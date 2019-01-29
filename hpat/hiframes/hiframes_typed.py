from __future__ import print_function, division, absolute_import

import operator
from collections import defaultdict
import numpy as np
import pandas as pd
from collections import namedtuple
import warnings
import numba
from numba import ir, ir_utils, types
from numba.ir_utils import (replace_arg_nodes, compile_to_numba_ir,
                            find_topo_order, gen_np_call, get_definition, guard,
                            find_callname, mk_alloc, find_const, is_setitem,
                            is_getitem, mk_unique_var, dprint_func_ir,
                            build_definitions, find_build_sequence)
from numba.inline_closurecall import inline_closure_call
from numba.typing.templates import Signature, bound_function, signature
from numba.typing.arraydecl import ArrayAttribute
from numba.extending import overload
from numba.typing.templates import infer_global, AbstractTemplate, signature
import hpat
from hpat import hiframes
from hpat.utils import (debug_prints, inline_new_blocks, ReplaceFunc,
    is_whole_slice)
from hpat.str_ext import string_type, unicode_to_std_str, std_str_to_unicode
from hpat.str_arr_ext import (string_array_type, StringArrayType,
    is_str_arr_typ, pre_alloc_string_array)
from hpat.hiframes.pd_series_ext import (SeriesType, string_series_type,
    series_to_array_type, BoxedSeriesType, dt_index_series_type,
    if_series_to_array_type, if_series_to_unbox, is_series_type,
    series_str_methods_type, SeriesRollingType, SeriesIatType,
    explicit_binop_funcs, series_dt_methods_type)
from hpat.pio_api import h5dataset_type
from hpat.hiframes.rolling import get_rolling_setup_args
from hpat.hiframes.aggregate import Aggregate
import datetime

LARGE_WIN_SIZE = 10

_dt_index_binops = ('==', '!=', '>=', '>', '<=', '<', '-',
                operator.eq, operator.ne, operator.ge, operator.gt,
                operator.le, operator.lt, operator.sub)

_string_array_comp_ops = ('==', '!=', '>=', '>', '<=', '<',
                operator.eq, operator.ne, operator.ge, operator.gt,
                operator.le, operator.lt)

_binop_to_str = {
    operator.eq: '==',
    operator.ne: '!=',
    operator.ge: '>=',
    operator.gt: '>',
    operator.le: '<=',
    operator.lt: '<',
    operator.sub: '-',
    operator.add: '+',
    operator.mul: '*',
    operator.truediv: '/',
    operator.floordiv: '//',
    operator.mod: '%',
    operator.pow: '**',
    '==': '==',
    '!=': '!=',
    '>=': '>=',
    '>': '>',
    '<=': '<=',
    '<': '<',
    '-': '-',
    '+': '+',
    '*': '*',
    '/': '/',
    '//': '//',
    '%': '%',
    '**': '**',
}


class HiFramesTyped(object):
    """Analyze and transform hiframes calls after typing"""

    def __init__(self, func_ir, typingctx, typemap, calltypes, return_type=None):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.typemap = typemap
        self.calltypes = calltypes
        self.return_type = return_type
        # keep track of tuple variables change by to_const_tuple
        self._type_changed_vars = []

    def run(self):
        blocks = self.func_ir.blocks
        work_list = list(blocks.items())
        while work_list:
            label, block = work_list.pop()
            new_body = []
            replaced = False
            for i, inst in enumerate(block.body):
                if isinstance(inst, Aggregate):
                    #import pdb; pdb.set_trace()
                    inst.out_typer_vars = None
                if isinstance(inst, ir.Assign):
                    out_nodes = self._run_assign(inst)
                    if isinstance(out_nodes, list):
                        new_body.extend(out_nodes)
                    if isinstance(out_nodes, ReplaceFunc):
                        rp_func = out_nodes
                        if rp_func.pre_nodes is not None:
                            new_body.extend(rp_func.pre_nodes)
                        # replace inst.value to a call with target args
                        # as expected by inline_closure_call
                        inst.value = ir.Expr.call(None, rp_func.args, (), inst.loc)
                        block.body = new_body + block.body[i:]
                        inline_closure_call(self.func_ir, rp_func.glbls,
                            block, len(new_body), rp_func.func, self.typingctx,
                            rp_func.arg_types,
                            self.typemap, self.calltypes, work_list)
                        replaced = True
                        break
                    if isinstance(out_nodes, dict):
                        block.body = new_body + block.body[i:]
                        inline_new_blocks(self.func_ir, block, i, out_nodes, work_list)
                        replaced = True
                        break
                elif (isinstance(inst, ir.StaticSetItem)
                        and self.typemap[inst.target.name] == h5dataset_type):
                    out_nodes = self._handle_h5_write(inst.target, inst.index, inst.value)
                    new_body.extend(out_nodes)
                elif (isinstance(inst, (ir.SetItem, ir.StaticSetItem))
                        and isinstance(self.typemap[inst.target.name], SeriesIatType)):
                    val_def = guard(get_definition, self.func_ir, inst.target)
                    assert isinstance(val_def, ir.Expr) and val_def.op == 'getattr' and val_def.attr in ('iat', 'iloc', 'loc')
                    series_var = val_def.value
                    inst.target = series_var
                    new_body.append(inst)
                else:
                    new_body.append(inst)
            if not replaced:
                blocks[label].body = new_body

        if debug_prints():  # pragma: no cover
            print("--- types before Series replacement:", self.typemap)
            print("calltypes: ", self.calltypes)

        replace_series = {}
        for vname, typ in self.typemap.items():
            new_typ = if_series_to_array_type(typ)
            if new_typ != typ:
                # print("replacing series type", vname)
                replace_series[vname] = new_typ
            # replace array.call() variable types
            if isinstance(typ, types.BoundFunction) and isinstance(typ.this, SeriesType):
                # TODO: handle string arrays, etc.
                assert (typ.typing_key in explicit_binop_funcs.keys()
                    or typ.typing_key.startswith('array.')
                    or typ.typing_key.startswith('series.'))
                # skip if series.func since it is replaced here
                if (not isinstance(typ.typing_key, str)
                        or not typ.typing_key.startswith('array.')):
                    continue
                this = series_to_array_type(typ.this)
                attr = typ.typing_key[len('array.'):]
                # string array copy() shouldn't go to np array resolver
                if this == string_array_type and attr == 'copy':
                    replace_series[vname] = hpat.str_arr_ext.StrArrayAttribute(
                        self.typingctx).resolve_copy(this)
                    continue
                resolver = getattr(ArrayAttribute, 'resolve_'+attr)
                # methods are either installed with install_array_method or
                # using @bound_function in arraydecl.py
                if hasattr(resolver, '__wrapped__'):
                    resolver = bound_function(typ.typing_key)(resolver.__wrapped__)
                new_typ = resolver(ArrayAttribute(self.typingctx), this)
                replace_series[vname] = new_typ

        for vname, typ in replace_series.items():
            self.typemap.pop(vname)
            self.typemap[vname] = typ

        replace_calltype = {}
        # replace sig of getitem/setitem/... series type with array
        for call, sig in self.calltypes.items():
            if sig is None:
                continue
            assert isinstance(sig, Signature)
            # XXX using replace() since it copies, otherwise cached overload
            # functions fail
            sig = sig.replace(return_type=if_series_to_array_type(sig.return_type))
            sig.args = tuple(map(if_series_to_array_type, sig.args))
            replace_calltype[call] = sig
            # XXX: side effect: force update of call signatures
            if isinstance(call, ir.Expr) and call.op == 'call':
                # StencilFunc requires kws for typing so sig.args can't be used
                # reusing sig.args since some types become Const in sig
                argtyps = sig.args[:len(call.args)]
                kwtyps = {name: self.typemap[v.name] for name, v in call.kws}
                new_sig = self.typemap[call.func.name].get_call_type(
                    self.typingctx , argtyps, kwtyps)
                # calltypes of things like BoundFunction (array.call) need to
                # be update for lowering to work
                # XXX: new_sig could be None for things like np.int32()
                if call in self.calltypes and new_sig is not None:
                    # for box_df, don't change return type so that information
                    # such as Categorical dtype is preserved
                    if isinstance(sig.return_type, hpat.hiframes.api.PandasDataFrameType):
                        new_sig.return_type = sig.return_type
                        replace_calltype[call] = new_sig
                        continue
                    old_sig = self.calltypes[call]
                    # fix types with undefined dtypes in empty_inferred, etc.
                    return_type = _fix_typ_undefs(new_sig.return_type, old_sig.return_type)
                    args = tuple(_fix_typ_undefs(a, b) for a,b  in zip(new_sig.args, old_sig.args))
                    replace_calltype[call] = Signature(return_type, args, new_sig.recvr, new_sig.pysig)

        for call, sig in replace_calltype.items():
            self.calltypes.pop(call)
            self.calltypes[call] = sig

        if debug_prints():  # pragma: no cover
            print("--- types after Series replacement:", self.typemap)
            print("calltypes: ", self.calltypes)

        # XXX remove slice() of h5 read due to Numba's #3380 bug
        while ir_utils.remove_dead(self.func_ir.blocks, self.func_ir.arg_names, self.func_ir, self.typemap):
            pass

        self.func_ir._definitions = build_definitions(self.func_ir.blocks)
        dprint_func_ir(self.func_ir, "after hiframes_typed")
        return if_series_to_unbox(self.return_type)

    def _run_assign(self, assign):
        lhs = assign.target.name
        rhs = assign.value

        # fix type of lhs if type of rhs has been changed
        if isinstance(rhs, ir.Var) and rhs.name in self._type_changed_vars:
            self.typemap.pop(lhs)
            self.typemap[lhs] = self.typemap[rhs.name]
            self._type_changed_vars.append(lhs)

        if isinstance(rhs, ir.Expr):
            # arr = S.values
            if (rhs.op == 'getattr'):
                rhs_type = self.typemap[rhs.value.name]  # get type of rhs value "S"

                # replace arr.dtype for dt64 since PA replaces with
                # np.datetime64[ns] which invalid, TODO: fix PA
                if (rhs.attr == 'dtype' and (is_series_type(rhs_type)
                        or isinstance(rhs_type, types.Array)) and isinstance(
                            rhs_type.dtype,
                            (types.NPDatetime, types.NPTimedelta))):
                    assign.value = ir.Global("numpy.datetime64", rhs_type.dtype, rhs.loc)
                    return [assign]

                # replace arr.dtype since PA replacement inserts in the
                # beginning of block, preventing fusion. TODO: fix PA
                if (rhs.attr == 'dtype' and isinstance(
                        if_series_to_array_type(rhs_type), types.Array)):
                    typ_str = str(rhs_type.dtype)
                    assign.value = ir.Global("np.dtype({})".format(typ_str), np.dtype(typ_str), rhs.loc)
                    return [assign]

                if isinstance(rhs_type, SeriesType) and rhs.attr == 'values':
                    # simply return the column
                    assign.value = rhs.value
                    return [assign]

                if isinstance(rhs_type, SeriesType) and isinstance(rhs_type.dtype, types.scalars.NPDatetime):
                    if rhs.attr in hpat.hiframes.pd_timestamp_ext.date_fields:
                        return self._run_DatetimeIndex_field(assign, assign.target, rhs)
                    if rhs.attr == 'date':
                        return self._run_DatetimeIndex_date(assign, assign.target, rhs)

                if rhs_type == series_dt_methods_type:
                    dt_def = guard(get_definition, self.func_ir, rhs.value)
                    if dt_def is None:  # TODO: check for errors
                        raise ValueError("invalid series.dt")
                    rhs.value = dt_def.value
                    return self._run_DatetimeIndex_field(assign, assign.target, rhs)

                if isinstance(rhs_type, SeriesType) and isinstance(rhs_type.dtype, types.scalars.NPTimedelta):
                    if rhs.attr in hpat.hiframes.pd_timestamp_ext.timedelta_fields:
                        return self._run_Timedelta_field(assign, assign.target, rhs)

            res = self._handle_string_array_expr(lhs, rhs, assign)
            if res is not None:
                return res

            # replace getitems on Series.iat
            if (rhs.op in ['getitem', 'static_getitem']
                    and isinstance(self.typemap[rhs.value.name], SeriesIatType)):
                val_def = guard(get_definition, self.func_ir, rhs.value)
                assert isinstance(val_def, ir.Expr) and val_def.op == 'getattr' and val_def.attr in ('iat', 'iloc', 'loc')
                series_var = val_def.value
                rhs.value = series_var

            # replace getitems on dt_index/dt64 series with Timestamp function
            if (rhs.op in ['getitem', 'static_getitem']
                    and self.typemap[rhs.value.name] == dt_index_series_type):
                if rhs.op == 'getitem':
                    ind_var = rhs.index
                else:
                    ind_var = rhs.index_var

                in_arr = rhs.value
                def f(_in_arr, _ind):
                    dt = _in_arr[_ind]
                    s = np.int64(dt)
                    res = hpat.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(s)

                assert isinstance(self.typemap[ind_var.name],
                    (types.Integer, types.IntegerLiteral))
                f_block = compile_to_numba_ir(f, {'numba': numba, 'np': np,
                                                'hpat': hpat}, self.typingctx,
                                            (if_series_to_array_type(self.typemap[in_arr.name]), types.intp),
                                            self.typemap, self.calltypes).blocks.popitem()[1]
                replace_arg_nodes(f_block, [in_arr, ind_var])
                nodes = f_block.body[:-3]  # remove none return
                nodes[-1].target = assign.target
                return nodes

            if rhs.op == 'call':

                fdef = guard(find_callname, self.func_ir, rhs, self.typemap)
                if fdef is None:
                    from numba.stencil import StencilFunc
                    # could be make_function from list comprehension which is ok
                    func_def = guard(get_definition, self.func_ir, rhs.func)
                    if isinstance(func_def, ir.Expr) and func_def.op == 'make_function':
                        return [assign]
                    if isinstance(func_def, ir.Global) and isinstance(func_def.value, StencilFunc):
                        return [assign]
                    warnings.warn(
                        "function call couldn't be found for initial analysis")
                    return [assign]
                else:
                    func_name, func_mod = fdef

                if (isinstance(func_mod, ir.Var)
                        and self.typemap[func_mod.name]
                        == series_str_methods_type):
                    f_def = guard(get_definition, self.func_ir, rhs.func)
                    str_def = guard(get_definition, self.func_ir, f_def.value)
                    if str_def is None:  # TODO: check for errors
                        raise ValueError("invalid series.str")

                    series_var = str_def.value
                    if func_name == 'contains':  # TODO: refactor
                        return self._handle_series_str_contains(
                            rhs, series_var)

                    return self._run_series_str_method(
                        assign, assign.target, series_var, func_name, rhs)

                # replace _get_type_max_value(arr.dtype) since parfors
                # arr.dtype transformation produces invalid code for dt64
                # TODO: min
                if fdef == ('_get_type_max_value', 'hpat.hiframes.hiframes_typed'):
                    if self.typemap[rhs.args[0].name] == types.DType(types.NPDatetime('ns')):
                        return self._replace_func(
                            lambda: hpat.hiframes.pd_timestamp_ext.integer_to_dt64(
                                numba.targets.builtins.get_type_max_value(
                                    numba.types.int64)), [])
                    return self._replace_func(
                        lambda d: numba.targets.builtins.get_type_max_value(
                                    d), rhs.args)

                if fdef == ('h5_read_dummy', 'hpat.pio_api'):
                    ndim = guard(find_const, self.func_ir, rhs.args[1])
                    dtype_str = guard(find_const, self.func_ir, rhs.args[2])
                    index_var = rhs.args[3]
                    filter_read = False

                    func_text = "def _h5_read_impl(dset_id, ndim, dtype_str, index):\n"
                    if guard(is_whole_slice, self.typemap, self.func_ir, index_var):
                        func_text += "  size_0 = hpat.pio_api.h5size(dset_id, np.int32(0))\n"
                    else:
                        # TODO: check index format for this case
                        filter_read = True
                        assert isinstance(self.typemap[index_var.name], types.BaseTuple)
                        func_text += "  read_indices = hpat.pio_api.get_filter_read_indices(index[0])\n"
                        func_text += "  size_0 = len(read_indices)\n"
                    for i in range(1, ndim):
                        func_text += "  size_{} = hpat.pio_api.h5size(dset_id, np.int32({}))\n".format(i, i)
                    func_text += "  arr_shape = ({},)\n".format(
                        ", ".join(["size_{}".format(i) for i in range(ndim)]))
                    func_text += "  zero_tup = ({},)\n".format(", ".join(["0"]*ndim))
                    func_text += "  A = np.empty(arr_shape, np.{})\n".format(
                        dtype_str)
                    if filter_read:
                        func_text += "  err = hpat.pio_api.h5read_filter(dset_id, np.int32({}), zero_tup, arr_shape, 0, A, read_indices)\n".format(ndim)
                    else:
                        func_text += "  err = hpat.pio_api.h5read(dset_id, np.int32({}), zero_tup, arr_shape, 0, A)\n".format(ndim)
                    func_text += "  return A\n"

                    loc_vars = {}
                    exec(func_text, {}, loc_vars)
                    _h5_read_impl = loc_vars['_h5_read_impl']
                    return self._replace_func(_h5_read_impl, rhs.args)

                if fdef == ('DatetimeIndex', 'pandas'):
                    return self._run_pd_DatetimeIndex(assign, assign.target, rhs)

                if fdef == ('Series', 'pandas'):
                    in_typ = self.typemap[rhs.args[0].name]
                    impl = hpat.hiframes.pd_series_ext.pd_series_overload(in_typ)
                    return self._replace_func(impl, rhs.args)

                if func_mod == 'hpat.hiframes.api':
                    return self._run_call_hiframes(assign, assign.target, rhs, func_name)

                if func_mod == 'hpat.hiframes.rolling':
                    return self._run_call_rolling(assign, assign.target, rhs, func_name)

                if fdef == ('empty_like', 'numpy'):
                    return self._handle_empty_like(assign, lhs, rhs)

                if (isinstance(func_mod, ir.Var)
                        and is_series_type(self.typemap[func_mod.name])):
                    return self._run_call_series(
                        assign, assign.target, rhs, func_mod, func_name)

                if (isinstance(func_mod, ir.Var) and isinstance(
                        self.typemap[func_mod.name], SeriesRollingType)):
                    return self._run_call_series_rolling(
                        assign, assign.target, rhs, func_mod, func_name)

                # handle sorted() with key lambda input
                if fdef == ('sorted', 'builtins') and 'key' in dict(rhs.kws):
                    return self._handle_sorted_by_key(rhs)

            if self._is_dt_index_binop(rhs):
                return self._handle_dt_index_binop(lhs, rhs, assign)

        return [assign]

    def _run_call_hiframes(self, assign, lhs, rhs, func_name):
        if func_name in ('to_series_type', 'to_arr_from_series'):
            assign.value = rhs.args[0]
            return [assign]

        if func_name in ('str_contains_regex', 'str_contains_noregex'):
            return self._handle_str_contains(assign, lhs, rhs, func_name)

        # arr = fix_df_array(col) -> arr=col if col is array
        if func_name == 'fix_df_array':
            in_typ = self.typemap[rhs.args[0].name]
            impl = hpat.hiframes.api.fix_df_array_overload(in_typ)
            return self._replace_func(impl, rhs.args)

        # arr = fix_rolling_array(col) -> arr=col if col is float array
        if func_name == 'fix_rolling_array':
            in_arr = rhs.args[0]
            if isinstance(self.typemap[in_arr.name].dtype, types.Float):
                assign.value = rhs.args[0]
                return [assign]
            else:
                def f(column):  # pragma: no cover
                    a = column.astype(np.float64)
                f_block = compile_to_numba_ir(f,
                                              {'hpat': hpat, 'np': np}, self.typingctx,
                                              (if_series_to_array_type(self.typemap[in_arr.name]),),
                                              self.typemap, self.calltypes).blocks.popitem()[1]
                replace_arg_nodes(f_block, [in_arr])
                nodes = f_block.body[:-3]
                nodes[-1].target = assign.target
                return nodes

        if func_name == 'set_df_col':
            return self._handle_df_col_filter(assign, lhs, rhs)

        if func_name == 'to_const_tuple':
            tup = rhs.args[0]
            tup_items = self._get_const_tup(tup)
            new_tup = ir.Expr.build_tuple(tup_items, tup.loc)
            assign.value = new_tup
            # fix type and definition of lhs
            self.typemap.pop(lhs.name)
            self._type_changed_vars.append(lhs.name)
            self.typemap[lhs.name] = types.Tuple(tuple(if_series_to_array_type(
                                     self.typemap[a.name]) for a in tup_items))
            self.func_ir._definitions[lhs.name] = [new_tup]
            return [assign]

        if func_name == 'concat':
            # concat() case where tuple type changes by to_const_type()
            if any([a.name in self._type_changed_vars for a in rhs.args]):
                argtyps = tuple(self.typemap[a.name] for a in rhs.args)
                old_sig = self.calltypes.pop(rhs)
                new_sig = self.typemap[rhs.func.name].get_call_type(
                    self.typingctx , argtyps, rhs.kws)
                self.calltypes[rhs] = new_sig

        # replace isna early to enable more optimization in PA
        # TODO: handle more types
        if func_name == 'isna':
            arr = rhs.args[0]
            ind = rhs.args[1]
            arr_typ = self.typemap[arr.name]
            if isinstance(arr_typ, (types.Array, SeriesType)):
                if isinstance(arr_typ.dtype, types.Float):
                    func = lambda arr,i: np.isnan(arr[i])
                    return self._replace_func(func, [arr, ind])
                elif isinstance(
                        arr_typ.dtype, (types.NPDatetime, types.NPTimedelta)):
                    nat = arr_typ.dtype('NaT')
                    # TODO: replace with np.isnat
                    return self._replace_func(
                        lambda arr,i: arr[i] == nat, [arr, ind])
                elif arr_typ.dtype != string_type:
                    return self._replace_func(lambda arr,i: False, [arr, ind])

        if func_name == 'df_isin':
            # XXX df isin is different than Series.isin, df.isin considers
            #  index but Series.isin ignores it (everything is set)
            # TODO: support strings and other types
            def _isin_series(A, B):
                numba.parfor.init_prange()
                n = len(A)
                m = len(B)
                S = np.empty(n, np.bool_)
                for i in numba.parfor.internal_prange(n):
                    S[i] = (A[i] == B[i] if i < m else False)
                return S

            return self._replace_func(_isin_series, rhs.args)

        if func_name == 'df_isin_vals':
            def _isin_series(A, vals):
                numba.parfor.init_prange()
                n = len(A)
                S = np.empty(n, np.bool_)
                for i in numba.parfor.internal_prange(n):
                    S[i] = A[i] in vals
                return S

            return self._replace_func(_isin_series, rhs.args)

        if func_name == 'flatten_to_series':
            def _flatten_impl(A):
                numba.parfor.init_prange()
                flat_list = []
                n = len(A)
                for i in numba.parfor.internal_prange(n):
                    l = A[i]
                    for s in l:
                        flat_list.append(s)

                return hpat.hiframes.api.to_series_type(
                    hpat.hiframes.api.parallel_fix_df_array(flat_list))
            return self._replace_func(_flatten_impl, [rhs.args[0]])

        if func_name == 'to_numeric':
            out_dtype = self.typemap[lhs.name].dtype
            conv_func = int
            if out_dtype == types.float64:
                conv_func = float
            else:
                assert out_dtype == types.int64

            def _to_numeric_impl(A):
                numba.parfor.init_prange()
                n = len(A)
                B = np.empty(n, out_dtype)
                for i in numba.parfor.internal_prange(n):
                    B[i] = conv_func(A[i])

                return hpat.hiframes.api.to_series_type(B)
            return self._replace_func(_to_numeric_impl, [rhs.args[0]],
                extra_globals={'out_dtype': out_dtype, 'conv_func': conv_func})

        return self._handle_df_col_calls(assign, lhs, rhs, func_name)

    def _run_call_series(self, assign, lhs, rhs, series_var, func_name):
        # single arg functions
        if func_name in ['sum', 'count', 'mean', 'var', 'min', 'max', 'prod']:
            if rhs.args or rhs.kws:
                raise ValueError("unsupported Series.{}() arguments".format(
                    func_name))
            # TODO: handle skipna, min_count arguments
            series_typ = self.typemap[series_var.name]
            series_dtype = series_typ.dtype
            func = series_replace_funcs[func_name]
            if isinstance(func, dict):
                func = func[series_dtype]
            return self._replace_func(func, [series_var])

        if func_name in ['std', 'nunique', 'describe', 'abs', 'isna',
                         'isnull', 'median', 'idxmin', 'idxmax', 'unique']:
            if rhs.args or rhs.kws:
                raise ValueError("unsupported Series.{}() arguments".format(
                    func_name))
            func = series_replace_funcs[func_name]
            # TODO: handle skipna, min_count arguments
            return self._replace_func(func, [series_var])

        if func_name == 'quantile':
            return self._replace_func(
                lambda A, q: hpat.hiframes.api.quantile(A, q),
                [series_var, rhs.args[0]]
            )

        if func_name == 'fillna':
            return self._run_call_series_fillna(assign, lhs, rhs, series_var)

        if func_name == 'dropna':
            return self._run_call_series_dropna(assign, lhs, rhs, series_var)

        if func_name in ('shift', 'pct_change'):
            # TODO: support default period argument
            if len(rhs.args) == 0:
                args = [series_var]
                func = series_replace_funcs[func_name + "_default"]
            else:
                assert len(rhs.args) == 1, "invalid args for " + func_name
                shift_const = rhs.args[0]
                args = [series_var, shift_const]
                func = series_replace_funcs[func_name]
            return self._replace_func(func, args)

        if func_name in ('nlargest', 'nsmallest'):
            # TODO: kws
            if len(rhs.args) == 0 and not rhs.kws:
                return self._replace_func(
                    series_replace_funcs[func_name + '_default'], [series_var],
                                    extra_globals={'gt_f': gt_f, 'lt_f': lt_f})
            n_arg = rhs.args[0]
            func = series_replace_funcs[func_name]
            return self._replace_func(func, [series_var, n_arg],
                                    extra_globals={'gt_f': gt_f, 'lt_f': lt_f})

        if func_name == 'head':
            # TODO: kws
            if len(rhs.args) == 0 and not rhs.kws:
                return self._replace_func(
                    series_replace_funcs['head_default'], [series_var])
            n_arg = rhs.args[0]
            func = series_replace_funcs[func_name]
            return self._replace_func(func, [series_var, n_arg])

        if func_name in ('cov', 'corr'):
            S2 = rhs.args[0]
            func = series_replace_funcs[func_name]
            return self._replace_func(func, [series_var, S2])

        if func_name in ('argsort', 'sort_values'):
            return self._handle_series_sort(
                lhs, rhs, series_var, func_name == 'argsort')

        if func_name == 'rolling':
            # XXX: remove rolling setup call, assuming still available in definitions
            return []

        if func_name == 'combine':
            return self._handle_series_combine(assign, lhs, rhs, series_var)

        if func_name in ('map', 'apply'):
            return self._handle_series_map(assign, lhs, rhs, series_var)

        if func_name == 'append':
            other = rhs.args[0]
            if isinstance(self.typemap[other.name], SeriesType):
                func = series_replace_funcs['append_single']
            else:
                func = series_replace_funcs['append_tuple']
            return self._replace_func(func, [series_var, other])

        if func_name == 'notna':
            # TODO: make sure this is fused and optimized properly
            return self._replace_func(
                lambda S: S.isna()==False, [series_var],
                array_typ_convert=False)

        # astype with string output
        if func_name == 'astype' and self.typemap[lhs.name] == string_series_type:
            # just return input if string
            if self.typemap[series_var.name] == string_series_type:
                return self._replace_func(lambda a: a, [series_var])
            func = series_replace_funcs['astype_str']
            return self._replace_func(func, [series_var])

        if func_name in explicit_binop_funcs.values():
            binop_map = {v: _binop_to_str[k] for k, v in explicit_binop_funcs.items()}
            func_text = "def _binop_impl(A, B):\n"
            func_text += "  return A {} B\n".format(binop_map[func_name])

            loc_vars = {}
            exec(func_text, {}, loc_vars)
            _binop_impl = loc_vars['_binop_impl']
            return self._replace_func(_binop_impl, [series_var] + rhs.args)

        # functions we revert to Numpy for now, otherwise warning
        # TODO: handle series-specific cases for this funcs
        if (not func_name.startswith("values.") and func_name
                not in ('copy', 'cumsum', 'cumprod', 'take', 'astype')):
            warnings.warn("unknown Series call {}, reverting to Numpy".format(
                func_name))

        return [assign]

    def _handle_series_sort(self, lhs, rhs, series_var, is_argsort):
        """creates an index list and passes it to a Sort node as data
        """
        in_df = {}
        out_df = {}
        out_key_arr = lhs
        nodes = []
        if is_argsort:
            def _get_data(S):  # pragma: no cover
                n = len(S)
                return np.arange(n)

            f_block = compile_to_numba_ir(
                _get_data, {'np': np}, self.typingctx,
                (if_series_to_array_type(self.typemap[series_var.name]),),
                self.typemap, self.calltypes).blocks.popitem()[1]
            replace_arg_nodes(f_block, [series_var])
            nodes = f_block.body[:-2]
            in_df = {'inds': nodes[-1].target}
            out_df = {'inds': lhs}
            # dummy output key, TODO: remove
            out_key_arr = ir.Var(lhs.scope, mk_unique_var('dummy'), lhs.loc)
            self.typemap[out_key_arr.name] = if_series_to_array_type(
                self.typemap[series_var.name])


        nodes.append(hiframes.sort.Sort(series_var.name, lhs.name, [series_var],
            [out_key_arr], in_df, out_df, False, lhs.loc))
        return nodes

    def _run_call_series_fillna(self, assign, lhs, rhs, series_var):
        dtype = self.typemap[series_var.name].dtype
        val = rhs.args[0]
        kws = dict(rhs.kws)
        inplace = False
        if 'inplace' in kws:
            inplace = guard(find_const, self.func_ir, kws['inplace'])
            if inplace == None:  # pragma: no cover
                raise ValueError("inplace arg to fillna should be constant")

        if inplace:
            if dtype == string_type:
                # optimization: just set null bit if fill is empty
                if guard(find_const, self.func_ir, val) == "":
                    return self._replace_func(
                        lambda A: hpat.str_arr_ext.set_null_bits(A),
                        [series_var])
                # Since string arrays can't be changed, we have to create a new
                # array and assign it back to the same Series variable
                # result back to the same variable
                # TODO: handle string array reflection
                def str_fillna_impl(A, fill):
                    # not using A.fillna since definition list is not working
                    # for A to find callname
                    hpat.hiframes.api.fillna_str_alloc(A, fill)
                    #A.fillna(fill)
                fill_var = rhs.args[0]
                arg_typs = (self.typemap[series_var.name], self.typemap[fill_var.name])
                f_block = compile_to_numba_ir(str_fillna_impl,
                                  {'hpat': hpat},
                                  self.typingctx, arg_typs,
                                  self.typemap, self.calltypes).blocks.popitem()[1]
                replace_arg_nodes(f_block, [series_var, fill_var])
                # assign output back to series variable
                f_block.body[-4].target = series_var
                return {0: f_block}
            else:
                return self._replace_func(
                    lambda a,b,c: hpat.hiframes.api.fillna(a,b,c),
                    [series_var, series_var, val])
        else:
            if dtype == string_type:
                func = series_replace_funcs['fillna_str_alloc']
            else:
                func = series_replace_funcs['fillna_alloc']
            return self._replace_func(func, [series_var, val])

    def _run_call_series_dropna(self, assign, lhs, rhs, series_var):
        dtype = self.typemap[series_var.name].dtype
        kws = dict(rhs.kws)
        inplace = False
        if 'inplace' in kws:
            inplace = guard(find_const, self.func_ir, kws['inplace'])
            if inplace == None:  # pragma: no cover
                raise ValueError("inplace arg to dropna should be constant")

        if inplace:
            # Since arrays can't resize inplace, we have to create a new
            # array and assign it back to the same Series variable
            # result back to the same variable
            def dropna_impl(A):
                # not using A.dropna since definition list is not working
                # for A to find callname
                res = hpat.hiframes.api.dropna(A)

            arg_typs = (self.typemap[series_var.name],)
            f_block = compile_to_numba_ir(dropna_impl,
                                {'hpat': hpat},
                                self.typingctx, arg_typs,
                                self.typemap, self.calltypes).blocks.popitem()[1]
            replace_arg_nodes(f_block, [series_var])
            # assign output back to series variable
            f_block.body[-4].target = series_var
            return {0: f_block}
        else:
            if dtype == string_type:
                func = series_replace_funcs['dropna_str_alloc']
            elif isinstance(dtype, types.Float):
                func = series_replace_funcs['dropna_float']
            else:
                # integer case, TODO: bool, date etc.
                func = lambda A: A
            return self._replace_func(func, [series_var])

    def _handle_series_map(self, assign, lhs, rhs, series_var):
        """translate df.A.map(lambda a:...) to prange()
        """
        # error checking: make sure there is function input only
        if len(rhs.args) != 1:
            raise ValueError("map expects 1 argument")
        func = guard(get_definition, self.func_ir, rhs.args[0])
        if func is None or not (isinstance(func, ir.Expr)
                                and func.op == 'make_function'):
            raise ValueError("lambda for map not found")

        out_typ = self.typemap[lhs.name].dtype

        # TODO: handle non numpy alloc types like string array
        # prange func to inline
        func_text = "def f(A):\n"
        func_text += "  numba.parfor.init_prange()\n"
        func_text += "  n = len(A)\n"
        func_text += "  S = numba.unsafe.ndarray.empty_inferred((n,))\n"
        func_text += "  for i in numba.parfor.internal_prange(n):\n"
        func_text += "    t = A[i]\n"
        func_text += "    v = map_func(t)\n"
        func_text += "    S[i] = hpat.hiframes.api.convert_tup_to_rec(v)\n"
        # func_text += "    print(S[i])\n"
        if out_typ == hpat.hiframes.pd_timestamp_ext.datetime_date_type:
            func_text += "  ret = hpat.hiframes.api.to_date_series_type(S)\n"
        else:
            func_text += "  ret = S\n"
        #func_text += "  return hpat.hiframes.api.to_series_type(ret)\n"
        func_text += "  return ret\n"

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        _globals = self.func_ir.func_id.func.__globals__
        f_ir = compile_to_numba_ir(f, {'numba': numba, 'np': np, 'hpat': hpat})

        # fix definitions to enable finding sentinel
        f_ir._definitions = build_definitions(f_ir.blocks)
        topo_order = find_topo_order(f_ir.blocks)

        # find sentinel function and replace with user func
        for l in topo_order:
            block = f_ir.blocks[l]
            for i, stmt in enumerate(block.body):
                if (isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op == 'call'):
                    fdef = guard(get_definition, f_ir, stmt.value.func)
                    if isinstance(fdef, ir.Global) and fdef.name == 'map_func':
                        inline_closure_call(f_ir, _globals, block, i, func)
                        break

        # remove sentinel global to avoid type inference issues
        ir_utils.remove_dead(f_ir.blocks, f_ir.arg_names, f_ir)
        f_ir._definitions = build_definitions(f_ir.blocks)
        arg_typs = (self.typemap[series_var.name],)
        f_typemap, f_return_type, f_calltypes = numba.compiler.type_inference_stage(
                self.typingctx, f_ir, arg_typs, None)
        # remove argument entries like arg.a from typemap
        arg_names = [vname for vname in f_typemap if vname.startswith("arg.")]
        for a in arg_names:
            f_typemap.pop(a)
        self.typemap.update(f_typemap)
        self.calltypes.update(f_calltypes)
        replace_arg_nodes(f_ir.blocks[topo_order[0]], [series_var])
        return f_ir.blocks

    def _run_call_rolling(self, assign, lhs, rhs, func_name):
        if func_name == 'rolling_corr':
            def rolling_corr_impl(arr, other, win, center):
                cov = hpat.hiframes.rolling.rolling_cov(arr, other, win, center)
                a_std = hpat.hiframes.rolling.rolling_fixed(arr, win, center, False, 'std')
                b_std = hpat.hiframes.rolling.rolling_fixed(other, win, center, False, 'std')
                return cov / (a_std * b_std)
            return self._replace_func(rolling_corr_impl, rhs.args)
        if func_name == 'rolling_cov':
            def rolling_cov_impl(arr, other, w, center):  # pragma: no cover
                ddof = 1
                X = arr.astype(np.float64)
                Y = other.astype(np.float64)
                XpY = X + Y
                XtY = X * Y
                count = hpat.hiframes.rolling.rolling_fixed(XpY, w, center, False, 'count')
                mean_XtY = hpat.hiframes.rolling.rolling_fixed(XtY, w, center, False, 'mean')
                mean_X = hpat.hiframes.rolling.rolling_fixed(X, w, center, False, 'mean')
                mean_Y = hpat.hiframes.rolling.rolling_fixed(Y, w, center, False, 'mean')
                bias_adj = count / (count - ddof)
                return (mean_XtY - mean_X * mean_Y) * bias_adj
            return self._replace_func(rolling_cov_impl, rhs.args)
        # replace apply function with dispatcher obj, now the type is known
        if (func_name == 'rolling_fixed' and isinstance(
                self.typemap[rhs.args[4].name], types.MakeFunctionLiteral)):
            # for apply case, create a dispatcher for the kernel and pass it
            # TODO: automatically handle lambdas in Numba
            dtype = self.typemap[rhs.args[0].name].dtype
            out_dtype = self.typemap[lhs.name].dtype
            func_node = guard(get_definition, self.func_ir, rhs.args[4])
            imp_dis = self._handle_rolling_apply_func(
                func_node, dtype, out_dtype)
            def f(arr, w, center):  # pragma: no cover
                df_arr = hpat.hiframes.rolling.rolling_fixed(
                                                arr, w, center, False, _func)
            f_block = compile_to_numba_ir(f, {'hpat': hpat, '_func': imp_dis},
                        self.typingctx,
                        tuple(self.typemap[v.name] for v in rhs.args[:-2]),
                        self.typemap, self.calltypes).blocks.popitem()[1]
            replace_arg_nodes(f_block, rhs.args[:-2])
            nodes = f_block.body[:-3]  # remove none return
            nodes[-1].target = lhs
            return nodes
        elif (func_name == 'rolling_variable' and isinstance(
                self.typemap[rhs.args[5].name], types.MakeFunctionLiteral)):
            # for apply case, create a dispatcher for the kernel and pass it
            # TODO: automatically handle lambdas in Numba
            dtype = self.typemap[rhs.args[0].name].dtype
            out_dtype = self.typemap[lhs.name].dtype
            func_node = guard(get_definition, self.func_ir, rhs.args[5])
            imp_dis = self._handle_rolling_apply_func(
                func_node, dtype, out_dtype)
            def f(arr, on_arr, w, center):  # pragma: no cover
                df_arr = hpat.hiframes.rolling.rolling_variable(
                                                arr, on_arr, w, center, False, _func)
            f_block = compile_to_numba_ir(f, {'hpat': hpat, '_func': imp_dis},
                        self.typingctx,
                        tuple(self.typemap[v.name] for v in rhs.args[:-2]),
                        self.typemap, self.calltypes).blocks.popitem()[1]
            replace_arg_nodes(f_block, rhs.args[:-2])
            nodes = f_block.body[:-3]  # remove none return
            nodes[-1].target = lhs
            return nodes
        return [assign]

    def _handle_series_combine(self, assign, lhs, rhs, series_var):
        """translate s1.combine(s2,lambda x1,x2 :...) to prange()
        """
        # error checking: make sure there is function input only
        if len(rhs.args) < 2:
            raise ValueError("not enough arguments in call to combine")
        if len(rhs.args) > 3:
            raise ValueError("too many arguments in call to combine")
        func = guard(get_definition, self.func_ir, rhs.args[1])
        if func is None or not (isinstance(func, ir.Expr)
                                and func.op == 'make_function'):
            raise ValueError("lambda for combine not found")

        out_typ = self.typemap[lhs.name].dtype

        # If we are called with 3 arguments, we must use 3rd arg as a fill value,
        # instead of Nan.
        use_nan = len(rhs.args) == 2

        # prange func to inline
        if use_nan:
            func_text = "def f(A, B):\n"
        else:
            func_text = "def f(A, B, C):\n"
        func_text += "  n1 = len(A)\n"
        func_text += "  n2 = len(B)\n"
        func_text += "  n = max(n1, n2)\n"
        if not isinstance(self.typemap[series_var.name].dtype, types.Float) and use_nan:
            func_text += "  assert n1 == n, 'can not use NAN for non-float series, with different length'\n"
        if not isinstance(self.typemap[rhs.args[0].name].dtype, types.Float) and use_nan:
            func_text += "  assert n2 == n, 'can not use NAN for non-float series, with different length'\n"
        func_text += "  numba.parfor.init_prange()\n"
        func_text += "  S = numba.unsafe.ndarray.empty_inferred((n,))\n"
        func_text += "  for i in numba.parfor.internal_prange(n):\n"
        if use_nan and isinstance(self.typemap[series_var.name].dtype, types.Float):
            func_text += "    t1 = np.nan\n"
            func_text += "    if i < n1:\n"
            func_text += "      t1 = A[i]\n"
        # length is equal, due to assertion above
        elif use_nan:
            func_text += "    t1 = A[i]\n"
        else:
            func_text += "    t1 = C\n"
            func_text += "    if i < n1:\n"
            func_text += "      t1 = A[i]\n"
        # same, but for 2nd argument
        if use_nan and isinstance(self.typemap[rhs.args[0].name].dtype, types.Float):
            func_text += "    t2 = np.nan\n"
            func_text += "    if i < n2:\n"
            func_text += "      t2 = B[i]\n"
        elif use_nan:
            func_text += "    t2 = B[i]\n"
        else:
            func_text += "    t2 = C\n"
            func_text += "    if i < n2:\n"
            func_text += "      t2 = B[i]\n"
        func_text += "    S[i] = map_func(t1, t2)\n"
        if out_typ == hpat.hiframes.pd_timestamp_ext.datetime_date_type:
            func_text += "  ret = hpat.hiframes.api.to_date_series_type(S)\n"
        else:
            func_text += "  ret = S\n"
        func_text += "  return ret\n"

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        _globals = self.func_ir.func_id.func.__globals__
        f_ir = compile_to_numba_ir(f, {'numba': numba, 'np': np, 'hpat': hpat})

        # fix definitions to enable finding sentinel
        f_ir._definitions = build_definitions(f_ir.blocks)
        topo_order = find_topo_order(f_ir.blocks)

        # find sentinel function and replace with user func
        for l in topo_order:
            block = f_ir.blocks[l]
            for i, stmt in enumerate(block.body):
                if (isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op == 'call'):
                    fdef = guard(get_definition, f_ir, stmt.value.func)
                    if isinstance(fdef, ir.Global) and fdef.name == 'map_func':
                        inline_closure_call(f_ir, _globals, block, i, func)
                        break

        # remove sentinel global to avoid type inference issues
        ir_utils.remove_dead(f_ir.blocks, f_ir.arg_names, f_ir)
        f_ir._definitions = build_definitions(f_ir.blocks)
        if use_nan:
            arg_typs = (self.typemap[series_var.name], self.typemap[rhs.args[0].name],)
        else:
            arg_typs = (self.typemap[series_var.name], self.typemap[rhs.args[0].name], self.typemap[rhs.args[2].name],)
        f_typemap, f_return_type, f_calltypes = numba.compiler.type_inference_stage(
                self.typingctx, f_ir, arg_typs, None)
        # remove argument entries like arg.a from typemap
        arg_names = [vname for vname in f_typemap if vname.startswith("arg.")]
        for a in arg_names:
            f_typemap.pop(a)
        self.typemap.update(f_typemap)
        self.calltypes.update(f_calltypes)
        func_args = [series_var, rhs.args[0]]
        if not use_nan:
            func_args.append(rhs.args[2])
        replace_arg_nodes(f_ir.blocks[topo_order[0]], func_args)
        return f_ir.blocks

    def _run_call_series_rolling(self, assign, lhs, rhs, rolling_var, func_name):
        """
        Handle Series rolling calls like:
          A = df.column.rolling(3).sum()
        """
        rolling_call = guard(get_definition, self.func_ir, rolling_var)
        assert isinstance(rolling_call, ir.Expr) and rolling_call.op == 'call'
        call_def = guard(get_definition, self.func_ir, rolling_call.func)
        assert isinstance(call_def, ir.Expr) and call_def.op == 'getattr'
        series_var = call_def.value
        nodes = []
        window, center, on = get_rolling_setup_args(self.func_ir, rolling_call, False)
        if not isinstance(center, ir.Var):
            center_var = ir.Var(lhs.scope, mk_unique_var("center"), lhs.loc)
            self.typemap[center_var.name] = types.bool_
            nodes.append(ir.Assign(ir.Const(center, lhs.loc), center_var, lhs.loc))
            center = center_var

        if func_name in ('cov', 'corr'):
            # TODO: variable window
            if len(rhs.args) == 1:
                other = rhs.args[0]
            else:
                other = series_var
            if func_name == 'cov':
                f = lambda a,b,w,c: hpat.hiframes.rolling.rolling_cov(a,b,w,c)
            if func_name == 'corr':
                f = lambda a,b,w,c: hpat.hiframes.rolling.rolling_corr(a,b,w,c)
            return self._replace_func(f, [series_var, other, window, center],
                                      pre_nodes=nodes)
        elif func_name == 'apply':
            func_node = guard(get_definition, self.func_ir, rhs.args[0])
            dtype = self.typemap[series_var.name].dtype
            out_dtype = self.typemap[lhs.name].dtype
            func_global = self._handle_rolling_apply_func(
                func_node, dtype, out_dtype)
        else:
            func_global = func_name
        def f(arr, w, center):  # pragma: no cover
            return hpat.hiframes.rolling.rolling_fixed(arr, w, center, False, _func)
        args = [series_var, window, center]
        return self._replace_func(
            f, args, pre_nodes=nodes, extra_globals={'_func': func_global})

    def _handle_rolling_apply_func(self, func_node, dtype, out_dtype):
        if func_node is None:
                raise ValueError(
                    "cannot find kernel function for rolling.apply() call")
        # TODO: more error checking on the kernel to make sure it doesn't
        # use global/closure variables
        if func_node.closure is not None:
            raise ValueError(
                "rolling apply kernel functions cannot have closure variables")
        if func_node.defaults is not None:
            raise ValueError(
                "rolling apply kernel functions cannot have default arguments")
        # create a function from the code object
        glbs = self.func_ir.func_id.func.__globals__
        lcs = {}
        exec("def f(A): return A", glbs, lcs)
        kernel_func = lcs['f']
        kernel_func.__code__ = func_node.code
        kernel_func.__name__ = func_node.code.co_name
        # use hpat's sequential pipeline to enable pandas operations
        # XXX seq pipeline used since dist pass causes a hang
        m = numba.ir_utils._max_label
        impl_disp = numba.njit(
            kernel_func, pipeline_class=hpat.compiler.HPATPipelineSeq)
        # precompile to avoid REP counting conflict in testing
        sig = out_dtype(types.Array(dtype, 1, 'C'))
        impl_disp.compile(sig)
        numba.ir_utils._max_label += m
        return impl_disp

    def _run_DatetimeIndex_field(self, assign, lhs, rhs):
        """transform DatetimeIndex.<field>
        """
        arr = rhs.value
        field = rhs.attr

        func_text = 'def f(dti):\n'
        func_text += '    numba.parfor.init_prange()\n'
        func_text += '    n = len(dti)\n'
        #func_text += '    S = numba.unsafe.ndarray.empty_inferred((n,))\n'
        # TODO: why doesn't empty_inferred work for t4 mortgage test?
        func_text += '    S = np.empty(n, np.int64)\n'
        func_text += '    for i in numba.parfor.internal_prange(n):\n'
        func_text += '        dt64 = hpat.hiframes.pd_timestamp_ext.dt64_to_integer(dti[i])\n'
        func_text += '        ts = hpat.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)\n'
        func_text += '        S[i] = ts.' + field + '\n'
        func_text += '    return S\n'
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        return self._replace_func(f, [arr])

    def _run_DatetimeIndex_date(self, assign, lhs, rhs):
        """transform DatetimeIndex.date
        """
        arr = rhs.value
        field = rhs.attr

        func_text = 'def f(dti):\n'
        func_text += '    numba.parfor.init_prange()\n'
        func_text += '    n = len(dti)\n'
        func_text += '    S = numba.unsafe.ndarray.empty_inferred((n,))\n'
        func_text += '    for i in numba.parfor.internal_prange(n):\n'
        func_text += '        dt64 = hpat.hiframes.pd_timestamp_ext.dt64_to_integer(dti[i])\n'
        func_text += '        ts = hpat.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)\n'
        func_text += '        S[i] = hpat.hiframes.pd_timestamp_ext.datetime_date_ctor(ts.year, ts.month, ts.day)\n'
        #func_text += '        S[i] = datetime.date(ts.year, ts.month, ts.day)\n'
        #func_text += '        S[i] = ts.day + (ts.month << 16) + (ts.year << 32)\n'
        func_text += '    return S\n'
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        return self._replace_func(f, [arr])

    def _run_Timedelta_field(self, assign, lhs, rhs):
        """transform Timedelta.<field>
        """
        arr = rhs.value
        field = rhs.attr

        func_text = 'def f(dti):\n'
        func_text += '    numba.parfor.init_prange()\n'
        func_text += '    n = len(dti)\n'
        func_text += '    S = numba.unsafe.ndarray.empty_inferred((n,))\n'
        func_text += '    for i in numba.parfor.internal_prange(n):\n'
        func_text += '        dt64 = hpat.hiframes.pd_timestamp_ext.timedelta64_to_integer(dti[i])\n'
        if field == 'nanoseconds':
            func_text += '        S[i] = dt64 % 1000\n'
        elif field == 'microseconds':
            func_text += '        S[i] = dt64 // 1000 % 100000\n'
        elif field == 'seconds':
            func_text += '        S[i] = dt64 // (1000 * 1000000) % (60 * 60 * 24)\n'
        elif field == 'days':
            func_text += '        S[i] = dt64 // (1000 * 1000000 * 60 * 60 * 24)\n'
        else:
            assert(0)
        func_text += '    return S\n'
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        return self._replace_func(f, [arr])

    def _run_pd_DatetimeIndex(self, assign, lhs, rhs):
        """transform pd.DatetimeIndex() call with string array argument
        """
        kws = dict(rhs.kws)
        if 'data' in kws:
            data = kws['data']
            if len(rhs.args) != 0:  # pragma: no cover
                raise ValueError(
                    "only data argument suppoted in pd.DatetimeIndex()")
        else:
            if len(rhs.args) != 1:  # pragma: no cover
                raise ValueError(
                    "data argument in pd.DatetimeIndex() expected")
            data = rhs.args[0]

        in_typ = self.typemap[data.name]
        if not (in_typ == string_array_type or in_typ == string_series_type):
            # already dt_index or int64
            # TODO: check for other types
            f = lambda A: hpat.hiframes.api.ts_series_to_arr_typ(A)
            return self._replace_func(f, [data])

        def f(str_arr):
            numba.parfor.init_prange()
            n = len(str_arr)
            S = numba.unsafe.ndarray.empty_inferred((n,))
            for i in numba.parfor.internal_prange(n):
                S[i] = hpat.hiframes.pd_timestamp_ext.parse_datetime_str(str_arr[i])
            return S

        return self._replace_func(f, [data])

    def _run_series_str_method(self, assign, lhs, arr, func_name, rhs):

        if func_name not in ('len', 'replace', 'split', 'get'):
            raise NotImplementedError(
                "Series.str.{} not supported yet".format(func_name))

        if func_name == 'replace':
            return self._run_series_str_replace(assign, lhs, arr, rhs)

        if func_name == 'split':
            return self._run_series_str_split(assign, lhs, arr, rhs)

        if func_name == 'get':
            return self._run_series_str_get(assign, lhs, arr, rhs)

        if func_name == 'len':
            out_typ = 'np.int64'

        func_text = 'def f(str_arr):\n'
        func_text += '    numba.parfor.init_prange()\n'
        func_text += '    n = len(str_arr)\n'
        #func_text += '    S = numba.unsafe.ndarray.empty_inferred((n,))\n'
        # TODO: use empty_inferred after it is fixed
        func_text += '    S = np.empty(n, {})\n'.format(out_typ)
        func_text += '    for i in numba.parfor.internal_prange(n):\n'
        func_text += '        val = str_arr[i]\n'
        func_text += '        S[i] = len(val)\n'
        func_text += '    return S\n'
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        return self._replace_func(f, [arr])

    def _run_series_str_replace(self, assign, lhs, arr, rhs):
        regex = True
        # TODO: refactor arg parsing
        kws = dict(rhs.kws)
        if 'regex' in kws:
            regex = guard(find_const, self.func_ir, kws['regex'])
            if regex is None:
                raise ValueError(
                    "str.replace regex argument should be constant")

        impl = _str_replace_regex_impl if regex else _str_replace_noregex_impl

        return self._replace_func(
            impl,
            [arr, rhs.args[0], rhs.args[1]],
            extra_globals={'unicode_to_std_str': unicode_to_std_str,
                            'std_str_to_unicode': std_str_to_unicode,
                            'pre_alloc_string_array': pre_alloc_string_array}
        )


    def _run_series_str_split(self, assign, lhs, arr, rhs):
        sep = rhs.args[0]  # TODO: support default whitespace separator

        def _str_split_impl(str_arr, sep):
            numba.parfor.init_prange()
            n = len(str_arr)
            out_arr = hpat.str_ext.alloc_list_list_str(n)
            for i in numba.parfor.internal_prange(n):
                in_str = str_arr[i]
                out_arr[i] = in_str.split(sep)

            return out_arr

        return self._replace_func(_str_split_impl, [arr, sep])

    def _run_series_str_get(self, assign, lhs, arr, rhs):
        # XXX only supports get for list(list(str)) input
        assert (self.typemap[arr.name]
                    == SeriesType(types.List(string_type)))
        ind_var = rhs.args[0]

        def _str_get_impl(str_arr, ind):
            numba.parfor.init_prange()
            n = len(str_arr)
            n_total_chars = 0
            str_list = hpat.str_ext.alloc_str_list(n)
            for i in numba.parfor.internal_prange(n):
                # TODO: support NAN
                in_list_str = str_arr[i]
                out_str = in_list_str[ind]
                str_list[i] = out_str
                n_total_chars += len(out_str)
            numba.parfor.init_prange()
            out_arr = pre_alloc_string_array(n, n_total_chars)
            for i in numba.parfor.internal_prange(n):
                _str = str_list[i]
                out_arr[i] = _str
            return out_arr

        return self._replace_func(_str_get_impl, [arr, ind_var],
            extra_globals={'pre_alloc_string_array': pre_alloc_string_array})

    def _is_dt_index_binop(self, rhs):
        if rhs.op != 'binop':
            return False

        if rhs.fn not in _dt_index_binops:
            return False

        arg1, arg2 = self.typemap[rhs.lhs.name], self.typemap[rhs.rhs.name]
        # one of them is dt_index but not both
        if ((arg1 == dt_index_series_type or arg2 == dt_index_series_type)
                and not (arg1 == dt_index_series_type and arg2 == dt_index_series_type)):
            return True

        return False

    def _handle_dt_index_binop(self, lhs, rhs, assign):
        arg1, arg2 = rhs.lhs, rhs.rhs
        allowed_types = (dt_index_series_type, string_type)

        # TODO: this has to be more generic to support all combinations.
        if (self.typemap[arg1.name] == dt_index_series_type and
            self.typemap[arg2.name] == hpat.hiframes.pd_timestamp_ext.pandas_timestamp_type and
            rhs.fn in ('-', operator.sub)):
            return self._replace_func(_column_sub_impl_datetimeindex_timestamp, [arg1, arg2])

        if (types.unliteral(self.typemap[arg1.name]) not in allowed_types
                or types.unliteral(self.typemap[arg2.name]) not in allowed_types):
            raise ValueError("DatetimeIndex operation not supported")

        op_str = _binop_to_str[rhs.fn]

        func_text = 'def f(arg1, arg2):\n'
        if self.typemap[arg1.name] == dt_index_series_type:
            func_text += '  dt_index, _str = arg1, arg2\n'
            comp = 'dt_index[i] {} other'.format(op_str)
        else:
            func_text += '  dt_index, _str = arg2, arg1\n'
            comp = 'other {} dt_index[i]'.format(op_str)
        func_text += '  l = len(dt_index)\n'
        func_text += '  other = hpat.hiframes.pd_timestamp_ext.parse_datetime_str(_str)\n'
        func_text += '  S = numba.unsafe.ndarray.empty_inferred((l,))\n'
        func_text += '  for i in numba.parfor.internal_prange(l):\n'
        func_text += '    S[i] = {}\n'.format(comp)
        func_text += '  return S\n'
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']
        # print(func_text)
        return self._replace_func(f, [arg1, arg2])

    def _handle_string_array_expr(self, lhs, rhs, assign):
        # convert str_arr==str into parfor
        if (rhs.op == 'binop'
                and rhs.fn in _string_array_comp_ops
                and (is_str_arr_typ(self.typemap[rhs.lhs.name])
                     or is_str_arr_typ(self.typemap[rhs.rhs.name]))):
            arg1 = rhs.lhs
            arg2 = rhs.rhs
            arg1_access = 'A'
            arg2_access = 'B'
            len_call = 'len(A)'
            if is_str_arr_typ(self.typemap[arg1.name]):
                arg1_access = 'A[i]'
                # replace type now for correct typing of len, etc.
                self.typemap.pop(arg1.name)
                self.typemap[arg1.name] = string_array_type

            if is_str_arr_typ(self.typemap[arg2.name]):
                arg1_access = 'B[i]'
                len_call = 'len(B)'
                self.typemap.pop(arg2.name)
                self.typemap[arg2.name] = string_array_type

            op_str = _binop_to_str[rhs.fn]

            func_text = 'def f(A, B):\n'
            func_text += '  l = {}\n'.format(len_call)
            func_text += '  S = np.empty(l, dtype=np.bool_)\n'
            func_text += '  for i in numba.parfor.internal_prange(l):\n'
            func_text += '    S[i] = {} {} {}\n'.format(arg1_access, op_str,
                                                        arg2_access)
            func_text += '  return S\n'

            loc_vars = {}
            exec(func_text, {}, loc_vars)
            f = loc_vars['f']
            return self._replace_func(f, [arg1, arg2])

        return None

    def _handle_series_str_contains(self, rhs, series_var):
        """
        Handle string contains like:
          B = df.column.str.contains('oo*', regex=True)
        """
        kws = dict(rhs.kws)
        pat = rhs.args[0]
        regex = True  # default regex arg is True
        if 'regex' in kws:
            regex = guard(find_const, self.func_ir, kws['regex'])
            if regex is None:
                raise ValueError("str.contains expects constant regex argument")
        if regex:
            fname = "str_contains_regex"
        else:
            fname = "str_contains_noregex"

        return self._replace_func(series_replace_funcs[fname], [series_var, pat])


    def _handle_empty_like(self, assign, lhs, rhs):
        # B = empty_like(A) -> B = empty(len(A), dtype)
        in_arr = rhs.args[0]

        if self.typemap[in_arr.name].ndim == 1:
            # generate simpler len() for 1D case
            def f(_in_arr):  # pragma: no cover
                _alloc_size = len(_in_arr)
                _out_arr = np.empty(_alloc_size, _in_arr.dtype)
        else:
            def f(_in_arr):  # pragma: no cover
                _alloc_size = _in_arr.shape
                _out_arr = np.empty(_alloc_size, _in_arr.dtype)

        f_block = compile_to_numba_ir(f, {'np': np}, self.typingctx, (if_series_to_array_type(self.typemap[in_arr.name]),),
                                        self.typemap, self.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, [in_arr])
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = assign.target
        return nodes

    def _handle_str_contains(self, assign, lhs, rhs, fname):

        if fname == 'str_contains_regex':
            comp_func = 'hpat.str_ext.contains_regex'
        elif fname == 'str_contains_noregex':
            comp_func = 'hpat.str_ext.contains_noregex'
        else:
            assert False

        func_text = 'def f(str_arr, pat):\n'
        func_text += '  l = len(str_arr)\n'
        func_text += '  S = np.empty(l, dtype=np.bool_)\n'
        func_text += '  for i in numba.parfor.internal_prange(l):\n'
        func_text += '    S[i] = {}(str_arr[i], pat)\n'.format(comp_func)
        func_text += '  return S\n'
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']
        return self._replace_func(f, rhs.args)

    def _handle_df_col_filter(self, assign, lhs, rhs):
        arr_def = guard(get_definition, self.func_ir, rhs.args[2])
        # find df['col2'] = df['col1'][arr]
        # since columns should have the same size, output is filled with NaNs
        # TODO: check for float, make sure col1 and col2 are in the same df
        if (isinstance(arr_def, ir.Expr)  and arr_def.op == 'getitem'
                and is_series_type(self.typemap[arr_def.value.name])
                and self.is_bool_arr(arr_def.index.name)):
            # TODO: handle filter str arr, etc.
            # XXX: can't handle int64 to float64 nans properly since df column
            # bookkeeping is before typing
            return self._replace_func(_column_filter_impl_float,
                [rhs.args[0], rhs.args[1], arr_def.value, arr_def.index], True)
        return [assign]


    def _handle_df_col_calls(self, assign, lhs, rhs, func_name):

        if func_name == 'count':
            return self._replace_func(_column_count_impl, rhs.args)

        if func_name == 'fillna':
            return self._replace_func(_column_fillna_impl, rhs.args)

        if func_name == 'fillna_str_alloc':
            return self._replace_func(_series_fillna_str_alloc_impl, rhs.args)

        if func_name == 'dropna':
            # df.dropna case
            if isinstance(self.typemap[rhs.args[0].name], types.BaseTuple):
                return self._handle_df_dropna(assign, lhs, rhs)
            dtype = self.typemap[rhs.args[0].name].dtype
            if dtype == string_type:
                func = series_replace_funcs['dropna_str_alloc']
            elif isinstance(dtype, types.Float):
                func = series_replace_funcs['dropna_float']
            else:
                # integer case, TODO: bool, date etc.
                func = lambda A: A
            return self._replace_func(func, rhs.args)

        if func_name == 'column_sum':
            return self._replace_func(_column_sum_impl_basic, rhs.args)

        if func_name == 'mean':
            return self._replace_func(_column_mean_impl, rhs.args)

        if func_name == 'var':
            return self._replace_func(_column_var_impl, rhs.args)

        return [assign]

    def _handle_df_dropna(self, assign, lhs, rhs):
        in_typ = self.typemap[rhs.args[0].name]

        in_vars, _ = guard(find_build_sequence, self.func_ir, rhs.args[0])
        in_names = [mk_unique_var(in_vars[i].name).replace('.', '_')
                     for i in range(len(in_vars))]
        out_names = [mk_unique_var(in_vars[i].name).replace('.', '_')
                     for i in range(len(in_vars))]
        str_colnames = [in_names[i] for i, t in enumerate(in_typ.types)
                                                    if t == string_series_type]
        list_str_colnames = [in_names[i] for i, t in enumerate(in_typ.types)
                        if t == SeriesType(types.List(string_type))]
        isna_calls = ['hpat.hiframes.api.isna({}, i)'.format(v) for v in in_names]

        func_text = "def _dropna_impl(arr_tup, inplace):\n"
        func_text += "  ({},) = arr_tup\n".format(", ".join(in_names))
        func_text += "  old_len = len({})\n".format(in_names[0])
        func_text += "  new_len = 0\n"
        for c in str_colnames:
            func_text += "  num_chars_{} = 0\n".format(c)
        func_text += "  for i in numba.parfor.internal_prange(old_len):\n"
        func_text += "    if not ({}):\n".format(' or '.join(isna_calls))
        func_text += "      new_len += 1\n"
        for c in str_colnames:
            func_text += "      num_chars_{} += len({}[i])\n".format(c, c)
        for v, out in zip(in_names, out_names):
            if v in str_colnames:
                func_text += "  {} = hpat.str_arr_ext.pre_alloc_string_array(new_len, num_chars_{})\n".format(out, v)
            elif v in list_str_colnames:
                func_text += "  {} = hpat.str_ext.alloc_list_list_str(new_len)\n".format(out)
            else:
                func_text += "  {} = np.empty(new_len, {}.dtype)\n".format(out, v)
        func_text += "  curr_ind = 0\n"
        func_text += "  for i in numba.parfor.internal_prange(old_len):\n"
        func_text += "    if not ({}):\n".format(' or '.join(isna_calls))
        for v, out in zip(in_names, out_names):
            func_text += "      {}[curr_ind] = {}[i]\n".format(out, v)
        func_text += "      curr_ind += 1\n"
        func_text += "  return ({},)\n".format(", ".join(out_names))

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _dropna_impl = loc_vars['_dropna_impl']
        return self._replace_func(_dropna_impl, rhs.args)

    def _handle_h5_write(self, dset, index, arr):
        if index != slice(None):
            raise ValueError("Only HDF5 write of full array supported")
        assert isinstance(self.typemap[arr.name], types.Array)
        ndim = self.typemap[arr.name].ndim

        func_text = "def _h5_write_impl(dset_id, arr):\n"
        func_text += "  zero_tup = ({},)\n".format(", ".join(["0"]*ndim))
        # TODO: remove after support arr.shape in parallel
        func_text += "  arr_shape = ({},)\n".format(
            ", ".join(["arr.shape[{}]".format(i) for i in range(ndim)]))
        func_text += "  err = hpat.pio_api.h5write(dset_id, np.int32({}), zero_tup, arr_shape, 0, arr)\n".format(ndim)

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _h5_write_impl = loc_vars['_h5_write_impl']
        f_block = compile_to_numba_ir(_h5_write_impl, {'np': np,
                                        'hpat': hpat}, self.typingctx,
                                    (self.typemap[dset.name], self.typemap[arr.name]),
                                    self.typemap, self.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, [dset, arr])
        nodes = f_block.body[:-3]  # remove none return
        return nodes

    def _handle_sorted_by_key(self, rhs):
        """generate a sort function with the given key lambda
        """
        # TODO: handle reverse
        from numba.targets import quicksort
        # get key lambda
        key_lambda_var = dict(rhs.kws)['key']
        key_lambda = guard(
            get_definition, self.func_ir, key_lambda_var)
        if key_lambda is None or not (
                isinstance(key_lambda, ir.Expr)
                and key_lambda.op == 'make_function'):
            raise ValueError("sorted(): lambda for key not found")

        # wrap lambda in function
        def key_lambda_wrapper(A):
            return A
        key_lambda_wrapper.__code__ = key_lambda.code
        key_func = numba.njit(key_lambda_wrapper)

        # make quicksort with new lt
        def lt(a, b):
            return key_func(a) < key_func(b)
        sort_func = quicksort.make_jit_quicksort(
            lt=lt).run_quicksort

        return self._replace_func(
            lambda a: _sort_func(a), rhs.args,
            extra_globals={'_sort_func': numba.njit(sort_func)})

    def _get_const_tup(self, tup_var):
        tup_def = guard(get_definition, self.func_ir, tup_var)
        if isinstance(tup_def, ir.Expr):
            if tup_def.op == 'binop' and tup_def.fn in ('+', operator.add):
                return (self._get_const_tup(tup_def.lhs)
                        + self._get_const_tup(tup_def.rhs))
            if tup_def.op in ('build_tuple', 'build_list'):
                return tup_def.items
        raise ValueError("constant tuple expected")

    def _replace_func(self, func, args, const=False, array_typ_convert=True,
                      pre_nodes=None, extra_globals=None):
        glbls = {'numba': numba, 'np': np, 'hpat': hpat}
        if extra_globals is not None:
            glbls.update(extra_globals)

        # create explicit arg variables for defaults if func has any
        # XXX: inine_closure_call() can't handle defaults properly
        if func.__defaults__:
            defaults = func.__defaults__[len(args):]
            scope = next(iter(self.func_ir.blocks.values())).scope
            loc = scope.loc
            pre_nodes = [] if pre_nodes is None else pre_nodes
            for val in defaults:
                d_var = ir.Var(scope, mk_unique_var('defaults'), loc)
                self.typemap[d_var.name] = numba.typeof(val)
                node = ir.Assign(ir.Const(val, loc), d_var, loc)
                args.append(d_var)
                pre_nodes.append(node)

        arg_typs = tuple(self.typemap[v.name] for v in args)
        if array_typ_convert:
            arg_typs = tuple(if_series_to_array_type(a) for a in arg_typs)
        if const:
            new_args = []
            for i, arg in enumerate(args):
                val = guard(find_const, self.func_ir, arg)
                if val:
                    new_args.append(types.literal(val))
                else:
                    new_args.append(arg_typs[i])
            arg_typs = tuple(new_args)
        return ReplaceFunc(func, arg_typs, args, glbls, pre_nodes)

    def is_bool_arr(self, varname):
        typ = self.typemap[varname]
        return isinstance(if_series_to_array_type(typ), types.npytypes.Array) and typ.dtype == types.bool_

def _fix_typ_undefs(new_typ, old_typ):
    if isinstance(old_typ, (types.Array, SeriesType)):
        assert isinstance(new_typ, (types.Array, SeriesType, StringArrayType,
            types.List))
        if new_typ.dtype == types.undefined:
            return new_typ.copy(old_typ.dtype)
    if isinstance(old_typ, (types.Tuple, types.UniTuple)):
        return types.Tuple([_fix_typ_undefs(t, u)
                                for t, u in zip(new_typ.types, old_typ.types)])
    # TODO: fix List, Set
    return new_typ


# float columns can have regular np.nan


def _column_filter_impl_float(df, cname, B, ind):  # pragma: no cover
    dtype = hpat.hiframes.api.shift_dtype(B.dtype)
    A = np.empty(len(B), dtype)
    for i in numba.parfor.internal_prange(len(A)):
        s = 0
        if ind[i]:
            s = B[i]
        else:
            s = np.nan
        A[i] = s
    hpat.hiframes.api.set_df_col(df, cname, A)
    return


def _column_count_impl(A):  # pragma: no cover
    numba.parfor.init_prange()
    count = 0
    for i in numba.parfor.internal_prange(len(A)):
        if not hpat.hiframes.api.isna(A, i):
            count += 1

    res = count
    return res


def _column_fillna_impl(A, B, fill):  # pragma: no cover
    for i in numba.parfor.internal_prange(len(A)):
        s = B[i]
        if hpat.hiframes.api.isna(B, i):
            s = fill
        A[i] = s

def _series_fillna_str_alloc_impl(B, fill):  # pragma: no cover
    n = len(B)
    num_chars = 0
    # get total chars in new array
    for i in numba.parfor.internal_prange(n):
        s = B[i]
        if hpat.hiframes.api.isna(B, i):
            num_chars += len(fill)
        else:
            num_chars += len(s)
    A = hpat.str_arr_ext.pre_alloc_string_array(n, num_chars)
    hpat.hiframes.api.fillna(A, B, fill)
    return A

def _series_dropna_float_impl(S):  # pragma: no cover
    old_len = len(S)
    new_len = old_len - hpat.hiframes.api.to_series_type(S).isna().sum()
    A = np.empty(new_len, S.dtype)
    curr_ind = 0
    for i in numba.parfor.internal_prange(old_len):
        val = S[i]
        if not np.isnan(val):
            A[curr_ind] = val
            curr_ind += 1

    return A

def _series_dropna_str_alloc_impl(B):  # pragma: no cover
    old_len = len(B)
    # TODO: more efficient null counting
    new_len = old_len - hpat.hiframes.api.to_series_type(B).isna().sum()
    num_chars = hpat.str_arr_ext.num_total_chars(B)
    A = hpat.str_arr_ext.pre_alloc_string_array(new_len, num_chars)
    hpat.str_arr_ext.copy_non_null_offsets(A, B)
    hpat.str_arr_ext.copy_data(A, B)
    return A

# return the nan value for the type (handle dt64)
def _get_nan(val):
    return np.nan

@overload(_get_nan)
def _get_nan_overload(val):
    if isinstance(val, (types.NPDatetime, types.NPTimedelta)):
        nat = val('NaT')
        return lambda val: nat
    # TODO: other types
    return lambda val: np.nan

def _get_type_max_value(dtype):
    return 0

@overload(_get_type_max_value)
def _get_type_max_value_overload(dtype):
    if isinstance(dtype.dtype, (types.NPDatetime, types.NPTimedelta)):
        return lambda dtype: hpat.hiframes.pd_timestamp_ext.integer_to_dt64(
            numba.targets.builtins.get_type_max_value(numba.types.int64))
    return lambda dtype: numba.targets.builtins.get_type_max_value(dtype)

# type(dtype) is called by np.full (used in agg_typer)
@infer_global(type)
class TypeDt64(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if len(args) == 1 and isinstance(args[0], (types.NPDatetime, types.NPTimedelta)):
            classty = types.DType(args[0])
            return signature(classty, *args)

@numba.njit
def _sum_handle_nan(s, count):  # pragma: no cover
    if not count:
        s = hpat.hiframes.hiframes_typed._get_nan(s)
    return s

def _column_sum_impl_basic(A):  # pragma: no cover
    numba.parfor.init_prange()
    # TODO: fix output type
    s = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            s += val

    res = s
    return res


def _column_sum_impl_count(A):  # pragma: no cover
    numba.parfor.init_prange()
    count = 0
    s = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            s += val
            count += 1

    res = hpat.hiframes.hiframes_typed._sum_handle_nan(s, count)
    return res

def _column_prod_impl_basic(A):  # pragma: no cover
    numba.parfor.init_prange()
    # TODO: fix output type
    s = 1
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            s *= val

    res = s
    return res

@numba.njit
def _mean_handle_nan(s, count):  # pragma: no cover
    if not count:
        s = np.nan
    else:
        s = s / count
    return s


def _column_mean_impl(A):  # pragma: no cover
    numba.parfor.init_prange()
    count = 0
    s = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            s += val
            count += 1

    res = hpat.hiframes.hiframes_typed._mean_handle_nan(s, count)
    return res


@numba.njit
def _var_handle_nan(s, count):  # pragma: no cover
    if count <= 1:
        s = np.nan
    else:
        s = s / (count - 1)
    return s


def _column_var_impl(A):  # pragma: no cover
    numba.parfor.init_prange()
    count_m = 0
    m = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            m += val
            count_m += 1

    numba.parfor.init_prange()
    m = hpat.hiframes.hiframes_typed._mean_handle_nan(m, count_m)
    s = 0
    count = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            s += (val - m)**2
            count += 1

    res = hpat.hiframes.hiframes_typed._var_handle_nan(s, count)
    return res

def _column_std_impl(A):  # pragma: no cover
    var = hpat.hiframes.api.var(A)
    return var**0.5

def _column_min_impl(in_arr):  # pragma: no cover
    numba.parfor.init_prange()
    count = 0
    s = hpat.hiframes.hiframes_typed._get_type_max_value(in_arr.dtype)
    for i in numba.parfor.internal_prange(len(in_arr)):
        val = in_arr[i]
        if not hpat.hiframes.api.isna(in_arr, i):
            s = min(s, val)
            count += 1
    res = hpat.hiframes.hiframes_typed._sum_handle_nan(s, count)
    return res

def _column_min_impl_no_isnan(in_arr):  # pragma: no cover
    numba.parfor.init_prange()
    s = numba.targets.builtins.get_type_max_value(numba.types.int64)
    for i in numba.parfor.internal_prange(len(in_arr)):
        val = hpat.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i])
        s = min(s, val)
    return hpat.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(s)

# TODO: fix for dt64
def _column_max_impl(in_arr):  # pragma: no cover
    numba.parfor.init_prange()
    count = 0
    s = numba.targets.builtins.get_type_min_value(in_arr.dtype)
    for i in numba.parfor.internal_prange(len(in_arr)):
        val = in_arr[i]
        if not np.isnan(val):
            s = max(s, val)
            count += 1
    res = hpat.hiframes.hiframes_typed._sum_handle_nan(s, count)
    return res

def _column_max_impl_no_isnan(in_arr):  # pragma: no cover
    numba.parfor.init_prange()
    s = numba.targets.builtins.get_type_min_value(numba.types.int64)
    for i in numba.parfor.internal_prange(len(in_arr)):
        val = in_arr[i]
        s = max(s, hpat.hiframes.pd_timestamp_ext.dt64_to_integer(val))
    return hpat.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(s)

def _column_sub_impl_datetimeindex_timestamp(in_arr, ts):  # pragma: no cover
    numba.parfor.init_prange()
    n = len(in_arr)
    S = numba.unsafe.ndarray.empty_inferred((n,))
    tsint = hpat.hiframes.pd_timestamp_ext.convert_timestamp_to_datetime64(ts)
    for i in numba.parfor.internal_prange(n):
        S[i] = hpat.hiframes.pd_timestamp_ext.integer_to_timedelta64(hpat.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i]) - tsint)
    return S

def _column_describe_impl(A):  # pragma: no cover
    S = hpat.hiframes.api.to_series_type(A)
    a_count = np.float64(hpat.hiframes.api.count(A))
    a_min = S.min()
    a_max = S.max()
    a_mean = hpat.hiframes.api.mean(A)
    a_std = hpat.hiframes.api.var(A)**0.5
    q25 = hpat.hiframes.api.quantile(A, .25)
    q50 = hpat.hiframes.api.quantile(A, .5)
    q75 = hpat.hiframes.api.quantile(A, .75)
    # TODO: pandas returns dataframe, maybe return namedtuple instread of
    # string?
    # TODO: fix string formatting to match python/pandas
    res = "count    " + str(a_count) + "\n"\
        "mean     " + str(a_mean) + "\n"\
        "std      " + str(a_std) + "\n"\
        "min      " + str(a_min) + "\n"\
        "25%      " + str(q25) + "\n"\
        "50%      " + str(q50) + "\n"\
        "75%      " + str(q75) + "\n"\
        "max      " + str(a_max) + "\n"
    return res

def _column_fillna_alloc_impl(S, val):  # pragma: no cover
    # TODO: handle string, etc.
    B = np.empty(len(S), S.dtype)
    hpat.hiframes.api.fillna(B, S, val)
    return B


def _str_contains_regex_impl(str_arr, pat):  # pragma: no cover
    e = hpat.str_ext.compile_regex(pat)
    return hpat.hiframes.api.str_contains_regex(str_arr, e)

def _str_contains_noregex_impl(str_arr, pat):  # pragma: no cover
    return hpat.hiframes.api.str_contains_noregex(str_arr, pat)



# TODO: use online algorithm, e.g. StatFunctions.scala
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
def _column_cov_impl(A, B):  # pragma: no cover
    S1 = hpat.hiframes.api.to_series_type(A)
    S2 = hpat.hiframes.api.to_series_type(B)
    # TODO: check lens
    ma = S1.mean()
    mb = S2.mean()
    # TODO: check aligned nans, (S1.notna() != S2.notna()).any()
    return ((S1-ma)*(S2-mb)).sum()/(S1.count()-1.0)


def _column_corr_impl(A, B):  # pragma: no cover
    S1 = hpat.hiframes.api.to_series_type(A)
    S2 = hpat.hiframes.api.to_series_type(B)
    n = S1.count()
    # TODO: check lens
    ma = S1.sum()
    mb = S2.sum()
    # TODO: check aligned nans, (S1.notna() != S2.notna()).any()
    a = n * ((S1*S2).sum()) - ma * mb
    b1 = n * (S1**2).sum() - ma**2
    b2 = n * (S2**2).sum() - mb**2
    # TODO: np.clip
    # TODO: np.true_divide?
    return a / np.sqrt(b1*b2)


def _series_append_single_impl(arr, other):
    return hpat.hiframes.api.concat((arr, other))

def _series_append_tuple_impl(arr, other):
    tup_other = hpat.hiframes.api.to_const_tuple(other)
    arrs = (arr,) + tup_other
    c_arrs = hpat.hiframes.api.to_const_tuple(arrs)
    return hpat.hiframes.api.concat(c_arrs)

def _series_isna_impl(arr):
    numba.parfor.init_prange()
    n = len(arr)
    out_arr = np.empty(n, np.bool_)
    for i in numba.parfor.internal_prange(n):
        out_arr[i] = hpat.hiframes.api.isna(arr, i)
    return out_arr

def _series_astype_str_impl(arr):
    n = len(arr)
    num_chars = 0
    # get total chars in new array
    for i in numba.parfor.internal_prange(n):
        s = arr[i]
        num_chars += len(str(s))  # TODO: check NA

    A = hpat.str_arr_ext.pre_alloc_string_array(n, num_chars)
    for i in numba.parfor.internal_prange(n):
        s = arr[i]
        A[i] = str(s)  # TODO: check NA
    return A


# TODO: refactor regex and noregex
def _str_replace_regex_impl(str_arr, pat, val):
    numba.parfor.init_prange()
    e = hpat.str_ext.compile_regex(unicode_to_std_str(pat))
    val = unicode_to_std_str(val)
    n = len(str_arr)
    n_total_chars = 0
    str_list = hpat.str_ext.alloc_str_list(n)
    for i in numba.parfor.internal_prange(n):
        in_str = unicode_to_std_str(str_arr[i])
        out_str = std_str_to_unicode(
            hpat.str_ext.str_replace_regex(in_str, e, val))
        str_list[i] = out_str
        n_total_chars += len(out_str)
    numba.parfor.init_prange()
    out_arr = pre_alloc_string_array(n, n_total_chars)
    for i in numba.parfor.internal_prange(n):
        _str = str_list[i]
        out_arr[i] = _str
    return out_arr


def _str_replace_noregex_impl(str_arr, pat, val):
    numba.parfor.init_prange()
    e = unicode_to_std_str(pat)
    val = unicode_to_std_str(val)
    n = len(str_arr)
    n_total_chars = 0
    str_list = hpat.str_ext.alloc_str_list(n)
    for i in numba.parfor.internal_prange(n):
        in_str = unicode_to_std_str(str_arr[i])
        out_str = std_str_to_unicode(
            hpat.str_ext.str_replace_noregex(in_str, e, val))
        str_list[i] = out_str
        n_total_chars += len(out_str)
    numba.parfor.init_prange()
    out_arr = pre_alloc_string_array(n, n_total_chars)
    for i in numba.parfor.internal_prange(n):
        _str = str_list[i]
        out_arr[i] = _str
    return out_arr


@numba.njit
def lt_f(a, b):
    return a < b

@numba.njit
def gt_f(a, b):
    return a > b

series_replace_funcs = {
    'sum': _column_sum_impl_basic,
    'prod': _column_prod_impl_basic,
    'count': _column_count_impl,
    'mean': _column_mean_impl,
    'max': defaultdict(lambda: _column_max_impl, [(numba.types.scalars.NPDatetime('ns'), _column_max_impl_no_isnan)]),
    'min': defaultdict(lambda: _column_min_impl, [(numba.types.scalars.NPDatetime('ns'), _column_min_impl_no_isnan)]),
    'var': _column_var_impl,
    'std': _column_std_impl,
    'nunique': lambda A: hpat.hiframes.api.nunique(A),
    'unique': lambda A: hpat.hiframes.api.unique(A),
    'describe': _column_describe_impl,
    'fillna_alloc': _column_fillna_alloc_impl,
    'fillna_str_alloc': _series_fillna_str_alloc_impl,
    'dropna_float': _series_dropna_float_impl,
    'dropna_str_alloc': _series_dropna_str_alloc_impl,
    'shift': lambda A, shift: hpat.hiframes.rolling.shift(A, shift, False),
    'shift_default': lambda A: hpat.hiframes.rolling.shift(A, 1, False),
    'pct_change': lambda A, shift: hpat.hiframes.rolling.pct_change(A, shift, False),
    'pct_change_default': lambda A: hpat.hiframes.rolling.pct_change(A, 1, False),
    'str_contains_regex': _str_contains_regex_impl,
    'str_contains_noregex': _str_contains_noregex_impl,
    'abs': lambda A: np.abs(A),  # TODO: timedelta
    'cov': _column_cov_impl,
    'corr': _column_corr_impl,
    'append_single': _series_append_single_impl,
    'append_tuple': _series_append_tuple_impl,
    'isna': _series_isna_impl,
    # isnull is just alias of isna
    'isnull': _series_isna_impl,
    'astype_str': _series_astype_str_impl,
    'nlargest': lambda A, k: hpat.hiframes.api.nlargest(A, k, True, gt_f),
    'nlargest_default': lambda A: hpat.hiframes.api.nlargest(A, 5, True, gt_f),
    'nsmallest': lambda A, k: hpat.hiframes.api.nlargest(A, k, False, lt_f),
    'nsmallest_default': lambda A: hpat.hiframes.api.nlargest(A, 5, False, lt_f),
    'head': lambda A, k: A[:k],
    'head_default': lambda A: A[:5],
    'median': lambda A: hpat.hiframes.api.median(A),
    # TODO: handle NAs in argmin/argmax
    'idxmin': lambda A: A.argmin(),
    'idxmax': lambda A: A.argmax(),
}

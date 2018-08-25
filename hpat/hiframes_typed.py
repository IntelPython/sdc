from __future__ import print_function, division, absolute_import

import numpy as np
from collections import namedtuple
import warnings
import numba
from numba import ir, ir_utils, types
from numba.ir_utils import (replace_arg_nodes, compile_to_numba_ir,
                            find_topo_order, gen_np_call, get_definition, guard,
                            find_callname, mk_alloc, find_const, is_setitem,
                            is_getitem, mk_unique_var, dprint_func_ir,
                            build_definitions)
from numba.inline_closurecall import inline_closure_call
from numba.typing.templates import Signature, bound_function, signature
from numba.typing.arraydecl import ArrayAttribute
import hpat
from hpat.utils import debug_prints, inline_new_blocks
from hpat.str_ext import string_type
from hpat.str_arr_ext import string_array_type, StringArrayType, is_str_arr_typ
from hpat.pd_series_ext import (SeriesType, string_series_type,
    series_to_array_type, BoxedSeriesType, dt_index_series_type,
    if_series_to_array_type, if_series_to_unbox, is_series_type,
    series_str_methods_type, SeriesRollingType)

ReplaceFunc = namedtuple("ReplaceFunc", ["func", "arg_types", "args", "glbls"])

LARGE_WIN_SIZE = 10


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
                if isinstance(inst, ir.Assign):
                    out_nodes = self._run_assign(inst)
                    if isinstance(out_nodes, list):
                        new_body.extend(out_nodes)
                    if isinstance(out_nodes, ReplaceFunc):
                        rp_func = out_nodes
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
                assert (typ.typing_key.startswith('array.')
                    or typ.typing_key.startswith('series.'))
                # skip if series.func since it is replaced here
                if typ.typing_key.startswith('series.'):
                    continue
                this = series_to_array_type(typ.this)
                attr = typ.typing_key[len('array.'):]
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
            sig.return_type = if_series_to_array_type(sig.return_type)
            sig.args = tuple(map(if_series_to_array_type, sig.args))
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
            if (rhs.op == 'getattr' and isinstance(self.typemap[rhs.value.name], SeriesType)
                    and rhs.attr == 'values'):
                # simply return the column
                assign.value = rhs.value
                return [assign]

            res = self._handle_string_array_expr(lhs, rhs, assign)
            if res is not None:
                return res

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
                    res = hpat.pd_timestamp_ext.convert_datetime64_to_timestamp(s)

                assert self.typemap[ind_var.name] == types.intp
                f_block = compile_to_numba_ir(f, {'numba': numba, 'np': np,
                                                'hpat': hpat}, self.typingctx,
                                            (if_series_to_array_type(self.typemap[in_arr.name]), types.intp),
                                            self.typemap, self.calltypes).blocks.popitem()[1]
                replace_arg_nodes(f_block, [in_arr, ind_var])
                nodes = f_block.body[:-3]  # remove none return
                nodes[-1].target = assign.target
                return nodes

            if rhs.op == 'call':

                fdef = guard(find_callname, self.func_ir, rhs)
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

                if fdef == ('DatetimeIndex', 'pandas'):
                    return self._run_pd_DatetimeIndex(assign, assign.target, rhs)

                if func_mod == 'hpat.hiframes_api':
                    return self._run_call_hiframes(assign, assign.target, rhs, func_name)

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
        if (func_name == 'fix_df_array'
                and isinstance(self.typemap[rhs.args[0].name],
                               (types.Array, StringArrayType))):
            assign.value = rhs.args[0]
            return [assign]

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

        return self._handle_df_col_calls(assign, lhs, rhs, func_name)

    def _run_call_series(self, assign, lhs, rhs, series_var, func_name):
        # single arg functions
        if func_name in ['sum', 'count', 'mean', 'var', 'std', 'min', 'max',
                         'nunique', 'describe', 'abs', 'str.len', 'isna',
                         'isnull', 'median']:
            if rhs.args or rhs.kws:
                raise ValueError("unsupported Series.{}() arguments".format(
                    func_name))
            func = series_replace_funcs[func_name]
            # TODO: handle skipna, min_count arguments
            return self._replace_func(func, [series_var])

        if func_name == 'quantile':
            return self._replace_func(
                lambda A, q: hpat.hiframes_api.quantile(A, q),
                [series_var, rhs.args[0]]
            )

        if func_name == 'fillna':
            return self._run_call_series_fillna(assign, lhs, rhs, series_var)

        if func_name == 'dropna':
            return self._run_call_series_dropna(assign, lhs, rhs, series_var)

        if func_name in ('shift', 'pct_change'):
            # TODO: support default period argument
            shift_const = rhs.args[0]
            func = series_replace_funcs[func_name]
            return self._replace_func(func, [series_var, shift_const])

        if func_name == 'nlargest':
            # TODO: support default n=5 argument
            n_arg = rhs.args[0]
            func = series_replace_funcs[func_name]
            return self._replace_func(func, [series_var, n_arg])

        if func_name in ('cov', 'corr'):
            S2 = rhs.args[0]
            func = series_replace_funcs[func_name]
            return self._replace_func(func, [series_var, S2])

        if func_name == 'str.contains':
            return self._handle_series_str_contains(rhs, series_var)

        if func_name == 'rolling':
            # XXX: remove rolling setup call, assuming still available in definitions
            return []

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

        # functions we revert to Numpy for now, otherwise warning
        # TODO: handle series-specific cases for this funcs
        if (not func_name.startswith("values.") and func_name
                not in ('copy', 'cumsum', 'cumprod', 'take', 'astype')):
            warnings.warn("unknown Series call {}, reverting to Numpy".format(
                func_name))

        return [assign]

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
                    hpat.hiframes_api.fillna_str_alloc(A, fill)
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
                    lambda a,b,c: hpat.hiframes_api.fillna(a,b,c),
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
                res = hpat.hiframes_api.dropna(A)

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
        func_text += "    S[i] = map_func(t)\n"
        if out_typ == hpat.pd_timestamp_ext.datetime_date_type:
            func_text += "  ret = hpat.hiframes_api.to_date_series_type(S)\n"
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
        window, center = self._get_rolling_setup_args(rolling_call)

        nodes = self._gen_rolling_call(
            rhs.args, series_var, window, center, func_name, lhs)
        return nodes

    def _get_rolling_setup_args(self, rhs):
        """
        Handle Series rolling calls like:
          r = df.column.rolling(3)
        """
        center = False
        kws = dict(rhs.kws)
        if rhs.args:
            window = rhs.args[0]
        elif 'window' in kws:
            window = kws['window']
        else:  # pragma: no cover
            raise ValueError("window argument to rolling() required")
        window_const = guard(find_const, self.func_ir, window)
        window = window_const if window_const is not None else window
        if 'center' in kws:
            center_const = guard(find_const, self.func_ir, kws['center'])
            center = center_const if center_const is not None else center
        return window, center

    def _gen_rolling_call(self, args, col_var, win_size, center, func, out_var):
        loc = col_var.loc
        scope = col_var.scope
        if func == 'apply':
            if len(args) != 1:  # pragma: no cover
                raise ValueError("One argument expected for rolling apply")
            kernel_func = guard(get_definition, self.func_ir, args[0])
        elif func in ['sum', 'mean', 'min', 'max', 'std', 'var']:
            if len(args) != 0:  # pragma: no cover
                raise ValueError("No argument expected for rolling {}".format(
                    func))
            g_pack = "np"
            if func in ['std', 'var', 'mean']:
                g_pack = "hpat.hiframes_api"
            if isinstance(win_size, int) and win_size < LARGE_WIN_SIZE:
                # unroll if size is less than 5
                kernel_args = ','.join(['a[{}]'.format(-i)
                                        for i in range(win_size)])
                kernel_expr = '{}.{}(np.array([{}]))'.format(
                    g_pack, func, kernel_args)
                if func == 'sum':  # simplify sum
                    kernel_expr = '+'.join(['a[{}]'.format(-i)
                                            for i in range(win_size)])
            else:
                kernel_expr = '{}.{}(a[(-w+1):1])'.format(g_pack, func)
            func_text = 'def g(a, w):\n  return {}\n'.format(kernel_expr)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            kernel_func = loc_vars['g']

        init_nodes = []

        if isinstance(win_size, int):
            win_size_var = ir.Var(scope, mk_unique_var("win_size"), loc)
            self.typemap[win_size_var.name] = types.intp
            init_nodes.append(
                ir.Assign(ir.Const(win_size, loc), win_size_var, loc))
            win_size = win_size_var

        index_offsets, win_tuple, option_nodes = self._gen_rolling_init(
            win_size, func, center)

        init_nodes += option_nodes
        other_args = [win_size]
        if func == 'apply':
            other_args = None
        options = {'neighborhood': win_tuple}
        fir_globals = self.func_ir.func_id.func.__globals__
        stencil_nodes = self._gen_stencil_call(init_nodes, col_var, out_var, kernel_func,
                                         index_offsets, fir_globals, other_args,
                                         options)

        def f(A, w):  # pragma: no cover
            A[0:w - 1] = np.nan
        fargs = [out_var, win_size]
        arg_typs = tuple(if_series_to_array_type(self.typemap[v.name]) for v in fargs)
        f_block = compile_to_numba_ir(f, {'np': np}, self.typingctx, arg_typs,
                                        self.typemap, self.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, fargs)
        setitem_nodes = f_block.body[:-3]  # remove none return

        if center:
            def f1(A, w):  # pragma: no cover
                A[0:w // 2] = np.nan

            def f2(A, w):  # pragma: no cover
                n = len(A)
                A[n - (w // 2):n] = np.nan
            f_block = compile_to_numba_ir(f1, {'np': np}, self.typingctx, arg_typs,
                                        self.typemap, self.calltypes).blocks.popitem()[1]
            replace_arg_nodes(f_block, [out_var, win_size])
            setitem_nodes1 = f_block.body[:-3]  # remove none return
            f_block = compile_to_numba_ir(f2, {'np': np}, self.typingctx, arg_typs,
                                        self.typemap, self.calltypes).blocks.popitem()[1]
            replace_arg_nodes(f_block, [out_var, win_size])
            setitem_nodes2 = f_block.body[:-3]  # remove none return
            setitem_nodes = setitem_nodes1 + setitem_nodes2

        return stencil_nodes + setitem_nodes

    def _gen_rolling_init(self, win_size, func, center):
        nodes = []
        right_length = 0
        scope = win_size.scope
        loc = win_size.loc
        right_length = ir.Var(scope, mk_unique_var('zero_var'), scope)
        self.typemap[right_length.name] = types.intp
        nodes.append(ir.Assign(ir.Const(0, loc), right_length, win_size.loc))

        def f(w):  # pragma: no cover
            return -w + 1

        fargs = [win_size]
        arg_typs = tuple(if_series_to_array_type(self.typemap[v.name]) for v in fargs)
        f_block = compile_to_numba_ir(f, {}, self.typingctx, arg_typs,
                                        self.typemap, self.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, fargs)
        nodes.extend(f_block.body[:-2])  # remove none return
        left_length = nodes[-1].target

        if center:
            def f(w):  # pragma: no cover
                return -(w // 2)
            f_block = compile_to_numba_ir(f, {}, self.typingctx, arg_typs,
                                        self.typemap, self.calltypes).blocks.popitem()[1]
            replace_arg_nodes(f_block, [win_size])
            nodes.extend(f_block.body[:-2])  # remove none return
            left_length = nodes[-1].target

            def f(w):  # pragma: no cover
                return (w // 2)
            f_block = compile_to_numba_ir(f, {}, self.typingctx, arg_typs,
                                        self.typemap, self.calltypes).blocks.popitem()[1]
            replace_arg_nodes(f_block, [win_size])
            nodes.extend(f_block.body[:-2])  # remove none return
            right_length = nodes[-1].target

        def f(a, b):  # pragma: no cover
            return ((a, b),)
        fargs = [left_length, right_length]
        arg_typs = tuple(if_series_to_array_type(self.typemap[v.name]) for v in fargs)
        f_block = compile_to_numba_ir(f, {}, self.typingctx, arg_typs,
                                        self.typemap, self.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, fargs)
        nodes.extend(f_block.body[:-2])  # remove none return
        win_tuple = nodes[-1].target

        index_offsets = [right_length]

        if func == 'apply':
            index_offsets = [left_length]

        def f(a):  # pragma: no cover
            return (a,)

        fargs = index_offsets
        arg_typs = tuple(if_series_to_array_type(self.typemap[v.name]) for v in fargs)
        f_block = compile_to_numba_ir(f, {}, self.typingctx, arg_typs,
                                        self.typemap, self.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, index_offsets)
        nodes.extend(f_block.body[:-2])  # remove none return
        index_offsets = nodes[-1].target

        return index_offsets, win_tuple, nodes

    def _gen_stencil_call(self, init_nodes, in_arr, out_arr, kernel_func, index_offsets, fir_globals,
                        other_args=None, options=None):
        if other_args is None:
            other_args = []
        if options is None:
            options = {}
        if index_offsets != [0]:
            options['index_offsets'] = index_offsets
        scope = in_arr.scope
        loc = in_arr.loc
        stencil_nodes = []

        # alloc output
        stencil_nodes += self._gen_empty_like(in_arr, out_arr)

        # make kernel_var mk_function expr
        kernel_var = ir.Var(scope, mk_unique_var("kernel_var"), scope)
        self.typemap[kernel_var.name] = types.pyfunc_type
        if not isinstance(kernel_func, ir.Expr):
            kernel_func = ir.Expr.make_function("kernel", kernel_func.__code__,
                                                kernel_func.__closure__,
                                                kernel_func.__defaults__, loc)
        stencil_nodes.append(ir.Assign(kernel_func, kernel_var, loc))

        # compile vanilla function without types
        def f(A, B, f):  # pragma: no cover
            numba.stencil(f)(A, out=B)

        f_ir = compile_to_numba_ir(f, {'numba': numba,
            'hpat': hpat, 'np': np},)
        f_block = f_ir.blocks[min(f_ir.blocks.keys())]
        fargs = [in_arr, out_arr, kernel_var]
        replace_arg_nodes(f_block, fargs)

        # create arguments with array + other args for type inference
        other_arg_vars = [ir.Var(scope, mk_unique_var("w"), loc) for i in range(len(other_args))]
        arg_nodes = ([ir.Assign(ir.Arg('A', 0, loc), stencil_nodes[0].value, loc)]
            + [ir.Assign(ir.Arg('w'+str(i), i+1, loc), other_arg_vars[i], loc) for i in range(len(other_args))])
        f_block.body = arg_nodes + init_nodes + stencil_nodes + f_block.body

        # fix stencil call args
        setup_call = f_block.body[-5].value
        stencil_call = f_block.body[-4].value
        setup_call.kws = list(options.items())
        stencil_call.args += other_args

        # run inlining and type inference
        f_ir.arg_names = ['A'] + ["w"+str(i) for i in range(len(other_args))]
        f_ir.arg_count = 1 + len(other_args)
        f_ir._definitions = build_definitions(f_ir.blocks)
        inline_pass = numba.inline_closurecall.InlineClosureCallPass(
        f_ir, numba.targets.cpu.ParallelOptions(False))
        inline_pass.run()

        fargs = [in_arr] + other_args
        arg_typs = tuple(if_series_to_array_type(self.typemap[v.name]) for v in fargs)
        f_typemap, f_return_type, f_calltypes = numba.compiler.type_inference_stage(
                self.typingctx, f_ir, arg_typs, None)
        # remove argument entries like arg.a from typemap
        arg_names = [vname for vname in f_typemap if vname.startswith("arg.")]
        for a in arg_names:
            f_typemap.pop(a)
        self.typemap.update(f_typemap)
        self.calltypes.update(f_calltypes)
        replace_arg_nodes(f_block, fargs)

        return f_block.body[:-3]

    def _gen_empty_like(self, in_arr, out_arr):
        def f(A):  # pragma: no cover
            dtype = hpat.hiframes_api.shift_dtype(A.dtype)
            B = np.empty(A.shape, dtype)

        fargs = [in_arr]
        arg_typs = tuple(if_series_to_array_type(self.typemap[v.name]) for v in fargs)
        f_block = compile_to_numba_ir(f, {'hpat': hpat, 'np': np}, self.typingctx, arg_typs,
                                        self.typemap, self.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, fargs)
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = out_arr
        return nodes

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

        def f(str_arr):
            numba.parfor.init_prange()
            n = len(str_arr)
            S = numba.unsafe.ndarray.empty_inferred((n,))
            for i in numba.parfor.internal_prange(n):
                S[i] = hpat.pd_timestamp_ext.parse_datetime_str(str_arr[i])
            return S

        return self._replace_func(f, [data])

    def _is_dt_index_binop(self, rhs):
        if rhs.op != 'binop' or rhs.fn not in ('==', '!=', '>=', '>', '<=', '<'):
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

        if (self.typemap[arg1.name] not in allowed_types
                or self.typemap[arg2.name] not in allowed_types):
            raise ValueError("DatetimeIndex operation not supported")

        func_text = 'def f(arg1, arg2):\n'
        if self.typemap[arg1.name] == dt_index_series_type:
            func_text += '  dt_index, _str = arg1, arg2\n'
            comp = 'dt_index[i] {} other'.format(rhs.fn)
        else:
            func_text += '  dt_index, _str = arg2, arg1\n'
            comp = 'other {} dt_index[i]'.format(rhs.fn)
        func_text += '  l = len(dt_index)\n'
        func_text += '  other = hpat.pd_timestamp_ext.parse_datetime_str(_str)\n'
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
                and rhs.fn in ['==', '!=', '>=', '>', '<=', '<']
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

            func_text = 'def f(A, B):\n'
            func_text += '  l = {}\n'.format(len_call)
            func_text += '  S = np.empty(l, dtype=np.bool_)\n'
            func_text += '  for i in numba.parfor.internal_prange(l):\n'
            func_text += '    S[i] = {} {} {}\n'.format(arg1_access, rhs.fn,
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

    def _get_const_tup(self, tup_var):
        tup_def = guard(get_definition, self.func_ir, tup_var)
        if isinstance(tup_def, ir.Expr):
            if tup_def.op == 'binop' and tup_def.fn == '+':
                return (self._get_const_tup(tup_def.lhs)
                        + self._get_const_tup(tup_def.rhs))
            if tup_def.op in ('build_tuple', 'build_list'):
                return tup_def.items
        raise ValueError("constant tuple expected")

    def _replace_func(self, func, args, const=False, array_typ_convert=True):
        glbls = {'numba': numba, 'np': np, 'hpat': hpat}
        arg_typs = tuple(self.typemap[v.name] for v in args)
        if array_typ_convert:
            arg_typs = tuple(if_series_to_array_type(a) for a in arg_typs)
        if const:
            new_args = []
            for i, arg in enumerate(args):
                val = guard(find_const, self.func_ir, arg)
                if val:
                    new_args.append(types.Const(val))
                else:
                    new_args.append(arg_typs[i])
            arg_typs = tuple(new_args)
        return ReplaceFunc(func, arg_typs, args, glbls)

    def is_bool_arr(self, varname):
        typ = self.typemap[varname]
        return isinstance(if_series_to_array_type(typ), types.npytypes.Array) and typ.dtype == types.bool_

def _fix_typ_undefs(new_typ, old_typ):
    if isinstance(old_typ, (types.Array, SeriesType)):
        assert isinstance(new_typ, (types.Array, SeriesType))
        if new_typ.dtype == types.undefined:
            return new_typ.copy(old_typ.dtype)
    if isinstance(old_typ, (types.Tuple, types.UniTuple)):
        return types.Tuple([_fix_typ_undefs(t, u)
                                for t, u in zip(new_typ.types, old_typ.types)])
    # TODO: fix List, Set
    return new_typ


# float columns can have regular np.nan


def _column_filter_impl_float(df, cname, B, ind):  # pragma: no cover
    dtype = hpat.hiframes_api.shift_dtype(B.dtype)
    A = np.empty(len(B), dtype)
    for i in numba.parfor.internal_prange(len(A)):
        s = 0
        if ind[i]:
            s = B[i]
        else:
            s = np.nan
        A[i] = s
    hpat.hiframes_api.set_df_col(df, cname, A)
    return


def _column_count_impl(A):  # pragma: no cover
    numba.parfor.init_prange()
    count = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            count += 1

    res = count
    return res


def _column_fillna_impl(A, B, fill):  # pragma: no cover
    for i in numba.parfor.internal_prange(len(A)):
        s = B[i]
        if hpat.hiframes_api.isna(B, i):
            s = fill
        A[i] = s

def _series_fillna_str_alloc_impl(B, fill):  # pragma: no cover
    n = len(B)
    num_chars = 0
    # get total chars in new array
    for i in numba.parfor.internal_prange(n):
        s = B[i]
        if hpat.hiframes_api.isna(B, i):
            num_chars += len(fill)
        else:
            num_chars += len(s)
    A = hpat.str_arr_ext.pre_alloc_string_array(n, num_chars)
    hpat.hiframes_api.fillna(A, B, fill)
    return A

def _series_dropna_float_impl(S):  # pragma: no cover
    old_len = len(S)
    new_len = old_len - hpat.hiframes_api.to_series_type(S).isna().sum()
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
    new_len = old_len - hpat.hiframes_api.to_series_type(B).isna().sum()
    num_chars = hpat.str_arr_ext.num_total_chars(B)
    A = hpat.str_arr_ext.pre_alloc_string_array(new_len, num_chars)
    hpat.str_arr_ext.copy_non_null_offsets(A, B)
    hpat.str_arr_ext.copy_data(A, B)
    return A


@numba.njit
def _sum_handle_nan(s, count):  # pragma: no cover
    if not count:
        s = np.nan
    return s

def _column_sum_impl_basic(A):  # pragma: no cover
    numba.parfor.init_prange()
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

    res = hpat.hiframes_typed._sum_handle_nan(s, count)
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

    res = hpat.hiframes_typed._mean_handle_nan(s, count)
    return res


@numba.njit
def _var_handle_nan(s, count):  # pragma: no cover
    if count <= 1:
        s = np.nan
    else:
        s = s / (count - 1)
    return s


def _column_var_impl(A):  # pragma: no cover
    count_m = 0
    m = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            m += val
            count_m += 1

    m = hpat.hiframes_typed._mean_handle_nan(m, count_m)
    s = 0
    count = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            s += (val - m)**2
            count += 1

    res = hpat.hiframes_typed._var_handle_nan(s, count)
    return res

def _column_std_impl(A):  # pragma: no cover
    var = hpat.hiframes_api.var(A)
    return var**0.5

def _column_min_impl(in_arr):  # pragma: no cover
    numba.parfor.init_prange()
    count = 0
    s = numba.targets.builtins.get_type_max_value(in_arr.dtype)
    for i in numba.parfor.internal_prange(len(in_arr)):
        val = in_arr[i]
        if not np.isnan(val):
            s = min(s, val)
            count += 1
    res = hpat.hiframes_typed._sum_handle_nan(s, count)
    return res

def _column_max_impl(in_arr):  # pragma: no cover
    numba.parfor.init_prange()
    count = 0
    s = numba.targets.builtins.get_type_min_value(in_arr.dtype)
    for i in numba.parfor.internal_prange(len(in_arr)):
        val = in_arr[i]
        if not np.isnan(val):
            s = max(s, val)
            count += 1
    res = hpat.hiframes_typed._sum_handle_nan(s, count)
    return res


def _column_describe_impl(A):  # pragma: no cover
    S = hpat.hiframes_api.to_series_type(A)
    a_count = np.float64(hpat.hiframes_api.count(A))
    a_min = S.min()
    a_max = S.max()
    a_mean = hpat.hiframes_api.mean(A)
    a_std = hpat.hiframes_api.var(A)**0.5
    q25 = hpat.hiframes_api.quantile(A, .25)
    q50 = hpat.hiframes_api.quantile(A, .5)
    q75 = hpat.hiframes_api.quantile(A, .75)
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
    hpat.hiframes_api.fillna(B, S, val)
    return B

def _column_shift_impl(A, shift):  # pragma: no cover
    # TODO: alloc_shift
    #B = hpat.hiframes_api.alloc_shift(A)
    B = np.empty_like(A)
    #numba.stencil(lambda a, b: a[-b], out=B, neighborhood=((-shift, 1-shift), ))(A, shift)
    numba.stencil(lambda a, b: a[-b], out=B)(A, shift)
    B[0:shift] = np.nan
    return B


def _column_pct_change_impl(A, shift):  # pragma: no cover
    # TODO: alloc_shift
    #B = hpat.hiframes_api.alloc_shift(A)
    B = np.empty_like(A)
    #numba.stencil(lambda a, b: a[-b], out=B, neighborhood=((-shift, 1-shift), ))(A, shift)
    numba.stencil(lambda a, b: (a[0]-a[-b])/a[-b], out=B)(A, shift)
    B[0:shift] = np.nan
    return B


def _str_contains_regex_impl(str_arr, pat):  # pragma: no cover
    e = hpat.str_ext.compile_regex(pat)
    return hpat.hiframes_api.str_contains_regex(str_arr, e)

def _str_contains_noregex_impl(str_arr, pat):  # pragma: no cover
    return hpat.hiframes_api.str_contains_noregex(str_arr, pat)

def _str_len_impl(str_arr):
    numba.parfor.init_prange()
    n = len(str_arr)
    out_arr = np.empty(n, np.int64)
    for i in numba.parfor.internal_prange(n):
        val = str_arr[i]
        out_arr[i] = len(val)
    return out_arr



# TODO: use online algorithm, e.g. StatFunctions.scala
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
def _column_cov_impl(A, B):  # pragma: no cover
    S1 = hpat.hiframes_api.to_series_type(A)
    S2 = hpat.hiframes_api.to_series_type(B)
    # TODO: check lens
    ma = S1.mean()
    mb = S2.mean()
    # TODO: check aligned nans, (S1.notna() != S2.notna()).any()
    return ((S1-ma)*(S2-mb)).sum()/(S1.count()-1.0)


def _column_corr_impl(A, B):  # pragma: no cover
    S1 = hpat.hiframes_api.to_series_type(A)
    S2 = hpat.hiframes_api.to_series_type(B)
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
    return hpat.hiframes_api.concat((arr, other))

def _series_append_tuple_impl(arr, other):
    tup_other = hpat.hiframes_api.to_const_tuple(other)
    arrs = (arr,) + tup_other
    c_arrs = hpat.hiframes_api.to_const_tuple(arrs)
    return hpat.hiframes_api.concat(c_arrs)

def _series_isna_impl(arr):
    numba.parfor.init_prange()
    n = len(arr)
    out_arr = np.empty(n, np.bool_)
    for i in numba.parfor.internal_prange(n):
        out_arr[i] = hpat.hiframes_api.isna(arr, i)
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


series_replace_funcs = {
    'sum': _column_sum_impl_basic,
    'count': _column_count_impl,
    'mean': _column_mean_impl,
    'max': _column_max_impl,
    'min': _column_min_impl,
    'var': _column_var_impl,
    'std': _column_std_impl,
    'nunique': lambda A: hpat.hiframes_api.nunique(A),
    'describe': _column_describe_impl,
    'fillna_alloc': _column_fillna_alloc_impl,
    'fillna_str_alloc': _series_fillna_str_alloc_impl,
    'dropna_float': _series_dropna_float_impl,
    'dropna_str_alloc': _series_dropna_str_alloc_impl,
    'shift': _column_shift_impl,
    'pct_change': _column_pct_change_impl,
    'str_contains_regex': _str_contains_regex_impl,
    'str_contains_noregex': _str_contains_noregex_impl,
    'abs': lambda A: np.abs(A),  # TODO: timedelta
    'cov': _column_cov_impl,
    'corr': _column_corr_impl,
    'str.len': _str_len_impl,
    'append_single': _series_append_single_impl,
    'append_tuple': _series_append_tuple_impl,
    'isna': _series_isna_impl,
    # isnull is just alias of isna
    'isnull': _series_isna_impl,
    'astype_str': _series_astype_str_impl,
    'nlargest': lambda A, k: hpat.hiframes_api.nlargest(A, k),
    'median': lambda A: hpat.hiframes_api.median(A),
}

from __future__ import print_function, division, absolute_import

import numba
from numba import ir, ir_utils, types
from numba import compiler as numba_compiler

from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir, remove_dead, mk_alloc, remove_dels,
                            get_name_var_table, replace_var_names,
                            add_offset_to_labels, get_ir_of_code,
                            compile_to_numba_ir, replace_arg_nodes,
                            find_callname, guard, require, get_definition)

import hpat
from hpat import hiframes_api, utils, parquet_pio, config, hiframes_filter
from hpat.utils import get_constant, NOT_CONSTANT, get_definitions
import numpy as np
import math
from hpat.parquet_pio import ParquetHandler


df_col_funcs = ['shift', 'pct_change', 'fillna', 'sum', 'mean', 'var', 'std',
                                                'quantile', 'count', 'describe']
LARGE_WIN_SIZE = 10

def remove_hiframes(rhs, lives, call_list):
    # used in stencil generation of rolling
    if len(call_list) == 1 and call_list[0] in [min, max, abs]:
        return True
    # used in stencil generation of rolling
    if (len(call_list) == 1 and isinstance(call_list[0],
            numba.targets.registry.CPUDispatcher)
            and call_list[0].py_func == numba.stencilparfor._compute_last_ind):
        return True
    # used in stencil generation of rolling
    if call_list == ['ceil', math]:
        return True
    if (len(call_list) == 3 and call_list[1:] == ['hiframes_api', hpat] and
            call_list[0] in ['fix_df_array', 'fix_rolling_array']):
        return True
    if (len(call_list) == 3 and call_list[1:] == ['hiframes_typed', hpat] and
            call_list[0]
                in ['_sum_handle_nan', '_mean_handle_nan', '_var_handle_nan']):
        return True
    return False

numba.ir_utils.remove_call_handlers.append(remove_hiframes)

class HiFrames(object):
    """analyze and transform hiframes calls"""
    def __init__(self, func_ir, typingctx, args, _locals):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.args = args
        self.locals = _locals
        ir_utils._max_label = max(func_ir.blocks.keys())

        # rolling call name -> [column_varname, win_size]
        self.rolling_calls = {}

        # df_var -> {col1:col1_var ...}
        self.df_vars = {}
        # arrays that are df columns actually (pd.Series)
        self.df_cols = set()
        self.arrow_tables = {}
        self.reverse_copies = {}
        self.pq_handler = ParquetHandler(func_ir, typingctx, args, _locals, self.reverse_copies)

    def run(self):
        dprint_func_ir(self.func_ir, "starting hiframes")
        topo_order = find_topo_order(self.func_ir.blocks)
        for label in topo_order:
            self._get_reverse_copies(self.func_ir.blocks[label].body)
            new_body = []
            for inst in self.func_ir.blocks[label].body:
                # df['col'] = arr
                if isinstance(inst, ir.StaticSetItem) and inst.target.name in self.df_vars:
                    df_name = inst.target.name
                    self.df_vars[df_name][inst.index] = inst.value
                    self._update_df_cols()
                elif isinstance(inst, ir.Assign):
                    out_nodes = self._run_assign(inst)
                    if isinstance(out_nodes, list):
                        new_body.extend(out_nodes)
                    if isinstance(out_nodes, dict):
                        label = include_new_blocks(self.func_ir.blocks, out_nodes, label, new_body)
                        new_body = []
                else:
                    new_body.append(inst)
            self.func_ir.blocks[label].body = new_body

        self.func_ir._definitions = get_definitions(self.func_ir.blocks)
        self.func_ir.df_cols = self.df_cols
        #remove_dead(self.func_ir.blocks, self.func_ir.arg_names)
        dprint_func_ir(self.func_ir, "after hiframes")
        if numba.config.DEBUG_ARRAY_OPT==1:  # pragma: no cover
            print("df_vars: ", self.df_vars)
        return

    def _run_assign(self, assign):
        lhs = assign.target.name
        rhs = assign.value

        if isinstance(rhs, ir.Expr):
            if rhs.op=='call':
                res = self._handle_pd_DataFrame(assign.target, rhs)
                if res is not None:
                    return res
                res = self._handle_pq_table(assign.target, rhs)
                if res is not None:
                    return res
                res = self._handle_ros(assign.target, rhs)
                if res is not None:
                    return res
                res = self._handle_column_call(assign.target, rhs)
                if res is not None:
                    return res
                res = self._handle_rolling_setup(assign.target, rhs)
                if res is not None:
                    return res
                res = self._handle_rolling_call(assign.target, rhs)
                if res is not None:
                    return res
                res = self._handle_str_contains(assign.target, rhs)
                if res is not None:
                    return res

            # d = df['column']
            if (rhs.op == 'static_getitem' and rhs.value.name in self.df_vars
                                            and isinstance(rhs.index, str)):
                df = rhs.value.name
                assign.value = self.df_vars[df][rhs.index]
                self.df_cols.add(lhs)  # save lhs as column

            # df1 = df[df.A > .5]
            if (rhs.op == 'getitem' and rhs.value.name in self.df_vars):
                # output df1 has same columns as df, create new vars
                scope = assign.target.scope
                loc = assign.target.loc
                self.df_vars[lhs] = {}
                for col, _ in self.df_vars[rhs.value.name].items():
                    self.df_vars[lhs][col] = ir.Var(scope, mk_unique_var(col),
                                                                            loc)
                self._update_df_cols()
                return [hiframes_filter.Filter(lhs, rhs.value.name, rhs.index,
                                                        self.df_vars, rhs.loc)]

            # df.loc or df.iloc
            if rhs.op=='getattr' and rhs.value.name in self.df_vars and rhs.attr in ['loc', 'iloc']:
                # FIXME: treat iloc and loc as regular df variables so getitem
                # turns them into filter. Only boolean array is supported
                self.df_vars[lhs] = self.df_vars[rhs.value.name]
                return []

            # if (rhs.op == 'getitem' and rhs.value.name in self.df_cols):
            #     self.col_filters.add(assign)

            # d = df.column
            if rhs.op=='getattr' and rhs.value.name in self.df_vars:
                df = rhs.value.name
                df_cols = self.df_vars[df]
                assert rhs.attr in df_cols
                assign.value = df_cols[rhs.attr]
                self.df_cols.add(lhs)  # save lhs as column

            # c = df.column.values
            if (rhs.op=='getattr' and rhs.value.name in self.df_cols and
                        rhs.attr == 'values'):
                # simply return the column
                # output is array so it's not added to df_cols
                assign.value = rhs.value
                return [assign]

        # handle copies lhs = f
        if isinstance(rhs, ir.Var) and rhs.name in self.df_vars:
            self.df_vars[lhs] = self.df_vars[rhs.name]
        if isinstance(rhs, ir.Var) and rhs.name in self.df_cols:
            self.df_cols.add(lhs)
        if isinstance(rhs, ir.Var) and rhs.name in self.arrow_tables:
            self.arrow_tables[lhs] = self.arrow_tables[rhs.name]
        return [assign]

    def _get_reverse_copies(self, body):
        for inst in body:
            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Var):
                self.reverse_copies[inst.value.name] = inst.target.name
        return

    def _handle_pd_DataFrame(self, lhs, rhs):
        if guard(find_callname, self.func_ir, rhs) == ('DataFrame', 'pandas'):
            if len(rhs.args) != 1:  # pragma: no cover
                raise ValueError("Invalid DataFrame() arguments (one expected)")
            arg_def = guard(get_definition, self.func_ir, rhs.args[0])
            if not isinstance(arg_def, ir.Expr) or arg_def.op != 'build_map':  # pragma: no cover
                raise ValueError("Invalid DataFrame() arguments (map expected)")
            out, items = self._fix_df_arrays(arg_def.items)
            self.df_vars[lhs.name] = self._process_df_build_map(items)
            self._update_df_cols()
            # remove DataFrame call
            return out
        return None

    def _handle_pq_table(self, lhs, rhs):
        if guard(find_callname, self.func_ir, rhs) == ('read_table',
                                                        'pyarrow.parquet'):
            if len(rhs.args) != 1:  # pragma: no cover
                raise ValueError("Invalid read_table() arguments")
            self.arrow_tables[lhs.name] = rhs.args[0]
            return []
        # match t.to_pandas()
        func_def = guard(get_definition, self.func_ir, rhs.func)
        assert func_def is not None
        # rare case where function variable is assigned to a new variable
        if isinstance(func_def, ir.Var):
            rhs.func = func_def
            return self._handle_pq_table(lhs, rhs)
        if (isinstance(func_def, ir.Expr) and func_def.op == 'getattr'
                and func_def.value.name in self.arrow_tables
                and func_def.attr == 'to_pandas'):

            col_items, nodes = self.pq_handler.gen_parquet_read(
                            self.arrow_tables[func_def.value.name], lhs)
            self.df_vars[lhs.name] = self._process_df_build_map(col_items)
            self._update_df_cols()
            return nodes
        return None

    def _handle_ros(self, lhs, rhs):
        if guard(find_callname, self.func_ir, rhs) == ('read_ros_images',
                                                        'hpat.ros'):
            if len(rhs.args) != 1:  # pragma: no cover
                raise ValueError("Invalid read_ros_images() arguments")
            import hpat.ros
            return hpat.ros._handle_read_images(lhs, rhs)
        return None

    def _fix_df_arrays(self, items_list):
        nodes = []
        new_list = []
        for item in items_list:
            col_varname = item[0]
            col_arr = item[1]
            def f(arr):  # pragma: no cover
                df_arr = hpat.hiframes_api.fix_df_array(arr)
            f_block = compile_to_numba_ir(f, {'hpat': hpat}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [col_arr])
            nodes += f_block.body[:-3]  # remove none return
            new_col_arr = nodes[-1].target
            new_list.append((col_varname, new_col_arr))
        return nodes, new_list

    def _process_df_build_map(self, items_list):
        df_cols = {}
        for item in items_list:
            col_var = item[0]
            if isinstance(col_var, str):
                col_name = col_var
            else:
                col_name = get_constant(self.func_ir, col_var)
                if col_name is NOT_CONSTANT:  # pragma: no cover
                    raise ValueError("data frame column names should be constant")
            df_cols[col_name] = item[1]
        return df_cols

    def _update_df_cols(self):
        for df_name, cols_map in self.df_vars.items():
            for col_name, col_var in cols_map.items():
                self.df_cols.add(col_var.name)
        return

    def _handle_column_call(self, lhs, rhs):
        """
        Handle Series calls like:
          A = df.column.shift(3)
        """
        func_def = guard(get_definition, self.func_ir, rhs.func)
        assert func_def is not None
        # rare case where function variable is assigned to a new variable
        if isinstance(func_def, ir.Var):
            rhs.func = func_def
            return self._handle_column_call(lhs, rhs)
        if (isinstance(func_def, ir.Expr) and func_def.op == 'getattr'
                and func_def.value.name in self.df_cols
                and func_def.attr in df_col_funcs):
            func_name = func_def.attr
            col_var = func_def.value
            return self._gen_column_call(lhs, rhs.args, col_var, func_name, dict(rhs.kws))
        return None

    def _handle_rolling_setup(self, lhs, rhs):
        """
        Handle Series rolling calls like:
          r = df.column.rolling(3)
        """
        func_def = guard(get_definition, self.func_ir, rhs.func)
        assert func_def is not None
        # rare case where function variable is assigned to a new variable
        if isinstance(func_def, ir.Var):
            rhs.func = func_def
            return self._handle_rolling_setup(lhs, rhs)
        # df.column.rolling
        if (isinstance(func_def, ir.Expr) and func_def.op == 'getattr'
                and func_def.value.name in self.df_cols
                and func_def.attr == 'rolling'):
            center = False
            kws = dict(rhs.kws)
            if rhs.args:
                window = rhs.args[0]
            elif 'window' in kws:
                window = kws['window']
            else:  # pragma: no cover
                raise ValueError("window argument to rolling() required")
            window =  get_constant(self.func_ir, window, window)
            if 'center' in kws:
                center =  get_constant(self.func_ir, kws['center'], center)
            self.rolling_calls[lhs.name] = [func_def.value, window, center]
            return []  # remove
        return None

    def _handle_rolling_call(self, lhs, rhs):
        """
        Handle Series rolling calls like:
          A = df.column.rolling(3).sum()
        """
        func_def = guard(get_definition, self.func_ir, rhs.func)
        assert func_def is not None
        # rare case where function variable is assigned to a new variable
        if isinstance(func_def, ir.Var):
            rhs.func = func_def
            return self._handle_rolling_setup(lhs, rhs)
        # df.column.rolling(3).sum()
        if (isinstance(func_def, ir.Expr) and func_def.op == 'getattr'
                and func_def.value.name in self.rolling_calls):
            func_name =  func_def.attr
            self.df_cols.add(lhs.name)  # output is Series
            return self._gen_rolling_call(rhs.args,
                *self.rolling_calls[func_def.value.name]+[func_name, lhs])
        return None

    def _handle_str_contains(self, lhs, rhs):
        """
        Handle string contains like:
          B = df.column.str.contains('oo*', regex=True)
        """
        func_def = guard(get_definition, self.func_ir, rhs.func)
        assert func_def is not None
        # rare case where function variable is assigned to a new variable
        if isinstance(func_def, ir.Var):
            rhs.func = func_def
            return self._handle_str_contains(lhs, rhs)
        str_col = guard(self._get_str_contains_col, func_def)
        if str_col is None:
            return None
        kws = dict(rhs.kws)
        pat = rhs.args[0]
        regex = True  # default regex arg is True
        if 'regex' in kws:
            regex = get_constant(self.func_ir, kws['regex'], regex)
        if regex:
            def f(str_arr, pat):  # pragma: no cover
                e = hpat.str_ext.compile_regex(pat)
                hpat.hiframes_api.str_contains_regex(str_arr, e)
        else:
            def f(str_arr, pat):  # pragma: no cover
                hpat.hiframes_api.str_contains_noregex(str_arr, pat)

        f_block = compile_to_numba_ir(f, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [str_col, pat])
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = lhs
        return nodes

    def _get_str_contains_col(self, func_def):
        require(isinstance(func_def, ir.Expr) and func_def.op == 'getattr')
        require(func_def.attr == 'contains')
        str_def = get_definition(self.func_ir, func_def.value)
        require(isinstance(str_def, ir.Expr) and str_def.op == 'getattr')
        require(str_def.attr == 'str')
        col = str_def.value
        require(col.name in self.df_cols)
        return col

    def _gen_column_call(self, out_var, args, col_var, func, kws):
        if func in ['fillna', 'pct_change', 'shift']:
            self.df_cols.add(out_var.name) # output is Series except sum
        if func == 'count':
            return self._gen_col_count(out_var, args, col_var)
        if func == 'fillna':
            return self._gen_fillna(out_var, args, col_var, kws)
        if func == 'sum':
            return self._gen_col_sum(out_var, args, col_var)
        if func == 'mean':
            return self._gen_col_mean(out_var, args, col_var)
        if func == 'var':
            return self._gen_col_var(out_var, args, col_var)
        if func == 'std':
            return self._gen_col_std(out_var, args, col_var)
        if func == 'quantile':
            return self._gen_col_quantile(out_var, args, col_var)
        if func == 'describe':
            return self._gen_col_describe(out_var, args, col_var)
        else:
            assert func in ['pct_change', 'shift']
            return self._gen_column_shift_pct(out_var, args, col_var, func)

    def _gen_column_shift_pct(self, out_var, args, col_var, func):
        loc = col_var.loc
        if func == 'pct_change':
            shift_const = 1
            if args:
                shift_const = get_constant(self.func_ir, args[0])
                assert shift_const is not NOT_CONSTANT
            func_text = 'def g(a):\n  return (a[0]-a[{}])/a[{}]\n'.format(
                                                    -shift_const, -shift_const)
        else:
            assert func == 'shift'
            shift_const = get_constant(self.func_ir, args[0])
            assert shift_const is not NOT_CONSTANT
            func_text = 'def g(a):\n  return a[{}]\n'.format(-shift_const)

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        kernel_func = loc_vars['g']

        index_offsets = [0]
        fir_globals = self.func_ir.func_id.func.__globals__
        stencil_nodes = gen_stencil_call(col_var, out_var, kernel_func, index_offsets, fir_globals)

        border_text = 'def f(A):\n  A[0:{}] = np.nan\n'.format(shift_const)
        loc_vars = {}
        exec(border_text, {}, loc_vars)
        border_func = loc_vars['f']

        f_blocks = compile_to_numba_ir(border_func, {'np': np}).blocks
        block = f_blocks[min(f_blocks.keys())]
        replace_arg_nodes(block, [out_var])
        setitem_nodes = block.body[:-3]  # remove none return

        return stencil_nodes+setitem_nodes

    def _gen_col_count(self, out_var, args, col_var):
        def f(A):  # pragma: no cover
            s = hpat.hiframes_api.count(A)

        f_block = compile_to_numba_ir(f, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [col_var])
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = out_var
        return nodes

    def _gen_fillna(self, out_var, args, col_var, kws):
        inplace = False
        if 'inplace' in kws:
            inplace = get_constant(self.func_ir, kws['inplace'])
            if inplace==NOT_CONSTANT:  # pragma: no cover
                raise ValueError("inplace arg to fillna should be constant")

        if inplace:
            out_var = col_var  # output array is same as input array
            alloc_nodes = []
        else:
            alloc_nodes = gen_empty_like(col_var, out_var)

        val = args[0]
        def f(A, B, fill):  # pragma: no cover
            hpat.hiframes_api.fillna(A, B, fill)
        f_block = compile_to_numba_ir(f, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [out_var, col_var, val])
        nodes = f_block.body[:-3]  # remove none return
        return alloc_nodes + nodes

    def _gen_col_sum(self, out_var, args, col_var):
        def f(A):  # pragma: no cover
            s = hpat.hiframes_api.column_sum(A)

        f_block = compile_to_numba_ir(f, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [col_var])
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = out_var
        return nodes

    def _gen_col_mean(self, out_var, args, col_var):
        def f(A):  # pragma: no cover
            s = hpat.hiframes_api.mean(A)

        f_block = compile_to_numba_ir(f, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [col_var])
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = out_var
        return nodes

    def _gen_col_var(self, out_var, args, col_var):
        def f(A):  # pragma: no cover
            s = hpat.hiframes_api.var(A)

        f_block = compile_to_numba_ir(f, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [col_var])
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = out_var
        return nodes

    def _gen_col_std(self, out_var, args, col_var):
        loc = out_var.loc
        scope = out_var.scope
        # calculate var() first
        var_var = ir.Var(scope, mk_unique_var("var_val"), loc)
        v_nodes = self._gen_col_var(var_var, args, col_var)

        def f(a):  # pragma: no cover
            a ** 0.5
        s_block = compile_to_numba_ir(f, {}).blocks.popitem()[1]
        replace_arg_nodes(s_block, [var_var])
        s_nodes = s_block.body[:-3]
        assert len(s_nodes) == 3
        s_nodes[-1].target = out_var
        return v_nodes + s_nodes

    def _gen_col_quantile(self, out_var, args, col_var):
        def f(A, q):  # pragma: no cover
            s = hpat.hiframes_api.quantile(A, q)

        f_block = compile_to_numba_ir(f, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [col_var, args[0]])
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = out_var
        return nodes

    def _gen_col_describe(self, out_var, args, col_var):
        def f(A):  # pragma: no cover
            a_count = hpat.hiframes_api.count(A)
            a_min = np.min(A)
            a_max = np.max(A)
            a_mean = hpat.hiframes_api.mean(A)
            a_std = hpat.hiframes_api.var(A)**0.5
            q25 = hpat.hiframes_api.quantile(A, .25)
            q50 = hpat.hiframes_api.quantile(A, .5)
            q75 = hpat.hiframes_api.quantile(A, .75)
            s = "count    "+str(a_count)+"\n"\
                "mean     "+str(a_mean)+"\n"\
                "std      "+str(a_std)+"\n"\
                "min      "+str(a_min)+"\n"\
                "25%      "+str(q25)+"\n"\
                "50%      "+str(q50)+"\n"\
                "75%      "+str(q75)+"\n"\
                "max      "+str(a_max)+"\n"

        f_block = compile_to_numba_ir(f, {'hpat': hpat, 'np': np}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [col_var])
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = out_var
        return nodes

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
                kernel_args = ','.join(['a[{}]'.format(-i) for i in range(win_size)])
                kernel_expr = '{}.{}(np.array([{}]))'.format(g_pack, func, kernel_args)
                if func == 'sum':  # simplify sum
                    kernel_expr = '+'.join(['a[{}]'.format(-i) for i in range(win_size)])
            else:
                kernel_expr = '{}.{}(a[(-w+1):1])'.format(g_pack, func)
            func_text = 'def g(a, w):\n  return {}\n'.format(kernel_expr)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            kernel_func = loc_vars['g']

        init_nodes = []
        col_var, init_nodes = self._fix_rolling_array(col_var, func)

        if isinstance(win_size, int):
            win_size_var = ir.Var(scope, mk_unique_var("win_size"), loc)
            init_nodes.append(
                        ir.Assign(ir.Const(win_size, loc), win_size_var, loc))
            win_size = win_size_var

        index_offsets, win_tuple, option_nodes = self._gen_rolling_init(win_size,
                                                                func, center)

        init_nodes += option_nodes
        other_args = [win_size]
        if func == 'apply':
            other_args = None
        options = {'neighborhood': win_tuple}
        fir_globals = self.func_ir.func_id.func.__globals__
        stencil_nodes = gen_stencil_call(col_var, out_var, kernel_func,
                                    index_offsets, fir_globals, other_args, options)


        def f(A, w):  # pragma: no cover
            A[0:w-1] = np.nan
        f_block = compile_to_numba_ir(f, {'np': np}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [out_var, win_size])
        setitem_nodes = f_block.body[:-3]  # remove none return

        if center:
            def f1(A, w):  # pragma: no cover
                A[0:w//2] = np.nan
            def f2(A, w):  # pragma: no cover
                n = len(A)
                A[n-(w//2):n] = np.nan
            f_block = compile_to_numba_ir(f1, {'np': np}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [out_var, win_size])
            setitem_nodes1 = f_block.body[:-3]  # remove none return
            f_block = compile_to_numba_ir(f2, {'np': np}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [out_var, win_size])
            setitem_nodes2 = f_block.body[:-3]  # remove none return
            setitem_nodes = setitem_nodes1 + setitem_nodes2

        return init_nodes + stencil_nodes + setitem_nodes

    def _fix_rolling_array(self, col_var, func):
        """
        for integers and bools, the output should be converted to float64
        """
        # TODO: check all possible funcs
        def f(arr):  # pragma: no cover
            df_arr = hpat.hiframes_api.fix_rolling_array(arr)
        f_block = compile_to_numba_ir(f, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [col_var])
        nodes = f_block.body[:-3]  # remove none return
        new_col_var = nodes[-1].target
        return new_col_var, nodes


    def _gen_rolling_init(self, win_size, func, center):
        nodes = []
        right_length = 0
        scope = win_size.scope
        loc = win_size.loc
        right_length = ir.Var(scope, mk_unique_var('zero_var'), scope)
        nodes.append(ir.Assign(ir.Const(0, loc), right_length, win_size.loc))

        def f(w):  # pragma: no cover
            return -w+1
        f_block = compile_to_numba_ir(f, {}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [win_size])
        nodes.extend(f_block.body[:-2])  # remove none return
        left_length = nodes[-1].target

        if center:
            def f(w):  # pragma: no cover
                return -(w//2)
            f_block = compile_to_numba_ir(f, {}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [win_size])
            nodes.extend(f_block.body[:-2])  # remove none return
            left_length = nodes[-1].target
            def f(w):  # pragma: no cover
                return (w//2)
            f_block = compile_to_numba_ir(f, {}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [win_size])
            nodes.extend(f_block.body[:-2])  # remove none return
            right_length = nodes[-1].target


        def f(a, b):  # pragma: no cover
            return ((a, b),)
        f_block = compile_to_numba_ir(f, {}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [left_length, right_length])
        nodes.extend(f_block.body[:-2])  # remove none return
        win_tuple = nodes[-1].target

        index_offsets = [right_length]

        if func == 'apply':
            index_offsets = [left_length]

        def f(a):  # pragma: no cover
            return (a,)
        f_block = compile_to_numba_ir(f, {}).blocks.popitem()[1]
        replace_arg_nodes(f_block, index_offsets)
        nodes.extend(f_block.body[:-2])  # remove none return
        index_offsets = nodes[-1].target

        return index_offsets, win_tuple, nodes

def gen_empty_like(in_arr, out_arr):
    scope = in_arr.scope
    loc = in_arr.loc
    # g_np_var = Global(numpy)
    g_np_var = ir.Var(scope, mk_unique_var("$np_g_var"), loc)
    g_np = ir.Global('np', np, loc)
    g_np_assign = ir.Assign(g_np, g_np_var, loc)
    # attr call: empty_attr = getattr(g_np_var, empty_like)
    empty_attr_call = ir.Expr.getattr(g_np_var, "empty_like", loc)
    attr_var = ir.Var(scope, mk_unique_var("$empty_attr_attr"), loc)
    attr_assign = ir.Assign(empty_attr_call, attr_var, loc)
    # alloc call: out_arr = empty_attr(in_arr)
    alloc_call = ir.Expr.call(attr_var, [in_arr], (), loc)
    alloc_assign = ir.Assign(alloc_call, out_arr, loc)
    return [g_np_assign, attr_assign, alloc_assign]

def gen_stencil_call(in_arr, out_arr, kernel_func, index_offsets, fir_globals,
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
    stencil_nodes += gen_empty_like(in_arr, out_arr)

    kernel_var = ir.Var(scope, mk_unique_var("kernel_var"), scope)
    if not isinstance(kernel_func, ir.Expr):
        kernel_func = ir.Expr.make_function("kernel", kernel_func.__code__,
                    kernel_func.__closure__, kernel_func.__defaults__, loc)
    stencil_nodes.append(ir.Assign(kernel_func, kernel_var, loc))

    def f(A, B, f):  # pragma: no cover
        numba.stencil(f)(A, out=B)
    f_block = compile_to_numba_ir(f, {'numba': numba}).blocks.popitem()[1]
    replace_arg_nodes(f_block, [in_arr, out_arr, kernel_var])
    stencil_nodes += f_block.body[:-3]  # remove none return
    setup_call = stencil_nodes[-2].value
    stencil_call = stencil_nodes[-1].value
    setup_call.kws = list(options.items())
    stencil_call.args += other_args

    return stencil_nodes


def remove_none_return_from_block(last_block):
    # remove const none, cast, return nodes
    assert isinstance(last_block.body[-1], ir.Return)
    last_block.body.pop()
    assert (isinstance(last_block.body[-1], ir.Assign)
        and isinstance(last_block.body[-1].value, ir.Expr)
        and last_block.body[-1].value.op == 'cast')
    last_block.body.pop()
    assert (isinstance(last_block.body[-1], ir.Assign)
        and isinstance(last_block.body[-1].value, ir.Const)
        and last_block.body[-1].value.value == None)
    last_block.body.pop()

def include_new_blocks(blocks, new_blocks, label, new_body):
    inner_blocks = add_offset_to_labels(new_blocks, ir_utils._max_label+1)
    blocks.update(inner_blocks)
    ir_utils._max_label = max(blocks.keys())
    scope = blocks[label].scope
    loc = blocks[label].loc
    inner_topo_order = find_topo_order(inner_blocks)
    inner_first_label = inner_topo_order[0]
    inner_last_label = inner_topo_order[-1]
    remove_none_return_from_block(inner_blocks[inner_last_label])
    new_body.append(ir.Jump(inner_first_label, loc))
    blocks[label].body = new_body
    label = ir_utils.next_label()
    blocks[label] = ir.Block(scope, loc)
    inner_blocks[inner_last_label].body.append(ir.Jump(label, loc))
    #new_body.clear()
    return label

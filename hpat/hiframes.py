from __future__ import print_function, division, absolute_import
import warnings
from collections import namedtuple

import numba
from numba import ir, ir_utils, types
from numba import compiler as numba_compiler
from numba.targets.registry import CPUDispatcher

from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir, remove_dead, mk_alloc, remove_dels,
                            get_name_var_table, replace_var_names,
                            add_offset_to_labels, get_ir_of_code,
                            compile_to_numba_ir, replace_arg_nodes,
                            find_callname, guard, require, get_definition,
                            build_definitions)

from numba.inline_closurecall import inline_closure_call

import hpat
from hpat import (hiframes_api, utils, parquet_pio, config, hiframes_filter,
                  hiframes_join)
from hpat.utils import get_constant, NOT_CONSTANT, get_definitions
from hpat.hiframes_api import PandasDataFrameType
from hpat.str_ext import StringType, string_type
from hpat.str_arr_ext import StringArray, StringArrayType, string_array_type

import numpy as np
import math
from hpat.parquet_pio import ParquetHandler
from hpat.pd_timestamp_ext import timestamp_series_type

df_col_funcs = ['shift', 'pct_change', 'fillna', 'sum', 'mean', 'var', 'std',
                'quantile', 'count', 'describe']
LARGE_WIN_SIZE = 10


def remove_hiframes(rhs, lives, call_list):
    # used in stencil generation of rolling
    if len(call_list) == 1 and call_list[0] in [int, min, max, abs]:
        return True
    # used in stencil generation of rolling
    if (len(call_list) == 1 and isinstance(call_list[0], CPUDispatcher)
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
    if call_list == ['dist_return', 'distributed_api', hpat]:
        return True
    if call_list == ['unbox_df_column', 'hiframes_api', hpat]:
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
        # keep track of series that are timestamp to replace getitem
        # FIXME: this is possibly fragile, maybe replace all series getitems
        # with a function and handle this after type inference
        self.ts_series_vars = set()
        self.arrow_tables = {}
        self.reverse_copies = {}
        self.pq_handler = ParquetHandler(
            func_ir, typingctx, args, _locals, self.reverse_copies)

    def run(self):
        dprint_func_ir(self.func_ir, "starting hiframes")
        topo_order = find_topo_order(self.func_ir.blocks)
        for label in topo_order:
            self._get_reverse_copies(self.func_ir.blocks[label].body)
            new_body = []
            for inst in self.func_ir.blocks[label].body:
                # df['col'] = arr
                if (isinstance(inst, ir.StaticSetItem)
                        and inst.target.name in self.df_vars):
                    df_name = inst.target.name
                    self.df_vars[df_name][inst.index] = inst.value
                    self._update_df_cols()
                elif isinstance(inst, ir.Assign):
                    out_nodes = self._run_assign(inst)
                    if isinstance(out_nodes, list):
                        new_body.extend(out_nodes)
                    if isinstance(out_nodes, dict):
                        label = include_new_blocks(
                            self.func_ir.blocks, out_nodes, label, new_body)
                        new_body = []
                elif isinstance(inst, ir.Return):
                    nodes = self._run_return(inst)
                    new_body += nodes
                else:
                    new_body.append(inst)
            self.func_ir.blocks[label].body = new_body

        self.func_ir._definitions = get_definitions(self.func_ir.blocks)
        self.func_ir.df_cols = self.df_cols
        # remove_dead(self.func_ir.blocks, self.func_ir.arg_names)
        dprint_func_ir(self.func_ir, "after hiframes")
        if numba.config.DEBUG_ARRAY_OPT == 1:  # pragma: no cover
            print("df_vars: ", self.df_vars)
        return

    def _run_assign(self, assign):
        lhs = assign.target.name
        rhs = assign.value

        if isinstance(rhs, ir.Expr):
            if rhs.op == 'call':
                return self._run_call(assign)

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
            if (rhs.op == 'getattr' and rhs.value.name in self.df_vars
                    and rhs.attr in ['loc', 'iloc']):
                # FIXME: treat iloc and loc as regular df variables so getitem
                # turns them into filter. Only boolean array is supported
                self.df_vars[lhs] = self.df_vars[rhs.value.name]
                return []

            # if (rhs.op == 'getitem' and rhs.value.name in self.df_cols):
            #     self.col_filters.add(assign)

            # d = df.column
            if (rhs.op == 'getattr' and rhs.value.name in self.df_vars
                    and rhs.attr in self.df_vars[rhs.value.name]):
                df = rhs.value.name
                df_cols = self.df_vars[df]
                # assert rhs.attr in df_cols
                assign.value = df_cols[rhs.attr]
                self.df_cols.add(lhs)  # save lhs as column
                if df_cols[rhs.attr].name in self.ts_series_vars:
                    self.ts_series_vars.add(lhs)
                # need to remove the lhs definition so that find_callname can
                # match column function calls (i.e. A.f instead of df.A.f)
                assert self.func_ir._definitions[lhs] == [rhs], "invalid def"
                self.func_ir._definitions[lhs] = [None]

            # c = df.column.values
            if (rhs.op == 'getattr' and rhs.value.name in self.df_cols and
                    rhs.attr == 'values'):
                # simply return the column
                # output is array so it's not added to df_cols
                assign.value = rhs.value
                return [assign]

            # replace getitems on timestamp series with function
            # for proper type inference
            if (rhs.op in ['getitem', 'static_getitem']
                    and rhs.value.name in self.ts_series_vars):
                if rhs.op == 'getitem':
                    ind_var = rhs.index
                else:
                    ind_var = rhs.index_var
                def f(_ts_series, _ind):  # pragma: no cover
                    _val = hpat.hiframes_api.ts_series_getitem(_ts_series, _ind)

                f_block = compile_to_numba_ir(
                    f, {'hpat': hpat}).blocks.popitem()[1]
                replace_arg_nodes(f_block, [rhs.value, ind_var])
                nodes = f_block.body[:-3]  # remove none return
                nodes[-1].target = assign.target
                # output could be series in case of slice index
                # FIXME: this is fragile
                self.ts_series_vars.add(lhs)
                return nodes

        if isinstance(rhs, ir.Arg):
            return self._run_arg(assign)

        # handle copies lhs = f
        if isinstance(rhs, ir.Var) and rhs.name in self.df_vars:
            self.df_vars[lhs] = self.df_vars[rhs.name]
        if isinstance(rhs, ir.Var) and rhs.name in self.df_cols:
            self.df_cols.add(lhs)
        if isinstance(rhs, ir.Var) and rhs.name in self.arrow_tables:
            self.arrow_tables[lhs] = self.arrow_tables[rhs.name]
        if isinstance(rhs, ir.Var) and rhs.name in self.ts_series_vars:
            self.ts_series_vars[lhs] = self.ts_series_vars[rhs.name]
        return [assign]

    def _run_call(self, assign):
        """handle calls and return new nodes if needed
        """
        lhs = assign.target
        rhs = assign.value

        func_name = ""
        func_mod = ""
        fdef = guard(find_callname, self.func_ir, rhs)
        if fdef is None:
            warnings.warn(
                "function call couldn't be found for initial analysis")
            return [assign]
        else:
            func_name, func_mod = fdef

        if fdef == ('DataFrame', 'pandas'):
            return self._handle_pd_DataFrame(assign, lhs, rhs)

        if fdef == ('read_table', 'pyarrow.parquet'):
            return self._handle_pq_read_table(assign, lhs, rhs)

        if (func_name == 'to_pandas' and isinstance(func_mod, ir.Var)
                and func_mod.name in self.arrow_tables):
            return self._handle_pq_to_pandas(assign, lhs, rhs, func_mod)

        if fdef == ('merge', 'pandas'):
            return self._handle_merge(assign, lhs, rhs)

        if fdef == ('concat', 'pandas'):
            return self._handle_concat(assign, lhs, rhs)

        if fdef == ('read_ros_images', 'hpat.ros'):
            return self._handle_ros(assign, lhs, rhs)


        # df.column.shift(3)
        if isinstance(func_mod, ir.Var) and func_mod.name in self.df_cols:
            return self._handle_column_call(assign, lhs, rhs, func_mod, func_name)

        # df.apply(lambda a:..., axis=1)
        if (isinstance(func_mod, ir.Var) and func_mod.name in self.df_vars
                and func_name == 'apply'):
            return self._handle_df_apply(assign, lhs, rhs, func_mod)

        # df.describe()
        if (isinstance(func_mod, ir.Var) and func_mod.name in self.df_vars
                and func_name == 'describe'):
            return self._handle_df_describe(assign, lhs, rhs, func_mod)

        res = self._handle_rolling_call(assign.target, rhs)
        if res is not None:
            return res

        if fdef == ('fromfile', 'numpy'):
            return hpat.io._handle_np_fromfile(assign, lhs, rhs)

        return [assign]

    def _get_reverse_copies(self, body):
        for inst in body:
            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Var):
                self.reverse_copies[inst.value.name] = inst.target.name
        return

    def _handle_pd_DataFrame(self, assign, lhs, rhs):
        """transform pd.DataFrame({'A': A}) call
        """
        if len(rhs.args) != 1:  # pragma: no cover
            raise ValueError(
                "Invalid DataFrame() arguments (one expected)")
        arg_def = guard(get_definition, self.func_ir, rhs.args[0])
        if (not isinstance(arg_def, ir.Expr)
                or arg_def.op != 'build_map'):  # pragma: no cover
            raise ValueError(
                "Invalid DataFrame() arguments (map expected)")
        out, items = self._fix_df_arrays(arg_def.items)
        self.df_vars[lhs.name] = self._process_df_build_map(items)
        self._update_df_cols()
        # remove DataFrame call
        return out

    def _handle_pq_read_table(self, assign, lhs, rhs):
        if len(rhs.args) != 1:  # pragma: no cover
            raise ValueError("Invalid read_table() arguments")
        self.arrow_tables[lhs.name] = rhs.args[0]
        return []

    def _handle_pq_to_pandas(self, assign, lhs, rhs, t_var):
        col_items, nodes = self.pq_handler.gen_parquet_read(
            self.arrow_tables[t_var.name], lhs)
        self.df_vars[lhs.name] = self._process_df_build_map(col_items)
        self._update_df_cols()
        return nodes

    def _handle_merge(self, assign, lhs, rhs):
        """transform pd.merge() into a Join node
        """
        if len(rhs.args) < 2:
            raise ValueError("left and right arguments required for merge")
        left_df = rhs.args[0]
        right_df = rhs.args[1]
        kws = dict(rhs.kws)
        if 'on' in kws:
            left_on = get_constant(self.func_ir, kws['on'], None)
            right_on = left_on
        else:  # pragma: no cover
            if 'left_on' not in kws or 'right_on' not in kws:
                raise ValueError("merge 'on' or 'left_on'/'right_on'"
                                 "arguments required")
            left_on = get_constant(self.func_ir, kws['left_on'], None)
            right_on = get_constant(self.func_ir, kws['right_on'], None)
        if left_on is None or right_on is None:
            raise ValueError("merge key values should be constant strings")
        scope = lhs.scope
        loc = lhs.loc
        self.df_vars[lhs.name] = {}
        # add columns from left to output
        for col, _ in self.df_vars[left_df.name].items():
            self.df_vars[lhs.name][col] = ir.Var(
                scope, mk_unique_var(col), loc)
        # add columns from right to output
        for col, _ in self.df_vars[right_df.name].items():
            self.df_vars[lhs.name][col] = ir.Var(
                scope, mk_unique_var(col), loc)
        self._update_df_cols()
        return [hiframes_join.Join(lhs.name, left_df.name, right_df.name,
                                   left_on, right_on, self.df_vars, lhs.loc)]

    def _handle_concat(self, assign, lhs, rhs):
        if len(rhs.args) != 1 or len(rhs.kws) != 0:
            raise ValueError(
                "only a list/tuple argument is supported in concat")
        df_list = guard(get_definition, self.func_ir, rhs.args[0])
        assert isinstance(df_list, ir.Expr) and df_list.op == 'build_list'

        nodes = []
        done_cols = {}
        i = 0
        for df in df_list.items:
            for (c, v) in self.df_vars[df.name].items():
                if c in done_cols:
                    continue
                # arguments to the generated function
                args = [v]
                # names of arguments to the generated function
                arg_names = ['_hpat_c' + str(i)]
                # arguments to the concatenate function
                conc_arg_names = ['_hpat_c' + str(i)]
                allocs = ""
                i += 1
                for other_df in df_list.items:
                    if other_df.name == df.name:
                        continue
                    if c in self.df_vars[other_df.name]:
                        args.append(self.df_vars[other_df.name][c])
                        arg_names.append('_hpat_c' + str(i))
                        conc_arg_names.append('_hpat_c' + str(i))
                        i += 1
                    else:
                        # use a df column for length
                        len_arg = list(
                            self.df_vars[other_df.name].values())[0]
                        len_name = '_hpat_len' + str(i)
                        args.append(len_arg)
                        arg_names.append(len_name)
                        i += 1
                        out_name = '_hpat_out' + str(i)
                        conc_arg_names.append(out_name)
                        i += 1
                        # TODO: fix type
                        allocs += "    {} = np.full(len({}), np.nan)\n".format(
                            out_name, len_name)

                func_text = "def f({}):\n".format(",".join(arg_names))
                func_text += allocs
                func_text += "    s = np.concatenate(({}))\n".format(
                    ",".join(conc_arg_names))
                loc_vars = {}
                exec(func_text, {}, loc_vars)
                f = loc_vars['f']

                f_block = compile_to_numba_ir(f,
                            {'hpat': hpat, 'np': np}).blocks.popitem()[1]
                replace_arg_nodes(f_block, args)
                nodes += f_block.body[:-3]
                done_cols[c] = nodes[-1].target

        self.df_vars[lhs.name] = done_cols
        self._update_df_cols()
        return nodes

    def _handle_ros(self, assign, lhs, rhs):
        if len(rhs.args) != 1:  # pragma: no cover
            raise ValueError("Invalid read_ros_images() arguments")
        import hpat.ros
        return hpat.ros._handle_read_images(lhs, rhs)

    def _fix_df_arrays(self, items_list):
        nodes = []
        new_list = []
        for item in items_list:
            col_varname = item[0]
            col_arr = item[1]

            def f(arr):  # pragma: no cover
                df_arr = hpat.hiframes_api.fix_df_array(arr)
            f_block = compile_to_numba_ir(
                f, {'hpat': hpat}).blocks.popitem()[1]
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
                    raise ValueError(
                        "data frame column names should be constant")
            df_cols[col_name] = item[1]
        return df_cols

    def _update_df_cols(self):
        for df_name, cols_map in self.df_vars.items():
            for col_name, col_var in cols_map.items():
                self.df_cols.add(col_var.name)
        return

    def _handle_df_apply(self, assign, lhs, rhs, func_mod):
        # check for axis=1
        if not (len(rhs.kws) == 1 and rhs.kws[0][0] == 'axis'
                and get_constant(self.func_ir, rhs.kws[0][1]) == 1):
            raise ValueError("only apply() with axis=1 supported")

        if len(rhs.args) != 1:
            raise ValueError("lambda arg to apply() expected")

        # get apply function
        func = guard(get_definition, self.func_ir, rhs.args[0])
        if func is None or not (isinstance(func, ir.Expr)
                                and func.op == 'make_function'):
            raise ValueError("lambda for apply not found")

        _globals = self.func_ir.func_id.func.__globals__
        col_names = self.df_vars[func_mod.name].keys()

        # find columns that are actually used if possible
        used_cols = []
        lambda_ir = compile_to_numba_ir(func, _globals)
        l_topo_order = find_topo_order(lambda_ir.blocks)
        first_stmt = lambda_ir.blocks[l_topo_order[0]].body[0]
        assert isinstance(first_stmt, ir.Assign) and isinstance(first_stmt.value, ir.Arg)
        arg_var = first_stmt.target
        use_all_cols = False
        for bl in lambda_ir.blocks.values():
            for stmt in bl.body:
                vnames = [v.name for v in stmt.list_vars()]
                if arg_var.name in vnames:
                    if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Arg):
                        continue
                    if (isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr)
                            and stmt.value.op == 'getattr'):
                        assert stmt.value.attr in col_names
                        used_cols.append(stmt.value.attr)
                    else:
                        # argument is used in some other form
                        # be conservative and use all cols
                        use_all_cols = True
                        used_cols = col_names
                        break

            if use_all_cols:
                break

        Row = namedtuple(func_mod.name, used_cols)
        # TODO: handle non numpy alloc types
        # prange func to inline
        col_name_args = ', '.join(["c"+str(i) for i in range(len(used_cols))])
        row_args = ', '.join(["c"+str(i)+"[i]" for i in range(len(used_cols))])

        func_text = "def f({}):\n".format(col_name_args)
        func_text += "  numba.parfor.init_prange()\n"
        func_text += "  n = len(c0)\n"
        func_text += "  S = numba.unsafe.ndarray.empty_inferred((n,))\n"
        func_text += "  for i in numba.parfor.internal_prange(n):\n"
        func_text += "     row = Row({})\n".format(row_args)
        func_text += "     S[i] = map_func(row)\n"
        func_text += "  ret = S\n"

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        f_ir = compile_to_numba_ir(f, {'numba': numba, 'np': np, 'Row': Row})
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

        f_ir.blocks[topo_order[-1]].body[-4].target = lhs
        col_vars = [self.df_vars[func_mod.name][c] for c in used_cols]
        replace_arg_nodes(f_ir.blocks[topo_order[0]], col_vars)
        return f_ir.blocks

    def _handle_df_describe(self, assign, lhs, rhs, func_mod):
        """translate df.describe() call with no input or just include='all'
        """
        # check for no arg or just include='all'
        if not (len(rhs.args) == 0 and (len(rhs.kws) == 0 or (len(rhs.kws) == 1
                and rhs.kws[0][0] == 'include'
                and get_constant(self.func_ir, rhs.kws[0][1]) == 'all'))):
            raise ValueError("only describe() with include='all' supported")

        col_names = self.df_vars[func_mod.name].keys()
        col_name_args = ["c"+str(i) for i in range(len(col_names))]
        # TODO: pandas returns dataframe, maybe return namedtuple instread of
        # string?

        func_text = "def f({}):\n".format(', '.join(col_name_args))
        # compute stat values
        for c in col_name_args:
            func_text += "  {}_count = np.float64(hpat.hiframes_api.count({}))\n".format(c, c)
            func_text += "  {}_min = np.min({})\n".format(c, c)
            func_text += "  {}_max = np.max({})\n".format(c, c)
            func_text += "  {}_mean = hpat.hiframes_api.mean({})\n".format(c, c)
            func_text += "  {}_std = hpat.hiframes_api.var({})**0.5\n".format(c, c)
            func_text += "  {}_q25 = hpat.hiframes_api.quantile({}, .25)\n".format(c, c)
            func_text += "  {}_q50 = hpat.hiframes_api.quantile({}, .5)\n".format(c, c)
            func_text += "  {}_q75 = hpat.hiframes_api.quantile({}, .75)\n".format(c, c)


        col_header = "      ".join([c for c in col_names])
        func_text += "  res = '        {}\\n' + \\\n".format(col_header)
        count_strs = "+ '   ' + ".join(["str({}_count)".format(c) for c in col_name_args])
        func_text += "   'count   ' + {} + '\\n' + \\\n".format(count_strs)
        mean_strs = "+ '   ' + ".join(["str({}_mean)".format(c) for c in col_name_args])
        func_text += "   'mean    ' + {} + '\\n' + \\\n".format(mean_strs)
        std_strs = "+ '   ' + ".join(["str({}_std)".format(c) for c in col_name_args])
        func_text += "   'std     ' + {} + '\\n' + \\\n".format(std_strs)
        min_strs = "+ '   ' + ".join(["str({}_min)".format(c) for c in col_name_args])
        func_text += "   'min     ' + {} + '\\n' + \\\n".format(min_strs)
        q25_strs = "+ '   ' + ".join(["str({}_q25)".format(c) for c in col_name_args])
        func_text += "   '25%     ' + {} + '\\n' + \\\n".format(q25_strs)
        q50_strs = "+ '   ' + ".join(["str({}_q50)".format(c) for c in col_name_args])
        func_text += "   '50%     ' + {} + '\\n' + \\\n".format(q50_strs)
        q75_strs = "+ '   ' + ".join(["str({}_q75)".format(c) for c in col_name_args])
        func_text += "   '75%     ' + {} + '\\n' + \\\n".format(q75_strs)
        max_strs = "+ '   ' + ".join(["str({}_max)".format(c) for c in col_name_args])
        func_text += "   'max     ' + {}\n".format(max_strs)

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        f_block = compile_to_numba_ir(
            f, {'hpat': hpat, 'np': np}).blocks.popitem()[1]
        col_vars = list(self.df_vars[func_mod.name].values())
        replace_arg_nodes(f_block, col_vars)
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = lhs
        return nodes

    def _handle_column_call(self, assign, lhs, rhs, col_var, func_name):
        """
        Handle Series calls like:
          A = df.column.shift(3)
        """
        # TODO: handle map/apply differences
        if func_name in ['map', 'apply']:
            return self._handle_map(assign, lhs, rhs, col_var)

        if func_name == 'rolling':
            return self._handle_rolling_setup(assign, lhs, rhs, col_var)

        if func_name == 'str.contains':
            return self._handle_str_contains(assign, lhs, rhs, col_var)

        if func_name in df_col_funcs:
            return self._gen_column_call(lhs, rhs.args, col_var, func_name,
                                         dict(rhs.kws))
        return [assign]

    def _handle_map(self, assign, lhs, rhs, col_var):
        """translate df.A.map(lambda a:...) to prange()
        """
        # error checking: make sure there is function input only
        if len(rhs.args) != 1:
            raise ValueError("map expects 1 argument")
        func = guard(get_definition, self.func_ir, rhs.args[0])
        if func is None or not (isinstance(func, ir.Expr)
                                and func.op == 'make_function'):
            raise ValueError("lambda for map not found")

        # TODO: handle non numpy alloc types like string array
        # prange func to inline
        def f(A):
            numba.parfor.init_prange()
            n = len(A)
            S = numba.unsafe.ndarray.empty_inferred((n,))
            for i in numba.parfor.internal_prange(n):
                S[i] = map_func(A[i])
            ret = S

        if col_var.name in self.ts_series_vars:
            def f(A):
                numba.parfor.init_prange()
                n = len(A)
                S = numba.unsafe.ndarray.empty_inferred((n,))
                for i in numba.parfor.internal_prange(n):
                    t = hpat.hiframes_api.ts_series_getitem(A, i)
                    S[i] = map_func(t)
                ret = S

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

        f_ir.blocks[topo_order[-1]].body[-4].target = lhs
        replace_arg_nodes(f_ir.blocks[topo_order[0]], [col_var])
        return f_ir.blocks

    def _handle_rolling_setup(self, assign, lhs, rhs, col_var):
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
        window = get_constant(self.func_ir, window, window)
        if 'center' in kws:
            center = get_constant(self.func_ir, kws['center'], center)
        self.rolling_calls[lhs.name] = [col_var, window, center]
        return []  # remove

    def _handle_rolling_call(self, lhs, rhs):
        """
        Handle Series rolling calls like:
          A = df.column.rolling(3).sum()
        """
        func_def = guard(get_definition, self.func_ir, rhs.func)
        assert func_def is not None
        # df.column.rolling(3).sum()
        if (isinstance(func_def, ir.Expr) and func_def.op == 'getattr'
                and func_def.value.name in self.rolling_calls):
            func_name = func_def.attr
            self.df_cols.add(lhs.name)  # output is Series
            return self._gen_rolling_call(rhs.args,
                                    *self.rolling_calls[func_def.value.name]
                                    + [func_name, lhs])
        return None

    def _handle_str_contains(self, assign, lhs, rhs, str_col):
        """
        Handle string contains like:
          B = df.column.str.contains('oo*', regex=True)
        """
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
            self.df_cols.add(out_var.name)  # output is Series except sum
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
        stencil_nodes = gen_stencil_call(
            col_var, out_var, kernel_func, index_offsets, fir_globals)

        border_text = 'def f(A):\n  A[0:{}] = np.nan\n'.format(shift_const)
        loc_vars = {}
        exec(border_text, {}, loc_vars)
        border_func = loc_vars['f']

        f_blocks = compile_to_numba_ir(border_func, {'np': np}).blocks
        block = f_blocks[min(f_blocks.keys())]
        replace_arg_nodes(block, [out_var])
        setitem_nodes = block.body[:-3]  # remove none return

        return stencil_nodes + setitem_nodes

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
            if inplace == NOT_CONSTANT:  # pragma: no cover
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
            a_count = np.float64(hpat.hiframes_api.count(A))
            a_min = np.min(A)
            a_max = np.max(A)
            a_mean = hpat.hiframes_api.mean(A)
            a_std = hpat.hiframes_api.var(A)**0.5
            q25 = hpat.hiframes_api.quantile(A, .25)
            q50 = hpat.hiframes_api.quantile(A, .5)
            q75 = hpat.hiframes_api.quantile(A, .75)
            # TODO: pandas returns dataframe, maybe return namedtuple instread of
            # string?
            # TODO: fix string formatting to match python/pandas
            s = "count    " + str(a_count) + "\n"\
                "mean     " + str(a_mean) + "\n"\
                "std      " + str(a_std) + "\n"\
                "min      " + str(a_min) + "\n"\
                "25%      " + str(q25) + "\n"\
                "50%      " + str(q50) + "\n"\
                "75%      " + str(q75) + "\n"\
                "max      " + str(a_max) + "\n"

        f_block = compile_to_numba_ir(
            f, {'hpat': hpat, 'np': np}).blocks.popitem()[1]
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
        col_var, init_nodes = self._fix_rolling_array(col_var, func)

        if isinstance(win_size, int):
            win_size_var = ir.Var(scope, mk_unique_var("win_size"), loc)
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
        stencil_nodes = gen_stencil_call(col_var, out_var, kernel_func,
                                         index_offsets, fir_globals, other_args,
                                         options)

        def f(A, w):  # pragma: no cover
            A[0:w - 1] = np.nan
        f_block = compile_to_numba_ir(f, {'np': np}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [out_var, win_size])
        setitem_nodes = f_block.body[:-3]  # remove none return

        if center:
            def f1(A, w):  # pragma: no cover
                A[0:w // 2] = np.nan

            def f2(A, w):  # pragma: no cover
                n = len(A)
                A[n - (w // 2):n] = np.nan
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

    def _run_arg(self, arg_assign):
        nodes = [arg_assign]
        arg_name = arg_assign.value.name
        arg_ind = arg_assign.value.index
        arg_var = arg_assign.target
        scope = arg_var.scope
        loc = arg_var.loc

        # e.g. {"A:return":"distributed"} -> "A"
        flagged_inputs = { var_name.split(":")[0]: flag
                    for (var_name, flag) in self.locals.items()
                    if var_name.endswith(":input") }

        if arg_name in flagged_inputs.keys():
            self.locals.pop(arg_name + ":input")
            flag = flagged_inputs[arg_name]
            # replace assign target with tmp
            in_arr_tmp = ir.Var(scope, mk_unique_var(flag + "_input"), loc)
            arg_assign.target = in_arr_tmp
            if flag == 'distributed':
                def f(_dist_arr):  # pragma: no cover
                    _d_arr = hpat.distributed_api.dist_input(_dist_arr)
            elif flag == 'threaded':
                def f(_thread_arr):  # pragma: no cover
                    _th_arr = hpat.distributed_api.threaded_input(_thread_arr)
            else:
                raise ValueError("Invalid input flag")
            f_block = compile_to_numba_ir(
                f, {'hpat': hpat}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [in_arr_tmp])
            nodes += f_block.body[:-3]  # remove none return
            nodes[-1].target = arg_var

        # handle timestamp Series
        # transform to array[dt64]
        # could be combined with distributed/threaded input
        if self.args[arg_ind] == timestamp_series_type:
            # replace arg var with tmp
            in_arr_tmp = ir.Var(scope, mk_unique_var("ts_series_input"), loc)
            nodes[-1].target = in_arr_tmp

            def f(_ts_series):  # pragma: no cover
                _dt_arr = hpat.hiframes_api.ts_series_to_arr_typ(_ts_series)

            f_block = compile_to_numba_ir(
                f, {'hpat': hpat}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [in_arr_tmp])
            nodes += f_block.body[:-3]  # remove none return
            nodes[-1].target = arg_var
            # remember that this variable is actually a Series
            self.df_cols.add(arg_var.name)
            self.ts_series_vars.add(arg_var.name)

        # input dataframe arg
        # TODO: support distributed input
        if isinstance(self.args[arg_ind], PandasDataFrameType):
            df_typ = self.args[arg_ind]
            df_items = {}
            for i, col in enumerate(df_typ.col_names):
                col_dtype = df_typ.col_types[i]
                if col_dtype == string_type:
                    alloc_dt = 11  # dummy string value
                elif col_dtype == types.boolean:
                    alloc_dt = "np.bool_"
                elif col_dtype == types.NPDatetime('ns'):
                    alloc_dt = 12  # XXX const code for dt64 since we can't init dt64 dtype
                else:
                    alloc_dt = "np.{}".format(col_dtype)

                func_text = "def f(_df):\n"
                func_text += "  _col_input_{} = hpat.hiframes_api.unbox_df_column(_df, {}, {})\n".format(col, i, alloc_dt)
                loc_vars = {}
                exec(func_text, {}, loc_vars)
                f = loc_vars['f']
                f_block = compile_to_numba_ir(f,
                            {'hpat': hpat, 'np': np}).blocks.popitem()[1]
                replace_arg_nodes(f_block, [arg_var])
                nodes += f_block.body[:-3]
                #
                if col_dtype == types.NPDatetime('ns'):
                    self.ts_series_vars.add(nodes[-1].target.name)
                df_items[col] = nodes[-1].target

            self.df_vars[arg_var.name] = df_items
            self._update_df_cols()

        return nodes

    def _run_return(self, ret_node):
        # e.g. {"A:return":"distributed"} -> "A"
        flagged_returns = { var_name.split(":")[0]: flag
                    for (var_name, flag) in self.locals.items()
                    if var_name.endswith(":return") }
        for v in flagged_returns.keys():
            self.locals.pop(v + ":return")
        nodes = [ret_node]
        # shortcut if no dist return
        if len(flagged_returns) == 0:
            return nodes
        cast = guard(get_definition, self.func_ir, ret_node.value)
        assert cast is not None, "return cast not found"
        assert isinstance(cast, ir.Expr) and cast.op == 'cast'
        scope = cast.value.scope
        loc = cast.loc
        # XXX: using split('.') since the variable might be renamed (e.g. A.2)
        ret_name = cast.value.name.split('.')[0]
        if ret_name in flagged_returns.keys():
            flag = flagged_returns[ret_name]
            nodes = self._gen_replace_dist_return(cast.value, flag)
            new_arr = nodes[-1].target
            new_cast = ir.Expr.cast(new_arr, loc)
            new_out = ir.Var(scope, mk_unique_var(flag + "_return"), loc)
            nodes.append(ir.Assign(new_cast, new_out, loc))
            ret_node.value = new_out
            nodes.append(ret_node)
            return nodes

        cast_def = guard(get_definition, self.func_ir, cast.value)
        if (cast_def is not None and isinstance(cast_def, ir.Expr)
                and cast_def.op == 'build_tuple'):
            nodes = []
            new_var_list = []
            for v in cast_def.items:
                vname = v.name.split('.')[0]
                if vname in flagged_returns.keys():
                    flag = flagged_returns[vname]
                    nodes += self._gen_replace_dist_return(v, flag)
                    new_var_list.append(nodes[-1].target)
                else:
                    new_var_list.append(v)
            new_tuple_node = ir.Expr.build_tuple(new_var_list, loc)
            new_tuple_var = ir.Var(scope, mk_unique_var("dist_return_tp"), loc)
            nodes.append(ir.Assign(new_tuple_node, new_tuple_var, loc))
            new_cast = ir.Expr.cast(new_tuple_var, loc)
            new_out = ir.Var(scope, mk_unique_var("dist_return"), loc)
            nodes.append(ir.Assign(new_cast, new_out, loc))
            ret_node.value = new_out
            nodes.append(ret_node)

        return nodes

    def _gen_replace_dist_return(self, var, flag):
        if flag == 'distributed':
            def f(_dist_arr):  # pragma: no cover
                _d_arr = hpat.distributed_api.dist_return(_dist_arr)
        elif flag == 'threaded':
            def f(_threaded_arr):  # pragma: no cover
                _th_arr = hpat.distributed_api.threaded_return(_threaded_arr)
        else:
            raise ValueError("Invalid return flag {}".format(flag))
        f_block = compile_to_numba_ir(
            f, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [var])
        return f_block.body[:-3]  # remove none return

    def _gen_rolling_init(self, win_size, func, center):
        nodes = []
        right_length = 0
        scope = win_size.scope
        loc = win_size.loc
        right_length = ir.Var(scope, mk_unique_var('zero_var'), scope)
        nodes.append(ir.Assign(ir.Const(0, loc), right_length, win_size.loc))

        def f(w):  # pragma: no cover
            return -w + 1
        f_block = compile_to_numba_ir(f, {}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [win_size])
        nodes.extend(f_block.body[:-2])  # remove none return
        left_length = nodes[-1].target

        if center:
            def f(w):  # pragma: no cover
                return -(w // 2)
            f_block = compile_to_numba_ir(f, {}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [win_size])
            nodes.extend(f_block.body[:-2])  # remove none return
            left_length = nodes[-1].target

            def f(w):  # pragma: no cover
                return (w // 2)
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
                                            kernel_func.__closure__,
                                            kernel_func.__defaults__, loc)
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
            and last_block.body[-1].value.value is None)
    last_block.body.pop()


def include_new_blocks(blocks, new_blocks, label, new_body):
    inner_blocks = add_offset_to_labels(new_blocks, ir_utils._max_label + 1)
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
    # new_body.clear()
    return label

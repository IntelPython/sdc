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
                            add_offset_to_labels, get_ir_of_code, find_const,
                            compile_to_numba_ir, replace_arg_nodes,
                            find_callname, guard, require, get_definition,
                            build_definitions, replace_vars_stmt, replace_vars_inner)

from numba.inline_closurecall import inline_closure_call
from numba.analysis import compute_cfg_from_blocks

import hpat
from hpat import (hiframes_api, utils, parquet_pio, config, hiframes_filter,
                  hiframes_join, hiframes_aggregate, hiframes_sort, hiframes_typed)
from hpat.utils import get_constant, NOT_CONSTANT, get_definitions, debug_prints, include_new_blocks
from hpat.hiframes_api import PandasDataFrameType
from hpat.str_ext import string_type

import numpy as np
import math
from hpat.parquet_pio import ParquetHandler
from hpat.pd_timestamp_ext import (datetime_date_type,
                                    datetime_date_to_int, int_to_datetime_date)
from hpat.pd_series_ext import SeriesType, BoxedSeriesType

df_col_funcs = ['pct_change',]
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
            call_list[0] in ['fix_df_array', 'fix_rolling_array',
            'concat', 'count', 'mean', 'quantile', 'var',
            'str_contains_regex', 'str_contains_noregex', 'column_sum',
            'nunique']):
        return True
    if (len(call_list) == 3 and call_list[1:] == ['hiframes_typed', hpat] and
            call_list[0]
            in ['_sum_handle_nan', '_mean_handle_nan', '_var_handle_nan']):
        return True
    if call_list == ['dist_return', 'distributed_api', hpat]:
        return True
    if call_list == ['unbox_df_column', 'hiframes_api', hpat]:
        return True
    if call_list == [list]:
        return True
    if call_list == ['groupby']:
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
        # replace inst variables as determined previously during the pass
        # currently use to keep lhs of Arg nodes intact
        self.replace_var_dict = {}

        # rolling call name -> [column_varname, win_size]
        self.rolling_calls = {}

        # df_var -> {col1:col1_var ...}
        self.df_vars = {}
        # df_var -> label where it is defined
        self.df_labels = {}
        # arrays that are df columns actually (pd.Series)
        self.df_cols = set()

        self.arrow_tables = {}
        self.reverse_copies = {}
        self.pq_handler = ParquetHandler(
            func_ir, typingctx, args, _locals, self.reverse_copies)

    def run(self):
        # FIXME: see why this breaks test_kmeans
        # remove_dels(self.func_ir.blocks)
        dprint_func_ir(self.func_ir, "starting hiframes")
        cfg = compute_cfg_from_blocks(self.func_ir.blocks)
        topo_order = find_topo_order(self.func_ir.blocks)
        for label in topo_order:
            self._get_reverse_copies(self.func_ir.blocks[label].body)
            new_body = []
            self._working_body = new_body
            old_body = self.func_ir.blocks[label].body
            for inst in old_body:
                ir_utils.replace_vars_stmt(inst, self.replace_var_dict)
                # df['col'] = arr
                if (isinstance(inst, ir.StaticSetItem)
                        and self._is_df_var(inst.target)):
                    new_body += self._run_df_set_column(inst, label, cfg)
                elif isinstance(inst, ir.Assign):
                    out_nodes = self._run_assign(inst, label)
                    if isinstance(out_nodes, list):
                        new_body.extend(out_nodes)
                    if isinstance(out_nodes, dict):
                        old_label = label
                        label = include_new_blocks(
                            self.func_ir.blocks, out_nodes, label, new_body)
                        # cfg needs to be updated since label is updated
                        # needed for set df column
                        # new block will have the same jump as old block
                        self.func_ir.blocks[label].body.append(old_body[-1])
                        cfg = compute_cfg_from_blocks(self.func_ir.blocks)
                        self.func_ir.blocks[label].body.pop()
                        new_body = []
                        self._working_body = new_body
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
        if debug_prints():  # pragma: no cover
            print("df_vars: ", self.df_vars)
        return

    def _run_assign(self, assign, label):
        lhs = assign.target.name
        rhs = assign.value

        if isinstance(rhs, ir.Expr):
            if rhs.op == 'call':
                return self._run_call(assign, label)

            # d = df['column']
            if (rhs.op == 'static_getitem' and self._is_df_var(rhs.value)
                    and isinstance(rhs.index, str)):
                assign.value = self._get_df_cols(rhs.value)[rhs.index]
                self.df_cols.add(lhs)  # save lhs as column

            # df1 = df[df.A > .5]
            if rhs.op == 'getitem' and self._is_df_var(rhs.value):
                # output df1 has same columns as df, create new vars
                scope = assign.target.scope
                loc = assign.target.loc
                in_df_col_names = self._get_df_col_names(rhs.value)
                df_col_map = {col: ir.Var(scope, mk_unique_var(col), loc)
                                for col in in_df_col_names}
                self._create_df(lhs, df_col_map, label)
                in_df = self._get_renamed_df(rhs.value)
                return [hiframes_filter.Filter(lhs, in_df.name, rhs.index,
                                               self.df_vars, rhs.loc)]

            # df.loc or df.iloc
            if (rhs.op == 'getattr' and self._is_df_var(rhs.value)
                    and rhs.attr in ['loc', 'iloc']):
                # FIXME: treat iloc and loc as regular df variables so getitem
                # turns them into filter. Only boolean array is supported
                self.df_vars[lhs] = self._get_df_cols(rhs.value)
                return []

            # if (rhs.op == 'getitem' and rhs.value.name in self.df_cols):
            #     self.col_filters.add(assign)

            # d = df.column
            if (rhs.op == 'getattr' and self._is_df_var(rhs.value)
                    and self._is_df_colname(rhs.value, rhs.attr)):
                df = rhs.value.name
                col_var = self._get_df_colvar(rhs.value, rhs.attr)
                assign.value = col_var
                self.df_cols.add(lhs)  # save lhs as column
                # need to remove the lhs definition so that find_callname can
                # match column function calls (i.e. A.f instead of df.A.f)
                assert self.func_ir._definitions[lhs] == [rhs], "invalid def"
                self.func_ir._definitions[lhs] = [None]

            if rhs.op == 'cast' and rhs.value.name in self.df_vars:
                return self._box_return_df(assign, self.df_vars[rhs.value.name])

        if isinstance(rhs, ir.Arg):
            return self._run_arg(assign, label)

        # handle copies lhs = f
        if isinstance(rhs, ir.Var) and rhs.name in self.df_vars:
            self.df_vars[lhs] = self.df_vars[rhs.name]
        if isinstance(rhs, ir.Var) and rhs.name in self.df_labels:
            self.df_labels[lhs] = self.df_labels[rhs.name]
        if isinstance(rhs, ir.Var) and rhs.name in self.df_cols:
            self.df_cols.add(lhs)
        if isinstance(rhs, ir.Var) and rhs.name in self.arrow_tables:
            self.arrow_tables[lhs] = self.arrow_tables[rhs.name]
        return [assign]

    def _run_call(self, assign, label):
        """handle calls and return new nodes if needed
        """
        lhs = assign.target
        rhs = assign.value

        func_name = ""
        func_mod = ""
        fdef = guard(find_callname, self.func_ir, rhs)
        if fdef is None:
            # could be make_function from list comprehension which is ok
            func_def = guard(get_definition, self.func_ir, rhs.func)
            if isinstance(func_def, ir.Expr) and func_def.op == 'make_function':
                return [assign]
            warnings.warn(
                "function call couldn't be found for initial analysis")
            return [assign]
        else:
            func_name, func_mod = fdef

        if fdef == ('DataFrame', 'pandas'):
            return self._handle_pd_DataFrame(assign, lhs, rhs, label)

        if fdef == ('Series', 'pandas'):
            return self._handle_pd_Series(assign, lhs, rhs)

        if fdef == ('len', 'builtins') and self._is_df_var(rhs.args[0]):
            return self._df_len(lhs, rhs.args[0])

        if fdef == ('read_table', 'pyarrow.parquet'):
            return self._handle_pq_read_table(assign, lhs, rhs)

        if (func_name == 'to_pandas' and isinstance(func_mod, ir.Var)
                and func_mod.name in self.arrow_tables):
            return self._handle_pq_to_pandas(assign, lhs, rhs, func_mod, label)

        if fdef == ('read_parquet', 'pandas'):
            return self._handle_pd_read_parquet(assign, lhs, rhs, label)

        if fdef == ('merge', 'pandas'):
            return self._handle_merge(assign, lhs, rhs, label)

        if fdef == ('concat', 'pandas'):
            return self._handle_concat(assign, lhs, rhs, label)

        if fdef == ('read_ros_images', 'hpat.ros'):
            return self._handle_ros(assign, lhs, rhs)

        # df.column.shift(3)
        if isinstance(func_mod, ir.Var) and func_mod.name in self.df_cols:
            return self._handle_column_call(assign, lhs, rhs, func_mod, func_name, label)

        # df.apply(lambda a:..., axis=1)
        if (isinstance(func_mod, ir.Var) and self._is_df_var(func_mod)
                and func_name == 'apply'):
            return self._handle_df_apply(assign, lhs, rhs, func_mod)

        # df.describe()
        if (isinstance(func_mod, ir.Var) and self._is_df_var(func_mod)
                and func_name == 'describe'):
            return self._handle_df_describe(assign, lhs, rhs, func_mod)

        # df.sort_values()
        if (isinstance(func_mod, ir.Var) and self._is_df_var(func_mod)
                and func_name == 'sort_values'):
            return self._handle_df_sort_values(assign, lhs, rhs, func_mod, label)

        # df.itertuples()
        if (isinstance(func_mod, ir.Var) and self._is_df_var(func_mod)
                and func_name == 'itertuples'):
            return self._handle_df_itertuples(assign, lhs, rhs, func_mod)

        # df.pivot_table()
        if (isinstance(func_mod, ir.Var) and self._is_df_var(func_mod)
                and func_name == 'pivot_table'):
            return self._handle_df_pivot_table(lhs, rhs, func_mod, label)

        res = self._handle_rolling_call(assign.target, rhs)
        if res is not None:
            return res

        # groupby aggregate
        # e.g. df.groupby('A')['B'].agg(lambda x: x.max()-x.min())
        if isinstance(func_mod, ir.Var) and self._is_groupby(func_mod):
            return self._handle_aggregate(lhs, rhs, func_mod, func_name, label)

        if fdef == ('fromfile', 'numpy'):
            return hpat.io._handle_np_fromfile(assign, lhs, rhs)

        if fdef == ('read_xenon', 'hpat.xenon_ext'):
            col_items, nodes = hpat.xenon_ext._handle_read(assign, lhs, rhs, self.func_ir)
            df_nodes, col_map = self._process_df_build_map(col_items)
            self._create_df(lhs.name, col_map, label)
            nodes += df_nodes
            return nodes

        return [assign]

    def _get_reverse_copies(self, body):
        for inst in body:
            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Var):
                self.reverse_copies[inst.value.name] = inst.target.name
        return

    def _handle_pd_DataFrame(self, assign, lhs, rhs, label):
        """transform pd.DataFrame({'A': A}) call
        """
        kws = dict(rhs.kws)
        if 'data' in kws:
            data = kws['data']
            if len(rhs.args) != 0:  # pragma: no cover
                raise ValueError(
                    "only data argument suppoted in pd.DataFrame()")
        else:
            if len(rhs.args) != 1:  # pragma: no cover
                raise ValueError(
                    "data argument in pd.DataFrame() expected")
            data = rhs.args[0]

        arg_def = guard(get_definition, self.func_ir, data)
        if (not isinstance(arg_def, ir.Expr)
                or arg_def.op != 'build_map'):  # pragma: no cover
            raise ValueError(
                "Invalid DataFrame() arguments (constant dict of columns expected)")
        nodes, items = self._fix_df_arrays(arg_def.items)
        df_nodes, col_map = self._process_df_build_map(items)
        nodes += df_nodes
        self._create_df(lhs.name, col_map, label)
        # remove DataFrame call
        return nodes

    def _handle_pd_Series(self, assign, lhs, rhs):
        """transform pd.Series(A) call
        """
        kws = dict(rhs.kws)
        if 'data' in kws:
            data = kws['data']
            if len(rhs.args) != 0:  # pragma: no cover
                raise ValueError(
                    "only data argument suppoted in pd.Series()")
        else:
            if len(rhs.args) != 1:  # pragma: no cover
                raise ValueError(
                    "data argument in pd.Series() expected")
            data = rhs.args[0]

        def f(arr):  # pragma: no cover
            df_arr = hpat.hiframes_api.to_series_type(hpat.hiframes_api.fix_df_array(arr))
        f_block = compile_to_numba_ir(
                f, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [data])
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = lhs
        self.df_cols.add(lhs.name)
        return nodes

    def _df_len(self, lhs, df_var):
        # run len on one of the columns
        # FIXME: it could potentially avoid remove dead for the column if
        # array analysis doesn't replace len() with it's size
        df_arrs = list(self.df_vars[df_var.name].values())
        # empty dataframe has 0 len
        if len(df_arrs) == 0:
            return [ir.Assign(ir.Const(0, lhs.loc), lhs, lhs.loc)]
        arr = df_arrs[0]
        def f(df_arr):  # pragma: no cover
            df_len = len(df_arr)
        f_block = compile_to_numba_ir(f, {}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [arr])
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = lhs
        return nodes

    def _handle_pq_read_table(self, assign, lhs, rhs):
        if len(rhs.args) != 1:  # pragma: no cover
            raise ValueError("Invalid read_table() arguments")
        self.arrow_tables[lhs.name] = rhs.args[0]
        return []

    def _handle_pq_to_pandas(self, assign, lhs, rhs, t_var, label):
        return self._gen_parquet_read(self.arrow_tables[t_var.name], lhs, label)

    def _gen_parquet_read(self, fname, lhs, label):
        col_items, col_types, nodes = self.pq_handler.gen_parquet_read(
            fname, lhs)
        df_nodes, col_map = self._process_df_build_map(col_items)
        nodes += df_nodes
        self._create_df(lhs.name, col_map, label)
        return nodes

    def _handle_pd_read_parquet(self, assign, lhs, rhs, label):
        fname = rhs.args[0]
        return self._gen_parquet_read(fname, lhs, label)

    def _handle_merge(self, assign, lhs, rhs, label):
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
        # add columns from left to output
        left_colnames = self._get_df_col_names(left_df)
        df_col_map = {col: ir.Var(scope, mk_unique_var(col), loc)
                                for col in left_colnames}
        # add columns from right to output
        right_colnames = self._get_df_col_names(right_df)
        df_col_map.update({col: ir.Var(scope, mk_unique_var(col), loc)
                                for col in right_colnames})
        self._create_df(lhs.name, df_col_map, label)
        return [hiframes_join.Join(lhs.name, self._get_renamed_df(left_df).name,
                                   self._get_renamed_df(right_df).name,
                                   left_on, right_on, self.df_vars, lhs.loc)]

    def _handle_concat(self, assign, lhs, rhs, label):
        if len(rhs.args) != 1 or len(rhs.kws) != 0:
            raise ValueError(
                "only a list/tuple argument is supported in concat")
        df_list = guard(get_definition, self.func_ir, rhs.args[0])
        if not isinstance(df_list, ir.Expr) or not (df_list.op
                                            in ['build_tuple', 'build_list']):
            raise ValueError("pd.concat input should be constant list or tuple")

        if len(df_list.items) == 0:
            # copied error from pandas
            raise ValueError("No objects to concatenate")

        first_varname = df_list.items[0].name
        # the inputs should be either dataframe or series
        if not (first_varname in self.df_vars
                or first_varname in self.df_cols):
            raise ValueError("pd.concat input should be pd.Series or pd.DataFrame")

        if first_varname in self.df_vars:
            return self._handle_concat_df(lhs, df_list, label)
        assert first_varname in self.df_cols
        # XXX convert build_list to build_tuple since Numba doesn't handle list of
        # arrays
        if df_list.op == 'build_list':
            df_list.op = 'build_tuple'
        return self._handle_concat_series(lhs, rhs)

    def _handle_concat_df(self, lhs, df_list, label):
        # TODO: handle non-numerical (e.g. string, datetime) columns
        nodes = []
        done_cols = {}
        i = 0
        for df in df_list.items:
            df_col_map = self._get_df_cols(df)
            for (c, v) in df_col_map.items():
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
                    if self._is_df_colname(other_df, c):
                        other_var = self._get_df_colvar(other_df, c)
                        args.append(other_var)
                        arg_names.append('_hpat_c' + str(i))
                        conc_arg_names.append('_hpat_c' + str(i))
                        i += 1
                    else:
                        # use a df column variable just for computing length
                        len_arg = self._get_df_col_vars(other_df)[0]
                        len_name = '_hpat_len' + str(i)
                        args.append(len_arg)
                        arg_names.append(len_name)
                        i += 1
                        out_name = '_hpat_out' + str(i)
                        conc_arg_names.append(out_name)
                        i += 1
                        # TODO: fix type
                        # TODO: allocate string array of NAs
                        allocs += "    {} = np.full(len({}), np.nan)\n".format(
                            out_name, len_name)

                func_text = "def f({}):\n".format(",".join(arg_names))
                func_text += allocs
                func_text += "    s = hpat.hiframes_api.to_series_type(hpat.hiframes_api.concat(({})))\n".format(
                    ",".join(conc_arg_names))
                loc_vars = {}
                exec(func_text, {}, loc_vars)
                f = loc_vars['f']

                f_block = compile_to_numba_ir(f,
                            {'hpat': hpat, 'np': np}).blocks.popitem()[1]
                replace_arg_nodes(f_block, args)
                nodes += f_block.body[:-3]
                done_cols[c] = nodes[-1].target

        self._create_df(lhs.name, done_cols, label)
        return nodes

    def _handle_concat_series(self, lhs, rhs):
        # defer to typed pass since the type might be non-numerical
        def f(arr_list):  # pragma: no cover
            concat_arr = hpat.hiframes_api.to_series_type(hpat.hiframes_api.concat(arr_list))
        f_block = compile_to_numba_ir(f, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, rhs.args)
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = lhs
        self.df_cols.add(lhs.name)
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
            # fix list(multi-dim arrays) (packing images)
            # FIXME: does this break for list(other things)?
            col_arr = self._fix_df_list_of_array(col_arr)

            def f(arr):  # pragma: no cover
                df_arr = hpat.hiframes_api.fix_df_array(arr)
            f_block = compile_to_numba_ir(
                f, {'hpat': hpat}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [col_arr])
            nodes += f_block.body[:-3]  # remove none return
            new_col_arr = nodes[-1].target
            new_list.append((col_varname, new_col_arr))
        return nodes, new_list

    def _fix_df_list_of_array(self, col_arr):
        list_call = guard(get_definition, self.func_ir, col_arr)
        if guard(find_callname, self.func_ir, list_call) == ('list', 'builtins'):
            return list_call.args[0]
        return col_arr

    def _process_df_build_map(self, items_list):
        df_cols = {}
        nodes = []
        for item in items_list:
            col_var = item[0]
            if isinstance(col_var, str):
                col_name = col_var
            else:
                col_name = get_constant(self.func_ir, col_var)
                if col_name is NOT_CONSTANT:  # pragma: no cover
                    raise ValueError(
                        "data frame column names should be constant")
            # cast to series type
            def f(arr):  # pragma: no cover
                df_arr = hpat.hiframes_api.to_series_type(arr)
            f_block = compile_to_numba_ir(
                f, {'hpat': hpat}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [item[1]])
            nodes += f_block.body[:-3]  # remove none return
            new_col_arr = nodes[-1].target
            df_cols[col_name] = new_col_arr
        return nodes, df_cols

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
        col_names = self._get_df_col_names(func_mod)

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

        # remove duplicates with set() since a column can be used multiple times
        used_cols = set(used_cols)
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
        df_col_map = self._get_df_cols(func_mod)
        col_vars = [df_col_map[c] for c in used_cols]
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

        col_names = self._get_df_col_names(func_mod)
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
        col_vars = self._get_df_col_vars(func_mod)
        replace_arg_nodes(f_block, col_vars)
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = lhs
        return nodes

    def _handle_df_sort_values(self, assign, lhs, rhs, df, label):
        kws = dict(rhs.kws)
        # find key array for sort ('by' arg)
        key_name = None
        if len(rhs.args) > 0:
            key_name = guard(find_const, self.func_ir, rhs.args[0])
        else:
            key_name = kws['by'] if 'by' in kws else None
        if key_name is None:
            raise ValueError("'by' argument is required for sort_values() "
                             "which should be a constant string")

        inplace = False
        if 'inplace' in kws and guard(find_const, self.func_ir, kws['inplace']) == True:
            inplace = True

        # TODO: support multiple columns as key
        # TODO: support ascending=False

        out = []
        df_cols = self._get_df_cols(df).copy()  # copy since it'll be modified
        if not inplace:
            new_df_cols = {}
            def cp_func(arr):
                arr_cp = arr.copy()
            for cname, cvar in df_cols.items():
                f_block = compile_to_numba_ir(cp_func, {}).blocks.popitem()[1]
                replace_arg_nodes(f_block, [cvar])
                out += f_block.body[:-3]
                new_df_cols[cname] = out[-1].target
            df_cols = new_df_cols
            self._create_df(lhs.name, df_cols.copy(), label)

        if key_name not in df_cols:
            raise ValueError("invalid sort key {}".format(key_name))
        key_var = df_cols.pop(key_name)

        out.append(hiframes_sort.Sort(df.name, key_var, df_cols, lhs.loc))
        return out

    def _handle_df_itertuples(self, assign, lhs, rhs, df_var):
        """pass df column names and variables to get_itertuples() to be able
        to create the iterator.
        e.g. get_itertuples("A", "B", A_arr, B_arr)
        """
        col_names = self._get_df_col_names(df_var)

        col_name_args = ', '.join(["c"+str(i) for i in range(len(col_names))])
        name_consts = ', '.join(["'{}'".format(c) for c in col_names])

        func_text = "def f({}):\n".format(col_name_args)
        func_text += "  it = hpat.hiframes_api.get_itertuples({}, {})\n"\
                                            .format(name_consts, col_name_args)

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        f_block = compile_to_numba_ir(
            f, {'hpat': hpat, 'np': np}).blocks.popitem()[1]

        col_vars = self._get_df_col_vars(df_var)
        replace_arg_nodes(f_block, col_vars)
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = lhs
        return nodes

    def _handle_column_call(self, assign, lhs, rhs, col_var, func_name, label):
        """
        Handle Series calls like:
          A = df.column.shift(3)
        """
        # TODO: handle map/apply differences
        if func_name in ['map', 'apply']:
            return self._handle_map(assign, lhs, rhs, col_var, label)

        if func_name == 'rolling':
            return self._handle_rolling_setup(assign, lhs, rhs, col_var)

        if func_name == 'str.contains':
            return self._handle_str_contains(assign, lhs, rhs, col_var)

        if func_name in df_col_funcs:
            return self._gen_column_call(lhs, rhs.args, col_var, func_name,
                                         dict(rhs.kws))
        return [assign]

    def _handle_map(self, assign, lhs, rhs, col_var, label):
        """translate df.A.map(lambda a:...) to prange()
        """
        # error checking: make sure there is function input only
        if len(rhs.args) != 1:
            raise ValueError("map expects 1 argument")
        func = guard(get_definition, self.func_ir, rhs.args[0])
        if func is None or not (isinstance(func, ir.Expr)
                                and func.op == 'make_function'):
            raise ValueError("lambda for map not found")

        out_typ = self._get_map_output_typ(col_var, func, label)

        # TODO: handle non numpy alloc types like string array
        # prange func to inline
        func_text = "def f(A):\n"
        func_text += "  numba.parfor.init_prange()\n"
        func_text += "  n = len(A)\n"
        func_text += "  S = numba.unsafe.ndarray.empty_inferred((n,))\n"
        func_text += "  for i in numba.parfor.internal_prange(n):\n"
        func_text += "    t = A[i]\n"
        func_text += "    S[i] = map_func(t)\n"
        if out_typ == datetime_date_type:
            func_text += "  ret = hpat.hiframes_api.to_date_series_type(S)\n"
        else:
            func_text += "  ret = S\n"

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        _globals = self.func_ir.func_id.func.__globals__
        f_ir = compile_to_numba_ir(f, {'numba': numba, 'np': np, 'hpat': hpat,
            'datetime_date_to_int': datetime_date_to_int,
            'int_to_datetime_date': int_to_datetime_date})
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

    def _get_map_output_typ(self, col_var, func, label):
        # wrapper function to get the output type
        def f(A):
            return map_func(A[0])

        return self._get_func_output_typ(col_var, func, f, label)

    def _get_func_output_typ(self, col_var, func, wrapper_func, label):
        # stich together all blocks before the current block for type inference
        # XXX: does control flow affect type inference in Numba?
        dummy_ir = self.func_ir.copy()
        dummy_ir.blocks[label].body.append(ir.Return(0, col_var.loc))
        topo_order = find_topo_order(dummy_ir.blocks)
        all_body = []
        for l in topo_order:
            if l == label:
                break
            all_body += dummy_ir.blocks[l].body

        # add nodes created for current block so far
        all_body += self._working_body
        dummy_ir.blocks = {0: ir.Block(col_var.scope, col_var.loc)}
        dummy_ir.blocks[0].body = all_body

        _globals = self.func_ir.func_id.func.__globals__
        _globals.update({'hpat': hpat, 'numba': numba, 'np': np})
        f_ir = compile_to_numba_ir(wrapper_func, {'hpat': hpat})
        # fix definitions to enable finding sentinel
        f_ir._definitions = build_definitions(f_ir.blocks)
        first_label = min(f_ir.blocks)
        for i, stmt in enumerate(f_ir.blocks[first_label].body):
            if (isinstance(stmt, ir.Assign)
                    and isinstance(stmt.value, ir.Expr)
                    and stmt.value.op == 'call'):
                fdef = guard(get_definition, f_ir, stmt.value.func)
                if isinstance(fdef, ir.Global) and fdef.name == 'map_func':
                    inline_closure_call(f_ir, _globals, f_ir.blocks[first_label], i, func)
                    break

        f_ir.blocks = ir_utils.simplify_CFG(f_ir.blocks)
        f_topo_order = find_topo_order(f_ir.blocks)
        assert isinstance(f_ir.blocks[f_topo_order[-1]].body[-1], ir.Return)
        output_var = f_ir.blocks[f_topo_order[-1]].body[-1].value
        first_label = f_topo_order[0]
        replace_arg_nodes(f_ir.blocks[first_label], [col_var])
        assert first_label != topo_order[0]  #  TODO: check for 0 and handle
        dummy_ir.blocks.update(f_ir.blocks)
        dummy_ir.blocks[0].body.append(ir.Jump(first_label, col_var.loc))
        # dead df code can cause type inference issues
        remove_dead(dummy_ir.blocks, dummy_ir.arg_names, dummy_ir)

        # run type inference on the dummy IR
        warnings = numba.errors.WarningsFixer(numba.errors.NumbaWarning)
        infer = numba.typeinfer.TypeInferer(self.typingctx, dummy_ir, warnings)
        for index, (name, ty) in enumerate(zip(dummy_ir.arg_names, self.args)):
            infer.seed_argument(name, index, ty)
        infer.build_constraint()
        infer.propagate()
        out_tp = infer.typevars[output_var.name].getone()
        # typemap, restype, calltypes = numba.compiler.type_inference_stage(self.typingctx, dummy_ir, self.args, None)
        return out_tp


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
            tmp_out = ir.Var(lhs.scope, mk_unique_var("rolling_out"), lhs.loc)
            self.df_cols.add(lhs.name)  # output is Series
            nodes = self._gen_rolling_call(rhs.args,
                                    *self.rolling_calls[func_def.value.name]
                                    + [func_name, tmp_out])
            def f(_arr):  # pragma: no cover
                _dt_arr = hpat.hiframes_api.to_series_type(_arr)

            f_block = compile_to_numba_ir(
                f, {'hpat': hpat}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [tmp_out])
            nodes += f_block.body[:-3]  # remove none return
            nodes[-1].target = lhs
            return nodes

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
        if func in ['pct_change', 'shift']:
            self.df_cols.add(out_var.name)  # output is Series except sum
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

        tmp_var = ir.Var(out_var.scope, mk_unique_var("tmp_shift"), out_var.loc)

        index_offsets = [0]
        fir_globals = self.func_ir.func_id.func.__globals__
        stencil_nodes = gen_stencil_call(
            col_var, tmp_var, kernel_func, index_offsets, fir_globals)

        border_text = 'def f(A):\n  A[0:{}] = np.nan\n'.format(shift_const)
        loc_vars = {}
        exec(border_text, {}, loc_vars)
        border_func = loc_vars['f']

        f_blocks = compile_to_numba_ir(border_func, {'np': np}).blocks
        block = f_blocks[min(f_blocks.keys())]
        replace_arg_nodes(block, [tmp_var])
        setitem_nodes = block.body[:-3]  # remove none return

        def f(arr):  # pragma: no cover
            df_arr = hpat.hiframes_api.to_series_type(arr)
        f_block = compile_to_numba_ir(
                f, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [tmp_var])
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = out_var

        return stencil_nodes + setitem_nodes + nodes

    def _is_groupby(self, agg_var):
        """determines whether variable is coming from groupby() or groupby()[]
        """
        var_def = guard(get_definition, self.func_ir, agg_var)
        # groupby()['B'] case
        if (isinstance(var_def, ir.Expr)
                and var_def.op in ['getitem', 'static_getitem']):
            return self._is_groupby(var_def.value)
        # groupby() called on column or df
        call_def = guard(find_callname, self.func_ir, var_def)
        if (call_def is not None and call_def[0] == 'groupby'
                and isinstance(call_def[1], ir.Var)
                and (call_def[1].name in self.df_cols
                or self._is_df_var(call_def[1]))):
            return True
        return False

    def _handle_df_pivot_table(self, lhs, rhs, df_var, label):
        kws = dict(rhs.kws)
        values_arg = self._get_str_arg('pivot_table', rhs.args, kws, 0, 'values')
        index_arg = self._get_str_arg('pivot_table', rhs.args, kws, 1, 'index')
        columns_arg = self._get_str_arg('pivot_table', rhs.args, kws, 2, 'columns')
        agg_func_arg = self._get_str_arg('pivot_table', rhs.args, kws, 3, 'aggfunc', 'mean')

        agg_func = self._get_agg_func(agg_func_arg, rhs)

        in_vars = {values_arg: self.df_vars[df_var.name][values_arg]}
        def _map_dummy_f(A):
            return map_func(A)
        out_typ = self._get_func_output_typ(in_vars[values_arg], agg_func, _map_dummy_f, label)
        out_types = {values_arg: out_typ}

        pivot_values = self._get_pivot_values(lhs.name)
        df_col_map = ({col: ir.Var(lhs.scope, mk_unique_var(col), lhs.loc)
                                for col in pivot_values})
        # df_col_map = ({col: ir.Var(lhs.scope, mk_unique_var(col), lhs.loc)
        #                         for col in [values_arg]})
        out_df = df_col_map.copy()
        self._create_df(lhs.name, out_df, label)
        pivot_arr = self.df_vars[df_var.name][columns_arg]

        return [hiframes_aggregate.Aggregate(
            lhs.name, df_var.name, index_arg, None, df_col_map,
            in_vars, self.df_vars[df_var.name][index_arg],
            agg_func, out_types, lhs.loc, pivot_arr, pivot_values)]


    def _get_pivot_values(self, varname):
        if varname not in self.reverse_copies or (self.reverse_copies[varname] + ':pivot') not in self.locals:
            raise ValueError("pivot_table() requires annotation of pivot values")
        new_name = self.reverse_copies[varname]
        values = self.locals.pop(new_name + ":pivot")
        return values

    def _get_str_arg(self, f_name, args, kws, arg_no, arg_name, default=None):
        arg = None
        if len(args) > arg_no:
            arg = guard(find_const, self.func_ir, args[arg_no])
        elif arg_name in kws:
            arg = guard(find_const, self.func_ir, kws[arg_name])

        if arg is None:
            if default is not None:
                return default
            raise ValueError(("{} requires '{}' argument as a"
                             "constant string").format(f_name, arg_name))
        return arg



    def _handle_aggregate(self, lhs, rhs, obj_var, func_name, label):
        # format df.groupby('A')['B'].agg(lambda x: x.max()-x.min())
        _supported_agg_funcs = ['agg', 'aggregate', 'sum', 'count', 'mean',
                                'min', 'max']
        # TODO: support aggregation functions sum, count, etc.
        if func_name not in _supported_agg_funcs:
            raise ValueError("only {} supported in groupby".format(
                                             ", ".join(_supported_agg_funcs)))

        agg_func = self._get_agg_func(func_name, rhs)

        # find selected output columns
        df_var, key_colname, as_index, out_colnames, explicit_select = self._analyze_agg_select(
                                                                       obj_var)

        # find input vars and output types
        out_types = {}
        in_vars = {}
        for out_cname in out_colnames:
            in_var = self.df_vars[df_var.name][out_cname]
            in_vars[out_cname] = in_var
            def f(A):
                return map_func(A)
            out_typ = self._get_func_output_typ(in_var, agg_func, f, label)
            out_types[out_cname] = out_typ

        # output column map, create dataframe if multiple outputs
        out_key_var = None
        if len(out_colnames) == 1 and explicit_select:
            df_col_map = {out_colnames[0]: lhs}
            self.df_cols.add(lhs.name)  # output is series
        else:
            df_col_map = ({col: ir.Var(lhs.scope, mk_unique_var(col), lhs.loc)
                                for col in out_colnames})
            out_df = df_col_map.copy()
            if as_index is False:
                out_key_var = ir.Var(lhs.scope, mk_unique_var(key_colname), lhs.loc)
                out_df[key_colname] = out_key_var

            self._create_df(lhs.name, out_df, label)

        return [hiframes_aggregate.Aggregate(
            lhs.name, df_var.name, key_colname, out_key_var, df_col_map,
            in_vars, self.df_vars[df_var.name][key_colname],
            agg_func, out_types, lhs.loc)]

    def _analyze_agg_select(self, obj_var):
        """analyze selection of columns in after groupby()
        e.g. groupby('A')['B'], groupby('A')['B', 'C'], groupby('A')
        """
        select_def = guard(get_definition, self.func_ir, obj_var)
        out_colnames = None
        explicit_select = False
        if isinstance(select_def, ir.Expr) and select_def.op == 'getitem':
            agg_var = select_def.value
            out_colnames = guard(find_const, self.func_ir, select_def.index)
            if not isinstance(out_colnames, (str, tuple)):
                raise ValueError("Groupby output column names should be constant")
            if isinstance(out_colnames, str):
                out_colnames = [out_colnames]
            explicit_select = True
        else:
            agg_var = obj_var

        # find groupby key and as_index
        groubpy_call = guard(get_definition, self.func_ir, agg_var)
        assert isinstance(groubpy_call, ir.Expr) and groubpy_call.op == 'call'
        kws = dict(groubpy_call.kws)
        as_index = True
        if 'as_index' in kws:
            as_index = guard(find_const, self.func_ir, kws['as_index'])
            if as_index is None:
                raise ValueError(
                    "groupby as_index argument should be constant")
        if len(groubpy_call.args) == 1:
            by_arg = groubpy_call.args[0]
        elif 'by' in kws:
            by_arg = kws['by']
        else:  # pragma: no cover
            raise ValueError("by argument for groupby() required")
        key_colname = guard(find_const, self.func_ir, by_arg)

        # find dataframe
        call_def = guard(find_callname, self.func_ir, groubpy_call)
        assert (call_def is not None and call_def[0] == 'groupby'
                and isinstance(call_def[1], ir.Var)
                and self._is_df_var(call_def[1]))
        df_var = call_def[1]

        if out_colnames is None:
            out_colnames = list(self.df_vars[df_var.name].keys())
            # key arr is not output by default
            # as_index should be handled separately since it just returns keys
            out_colnames.remove(key_colname)

        return df_var, key_colname, as_index, out_colnames, explicit_select

    def _get_agg_func(self, func_name, rhs):

        if func_name in ['sum', 'count', 'mean', 'max', 'min']:
            return hiframes_typed.series_replace_funcs[func_name]

        assert func_name in ['agg', 'aggregate']
        # agg case
        # error checking: make sure there is function input only
        if len(rhs.args) != 1:
            raise ValueError("agg expects 1 argument")
        agg_func = guard(get_definition, self.func_ir, rhs.args[0])
        if agg_func is None or not (isinstance(agg_func, ir.Expr)
                                and agg_func.op == 'make_function'):
            raise ValueError("lambda for map not found")

        return agg_func

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

    def _run_arg(self, arg_assign, label):
        nodes = [arg_assign]
        arg_name = arg_assign.value.name
        arg_ind = arg_assign.value.index
        arg_var = arg_assign.target
        arg_typ = self.args[arg_ind]
        scope = arg_var.scope
        loc = arg_var.loc

        # e.g. {"A:return":"distributed"} -> "A"
        flagged_inputs = { var_name.split(":")[0]: flag
                    for (var_name, flag) in self.locals.items()
                    if var_name.endswith(":input") }

        if arg_name in flagged_inputs.keys():
            self.locals.pop(arg_name + ":input")
            flag = flagged_inputs[arg_name]
            if flag== 'series':
                # XXX hack for agg functions, replace with proper Series type
                self.df_cols.add(arg_var.name)
                return nodes
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
            replace_arg_nodes(f_block, [arg_var])
            nodes += f_block.body[:-3]  # remove none return
            new_arg_var = ir.Var(scope, mk_unique_var(arg_name), loc)
            nodes[-1].target = new_arg_var
            self.replace_var_dict[arg_var.name] = new_arg_var
            self._add_node_defs(nodes)

        # TODO: handle datetime.date() series

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
                func_text += "  _col_input_{} = hpat.hiframes_api.to_series_type(hpat.hiframes_api.unbox_df_column(_df, {}, {}))\n".format(col, i, alloc_dt)
                loc_vars = {}
                exec(func_text, {}, loc_vars)
                f = loc_vars['f']
                f_block = compile_to_numba_ir(f,
                            {'hpat': hpat, 'np': np}).blocks.popitem()[1]
                replace_arg_nodes(f_block, [arg_var])
                nodes += f_block.body[:-3]
                df_items[col] = nodes[-1].target

            self._create_df(arg_var.name, df_items, label)

        if isinstance(arg_typ, BoxedSeriesType):
            # self.args[arg_ind] = SeriesType(arg_typ.dtype, 1, 'C')
            # replace arg var with tmp
            def f(_boxed_series):  # pragma: no cover
                _dt_arr = hpat.hiframes_api.to_series_type(hpat.hiframes_api.dummy_unbox_series(_boxed_series))

            f_block = compile_to_numba_ir(
                f, {'hpat': hpat}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [arg_var])
            nodes += f_block.body[:-3]  # remove none return
            new_arg_var = ir.Var(scope, mk_unique_var(arg_name), loc)
            nodes[-1].target = new_arg_var
            self.df_cols.add(new_arg_var.name)
            self.replace_var_dict[arg_var.name] = new_arg_var
            self._add_node_defs(nodes)

        return nodes

    def _box_return_df(self, cast_assign, df_map):
        #
        arrs = list(df_map.values())
        names = list(df_map.keys())
        n_cols = len(arrs)

        arg_names = ", ".join(['in_{}'.format(i) for i in range(n_cols)])
        col_names = ", ".join(['"{}"'.format(cname) for cname in names])

        func_text = "def f({}):\n".format(arg_names)
        func_text += "  _dt_arr = hpat.hiframes_api.box_df({}, {})\n".format(col_names, arg_names)
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        f_block = compile_to_numba_ir(
            f, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, arrs)
        nodes = f_block.body[:-3]  # remove none return
        cast_assign.value = nodes[-1].target
        nodes.append(cast_assign)
        return nodes


    def _add_node_defs(self, nodes):
        # TODO: add node defs for all new nodes
        loc = ir.Loc("", -1)
        dummy_block = ir.Block(ir.Scope(None, loc), loc)
        dummy_block.body = nodes
        build_definitions({0: dummy_block}, self.func_ir._definitions)

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

    def _run_df_set_column(self, inst, label, cfg):
        """handle setitem: df['col_name'] = arr
        """
        # TODO: generalize to more cases
        # TODO: rename the dataframe variable to keep schema static
        df_label = self.df_labels[inst.target.name]
        # setting column possible only when it dominates the df creation to
        # keep schema consistent
        if label not in cfg.backbone() and label not in cfg.post_dominators()[df_label]:
            raise ValueError("setting dataframe columns inside conditionals and"
                             " loops not supported yet")
        if not isinstance(inst.index, str):
            raise ValueError("dataframe column name should be a string constant")

        df_name = inst.target.name
        self.df_vars[df_name][inst.index] = inst.value
        self._update_df_cols()

        # set dataframe column if it is input and needs to be reflected
        df_def = guard(get_definition, self.func_ir, df_name)
        if isinstance(df_def, ir.Arg):
            # assign column name to variable
            cname_var = ir.Var(inst.value.scope, mk_unique_var("$cname_const"), inst.loc)
            nodes = [ir.Assign(ir.Const(inst.index, inst.loc), cname_var, inst.loc)]
            series_arr = inst.value

            def f(_df, _cname, _arr):  # pragma: no cover
                s = hpat.hiframes_api.set_df_col(_df, _cname, _arr)

            f_block = compile_to_numba_ir(f, {'hpat': hpat}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [inst.target, cname_var, series_arr])
            # copy propagate to enable string Const in typing and lowering
            simple_block_copy_propagate(f_block)
            nodes += f_block.body[:-3]  # remove none return
            return nodes

        return []

    def _create_df(self, df_varname, df_col_map, label):
        # order is important for proper handling of itertuples, apply, etc.
        # starting pandas 0.23 and Python 3.6, regular dict order is OK
        # for <0.23 ordered_df_map = OrderedDict(sorted(df_col_map.items()))
        self.df_vars[df_varname] = df_col_map
        self._update_df_cols()
        self.df_labels[df_varname] = label

    def _is_df_colname(self, df_var, cname):
        """ is cname a column name in df_var
        """
        df_var_renamed = self._get_renamed_df(df_var)
        return cname in self.df_vars[df_var_renamed.name]


    def _is_df_var(self, var):
        assert isinstance(var, ir.Var)
        return (var.name in self.df_vars)

    def _get_df_cols(self, df_var):
        #
        assert isinstance(df_var, ir.Var)
        df_var_renamed = self._get_renamed_df(df_var)
        return self.df_vars[df_var_renamed.name]

    def _get_df_col_names(self, df_var):
        assert isinstance(df_var, ir.Var)
        df_var_renamed = self._get_renamed_df(df_var)
        return list(self.df_vars[df_var_renamed.name].keys())

    def _get_df_col_vars(self, df_var):
        #
        assert isinstance(df_var, ir.Var)
        df_var_renamed = self._get_renamed_df(df_var)
        return list(self.df_vars[df_var_renamed.name].values())

    def _get_df_colvar(self, df_var, cname):
        assert isinstance(df_var, ir.Var)
        df_var_renamed = self._get_renamed_df(df_var)
        return self.df_vars[df_var_renamed.name][cname]

    def _get_renamed_df(self, df_var):
        # XXX placeholder for df variable renaming
        assert isinstance(df_var, ir.Var)
        return df_var


def gen_empty_like(in_arr, out_arr):
    def f(A):  # pragma: no cover
        B = np.empty(A.shape, A.dtype)
    f_block = compile_to_numba_ir(f, {'hpat': hpat, 'np': np}).blocks.popitem()[1]
    replace_arg_nodes(f_block, [in_arr])
    nodes = f_block.body[:-3]  # remove none return
    nodes[-1].target = out_arr
    return nodes


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
        in_arr = hpat.hiframes_api.to_arr_from_series(A)
        numba.stencil(f)(in_arr, out=B)
    f_block = compile_to_numba_ir(f, {'numba': numba,
        'hpat': hpat}).blocks.popitem()[1]
    replace_arg_nodes(f_block, [in_arr, out_arr, kernel_var])
    stencil_nodes += f_block.body[:-3]  # remove none return
    setup_call = stencil_nodes[-2].value
    stencil_call = stencil_nodes[-1].value
    setup_call.kws = list(options.items())
    stencil_call.args += other_args

    return stencil_nodes


def simple_block_copy_propagate(block):
    """simple copy propagate for a single block before typing, without Parfor"""

    var_dict = {}
    # assignments as dict to replace with latest value
    for stmt in block.body:
        # only rhs of assignments should be replaced
        # e.g. if x=y is available, x in x=z shouldn't be replaced
        if isinstance(stmt, ir.Assign):
            stmt.value = replace_vars_inner(stmt.value, var_dict)
        else:
            replace_vars_stmt(stmt, var_dict)
        if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Var):
            lhs = stmt.target.name
            rhs = stmt.value.name
            # rhs could be replaced with lhs from previous copies
            if lhs != rhs:
                var_dict[lhs] = stmt.value
                # a=b kills previous t=a
                lhs_kill = []
                for k, v in var_dict.items():
                    if v.name == lhs:
                        lhs_kill.append(k)
                for k in lhs_kill:
                    var_dict.pop(k, None)
        if (isinstance(stmt, ir.Assign)
                                    and not isinstance(stmt.value, ir.Var)):
            lhs = stmt.target.name
            var_dict.pop(lhs, None)
            # previous t=a is killed if a is killed
            lhs_kill = []
            for k, v in var_dict.items():
                if v.name == lhs:
                    lhs_kill.append(k)
            for k in lhs_kill:
                var_dict.pop(k, None)
    return

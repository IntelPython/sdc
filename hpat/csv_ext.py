
from collections import defaultdict
import numba
from numba import typeinfer, ir, ir_utils, config, types
from numba.extending import overload
from numba.ir_utils import (visit_vars_inner, replace_vars_inner,
                            compile_to_numba_ir, replace_arg_nodes)
import hpat
from hpat import distributed, distributed_analysis
from hpat.utils import debug_prints, alloc_arr_tup, empty_like_type
from hpat.distributed_analysis import Distribution
from hpat.hiframes_sort import (
    alloc_shuffle_metadata, data_alloc_shuffle_metadata, alltoallv,
    alltoallv_tup, finalize_shuffle_meta, finalize_data_shuffle_meta,
    update_shuffle_meta, update_data_shuffle_meta,
    )
from hpat.str_arr_ext import (string_array_type, to_string_list,
                              cp_str_list_to_array, str_list_to_array,
                              get_offset_ptr, get_data_ptr, convert_len_arr_to_offset,
                              pre_alloc_string_array, del_str, num_total_chars,
                              getitem_str_offset, copy_str_arr_slice, setitem_string_array)
from hpat.hiframes_api import str_copy_ptr
from hpat.timsort import copyElement_tup, getitem_arr_tup
import numpy as np


class CsvReader(ir.Stmt):
    def __init__(self, df_out, out_vars, out_types, loc):
        self.df_out = df_out
        self.out_vars = out_vars
        self.out_types = out_types
        self.loc = loc

    def __repr__(self):  # pragma: no cover
        # TODO
        return "{} = ReadCsv()".format(self.df_out)


def csv_array_analysis(csv_node, equiv_set, typemap, array_analysis):
    post = []
    # empty csv nodes should be deleted in remove dead
    assert len(csv_node.out_vars) > 0, "empty csv in array analysis"

    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variable for an output column
    all_shapes = []
    for col_var in csv_node.out_vars:
        typ = typemap[col_var.name]
        if typ == string_array_type:
            continue
        (shape, c_post) = array_analysis._gen_shape_call(
            equiv_set, col_var, typ.ndim, None)
        equiv_set.insert_equiv(col_var, shape)
        post.extend(c_post)
        all_shapes.append(shape[0])
        equiv_set.define(col_var)

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], post


numba.array_analysis.array_analysis_extensions[CsvReader] = csv_array_analysis


def csv_distributed_analysis(csv_node, array_dists):
    # TODO
    return


distributed_analysis.distributed_analysis_extensions[CsvReader] = csv_distributed_analysis


def csv_typeinfer(csv_node, typeinferer):
    for col_var, typ in zip(csv_node.out_vars, csv_node.out_types):
        typeinferer.lock_type(col_var.name, typ, loc=csv_node.loc)
    return


typeinfer.typeinfer_extensions[CsvReader] = csv_typeinfer


def visit_vars_csv(csv_node, callback, cbdata):
    if debug_prints():  # pragma: no cover
        print("visiting csv vars for:", csv_node)
        print("cbdata: ", sorted(cbdata.items()))

    # update output_vars
    new_out_vars = []
    for col_var in csv_node.out_vars:
        new_var = visit_vars_inner(col_var, callback, cbdata)
        new_out_vars.append(new_var)

    csv_node.out_vars = new_out_vars
    return

# add call to visit csv variable
ir_utils.visit_vars_extensions[CsvReader] = visit_vars_csv


def remove_dead_csv(csv_node, lives, arg_aliases, alias_map, func_ir, typemap):
    # TODO
    return csv_node


ir_utils.remove_dead_extensions[CsvReader] = remove_dead_csv


def csv_usedefs(csv_node, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # output columns are defined
    def_set.update({v.name for v in csv_node.out_vars})

    return numba.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.analysis.ir_extension_usedefs[CsvReader] = csv_usedefs


def get_copies_csv(csv_node, typemap):
    # csv doesn't generate copies, it just kills the output columns
    kill_set = set(v.name for v in csv_node.out_vars)
    return set(), kill_set


ir_utils.copy_propagate_extensions[CsvReader] = get_copies_csv


def apply_copies_csv(csv_node, var_dict, name_var_table,
                      typemap, calltypes, save_copies):
    """apply copy propagate in csv node"""

    # update output_vars
    new_out_vars = []
    for col_var in csv_node.out_vars:
        new_var = replace_vars_inner(col_var, var_dict)
        new_out_vars.append(new_var)

    csv_node.out_vars = new_out_vars
    return


ir_utils.apply_copy_propagate_extensions[CsvReader] = apply_copies_csv

def build_csv_definitions(csv_node, definitions=None):
    if definitions is None:
        definitions = defaultdict(list)

    for col_var in csv_node.out_vars:
        definitions[col_var.name].append(csv_node)

    return definitions

ir_utils.build_defs_extensions[CsvReader] = build_csv_definitions

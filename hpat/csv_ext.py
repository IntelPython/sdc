
from collections import defaultdict
import numba
from numba import typeinfer, ir, ir_utils, config, types, cgutils
from numba.typing.templates import signature
from numba.extending import overload, intrinsic, register_model, models, box
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
from hpat.hiframes_api import PandasDataFrameType, lower_unbox_df_column
from hpat.str_ext import string_type
from hpat.str_arr_ext import (string_array_type, to_string_list,
                              cp_str_list_to_array, str_list_to_array,
                              get_offset_ptr, get_data_ptr, convert_len_arr_to_offset,
                              pre_alloc_string_array, del_str, num_total_chars,
                              getitem_str_offset, copy_str_arr_slice, setitem_string_array)
from hpat.timsort import copyElement_tup, getitem_arr_tup
from hpat.utils import _numba_to_c_type_map
from hpat import objmode
import pandas as pd
import numpy as np


class CsvReader(ir.Stmt):
    def __init__(self, file_name, df_out, df_colnames, out_vars, out_types, loc):
        self.file_name = file_name
        self.df_out = df_out
        self.df_colnames = df_colnames
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
    csv_node.file_name = visit_vars_inner(csv_node.file_name, callback, cbdata)
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
    use_set.add(csv_node.file_name.name)

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
    csv_node.file_name = replace_vars_inner(csv_node.file_name, var_dict)
    return


ir_utils.apply_copy_propagate_extensions[CsvReader] = apply_copies_csv

def build_csv_definitions(csv_node, definitions=None):
    if definitions is None:
        definitions = defaultdict(list)

    for col_var in csv_node.out_vars:
        definitions[col_var.name].append(csv_node)

    return definitions

ir_utils.build_defs_extensions[CsvReader] = build_csv_definitions

import hio
from llvmlite import ir as lir
import llvmlite.binding as ll
ll.add_symbol('csv_file_chunk_reader', hio.csv_file_chunk_reader)

@intrinsic(support_literals=True)
def _csv_read(typingctx, fname_typ, cols_to_read_typ, cols_to_read_names_typ, dtypes_typ, n_cols_to_read_typ, delims_typ, quotes_typ):
    '''_csv_read(        fname,     (0, 1, 2, 3,),    'A,B,C,D',              (4, 6, 6, 4,), 4, ',', '"')
    This is creating the llvm wrapper calling the C function.
    '''
    # FIXME how to check types of const inputs?
    assert fname_typ == string_type
    assert isinstance(cols_to_read_typ, types.Const)
    assert isinstance(cols_to_read_names_typ, types.Const)
    assert isinstance(dtypes_typ, types.Const)
    # assert cols_to_read_typ.count == dtypes_typ.count
    assert isinstance(n_cols_to_read_typ, types.Const)
    assert delims_typ == string_type or isinstance(delims_typ, types.Const)
    assert quotes_typ == string_type or isinstance(quotes_typ, types.Const)
    ctype_enum_to_nb_typ = {v: k for k, v in _numba_to_c_type_map.items()}
    return_typ = types.Tuple([types.Array(ctype_enum_to_nb_typ[t], 1, 'C')
        for t in dtypes_typ.value])
    ncols = len(cols_to_read_typ.value)
    colnames = cols_to_read_names_typ.value.split(',')

    def codegen(context, builder, sig, args):
        # we simply cast the input tuples to a c-pointers
        cols_ptr = cgutils.alloca_once(builder, args[1].type)
        builder.store(args[1], cols_ptr)
        cols_ptr = builder.bitcast(cols_ptr, lir.IntType(64).as_pointer())
        dtypes_ptr = cgutils.alloca_once(builder, args[2].type)
        builder.store(args[2], dtypes_ptr)
        dtypes_ptr = builder.bitcast(dtypes_ptr, lir.IntType(64).as_pointer())
        # we need extra pointers for output parameters
        first_row_ptr = cgutils.alloca_once(builder, lir.IntType(64))
        n_rows_ptr = cgutils.alloca_once(builder, lir.IntType(64))
        # define function type, it returns an pandas dataframe (PyObject*)
        fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                                [lir.IntType(8).as_pointer(),  # const std::string * fname,
                                 lir.IntType(64).as_pointer(), # size_t * cols_to_read
                                 lir.IntType(64).as_pointer(), # int64_t * dtypes
                                 lir.IntType(64),              # size_t n_cols_to_read
                                 lir.IntType(64).as_pointer(), # size_t * first_row,
                                 lir.IntType(64).as_pointer(), # size_t * n_rows,
                                 lir.IntType(8).as_pointer(),  # std::string * delimiters
                                 lir.IntType(8).as_pointer(),  # std::string * quotes
                                ])
        fn = builder.module.get_or_insert_function(fnty, name='csv_read_file')
        call_args = [args[0], cols_ptr, dtypes_ptr, args[4], first_row_ptr, n_rows_ptr, args[5], args[6]]
        df = builder.call(fn, call_args)
        # pyapi = context.get_python_api(builder)
        # pyapi.print_object(df)

        # pyapi = context.get_python_api(builder)
        # ub_ctxt = numba.pythonapi._UnboxContext(context, builder, pyapi)
        # df = ub_ctxt.pyapi.to_native_value(PandasDataFrameType, pydf)

        # create a tuple of arrays from returned meminfo pointers
        ll_ret_typ = context.get_data_type(sig.return_type)
        out_arr_tup = cgutils.alloca_once(builder, ll_ret_typ)
        num_rows = builder.load(n_rows_ptr)

        for i, arr_typ in enumerate(sig.return_type.types):
            unbox_sig = signature(
                0, PandasDataFrameType(colnames, sig.return_type.types),
                types.Const(i), arr_typ)
            arr = lower_unbox_df_column(context,
                                        builder,
                                        unbox_sig,
                                        [df, context.get_constant(types.intp, i), 0])
            builder.store(arr,
                          cgutils.gep_inbounds(builder, out_arr_tup, 0, i))

        return builder.load(out_arr_tup)

    #    return types.CPointer(types.MemInfoPointer(types.byte))(cols_to_read_typ, dtypes_typ, n_cols_to_read_typ, delims_typ, quotes_typ)
    cols_int_tup_typ = types.UniTuple(types.intp, ncols)
    arg_typs = (string_type, cols_int_tup_typ, string_type, cols_int_tup_typ, types.intp, string_type, string_type)
    return return_typ(*arg_typs), codegen


def csv_distributed_run(csv_node, array_dists, typemap, calltypes, typingctx, targetctx):
    parallel = True
    for v in csv_node.out_vars:
        if (array_dists[v.name] != distributed.Distribution.OneD
                and array_dists[v.name] != distributed.Distribution.OneD_Var):
            parallel = False

    n_cols = len(csv_node.out_vars)
    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    # get column variables
    arg_names = ", ".join("arr" + str(i) for i in range(n_cols))
    col_inds = ", ".join(str(i) for i in range(n_cols))
    col_names = ",".join("'{}'".format(nm) for nm in csv_node.df_colnames)
    col_typs = ", ".join(str(_numba_to_c_type_map[arr_typ.dtype]) for arr_typ in csv_node.out_types)
    #col_typs = [str(_numba_to_c_type_map[arr_typ.dtype]) for arr_typ in csv_node.out_types]
    func_text  = "def csv_impl(fname):\n"
    # func_text += "    ({},) = _csv_reader_py(fname, ({},), [{},], ({},))\n".format(
    #     arg_names, col_inds, col_names, col_typs)
    func_text += "    ({},) = _csv_reader_py(fname)\n".format(
    arg_names)
    print(func_text)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    csv_impl = loc_vars['csv_impl']

    csv_reader_py = _gen_csv_reader_py(col_inds, csv_node.df_colnames, csv_node.out_types, typingctx, targetctx)

    f_block = compile_to_numba_ir(csv_impl,
                                  {'_csv_reader_py': csv_reader_py},
                                  typingctx, (string_type,),
                                  typemap, calltypes).blocks.popitem()[1]
    replace_arg_nodes(f_block, [csv_node.file_name])
    nodes = f_block.body[:-3]
    for i in range(len(csv_node.out_vars)):
        nodes[-len(csv_node.out_vars) + i].target = csv_node.out_vars[i]

    return nodes


distributed.distributed_run_extensions[CsvReader] = csv_distributed_run


class HPATIOType(types.Opaque):
    def __init__(self):
        super(HPATIOType, self).__init__(name='HPATIOType')

hpatio_type = HPATIOType()
register_model(HPATIOType)(models.OpaqueModel)

@box(HPATIOType)
def box_hpatio(typ, val, c):
    return val

csv_file_chunk_reader = types.ExternalFunction("csv_file_chunk_reader", hpatio_type(string_type))


def _gen_csv_reader_py(col_inds, col_names, col_typs, typingctx, targetctx):

    func_text = "def csv_reader_py(fname):\n"
    func_text += "  f_reader = csv_file_chunk_reader(fname)\n"
    func_text += "  with objmode(A='float64[::1]'):\n"
    func_text += "    df = pd.read_csv(f_reader, names=['A'])\n"
    func_text += "    A = df.A.values\n"
    func_text += "  return (A,)\n"

    glbls = {'objmode': objmode, 'csv_file_chunk_reader': csv_file_chunk_reader, 'pd': pd, 'np': np}
    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    csv_reader_py = loc_vars['csv_reader_py']

    # return numba.njit(csv_reader_py)
    from numba import compiler
    from numba.ir_utils import compile_to_numba_ir, get_ir_of_code

    f_ir = get_ir_of_code(glbls, csv_reader_py.__code__)
    main_ir, withs = numba.transforms.with_lifting(
        func_ir=f_ir,
        typingctx=typingctx,
        targetctx=targetctx,
        flags=compiler.DEFAULT_FLAGS,
        locals={},
    )
    out_tp = types.Tuple([types.Array(types.float64, 1, 'C')])
    csv_reader_py_func = compiler.compile_ir(
            typingctx,
            targetctx,
            main_ir,
            (string_type,),
            out_tp,
            compiler.DEFAULT_FLAGS,
            {},
            lifted=tuple(withs), lifted_from=None
    )
    imp_dis = numba.targets.registry.dispatcher_registry['cpu'](csv_reader_py)
    imp_dis.add_overload(csv_reader_py_func)
    return imp_dis

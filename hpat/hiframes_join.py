from __future__ import print_function, division, absolute_import

import numba
from numba import typeinfer, ir, ir_utils, config, types
from numba.ir_utils import (visit_vars_inner, replace_vars_inner,
    compile_to_numba_ir, replace_arg_nodes)
import hpat
from hpat import distributed, distributed_analysis
from hpat.distributed_analysis import Distribution
from hpat.str_arr_ext import string_array_type
import numpy as np

class Join(ir.Stmt):
    def __init__(self, df_out, left_df, right_df, left_key, right_key, df_vars, loc):
        self.df_out = df_out
        self.left_df = left_df
        self.right_df = right_df
        self.left_key = left_key
        self.right_key = right_key
        self.df_out_vars = df_vars[self.df_out]
        self.left_vars = df_vars[left_df]
        self.right_vars = df_vars[right_df]
        # needs df columns for type inference stage
        self.df_vars = df_vars
        self.loc = loc

    def __repr__(self):  # pragma: no cover
        out_cols = ""
        for (c, v) in self.df_out_vars.items():
            out_cols += "'{}':{}, ".format(c, v.name)
        df_out_str = "{}{{{}}}".format(self.df_out, out_cols)

        in_cols = ""
        for (c, v) in self.left_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_left_str = "{}{{{}}}".format(self.left_df, in_cols)

        in_cols = ""
        for (c, v) in self.right_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_right_str = "{}{{{}}}".format(self.right_df, in_cols)
        return "join [{}={}]: {} , {}, {}".format(self.left_key,
            self.right_key, df_out_str, df_left_str, df_right_str)

def join_array_analysis(join_node, equiv_set, typemap, array_analysis):
    post = []
    # empty join nodes should be deleted in remove dead
    assert len(join_node.df_out_vars) > 0, "empty join in array analysis"

    # arrays of left_df and right_df have same size in first dimension
    all_shapes = []
    for _, col_var in (list(join_node.left_vars.items())
                        +list(join_node.right_vars.items())):
        typ = typemap[col_var.name]
        if typ == string_array_type:
            continue
        col_shape = equiv_set.get_shape(col_var)
        all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variable for an output column
    all_shapes = []
    for _, col_var in join_node.df_out_vars.items():
        typ = typemap[col_var.name]
        if typ == string_array_type:
            continue
        (shape, c_post) = array_analysis._gen_shape_call(equiv_set, col_var, typ.ndim, None)
        equiv_set.insert_equiv(col_var, shape)
        post.extend(c_post)
        all_shapes.append(shape[0])
        equiv_set.define(col_var)

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], post

numba.array_analysis.array_analysis_extensions[Join] = join_array_analysis

def join_distributed_analysis(join_node, array_dists):

    # input columns have same distribution
    in_dist = Distribution.OneD
    for _, col_var in (list(join_node.left_vars.items())
                        +list(join_node.right_vars.items())):
        in_dist = Distribution(min(in_dist.value, array_dists[col_var.name].value))


    # output columns have same distribution
    out_dist = Distribution.OneD_Var
    for _, col_var in join_node.df_out_vars.items():
        # output dist might not be assigned yet
        if col_var.name in array_dists:
            out_dist = Distribution(min(out_dist.value, array_dists[col_var.name].value))

    # out dist should meet input dist (e.g. REP in causes REP out)
    out_dist = Distribution(min(out_dist.value, in_dist.value))
    for _, col_var in join_node.df_out_vars.items():
        array_dists[col_var.name] = out_dist

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        array_dists[join_node.bool_arr.name] = out_dist
        for _, col_var in join_node.df_in_vars.items():
            array_dists[col_var.name] = out_dist

    return

distributed_analysis.distributed_analysis_extensions[Join] = join_distributed_analysis

def join_typeinfer(join_node, typeinferer):
    # TODO: consider keys with same name, cols with suffix
    for col_name, col_var in (list(join_node.left_vars.items())
                        +list(join_node.right_vars.items())):
        out_col_var = join_node.df_out_vars[col_name]
        typeinferer.constraints.append(typeinfer.Propagate(dst=out_col_var.name,
                                              src=col_var.name, loc=join_node.loc))
    return

typeinfer.typeinfer_extensions[Join] = join_typeinfer


def visit_vars_join(join_node, callback, cbdata):
    if config.DEBUG_ARRAY_OPT == 1:  # pragma: no cover
        print("visiting join vars for:", join_node)
        print("cbdata: ", sorted(cbdata.items()))

    # left
    for col_name in list(join_node.left_vars.keys()):
        join_node.left_vars[col_name] = visit_vars_inner(join_node.left_vars[col_name], callback, cbdata)
    # right
    for col_name in list(join_node.right_vars.keys()):
        join_node.right_vars[col_name] = visit_vars_inner(join_node.right_vars[col_name], callback, cbdata)
    # output
    for col_name in list(join_node.df_out_vars.keys()):
        join_node.df_out_vars[col_name] = visit_vars_inner(join_node.df_out_vars[col_name], callback, cbdata)

# add call to visit Join variable
ir_utils.visit_vars_extensions[Join] = visit_vars_join

def remove_dead_join(join_node, lives, arg_aliases, alias_map, typemap):
    # if an output column is dead, the related input column is not needed
    # anymore in the join
    dead_cols = []
    left_key_dead = False
    right_key_dead = False
    # TODO: remove output of dead keys

    for col_name, col_var in join_node.df_out_vars.items():
        if col_var.name not in lives:
            if col_name == join_node.left_key:
                left_key_dead = True
            elif col_name == join_node.right_key:
                right_key_dead = True
            else:
                dead_cols.append(col_name)

    for cname in dead_cols:
        assert cname in join_node.left_vars or cname in join_node.right_vars
        join_node.left_vars.pop(cname, None)
        join_node.right_vars.pop(cname, None)
        join_node.df_out_vars.pop(cname)

    # remove empty join node
    if len(join_node.df_out_vars) == 0:
        return None

    return join_node

ir_utils.remove_dead_extensions[Join] = remove_dead_join

def join_usedefs(join_node, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # input columns are used
    use_set.update({v.name for v in join_node.left_vars.values()})
    use_set.update({v.name for v in join_node.right_vars.values()})

    # output columns are defined
    def_set.update({v.name for v in join_node.df_out_vars.values()})

    return numba.analysis._use_defs_result(usemap=use_set, defmap=def_set)

numba.analysis.ir_extension_usedefs[Join] = join_usedefs

def get_copies_join(join_node, typemap):
    # join doesn't generate copies, it just kills the output columns
    kill_set = set(v.name for v in join_node.df_out_vars.values())
    return set(), kill_set

ir_utils.copy_propagate_extensions[Join] = get_copies_join

def apply_copies_join(join_node, var_dict, name_var_table, ext_func, ext_data,
                        typemap, calltypes, save_copies):
    """apply copy propagate in join node"""

    # left
    for col_name in list(join_node.left_vars.keys()):
        join_node.left_vars[col_name] = replace_vars_inner(join_node.left_vars[col_name], var_dict)
    # right
    for col_name in list(join_node.right_vars.keys()):
        join_node.right_vars[col_name] = replace_vars_inner(join_node.right_vars[col_name], var_dict)
    # output
    for col_name in list(join_node.df_out_vars.keys()):
        join_node.df_out_vars[col_name] = replace_vars_inner(join_node.df_out_vars[col_name], var_dict)

    return

ir_utils.apply_copy_propagate_extensions[Join] = apply_copies_join

def join_distributed_run(join_node, typemap, calltypes, typingctx):
    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    loc = join_node.loc
    # get column variables
    left_key_var = join_node.left_vars[join_node.left_key]
    right_key_var = join_node.right_vars[join_node.right_key]
    left_other_col_vars = [v for (n, v) in sorted(join_node.left_vars.items())
                                            if n != join_node.left_key]
    right_other_col_vars = [v for (n, v) in sorted(join_node.right_vars.items())
                                            if n != join_node.right_key]
    # get column types
    left_other_col_typ = [typemap[v.name] for v in left_other_col_vars]
    right_other_col_typ = [typemap[v.name] for v in right_other_col_vars]
    arg_typs = tuple([typemap[left_key_var.name], typemap[right_key_var.name]]
                                + left_other_col_typ + right_other_col_typ)
    # arg names of non-key columns
    left_other_names = ["t1_c"+str(i) for i in range(len(left_other_col_vars))]
    right_other_names = ["t2_c"+str(i) for i in range(len(right_other_col_vars))]
    func_text = """def f(t1_key, t2_key,{}{}{}):
    (t1_send_counts, t1_recv_counts, t1_send_disp, t1_recv_disp,
        t1_recv_size) = hpat.hiframes_join.get_sendrecv_counts(t1_key)
    (t2_send_counts, t2_recv_counts, t2_send_disp, t2_recv_disp,
        t2_recv_size) = hpat.hiframes_join.get_sendrecv_counts(t2_key)
    #hpat.cprint(t1_recv_size, t2_recv_size)\n""".format(",".join(left_other_names),
                ("," if len(left_other_names) != 0 else ""),
                ",".join(right_other_names))
    # prepare for shuffle
    left_arg_names = ['t1_key'] + left_other_names
    right_arg_names = ['t2_key'] + right_other_names
    # allocate send/recv buffers
    for a in left_arg_names:
        func_text += "    send_{} = np.empty_like({})\n".format(a, a)
        func_text += "    recv_{} = np.empty(t1_recv_size, {}.dtype)\n".format(a, a)

    for a in right_arg_names:
        func_text += "    send_{} = np.empty_like({})\n".format(a, a)
        func_text += "    recv_{} = np.empty(t2_recv_size, {}.dtype)\n".format(a, a)
    left_send_names = ",".join(["send_"+v for v in left_arg_names])
    left_recv_names = ",".join(["recv_"+v for v in left_arg_names])
    func_text += ("    hpat.hiframes_join.shuffle_data(t1_send_counts,"
            + " t1_recv_counts, t1_send_disp, t1_recv_disp, {},{},{})\n".format(
                ",".join(left_arg_names), left_send_names, left_recv_names))
    right_send_names = ",".join(["send_"+v for v in right_arg_names])
    right_recv_names = ",".join(["recv_"+v for v in right_arg_names])
    func_text += ("    hpat.hiframes_join.shuffle_data(t2_send_counts,"
            + " t2_recv_counts, t2_send_disp, t2_recv_disp, {},{},{})\n".format(
                ",".join(right_arg_names), right_send_names, right_recv_names))
    func_text += "    hpat.hiframes_join.sort({})\n".format(left_recv_names)
    func_text += "    hpat.hiframes_join.sort({})\n".format(right_recv_names)

    # align output variables for local merge
    # add keys first (TODO: remove dead keys)
    merge_out = [join_node.df_out_vars[join_node.left_key]]
    merge_out += [join_node.df_out_vars[n] for (n, v) in sorted(join_node.left_vars.items())
                                            if n != join_node.left_key]
    merge_out.append(join_node.df_out_vars[join_node.right_key])
    merge_out += [join_node.df_out_vars[n] for (n, v) in sorted(join_node.right_vars.items())
                                            if n != join_node.right_key]
    out_names = ["t3_c"+str(i) for i in range(len(merge_out))]

    func_text += "    {} = hpat.hiframes_join.local_merge({}, {}, {})\n".format(",".join(out_names), len(left_arg_names), left_recv_names, right_recv_names)

    #func_text += "    print(recv_t2_c0)\n"
    #func_text += "    hpat.cprint(recv_t2_c0[0])"

    #delete_buffers((t1_send_counts, t1_recv_counts, t1_send_disp, t1_recv_disp))
    #delete_buffers((t2_send_counts, t2_recv_counts, t2_send_disp, t2_recv_disp))

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    f = loc_vars['f']

    f_block = compile_to_numba_ir(f,
            {'hpat': hpat, 'np': np}, typingctx, arg_typs,
            typemap, calltypes).blocks.popitem()[1]
    replace_arg_nodes(f_block, [left_key_var, right_key_var]
                                + left_other_col_vars + right_other_col_vars)

    nodes = f_block.body[:-3]
    # XXX: create dummy output arrays to allow testing for now
    from numba.ir_utils import mk_alloc
    for _, col_var in join_node.df_out_vars.items():
        nodes += mk_alloc(typemap, calltypes, col_var, (0,), typemap[col_var.name].dtype, col_var.scope, col_var.loc)
    return nodes

distributed.distributed_run_extensions[Join] = join_distributed_run


from numba.typing.templates import (signature, AbstractTemplate, infer_global, infer)
from numba.extending import (register_model, models, lower_builtin)
from numba import cgutils

# a native buffer pointer managed explicity (e.g. deleted manually)
class CBufferType(types.Opaque):
    def __init__(self):
        super(CBufferType, self).__init__(name='CBufferType')

c_buffer_type = CBufferType()

register_model(CBufferType)(models.OpaqueModel)


def get_sendrecv_counts():
    return 0

def shuffle_data():
    return 0

def sort():
    return 0

def local_merge():
    return 0

@infer_global(get_sendrecv_counts)
class SendRecvCountTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        out_typ = types.Tuple([c_buffer_type, c_buffer_type, c_buffer_type,
                                                    c_buffer_type, types.intp])
        return signature(out_typ, *args)

@infer_global(shuffle_data)
class ShuffleTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.int32, *args)

@infer_global(sort)
class ShuffleTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.int32, *args)

@infer_global(local_merge)
class LocalMergeTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        out_typ = types.Tuple(args[1:])
        return signature(out_typ, *args)

LocalMergeTyper.support_literals = True

from llvmlite import ir as lir
import llvmlite.binding as ll
from numba.targets.arrayobj import make_array
from hpat.distributed_lower import _h5_typ_table
import chiframes
ll.add_symbol('get_join_sendrecv_counts', chiframes.get_join_sendrecv_counts)
ll.add_symbol('timsort', chiframes.timsort)
import hdist
ll.add_symbol('c_alltoallv', hdist.c_alltoallv)

@lower_builtin(get_sendrecv_counts, types.Array)
def lower_get_sendrecv_counts(context, builder, sig, args):
    # prepare buffer args
    pointer_to_cbuffer_typ = lir.IntType(8).as_pointer().as_pointer()
    send_counts = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
    recv_counts = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
    send_disp = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
    recv_disp = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())

    # prepare key array args
    key_arr = make_array(sig.args[0])(context, builder, args[0])
    # XXX: assuming key arr is 1D
    assert key_arr.shape.type.count == 1
    arr_len = builder.extract_value(key_arr.shape, 0)
    # TODO: extend to other key types
    assert sig.args[0].dtype == types.intp
    key_typ_enum = _h5_typ_table[sig.args[0].dtype]
    key_typ_arg = builder.load(cgutils.alloca_once_value(builder,
                                lir.Constant(lir.IntType(32), key_typ_enum)))
    key_arr_data = builder.bitcast(key_arr.data, lir.IntType(8).as_pointer())

    call_args = [send_counts, recv_counts, send_disp, recv_disp, arr_len,
                                                    key_typ_arg, key_arr_data]

    fnty = lir.FunctionType(lir.IntType(64), [pointer_to_cbuffer_typ] * 4
            + [lir.IntType(64), lir.IntType(32), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty,
                                                name="get_join_sendrecv_counts")
    total_size = builder.call(fn, call_args)
    items = [builder.load(send_counts), builder.load(recv_counts),
        builder.load(send_disp), builder.load(recv_disp), total_size]
    out_tuple_typ = types.Tuple([c_buffer_type, c_buffer_type, c_buffer_type,
                                                c_buffer_type, types.intp])
    return context.make_tuple(builder, out_tuple_typ, items)

@lower_builtin(shuffle_data, c_buffer_type, c_buffer_type, c_buffer_type,
                        c_buffer_type, types.VarArg(types.Any))
def lower_shuffle(context, builder, sig, args):
    # assuming there are 4 buffer arguments, column vars, send arrs, recv arrs
    assert (len(args) - 4) % 3 == 0
    num_cols = (len(args) - 4) // 3
    send_counts, recv_counts, send_disp, recv_disp = args[:4]
    col_names = ["c"+str(i) for i in range(num_cols)]
    send_names = ["send_c"+str(i) for i in range(num_cols)]
    # create send buffer building function
    func_text = "def f(send_disp, {}, {}):\n".format(",".join(col_names), ",".join(send_names))
    func_text += "    n_pes = hpat.distributed_api.get_size()\n"
    func_text += "    tmp_count = np.zeros(n_pes, dtype=np.int64)\n"
    func_text += "    for i in range(len(c0)):\n"
    func_text += "        node_id = c0[i] % n_pes\n"
    func_text += "        ind = send_disp[node_id] + tmp_count[node_id]\n"
    func_text += "        send_c0[ind] = c0[i]\n"
    for i in range(1, num_cols):
        func_text += "        send_c{}[ind] = c{}[i]\n".format(i, i)
    func_text += "        tmp_count[node_id] = tmp_count[node_id] + 1\n"
    #func_text += "        hpat.cprint(node_id)\n"

    loc_vars = {}
    exec(func_text, {'hpat': hpat, 'np': np}, loc_vars)
    f = loc_vars['f']
    f_args = [send_disp]+args[4:4+2*num_cols]
    f_sig = signature(types.void, *((c_buffer_type,)+sig.args[4:4+2*num_cols]))
    context.compile_internal(builder, f, f_sig, f_args)
    # generate alltoallv calls
    for i in range(0, num_cols):
        arr_typ = sig.args[4+i]
        send_arg = args[4+num_cols+i]
        recv_arg = args[4+2*num_cols+i]
        gen_alltoallv(context, builder, arr_typ, send_arg, recv_arg, send_counts,
                                            recv_counts, send_disp, recv_disp)

    return lir.Constant(lir.IntType(32), 0)

def gen_alltoallv(context, builder, arr_typ, send_arg, recv_arg, send_counts,
                                            recv_counts, send_disp, recv_disp):
    #
    typ_enum = _h5_typ_table[arr_typ.dtype]
    typ_arg = builder.load(cgutils.alloca_once_value(builder,
                                lir.Constant(lir.IntType(32), typ_enum)))
    send_data = make_array(arr_typ)(context, builder, send_arg).data
    recv_data = make_array(arr_typ)(context, builder, recv_arg).data
    send_data = builder.bitcast(send_data, lir.IntType(8).as_pointer())
    recv_data = builder.bitcast(recv_data, lir.IntType(8).as_pointer())

    call_args = [send_data, recv_data, send_counts, recv_counts, send_disp, recv_disp, typ_arg]

    fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()]*6 + [lir.IntType(32)])
    fn = builder.module.get_or_insert_function(fnty, name="c_alltoallv")
    builder.call(fn, call_args)

@lower_builtin(sort, types.VarArg(types.Any))
def lower_sort(context, builder, sig, args):
    #
    key_arr = make_array(sig.args[0])(context, builder, args[0])
    key_data = builder.bitcast(key_arr.data, lir.IntType(8).as_pointer())
    # XXX: assuming key arr is 1D
    assert key_arr.shape.type.count == 1
    arr_len = builder.extract_value(key_arr.shape, 0)
    num_other_cols = len(args) - 1
    # build array of other column arrays arg
    other_arrs = cgutils.alloca_once(builder, lir.IntType(8).as_pointer(), num_other_cols)
    for i in range(num_other_cols):
        ptr = cgutils.gep_inbounds(builder, other_arrs, i)
        arr = make_array(sig.args[i+1])(context, builder, args[i+1])
        arr_data = builder.bitcast(arr.data, lir.IntType(8).as_pointer())
        builder.store(arr_data, ptr)

    call_args = [key_data, arr_len, other_arrs,
                                lir.Constant(lir.IntType(64), num_other_cols)]
    fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer(),
        lir.IntType(64), lir.IntType(8).as_pointer().as_pointer(), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="timsort")
    builder.call(fn, call_args)
    return lir.Constant(lir.IntType(32), 0)

@lower_builtin(local_merge, types.Const, types.VarArg(types.Any))
def lower_local_merge(context, builder, sig, args):
    #
    num_left_cols = sig.args[0].value
    num_right_cols = len(args) - num_left_cols - 1
    left_other_names = ["t1_c"+str(i) for i in range(num_left_cols-1)]
    right_other_names = ["t2_c"+str(i) for i in range(num_right_cols-1)]

    # create merge function
    func_text = "def f(left_key, right_key, {}{} {}):\n".format(
                            ",".join(left_other_names),
                            ("," if len(left_other_names) != 0 else ""),
                            ",".join(right_other_names))
    # initialize output arrays with a heuristic starting size
    func_text += "    init_size = 100 + min(len(left_key), len(right_key)) // 10 + 1\n"
    func_text += "    out_left_key = np.empty(init_size, left_key.dtype)\n"
    func_text += "    out_right_key = np.empty(init_size, right_key.dtype)\n"
    for v in (left_other_names + right_other_names):
        func_text += "    out_{} = np.empty(init_size, {}.dtype)\n".format(v, v)
    func_text += "    out_ind = 0\n"
    func_text += "    left_ind = 0\n"
    func_text += "    right_ind = 0\n"
    func_text += "    while left_ind < len(left_key) and right_ind < len(right_key):\n"
    func_text += "        if left_key[left_ind] == right_key[right_ind]:\n"
    func_text += _set_merge_output("            ", left_other_names, right_other_names, "left_ind", "right_ind")
    func_text += "            left_run = left_ind + 1\n"
    func_text += "            while left_run < len(left_key) and left_key[left_run] == right_key[right_ind]:\n"
    func_text += _set_merge_output("                ", left_other_names, right_other_names, "left_run", "right_ind")
    func_text += "                left_run += 1\n"
    func_text += "            right_run = right_ind + 1\n"
    func_text += "            while right_run < len(right_key) and right_key[right_run] == left_key[left_ind]:\n"
    func_text += _set_merge_output("                ", left_other_names, right_other_names, "left_ind", "right_run")
    func_text += "                right_run += 1\n"
    func_text += "            left_ind += 1\n"
    func_text += "            right_ind += 1\n"
    func_text += "        elif left_key[left_ind] < right_key[right_ind]:\n"
    func_text += "            left_ind += 1\n"
    func_text += "        else:\n"
    func_text += "            right_ind += 1\n"
    # return output
    out_left_other_names =  ["out_"+v for v in left_other_names]
    out_right_other_names =  ["out_"+v for v in right_other_names]
    func_text += "    return out_left_key,{}{} out_right_key,{}\n".format(
                                ",".join(out_left_other_names),
                                ("," if len(out_left_other_names) != 0 else ""),
                                ",".join(out_right_other_names))

    loc_vars = {}
    exec(func_text, {'hpat': hpat, 'np': np}, loc_vars)
    f = loc_vars['f']
    # args: left key, right key, left other cols, right other cols
    f_args = [args[1], args[num_left_cols+1]] + args[2:num_left_cols+1] + args[num_left_cols+2:]
    f_sig = signature(sig.return_type, *sig.args[1:])
    return context.compile_internal(builder, f, f_sig, f_args)

def _set_merge_output(indent, left_other_names, right_other_names, left_ind, right_ind):
    func_text = indent + _set_expand_arr("out_left_key", "left_key[{}]".format(left_ind), "out_ind")
    func_text += indent + _set_expand_arr("out_right_key", "right_key[{}]".format(right_ind), "out_ind")
    for v in left_other_names:
        func_text += indent + _set_expand_arr("out_"+v, v+"[{}]".format(left_ind), "out_ind")
    for v in right_other_names:
        func_text += indent + _set_expand_arr("out_"+v, v+"[{}]".format(right_ind), "out_ind")
    func_text += indent + "out_ind += 1\n"
    return func_text

def _set_expand_arr(out, val, ind):
    # XXX expand array
    return "{}[{}] = {}\n".format(out, ind, val)

@infer
class GetItemCBuf(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        c_buff, idx = args
        if isinstance(c_buff, CBufferType):
            if isinstance(idx, types.Integer):
                return signature(types.int32, c_buff, idx)


@lower_builtin('getitem', c_buffer_type, types.intp)
def c_buffer_type_getitem(context, builder, sig, args):
    base_ptr = builder.bitcast(args[0], lir.IntType(32).as_pointer())
    return builder.load(builder.gep(base_ptr, [args[1]], inbounds=True))

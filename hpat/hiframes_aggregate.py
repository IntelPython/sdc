from __future__ import print_function, division, absolute_import

from collections import namedtuple
from functools import reduce
import numpy as np
import numba
from numba import typeinfer, ir, ir_utils, config, types, compiler
from numba.ir_utils import (visit_vars_inner, replace_vars_inner,
                            compile_to_numba_ir, replace_arg_nodes,
                            replace_vars_stmt, find_callname, guard,
                            mk_unique_var)
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
import hpat
from hpat.utils import is_call, is_var_assign, is_assign
from hpat import distributed, distributed_analysis
from hpat.distributed_analysis import Distribution
from hpat.str_ext import string_type
from hpat.str_arr_ext import string_array_type

AggFuncStruct = namedtuple('AggFuncStruct', ['vars', 'var_typs', 'init',
                            'input_var', 'update', 'combine', 'eval', 'pm'])


class Aggregate(ir.Stmt):
    def __init__(self, df_out, df_in, key_name, df_out_vars, df_in_vars,
                                             key_arr, agg_func, out_typs, loc):
        # name of output dataframe (just for printing purposes)
        self.df_out = df_out
        # name of input dataframe (just for printing purposes)
        self.df_in = df_in
        # key name (for printing)
        self.key_name = key_name

        self.df_out_vars = df_out_vars
        self.df_in_vars = df_in_vars
        self.key_arr = key_arr

        self.agg_func = agg_func
        self.out_typs = out_typs

        self.loc = loc

    def __repr__(self):  # pragma: no cover
        out_cols = ""
        for (c, v) in self.df_out_vars.items():
            out_cols += "'{}':{}, ".format(c, v.name)
        df_out_str = "{}{{{}}}".format(self.df_out, out_cols)
        in_cols = ""
        for (c, v) in self.df_in_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_in_str = "{}{{{}}}".format(self.df_in, in_cols)
        return "aggregate: {} = {} [key: {}:{}] ".format(df_out_str, df_in_str,
                                            self.key_name, self.key_arr.name)


def aggregate_typeinfer(aggregate_node, typeinferer):
    for out_name, out_var in aggregate_node.df_out_vars.items():
        typ = aggregate_node.out_typs[out_name]
        # TODO: are there other non-numpy array types?
        if typ == string_type:
            arr_type = string_array_type
        else:
            arr_type = types.Array(typ, 1, 'C')

        typeinferer.lock_type(out_var.name, arr_type, loc=aggregate_node.loc)

    return

typeinfer.typeinfer_extensions[Aggregate] = aggregate_typeinfer


def aggregate_usedefs(aggregate_node, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # key array and input columns are used
    use_set.add(aggregate_node.key_arr.name)
    use_set.update({v.name for v in aggregate_node.df_in_vars.values()})

    # output columns are defined
    def_set.update({v.name for v in aggregate_node.df_out_vars.values()})

    return numba.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.analysis.ir_extension_usedefs[Aggregate] = aggregate_usedefs


def remove_dead_aggregate(aggregate_node, lives, arg_aliases, alias_map, typemap):
    #
    dead_cols = []

    for col_name, col_var in aggregate_node.df_out_vars.items():
        if col_var.name not in lives:
            dead_cols.append(col_name)

    for cname in dead_cols:
        aggregate_node.df_in_vars.pop(cname)
        aggregate_node.df_out_vars.pop(cname)

    # TODO: test agg remove
    # remove empty aggregate node
    if len(aggregate_node.df_in_vars) == 0:
        return None

    return aggregate_node


ir_utils.remove_dead_extensions[Aggregate] = remove_dead_aggregate

def get_copies_aggregate(aggregate_node, typemap):
    # aggregate doesn't generate copies, it just kills the output columns
    kill_set = set(v.name for v in aggregate_node.df_out_vars.values())
    return set(), kill_set


ir_utils.copy_propagate_extensions[Aggregate] = get_copies_aggregate


def apply_copies_aggregate(aggregate_node, var_dict, name_var_table,
                        typemap, calltypes, save_copies):
    """apply copy propagate in aggregate node"""
    aggregate_node.key_arr = replace_vars_inner(aggregate_node.key_arr,
                                                                     var_dict)

    for col_name in list(aggregate_node.df_in_vars.keys()):
        aggregate_node.df_in_vars[col_name] = replace_vars_inner(
            aggregate_node.df_in_vars[col_name], var_dict)
    for col_name in list(aggregate_node.df_out_vars.keys()):
        aggregate_node.df_out_vars[col_name] = replace_vars_inner(
            aggregate_node.df_out_vars[col_name], var_dict)

    return


ir_utils.apply_copy_propagate_extensions[Aggregate] = apply_copies_aggregate


def visit_vars_aggregate(aggregate_node, callback, cbdata):
    if config.DEBUG_ARRAY_OPT == 1:  # pragma: no cover
        print("visiting aggregate vars for:", aggregate_node)
        print("cbdata: ", sorted(cbdata.items()))

    aggregate_node.key_arr = visit_vars_inner(
        aggregate_node.key_arr, callback, cbdata)

    for col_name in list(aggregate_node.df_in_vars.keys()):
        aggregate_node.df_in_vars[col_name] = visit_vars_inner(
            aggregate_node.df_in_vars[col_name], callback, cbdata)
    for col_name in list(aggregate_node.df_out_vars.keys()):
        aggregate_node.df_out_vars[col_name] = visit_vars_inner(
            aggregate_node.df_out_vars[col_name], callback, cbdata)


# add call to visit aggregate variable
ir_utils.visit_vars_extensions[Aggregate] = visit_vars_aggregate


def aggregate_array_analysis(aggregate_node, equiv_set, typemap,
                                                            array_analysis):
    # empty aggregate nodes should be deleted in remove dead
    assert len(aggregate_node.df_in_vars) > 0, ("empty aggregate in array"
                                                                   "analysis")

    # arrays of input df have same size in first dimension as key array
    col_shape = equiv_set.get_shape(aggregate_node.key_arr)
    all_shapes = [col_shape[0]]
    for _, col_var in aggregate_node.df_in_vars.items():
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
    post = []
    all_shapes = []
    for _, col_var in aggregate_node.df_out_vars.items():
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


numba.array_analysis.array_analysis_extensions[Aggregate] = aggregate_array_analysis


def aggregate_distributed_analysis(aggregate_node, array_dists):
    # input columns have same distribution
    in_dist = Distribution.OneD
    for _, col_var in aggregate_node.df_in_vars.items():
        in_dist = Distribution(
            min(in_dist.value, array_dists[col_var.name].value))

    # key arr
    in_dist = Distribution(
        min(in_dist.value, array_dists[aggregate_node.key_arr.name].value))
    for _, col_var in aggregate_node.df_in_vars.items():
        array_dists[col_var.name] = in_dist
    array_dists[aggregate_node.key_arr.name] = in_dist

    # output columns have same distribution
    out_dist = Distribution.OneD_Var
    for _, col_var in aggregate_node.df_out_vars.items():
        # output dist might not be assigned yet
        if col_var.name in array_dists:
            out_dist = Distribution(
                min(out_dist.value, array_dists[col_var.name].value))

    # out dist should meet input dist (e.g. REP in causes REP out)
    out_dist = Distribution(min(out_dist.value, in_dist.value))
    for _, col_var in aggregate_node.df_out_vars.items():
        array_dists[col_var.name] = out_dist

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        array_dists[aggregate_node.key_arr.name] = out_dist
        for _, col_var in aggregate_node.df_in_vars.items():
            array_dists[col_var.name] = out_dist

    return


distributed_analysis.distributed_analysis_extensions[Aggregate] = aggregate_distributed_analysis

def __update_redvars():
    pass

@infer_global(__update_redvars)
class UpdateDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.void, *args)

def __eval_res():
    pass

@infer_global(__eval_res)
class EvalDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        # takse the output array as first argument to know the output dtype
        return signature(args[0].dtype, *args)

def agg_distributed_run(agg_node, array_dists, typemap, calltypes, typingctx, targetctx):
    parallel = True
    for v in (list(agg_node.df_in_vars.values())
              + list(agg_node.df_out_vars.values()) + [agg_node.key_arr]):
        if (array_dists[v.name] != distributed.Distribution.OneD
                and array_dists[v.name] != distributed.Distribution.OneD_Var):
            parallel = False
        # TODO: check supported types
        # if (typemap[v.name] != types.Array(types.intp, 1, 'C')
        #         and typemap[v.name] != types.Array(types.float64, 1, 'C')):
        #     raise ValueError(
        #         "Only int64 and float64 columns are currently supported in aggregate")
        # if (typemap[left_key_var.name] != types.Array(types.intp, 1, 'C')
        #     or typemap[right_key_var.name] != types.Array(types.intp, 1, 'C')):
        # raise ValueError("Only int64 keys are currently supported in aggregate")

    # TODO: rebalance if output distributions are 1D instead of 1D_Var

    # TODO: handle key column being part of output

    key_typ = typemap[agg_node.key_arr.name]
    # get column variables
    in_col_vars = [v for (n, v) in sorted(agg_node.df_in_vars.items())]
    out_col_vars = [v for (n, v) in sorted(agg_node.df_out_vars.items())]
    # get column types
    in_col_typs = [typemap[v.name] for v in in_col_vars]
    out_col_typs = [typemap[v.name] for v in out_col_vars]
    arg_typs = tuple([key_typ] + in_col_typs)
    # arg names
    in_names = ["in_c{}".format(i) for i in range(len(in_col_vars))]
    out_names = ["out_c{}".format(i) for i in range(len(out_col_vars))]
    # key and arg names
    col_names = ['key_arr'] + in_names

    agg_func_struct = get_agg_func_struct(agg_node.agg_func, in_col_typs[0], typingctx)
    typemap.update(agg_func_struct.pm.typemap)
    calltypes.update(agg_func_struct.pm.calltypes)

    func_text = "def f(key_arr, {}):\n".format(", ".join(in_names))

    if parallel:
        # get send/recv counts
        func_text += "    send_counts, recv_counts = agg_send_recv_counts(key_arr)\n"
        func_text += "    n_uniq_keys = send_counts.sum()\n"
        func_text += "    recv_size = recv_counts.sum()\n"
        # func_text += "    hpat.cprint(n_uniq_keys, recv_size)\n"

        # prepare for shuffle
        # allocate send/recv buffers
        for a in col_names:
            func_text += "    send_{} = np.empty(n_uniq_keys, {}.dtype)\n".format(
                a, a)
            func_text += "    recv_{} = np.empty(recv_size, {}.dtype)\n".format(
                a, a)

    else:
        func_text += "    n_uniq_keys = len(set(key_arr))\n"
        # allocate output
        for i, col in enumerate(agg_node.df_in_vars.keys()):
            func_text += "    out_c{} = np.empty(n_uniq_keys, np.{})\n".format(
                                                    i, agg_node.out_typs[col])

    func_text += "    __agg_func(n_uniq_keys, key_arr, {}, {})\n".format(", ".join(out_names), ", ".join(in_names))
    func_text += "    c = out_c0\n"

    # print(func_text)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    f = loc_vars['f']

    agg_impl = gen_agg_func(agg_func_struct, key_typ, in_col_typs,
                        out_col_typs, typingctx, typemap, calltypes, targetctx)

    f_block = compile_to_numba_ir(f,
                                  {'hpat': hpat, 'np': np,
                                  'agg_send_recv_counts': agg_send_recv_counts,
                                  '__agg_func': agg_impl},
                                  typingctx, arg_typs,
                                  typemap, calltypes).blocks.popitem()[1]

    replace_arg_nodes(f_block, [agg_node.key_arr] + in_col_vars)
    nodes = f_block.body[:-3]
    nodes[-1].target = list(agg_node.df_out_vars.values())[0]
    return nodes


distributed.distributed_run_extensions[Aggregate] = agg_distributed_run




def gen_agg_func(agg_func_struct, key_typ, in_typs, out_typs, typingctx, typemap, calltypes, targetctx):

    arg_typs = tuple([types.intp, key_typ] + out_typs + in_typs)
    num_cols = len(in_typs)
    assert len(out_typs) == num_cols

    # arg names
    in_names = ["in_c{}".format(i) for i in range(num_cols)]
    out_names = ["out_c{}".format(i) for i in range(num_cols)]

    func_text = "def f(n_uniq_keys, key_arr, {}, {}):\n".format(", ".join(out_names), ", ".join(in_names))

    # allocate reduction var arrays
    for i, typ in enumerate(agg_func_struct.var_typs):
        func_text += "    _init_val_{} = np.{}(0)\n".format(i, typ)
        func_text += "    redvar_{}_arr = np.full(n_uniq_keys, _init_val_{}, np.{})\n".format(
            i, i, typ)

    # find write location
    # TODO: non-int dict
    func_text += "    key_write_map = hpat.DictIntInt()\n"
    func_text += "    curr_write_ind = 0\n"
    func_text += "    for i in range(len(key_arr)):\n"
    func_text += "      k = key_arr[i]\n"
    func_text += "      if k not in key_write_map:\n"
    func_text += "        key_write_map[k] = curr_write_ind\n"
    func_text += "        w_ind = curr_write_ind\n"
    func_text += "        curr_write_ind += 1\n"
    func_text += "      else:\n"
    func_text += "        w_ind = key_write_map[k]\n"
    # update reduce vars with input
    redvar_arrnames = ", ".join(["redvar_{}_arr".format(i)
                                for i in range(len(agg_func_struct.vars))])
    func_text += "      __update_redvars(w_ind, {}[i], {})\n".format(
        in_names[0], redvar_arrnames)
    # get final output from reduce varis
    redvar_arrnames = ", ".join(["redvar_{}_arr[j]".format(i)
                                for i in range(len(agg_func_struct.vars))])
    func_text += "    for j in range(n_uniq_keys):\n"
    func_text += "      out_c0[j] = __eval_res(out_c0, {})\n".format(
                                                            redvar_arrnames)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    f = loc_vars['f']
    #
    # print(func_text)

    f_ir = compile_to_numba_ir(f,
                                  {'hpat': hpat, 'np': np,
                                  '__update_redvars': __update_redvars,
                                  '__eval_res': __eval_res},
                                  typingctx, arg_typs,
                                  typemap, calltypes)

    f_ir._definitions = numba.ir_utils.build_definitions(f_ir.blocks)
    topo_order = numba.ir_utils.find_topo_order(f_ir.blocks)
    first_block = f_ir.blocks[topo_order[0]]

    # find reduce variables from names and store in the same order
    reduce_vars = [0] * len(agg_func_struct.vars)
    for node in agg_func_struct.init:
        if isinstance(node, ir.Assign) and node.target.name in agg_func_struct.vars:
            var_ind = agg_func_struct.vars.index(node.target.name)
            reduce_vars[var_ind] = node.target
    assert 0 not in reduce_vars

    # add initialization code to first block
    # make sure arg nodes are in the beginning
    arg_nodes = []
    for i in range(len(arg_typs)):
        arg_nodes.append(first_block.body[i])
    first_block.body = arg_nodes + agg_func_struct.init + first_block.body[len(arg_nodes):]

    # replace init val sentinels
    for l in topo_order:
        block = f_ir.blocks[l]
        for i, stmt in enumerate(block.body):
            if isinstance(stmt, ir.Assign) and stmt.target.name.startswith("_init_val_"):
                first_dot = stmt.target.name.index(".")
                var_ind = int(stmt.target.name[len("_init_val_"):first_dot])
                stmt.value = reduce_vars[var_ind]
            if is_call(stmt) and (guard(find_callname, f_ir, stmt.value)
                    == ('__update_redvars', 'hpat.hiframes_aggregate')):
                write_ind_var = stmt.value.args[0]
                column_val = stmt.value.args[1]
                red_arrs = stmt.value.args[2:]
                # replace_dict = {v:column_val for v in agg_func_struct.vars}
                update_nodes = []
                red_vals = []
                # generate getitem nodes for reduction arrays
                for k, rarr in enumerate(red_arrs):
                    val_typ = typemap[rarr.name].dtype
                    getitem_call = ir.Expr.getitem(rarr, write_ind_var, rarr.loc)
                    red_val = ir.Var(rarr.scope, mk_unique_var("#red_val{}".format(k)), rarr.loc)
                    red_vals.append(red_val)
                    typemap[red_val.name] = val_typ
                    calltypes[getitem_call] = signature(val_typ, typemap[rarr.name],
                                                            typemap[write_ind_var.name])
                    update_nodes.append(ir.Assign(getitem_call, red_val, rarr.loc))
                replace_dict = {}
                for k, v in enumerate(agg_func_struct.vars):
                    replace_dict[v] = red_vals[k]

                for inst in agg_func_struct.update:
                    # replace agg#input with column array value
                    if is_var_assign(inst) and inst.value.name == agg_func_struct.input_var.name:
                        inst.value = column_val
                    # replace red_var assignment with red_arr[w_ind] setitem
                    if is_assign(inst) and inst.target.name in agg_func_struct.vars:
                        var_ind = agg_func_struct.vars.index(inst.target.name)
                        arr = red_arrs[var_ind]
                        setitem_node = ir.SetItem(arr, write_ind_var, inst.value, inst.loc)
                        calltypes[setitem_node] = signature(
                        types.none, typemap[arr.name], typemap[write_ind_var.name], typemap[inst.value.name])
                        update_nodes.append(setitem_node)
                        continue  # remove call

                    # replace reduction vars with column input value
                    replace_vars_stmt(inst, replace_dict)
                    update_nodes.append(inst)

                # add new update nodes
                # assuming update sentinel is right before jump
                # XXX can modify since iterator is terminated
                assert i == len(block.body) - 2
                jump_node = block.body.pop()
                block.body.pop()  # remove update call
                block.body += update_nodes
                block.body.append(jump_node)
                break

            if is_call(stmt) and (guard(find_callname, f_ir, stmt.value)
                    == ('__eval_res', 'hpat.hiframes_aggregate')):
                red_vals = stmt.value.args[1:]
                replace_dict = {}
                for k, v in enumerate(agg_func_struct.vars):
                    replace_dict[v] = red_vals[k]
                for inst in agg_func_struct.eval:
                    replace_vars_stmt(inst, replace_dict)
                # add new eval nodes
                # assuming eval sentinel is before setitem and jump
                # XXX can modify since iterator is terminated
                assert i == len(block.body) - 3
                jump_node = block.body.pop()
                setitem_node = block.body.pop()
                agg_func_struct.eval[-1].target = setitem_node.value
                block.body.pop()  # remove update call
                block.body += agg_func_struct.eval
                block.body.append(setitem_node)
                block.body.append(jump_node)
                break

    # compile implementation to binary (Dispatcher)
    agg_impl_func = compiler.compile_ir(
            typingctx,
            targetctx,
            f_ir,
            arg_typs,
            types.none,
            compiler.DEFAULT_FLAGS,
            {}
    )

    imp_dis = numba.targets.registry.dispatcher_registry['cpu'](f)
    imp_dis.add_overload(agg_impl_func)
    return imp_dis


@numba.njit
def agg_send_recv_counts(key_arr):
    n_pes = hpat.distributed_api.get_size()
    send_counts = np.zeros(n_pes, np.int32)
    recv_counts = np.empty(n_pes, np.int32)
    # TODO: handle string
    key_set = set()
    for i in range(len(key_arr)):
        k = key_arr[i]
        if k not in key_set:
            key_set.add(k)
            node_id = hash(k) % n_pes
            send_counts[node_id] += 1

    hpat.distributed_api.alltoall(send_counts, recv_counts, 1)
    return send_counts, recv_counts


def compile_to_optimized_ir(func, arg_typs, typingctx):
    # XXX are outside function's globals needed?
    f_ir = numba.ir_utils.get_ir_of_code({}, func.code)
    typemap, return_type, calltypes = compiler.type_inference_stage(
                typingctx, f_ir, arg_typs, None)

    options = numba.targets.cpu.ParallelOptions(True)
    flags = compiler.Flags()
    targetctx = numba.targets.cpu.CPUContext(typingctx)

    DummyPipeline = namedtuple('DummyPipeline',
        ['typingctx', 'targetctx', 'args', 'func_ir', 'typemap', 'return_type',
        'calltypes'])
    pm = DummyPipeline(typingctx, targetctx, None, f_ir, typemap, return_type,
                        calltypes)
    preparfor_pass = numba.parfor.PreParforPass(
            f_ir,
            typemap,
            calltypes, typingctx,
            options
            )
    preparfor_pass.run()
    numba.rewrites.rewrite_registry.apply('after-inference', pm, f_ir)
    parfor_pass = numba.parfor.ParforPass(f_ir, typemap,
    calltypes, return_type, typingctx,
    options, flags)
    parfor_pass.run()
    numba.ir_utils.remove_dels(f_ir.blocks)
    return f_ir, pm

def get_agg_func_struct(agg_func, in_col_typ, typingctx):
    """find initialization, update, combine and final evaluation code of the
    aggregation function. Currently assuming that the function is single block
    and has one parfor.
    """
    f_ir, pm = compile_to_optimized_ir(
        agg_func, tuple([in_col_typ]), typingctx)
    assert len(f_ir.blocks) == 1 and 0 in f_ir.blocks, ("only simple functions"
                                  " with one block supported for aggregation")
    block = f_ir.blocks[0]
    parfor_ind = -1
    for i, stmt in enumerate(block.body):
        if isinstance(stmt, numba.parfor.Parfor):
            assert parfor_ind == -1, "only one parfor for aggregation function"
            parfor_ind = i
    parfor = block.body[parfor_ind]
    numba.ir_utils.remove_dels({0: parfor.init_block})
    redvars, reddict = numba.parfor.get_parfor_reductions(
        parfor, parfor.params, pm.calltypes)
    var_types = [pm.typemap[v] for v in redvars]
    combine_nodes = reduce(lambda a,b: a+b, [r[1] for r in reddict.values()])

    # ignore arg and size/shape nodes for input arr
    assert (isinstance(block.body[0], ir.Assign)
            and isinstance(block.body[0].value, ir.Arg)), "invalid agg func"
    init_nodes = block.body[3:parfor_ind] + parfor.init_block.body

    arr_var = block.body[0].target
    input_var = ir.Var(arr_var.scope, "agg#input", arr_var.loc)
    for bl in parfor.loop_body.values():
        for stmt in bl.body:
            if numba.ir_utils.is_getitem(stmt) and stmt.value.value.name == arr_var.name:
                stmt.value = input_var

    eval_nodes = block.body[parfor_ind+1:-2]  # ignore cast and return
    return AggFuncStruct(redvars, var_types, init_nodes, input_var, bl.body,
                         combine_nodes, eval_nodes, pm)

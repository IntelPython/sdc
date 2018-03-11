import numba
from numba import ir_utils, ir, types, cgutils
from numba.ir_utils import guard, get_definition, find_callname, require
from numba.parfor import wrap_parfor_blocks, unwrap_parfor_blocks
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.targets.imputils import lower_builtin
import collections
import numpy as np
from hpat.str_arr_ext import string_array_type

# sentinel value representing non-constant values


class NotConstant:
    pass


NOT_CONSTANT = NotConstant()


def get_constant(func_ir, var, default=NOT_CONSTANT):
    def_node = guard(get_definition, func_ir, var)
    if def_node is None:
        return default
    if isinstance(def_node, ir.Const):
        return def_node.value
    # call recursively if variable assignment
    if isinstance(def_node, ir.Var):
        return get_constant(func_ir, def_node, default)
    return default


def get_definitions(blocks, definitions=None):
    if definitions is None:
        definitions = collections.defaultdict(list)
    for block in blocks.values():
        update_node_definitions(block.body, definitions)
    return definitions

def update_node_definitions(nodes, definitions):
    for inst in nodes:
        if isinstance(inst, ir.Assign):
            definitions[inst.target.name].append(inst.value)
        if isinstance(inst, numba.parfor.Parfor):
            parfor_blocks = wrap_parfor_blocks(inst)
            get_definitions(parfor_blocks, definitions)
            unwrap_parfor_blocks(inst)

def is_alloc_call(func_var, call_table):
    """
    return trie of func_var represents an array creation call
    """
    assert func_var in call_table
    call_list = call_table[func_var]
    return ((len(call_list) == 2 and call_list[1] == np
             and call_list[0] in ['empty', 'zeros', 'ones', 'full'])
            or call_list == [numba.unsafe.ndarray.empty_inferred])


def cprint(*s):
    print(*s)


@infer_global(cprint)
class CprintInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.none, *args)


typ_to_format = {
    types.int32: 'd',
    types.int64: 'lld',
    types.float32: 'f',
    types.float64: 'lf',
}

from llvmlite import ir as lir
import llvmlite.binding as ll
import hstr_ext
ll.add_symbol('print_str', hstr_ext.print_str)


@lower_builtin(cprint, types.VarArg(types.Any))
def cprint_lower(context, builder, sig, args):
    from hpat.str_ext import string_type

    for i, val in enumerate(args):
        typ = sig.args[i]
        if typ == string_type:
            fnty = lir.FunctionType(
                lir.VoidType(), [lir.IntType(8).as_pointer()])
            fn = builder.module.get_or_insert_function(fnty, name="print_str")
            builder.call(fn, [val])
            cgutils.printf(builder, " ")
            continue
        format_str = typ_to_format[typ]
        cgutils.printf(builder, "%{} ".format(format_str), val)
    cgutils.printf(builder, "\n")
    return context.get_dummy_value()


def print_dist(d):
    from hpat.distributed_analysis import Distribution
    if d == Distribution.REP:
        return "REP"
    if d == Distribution.OneD:
        return "1D_Block"
    if d == Distribution.OneD_Var:
        return "1D_Block_Var"
    if d == Distribution.Thread:
        return "Multi-thread"
    if d == Distribution.TwoD:
        return "2D_Block"


def distribution_report():
    import hpat.distributed
    if hpat.distributed.dist_analysis is None:
        return
    print("Array distributions:")
    for arr, dist in hpat.distributed.dist_analysis.array_dists.items():
        print("   {0:20} {1}".format(arr, print_dist(dist)))
    print("\nParfor distributions:")
    for p, dist in hpat.distributed.dist_analysis.parfor_dists.items():
        print("   {0:<20} {1}".format(p, print_dist(dist)))


def is_whole_slice(typemap, func_ir, var):
    """ return True if var can be determined to be a whole slice """
    require(typemap[var.name] == types.slice2_type)
    call_expr = get_definition(func_ir, var)
    require(isinstance(call_expr, ir.Expr) and call_expr.op == 'call')
    assert len(call_expr.args) == 2
    assert find_callname(func_ir, call_expr) == ('slice', 'builtins')
    arg0_def = get_definition(func_ir, call_expr.args[0])
    arg1_def = get_definition(func_ir, call_expr.args[1])
    require(isinstance(arg0_def, ir.Const) and arg0_def.value == None)
    require(isinstance(arg1_def, ir.Const) and arg1_def.value == None)
    return True

def is_array(typemap, varname):
    return (varname in typemap
            and (is_np_array(typemap, varname)
                or typemap[varname] == string_array_type))

def is_np_array(typemap, varname):
    return (varname in typemap
            and isinstance(typemap[varname], types.Array))

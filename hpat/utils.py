import numba
from numba import ir_utils, ir, types, cgutils
from numba.ir_utils import guard, get_definition, find_callname, require
from numba.parfor import wrap_parfor_blocks, unwrap_parfor_blocks
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.targets.imputils import lower_builtin
from numba.extending import overload
import collections
import numpy as np
from hpat.str_ext import string_type
from hpat.str_arr_ext import string_array_type

# silence Numba error messages for now
# TODO: customize through @hpat.jit
numba.errors.error_extras = {'unsupported_error': '', 'typing': '', 'reportable': '', 'interpreter': '', 'constant_inference': ''}

# sentinel value representing non-constant values
class NotConstant:
    pass


NOT_CONSTANT = NotConstant()

np_alloc_callnames = ('empty', 'zeros', 'ones', 'full')

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
    return true if func_var represents an array creation call
    """
    assert func_var in call_table
    call_list = call_table[func_var]
    return ((len(call_list) == 2 and call_list[1] == np
             and call_list[0] in ['empty', 'zeros', 'ones', 'full'])
            or call_list == [numba.unsafe.ndarray.empty_inferred])

def is_alloc_callname(func_name, mod_name):
    """
    return true if function represents an array creation call
    """
    return isinstance(mod_name, str) and ((mod_name == 'numpy'
        and func_name in np_alloc_callnames)
        or (func_name == 'empty_inferred'
            and mod_name in ('numba.extending', 'numba.unsafe.ndarray')))

def find_build_tuple(func_ir, var):
    """Check if a variable is constructed via build_tuple
    and return the sequence or raise GuardException otherwise.
    """
    # variable or variable name
    require(isinstance(var, (ir.Var, str)))
    var_def = get_definition(func_ir, var)
    require(isinstance(var_def, ir.Expr))
    require(var_def.op == 'build_tuple')
    return var_def.items

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
ll.add_symbol('print_char', hstr_ext.print_char)


@lower_builtin(cprint, types.VarArg(types.Any))
def cprint_lower(context, builder, sig, args):
    from hpat.str_ext import string_type, char_type

    for i, val in enumerate(args):
        typ = sig.args[i]
        if typ == string_type:
            fnty = lir.FunctionType(
                lir.VoidType(), [lir.IntType(8).as_pointer()])
            fn = builder.module.get_or_insert_function(fnty, name="print_str")
            builder.call(fn, [val])
            cgutils.printf(builder, " ")
            continue
        if typ == char_type:
            fnty = lir.FunctionType(
                lir.VoidType(), [lir.IntType(8)])
            fn = builder.module.get_or_insert_function(fnty, name="print_char")
            builder.call(fn, [val])
            cgutils.printf(builder, " ")
            continue
        if isinstance(typ, types.ArrayCTypes):
            cgutils.printf(builder, "%p ", val)
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


def is_whole_slice(typemap, func_ir, var, accept_stride=False):
    """ return True if var can be determined to be a whole slice """
    require(typemap[var.name] == types.slice2_type
            or (accept_stride and typemap[var.name] == types.slice3_type))
    call_expr = get_definition(func_ir, var)
    require(isinstance(call_expr, ir.Expr) and call_expr.op == 'call')
    assert (len(call_expr.args) == 2
            or (accept_stride and len(call_expr.args) == 3))
    assert find_callname(func_ir, call_expr) == ('slice', 'builtins')
    arg0_def = get_definition(func_ir, call_expr.args[0])
    arg1_def = get_definition(func_ir, call_expr.args[1])
    require(isinstance(arg0_def, ir.Const) and arg0_def.value == None)
    require(isinstance(arg1_def, ir.Const) and arg1_def.value == None)
    return True

def get_slice_step(typemap, func_ir, var):
    require(typemap[var.name] == types.slice3_type)
    call_expr = get_definition(func_ir, var)
    require(isinstance(call_expr, ir.Expr) and call_expr.op == 'call')
    assert len(call_expr.args) == 3
    return call_expr.args[2]

def is_array(typemap, varname):
    return (varname in typemap
            and (is_np_array(typemap, varname)
                or typemap[varname] == string_array_type))

def is_np_array(typemap, varname):
    return (varname in typemap
            and isinstance(typemap[varname], types.Array))

# converts an iterable to array, similar to np.array, but can support
# other things like StringArray
# TODO: other types like datetime?
def to_array(A):
    return np.array(A)

@overload(to_array)
def to_array_overload(in_typ):
    # try regular np.array and return it if it works
    def to_array_impl(A):
        return np.array(A)
    try:
        numba.njit(to_array_impl).get_call_template((in_typ,), {})
        return to_array_impl
    except:
        pass  # should be handled elsewhere (e.g. Set)

def empty_like_type(n, arr):
    return np.empty(n, arr.dtype)

@overload(empty_like_type)
def empty_like_type_overload(size_t, arr_typ):
    if isinstance(arr_typ, types.Array):
        return lambda a,b: np.empty(a, b.dtype)
    if isinstance(arr_typ, types.List) and arr_typ.dtype == string_type:
        def empty_like_type_str_list(n, arr):
            return [''] * n
        return empty_like_type_str_list

def is_call(stmt):
    """true if stmt is a getitem or static_getitem assignment"""
    return (isinstance(stmt, ir.Assign)
            and isinstance(stmt.value, ir.Expr)
            and stmt.value.op == 'call')

def is_var_assign(inst):
    return isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Var)

def is_assign(inst):
    return isinstance(inst, ir.Assign)

def dump_node_list(node_list):
    for n in node_list:
        print("   ", n)

def debug_prints():
    return numba.config.DEBUG_ARRAY_OPT == 1

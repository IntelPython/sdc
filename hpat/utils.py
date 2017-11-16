import numba
from numba import ir_utils, ir, types, cgutils
from numba.ir_utils import guard, get_definition
from numba.parfor import wrap_parfor_blocks, unwrap_parfor_blocks
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.targets.imputils import lower_builtin
import collections
import numpy as np

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
        for inst in block.body:
            if isinstance(inst, ir.Assign):
                definitions[inst.target.name].append(inst.value)
            if isinstance(inst, numba.parfor.Parfor):
                parfor_blocks = wrap_parfor_blocks(inst)
                get_definitions(parfor_blocks, definitions)
                unwrap_parfor_blocks(inst)
    return definitions

def is_alloc_call(func_var, call_table):
    assert func_var in call_table
    return call_table[func_var] in [['empty', np],
                                    [numba.unsafe.ndarray.empty_inferred]]

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
            fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
            fn = builder.module.get_or_insert_function(fnty, name="print_str")
            builder.call(fn, [val])
            cgutils.printf(builder, " ")
            continue
        format_str = typ_to_format[typ]
        cgutils.printf(builder, "%{} ".format(format_str), val)
    cgutils.printf(builder, "\n")
    return context.get_dummy_value()

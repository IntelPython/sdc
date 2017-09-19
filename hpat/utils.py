import numba
from numba import ir_utils, ir
from numba.ir_utils import guard, get_definition

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

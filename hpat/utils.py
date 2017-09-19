import numba
from numba import ir_utils, ir
from numba.ir_utils import guard, get_definition

# sentinel value representing non-constant values
class NotConstant:
    pass

NOT_CONSTANT = NotConstant()

def get_constant(func_ir, var, default=NOT_CONSTANT):
    def_node = guard(get_definition, func_ir, var)
    if def_node is not None and isinstance(def_node, ir.Const):
        return def_node.value
    return default

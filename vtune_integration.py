import ctypes
import itt
import llvmlite.binding as ll
from llvmlite.llvmpy.core import Type as LLType
from llvmlite import ir as lir
from llvmlite.llvmpy.core import Constant

functype_domain = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)
ctypes_domain = functype_domain(itt.__itt_domain_create)
domain = ctypes_domain(b"VTune.Profiling.SDC\0")

functype_string_handle_create = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)
ctypes_string_handle_create = functype_string_handle_create(itt.__itt_string_handle_create)

ll.add_symbol('__itt_task_begin_new', itt.__itt_task_begin)
ll.add_symbol('__itt_task_end_new', itt.__itt_task_end)

def vtune_profiling_boxing(name_handle):
    def task(func):
        handle = ctypes_string_handle_create(name_handle)

        def wrapper(typ, val, c):
            fnty = LLType.function(LLType.void(), [c.pyapi.voidptr, c.pyapi.voidptr])
            fn = c.pyapi._get_function(fnty, name="__itt_task_begin_new")
            domain_const = lir.Constant(LLType.int(64), domain)
            handle_const = lir.Constant(LLType.int(64), handle)
            c.builder.call(fn, [Constant.inttoptr(domain_const, c.pyapi.voidptr),
                                Constant.inttoptr(handle_const, c.pyapi.voidptr)])

            return_value = func(typ, val, c)

            fnty_end = LLType.function(LLType.void(), [c.pyapi.voidptr])
            fn_end = c.pyapi._get_function(fnty_end, name="__itt_task_end_new")
            c.builder.call(fn_end, [Constant.inttoptr(domain_const, c.pyapi.voidptr)])

            return return_value

        return wrapper

    return task

import ctypes
import itt
import llvmlite.binding as ll
from llvmlite.llvmpy.core import Type as LLType
from llvmlite import ir as lir
from llvmlite.llvmpy.core import Constant
from inspect import signature
import numba

functype_domain = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)
ctypes_domain = functype_domain(itt.__itt_domain_create)
domain = ctypes_domain(b"VTune.Profiling.SDC\0")

functype_string_handle_create = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)
ctypes_string_handle_create = functype_string_handle_create(itt.__itt_string_handle_create)

functype_task_begin = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)
task_begin = functype_task_begin(itt.__itt_task_begin)

functype_task_end = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
task_end = functype_task_end(itt.__itt_task_end)

ll.add_symbol('__itt_task_begin_new', itt.__itt_task_begin)
ll.add_symbol('__itt_task_end_new', itt.__itt_task_end)


def string_handle_create(string):
    return ctypes_string_handle_create(string.encode())


def vtune_profiling_boxing(name_handle):
    def task(func):
        handle = string_handle_create(name_handle)

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


def vtune_profiling_overload(name_handle):
    def task(func):
        handle = string_handle_create(name_handle)
        args = signature(func)
        return exec_impl(func, handle)

    return task


def codegen_exec_impl(func, handle):
    sig = signature(func)
    sig_str = str(sig)
    args_str = ', '.join(sig.parameters.keys())
    func_lines = [f"def wrapper{sig_str}:",
                  f"  overload_result = func({args_str})",
                  f"  result = numba.njit(overload_result)",
                  f"  def for_jit{sig_str}:",
                  f"    task_begin(domain, handle)",
                  f"    return_value = result({args_str})",
                  f"    task_end(domain)",
                  f"    return return_value",
                  f"  return for_jit"]

    func_text = '\n'.join(func_lines)
    global_vars = {"func": func,
                   "numba": numba,
                   "domain": domain,
                   "handle": handle,
                   "task_begin": task_begin,
                   "task_end": task_end
                   }

    return func_text, global_vars


def exec_impl(func, handle):
    func_text, global_vars = codegen_exec_impl(func, handle)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _impl = loc_vars['wrapper']

    return _impl

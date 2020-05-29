from numba import njit, cfunc, literally
from numba.extending import intrinsic, overload
from numba import types
from numba.core import cgutils
from numba import typed
import ctypes as ct

from sdc import concurrent_sort

def bind(sym, sig):
    # Returns ctypes binding to symbol sym with signature sig
    addr = getattr(concurrent_sort, sym)
    return ct.cast(addr, sig)

parallel_sort_arithm_sig = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_uint64)

parallel_sort_sig = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_uint64,
                               ct.c_uint64, ct.c_void_p,)

parallel_sort_sym = bind('parallel_sort',
                         parallel_sort_sig)

parallel_stable_sort_sym = bind('parallel_stable_sort',
                                parallel_sort_sig)

parallel_sort_t_sig = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_uint64)


def less(left, right):
    pass


@overload(less, jit_options={'locals':{'result':types.int8}})
def less_overload(left, right):
    def less_impl(left, right):
        result = left < right
        return result

    return less_impl


@intrinsic
def adaptor(tyctx, thing, another):
    # This function creates a call specialisation on "custom_hash" based on the
    # type of "thing" and its literal value

    # resolve to function type
    sig = types.intp(thing, another)
    fnty = tyctx.resolve_value_type(less)

    def codegen(cgctx, builder, sig, args):
        ty = sig.args[0]
        # trigger resolution to get a "custom_hash" impl based on the call type
        # "ty" and its literal value
        # import pdb; pdb.set_trace()
        lsig = fnty.get_call_type(tyctx, (ty, ty), {})
        resolved = cgctx.get_function(fnty, lsig)

        # close over resolved function, this is to deal with python scoping
        def resolved_codegen(cgctx, builder, sig, args):
            return resolved(builder, args)

        # A python function "wrapper" is made for the `@cfunc` arg, this calls
        # the jitted function "wrappee", which will be compiled as part of the
        # compilation chain for the cfunc. In turn the wrappee jitted function
        # has an intrinsic call which is holding reference to the resolved type
        # specialised custom_hash call above.
        @intrinsic
        def dispatcher(_ityctx, _a, _b):
            return types.int8(thing, another), resolved_codegen

        @intrinsic
        def deref(_ityctx, _x):
            # to deref the void * passed. TODO: nrt awareness
            catchthing = thing
            sig = catchthing(_x)
            def codegen(cgctx, builder, sig, args):
                toty = cgctx.get_value_type(sig.return_type).as_pointer()
                addressable = builder.bitcast(args[0], toty)
                zero_intpt = cgctx.get_constant(types.intp, 0)
                vref = builder.gep(addressable, [zero_intpt], inbounds=True)
                return builder.load(vref)
            return sig, codegen

        @njit
        def wrappee(ap, bp):
            a = deref(ap)
            b = deref(bp)
            return dispatcher(a, b)

        def wrapper(a, b):
            return wrappee(a, b)

        callback = cfunc(types.int8(types.voidptr, types.voidptr))(wrapper)

        # bake in address as a int const
        address = callback.address
        return cgctx.get_constant(types.intp, address)

    return sig, codegen


@intrinsic
def asvoidp(tyctx, thing):
    sig = types.voidptr(thing)
    def codegen(cgctx, builder, sig, args):
        dm_thing = cgctx.data_model_manager[sig.args[0]]
        data_thing = dm_thing.as_data(builder, args[0])
        ptr_thing = cgutils.alloca_once_value(builder, data_thing)
        return builder.bitcast(ptr_thing, cgutils.voidptr_t)
    return sig, codegen


@intrinsic
def sizeof(context, t):
    sig = types.uint64(t)

    def codegen(cgctx, builder, sig, args):
        size =  cgctx.get_abi_sizeof(t)
        return cgctx.get_constant(types.uint64, size)

    return sig, codegen


types_to_postfix = {types.int8    : 'i8',
                    types.uint8   : 'u8',
                    types.int16   : 'i16',
                    types.uint16  : 'u16',
                    types.int32   : 'i32',
                    types.uint32  : 'u32',
                    types.int64   : 'i64',
                    types.uint64  : 'u64',
                    types.float32 : 'f32',
                    types.float64 : 'f64' }


def load_symbols(name, sig, types):
    result = {}

    func_text = '\n'.join([f"result[{typ}] = bind('{name}_{pstfx}', sig)" for typ, pstfx in types.items()])
    glbls = {f'{typ}' : typ for typ in types.keys()}
    glbls.update({'result': result, 'sig': sig, 'bind': bind})
    exec(func_text, glbls)

    return result


sort_map = load_symbols('parallel_sort', parallel_sort_arithm_sig, types_to_postfix)
stable_sort_map = load_symbols('parallel_stable_sort', parallel_sort_arithm_sig, types_to_postfix)

@intrinsic
def list_itemsize(tyctx, list_ty):
    sig = types.uint64(list_ty)

    def codegen(cgctx, builder, sig, args):
        nb_lty = sig.args[0]
        nb_item_ty = nb_lty.item_type
        ll_item_ty = cgctx.get_value_type(nb_item_ty)
        item_size = cgctx.get_abi_sizeof(ll_item_ty)
        return cgctx.get_constant(sig.return_type, item_size)

    return sig, codegen


def itemsize(arr):
    pass


@overload(itemsize)
def itemsize_overload(arr):
    if isinstance(arr, types.Array):
        def itemsize_impl(arr):
            return arr.itemsize

        return itemsize_impl

    if isinstance(arr, types.List):
        def itemsize_impl(arr):
            return list_itemsize(arr)

        return itemsize_impl

    raise NotImplementedError


def parallel_sort(arr):
    pass


@overload(parallel_sort)
def parallel_sort_overload(arr):

    if not isinstance(arr, types.Array):
        raise NotImplementedError

    dt = arr.dtype

    if dt in types_to_postfix.keys():
        sort_f = sort_map[dt]

        def parallel_sort_arithm_impl(arr):
            return sort_f(arr.ctypes, len(arr))

        return parallel_sort_arithm_impl

    def parallel_sort_impl(arr):
        item_size = itemsize(arr)
        return parallel_sort_sym(arr.ctypes, len(arr), item_size, adaptor(arr[0], arr[0]))

    return parallel_sort_impl

def parallel_stable_sort(arr):
    pass


@overload(parallel_stable_sort)
def parallel_stable_sort_overload(arr):

    if not isinstance(arr, types.Array):
        raise NotImplementedError

    dt = arr.dtype

    if dt in types_to_postfix.keys():
        sort_f = stable_sort_map[dt]

        def parallel_stable_sort_arithm_impl(arr):
            return sort_f(arr.ctypes, len(arr))

        return parallel_stable_sort_arithm_impl

    def parallel_stable_sort_impl(arr):
        item_size = itemsize(arr)
        return parallel_stable_sort_sym(arr.ctypes, len(arr), item_size, adaptor(arr[0], arr[0]))

    return parallel_stable_sort_impl

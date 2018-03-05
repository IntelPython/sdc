import numba
import hpat
from numba import types
from numba.typing.templates import infer_global, AbstractTemplate, infer, signature
from numba.extending import lower_builtin
from numba import cgutils
from hpat.str_ext import string_type
from numba.targets.imputils import impl_ret_new_ref, impl_ret_borrowed
from numba.targets.arrayobj import _empty_nd_impl

import cv2
import numpy as np

from llvmlite import ir as lir
import llvmlite.binding as ll
import cv_wrapper
ll.add_symbol('cv_imread', cv_wrapper.cv_imread)
ll.add_symbol('cv_resize', cv_wrapper.cv_resize)
ll.add_symbol('cv_mat_release', cv_wrapper.cv_mat_release)

@infer_global(cv2.imread, typing_key='cv2imread')
class ImreadInfer(AbstractTemplate):
    def generic(self, args, kws):
        if not kws and len(args) == 1 and args[0] == string_type:
            return signature(types.Array(types.uint8, 3, 'C'), *args)

@infer_global(cv2.resize, typing_key='cv2resize')
class ImreadInfer(AbstractTemplate):
    def generic(self, args, kws):
        if not kws and len(args) == 2 and args[0] == types.Array(types.uint8, 3, 'C'):
            return signature(types.Array(types.uint8, 3, 'C'), *args)

@lower_builtin('cv2imread', string_type)
def lower_cv2_imread(context, builder, sig, args):
    fname = args[0]
    arrtype = sig.return_type

    # read shapes and data pointer
    ll_shty = lir.ArrayType(cgutils.intp_t, arrtype.ndim)
    shapes_array = cgutils.alloca_once(builder, ll_shty)
    data = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())

    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [ll_shty.as_pointer(),
                             lir.IntType(8).as_pointer().as_pointer(),
                             lir.IntType(8).as_pointer()])
    fn_imread = builder.module.get_or_insert_function(fnty, name="cv_imread")
    img = builder.call(fn_imread, [shapes_array, data, fname])


    return _image_to_array(context, builder, shapes_array, arrtype, data, img)


@lower_builtin('cv2resize', types.Array, types.UniTuple)
def lower_cv2_resize(context, builder, sig, args):
    #
    in_arrtype = sig.args[0]
    in_array = context.make_array(in_arrtype)(context, builder, args[0])
    in_shapes = cgutils.unpack_tuple(builder, in_array.shape)
    # cgutils.printf(builder, "%lld\n", in_shapes[2])

    new_sizes = cgutils.unpack_tuple(builder, args[1])

    ary = _empty_nd_impl(context, builder, in_arrtype, [new_sizes[1], new_sizes[0], in_shapes[2]])

    fnty = lir.FunctionType(lir.VoidType(),
                            [lir.IntType(64), lir.IntType(64),
                             lir.IntType(8).as_pointer(),
                             lir.IntType(8).as_pointer(),
                             lir.IntType(64),
                             lir.IntType(64)])
    fn_resize = builder.module.get_or_insert_function(fnty, name="cv_resize")
    img = builder.call(fn_resize, [new_sizes[1], new_sizes[0], ary.data, in_array.data,
                                    in_shapes[0], in_shapes[1]])

    return impl_ret_new_ref(context, builder, in_arrtype, ary._getvalue())


def _image_to_array(context, builder, shapes_array, arrtype, data, img):
    # allocate array
    shapes = cgutils.unpack_tuple(builder, builder.load(shapes_array))
    ary = _empty_nd_impl(context, builder, arrtype, shapes)
    cgutils.raw_memcpy(builder, ary.data, builder.load(data), ary.nitems,
                       ary.itemsize, align=1)

    # clean up cv::Mat image
    fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
    fn_release = builder.module.get_or_insert_function(fnty, name="cv_mat_release")
    builder.call(fn_release, [img])

    return impl_ret_new_ref(context, builder, arrtype, ary._getvalue())

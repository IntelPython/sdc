# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

import numba
import sdc

from numba import types, typing, generated_jit
from numba.extending import models, register_model
from numba.extending import lower_builtin, overload_method, overload, intrinsic

from llvmlite import ir as lir
import llvmlite.binding as ll
from . import hconcurrent_hash
ll.add_symbol('create_int_hashmap', hconcurrent_hash.create_int_hashmap)
ll.add_symbol('delete_int_hashmap', hconcurrent_hash.delete_int_hashmap)
ll.add_symbol('addelem_int_hashmap', hconcurrent_hash.addelem_int_hashmap)

ll.add_symbol('createiter_int_hashmap', hconcurrent_hash.createiter_int_hashmap)
ll.add_symbol('enditer_int_hashmap', hconcurrent_hash.enditer_int_hashmap)
ll.add_symbol('nextiter_int_hashmap', hconcurrent_hash.nextiter_int_hashmap)
ll.add_symbol('iterkey_int_hashmap', hconcurrent_hash.iterkey_int_hashmap)
ll.add_symbol('iterval_int_hashmap', hconcurrent_hash.iterval_int_hashmap)
ll.add_symbol('deleteiter_int_hashmap', hconcurrent_hash.deleteiter_int_hashmap)

_create_int_hashmap = types.ExternalFunction("create_int_hashmap",
                                             types.voidptr())
_delete_int_hashmap = types.ExternalFunction("delete_int_hashmap",
                                             types.void(types.voidptr))
_addelem_int_hashmap = types.ExternalFunction("addelem_int_hashmap",
                                              types.void(types.voidptr, types.int64, types.intp))

_createiter_int_hashmap = types.ExternalFunction("createiter_int_hashmap",
                                                 types.voidptr(types.voidptr))
_enditer_int_hashmap = types.ExternalFunction("enditer_int_hashmap",
                                              types.int32(types.voidptr))
_nextiter_int_hashmap = types.ExternalFunction("nextiter_int_hashmap",
                                               types.void(types.voidptr))
_iterkey_int_hashmap = types.ExternalFunction("iterkey_int_hashmap",
                                              types.int64(types.voidptr))
_iterval_int_hashmap = types.ExternalFunction("iterval_int_hashmap",
                                              types.intp(types.voidptr))
_deleteiter_int_hashmap = types.ExternalFunction("deleteiter_int_hashmap",
                                                 types.void(types.voidptr))


def create_int_hashmap():
    pass


def delete_int_hashmap():
    pass


def addelem_int_hashmap():
    pass


def createiter_int_hashmap():
    pass


def enditer_int_hashmap():
    pass


def nextiter_int_hashmap():
    pass


def iterkey_int_hashmap():
    pass


def iterval_int_hashmap():
    pass


def deleteiter_int_hashmap():
    pass


@overload(create_int_hashmap)
def create_int_hashmap_overload():
    return lambda: _create_int_hashmap()


@overload(delete_int_hashmap)
def delete_int_hashmap_overload(h):
    return lambda h: _delete_int_hashmap(h)


@overload(addelem_int_hashmap)
def addelem_int_hashmap_overload(h, key, val):
    return lambda h, key, val: _addelem_int_hashmap(h, key, val)


@overload(createiter_int_hashmap)
def createiter_int_hashmap_overload(h):
    return lambda h: _createiter_int_hashmap(h)


@overload(enditer_int_hashmap)
def enditer_int_hashmap_overload(h):
    return lambda h: _enditer_int_hashmap(h)


@overload(nextiter_int_hashmap)
def nextiter_int_hashmap_overload(h):
    return lambda h: _nextiter_int_hashmap(h)


@overload(iterkey_int_hashmap)
def iterkey_int_hashmap_overload(h):
    return lambda h: _iterkey_int_hashmap(h)


@overload(iterval_int_hashmap)
def iterval_int_hashmap_overload(h):
    return lambda h: _iterval_int_hashmap(h)


@overload(deleteiter_int_hashmap)
def deleteiter_int_hashmap_overload(h):
    return lambda h: _deleteiter_int_hashmap(h)

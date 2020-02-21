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
ll.add_symbol('addelem_int_hashmap',hconcurrent_hash.addelem_int_hashmap)

ll.add_symbol('createiter_int_hashmap',hconcurrent_hash.createiter_int_hashmap)
ll.add_symbol('enditer_int_hashmap',hconcurrent_hash.enditer_int_hashmap)
ll.add_symbol('nextiter_int_hashmap',hconcurrent_hash.nextiter_int_hashmap)
ll.add_symbol('iterkey_int_hashmap',hconcurrent_hash.iterkey_int_hashmap)
ll.add_symbol('itersize_int_hashmap',hconcurrent_hash.itersize_int_hashmap)
ll.add_symbol('iterelem_int_hashmap',hconcurrent_hash.iterelem_int_hashmap)
ll.add_symbol('deleteiter_int_hashmap',hconcurrent_hash.deleteiter_int_hashmap)

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
_itersize_int_hashmap = types.ExternalFunction("itersize_int_hashmap",
                                               types.intp(types.voidptr))
_iterelem_int_hashmap = types.ExternalFunction("iterelem_int_hashmap",
                                               types.intp(types.voidptr, types.intp))
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

def itersize_int_hashmap():
    pass

def iterelem_int_hashmap():
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

@overload(itersize_int_hashmap)
def itersize_int_hashmap_overload(h):
    return lambda h: _itersize_int_hashmap(h)

@overload(iterelem_int_hashmap)
def iterelem_int_hashmap_overload(h, i):
    return lambda h, i: _iterelem_int_hashmap(h, i)

@overload(deleteiter_int_hashmap)
def deleteiter_int_hashmap_overload(h):
    return lambda h: _deleteiter_int_hashmap(h)

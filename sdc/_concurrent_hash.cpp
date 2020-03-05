//*****************************************************************************
// Copyright (c) 2020, Intel Corporation All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

#include <Python.h>
#include <cstdint>
#include <tbb/concurrent_unordered_map.h>


template<typename Key, typename Val>
using hashmap = tbb::concurrent_unordered_multimap<Key,Val>;

template<typename Key, typename Val>
using iter_range = std::pair<typename hashmap<Key, Val>::iterator, typename hashmap<Key, Val>::iterator>;

using int_hashmap = hashmap<int64_t, size_t>;
using int_hashmap_iters = iter_range<int64_t, size_t>;

extern "C"
{
void* create_int_hashmap()
{
    return new int_hashmap;
}

void delete_int_hashmap(void* obj)
{
    delete static_cast<int_hashmap*>(obj);
}

void addelem_int_hashmap(void* obj, int64_t key, size_t val)
{
    auto& h = *static_cast<int_hashmap*>(obj);
    h.insert({key,val});
}

void* createiter_int_hashmap(void* obj)
{
    auto& h = *static_cast<int_hashmap*>(obj);
    return new int_hashmap_iters{h.begin(), h.end()};
}

int32_t enditer_int_hashmap(void* it)
{
    auto& r = *static_cast<int_hashmap_iters*>(it);
    return static_cast<int32_t>(r.first == r.second);
}

void nextiter_int_hashmap(void* it)
{
    auto& r = *static_cast<int_hashmap_iters*>(it);
    ++r.first;
}

int64_t iterkey_int_hashmap(void* it)
{
    auto& r = *static_cast<int_hashmap_iters*>(it);
    return r.first->first;
}

size_t iterval_int_hashmap(void* it)
{
    auto& r = *static_cast<int_hashmap_iters*>(it);
    return r.first->second;
}

void deleteiter_int_hashmap(void* obj)
{
    delete static_cast<int_hashmap_iters*>(obj);
}

using funcptr_t = int32_t(*)(int32_t,int32_t,int32_t);
int32_t test_funcptr(funcptr_t func, int32_t a, int32_t b)
{
    int32_t res = 0;
    for (int i = 0; i < 10; ++i)
    {
        res += func(a, b, i);
    }
    return res;
}

PyMODINIT_FUNC PyInit_hconcurrent_hash()
{
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "hconcurrent_hash",
        "No docs",
        -1,
        NULL,
    };
    PyObject* m = PyModule_Create(&moduledef);
    if (m == NULL)
    {
        return NULL;
    }

#define REGISTER(func) PyObject_SetAttrString(m, #func, PyLong_FromVoidPtr((void*)(&func)));
    REGISTER(create_int_hashmap)
    REGISTER(delete_int_hashmap)
    REGISTER(addelem_int_hashmap)

    REGISTER(createiter_int_hashmap)
    REGISTER(enditer_int_hashmap)
    REGISTER(nextiter_int_hashmap)
    REGISTER(iterkey_int_hashmap)
    REGISTER(iterval_int_hashmap)
    REGISTER(deleteiter_int_hashmap)

    REGISTER(test_funcptr)
#undef REGISTER
    return m;
}
}

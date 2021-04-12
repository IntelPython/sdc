// *****************************************************************************
// Copyright (c) 2021, Intel Corporation All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//     Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
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
// *****************************************************************************

#include <Python.h>
#include "hashmap.hpp"


#define declare_hashmap_create(key_type, val_type, suffix) \
void hashmap_create_##suffix(NRT_MemInfo** meminfo, \
                             void* nrt_table, \
                             int8_t gen_key, \
                             int8_t gen_val, \
                             void* hash_func_ptr, \
                             void* eq_func_ptr, \
                             void* key_incref_func_ptr, \
                             void* key_decref_func_ptr, \
                             void* val_incref_func_ptr, \
                             void* val_decref_func_ptr, \
                             uint64_t key_size, \
                             uint64_t val_size) \
{ \
    hashmap_create<key_type, val_type>( \
                meminfo, nrt_table, \
                gen_key, gen_val, \
                hash_func_ptr, eq_func_ptr, \
                key_incref_func_ptr, key_decref_func_ptr, \
                val_incref_func_ptr, val_decref_func_ptr, \
                key_size, val_size); \
} \


#define declare_hashmap_size(key_type, val_type, suffix) \
uint64_t hashmap_size_##suffix(void* p_hash_map) \
{ \
    return hashmap_size<key_type, val_type>(p_hash_map); \
} \


#define declare_hashmap_set(key_type, val_type, suffix) \
void hashmap_set_##suffix(void* p_hash_map, key_type key, val_type val) \
{ \
    hashmap_set<key_type, val_type>(p_hash_map, key, val); \
} \


#define declare_hashmap_contains(key_type, val_type, suffix) \
int8_t hashmap_contains_##suffix(void* p_hash_map, key_type key) \
{ \
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map); \
    return p_hash_map_spec->contains(key); \
} \


#define declare_hashmap_lookup(key_type, val_type, suffix) \
int8_t hashmap_lookup_##suffix(void* p_hash_map, key_type key, val_type* res) \
{ \
    return hashmap_lookup<key_type, val_type>(p_hash_map, key, res); \
} \


#define declare_hashmap_clear(key_type, val_type, suffix) \
void hashmap_clear_##suffix(void* p_hash_map) \
{ \
    return hashmap_clear<key_type, val_type>(p_hash_map); \
} \


#define declare_hashmap_pop(key_type, val_type, suffix) \
int8_t hashmap_pop_##suffix(void* p_hash_map, key_type key, val_type* res) \
{ \
    return hashmap_unsafe_extract<key_type, val_type>(p_hash_map, key, res); \
} \


#define declare_hashmap_create_from_data(key_type, val_type) \
void hashmap_create_from_data_##key_type##_to_##val_type(NRT_MemInfo** meminfo, void* nrt_table, key_type* keys, val_type* values, int64_t size) \
{ \
    hashmap_numeric_from_arrays<key_type, val_type>(meminfo, nrt_table, keys, values, size); \
} \


#define declare_hashmap_update(key_type, val_type, suffix) \
void hashmap_update_##suffix(void* p_self_hash_map, void* p_arg_hash_map) \
{ \
    return hashmap_update<key_type, val_type>(p_self_hash_map, p_arg_hash_map); \
} \


#define declare_hashmap_dump(key_type, val_type, suffix) \
void hashmap_dump_##suffix(void* p_hash_map) \
{ \
    hashmap_dump<key_type, val_type>(p_hash_map); \
} \


#define declare_hashmap_getiter(key_type, val_type, suffix) \
void* hashmap_getiter_##suffix(NRT_MemInfo** meminfo, void* nrt_table, void* p_hash_map) \
{ \
    return hashmap_getiter<key_type, val_type>(meminfo, nrt_table, p_hash_map); \
} \


#define declare_hashmap_iternext(key_type, val_type, suffix) \
int8_t hashmap_iternext_##suffix(void* p_iter_state, key_type* ret_key, val_type* ret_val) \
{ \
    return hashmap_iternext<key_type, val_type>(p_iter_state, ret_key, ret_val); \
} \


extern "C"
{

// empty hashmap creation functions
declare_hashmap_create(int32_t, int32_t, int32_t_to_int32_t);
declare_hashmap_create(int32_t, int64_t, int32_t_to_int64_t);
declare_hashmap_create(int32_t, float, int32_t_to_float);
declare_hashmap_create(int32_t, double, int32_t_to_double);

declare_hashmap_create(int64_t, int32_t, int64_t_to_int32_t);
declare_hashmap_create(int64_t, int64_t, int64_t_to_int64_t);
declare_hashmap_create(int64_t, float, int64_t_to_float);
declare_hashmap_create(int64_t, double, int64_t_to_double);

declare_hashmap_create(void*, int32_t, voidptr_to_int32_t);
declare_hashmap_create(void*, int64_t, voidptr_to_int64_t);
declare_hashmap_create(void*, float, voidptr_to_float);
declare_hashmap_create(void*, double, voidptr_to_double);

declare_hashmap_create(int32_t, void*, int32_t_to_voidptr);
declare_hashmap_create(int64_t, void*, int64_t_to_voidptr);

declare_hashmap_create(void*, void*, voidptr_to_voidptr);

// hashmap size functions
declare_hashmap_size(int32_t, int32_t, int32_t_to_int32_t);
declare_hashmap_size(int32_t, int64_t, int32_t_to_int64_t);
declare_hashmap_size(int32_t, float, int32_t_to_float);
declare_hashmap_size(int32_t, double, int32_t_to_double);

declare_hashmap_size(int64_t, int32_t, int64_t_to_int32_t);
declare_hashmap_size(int64_t, int64_t, int64_t_to_int64_t);
declare_hashmap_size(int64_t, float, int64_t_to_float);
declare_hashmap_size(int64_t, double, int64_t_to_double);

declare_hashmap_size(void*, int32_t, voidptr_to_int32_t);
declare_hashmap_size(void*, int64_t, voidptr_to_int64_t);
declare_hashmap_size(void*, float, voidptr_to_float);
declare_hashmap_size(void*, double, voidptr_to_double);

declare_hashmap_size(int32_t, void*, int32_t_to_voidptr);
declare_hashmap_size(int64_t, void*, int64_t_to_voidptr);

declare_hashmap_size(void*, void*, voidptr_to_voidptr);

// hashmap set functions
declare_hashmap_set(int32_t, int32_t, int32_t_to_int32_t);
declare_hashmap_set(int32_t, int64_t, int32_t_to_int64_t);
declare_hashmap_set(int32_t, float, int32_t_to_float);
declare_hashmap_set(int32_t, double, int32_t_to_double);

declare_hashmap_set(int64_t, int32_t, int64_t_to_int32_t);
declare_hashmap_set(int64_t, int64_t, int64_t_to_int64_t);
declare_hashmap_set(int64_t, float, int64_t_to_float);
declare_hashmap_set(int64_t, double, int64_t_to_double);

declare_hashmap_set(void*, int32_t, voidptr_to_int32_t);
declare_hashmap_set(void*, int64_t, voidptr_to_int64_t);
declare_hashmap_set(void*, float, voidptr_to_float);
declare_hashmap_set(void*, double, voidptr_to_double);

declare_hashmap_set(int32_t, void*, int32_t_to_voidptr);
declare_hashmap_set(int64_t, void*, int64_t_to_voidptr);

declare_hashmap_set(void*, void*, voidptr_to_voidptr);

// hashmap contains functions
declare_hashmap_contains(int32_t, int32_t, int32_t_to_int32_t);
declare_hashmap_contains(int32_t, int64_t, int32_t_to_int64_t);
declare_hashmap_contains(int32_t, float, int32_t_to_float);
declare_hashmap_contains(int32_t, double, int32_t_to_double);

declare_hashmap_contains(int64_t, int32_t, int64_t_to_int32_t);
declare_hashmap_contains(int64_t, int64_t, int64_t_to_int64_t);
declare_hashmap_contains(int64_t, float, int64_t_to_float);
declare_hashmap_contains(int64_t, double, int64_t_to_double);

declare_hashmap_contains(void*, int32_t, voidptr_to_int32_t);
declare_hashmap_contains(void*, int64_t, voidptr_to_int64_t);
declare_hashmap_contains(void*, float, voidptr_to_float);
declare_hashmap_contains(void*, double, voidptr_to_double);

declare_hashmap_contains(int32_t, void*, int32_t_to_voidptr);
declare_hashmap_contains(int64_t, void*, int64_t_to_voidptr);

declare_hashmap_contains(void*, void*, voidptr_to_voidptr);

// hashmap lookup functions
declare_hashmap_lookup(int32_t, int32_t, int32_t_to_int32_t);
declare_hashmap_lookup(int32_t, int64_t, int32_t_to_int64_t);
declare_hashmap_lookup(int32_t, float, int32_t_to_float);
declare_hashmap_lookup(int32_t, double, int32_t_to_double);

declare_hashmap_lookup(int64_t, int32_t, int64_t_to_int32_t);
declare_hashmap_lookup(int64_t, int64_t, int64_t_to_int64_t);
declare_hashmap_lookup(int64_t, float, int64_t_to_float);
declare_hashmap_lookup(int64_t, double, int64_t_to_double);

declare_hashmap_lookup(void*, int32_t, voidptr_to_int32_t);
declare_hashmap_lookup(void*, int64_t, voidptr_to_int64_t);
declare_hashmap_lookup(void*, float, voidptr_to_float);
declare_hashmap_lookup(void*, double, voidptr_to_double);

declare_hashmap_lookup(int32_t, void*, int32_t_to_voidptr);
declare_hashmap_lookup(int64_t, void*, int64_t_to_voidptr);

declare_hashmap_lookup(void*, void*, voidptr_to_voidptr);

// hashmap clear functions
declare_hashmap_clear(int32_t, int32_t, int32_t_to_int32_t);
declare_hashmap_clear(int32_t, int64_t, int32_t_to_int64_t);
declare_hashmap_clear(int32_t, float, int32_t_to_float);
declare_hashmap_clear(int32_t, double, int32_t_to_double);

declare_hashmap_clear(int64_t, int32_t, int64_t_to_int32_t);
declare_hashmap_clear(int64_t, int64_t, int64_t_to_int64_t);
declare_hashmap_clear(int64_t, float, int64_t_to_float);
declare_hashmap_clear(int64_t, double, int64_t_to_double);

declare_hashmap_clear(void*, int32_t, voidptr_to_int32_t);
declare_hashmap_clear(void*, int64_t, voidptr_to_int64_t);
declare_hashmap_clear(void*, float, voidptr_to_float);
declare_hashmap_clear(void*, double, voidptr_to_double);

declare_hashmap_clear(int32_t, void*, int32_t_to_voidptr);
declare_hashmap_clear(int64_t, void*, int64_t_to_voidptr);

declare_hashmap_clear(void*, void*, voidptr_to_voidptr);

// hashmap pop functions
declare_hashmap_pop(int32_t, int32_t, int32_t_to_int32_t);
declare_hashmap_pop(int32_t, int64_t, int32_t_to_int64_t);
declare_hashmap_pop(int32_t, float, int32_t_to_float);
declare_hashmap_pop(int32_t, double, int32_t_to_double);

declare_hashmap_pop(int64_t, int32_t, int64_t_to_int32_t);
declare_hashmap_pop(int64_t, int64_t, int64_t_to_int64_t);
declare_hashmap_pop(int64_t, float, int64_t_to_float);
declare_hashmap_pop(int64_t, double, int64_t_to_double);

declare_hashmap_pop(void*, int32_t, voidptr_to_int32_t);
declare_hashmap_pop(void*, int64_t, voidptr_to_int64_t);
declare_hashmap_pop(void*, float, voidptr_to_float);
declare_hashmap_pop(void*, double, voidptr_to_double);

declare_hashmap_pop(int32_t, void*, int32_t_to_voidptr);
declare_hashmap_pop(int64_t, void*, int64_t_to_voidptr);

declare_hashmap_pop(void*, void*, voidptr_to_voidptr);

// hashmap update functions
declare_hashmap_update(int32_t, int32_t, int32_t_to_int32_t);
declare_hashmap_update(int32_t, int64_t, int32_t_to_int64_t);
declare_hashmap_update(int32_t, float, int32_t_to_float);
declare_hashmap_update(int32_t, double, int32_t_to_double);

declare_hashmap_update(int64_t, int32_t, int64_t_to_int32_t);
declare_hashmap_update(int64_t, int64_t, int64_t_to_int64_t);
declare_hashmap_update(int64_t, float, int64_t_to_float);
declare_hashmap_update(int64_t, double, int64_t_to_double);

declare_hashmap_update(void*, int32_t, voidptr_to_int32_t);
declare_hashmap_update(void*, int64_t, voidptr_to_int64_t);
declare_hashmap_update(void*, float, voidptr_to_float);
declare_hashmap_update(void*, double, voidptr_to_double);

declare_hashmap_update(int32_t, void*, int32_t_to_voidptr);
declare_hashmap_update(int64_t, void*, int64_t_to_voidptr);

declare_hashmap_update(void*, void*, voidptr_to_voidptr);

// numeric hashmap create_from_data functions
declare_hashmap_create_from_data(int32_t, int32_t);
declare_hashmap_create_from_data(int32_t, int64_t);
declare_hashmap_create_from_data(int32_t, float);
declare_hashmap_create_from_data(int32_t, double);

declare_hashmap_create_from_data(int64_t, int32_t);
declare_hashmap_create_from_data(int64_t, int64_t);
declare_hashmap_create_from_data(int64_t, float);
declare_hashmap_create_from_data(int64_t, double);

// hashmap dump functions
declare_hashmap_dump(int32_t, int32_t, int32_t_to_int32_t);
declare_hashmap_dump(int32_t, int64_t, int32_t_to_int64_t);
declare_hashmap_dump(int32_t, float, int32_t_to_float);
declare_hashmap_dump(int32_t, double, int32_t_to_double);

declare_hashmap_dump(int64_t, int32_t, int64_t_to_int32_t);
declare_hashmap_dump(int64_t, int64_t, int64_t_to_int64_t);
declare_hashmap_dump(int64_t, float, int64_t_to_float);
declare_hashmap_dump(int64_t, double, int64_t_to_double);

declare_hashmap_dump(void*, int32_t, voidptr_to_int32_t);
declare_hashmap_dump(void*, int64_t, voidptr_to_int64_t);
declare_hashmap_dump(void*, float, voidptr_to_float);
declare_hashmap_dump(void*, double, voidptr_to_double);

declare_hashmap_dump(int32_t, void*, int32_t_to_voidptr);
declare_hashmap_dump(int64_t, void*, int64_t_to_voidptr);

declare_hashmap_dump(void*, void*, voidptr_to_voidptr);


// hashmap getiter functions
declare_hashmap_getiter(int32_t, int32_t, int32_t_to_int32_t);
declare_hashmap_getiter(int32_t, int64_t, int32_t_to_int64_t);
declare_hashmap_getiter(int32_t, float, int32_t_to_float);
declare_hashmap_getiter(int32_t, double, int32_t_to_double);

declare_hashmap_getiter(int64_t, int32_t, int64_t_to_int32_t);
declare_hashmap_getiter(int64_t, int64_t, int64_t_to_int64_t);
declare_hashmap_getiter(int64_t, float, int64_t_to_float);
declare_hashmap_getiter(int64_t, double, int64_t_to_double);

declare_hashmap_getiter(void*, int32_t, voidptr_to_int32_t);
declare_hashmap_getiter(void*, int64_t, voidptr_to_int64_t);
declare_hashmap_getiter(void*, float, voidptr_to_float);
declare_hashmap_getiter(void*, double, voidptr_to_double);

declare_hashmap_getiter(int32_t, void*, int32_t_to_voidptr);
declare_hashmap_getiter(int64_t, void*, int64_t_to_voidptr);

declare_hashmap_getiter(void*, void*, voidptr_to_voidptr);

// hashmap iternext functions
declare_hashmap_iternext(int32_t, int32_t, int32_t_to_int32_t);
declare_hashmap_iternext(int32_t, int64_t, int32_t_to_int64_t);
declare_hashmap_iternext(int32_t, float, int32_t_to_float);
declare_hashmap_iternext(int32_t, double, int32_t_to_double);

declare_hashmap_iternext(int64_t, int32_t, int64_t_to_int32_t);
declare_hashmap_iternext(int64_t, int64_t, int64_t_to_int64_t);
declare_hashmap_iternext(int64_t, float, int64_t_to_float);
declare_hashmap_iternext(int64_t, double, int64_t_to_double);

declare_hashmap_iternext(void*, int32_t, voidptr_to_int32_t);
declare_hashmap_iternext(void*, int64_t, voidptr_to_int64_t);
declare_hashmap_iternext(void*, float, voidptr_to_float);
declare_hashmap_iternext(void*, double, voidptr_to_double);

declare_hashmap_iternext(int32_t, void*, int32_t_to_voidptr);
declare_hashmap_iternext(int64_t, void*, int64_t_to_voidptr);

declare_hashmap_iternext(void*, void*, voidptr_to_voidptr);


#define REGISTER(func) PyObject_SetAttrString(m, #func, PyLong_FromVoidPtr((void*)(&func)));

PyMODINIT_FUNC PyInit_hconc_dict()
{
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "htbb_hashmap",
        "No docs",
        -1,
        NULL,
    };
    PyObject* m = PyModule_Create(&moduledef);
    if (m == NULL)
    {
        return NULL;
    }

    // hashmap creation functions
    REGISTER(hashmap_create_int32_t_to_int32_t)
    REGISTER(hashmap_create_int32_t_to_int64_t)
    REGISTER(hashmap_create_int64_t_to_int32_t)
    REGISTER(hashmap_create_int64_t_to_int64_t)

    REGISTER(hashmap_create_int32_t_to_float)
    REGISTER(hashmap_create_int32_t_to_double)
    REGISTER(hashmap_create_int64_t_to_float)
    REGISTER(hashmap_create_int64_t_to_double)

    REGISTER(hashmap_create_voidptr_to_int32_t)
    REGISTER(hashmap_create_voidptr_to_int64_t)
    REGISTER(hashmap_create_voidptr_to_float)
    REGISTER(hashmap_create_voidptr_to_double)

    REGISTER(hashmap_create_int32_t_to_voidptr)
    REGISTER(hashmap_create_int64_t_to_voidptr)

    REGISTER(hashmap_create_voidptr_to_voidptr);

    // hashmap size functions
    REGISTER(hashmap_size_int32_t_to_int32_t)
    REGISTER(hashmap_size_int32_t_to_int64_t)
    REGISTER(hashmap_size_int64_t_to_int32_t)
    REGISTER(hashmap_size_int64_t_to_int64_t)

    REGISTER(hashmap_size_int32_t_to_float)
    REGISTER(hashmap_size_int32_t_to_double)
    REGISTER(hashmap_size_int64_t_to_float)
    REGISTER(hashmap_size_int64_t_to_double)

    REGISTER(hashmap_size_voidptr_to_int32_t)
    REGISTER(hashmap_size_voidptr_to_int64_t)
    REGISTER(hashmap_size_voidptr_to_float)
    REGISTER(hashmap_size_voidptr_to_double)

    REGISTER(hashmap_size_int32_t_to_voidptr)
    REGISTER(hashmap_size_int64_t_to_voidptr)

    REGISTER(hashmap_size_voidptr_to_voidptr);

    // hashmap set functions
    REGISTER(hashmap_set_int32_t_to_int32_t)
    REGISTER(hashmap_set_int32_t_to_int64_t)
    REGISTER(hashmap_set_int64_t_to_int32_t)
    REGISTER(hashmap_set_int64_t_to_int64_t)

    REGISTER(hashmap_set_int32_t_to_float)
    REGISTER(hashmap_set_int32_t_to_double)
    REGISTER(hashmap_set_int64_t_to_float)
    REGISTER(hashmap_set_int64_t_to_double)

    REGISTER(hashmap_set_voidptr_to_int32_t)
    REGISTER(hashmap_set_voidptr_to_int64_t)
    REGISTER(hashmap_set_voidptr_to_float)
    REGISTER(hashmap_set_voidptr_to_double)

    REGISTER(hashmap_set_int32_t_to_voidptr)
    REGISTER(hashmap_set_int64_t_to_voidptr)

    REGISTER(hashmap_set_voidptr_to_voidptr);

    // hashmap contains functions
    REGISTER(hashmap_contains_int32_t_to_int32_t)
    REGISTER(hashmap_contains_int32_t_to_int64_t)
    REGISTER(hashmap_contains_int64_t_to_int32_t)
    REGISTER(hashmap_contains_int64_t_to_int64_t)

    REGISTER(hashmap_contains_int32_t_to_float)
    REGISTER(hashmap_contains_int32_t_to_double)
    REGISTER(hashmap_contains_int64_t_to_float)
    REGISTER(hashmap_contains_int64_t_to_double)

    REGISTER(hashmap_contains_voidptr_to_int32_t)
    REGISTER(hashmap_contains_voidptr_to_int64_t)
    REGISTER(hashmap_contains_voidptr_to_float)
    REGISTER(hashmap_contains_voidptr_to_double)

    REGISTER(hashmap_contains_int32_t_to_voidptr)
    REGISTER(hashmap_contains_int64_t_to_voidptr)

    REGISTER(hashmap_contains_voidptr_to_voidptr);

    // hashmap lookup functions
    REGISTER(hashmap_lookup_int32_t_to_int32_t)
    REGISTER(hashmap_lookup_int32_t_to_int64_t)
    REGISTER(hashmap_lookup_int64_t_to_int32_t)
    REGISTER(hashmap_lookup_int64_t_to_int64_t)

    REGISTER(hashmap_lookup_int32_t_to_float)
    REGISTER(hashmap_lookup_int32_t_to_double)
    REGISTER(hashmap_lookup_int64_t_to_float)
    REGISTER(hashmap_lookup_int64_t_to_double)

    REGISTER(hashmap_lookup_voidptr_to_int32_t)
    REGISTER(hashmap_lookup_voidptr_to_int64_t)
    REGISTER(hashmap_lookup_voidptr_to_float)
    REGISTER(hashmap_lookup_voidptr_to_double)

    REGISTER(hashmap_lookup_int32_t_to_voidptr)
    REGISTER(hashmap_lookup_int64_t_to_voidptr)

    REGISTER(hashmap_lookup_voidptr_to_voidptr);

    // hashmap clear functions
    REGISTER(hashmap_clear_int32_t_to_int32_t)
    REGISTER(hashmap_clear_int32_t_to_int64_t)
    REGISTER(hashmap_clear_int64_t_to_int32_t)
    REGISTER(hashmap_clear_int64_t_to_int64_t)

    REGISTER(hashmap_clear_int32_t_to_float)
    REGISTER(hashmap_clear_int32_t_to_double)
    REGISTER(hashmap_clear_int64_t_to_float)
    REGISTER(hashmap_clear_int64_t_to_double)

    REGISTER(hashmap_clear_voidptr_to_int32_t)
    REGISTER(hashmap_clear_voidptr_to_int64_t)
    REGISTER(hashmap_clear_voidptr_to_float)
    REGISTER(hashmap_clear_voidptr_to_double)

    REGISTER(hashmap_clear_int32_t_to_voidptr)
    REGISTER(hashmap_clear_int64_t_to_voidptr)

    REGISTER(hashmap_clear_voidptr_to_voidptr);

    // hashmap pop functions
    REGISTER(hashmap_pop_int32_t_to_int32_t)
    REGISTER(hashmap_pop_int32_t_to_int64_t)
    REGISTER(hashmap_pop_int64_t_to_int32_t)
    REGISTER(hashmap_pop_int64_t_to_int64_t)

    REGISTER(hashmap_pop_int32_t_to_float)
    REGISTER(hashmap_pop_int32_t_to_double)
    REGISTER(hashmap_pop_int64_t_to_float)
    REGISTER(hashmap_pop_int64_t_to_double)

    REGISTER(hashmap_pop_voidptr_to_int32_t)
    REGISTER(hashmap_pop_voidptr_to_int64_t)
    REGISTER(hashmap_pop_voidptr_to_float)
    REGISTER(hashmap_pop_voidptr_to_double)

    REGISTER(hashmap_pop_int32_t_to_voidptr)
    REGISTER(hashmap_pop_int64_t_to_voidptr)

    REGISTER(hashmap_pop_voidptr_to_voidptr);

    // hashmap update functions
    REGISTER(hashmap_update_int32_t_to_int32_t)
    REGISTER(hashmap_update_int32_t_to_int64_t)
    REGISTER(hashmap_update_int64_t_to_int32_t)
    REGISTER(hashmap_update_int64_t_to_int64_t)

    REGISTER(hashmap_update_int32_t_to_float)
    REGISTER(hashmap_update_int32_t_to_double)
    REGISTER(hashmap_update_int64_t_to_float)
    REGISTER(hashmap_update_int64_t_to_double)

    REGISTER(hashmap_update_voidptr_to_int32_t)
    REGISTER(hashmap_update_voidptr_to_int64_t)
    REGISTER(hashmap_update_voidptr_to_float)
    REGISTER(hashmap_update_voidptr_to_double)

    REGISTER(hashmap_update_int32_t_to_voidptr)
    REGISTER(hashmap_update_int64_t_to_voidptr)

    REGISTER(hashmap_update_voidptr_to_voidptr);

    // hashmap create_from_data functions
    REGISTER(hashmap_create_from_data_int32_t_to_int32_t)
    REGISTER(hashmap_create_from_data_int32_t_to_int64_t)
    REGISTER(hashmap_create_from_data_int64_t_to_int32_t)
    REGISTER(hashmap_create_from_data_int64_t_to_int64_t)

    REGISTER(hashmap_create_from_data_int32_t_to_float)
    REGISTER(hashmap_create_from_data_int32_t_to_double)
    REGISTER(hashmap_create_from_data_int64_t_to_float)
    REGISTER(hashmap_create_from_data_int64_t_to_double)

    // hashmap dump functions
    REGISTER(hashmap_dump_int32_t_to_int32_t)
    REGISTER(hashmap_dump_int32_t_to_int64_t)
    REGISTER(hashmap_dump_int64_t_to_int32_t)
    REGISTER(hashmap_dump_int64_t_to_int64_t)

    REGISTER(hashmap_dump_int32_t_to_float)
    REGISTER(hashmap_dump_int32_t_to_double)
    REGISTER(hashmap_dump_int64_t_to_float)
    REGISTER(hashmap_dump_int64_t_to_double)

    REGISTER(hashmap_dump_voidptr_to_int32_t)
    REGISTER(hashmap_dump_voidptr_to_int64_t)
    REGISTER(hashmap_dump_voidptr_to_float)
    REGISTER(hashmap_dump_voidptr_to_double)

    REGISTER(hashmap_dump_int32_t_to_voidptr)
    REGISTER(hashmap_dump_int64_t_to_voidptr)

    REGISTER(hashmap_dump_voidptr_to_voidptr);

    // hashmap getiter functions
    REGISTER(hashmap_getiter_int32_t_to_int32_t)
    REGISTER(hashmap_getiter_int32_t_to_int64_t)
    REGISTER(hashmap_getiter_int64_t_to_int32_t)
    REGISTER(hashmap_getiter_int64_t_to_int64_t)

    REGISTER(hashmap_getiter_int32_t_to_float)
    REGISTER(hashmap_getiter_int32_t_to_double)
    REGISTER(hashmap_getiter_int64_t_to_float)
    REGISTER(hashmap_getiter_int64_t_to_double)

    REGISTER(hashmap_getiter_voidptr_to_int32_t)
    REGISTER(hashmap_getiter_voidptr_to_int64_t)
    REGISTER(hashmap_getiter_voidptr_to_float)
    REGISTER(hashmap_getiter_voidptr_to_double)

    REGISTER(hashmap_getiter_int32_t_to_voidptr)
    REGISTER(hashmap_getiter_int64_t_to_voidptr)

    REGISTER(hashmap_getiter_voidptr_to_voidptr);

    // hashmap iternext functions
    REGISTER(hashmap_iternext_int32_t_to_int32_t)
    REGISTER(hashmap_iternext_int32_t_to_int64_t)
    REGISTER(hashmap_iternext_int64_t_to_int32_t)
    REGISTER(hashmap_iternext_int64_t_to_int64_t)

    REGISTER(hashmap_iternext_int32_t_to_float)
    REGISTER(hashmap_iternext_int32_t_to_double)
    REGISTER(hashmap_iternext_int64_t_to_float)
    REGISTER(hashmap_iternext_int64_t_to_double)

    REGISTER(hashmap_iternext_voidptr_to_int32_t)
    REGISTER(hashmap_iternext_voidptr_to_int64_t)
    REGISTER(hashmap_iternext_voidptr_to_float)
    REGISTER(hashmap_iternext_voidptr_to_double)

    REGISTER(hashmap_iternext_int32_t_to_voidptr)
    REGISTER(hashmap_iternext_int64_t_to_voidptr)

    REGISTER(hashmap_iternext_voidptr_to_voidptr);

    utils::tbb_control::init();

    return m;
}

#undef declare_hashmap_create
#undef declare_hashmap_size
#undef declare_hashmap_set
#undef declare_hashmap_contains
#undef declare_hashmap_lookup
#undef declare_hashmap_clear
#undef declare_hashmap_pop
#undef declare_hashmap_create_from_data
#undef declare_hashmap_update
#undef declare_hashmap_dump
#undef declare_hashmap_getiter
#undef declare_hashmap_iternext
#undef REGISTER

}  // extern "C"


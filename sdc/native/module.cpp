// *****************************************************************************
// Copyright (c) 2020, Intel Corporation All rights reserved.
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
#include <cstdint>
#include "utils.hpp"

extern "C"
{
    void parallel_sort(void* begin, uint64_t len, uint64_t size, void* compare);

    void parallel_sort_i8(void* begin, uint64_t len);
    void parallel_sort_u8(void* begin, uint64_t len);
    void parallel_sort_i16(void* begin, uint64_t len);
    void parallel_sort_u16(void* begin, uint64_t len);
    void parallel_sort_i32(void* begin, uint64_t len);
    void parallel_sort_u32(void* begin, uint64_t len);
    void parallel_sort_i64(void* begin, uint64_t len);
    void parallel_sort_u64(void* begin, uint64_t len);

    void parallel_sort_f32(void* begin, uint64_t len);
    void parallel_sort_f64(void* begin, uint64_t len);

    void parallel_stable_sort(void* begin, uint64_t len, uint64_t size, void* compare);

    void parallel_stable_sort_i8(void* begin, uint64_t len);
    void parallel_stable_sort_u8(void* begin, uint64_t len);
    void parallel_stable_sort_i16(void* begin, uint64_t len);
    void parallel_stable_sort_u16(void* begin, uint64_t len);
    void parallel_stable_sort_i32(void* begin, uint64_t len);
    void parallel_stable_sort_u32(void* begin, uint64_t len);
    void parallel_stable_sort_i64(void* begin, uint64_t len);
    void parallel_stable_sort_u64(void* begin, uint64_t len);

    void parallel_stable_sort_f32(void* begin, uint64_t len);
    void parallel_stable_sort_f64(void* begin, uint64_t len);

    void parallel_argsort_u64v(void* index, void* begin, uint64_t len, uint64_t size, void* compare);

    void parallel_argsort_u64i8(void* index, void* begin, uint64_t len);
    void parallel_argsort_u64u8(void* index, void* begin, uint64_t len);
    void parallel_argsort_u64i16(void* index, void* begin, uint64_t len);
    void parallel_argsort_u64u16(void* index, void* begin, uint64_t len);
    void parallel_argsort_u64i32(void* index, void* begin, uint64_t len);
    void parallel_argsort_u64u32(void* index, void* begin, uint64_t len);
    void parallel_argsort_u64i64(void* index, void* begin, uint64_t len);
    void parallel_argsort_u64u64(void* index, void* begin, uint64_t len);

    void parallel_argsort_u64f32(void* index, void* begin, uint64_t len);
    void parallel_argsort_u64f64(void* index, void* begin, uint64_t len);

    void parallel_stable_argsort_u64v(void* index, void* begin, uint64_t len, uint64_t size, void* compare);

    void parallel_stable_argsort_u64i8(void* index, void* begin, uint64_t len);
    void parallel_stable_argsort_u64u8(void* index, void* begin, uint64_t len);
    void parallel_stable_argsort_u64i16(void* index, void* begin, uint64_t len);
    void parallel_stable_argsort_u64u16(void* index, void* begin, uint64_t len);
    void parallel_stable_argsort_u64i32(void* index, void* begin, uint64_t len);
    void parallel_stable_argsort_u64u32(void* index, void* begin, uint64_t len);
    void parallel_stable_argsort_u64i64(void* index, void* begin, uint64_t len);
    void parallel_stable_argsort_u64u64(void* index, void* begin, uint64_t len);

    void parallel_stable_argsort_u64f32(void* index, void* begin, uint64_t len);
    void parallel_stable_argsort_u64f64(void* index, void* begin, uint64_t len);

    void set_number_of_threads(uint64_t threads)
    {
        utils::set_threads_num(threads);
    }
}

PyMODINIT_FUNC PyInit_concurrent_sort()
{
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, /* m_base */
        "sort",                /* m_name */
        "No docs",             /* m_doc */
        -1,                    /* m_size */
        NULL,                  /* m_methods */
        NULL,                  /* m_reload */
        NULL,                  /* m_traverse */
        NULL,                  /* m_clear */
        utils::finalize_tbb,   /* m_free */
    };

    PyObject* m = PyModule_Create(&moduledef);
    if (m == NULL)
    {
        return NULL;
    }

#define REGISTER(func) PyObject_SetAttrString(m, #func, PyLong_FromVoidPtr((void*)(&func)));
    REGISTER(parallel_sort)

    REGISTER(parallel_sort_i8)
    REGISTER(parallel_sort_u8)
    REGISTER(parallel_sort_i16)
    REGISTER(parallel_sort_u16)
    REGISTER(parallel_sort_i32)
    REGISTER(parallel_sort_u32)
    REGISTER(parallel_sort_i64)
    REGISTER(parallel_sort_u64)

    REGISTER(parallel_sort_f32)
    REGISTER(parallel_sort_f64)

    REGISTER(parallel_stable_sort)

    REGISTER(parallel_stable_sort_i8)
    REGISTER(parallel_stable_sort_u8)
    REGISTER(parallel_stable_sort_i16)
    REGISTER(parallel_stable_sort_u16)
    REGISTER(parallel_stable_sort_i32)
    REGISTER(parallel_stable_sort_u32)
    REGISTER(parallel_stable_sort_i64)
    REGISTER(parallel_stable_sort_u64)

    REGISTER(parallel_stable_sort_f32)
    REGISTER(parallel_stable_sort_f64)

    REGISTER(parallel_argsort_u64v)

    REGISTER(parallel_argsort_u64i8)
    REGISTER(parallel_argsort_u64u8)
    REGISTER(parallel_argsort_u64i16)
    REGISTER(parallel_argsort_u64u16)
    REGISTER(parallel_argsort_u64i32)
    REGISTER(parallel_argsort_u64u32)
    REGISTER(parallel_argsort_u64i64)
    REGISTER(parallel_argsort_u64u64)

    REGISTER(parallel_argsort_u64f32)
    REGISTER(parallel_argsort_u64f64)

    REGISTER(parallel_stable_argsort_u64v)

    REGISTER(parallel_stable_argsort_u64i8)
    REGISTER(parallel_stable_argsort_u64u8)
    REGISTER(parallel_stable_argsort_u64i16)
    REGISTER(parallel_stable_argsort_u64u16)
    REGISTER(parallel_stable_argsort_u64i32)
    REGISTER(parallel_stable_argsort_u64u32)
    REGISTER(parallel_stable_argsort_u64i64)
    REGISTER(parallel_stable_argsort_u64u64)

    REGISTER(parallel_stable_argsort_u64f32)
    REGISTER(parallel_stable_argsort_u64f64)

    REGISTER(set_number_of_threads)
#undef REGISTER
    return m;
}

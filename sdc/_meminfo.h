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

#ifndef _MEMINFO_INCLUDED
#define _MEMINFO_INCLUDED

// #include "_import_py.h"
// #include <numba/runtime/nrt.h>

// /* Import MemInfo_* from numba.runtime._nrt_python.
//  */
// static void *
// import_meminfo_func(const char * func) {
// #define CHECK(expr, msg) if(!(expr)){std::cerr << msg << std::endl; PyGILState_Release(gilstate); return NULL;}
//     auto gilstate = PyGILState_Ensure();
//     PyObject * helperdct = import_sym("numba.runtime._nrt_python", "c_helpers");
//     CHECK(helperdct, "getting numba.runtime._nrt_python.c_helpers failed");
//     /* helperdct[func] */
//     PyObject * mi_rel_fn = PyDict_GetItemString(helperdct, func);
//     CHECK(mi_rel_fn, "getting meminfo func failed");
//     void * fnptr = PyLong_AsVoidPtr(mi_rel_fn);

//     Py_XDECREF(helperdct);
//     PyGILState_Release(gilstate);
//     return fnptr;
// #undef CHECK
// }

// typedef void (*MemInfo_release_type)(void*);
// typedef MemInfo* (*MemInfo_alloc_aligned_type)(size_t size, unsigned align);
// typedef void* (*MemInfo_data_type)(MemInfo* mi);

// ******** copied from Numba
// TODO: make Numba C library
typedef void (*NRT_dtor_function)(void* ptr, size_t size, void* info);
struct MemInfo
{
    size_t refct;
    NRT_dtor_function dtor;
    void* dtor_info;
    void* data;
    size_t size; /* only used for NRT allocated memory */
};

typedef struct MemInfo NRT_MemInfo;

void nrt_debug_print(const char* fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

#if 0
#define NRT_Debug(X) X
#else
#define NRT_Debug(X)                                                                                                   \
    if (0)                                                                                                             \
    {                                                                                                                  \
        X;                                                                                                             \
    }
#endif

#if !defined MIN
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#endif

void NRT_Free(void* ptr)
{
    NRT_Debug(nrt_debug_print("NRT_Free %p\n", ptr));
    free(ptr);
    // TheMSys.allocator.free(ptr);
    // TheMSys.atomic_inc(&TheMSys.stats_free);
}

void NRT_MemInfo_destroy(NRT_MemInfo* mi)
{
    NRT_Free(mi);
    // TheMSys.atomic_inc(&TheMSys.stats_mi_free);
}

void NRT_MemInfo_call_dtor(NRT_MemInfo* mi)
{
    NRT_Debug(nrt_debug_print("NRT_MemInfo_call_dtor %p\n", mi));
    if (mi->dtor) // && !TheMSys.shutting)
        /* We have a destructor and the system is not shutting down */
        mi->dtor(mi->data, mi->size, mi->dtor_info);
    /* Clear and release MemInfo */
    NRT_MemInfo_destroy(mi);
}

void* NRT_Allocate(size_t size)
{
    // void *ptr = TheMSys.allocator.malloc(size);
    void* ptr = malloc(size);
    NRT_Debug(nrt_debug_print("NRT_Allocate bytes=%zu ptr=%p\n", size, ptr));
    // TheMSys.atomic_inc(&TheMSys.stats_alloc);
    return ptr;
}

static void* nrt_allocate_meminfo_and_data(size_t size, NRT_MemInfo** mi_out)
{
    NRT_MemInfo* mi;
    char* base = (char*)NRT_Allocate(sizeof(NRT_MemInfo) + size);
    mi = (NRT_MemInfo*)base;
    *mi_out = mi;
    return base + sizeof(NRT_MemInfo);
}

void NRT_MemInfo_init(NRT_MemInfo* mi, void* data, size_t size, NRT_dtor_function dtor, void* dtor_info)
{
    mi->refct = 1; /* starts with 1 refct */
    mi->dtor = dtor;
    mi->dtor_info = dtor_info;
    mi->data = data;
    mi->size = size;
    /* Update stats */
    // TheMSys.atomic_inc(&TheMSys.stats_mi_alloc);
}

static void nrt_internal_dtor_safe(void* ptr, size_t size, void* info)
{
    NRT_Debug(nrt_debug_print("nrt_internal_dtor_safe %p, %p\n", ptr, info));
    /* See NRT_MemInfo_alloc_safe() */
    memset(ptr, 0xDE, MIN(size, 256));
}

static void nrt_internal_custom_dtor_safe(void* ptr, size_t size, void* info)
{
    NRT_dtor_function dtor = (NRT_dtor_function)info;
    NRT_Debug(nrt_debug_print("nrt_internal_custom_dtor_safe %p, %p\n", ptr, info));
    if (dtor)
    {
        dtor(ptr, size, NULL);
    }

    nrt_internal_dtor_safe(ptr, size, NULL);
}

NRT_MemInfo* NRT_MemInfo_alloc_dtor_safe(size_t size, NRT_dtor_function dtor)
{
    NRT_MemInfo* mi;
    void* data = nrt_allocate_meminfo_and_data(size, &mi);
    /* Only fill up a couple cachelines with debug markers, to minimize
       overhead. */
    memset(data, 0xCB, MIN(size, 256));
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc_dtor_safe %p %zu\n", data, size));
    NRT_MemInfo_init(mi, data, size, nrt_internal_custom_dtor_safe, (void*)dtor);
    return mi;
}

NRT_MemInfo* NRT_MemInfo_alloc_safe(size_t size)
{
    return NRT_MemInfo_alloc_dtor_safe(size, NULL);
}

#endif // #ifndef _MEMINFO_INCLUDED

#ifndef _MEMINFO_INCLUDED
#define _MEMINFO_INCLUDED

#include "_import_py.h"
#include <numba/runtime/nrt.h>

/* Import MemInfo_* from numba.runtime._nrt_python.
 */
static void *
import_meminfo_func(const char * func) {
#define CHECK(expr, msg) if(!(expr)){std::cerr << msg << std::endl; PyGILState_Release(gilstate); return NULL;}
    auto gilstate = PyGILState_Ensure();
    PyObject * helperdct = import_sym("numba.runtime._nrt_python", "c_helpers");
    CHECK(helperdct, "getting numba.runtime._nrt_python.c_helpers failed");
    /* helperdct[func] */
    PyObject * mi_rel_fn = PyDict_GetItemString(helperdct, func);
    CHECK(mi_rel_fn, "getting meminfo func failed");
    void * fnptr = PyLong_AsVoidPtr(mi_rel_fn);

    Py_XDECREF(helperdct);
    PyGILState_Release(gilstate);
    return fnptr;
#undef CHECK
}

typedef void (*MemInfo_release_type)(void*);
typedef MemInfo* (*MemInfo_alloc_aligned_type)(size_t size, unsigned align);
typedef void* (*MemInfo_data_type)(MemInfo* mi);

#endif

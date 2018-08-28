#ifndef _MEMINFO_INCLUDED
#define _MEMINFO_INCLUDED

#include <Python.h>
#include <numba/runtime/nrt.h>

/* Import MemInfo_* from numba.runtime._nrt_python.
 */
static void *
import_meminfo_func(const char * func) {
#define CHECK(expr, msg) if(!(expr)){std::cerr << msg << std::endl; PyGILState_Release(gilstate); return NULL;}
    PyObject *nrtmod = NULL;
    PyObject *helperdct = NULL;
    PyObject *mi_rel_fn = NULL;
    void *fnptr = NULL;
    auto gilstate = PyGILState_Ensure();
    /* from numba.runtime import _nrt_python */
    nrtmod = PyImport_ImportModule("numba.runtime._nrt_python");
    CHECK(nrtmod, "importing numba.runtime._nrt_python failed");
    /* helperdct = _nrt_python.c_helpers */
    helperdct = PyObject_GetAttrString(nrtmod, "c_helpers");
    CHECK(helperdct, "getting numba.runtime._nrt_python.c_helpers failed");
    /* helperdct[func] */
    mi_rel_fn = PyDict_GetItemString(helperdct, func);
    CHECK(mi_rel_fn, "getting meminfo func failed");
    fnptr = PyLong_AsVoidPtr(mi_rel_fn);

    PyGILState_Release(gilstate);
    Py_XDECREF(nrtmod);
    Py_XDECREF(helperdct);
    return fnptr;
}

typedef void (*MemInfo_release_type)(void*);
typedef MemInfo* (*MemInfo_alloc_aligned_type)(size_t size, unsigned align);
typedef void* (*MemInfo_data_type)(MemInfo* mi);

#endif

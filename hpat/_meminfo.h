#ifndef _MEMINFO_INCLUDED
#define _MEMINFO_INCLUDED

#include <Python.h>
#include <numba/runtime/nrt.h>

/* Import MemInfo_* from numba.runtime._nrt_python.
 */
static void *
import_meminfo_func(const char * func) {
    PyObject *nrtmod = NULL;
    PyObject *helperdct = NULL;
    PyObject *mi_rel_fn = NULL;
    void *fnptr = NULL;
    /* from numba.runtime import _nrt_python */
    nrtmod = PyImport_ImportModule("numba.runtime._nrt_python");
    if (!nrtmod) goto cleanup;
    /* helperdct = _nrt_python.c_helpers */
    helperdct = PyObject_GetAttrString(nrtmod, "c_helpers");
    if (!helperdct) goto cleanup;
    /* helperdct[func] */
    mi_rel_fn = PyDict_GetItemString(helperdct, func);
    if (!mi_rel_fn) goto cleanup;
    fnptr = PyLong_AsVoidPtr(mi_rel_fn);

cleanup:
    Py_XDECREF(nrtmod);
    Py_XDECREF(helperdct);
    return fnptr;
}

typedef void (*MemInfo_release_type)(void*);
typedef MemInfo* (*MemInfo_alloc_aligned_type)(size_t size, unsigned align);
typedef void* (*MemInfo_data_type)(MemInfo* mi);

#endif

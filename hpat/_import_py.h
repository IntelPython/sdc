#ifndef _IMPORT_PY_INCLUDED
#define _IMPORT_PY_INCLUDED

#include <Python.h>

/* Import 'sym' from module 'module'.
 */
static PyObject *
import_sym(const char * module, const char * sym) __UNUSED__ {
#define CHECK(expr, msg) if(!(expr)){std::cerr << msg << std::endl; PyGILState_Release(gilstate); return NULL;}
    PyObject *mod = NULL;
    PyObject *func = NULL;
    auto gilstate = PyGILState_Ensure();

    mod = PyImport_ImportModule(module);
    CHECK(mod, "importing failed");
    func = PyObject_GetAttrString(mod, sym);
    CHECK(func, "getting symbol from module failed");

    Py_XDECREF(mod);
    PyGILState_Release(gilstate);

    return func;
#undef CHECK
}

#endif // _IMPORT_PY_INCLUDED

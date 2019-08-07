#include <iostream>
#include <Python.h>
#include "_hpat_sort.h"

PyMODINIT_FUNC PyInit_chiframes(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "chiframes", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "timsort",
                        PyLong_FromVoidPtr((void*)(&__hpat_timsort)));

    return m;
}

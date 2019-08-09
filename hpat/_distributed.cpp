#include "_distributed.h"

PyMODINIT_FUNC PyInit_hdist(void)
{
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "hdist",
        "No docs",
        -1,
        NULL,
    };

    m = PyModule_Create(&moduledef);
    if (m == NULL)
    {
        return NULL;
    }

    PyObject_SetAttrString(m, "hpat_dist_get_start", PyLong_FromVoidPtr((void*)(&hpat_dist_get_start)));
    PyObject_SetAttrString(m, "hpat_dist_get_end", PyLong_FromVoidPtr((void*)(&hpat_dist_get_end)));
    PyObject_SetAttrString(m, "hpat_dist_get_node_portion", PyLong_FromVoidPtr((void*)(&hpat_dist_get_node_portion)));
    PyObject_SetAttrString(m, "hpat_dist_get_item_pointer", PyLong_FromVoidPtr((void*)(&hpat_dist_get_item_pointer)));
    PyObject_SetAttrString(m, "hpat_get_dummy_ptr", PyLong_FromVoidPtr((void*)(&hpat_get_dummy_ptr)));

    return m;
}

#include <Python.h>
#include <unordered_map>

void* init_dict_int_int();

PyMODINIT_FUNC PyInit_hdict_ext(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "hdict_ext", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "init_dict_int_int",
                                        PyLong_FromVoidPtr((void*)(&init_dict_int_int)));
    return m;
}

void* init_dict_int_int()
{
    return new std::unordered_map<int,int>();
}

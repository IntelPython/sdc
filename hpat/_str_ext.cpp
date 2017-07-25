#include <Python.h>
#include <string>
#include <iostream>

void* init_string(char*, int64_t);
const char* get_c_str(std::string* s);

PyMODINIT_FUNC PyInit_hstr_ext(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "hstr_ext", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "init_string",
                            PyLong_FromVoidPtr((void*)(&init_string)));
    PyObject_SetAttrString(m, "get_c_str",
                            PyLong_FromVoidPtr((void*)(&get_c_str)));
    return m;
}

void* init_string(char* in_str, int64_t size)
{
    // std::cout<<in_str<<std::endl;
    // std::cout<<size<<std::endl;
    return new std::string(in_str, size);
}

const char* get_c_str(std::string* s)
{
    // printf("in get %s\n", s->c_str());
    return s->c_str();
}

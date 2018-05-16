#include <Python.h>
#include <unordered_set>
#include <iostream>
#include <limits>
#include <string>


std::unordered_set<std::string>* init_set_string();
void insert_set_string(std::unordered_set<std::string>* str_set, std::string* val);
int64_t len_set_string(std::unordered_set<std::string>* str_set);


PyMODINIT_FUNC PyInit_hset_ext(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "hset_ext", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "init_set_string",
                            PyLong_FromVoidPtr((void*)(&init_set_string)));
    PyObject_SetAttrString(m, "insert_set_string",
                            PyLong_FromVoidPtr((void*)(&insert_set_string)));
    PyObject_SetAttrString(m, "len_set_string",
                            PyLong_FromVoidPtr((void*)(&len_set_string)));

    return m;
}

std::unordered_set<std::string>* init_set_string()
{
    return new std::unordered_set<std::string>();
}

void insert_set_string(std::unordered_set<std::string>* str_set, std::string* val)
{
    str_set->insert(*val);
}

int64_t len_set_string(std::unordered_set<std::string>* str_set)
{
    return str_set->size();
}

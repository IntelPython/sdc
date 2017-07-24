#include <Python.h>
#include <unordered_map>
#include <iostream>

void* init_dict_int_int();
void dict_int_int_setitem(std::unordered_map<int64_t, int64_t>* m, int64_t index, int64_t value);
void dict_int_int_print(std::unordered_map<int64_t, int64_t>* m);
int64_t dict_int_int_get(std::unordered_map<int64_t, int64_t>* m, int64_t index, int64_t default_val);
int64_t dict_int_int_getitem(std::unordered_map<int64_t, int64_t>* m, int64_t index);
int64_t dict_int_int_pop(std::unordered_map<int64_t, int64_t>* m, int64_t index);
void* dict_int_int_keys(std::unordered_map<int64_t, int64_t>* m);
int64_t dict_int_int_min(std::unordered_map<int64_t, int64_t>* m);
int64_t dict_int_int_max(std::unordered_map<int64_t, int64_t>* m);
bool dict_int_int_not_empty(std::unordered_map<int64_t, int64_t>* m);

PyMODINIT_FUNC PyInit_hdict_ext(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "hdict_ext", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "init_dict_int_int",
                            PyLong_FromVoidPtr((void*)(&init_dict_int_int)));
    PyObject_SetAttrString(m, "dict_int_int_setitem",
                            PyLong_FromVoidPtr((void*)(&dict_int_int_setitem)));
    PyObject_SetAttrString(m, "dict_int_int_print",
                            PyLong_FromVoidPtr((void*)(&dict_int_int_print)));
    PyObject_SetAttrString(m, "dict_int_int_get",
                            PyLong_FromVoidPtr((void*)(&dict_int_int_get)));
    PyObject_SetAttrString(m, "dict_int_int_getitem",
                            PyLong_FromVoidPtr((void*)(&dict_int_int_getitem)));
    PyObject_SetAttrString(m, "dict_int_int_pop",
                            PyLong_FromVoidPtr((void*)(&dict_int_int_pop)));
    PyObject_SetAttrString(m, "dict_int_int_keys",
                            PyLong_FromVoidPtr((void*)(&dict_int_int_keys)));
    PyObject_SetAttrString(m, "dict_int_int_min",
                            PyLong_FromVoidPtr((void*)(&dict_int_int_min)));
    PyObject_SetAttrString(m, "dict_int_int_max",
                            PyLong_FromVoidPtr((void*)(&dict_int_int_max)));
    PyObject_SetAttrString(m, "dict_int_int_not_empty",
                            PyLong_FromVoidPtr((void*)(&dict_int_int_not_empty)));
    return m;
}

void* init_dict_int_int()
{
    return new std::unordered_map<int64_t, int64_t>();
}

void dict_int_int_setitem(std::unordered_map<int64_t, int64_t>* m, int64_t index, int64_t value)
{
    (*m)[index] = value;
    return;
}

void dict_int_int_print(std::unordered_map<int64_t, int64_t>* m)
{
    // TODO: return python string and print in native mode
    for (auto& x: *m) {
        std::cout << x.first << ": " << x.second << std::endl;
    }
    return;
}

int64_t dict_int_int_get(std::unordered_map<int64_t, int64_t>* m, int64_t index, int64_t default_val)
{
    auto val = m->find(index);
    if (val==m->end())
        return default_val;
    return (*val).second;
}

int64_t dict_int_int_getitem(std::unordered_map<int64_t, int64_t>* m, int64_t index)
{
    return m->at(index);
}

int64_t dict_int_int_pop(std::unordered_map<int64_t, int64_t>* m, int64_t index)
{
    int64_t val = m->at(index);
    m->erase(index);
    return val;
}

void* dict_int_int_keys(std::unordered_map<int64_t, int64_t>* m)
{
    // TODO: return actual iterator
    return m;
}

int64_t dict_int_int_min(std::unordered_map<int64_t, int64_t>* m)
{
    // TODO: use actual iterator
    int64_t res = std::numeric_limits<int64_t>::max();
    for (auto& x: *m) {
        if (x.first<res)
            res = x.first;
    }
    return res;
}

int64_t dict_int_int_max(std::unordered_map<int64_t, int64_t>* m)
{
    // TODO: use actual iterator
    int64_t res = std::numeric_limits<int64_t>::min();
    for (auto& x: *m) {
        if (x.first>res)
            res = x.first;
    }
    return res;
}

bool dict_int_int_not_empty(std::unordered_map<int64_t, int64_t>* m)
{
    return !m->empty();
}

#include <Python.h>
#include <unordered_map>
#include <iostream>
#include <limits>
#include <string>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>

void* init_dict_int_int();
void dict_int_int_setitem(std::unordered_map<int64_t, int64_t>* m, int64_t index, int64_t value);
void dict_int_int_print(std::unordered_map<int64_t, int64_t>* m);
int64_t dict_int_int_get(std::unordered_map<int64_t, int64_t>* m, int64_t index, int64_t default_val);
int64_t dict_int_int_getitem(std::unordered_map<int64_t, int64_t>* m, int64_t index);
int64_t dict_int_int_pop(std::unordered_map<int64_t, int64_t>* m, int64_t index);
void* dict_int_int_keys(std::unordered_map<int64_t, int64_t>* m);
int64_t dict_int_int_min(std::unordered_map<int64_t, int64_t>* m);
int64_t dict_int_int_max(std::unordered_map<int64_t, int64_t>* m);
bool dict_int_int_in(std::unordered_map<int64_t, int64_t>* m, int64_t val);
bool dict_int_int_not_empty(std::unordered_map<int64_t, int64_t>* m);

// -- int32 versions --
void* init_dict_int32_int32();
void dict_int32_int32_setitem(std::unordered_map<int, int>* m, int index, int value);
void dict_int32_int32_print(std::unordered_map<int, int>* m);
int dict_int32_int32_get(std::unordered_map<int, int>* m, int index, int default_val);
int dict_int32_int32_getitem(std::unordered_map<int, int>* m, int index);
int dict_int32_int32_pop(std::unordered_map<int, int>* m, int index);
void* dict_int32_int32_keys(std::unordered_map<int, int>* m);
int dict_int32_int32_min(std::unordered_map<int, int>* m);
int dict_int32_int32_max(std::unordered_map<int, int>* m);
bool dict_int32_int32_not_empty(std::unordered_map<int, int>* m);

#define StringType_t std::string
#define bool_t bool

#define C_TYPE(a) BOOST_PP_CAT(a,_t)


#define DEF_DICT(key_typ, val_typ) \
/* create dictionary */ \
void* BOOST_PP_CAT(init_dict_##key_typ, _##val_typ)() { \
    return new std::unordered_map<C_TYPE(key_typ), C_TYPE(val_typ)>(); \
}

#define DEC_MOD_METHOD(func) PyObject_SetAttrString(m, BOOST_PP_STRINGIZE(func), PyLong_FromVoidPtr((void*)(&func)));

#define DEC_DICT_MOD(key_typ, val_typ) DEC_MOD_METHOD(BOOST_PP_CAT(init_dict_##key_typ, _##val_typ))


DEF_DICT(int64, int64)
DEF_DICT(StringType, int64)

PyMODINIT_FUNC PyInit_hdict_ext(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "hdict_ext", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    DEC_DICT_MOD(int64, int64)
    DEC_DICT_MOD(StringType, int64)

    DEC_MOD_METHOD(init_dict_int_int)

    // PyObject_SetAttrString(m, "init_dict_int_int",
    //                         PyLong_FromVoidPtr((void*)(&init_dict_int_int)));
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
    PyObject_SetAttrString(m, "dict_int_int_in",
                            PyLong_FromVoidPtr((void*)(&dict_int_int_in)));
    PyObject_SetAttrString(m, "dict_int_int_not_empty",
                            PyLong_FromVoidPtr((void*)(&dict_int_int_not_empty)));
    // ---- int32 versions ----
    PyObject_SetAttrString(m, "init_dict_int32_int32",
                            PyLong_FromVoidPtr((void*)(&init_dict_int32_int32)));
    PyObject_SetAttrString(m, "dict_int32_int32_setitem",
                            PyLong_FromVoidPtr((void*)(&dict_int32_int32_setitem)));
    PyObject_SetAttrString(m, "dict_int32_int32_print",
                            PyLong_FromVoidPtr((void*)(&dict_int32_int32_print)));
    PyObject_SetAttrString(m, "dict_int32_int32_get",
                            PyLong_FromVoidPtr((void*)(&dict_int32_int32_get)));
    PyObject_SetAttrString(m, "dict_int32_int32_getitem",
                            PyLong_FromVoidPtr((void*)(&dict_int32_int32_getitem)));
    PyObject_SetAttrString(m, "dict_int32_int32_pop",
                            PyLong_FromVoidPtr((void*)(&dict_int32_int32_pop)));
    PyObject_SetAttrString(m, "dict_int32_int32_keys",
                            PyLong_FromVoidPtr((void*)(&dict_int32_int32_keys)));
    PyObject_SetAttrString(m, "dict_int32_int32_min",
                            PyLong_FromVoidPtr((void*)(&dict_int32_int32_min)));
    PyObject_SetAttrString(m, "dict_int32_int32_max",
                            PyLong_FromVoidPtr((void*)(&dict_int32_int32_max)));
    PyObject_SetAttrString(m, "dict_int32_int32_not_empty",
                            PyLong_FromVoidPtr((void*)(&dict_int32_int32_not_empty)));
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

bool dict_int_int_in(std::unordered_map<int64_t, int64_t>* m, int64_t val)
{
    return (m->find(val) != m->end());
}

bool dict_int_int_not_empty(std::unordered_map<int64_t, int64_t>* m)
{
    return !m->empty();
}


// --------- int32 versions ------
void* init_dict_int32_int32()
{
    return new std::unordered_map<int, int>();
}

void dict_int32_int32_setitem(std::unordered_map<int, int>* m, int index, int value)
{
    (*m)[index] = value;
    return;
}

void dict_int32_int32_print(std::unordered_map<int, int>* m)
{
    // TODO: return python string and print in native mode
    for (auto& x: *m) {
        std::cout << x.first << ": " << x.second << std::endl;
    }
    return;
}

int dict_int32_int32_get(std::unordered_map<int, int>* m, int index, int default_val)
{
    auto val = m->find(index);
    if (val==m->end())
        return default_val;
    return (*val).second;
}

int dict_int32_int32_getitem(std::unordered_map<int, int>* m, int index)
{
    return m->at(index);
}

int dict_int32_int32_pop(std::unordered_map<int, int>* m, int index)
{
    int val = m->at(index);
    m->erase(index);
    return val;
}

void* dict_int32_int32_keys(std::unordered_map<int, int>* m)
{
    // TODO: return actual iterator
    return m;
}

int dict_int32_int32_min(std::unordered_map<int, int>* m)
{
    // TODO: use actual iterator
    int res = std::numeric_limits<int>::max();
    for (auto& x: *m) {
        if (x.first<res)
            res = x.first;
    }
    return res;
}

int dict_int32_int32_max(std::unordered_map<int, int>* m)
{
    // TODO: use actual iterator
    int res = std::numeric_limits<int>::min();
    for (auto& x: *m) {
        if (x.first>res)
            res = x.first;
    }
    return res;
}

bool dict_int32_int32_not_empty(std::unordered_map<int, int>* m)
{
    return !m->empty();
}

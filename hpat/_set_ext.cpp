#include <Python.h>
#include <unordered_set>
#include <iostream>
#include <limits>
#include <string>


std::unordered_set<std::string>* init_set_string();
void insert_set_string(std::unordered_set<std::string>* str_set, std::string* val);
int64_t len_set_string(std::unordered_set<std::string>* str_set);
int64_t num_total_chars_set_string(std::unordered_set<std::string>* str_set);
void populate_str_arr_from_set(std::unordered_set<std::string>* str_set,
                                uint32_t *offsets, char *data);

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
    PyObject_SetAttrString(m, "num_total_chars_set_string",
                            PyLong_FromVoidPtr((void*)(&num_total_chars_set_string)));
    PyObject_SetAttrString(m, "populate_str_arr_from_set",
                            PyLong_FromVoidPtr((void*)(&populate_str_arr_from_set)));

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

int64_t num_total_chars_set_string(std::unordered_set<std::string>* str_set)
{
    int64_t total_len = 0;
    for (auto it=str_set->cbegin(); it != str_set->cend(); ++it)
        total_len += (*it).size();
    return total_len;
}

void populate_str_arr_from_set(std::unordered_set<std::string>* str_set,
                                uint32_t *offsets, char *data)
{
    uint32_t curr_data_ind = 0;
    uint32_t index = 0;
    for (auto it=str_set->cbegin(); it != str_set->cend(); ++it)
    {
        uint32_t len = (*it).length();
        // std::cout << "start " << start << " len " << len << std::endl;
        memcpy(&data[curr_data_ind], (*it).c_str(), len);
        offsets[index] = curr_data_ind;
        curr_data_ind += len;
        index++;
    }
    offsets[index] = curr_data_ind;
}

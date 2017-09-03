#include <Python.h>
#include <string>
#include <iostream>
#include <vector>

void* init_string(char*, int64_t);
void* init_string_const(char* in_str);
const char* get_c_str(std::string* s);
void* str_concat(std::string* s1, std::string* s2);
bool str_equal(std::string* s1, std::string* s2);
void* str_split(std::string* str, std::string* sep, int64_t *size);

PyMODINIT_FUNC PyInit_hstr_ext(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "hstr_ext", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "init_string",
                            PyLong_FromVoidPtr((void*)(&init_string)));
    PyObject_SetAttrString(m, "init_string_const",
                            PyLong_FromVoidPtr((void*)(&init_string_const)));
    PyObject_SetAttrString(m, "get_c_str",
                            PyLong_FromVoidPtr((void*)(&get_c_str)));
    PyObject_SetAttrString(m, "str_concat",
                            PyLong_FromVoidPtr((void*)(&str_concat)));
    PyObject_SetAttrString(m, "str_equal",
                            PyLong_FromVoidPtr((void*)(&str_equal)));
    PyObject_SetAttrString(m, "str_split",
                            PyLong_FromVoidPtr((void*)(&str_split)));
    return m;
}

void* init_string(char* in_str, int64_t size)
{
    // std::cout<<"init str: "<<in_str<<" "<<size<<std::endl;
    return new std::string(in_str, size);
}

void* init_string_const(char* in_str)
{
    // std::cout<<"init str: "<<in_str<<" "<<size<<std::endl;
    return new std::string(in_str);
}

const char* get_c_str(std::string* s)
{
    // printf("in get %s\n", s->c_str());
    return s->c_str();
}

void* str_concat(std::string* s1, std::string* s2)
{
    // printf("in concat %s %s\n", s1->c_str(), s2->c_str());
    std::string* res = new std::string((*s1)+(*s2));
    return res;
}

bool str_equal(std::string* s1, std::string* s2)
{
    // printf("in str_equal %s %s\n", s1->c_str(), s2->c_str());
    return s1->compare(*s2)==0;
}

void* str_split(std::string* str, std::string* sep, int64_t *size)
{
    // std::cout << *str << " " << *sep << std::endl;
    std::vector<std::string*> res;

    size_t last = 0;
    size_t next = 0;
    while ((next = str->find(*sep, last)) != std::string::npos) {
        std::string *token = new std::string(str->substr(last, next-last));
        res.push_back(token);
        last = next + 1;
    }
    std::string *token = new std::string(str->substr(last));
    res.push_back(token);
    *size = res.size();
    // for(int i=0; i<*size; i++)
    //    std::cout<<*(res[i])<<std::endl;
    // TODO: avoid extra copy
    void* out = new void*[*size];
    memcpy(out, res.data(), (*size)*sizeof(void*));
    // std::cout<< *(((std::string**)(out))[1])<<std::endl;
    return out;
}

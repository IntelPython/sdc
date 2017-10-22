#include <Python.h>
#include <string>
#include <iostream>
#include <vector>


struct str_arr_payload {
    int64_t size;
    int32_t *offsets;
    char* data;
};

void* init_string(char*, int64_t);
void* init_string_const(char* in_str);
void dtor_string(std::string** in_str, int64_t size, void* in);
void dtor_string_array(str_arr_payload* in_str, int64_t size, void* in);
const char* get_c_str(std::string* s);
void* str_concat(std::string* s1, std::string* s2);
bool str_equal(std::string* s1, std::string* s2);
void* str_split(std::string* str, std::string* sep, int64_t *size);
void* str_substr_int(std::string* str, int64_t index);
int64_t str_to_int64(std::string* str);
double str_to_float64(std::string* str);
int64_t get_str_len(std::string* str);
void allocate_string_array(uint32_t **offsets, char **data, int64_t num_strings,
                                                            int64_t total_size);

void setitem_string_array(uint32_t *offsets, char *data, std::string* str,
                                                                int64_t index);
char* getitem_string_array(uint32_t *offsets, char *data, int64_t index);
void* getitem_string_array_std(uint32_t *offsets, char *data, int64_t index);
void print_int(int64_t val);

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
    PyObject_SetAttrString(m, "dtor_string",
                            PyLong_FromVoidPtr((void*)(&dtor_string)));
    PyObject_SetAttrString(m, "dtor_string_array",
                            PyLong_FromVoidPtr((void*)(&dtor_string_array)));
    PyObject_SetAttrString(m, "get_c_str",
                            PyLong_FromVoidPtr((void*)(&get_c_str)));
    PyObject_SetAttrString(m, "str_concat",
                            PyLong_FromVoidPtr((void*)(&str_concat)));
    PyObject_SetAttrString(m, "str_equal",
                            PyLong_FromVoidPtr((void*)(&str_equal)));
    PyObject_SetAttrString(m, "str_split",
                            PyLong_FromVoidPtr((void*)(&str_split)));
    PyObject_SetAttrString(m, "str_substr_int",
                            PyLong_FromVoidPtr((void*)(&str_substr_int)));
    PyObject_SetAttrString(m, "str_to_int64",
                            PyLong_FromVoidPtr((void*)(&str_to_int64)));
    PyObject_SetAttrString(m, "str_to_float64",
                            PyLong_FromVoidPtr((void*)(&str_to_float64)));
    PyObject_SetAttrString(m, "get_str_len",
                            PyLong_FromVoidPtr((void*)(&get_str_len)));
    PyObject_SetAttrString(m, "allocate_string_array",
                            PyLong_FromVoidPtr((void*)(&allocate_string_array)));
    PyObject_SetAttrString(m, "setitem_string_array",
                            PyLong_FromVoidPtr((void*)(&setitem_string_array)));
    PyObject_SetAttrString(m, "getitem_string_array",
                            PyLong_FromVoidPtr((void*)(&getitem_string_array)));
    PyObject_SetAttrString(m, "getitem_string_array_std",
                            PyLong_FromVoidPtr((void*)(&getitem_string_array_std)));
    PyObject_SetAttrString(m, "print_int",
                            PyLong_FromVoidPtr((void*)(&print_int)));
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

void dtor_string(std::string** in_str, int64_t size, void* info)
{
    printf("dtor size: %d\n", size); fflush(stdout);
    // std::cout<<"del str: "<< (*in_str)->c_str() <<std::endl;
    // delete (*in_str);
    return;
}

void dtor_string_array(str_arr_payload* in_str_arr, int64_t size, void* in)
{
    // printf("str arr dtor size: %lld\n", in_str_arr->size);
    // printf("num chars: %d\n", in_str_arr->offsets[in_str_arr->size]);
    delete[] in_str_arr->offsets;
    delete[] in_str_arr->data;
    return;
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

void* str_substr_int(std::string* str, int64_t index)
{
    return new std::string(*str, index, 1);
}

int64_t str_to_int64(std::string* str)
{
    return std::stoll(*str);
}

double str_to_float64(std::string* str)
{
    return std::stod(*str);
}

int64_t get_str_len(std::string* str)
{
    // std::cout << "str len called: " << *str << " " << str->length()<<std::endl;
    return str->length();
}

void allocate_string_array(uint32_t **offsets, char **data, int64_t num_strings,
                                                            int64_t total_size)
{
    // std::cout << "allocating string array: " << num_strings << " " <<
    //                                                 total_size << std::endl;
    *offsets = new uint32_t[num_strings+1];
    *data = new char[total_size];
    // *data = (char*) new std::string("gggg");
    return;
}

void setitem_string_array(uint32_t *offsets, char *data, std::string* str,
                                                                int64_t index)
{
    // std::cout << "setitem str: " << *str << " " << index << std::endl;
    if (index==0)
        offsets[index] = 0;
    uint32_t start = offsets[index];
    uint32_t len = str->length();
    // std::cout << "start " << start << " len " << len << std::endl;
    memcpy(&data[start], str->c_str(), len);
    offsets[index+1] = start+len;
    return;
}

char* getitem_string_array(uint32_t *offsets, char *data, int64_t index)
{
    // printf("getitem string arr index: %d offsets: %d %d", index,
    //                                  offsets[index], offsets[index+1]);
    uint32_t size = offsets[index+1]-offsets[index]+1;
    uint32_t start = offsets[index];
    char* res = new char[size];
    res[size-1] = '\0';
    memcpy(res, &data[start], size-1);
    // printf(" res %s\n", res);
    return res;
}

void* getitem_string_array_std(uint32_t *offsets, char *data, int64_t index)
{
    // printf("getitem string arr index: %d offsets: %d %d", index,
    //                                  offsets[index], offsets[index+1]);
    uint32_t size = offsets[index+1]-offsets[index];
    uint32_t start = offsets[index];
    return new std::string(&data[start], size);
}

void print_int(int64_t val)
{
    printf("%ld\n", val);
}

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <string>
#include <iostream>
#include <vector>
#include <boost/algorithm/string/replace.hpp>

#ifdef USE_BOOST_REGEX
#include <boost/regex.hpp>
using boost::regex;
using boost::regex_search;
#else
#include <regex>
using std::regex;
using std::regex_search;
#endif

#ifndef _WIN32
#include <glob.h>
#endif

extern "C" {

// XXX: equivalent to payload data model in str_arr_ext.py
struct str_arr_payload {
    uint32_t *offsets;
    char* data;
    uint8_t* null_bitmap;
};

// XXX: equivalent to payload data model in split_impl.py
struct str_arr_split_view_payload {
    uint32_t *index_offsets;
    uint32_t *data_offsets;
    // uint8_t* null_bitmap;
};

// taken from Arrow bin-util.h
static constexpr uint8_t kBitmask[] = {1, 2, 4, 8, 16, 32, 64, 128};

void* init_string(char*, int64_t);
void* init_string_const(char* in_str);
void dtor_string(std::string** in_str, int64_t size, void* in);
void dtor_string_array(str_arr_payload* in_str, int64_t size, void* in);
void dtor_str_arr_split_view(str_arr_split_view_payload* in_str_arr, int64_t size, void* in);
void str_arr_split_view_impl(str_arr_split_view_payload* out_view, int64_t n_strs, uint32_t* offsets, char* data, char sep);
const char* get_c_str(std::string* s);
const char* get_char_ptr(char c);
void* str_concat(std::string* s1, std::string* s2);
int str_compare(std::string* s1, std::string* s2);
bool str_equal(std::string* s1, std::string* s2);
bool str_equal_cstr(std::string* s1, char* s2);
void* str_split(std::string* str, std::string* sep, int64_t *size);
void* str_substr_int(std::string* str, int64_t index);
int64_t str_to_int64(std::string* str);
double str_to_float64(std::string* str);
int64_t get_str_len(std::string* str);
void string_array_from_sequence(PyObject * obj, int64_t * no_strings, uint32_t ** offset_table,
    char ** buffer, uint8_t **null_bitmap);
void* np_array_from_string_array(int64_t no_strings, const uint32_t * offset_table,
    const char *buffer, const uint8_t *null_bitmap);
void allocate_string_array(uint32_t **offsets, char **data, uint8_t **null_bitmap,
    int64_t num_strings, int64_t total_size);

void setitem_string_array(uint32_t *offsets, char *data, char* str, int64_t len, int64_t index);

void set_string_array_range(uint32_t *out_offsets, char *out_data,
                            uint32_t *in_offsets, char *in_data,
                            int64_t start_str_ind, int64_t start_chars_ind,
                            int64_t num_strs, int64_t num_chars);
void convert_len_arr_to_offset(uint32_t *offsets, int64_t num_strs);
char* getitem_string_array(uint32_t *offsets, char *data, int64_t index);
void* getitem_string_array_std(uint32_t *offsets, char *data, int64_t index);
void print_str(std::string* str);
void print_char(char c);
void print_int(int64_t val);
void* compile_regex(std::string* pat);
bool str_contains_regex(std::string* str, regex* e);
bool str_contains_noregex(std::string* str, std::string* pat);
std::string* str_replace_regex(std::string* str, regex* e, std::string* val);
std::string* str_replace_noregex(std::string* str, std::string* pat, std::string* val);
char get_char_from_string(std::string* str, int64_t index);

void* str_from_int32(int in);
void* str_from_int64(int64_t in);
void* str_from_float32(float in);
void* str_from_float64(double in);
bool is_na(const uint8_t* bull_bitmap, int64_t ind);
void del_str(std::string* in_str);
int64_t hash_str(std::string* in_str);
void c_glob(uint32_t **offsets, char **data, uint8_t **null_bitmap, int64_t* num_strings, char* path);
npy_intp array_size(PyArrayObject* arr);
void* array_getptr1(PyArrayObject* arr, npy_intp ind);
void array_setitem(PyArrayObject* arr, char* p, PyObject *s);


PyMODINIT_FUNC PyInit_hstr_ext(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "hstr_ext", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    // init numpy
    import_array();

    PyObject_SetAttrString(m, "init_string",
                            PyLong_FromVoidPtr((void*)(&init_string)));
    PyObject_SetAttrString(m, "init_string_const",
                            PyLong_FromVoidPtr((void*)(&init_string_const)));
    PyObject_SetAttrString(m, "dtor_string",
                            PyLong_FromVoidPtr((void*)(&dtor_string)));
    PyObject_SetAttrString(m, "dtor_string_array",
                            PyLong_FromVoidPtr((void*)(&dtor_string_array)));
    PyObject_SetAttrString(m, "dtor_str_arr_split_view",
                            PyLong_FromVoidPtr((void*)(&dtor_str_arr_split_view)));
    PyObject_SetAttrString(m, "str_arr_split_view_impl",
                            PyLong_FromVoidPtr((void*)(&str_arr_split_view_impl)));
    PyObject_SetAttrString(m, "get_c_str",
                            PyLong_FromVoidPtr((void*)(&get_c_str)));
    PyObject_SetAttrString(m, "get_char_ptr",
                            PyLong_FromVoidPtr((void*)(&get_char_ptr)));
    PyObject_SetAttrString(m, "str_concat",
                            PyLong_FromVoidPtr((void*)(&str_concat)));
    PyObject_SetAttrString(m, "str_compare",
                            PyLong_FromVoidPtr((void*)(&str_compare)));
    PyObject_SetAttrString(m, "str_equal",
                            PyLong_FromVoidPtr((void*)(&str_equal)));
    PyObject_SetAttrString(m, "str_equal_cstr",
                            PyLong_FromVoidPtr((void*)(&str_equal_cstr)));
    PyObject_SetAttrString(m, "str_split",
                            PyLong_FromVoidPtr((void*)(&str_split)));
    PyObject_SetAttrString(m, "str_substr_int",
                            PyLong_FromVoidPtr((void*)(&str_substr_int)));
    PyObject_SetAttrString(m, "get_char_from_string",
                            PyLong_FromVoidPtr((void*)(&get_char_from_string)));
    PyObject_SetAttrString(m, "str_to_int64",
                            PyLong_FromVoidPtr((void*)(&str_to_int64)));
    PyObject_SetAttrString(m, "str_to_float64",
                            PyLong_FromVoidPtr((void*)(&str_to_float64)));
    PyObject_SetAttrString(m, "get_str_len",
                            PyLong_FromVoidPtr((void*)(&get_str_len)));
    PyObject_SetAttrString(m, "string_array_from_sequence",
                            PyLong_FromVoidPtr((void*)(&string_array_from_sequence)));
    PyObject_SetAttrString(m, "np_array_from_string_array",
                            PyLong_FromVoidPtr((void*)(&np_array_from_string_array)));
    PyObject_SetAttrString(m, "allocate_string_array",
                            PyLong_FromVoidPtr((void*)(&allocate_string_array)));
    PyObject_SetAttrString(m, "setitem_string_array",
                            PyLong_FromVoidPtr((void*)(&setitem_string_array)));
    PyObject_SetAttrString(m, "set_string_array_range",
                            PyLong_FromVoidPtr((void*)(&set_string_array_range)));
    PyObject_SetAttrString(m, "convert_len_arr_to_offset",
                            PyLong_FromVoidPtr((void*)(&convert_len_arr_to_offset)));
    PyObject_SetAttrString(m, "getitem_string_array",
                            PyLong_FromVoidPtr((void*)(&getitem_string_array)));
    PyObject_SetAttrString(m, "getitem_string_array_std",
                            PyLong_FromVoidPtr((void*)(&getitem_string_array_std)));
    PyObject_SetAttrString(m, "print_str",
                            PyLong_FromVoidPtr((void*)(&print_str)));
    PyObject_SetAttrString(m, "print_char",
                            PyLong_FromVoidPtr((void*)(&print_char)));
    PyObject_SetAttrString(m, "print_int",
                            PyLong_FromVoidPtr((void*)(&print_int)));
    PyObject_SetAttrString(m, "compile_regex",
                            PyLong_FromVoidPtr((void*)(&compile_regex)));
    PyObject_SetAttrString(m, "str_contains_noregex",
                            PyLong_FromVoidPtr((void*)(&str_contains_noregex)));
    PyObject_SetAttrString(m, "str_contains_regex",
                            PyLong_FromVoidPtr((void*)(&str_contains_regex)));
    PyObject_SetAttrString(m, "str_replace_regex",
                            PyLong_FromVoidPtr((void*)(&str_replace_regex)));
    PyObject_SetAttrString(m, "str_replace_noregex",
                            PyLong_FromVoidPtr((void*)(&str_replace_noregex)));
    PyObject_SetAttrString(m, "str_from_int32",
                            PyLong_FromVoidPtr((void*)(&str_from_int32)));
    PyObject_SetAttrString(m, "str_from_int64",
                            PyLong_FromVoidPtr((void*)(&str_from_int64)));
    PyObject_SetAttrString(m, "str_from_float32",
                            PyLong_FromVoidPtr((void*)(&str_from_float32)));
    PyObject_SetAttrString(m, "str_from_float64",
                            PyLong_FromVoidPtr((void*)(&str_from_float64)));
    PyObject_SetAttrString(m, "is_na",
                            PyLong_FromVoidPtr((void*)(&is_na)));
    PyObject_SetAttrString(m, "del_str",
                            PyLong_FromVoidPtr((void*)(&del_str)));
    PyObject_SetAttrString(m, "hash_str",
                            PyLong_FromVoidPtr((void*)(&hash_str)));
    PyObject_SetAttrString(m, "c_glob",
                            PyLong_FromVoidPtr((void*)(&c_glob)));
    PyObject_SetAttrString(m, "array_size",
                            PyLong_FromVoidPtr((void*)(&array_size)));
    PyObject_SetAttrString(m, "array_getptr1",
                            PyLong_FromVoidPtr((void*)(&array_getptr1)));
    PyObject_SetAttrString(m, "array_setitem",
                            PyLong_FromVoidPtr((void*)(&array_setitem)));
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
    printf("dtor size: %ld\n", size); fflush(stdout);
    // std::cout<<"del str: "<< (*in_str)->c_str() <<std::endl;
    // delete (*in_str);
    return;
}

void del_str(std::string* in_str)
{
    delete in_str;
    return;
}

int64_t hash_str(std::string* in_str)
{
    std::size_t h1 = std::hash<std::string>{}(*in_str);
    return (int64_t)h1;
}

void dtor_string_array(str_arr_payload* in_str_arr, int64_t size, void* in)
{
    // printf("str arr dtor size: %lld\n", in_str_arr->size);
    // printf("num chars: %d\n", in_str_arr->offsets[in_str_arr->size]);
    delete[] in_str_arr->offsets;
    delete[] in_str_arr->data;
    if (in_str_arr->null_bitmap != nullptr)
        delete[] in_str_arr->null_bitmap;
    return;
}

void dtor_str_arr_split_view(str_arr_split_view_payload* in_str_arr, int64_t size, void* in)
{
    // printf("str arr dtor size: %lld\n", in_str_arr->size);
    // printf("num chars: %d\n", in_str_arr->offsets[in_str_arr->size]);
    delete[] in_str_arr->index_offsets;
    delete[] in_str_arr->data_offsets;
    // if (in_str_arr->null_bitmap != nullptr)
    //     delete[] in_str_arr->null_bitmap;
    return;
}

// example: ['AB,CC', 'C,ABB,D', 'G', '', 'g,f']
// offsets [0, 5, 12, 13, 13, 14, 17]
// data_offsets [-1, 2, 5,   4, 6, 10, 12,  11, 13,   12, 13,   12, 14, 16]
// index_offsets [0, 3, 7, 9, 11, 14]
void str_arr_split_view_impl(str_arr_split_view_payload* out_view, int64_t n_strs, uint32_t* offsets, char* data, char sep)
{
    uint32_t total_chars = offsets[n_strs];
    // printf("n_strs %d sep %c total chars:%d\n", n_strs, sep, total_chars);
    uint32_t* index_offsets = new uint32_t[n_strs+1];
    std::vector<uint32_t> data_offs;

    data_offs.push_back(-1);
    index_offsets[0] = 0;
    // uint32_t curr_data_off = 0;

    int data_ind = offsets[0];
    int str_ind = 0;
    // while there are chars to consume, equal since the first if will consume it
    while (data_ind <= total_chars)
    {
        // string has finished
        if (data_ind == offsets[str_ind+1])
        {
            data_offs.push_back(data_ind);
            index_offsets[str_ind+1] = data_offs.size();
            str_ind++;
            if (str_ind == n_strs) break;  // all finished
            // start new string
            data_offs.push_back(data_ind-1);
            continue;  // stay on same data_ind for start of next string
        }
        if (data[data_ind] == sep)
        {
            data_offs.push_back(data_ind);
        }
        data_ind++;
    }
    out_view->index_offsets = index_offsets;
    out_view->data_offsets = new uint32_t[data_offs.size()];
    // TODO: avoid copy
    std::copy(data_offs.cbegin(), data_offs.cend(), out_view->data_offsets);

    // printf("index_offsets: ");
    // for (int i=0; i<=n_strs; i++)
    //     printf("%d ", index_offsets[i]);
    // printf("\n");
    // printf("data_offsets: ");
    // for (int i=0; i<data_offs.size(); i++)
    //     printf("%d ", data_offs[i]);
    // printf("\n");
    return;
}

const char* get_c_str(std::string* s)
{
    // printf("in get %s\n", s->c_str());
    return s->c_str();
}

const char* get_char_ptr(char c)
{
    // printf("in get %s\n", s->c_str());
    char *str = new char[1];
    str[0] = c;
    return str;
}

void* str_concat(std::string* s1, std::string* s2)
{
    // printf("in concat %s %s\n", s1->c_str(), s2->c_str());
    std::string* res = new std::string((*s1)+(*s2));
    return res;
}

int str_compare(std::string* s1, std::string* s2)
{
    // printf("in str_comp %s %s\n", s1->c_str(), s2->c_str());
    return s1->compare(*s2);
}

bool str_equal(std::string* s1, std::string* s2)
{
    // printf("in str_equal %s %s\n", s1->c_str(), s2->c_str());
    return s1->compare(*s2)==0;
}

bool str_equal_cstr(std::string* s1, char* s2)
{
    // printf("in str_equal %s %s\n", s1->c_str(), s2->c_str());
    return s1->compare(s2)==0;
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

char get_char_from_string(std::string* str, int64_t index)
{
    return str->at(index);
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

void allocate_string_array(uint32_t **offsets, char **data, uint8_t **null_bitmap, int64_t num_strings,
                                                            int64_t total_size)
{
    // std::cout << "allocating string array: " << num_strings << " " <<
    //                                                 total_size << std::endl;
    *offsets = new uint32_t[num_strings+1];
    *data = new char[total_size];
    (*offsets)[0] = 0;
    (*offsets)[num_strings] = (uint32_t)total_size;  // in case total chars is read from here
    // allocate nulls
    int64_t n_bytes = (num_strings+sizeof(uint8_t)-1)/sizeof(uint8_t);
    *null_bitmap = new uint8_t[n_bytes];
    // set all bits to 1 indicating non-null as default
    memset(*null_bitmap, -1, n_bytes);
    // *data = (char*) new std::string("gggg");
    return;
}

void setitem_string_array(uint32_t *offsets, char *data, char* str, int64_t len, int64_t index)
{
    // std::cout << "setitem str: " << *str << " " << index << std::endl;
    if (index==0)
        offsets[index] = 0;
    uint32_t start = offsets[index];
    // std::cout << "start " << start << " len " << len << std::endl;
    memcpy(&data[start], str, len);
    assert(len < std::numeric_limits<uint32_t>::max());
    offsets[index+1] = start+ (uint32_t)len;
    return;
}

void set_string_array_range(uint32_t *out_offsets, char *out_data,
                            uint32_t *in_offsets, char *in_data,
                            int64_t start_str_ind, int64_t start_chars_ind,
                            int64_t num_strs, int64_t num_chars)
{
    // printf("%d %d\n", start_str_ind, start_chars_ind); fflush(stdout);
    uint32_t curr_offset = 0;
    if (start_str_ind!=0)
        curr_offset = out_offsets[start_str_ind];

    // set offsets
    for (size_t i=0; i<(size_t)num_strs; i++)
    {
        out_offsets[start_str_ind+i] = curr_offset;
        int32_t len = in_offsets[i+1]-in_offsets[i];
        curr_offset += len;
    }
    out_offsets[start_str_ind+num_strs] = curr_offset;
    // copy all chars
    memcpy(out_data+start_chars_ind, in_data, num_chars);
    return;
}

void convert_len_arr_to_offset(uint32_t *offsets, int64_t num_strs)
{
    uint32_t curr_offset = 0;
    for(int64_t i=0; i<num_strs; i++)
    {
        uint32_t val = offsets[i];
        offsets[i] = curr_offset;
        curr_offset += val;
    }
    offsets[num_strs] = curr_offset;
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

void* compile_regex(std::string* pat)
{
    // printf("compiling\n");
    // regex rr2("3");
    // printf("1 compiling\n");
    // regex * rr = new regex(*pat);
    // printf("done compiling\n");
    return new regex(*pat);
}

bool str_contains_regex(std::string* str, regex* e)
{
    // printf("regex matching\n");
    // regex e(*pat);
    // return regex_search(*str, e, regex_constants::match_any);
    return regex_search(*str, *e);
}

bool str_contains_noregex(std::string* str, std::string* pat)
{
    return (str->find(*pat) != std::string::npos);
}


std::string* str_replace_regex(std::string* str, regex* e, std::string* val)
{
    return new std::string(regex_replace(*str, *e, *val));
}

std::string* str_replace_noregex(std::string* str, std::string* pat, std::string* val)
{
    std::string* out = new std::string(*str);
    boost::replace_all(*out, *pat, *val);
    // std::cout << *out << std::endl;
    return out;
}

void print_str(std::string* str)
{
    std::cout<< *str;
    return;
}

void print_char(char c)
{
    std::cout << c;
    return;
}

void print_int(int64_t val)
{
    printf("%ld\n", val);
}

void* str_from_int32(int in)
{
    return new std::string(std::to_string(in));
}

void* str_from_int64(int64_t in)
{
    return new std::string(std::to_string(in));
}

void* str_from_float32(float in)
{
    return new std::string(std::to_string(in));
}

void* str_from_float64(double in)
{
    return new std::string(std::to_string(in));
}

bool is_na(const uint8_t* null_bitmap, int64_t i)
{
    // printf("%d\n", *null_bitmap);
    return (null_bitmap[i / 8] & kBitmask[i % 8]) == 0;
}

#if PY_VERSION_HEX >= 0x03000000
#define PyString_Check(name) PyUnicode_Check(name)
#define PyString_AsString(str) PyUnicode_AsUTF8(str)
#define PyString_FromStringAndSize(str, sz) PyUnicode_FromStringAndSize(str, sz)
#endif

/// @brief create a concatenated string and offset table from a pandas series of strings
/// @note strings in returned buffer will not be 0-terminated.
/// @param[out] buffer newly allocated buffer with concatenated strings, or NULL
/// @param[out] no_strings number of strings concatenated, value < 0 indicates an error
/// @param[out] offset_table newly allocated array of no_strings+1 integers
///                          first no_strings entries denote offsets, last entry indicates size of output array
/// @param[in]  obj Python Sequence object, intended to be a pandas series of string
void string_array_from_sequence(PyObject * obj, int64_t * no_strings, uint32_t ** offset_table, char ** buffer, uint8_t **null_bitmap)
{
#define CHECK(expr, msg) if(!(expr)){std::cerr << msg << std::endl; PyGILState_Release(gilstate); if(offsets != NULL) { delete [] offsets; } return;}

    uint32_t * offsets = NULL;

    auto gilstate = PyGILState_Ensure();

    if (no_strings == NULL || offset_table == NULL || buffer == NULL) {
        PyGILState_Release(gilstate);
        return;
    }

    *no_strings = -1;
    *offset_table = NULL;
    *buffer = NULL;

    CHECK(PySequence_Check(obj), "expecting a PySequence");
    CHECK(no_strings && offset_table && buffer, "output arguments must not be NULL");

    Py_ssize_t n = PyObject_Size(obj);
    if(n == 0 ) {
        // empty sequence, this is not an error, need to set size
        PyGILState_Release(gilstate);
        *no_strings = 0;
        *null_bitmap = new uint8_t[0];
        *offset_table = new uint32_t[1];
        (*offset_table)[0] = 0;
        *buffer = new char[0];
        return;
    }

    // allocate null bitmap
    int64_t n_bytes = (n+sizeof(uint8_t)-1)/sizeof(uint8_t);
    *null_bitmap = new uint8_t[n_bytes];
    memset(*null_bitmap, 0, n_bytes);

    // if obj is a pd.Series, get the numpy array for better performance
    // TODO: check actual Series class
    if (PyObject_HasAttrString(obj, "values"))
    {
        obj = PyObject_GetAttrString(obj, "values");
    }

    offsets = new uint32_t[n+1];
    std::vector<const char *> tmp_store(n);
    size_t len = 0;
    for(Py_ssize_t i = 0; i < n; ++i) {
        offsets[i] = len;
        PyObject * s = PySequence_GetItem(obj, i);
        CHECK(s, "getting element failed");
        // Pandas stores NA as either None or nan
        if (s == Py_None || (PyFloat_Check(s) && std::isnan(PyFloat_AsDouble(s))))
        {
            // leave null bit as 0
            tmp_store[i] = "";
        }
        else
        {
            // set null bit to 1 (Arrow bin-util.h)
            (*null_bitmap)[i / 8] |= kBitmask[i % 8];
            CHECK(PyString_Check(s), "expecting a string");
            tmp_store[i] = PyString_AsString(s);
            CHECK(tmp_store[i], "string conversion failed");
            len += strlen(tmp_store[i]);
        }
        Py_DECREF(s);
    }
    offsets[n] = len;

    char * outbuf = new char[len];
    for(Py_ssize_t i = 0; i < n; ++i) {
        memcpy(outbuf+offsets[i], tmp_store[i], offsets[i+1]-offsets[i]);
    }

    PyGILState_Release(gilstate);

    *offset_table = offsets;
    *no_strings = n;
    *buffer = outbuf;

    return;
#undef CHECK
}

/// @brief  From a StringArray create a numpy array of string objects
/// @return numpy array of str objects
/// @param[in] no_strings number of strings found in buffer
/// @param[in] offset_table offsets for strings in buffer
/// @param[in] buffer with concatenated strings (from StringArray)
void* np_array_from_string_array(int64_t no_strings, const uint32_t * offset_table, const char *buffer, const uint8_t *null_bitmap)
{
#define CHECK(expr, msg) if(!(expr)){std::cerr << msg << std::endl; PyGILState_Release(gilstate); return NULL;}
    auto gilstate = PyGILState_Ensure();

    npy_intp dims[] = {no_strings};
    PyObject* ret = PyArray_SimpleNew(1, dims, NPY_OBJECT);
    CHECK(ret, "allocating numpy array failed");
    int err;
    PyObject* np_mod = PyImport_ImportModule("numpy");
    CHECK(np_mod, "importing numpy module failed");
    PyObject* nan_obj = PyObject_GetAttrString(np_mod, "nan");
    CHECK(nan_obj, "getting np.nan failed");

    for(int64_t i = 0; i < no_strings; ++i) {
        PyObject * s = PyString_FromStringAndSize(buffer+offset_table[i], offset_table[i+1]-offset_table[i]);
        CHECK(s, "creating Python string/unicode object failed");
        auto p = PyArray_GETPTR1((PyArrayObject*)ret, i);
        CHECK(p, "getting offset in numpy array failed");
        if (!is_na(null_bitmap, i))
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, s);
        else
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, nan_obj);
        CHECK(err==0, "setting item in numpy array failed");
        Py_DECREF(s);
    }

    Py_DECREF(np_mod);
    Py_DECREF(nan_obj);
    PyGILState_Release(gilstate);
    return ret;
#undef CHECK
}

// helper functions for call Numpy APIs
npy_intp array_size(PyArrayObject* arr)
{
    // std::cout << "get size\n";
    return PyArray_SIZE(arr);
}

void* array_getptr1(PyArrayObject* arr, npy_intp ind)
{
    // std::cout << "get array ptr " << ind << '\n';
    return PyArray_GETPTR1(arr, ind);
}

void array_setitem(PyArrayObject* arr, char* p, PyObject *s)
{
#define CHECK(expr, msg) if(!(expr)){std::cerr << msg << std::endl; return;}
    // std::cout << "get array ptr " << ind << '\n';
    int err = PyArray_SETITEM(arr, p, s);
    CHECK(err==0, "setting item in numpy array failed");
    return;
#undef CHECK
}

// glob support
void c_glob(uint32_t **offsets, char **data, uint8_t **null_bitmap, int64_t* num_strings, char* path)
{
    // std::cout << "glob: " << std::string(path) << std::endl;
    *num_strings = 0;
    #ifndef _WIN32
    glob_t globBuf;
    int ret = glob(path, 0, 0, &globBuf);

    if (ret!=0)
    {
        if (ret==GLOB_NOMATCH)
        {
            globfree(&globBuf);
            return;
        }
        // TODO: match errors, e.g. GLOB_ABORTED GLOB_NOMATCH GLOB_NOSPACE
        std::cerr << "glob error" << '\n';
        globfree(&globBuf);
        return;
    }

    // std::cout << "num glob: " << globBuf.gl_pathc << std::endl;

    *num_strings = globBuf.gl_pathc;
    *offsets = new uint32_t[globBuf.gl_pathc+1];
    size_t total_size = 0;

    for (unsigned int i=0; i<globBuf.gl_pathc; i++)
    {
        (*offsets)[i] = (uint32_t)total_size;
        size_t curr_size = strlen(globBuf.gl_pathv[i]);
        total_size += curr_size;
    }
    (*offsets)[globBuf.gl_pathc] = (uint32_t) total_size;

    *data = new char[total_size];
    for (unsigned int i=0; i<globBuf.gl_pathc; i++)
    {
        strcpy(*data+(*offsets)[i], globBuf.gl_pathv[i]);
    }

    // allocate null bitmap
    int64_t n_bytes = (*num_strings+sizeof(uint8_t)-1)/sizeof(uint8_t);
    *null_bitmap = new uint8_t[n_bytes];
    memset(*null_bitmap, -1, n_bytes);  // set all bits to one for non-null

    // std::cout << "glob done" << std::endl;
    globfree(&globBuf);

    #else
    // TODO: support glob on Windows
    std::cerr << "no glob support on windows yet" << '\n';
    #endif

    return;
}

} // extern "C"

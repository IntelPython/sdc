#include "mpi.h"
#include <Python.h>
#include <string>
#include <iostream>
#include <cstring>
#include <cmath>

#include "parquet/arrow/reader.h"
using parquet::arrow::FileReader;

typedef std::vector< std::shared_ptr<FileReader> > FileReaderVec;

// just include parquet reader on Windows since the GCC ABI change issue
// doesn't exist, and VC linker removes unused lib symbols
#if defined(_MSC_VER) || defined(BUILTIN_PARQUET_READER)
#include <parquet_reader/hpat_parquet_reader.cpp>
#else

// parquet type sizes (NOT arrow)
// boolean, int32, int64, int96, float, double
int pq_type_sizes[] = {1, 4, 8, 12, 4, 8};

extern "C" {

void pq_init_reader(const char* file_name, std::shared_ptr<FileReader> *a_reader);
int64_t pq_get_size_single_file(std::shared_ptr<FileReader>, int64_t column_idx);
int64_t pq_read_single_file(std::shared_ptr<FileReader>, int64_t column_idx, uint8_t *out,
                int out_dtype);
int pq_read_parallel_single_file(std::shared_ptr<FileReader>, int64_t column_idx,
                uint8_t* out_data, int out_dtype, int64_t start, int64_t count);
int64_t pq_read_string_single_file(std::shared_ptr<FileReader>, int64_t column_idx,
                                uint32_t **out_offsets, uint8_t **out_data,
    std::vector<uint32_t> *offset_vec=NULL, std::vector<uint8_t> *data_vec=NULL);
int pq_read_string_parallel_single_file(std::shared_ptr<FileReader>, int64_t column_idx,
        uint32_t **out_offsets, uint8_t **out_data, int64_t start, int64_t count,
        std::vector<uint32_t> *offset_vec=NULL, std::vector<uint8_t> *data_vec=NULL);

}  // extern "C"

#endif  // _MSC_VER

FileReaderVec* get_arrow_readers(std::string* file_name);
void del_arrow_readers(FileReaderVec *readers);

PyObject* str_list_to_vec(PyObject* self, PyObject* str_list);
int64_t pq_get_size(FileReaderVec *readers, int64_t column_idx);
int64_t pq_read(FileReaderVec *readers, int64_t column_idx,
                uint8_t *out_data, int out_dtype);
int pq_read_parallel(FileReaderVec *readers, int64_t column_idx,
                uint8_t* out_data, int out_dtype, int64_t start, int64_t count);
int pq_read_string(FileReaderVec *readers, int64_t column_idx,
                                    uint32_t **out_offsets, uint8_t **out_data);
int pq_read_string_parallel(FileReaderVec *readers, int64_t column_idx,
        uint32_t **out_offsets, uint8_t **out_data, int64_t start, int64_t count);

static PyMethodDef parquet_cpp_methods[] = {
    {
        "str_list_to_vec", str_list_to_vec, METH_O, // METH_STATIC
        "convert Python string list to C++ std vector of strings"
    },
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC PyInit_parquet_cpp(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "parquet_cpp", "No docs", -1, parquet_cpp_methods, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "get_arrow_readers",
                            PyLong_FromVoidPtr((void*)(&get_arrow_readers)));
    PyObject_SetAttrString(m, "del_arrow_readers",
                            PyLong_FromVoidPtr((void*)(&del_arrow_readers)));
    PyObject_SetAttrString(m, "read",
                            PyLong_FromVoidPtr((void*)(&pq_read)));
    PyObject_SetAttrString(m, "read_parallel",
                            PyLong_FromVoidPtr((void*)(&pq_read_parallel)));
    PyObject_SetAttrString(m, "get_size",
                            PyLong_FromVoidPtr((void*)(&pq_get_size)));
    PyObject_SetAttrString(m, "read_string",
                            PyLong_FromVoidPtr((void*)(&pq_read_string)));
    PyObject_SetAttrString(m, "read_string_parallel",
                            PyLong_FromVoidPtr((void*)(&pq_read_string_parallel)));

    return m;
}

PyObject* str_list_to_vec(PyObject* self, PyObject* str_list)
{
    Py_INCREF(str_list);  // needed?
    // TODO: need to acquire GIL?
    std::vector<std::string> *strs_vec = new std::vector<std::string>();

    PyObject *iterator = PyObject_GetIter(str_list);
    Py_DECREF(str_list);
    PyObject *l_str;

    if (iterator == NULL) {
        Py_DECREF(iterator);
        return PyLong_FromVoidPtr((void*) strs_vec);
    }

    while (l_str = PyIter_Next(iterator)) {
        const char *c_path = PyUnicode_AsUTF8(l_str);
        // printf("str %s\n", c_path);
        strs_vec->push_back(std::string(c_path));
        Py_DECREF(l_str);
    }

    Py_DECREF(iterator);

    // CHECK(!PyErr_Occurred(), "Python error during Parquet dataset metadata")
    return PyLong_FromVoidPtr((void*) strs_vec);
}

std::vector<std::string> get_pq_pieces(std::string* file_name)
{
#define CHECK(expr, msg) if(!(expr)){std::cerr << msg << std::endl; PyGILState_Release(gilstate); return std::vector<std::string>();}

    std::vector<std::string> paths;

    auto gilstate = PyGILState_Ensure();

    // import pyarrow.parquet, FIXME: is this import reliable?
    PyObject* pq_mod = PyImport_ImportModule("pyarrow.parquet");

    // ds = pq.ParquetDataset(file_name)
    PyObject* ds = PyObject_CallMethod(pq_mod, "ParquetDataset", "s", file_name->c_str());
    CHECK(!PyErr_Occurred(), "Python error during Parquet dataset metadata")
    Py_DECREF(pq_mod);

    // all_peices = ds.pieces
    PyObject* all_peices = PyObject_GetAttrString(ds, "pieces");
    Py_DECREF(ds);

    // paths.append(piece.path) for piece in all peices
    PyObject *iterator = PyObject_GetIter(all_peices);
    Py_DECREF(all_peices);
    PyObject *piece;

    if (iterator == NULL) {
        // printf("empty\n");
        PyGILState_Release(gilstate);
        Py_DECREF(iterator);
        return paths;
    }

    while (piece = PyIter_Next(iterator)) {
        PyObject* p = PyObject_GetAttrString(piece, "path");
        const char *c_path = PyUnicode_AsUTF8(p);
        // printf("piece %s\n", c_path);
        paths.push_back(std::string(c_path));
        Py_DECREF(piece);
        Py_DECREF(p);
    }

    Py_DECREF(iterator);

    CHECK(!PyErr_Occurred(), "Python error during Parquet dataset metadata")
    PyGILState_Release(gilstate);
    return paths;
#undef CHECK
}


FileReaderVec* get_arrow_readers(std::string* file_name)
{
    FileReaderVec *readers = new FileReaderVec();

    std::vector<std::string> all_files = get_pq_pieces(file_name);
    for (const auto& inner_file : all_files)
    {
        std::shared_ptr<FileReader> arrow_reader;
        pq_init_reader(inner_file.c_str(), &arrow_reader);
        readers->push_back(arrow_reader);
    }

    return readers;
}

void del_arrow_readers(FileReaderVec *readers)
{
    delete readers;
    return;
}

int64_t pq_get_size(FileReaderVec *readers, int64_t column_idx)
{
    if (readers->size() == 0) {
        printf("empty parquet dataset\n");
        return 0;
    }

    if (readers->size() > 1)
    {
        // std::cout << "pq path is dir" << '\n';
        int64_t ret = 0;
        for (size_t i=0; i<readers->size(); i++)
        {
            ret += pq_get_size_single_file(readers->at(i), column_idx);
        }

        // std::cout << "total pq dir size: " << ret << '\n';
        return ret;
    }
    else
    {
        return pq_get_size_single_file(readers->at(0), column_idx);
    }
    return 0;
}

int64_t pq_read(FileReaderVec *readers, int64_t column_idx,
                uint8_t *out_data, int out_dtype)
{
    if (readers->size() == 0) {
        printf("empty parquet dataset\n");
        return 0;
    }

    if (readers->size() > 1)
    {
        // std::cout << "pq path is dir" << '\n';

        int64_t byte_offset = 0;
        for (size_t i=0; i<readers->size(); i++)
        {
            byte_offset += pq_read_single_file(readers->at(i), column_idx, out_data+byte_offset, out_dtype);
        }

        // std::cout << "total pq dir size: " << byte_offset << '\n';
        return byte_offset;
    }
    else
    {
        return pq_read_single_file(readers->at(0), column_idx, out_data, out_dtype);
    }
    return 0;
}

int pq_read_parallel(FileReaderVec *readers, int64_t column_idx,
                uint8_t* out_data, int out_dtype, int64_t start, int64_t count)
{
    // printf("read parquet parallel column: %lld start: %lld count: %lld\n",
    //                                                 column_idx, start, count);

    if (count==0) {
        return 0;
    }

    if (readers->size() == 0) {
        printf("empty parquet dataset\n");
        return 0;
    }

    if (readers->size() > 1)
    {
        // std::cout << "pq path is dir" << '\n';
        // TODO: get file sizes on root rank only

        // skip whole files if no need to read any rows
        int file_ind = 0;
        int64_t file_size = pq_get_size_single_file(readers->at(0), column_idx);
        while (start >= file_size)
        {
            start -= file_size;
            file_ind++;
            file_size = pq_get_size_single_file(readers->at(file_ind), column_idx);
        }

        int dtype_size = pq_type_sizes[out_dtype];
        // std::cout << "dtype_size: " << dtype_size << '\n';

        // read data
        int64_t read_rows = 0;
        while (read_rows<count)
        {
            int64_t rows_to_read = std::min(count-read_rows, file_size-start);
            pq_read_parallel_single_file(readers->at(file_ind), column_idx,
                out_data+read_rows*dtype_size, out_dtype, start, rows_to_read);
            read_rows += rows_to_read;
            start = 0;  // start becomes 0 after reading non-empty first chunk
            file_ind++;
            // std::cout << "next file: " << all_files[file_ind] << '\n';
            if (read_rows<count)
                file_size = pq_get_size_single_file(readers->at(file_ind), column_idx);
        }
        return 0;
        // std::cout << "total pq dir size: " << byte_offset << '\n';
    }
    else
    {
        return pq_read_parallel_single_file(readers->at(0), column_idx,
                                        out_data, out_dtype, start, count);
    }
    return 0;
}

int pq_read_string(FileReaderVec *readers, int64_t column_idx,
                                    uint32_t **out_offsets, uint8_t **out_data)
{

    if (readers->size() == 0) {
        printf("empty parquet dataset\n");
        return 0;
    }

    if (readers->size() > 1)
    {
        // std::cout << "pq path is dir" << '\n';

        std::vector<uint32_t> offset_vec;
        std::vector<uint8_t> data_vec;
        int32_t last_offset = 0;
        int64_t res = 0;
        for (size_t i=0; i<readers->size(); i++)
        {
            int64_t n_vals = pq_read_string_single_file(readers->at(i), column_idx, NULL, NULL, &offset_vec, &data_vec);
            if (n_vals==-1)
                continue;

            int size = offset_vec.size();
            for(int64_t i=1; i<=n_vals+1; i++)
                offset_vec[size-i] += last_offset;
            last_offset = offset_vec[size-1];
            offset_vec.pop_back();
            res += n_vals;
        }
        offset_vec.push_back(last_offset);

        *out_offsets = new uint32_t[offset_vec.size()];
        *out_data = new uint8_t[data_vec.size()];

        memcpy(*out_offsets, offset_vec.data(), offset_vec.size()*sizeof(uint32_t));
        memcpy(*out_data, data_vec.data(), data_vec.size());
        // for(int i=0; i<offset_vec.size(); i++)
        //     std::cout << (*out_offsets)[i] << ' ';
        // std::cout << '\n';
        // std::cout << "string dir read done" << '\n';
        return res;
    }
    else
    {
        return pq_read_string_single_file(readers->at(0), column_idx, out_offsets, out_data);
    }
    return 0;
}

int pq_read_string_parallel(FileReaderVec *readers, int64_t column_idx,
        uint32_t **out_offsets, uint8_t **out_data, int64_t start, int64_t count)
{
    // printf("read parquet parallel str file: %s column: %lld start: %lld count: %lld\n",
    //                                 file_name->c_str(), column_idx, start, count);

    if (readers->size() == 0) {
        printf("empty parquet dataset\n");
        return 0;
    }

    if (readers->size() > 1)
    {
        // std::cout << "pq path is dir" << '\n';

        // skip whole files if no need to read any rows
        int file_ind = 0;
        int64_t file_size = pq_get_size_single_file(readers->at(0), column_idx);
        while (start >= file_size)
        {
            start -= file_size;
            file_ind++;
            file_size = pq_get_size_single_file(readers->at(file_ind), column_idx);
        }

        int64_t res = 0;
        std::vector<uint32_t> offset_vec;
        std::vector<uint8_t> data_vec;

        // read data
        int64_t last_offset = 0;
        int64_t read_rows = 0;
        while (read_rows<count)
        {
            int64_t rows_to_read = std::min(count-read_rows, file_size-start);
            if (rows_to_read>0)
            {
                pq_read_string_parallel_single_file(readers->at(file_ind), column_idx,
                    NULL, NULL, start, rows_to_read, &offset_vec, &data_vec);

                int size = offset_vec.size();
                for(int64_t i=1; i<=rows_to_read+1; i++)
                    offset_vec[size-i] += last_offset;
                last_offset = offset_vec[size-1];
                offset_vec.pop_back();
                res += rows_to_read;
            }

            read_rows += rows_to_read;
            start = 0;  // start becomes 0 after reading non-empty first chunk
            file_ind++;
            if (read_rows<count)
                file_size = pq_get_size_single_file(readers->at(file_ind), column_idx);
        }
        offset_vec.push_back(last_offset);

        *out_offsets = new uint32_t[offset_vec.size()];
        *out_data = new uint8_t[data_vec.size()];

        memcpy(*out_offsets, offset_vec.data(), offset_vec.size()*sizeof(uint32_t));
        memcpy(*out_data, data_vec.data(), data_vec.size());
        return res;
    }
    else
    {
        return pq_read_string_parallel_single_file(readers->at(0), column_idx,
                out_offsets, out_data, start, count);
    }
    return 0;
}

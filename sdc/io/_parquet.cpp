//*****************************************************************************
// Copyright (c) 2019, Intel Corporation All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

#include <Python.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

#if _MSC_VER >= 1900
#undef timezone
#endif

#include "parquet/arrow/reader.h"
using parquet::arrow::FileReader;

typedef std::vector<std::shared_ptr<FileReader>> FileReaderVec;

// just include parquet reader on Windows since the GCC ABI change issue
// doesn't exist, and VC linker removes unused lib symbols
#if defined(_MSC_VER) || defined(BUILTIN_PARQUET_READER)
#include <parquet_reader/hpat_parquet_reader.cpp>
#else

void pq_init_reader(const char* file_name, std::shared_ptr<FileReader>* a_reader);
int64_t pq_get_size_single_file(std::shared_ptr<FileReader>, int64_t column_idx);
int64_t pq_read_single_file(std::shared_ptr<FileReader>, int64_t column_idx, uint8_t* out, int out_dtype);
int pq_read_parallel_single_file(
    std::shared_ptr<FileReader>, int64_t column_idx, uint8_t* out_data, int out_dtype, int64_t start, int64_t count);
int64_t pq_read_string_single_file(std::shared_ptr<FileReader>,
                                   int64_t column_idx,
                                   uint32_t** out_offsets,
                                   uint8_t** out_data,
                                   uint8_t** out_nulls,
                                   std::vector<uint32_t>* offset_vec = NULL,
                                   std::vector<uint8_t>* data_vec = NULL,
                                   std::vector<bool>* null_vec = NULL);
int pq_read_string_parallel_single_file(std::shared_ptr<FileReader>,
                                        int64_t column_idx,
                                        uint32_t** out_offsets,
                                        uint8_t** out_data,
                                        uint8_t** out_nulls,
                                        int64_t start,
                                        int64_t count,
                                        std::vector<uint32_t>* offset_vec = NULL,
                                        std::vector<uint8_t>* data_vec = NULL,
                                        std::vector<bool>* null_vec = NULL);

#endif // _MSC_VER

FileReaderVec* get_arrow_readers(char* file_name);
void del_arrow_readers(FileReaderVec* readers);

PyObject* str_list_to_vec(PyObject* self, PyObject* str_list);
int64_t pq_get_size(FileReaderVec* readers, int64_t column_idx);
int64_t pq_read(FileReaderVec* readers, int64_t column_idx, uint8_t* out_data, int out_dtype);
int pq_read_parallel(
    FileReaderVec* readers, int64_t column_idx, uint8_t* out_data, int out_dtype, int64_t start, int64_t count);
int pq_read_string(
    FileReaderVec* readers, int64_t column_idx, uint32_t** out_offsets, uint8_t** out_data, uint8_t** out_nulls);
int pq_read_string_parallel(FileReaderVec* readers,
                            int64_t column_idx,
                            uint32_t** out_offsets,
                            uint8_t** out_data,
                            uint8_t** out_nulls,
                            int64_t start,
                            int64_t count);

void pack_null_bitmap(uint8_t** out_nulls, std::vector<bool>& null_vec, int64_t n_all_vals);

static PyMethodDef parquet_cpp_methods[] = {{"str_list_to_vec",
                                             str_list_to_vec,
                                             METH_O, // METH_STATIC
                                             "convert Python string list to C++ std vector of strings"},
                                            {NULL, NULL, 0, NULL}};

PyMODINIT_FUNC PyInit_parquet_cpp(void)
{
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "parquet_cpp",
        "No docs",
        -1,
        parquet_cpp_methods,
    };

    m = PyModule_Create(&moduledef);
    if (m == NULL)
    {
        return NULL;
    }

    PyObject_SetAttrString(m, "get_arrow_readers", PyLong_FromVoidPtr((void*)(&get_arrow_readers)));
    PyObject_SetAttrString(m, "del_arrow_readers", PyLong_FromVoidPtr((void*)(&del_arrow_readers)));
    PyObject_SetAttrString(m, "read", PyLong_FromVoidPtr((void*)(&pq_read)));
    PyObject_SetAttrString(m, "read_parallel", PyLong_FromVoidPtr((void*)(&pq_read_parallel)));
    PyObject_SetAttrString(m, "get_size", PyLong_FromVoidPtr((void*)(&pq_get_size)));
    PyObject_SetAttrString(m, "read_string", PyLong_FromVoidPtr((void*)(&pq_read_string)));
    PyObject_SetAttrString(m, "read_string_parallel", PyLong_FromVoidPtr((void*)(&pq_read_string_parallel)));

    return m;
}

PyObject* str_list_to_vec(PyObject* self, PyObject* str_list)
{
    Py_INCREF(str_list); // needed?
    // TODO: need to acquire GIL?
    std::vector<std::string>* strs_vec = new std::vector<std::string>();

    PyObject* iterator = PyObject_GetIter(str_list);
    Py_DECREF(str_list);
    PyObject* l_str;

    if (iterator == NULL)
    {
        return PyLong_FromVoidPtr((void*)strs_vec);
    }

    while ((l_str = PyIter_Next(iterator)))
    {
        const char* c_path = PyUnicode_AsUTF8(l_str);
        // printf("str %s\n", c_path);
        strs_vec->push_back(std::string(c_path));
        Py_DECREF(l_str);
    }

    Py_DECREF(iterator);

    // CHECK(!PyErr_Occurred(), "Python error during Parquet dataset metadata")
    return PyLong_FromVoidPtr((void*)strs_vec);
}

std::vector<std::string> get_pq_pieces(char* file_name)
{
#define CHECK(expr, msg)                                                                                               \
    if (!(expr))                                                                                                       \
    {                                                                                                                  \
        std::cerr << msg << std::endl;                                                                                 \
        PyGILState_Release(gilstate);                                                                                  \
        return std::vector<std::string>();                                                                             \
    }

    std::vector<std::string> paths;

    auto gilstate = PyGILState_Ensure();

    // import pyarrow.parquet, FIXME: is this import reliable?
    PyObject* pq_mod = PyImport_ImportModule("pyarrow.parquet");

    // ds = pq.ParquetDataset(file_name)
    PyObject* ds = PyObject_CallMethod(pq_mod, "ParquetDataset", "s", file_name);
    CHECK(!PyErr_Occurred(), "Python error during Parquet dataset metadata")
    Py_DECREF(pq_mod);

    // all_peices = ds.pieces
    PyObject* all_peices = PyObject_GetAttrString(ds, "pieces");
    Py_DECREF(ds);

    // paths.append(piece.path) for piece in all peices
    PyObject* iterator = PyObject_GetIter(all_peices);
    Py_DECREF(all_peices);
    PyObject* piece;

    if (iterator == NULL)
    {
        // printf("empty\n");
        PyGILState_Release(gilstate);
        return paths;
    }

    while ((piece = PyIter_Next(iterator)))
    {
        PyObject* p = PyObject_GetAttrString(piece, "path");
        const char* c_path = PyUnicode_AsUTF8(p);
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

FileReaderVec* get_arrow_readers(char* file_name)
{
    FileReaderVec* readers = new FileReaderVec();

    std::vector<std::string> all_files = get_pq_pieces(file_name);
    for (const auto& inner_file : all_files)
    {
        std::shared_ptr<FileReader> arrow_reader;
        pq_init_reader(inner_file.c_str(), &arrow_reader);
        readers->push_back(arrow_reader);
    }

    return readers;
}

void del_arrow_readers(FileReaderVec* readers)
{
    delete readers;
    return;
}

int64_t pq_get_size(FileReaderVec* readers, int64_t column_idx)
{
    if (readers->size() == 0)
    {
        printf("empty parquet dataset\n");
        return 0;
    }

    if (readers->size() > 1)
    {
        // std::cout << "pq path is dir" << '\n';
        int64_t ret = 0;
        for (size_t i = 0; i < readers->size(); i++)
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

int64_t pq_read(FileReaderVec* readers, int64_t column_idx, uint8_t* out_data, int out_dtype)
{
    if (readers->size() == 0)
    {
        printf("empty parquet dataset\n");
        return 0;
    }

    if (readers->size() > 1)
    {
        // std::cout << "pq path is dir" << '\n';

        int64_t byte_offset = 0;
        for (size_t i = 0; i < readers->size(); i++)
        {
            byte_offset += pq_read_single_file(readers->at(i), column_idx, out_data + byte_offset, out_dtype);
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

int pq_read_parallel(
    FileReaderVec* readers, int64_t column_idx, uint8_t* out_data, int out_dtype, int64_t start, int64_t count)
{
    // printf("read parquet parallel column: %lld start: %lld count: %lld\n",
    //                                                 column_idx, start, count);

    if (count == 0)
    {
        return 0;
    }

    if (readers->size() == 0)
    {
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
        while (read_rows < count)
        {
            int64_t rows_to_read = std::min(count - read_rows, file_size - start);
            pq_read_parallel_single_file(
                readers->at(file_ind), column_idx, out_data + read_rows * dtype_size, out_dtype, start, rows_to_read);
            read_rows += rows_to_read;
            start = 0; // start becomes 0 after reading non-empty first chunk
            file_ind++;
            // std::cout << "next file: " << all_files[file_ind] << '\n';
            if (read_rows < count)
            {
                file_size = pq_get_size_single_file(readers->at(file_ind), column_idx);
            }
        }
        return 0;
        // std::cout << "total pq dir size: " << byte_offset << '\n';
    }
    else
    {
        return pq_read_parallel_single_file(readers->at(0), column_idx, out_data, out_dtype, start, count);
    }
    return 0;
}

int pq_read_string(
    FileReaderVec* readers, int64_t column_idx, uint32_t** out_offsets, uint8_t** out_data, uint8_t** out_nulls)
{
    if (readers->size() == 0)
    {
        printf("empty parquet dataset\n");
        return 0;
    }

    if (readers->size() > 1)
    {
        // std::cout << "pq path is dir" << '\n';

        std::vector<uint32_t> offset_vec;
        std::vector<uint8_t> data_vec;
        std::vector<bool> null_vec;
        int32_t last_offset = 0;
        int64_t n_all_vals = 0;
        for (size_t i = 0; i < readers->size(); i++)
        {
            int64_t n_vals = pq_read_string_single_file(
                readers->at(i), column_idx, NULL, NULL, NULL, &offset_vec, &data_vec, &null_vec);
            if (n_vals == -1)
            {
                continue;
            }

            int size = offset_vec.size();
            for (int64_t i = 1; i <= n_vals + 1; i++)
            {
                offset_vec[size - i] += last_offset;
            }
            last_offset = offset_vec[size - 1];
            offset_vec.pop_back();
            n_all_vals += n_vals;
        }
        offset_vec.push_back(last_offset);

        *out_offsets = new uint32_t[offset_vec.size()];
        *out_data = new uint8_t[data_vec.size()];

        memcpy(*out_offsets, offset_vec.data(), offset_vec.size() * sizeof(uint32_t));
        memcpy(*out_data, data_vec.data(), data_vec.size());
        pack_null_bitmap(out_nulls, null_vec, n_all_vals);

        // for(int i=0; i<offset_vec.size(); i++)
        //     std::cout << (*out_offsets)[i] << ' ';
        // std::cout << '\n';
        // std::cout << "string dir read done" << '\n';
        return n_all_vals;
    }
    else
    {
        return pq_read_string_single_file(readers->at(0), column_idx, out_offsets, out_data, out_nulls);
    }
    return 0;
}

int pq_read_string_parallel(FileReaderVec* readers,
                            int64_t column_idx,
                            uint32_t** out_offsets,
                            uint8_t** out_data,
                            uint8_t** out_nulls,
                            int64_t start,
                            int64_t count)
{
    // printf("read parquet parallel str file: %s column: %lld start: %lld count: %lld\n",
    //                                 file_name->c_str(), column_idx, start, count);

    if (readers->size() == 0)
    {
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

        int64_t n_all_vals = 0;
        std::vector<uint32_t> offset_vec;
        std::vector<uint8_t> data_vec;
        std::vector<bool> null_vec;

        // read data
        int64_t last_offset = 0;
        int64_t read_rows = 0;
        while (read_rows < count)
        {
            int64_t rows_to_read = std::min(count - read_rows, file_size - start);
            if (rows_to_read > 0)
            {
                pq_read_string_parallel_single_file(readers->at(file_ind),
                                                    column_idx,
                                                    NULL,
                                                    NULL,
                                                    NULL,
                                                    start,
                                                    rows_to_read,
                                                    &offset_vec,
                                                    &data_vec,
                                                    &null_vec);

                int size = offset_vec.size();
                for (int64_t i = 1; i <= rows_to_read + 1; i++)
                {
                    offset_vec[size - i] += last_offset;
                }
                last_offset = offset_vec[size - 1];
                offset_vec.pop_back();
                n_all_vals += rows_to_read;
            }

            read_rows += rows_to_read;
            start = 0; // start becomes 0 after reading non-empty first chunk
            file_ind++;
            if (read_rows < count)
            {
                file_size = pq_get_size_single_file(readers->at(file_ind), column_idx);
            }
        }
        offset_vec.push_back(last_offset);

        *out_offsets = new uint32_t[offset_vec.size()];
        *out_data = new uint8_t[data_vec.size()];

        memcpy(*out_offsets, offset_vec.data(), offset_vec.size() * sizeof(uint32_t));
        memcpy(*out_data, data_vec.data(), data_vec.size());
        pack_null_bitmap(out_nulls, null_vec, n_all_vals);
        return n_all_vals;
    }
    else
    {
        return pq_read_string_parallel_single_file(
            readers->at(0), column_idx, out_offsets, out_data, out_nulls, start, count);
    }
    return 0;
}

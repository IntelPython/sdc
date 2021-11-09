//*****************************************************************************
// Copyright (c) 2019-2021, Intel Corporation All rights reserved.
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

#ifndef HSDC_STR_ARR
#define HSDC_STR_ARR

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <iostream>
#include <numpy/arrayobject.h>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <string_view>
#include <charconv>
#include <cstring>
#include <regex>

#ifndef _WIN32
#include <glob.h>
#endif

#include "sdc_str_decode.hpp"

extern "C"
{
    // XXX: equivalent to payload data model in str_arr_ext.py
    struct str_arr_payload
    {
        uint32_t* offsets;
        char* data;
        uint8_t* null_bitmap;
    };

    // XXX: equivalent to payload data model in split_impl.py
    struct str_arr_split_view_payload
    {
        uint32_t* index_offsets;
        uint32_t* data_offsets;
        // uint8_t* null_bitmap;
    };

    // taken from Arrow bin-util.h
    static constexpr uint8_t kBitmask[] = {1, 2, 4, 8, 16, 32, 64, 128};

    void dtor_string_array(str_arr_payload* in_str_arr, int64_t size, void* in)
    {
        // printf("str arr dtor size: %lld\n", in_str_arr->size);
        // printf("num chars: %d\n", in_str_arr->offsets[in_str_arr->size]);
        delete[] in_str_arr->offsets;
        delete[] in_str_arr->data;
        if (in_str_arr->null_bitmap != nullptr)
        {
            delete[] in_str_arr->null_bitmap;
        }
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

    void str_arr_split_view_alloc(str_arr_split_view_payload* out_view, int64_t num_items, int64_t num_offsets)
    {
        out_view->index_offsets = new uint32_t[num_items + 1];
        out_view->data_offsets = new uint32_t[num_offsets];
        return;
    }

    // example: ['AB,CC', 'C,ABB,D', 'G', '', 'g,f']
    // offsets [0, 5, 12, 13, 13, 14, 17]
    // data_offsets [-1, 2, 5,   4, 6, 10, 12,  11, 13,   12, 13,   12, 14, 16]
    // index_offsets [0, 3, 7, 9, 11, 14]
    void str_arr_split_view_impl(
        str_arr_split_view_payload* out_view, int64_t n_strs, uint32_t* offsets, char* data, char sep)
    {
        uint32_t total_chars = offsets[n_strs];
        // printf("n_strs %d sep %c total chars:%d\n", n_strs, sep, total_chars);
        uint32_t* index_offsets = new uint32_t[n_strs + 1];
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
            if (data_ind == offsets[str_ind + 1])
            {
                data_offs.push_back(data_ind);
                index_offsets[str_ind + 1] = data_offs.size();
                str_ind++;
                if (str_ind == n_strs)
                {
                    break; // all finished
                }
                // start new string
                data_offs.push_back(data_ind - 1);
                continue; // stay on same data_ind for start of next string
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

    void allocate_string_array(
        uint32_t** offsets, char** data, uint8_t** null_bitmap, int64_t num_strings, int64_t total_size)
    {
        // std::cout << "allocating string array: " << num_strings << " " <<
        //                                                 total_size << std::endl;
        *offsets = new uint32_t[num_strings + 1];
        *data = new char[total_size];
        (*offsets)[0] = 0;
        (*offsets)[num_strings] = (uint32_t)total_size; // in case total chars is read from here
        // allocate nulls
        int64_t n_bytes = (num_strings + sizeof(uint8_t) - 1) / sizeof(uint8_t);
        *null_bitmap = new uint8_t[n_bytes];
        // set all bits to 1 indicating non-null as default
        memset(*null_bitmap, -1, n_bytes);
        // *data = (char*) new std::string("gggg");
        return;
    }

    void setitem_string_array(
        uint32_t* offsets, char* data, int64_t n_bytes, char* str, int64_t len, int kind, int is_ascii, int64_t index)
    {
#define CHECK(expr, msg)                                                                                               \
    if (!(expr))                                                                                                       \
    {                                                                                                                  \
        std::cerr << msg << std::endl;                                                                                 \
        return;                                                                                                        \
    }
        // std::cout << "setitem str: " << *str << " " << index << std::endl;
        if (index == 0)
        {
            offsets[index] = 0;
        }
        uint32_t start = offsets[index];
        int64_t utf8_len = -1;
        // std::cout << "start " << start << " len " << len << std::endl;

        if (is_ascii == 1)
        {
            memcpy(&data[start], str, len);
            utf8_len = len;
        }
        else
        {
            utf8_len = unicode_to_utf8(&data[start], str, len, kind);
        }

        CHECK(utf8_len < std::numeric_limits<uint32_t>::max(), "string array too large");
        CHECK(start + utf8_len <= n_bytes, "out of bounds string array setitem");
        offsets[index + 1] = start + (uint32_t)utf8_len;
        return;
#undef CHECK
    }

    void set_string_array_range(uint32_t* out_offsets,
                                char* out_data,
                                uint32_t* in_offsets,
                                char* in_data,
                                int64_t start_str_ind,
                                int64_t start_chars_ind,
                                int64_t num_strs,
                                int64_t num_chars)
    {
        // printf("%d %d\n", start_str_ind, start_chars_ind); fflush(stdout);
        uint32_t curr_offset = 0;
        if (start_str_ind != 0)
        {
            curr_offset = out_offsets[start_str_ind];
        }

        // set offsets
        for (size_t i = 0; i < (size_t)num_strs; i++)
        {
            out_offsets[start_str_ind + i] = curr_offset;
            int32_t len = in_offsets[i + 1] - in_offsets[i];
            curr_offset += len;
        }
        out_offsets[start_str_ind + num_strs] = curr_offset;
        // copy all chars
        memcpy(out_data + start_chars_ind, in_data, num_chars);
        return;
    }

    void convert_len_arr_to_offset(uint32_t* offsets, int64_t num_strs)
    {
        uint32_t curr_offset = 0;
        for (int64_t i = 0; i < num_strs; i++)
        {
            uint32_t val = offsets[i];
            offsets[i] = curr_offset;
            curr_offset += val;
        }
        offsets[num_strs] = curr_offset;
    }

    char* getitem_string_array(uint32_t* offsets, char* data, int64_t index)
    {
        // printf("getitem string arr index: %d offsets: %d %d", index,
        //                                  offsets[index], offsets[index+1]);
        uint32_t size = offsets[index + 1] - offsets[index] + 1;
        uint32_t start = offsets[index];
        char* res = new char[size];
        res[size - 1] = '\0';
        memcpy(res, &data[start], size - 1);
        // printf(" res %s\n", res);
        return res;
    }

    void* getitem_string_array_std(uint32_t* offsets, char* data, int64_t index)
    {
        // printf("getitem string arr index: %d offsets: %d %d", index,
        //                                  offsets[index], offsets[index+1]);
        uint32_t size = offsets[index + 1] - offsets[index];
        uint32_t start = offsets[index];
        return new std::string(&data[start], size);
    }

    int str_arr_to_int64(int64_t* out, uint32_t* offsets, char* data, int64_t index)
    {
        uint32_t size = offsets[index + 1] - offsets[index];
        uint32_t start = offsets[index];
        try
        {
            *out = stoll(std::string(data + start, (std::size_t)size));
            return 0;
        }
        catch (const std::exception&)
        {
            *out = 0;
            return -1;
        }
        return -1;
    }

    int str_arr_to_float64(double* out, uint32_t* offsets, char* data, int64_t index)
    {
        uint32_t size = offsets[index + 1] - offsets[index];
        uint32_t start = offsets[index];
        try
        {
            *out = stod(std::string(data + start, (std::size_t)size));
            return 0;
        }
        catch (const std::exception&)
        {
            *out = std::nan(""); // TODO: numpy NaN
            return -1;
        }
        return -1;
    }

    bool is_na(const uint8_t* null_bitmap, int64_t i)
    {
        // printf("%d\n", *null_bitmap);
        return (null_bitmap[i / 8] & kBitmask[i % 8]) == 0;
    }

    /// @brief create a concatenated string and offset table from a pandas series of strings
    /// @note strings in returned buffer will not be 0-terminated.
    /// @param[out] buffer newly allocated buffer with concatenated strings, or NULL
    /// @param[out] no_strings number of strings concatenated, value < 0 indicates an error
    /// @param[out] offset_table newly allocated array of no_strings+1 integers
    ///                          first no_strings entries denote offsets, last entry indicates size of output array
    /// @param[in]  obj Python Sequence object, intended to be a pandas series of string
    void string_array_from_sequence(
        PyObject* obj, int64_t* no_strings, uint32_t** offset_table, char** buffer, uint8_t** null_bitmap)
    {
#define CHECK(expr, msg)                                                                                               \
    if (!(expr))                                                                                                       \
    {                                                                                                                  \
        std::cerr << msg << std::endl;                                                                                 \
        PyGILState_Release(gilstate);                                                                                  \
        if (offsets != NULL)                                                                                           \
        {                                                                                                              \
            delete[] offsets;                                                                                          \
        }                                                                                                              \
        return;                                                                                                        \
    }

        uint32_t* offsets = NULL;

        auto gilstate = PyGILState_Ensure();

        if (no_strings == NULL || offset_table == NULL || buffer == NULL)
        {
            PyGILState_Release(gilstate);
            return;
        }

        *no_strings = -1;
        *offset_table = NULL;
        *buffer = NULL;

        CHECK(PySequence_Check(obj), "expecting a PySequence");
        CHECK(no_strings && offset_table && buffer, "output arguments must not be NULL");

        Py_ssize_t n = PyObject_Size(obj);
        if (n == 0)
        {
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
        int64_t n_bytes = (n + sizeof(uint8_t) - 1) / sizeof(uint8_t);
        *null_bitmap = new uint8_t[n_bytes];
        memset(*null_bitmap, 0, n_bytes);

        // if obj is a pd.Series, get the numpy array for better performance
        // TODO: check actual Series class
        if (PyObject_HasAttrString(obj, "values"))
        {
            obj = PyObject_GetAttrString(obj, "values");
        }

        offsets = new uint32_t[n + 1];
        std::vector<const char*> tmp_store(n);
        size_t len = 0;
        for (Py_ssize_t i = 0; i < n; ++i)
        {
            offsets[i] = len;
            PyObject* s = PySequence_GetItem(obj, i);
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
                // check string
                CHECK(PyUnicode_Check(s), "expecting a string");
                // convert to UTF-8 and get size
                Py_ssize_t size;
                tmp_store[i] = PyUnicode_AsUTF8AndSize(s, &size);
                CHECK(tmp_store[i], "string conversion failed");
                len += size;
            }
            Py_DECREF(s);
        }
        offsets[n] = len;

        char* outbuf = new char[len];
        for (Py_ssize_t i = 0; i < n; ++i)
        {
            memcpy(outbuf + offsets[i], tmp_store[i], offsets[i + 1] - offsets[i]);
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
    void* np_array_from_string_array(int64_t no_strings,
                                     const uint32_t* offset_table,
                                     const char* buffer,
                                     const uint8_t* null_bitmap)
    {
#define CHECK(expr, msg)                                                                                               \
    if (!(expr))                                                                                                       \
    {                                                                                                                  \
        std::cerr << msg << std::endl;                                                                                 \
        PyGILState_Release(gilstate);                                                                                  \
        return NULL;                                                                                                   \
    }
        auto gilstate = PyGILState_Ensure();

        npy_intp dims[] = {no_strings};
        PyObject* ret = PyArray_SimpleNew(1, dims, NPY_OBJECT);
        CHECK(ret, "allocating numpy array failed");
        int err;
        PyObject* np_mod = PyImport_ImportModule("numpy");
        CHECK(np_mod, "importing numpy module failed");
        PyObject* nan_obj = PyObject_GetAttrString(np_mod, "nan");
        CHECK(nan_obj, "getting np.nan failed");

        for (int64_t i = 0; i < no_strings; ++i)
        {
            PyObject* s = PyUnicode_FromStringAndSize(buffer + offset_table[i], offset_table[i + 1] - offset_table[i]);
            CHECK(s, "creating Python string/unicode object failed");
            auto p = PyArray_GETPTR1((PyArrayObject*)ret, i);
            CHECK(p, "getting offset in numpy array failed");
            if (!is_na(null_bitmap, i))
            {
                err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, s);
            }
            else
            {
                err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, nan_obj);
            }
            CHECK(err == 0, "setting item in numpy array failed");
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

    void array_setitem(PyArrayObject* arr, char* p, PyObject* s)
    {
#define CHECK(expr, msg)                                                                                               \
    if (!(expr))                                                                                                       \
    {                                                                                                                  \
        std::cerr << msg << std::endl;                                                                                 \
        return;                                                                                                        \
    }
        // std::cout << "get array ptr " << ind << '\n';
        int err = PyArray_SETITEM(arr, p, s);
        CHECK(err == 0, "setting item in numpy array failed");
        return;
#undef CHECK
    }

    // glob support
    void c_glob(uint32_t** offsets, char** data, uint8_t** null_bitmap, int64_t* num_strings, char* path)
    {
        // std::cout << "glob: " << std::string(path) << std::endl;
        *num_strings = 0;
#ifndef _WIN32
        glob_t globBuf;
        int ret = glob(path, 0, 0, &globBuf);

        if (ret != 0)
        {
            if (ret == GLOB_NOMATCH)
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
        *offsets = new uint32_t[globBuf.gl_pathc + 1];
        size_t total_size = 0;

        for (unsigned int i = 0; i < globBuf.gl_pathc; i++)
        {
            (*offsets)[i] = (uint32_t)total_size;
            size_t curr_size = strlen(globBuf.gl_pathv[i]);
            total_size += curr_size;
        }
        (*offsets)[globBuf.gl_pathc] = (uint32_t)total_size;

        *data = new char[total_size];
        for (unsigned int i = 0; i < globBuf.gl_pathc; i++)
        {
            strcpy(*data + (*offsets)[i], globBuf.gl_pathv[i]);
        }

        // allocate null bitmap
        int64_t n_bytes = (*num_strings + sizeof(uint8_t) - 1) / sizeof(uint8_t);
        *null_bitmap = new uint8_t[n_bytes];
        memset(*null_bitmap, -1, n_bytes); // set all bits to one for non-null

        // std::cout << "glob done" << std::endl;
        globfree(&globBuf);

#else
        // TODO: support glob on Windows
        std::cerr << "no glob support on windows yet" << '\n';
#endif

        return;
    }

    void stable_argsort(char* data_ptr, uint32_t* in_offsets, int64_t len, int8_t ascending, uint64_t* result)
    {
        using str_index_pair_type = std::pair<std::string, int64_t>;
        std::vector<str_index_pair_type> str_arr_indexed;
        str_arr_indexed.reserve(len);

        for (int64_t i=0; i < len; ++i)
        {
            uint32_t start = in_offsets[i];
            uint32_t size = in_offsets[i + 1] - in_offsets[i];
            str_arr_indexed.emplace_back(
                    std::move(std::string(&data_ptr[start], size)),
                    i
            );
        }

        std::stable_sort(str_arr_indexed.begin(),
                         str_arr_indexed.end(),
                         [=](const str_index_pair_type& left, const str_index_pair_type& right){
                            if (ascending)
                                return left.first < right.first;
                            else
                                return left.first > right.first;
                         }
        );

        for (int64_t i=0; i < len; ++i)
            result[i] = str_arr_indexed[i].second;
    }

} // extern "C"

#endif

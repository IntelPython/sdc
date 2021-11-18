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

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "str_ext/sdc_str_decode.hpp"
#include "str_ext/sdc_std_string.hpp"
#include "str_ext/sdc_str_arr.hpp"
#include "str_ext/sdc_std_string_view.hpp"

#define REGISTER(func) PyObject_SetAttrString(m, #func, PyLong_FromVoidPtr((void*)(&func)));

extern "C"
{
    PyMODINIT_FUNC PyInit_hstr_ext(void)
    {
        PyObject* m;
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "hstr_ext",
            "No docs",
            -1,
            NULL,
        };
        m = PyModule_Create(&moduledef);
        if (m == NULL)
        {
            return NULL;
        }

        // init numpy
        import_array();

        /* register basic string functions defined in sdc_std_string.hpp */
        REGISTER(init_string)
        REGISTER(init_string_const)
        REGISTER(dtor_string)
        REGISTER(get_c_str)
        REGISTER(get_char_ptr)
        REGISTER(str_concat)
        REGISTER(str_compare)
        REGISTER(str_equal)
        REGISTER(str_equal_cstr)
        REGISTER(str_split)
        REGISTER(str_substr_int)
        REGISTER(get_char_from_string)
        REGISTER(std_str_to_int64)
        REGISTER(str_to_int64)
        REGISTER(str_to_float64)
        REGISTER(get_str_len)
        REGISTER(print_str)
        REGISTER(print_char)
        REGISTER(print_int)
        REGISTER(compile_regex)
        REGISTER(str_contains_noregex)
        REGISTER(str_contains_regex)
        REGISTER(str_replace_regex)
        REGISTER(str_from_int32)
        REGISTER(str_from_int64)
        REGISTER(str_from_float32)
        REGISTER(str_from_float64)
        REGISTER(del_str)
        REGISTER(hash_str)

        REGISTER(get_utf8_size)
        REGISTER(decode_utf8)
        REGISTER(unicode_to_utf8)

        /* register string array native functions defined in sdc_str_arr.hpp */
        REGISTER(dtor_string_array)
        REGISTER(dtor_str_arr_split_view)
        REGISTER(str_arr_split_view_alloc)
        REGISTER(str_arr_split_view_impl)
        REGISTER(string_array_from_sequence)
        REGISTER(np_array_from_string_array)
        REGISTER(allocate_string_array)
        REGISTER(setitem_string_array)
        REGISTER(set_string_array_range)
        REGISTER(convert_len_arr_to_offset)
        REGISTER(getitem_string_array)
        REGISTER(getitem_string_array_std)
        REGISTER(str_arr_to_int64)
        REGISTER(str_arr_to_float64)
        REGISTER(is_na)
        REGISTER(c_glob)
        REGISTER(array_size)
        REGISTER(array_getptr1)
        REGISTER(array_setitem)
        REGISTER(stable_argsort)

        /* register string array native functions defined in sdc_std_string_view.hpp */
        REGISTER(string_view_create)
        REGISTER(string_view_print)
        REGISTER(string_view_len)
        REGISTER(string_view_get_data_ptr)
        REGISTER(string_view_set_data)
        REGISTER(string_view_to_int)
        REGISTER(string_view_create_with_data)
        REGISTER(string_view_to_float64)
        return m;
    }

} // extern "C"

#undef REGISTER

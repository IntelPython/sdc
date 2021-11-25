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

#ifndef HSDC_STD_STRING_VIEW
#define HSDC_STD_STRING_VIEW

#include <iostream>
#include <string>
#include <string_view>
#include <charconv>
#include <cstring>

#include "numba/core/runtime/nrt_external.h"

extern "C"
{

    void string_view_delete(void* p_str_view)
    {
        auto p_spec = (std::string_view*)p_str_view;
        delete p_spec;
    }

    void string_view_create(NRT_MemInfo** meminfo, void* nrt_table)
    {
        auto nrt = (NRT_api_functions*)nrt_table;

        auto p_str_view = new std::string_view;
        void* res = (void*)p_str_view;
        (*meminfo) = nrt->manage_memory(res, string_view_delete);
    }

    void string_view_create_with_data(NRT_MemInfo** meminfo, void* nrt_table, void* data_ptr, int64_t size)
    {
        auto nrt = (NRT_api_functions*)nrt_table;

        auto p_str_view = new std::string_view((char*)data_ptr, size);
        void* res = (void*)p_str_view;
        (*meminfo) = nrt->manage_memory(res, string_view_delete);
    }

    void string_view_print(void* p_str_view)
    {
        auto p_spec = (std::string_view*)p_str_view;
        std::cout << (*p_spec) << std::endl;
    }

    int64_t string_view_len(void* p_str_view)
    {
        auto p_spec = (std::string_view*)p_str_view;
        return p_spec->size();
    }

    const char* string_view_get_data_ptr(void* p_str_view)
    {
        auto p_spec = (std::string_view*)p_str_view;
        return p_spec->data();
    }

    void string_view_set_data(void* p_str_view, char* data, int64_t size)
    {
        auto p_spec = (std::string_view*)p_str_view;
        std::string_view tmp(data, size);
        p_spec->swap(tmp);
    }

    int8_t string_view_to_int(void* p_str_view, int64_t base, int64_t* p_res)
    {
        auto p_spec = (std::string_view*)p_str_view;

        char* p_data = (char*)(p_spec->data());
        size_t str_len = p_spec->size();
        if (!str_len)
            return 1;

        // std::from_chars doesn't recognize "0x" prefixes, so handle this ourselves
        if (!strncmp(p_data, "0x", 2) || !strncmp(p_data, "0X", 2)) {
            str_len -= 2;
            p_data += 2;
            base = 16;
        }

        int64_t res = 0;
        auto ret = std::from_chars(p_data, p_data + str_len, res, base);
        if (ret.ptr != p_data + str_len)
        {
            // TO-DO: propagate error code to python
            // std::cerr << "wrong data" << std::endl;
            return 1;
        } else {
            *p_res = res;
            return 0;
        }
    }

    int8_t string_view_to_float64(void* p_str_view, double* p_res)
    {
        auto p_spec = (std::string_view*)p_str_view;

        size_t str_len = p_spec->size();
        if (!str_len)
            return 1;

        // std::from_chars overloads for floats are supported from GCC 11.1 only
        std::string null_term_str(*p_spec);
        auto data_ptr = null_term_str.c_str();
        char* end_pos = nullptr;
        *p_res = std::strtod(data_ptr, &end_pos);
        if (data_ptr + str_len != end_pos)
            return 1;
        else
            return 0;
    }

} // extern "C"

#endif

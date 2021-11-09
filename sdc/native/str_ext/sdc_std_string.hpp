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

#ifndef HSDC_STD_STRING
#define HSDC_STD_STRING

#include <iostream>
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

// #include "_str_decode.cpp"

using std::regex;
using std::regex_search;


extern "C"
{
    void* init_string(char* in_str, int64_t size)
    {
        // std::cout<<"init str: "<<in_str<<" "<<size<<std::endl;
        return new std::string(in_str, size);
    }

    void* init_string_const(char* in_str, int64_t size)
    {
        // std::cout<<"init str: "<<in_str<<" "<<size<<std::endl;
        return new std::string(in_str, size);
    }

    void dtor_string(std::string** in_str, int64_t size, void* info)
    {
        printf("dtor size: %ld\n", size);
        fflush(stdout);
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

    const char* get_c_str(std::string* s)
    {
        // printf("in get %s\n", s->c_str());
        return s->c_str();
    }

    const char* get_char_ptr(char c)
    {
        // printf("in get %s\n", s->c_str());
        char* str = new char[1];
        str[0] = c;
        return str;
    }

    void* str_concat(std::string* s1, std::string* s2)
    {
        // printf("in concat %s %s\n", s1->c_str(), s2->c_str());
        std::string* res = new std::string((*s1) + (*s2));
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
        return s1->compare(*s2) == 0;
    }

    bool str_equal_cstr(std::string* s1, char* s2)
    {
        // printf("in str_equal %s %s\n", s1->c_str(), s2->c_str());
        return s1->compare(s2) == 0;
    }

    void* str_split(std::string* str, std::string* sep, int64_t* size)
    {
        // std::cout << *str << " " << *sep << std::endl;
        std::vector<std::string*> res;

        size_t last = 0;
        size_t next = 0;
        while ((next = str->find(*sep, last)) != std::string::npos)
        {
            std::string* token = new std::string(str->substr(last, next - last));
            res.push_back(token);
            last = next + 1;
        }
        std::string* token = new std::string(str->substr(last));
        res.push_back(token);
        *size = res.size();
        // for(int i=0; i<*size; i++)
        //    std::cout<<*(res[i])<<std::endl;
        // TODO: avoid extra copy
        void* out = new void*[*size];
        memcpy(out, res.data(), (*size) * sizeof(void*));
        // std::cout<< *(((std::string**)(out))[1])<<std::endl;
        return out;
    }

    void* str_substr_int(std::string* str, int64_t index) { return new std::string(*str, index, 1); }

    char get_char_from_string(std::string* str, int64_t index) { return str->at(index); }

    int64_t std_str_to_int64(std::string* str) { return std::stoll(*str); }

    double str_to_float64(std::string* str) { return std::stod(*str); }

    int64_t get_str_len(std::string* str)
    {
        // std::cout << "str len called: " << *str << " " << str->length()<<std::endl;
        return str->length();
    }

    int64_t get_utf8_size(char* str, int64_t len, int kind) { return unicode_to_utf8(NULL, str, len, kind); }

    int64_t str_to_int64(char* data, int64_t length)
    {
        try
        {
            return stoll(std::string(data, (std::size_t)length));
        }
        catch (const std::exception&)
        {
            std::cerr << "invalid string to int conversion" << std::endl;
            return -1;
        }
        return -1;
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

    bool str_contains_noregex(std::string* str, std::string* pat) { return (str->find(*pat) != std::string::npos); }

    std::string* str_replace_regex(std::string* str, regex* e, std::string* val)
    {
        return new std::string(regex_replace(*str, *e, *val));
    }

    void print_str(std::string* str)
    {
        std::cout << *str;
        return;
    }

    void print_char(char c)
    {
        std::cout << c;
        return;
    }

    void print_int(int64_t val) { printf("%ld\n", val); }

    void* str_from_int32(int in) { return new std::string(std::to_string(in)); }

    void* str_from_int64(int64_t in) { return new std::string(std::to_string(in)); }

    void* str_from_float32(float in) { return new std::string(std::to_string(in)); }

    void* str_from_float64(double in) { return new std::string(std::to_string(in)); }

} // extern "C"

#endif

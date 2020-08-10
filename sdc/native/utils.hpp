// *****************************************************************************
// Copyright (c) 2020, Intel Corporation All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//     Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
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
// *****************************************************************************

#pragma once

#include <cstdint>
#include <algorithm>
#include "tbb/task_arena.h"

namespace utils
{

using quant = int8_t;

using compare_func = bool (*)(const void*, const void*);

template<uint64_t ItemSize>
struct byte_range
{
    using data_type = std::array<quant, ItemSize>;

    byte_range(void* begin, uint64_t len)
    {
        _begin = reinterpret_cast<data_type*>(begin);
        _end   = std::next(_begin, len);
    }

    data_type* begin() { return _begin; }
    data_type* end()   { return _end; }

private:
    data_type* _begin;
    data_type* _end;
};

template<class Compare, int ItemSize>
struct ExternalCompare
{
    using data_type = typename byte_range<ItemSize>::data_type;

    ExternalCompare() {}

    ExternalCompare(const Compare in_cmp): cmp(in_cmp) {}

    bool operator() (const data_type& left, const data_type& right) const
    {
        return cmp(left.data(), right.data());
    }

    const Compare cmp = {};
};

template<class Data, class Compare>
struct IndexCompare
{
    IndexCompare() {}

    IndexCompare(Data* in_data, const Compare in_cmp = {}): cmp(in_cmp) {}

    template<typename index_type>
    bool operator() (const index_type& left, const index_type& right) const
    {
        return cmp(data[left], data[right]);
    }

    Data* data  = nullptr;
    Compare cmp = {};
};

tbb::task_arena& get_arena();

void set_threads_num(uint64_t);

template<int N> struct index {};

template<int N>
struct static_loop
{
    template<typename Body>
    void operator()(Body&& body)
    {
        static_loop<N - 1>()(body);
        body(index<N - 1>());
    }
};

template<>
struct static_loop<0>
{
    template<typename Body>
    void operator()(Body&& body)
    {
    }
};

} // namespace

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
#include <memory>
#include <cmath>
#include <type_traits>
#include "tbb/task_arena.h"
#include "tbb/tbb.h"

#define HAS_TASK_SCHEDULER_INIT (TBB_INTERFACE_VERSION < 12002)
#define HAS_TASK_SCHEDULER_HANDLE (TBB_INTERFACE_VERSION >= 12003)
#define SUPPORTED_TBB_VERSION (HAS_TASK_SCHEDULER_INIT || HAS_TASK_SCHEDULER_HANDLE)

namespace utils
{

using quant = int8_t;

using compare_func       = bool (*)(const void*, const void*);
using parallel_sort_call = void (*)(void*, uint64_t, compare_func);

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
    IndexCompare() { }

    IndexCompare(Data* in_data, const Compare in_cmp = {}): data(in_data), cmp(in_cmp) { }

    template<typename index_type>
    bool operator() (const index_type& left, const index_type& right) const
    {
        return cmp(data[left], data[right]);
    }

    Data* data  = nullptr;
    Compare cmp = {};
};

template<class Compare>
struct IndexCompare<void, Compare>
{
    IndexCompare() {}

    IndexCompare(void* in_data, uint64_t in_size, const Compare in_cmp):
        data(in_data), size(in_size), cmp(in_cmp)
    {
    }

    template<typename index_type>
    bool operator() (const index_type& left, const index_type& right) const
    {
        void* left_data  = &reinterpret_cast<quant*>(data)[size*left];
        void* right_data = &reinterpret_cast<quant*>(data)[size*right];
        return cmp(left_data, right_data);
    }

    void*    data = nullptr;
    uint64_t size = 0;
    Compare  cmp  = {};
};

template<int N> struct index {};

template<int N>
struct static_loop
{
    template<typename Body>
    void operator()(Body&& body)
    {
        static_loop<N - 1>()(body);
        body(utils::index<N - 1>());
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

template<int N, template<int> typename function_call>
struct fill_array_body
{
    fill_array_body(std::array<parallel_sort_call, N>& in_arr): arr(in_arr) {}

    template<int K>
    void operator() (utils::index<K>)
    {
        arr[K] = function_call<K+1>::call;
    }

    std::array<parallel_sort_call, N>& arr;
};

template<int N, template<int> typename function_call>
std::array<parallel_sort_call, N> fill_parallel_sort_array()
{
    auto result = std::array<parallel_sort_call, N>();

    static_loop<N>()(fill_array_body<N, function_call>(result));

    return result;
}

template<typename T>
void parallel_copy(T* src, T* dst, uint64_t len)
{
    using range_t = tbb::blocked_range<uint64_t>;
    tbb::parallel_for(range_t(0,len), [src, dst](const range_t& range)
    {
        for (auto i = range.begin(); i < range.end(); ++i)
            dst[i] = src[i];
    });
}

void parallel_copy(void* src, void* dst, uint64_t len, uint64_t size);

template<typename T>
void fill_index_parallel(T* index, T len)
{
    using range_t = tbb::blocked_range<T>;

    tbb::parallel_for(range_t(0,len), [index](const range_t& range)
    {
        for (auto i = range.begin(); i < range.end(); ++i)
            index[i] = i;
    });
}

template<typename I>
void reorder(void* src, I* index, uint64_t len, uint64_t size, void* dst)
{
    using range_t = tbb::blocked_range<uint64_t>;
    tbb::parallel_for(range_t(0,len), [src, index, dst, size](const range_t& range)
    {
        auto r_src = reinterpret_cast<quant*>(src) + range.begin()*size;
        auto r_dst = reinterpret_cast<quant*>(dst) + range.begin()*size;

        for (auto i = range.begin(); i < range.end(); ++i)
            std::copy_n(&r_src[index[i]*size], size, &r_dst[i*size]);
    });
}

template<typename T, typename I>
void reorder(T* src, T* index, uint64_t len, T* dst)
{
    using range_t = tbb::blocked_range<uint64_t>;
    tbb::parallel_for(range_t(0,len), [src, index, dst](const range_t& range)
    {
        for (auto i = range.begin(); i < range.end(); ++i)
            dst[i] = src[index[i]];
    });
}

template<typename T, typename I>
void reorder(T* data, T* index, uint64_t len)
{
    std::unique_ptr<T[]> temp(new T[len]);

    parallel_copy(data, temp.get(), len);
    reorder(temp.get(), index, len, data);
}

template<typename I>
void reorder(void* data, I* index, uint64_t len, uint64_t size)
{
    std::unique_ptr<quant[]> temp(new quant[len*size]);

    parallel_copy(data, temp.get(), len, size);
    reorder(temp.get(), index, len, size, data);
}

template<class I, typename Argsort>
void sort_by_argsort(void* data, uint64_t len, uint64_t size, compare_func cmp, Argsort argsort)
{
    std::unique_ptr<I[]> index(new I[len]);

    argsort(index.get(), data, len, size, cmp);
    reorder(data, index.get(), len, size);
}

template<typename T>
bool nanless(const T& left, const T& right)
{
    return std::less<T>()(left, right);
}

template<>
bool nanless<float>(const float& left, const float& right);

template<>
bool nanless<double>(const double& left, const double& right);

template<typename T>
struct less
{
    bool operator() (const T& left, const T& right) const
    {
        return nanless<T>(left, right);
    }
};

namespace tbb_control
{
    void init();

    tbb::task_arena& get_arena();

    void set_threads_num(uint64_t);

    void finalize();
}

} // namespace

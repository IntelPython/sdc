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

#include "utils.hpp"
#include "tbb/parallel_invoke.h"
#include <iostream>
#include <array>

using namespace utils;

namespace
{

template<class T>
struct buffer_queue
{
    using v_type = T;
    v_type* head;
    v_type* tail;

    buffer_queue(v_type* _head, int size)
    {
        head = _head;
        tail = head + size;
    }

    inline v_type* pop() { return head++; }

    inline bool not_empty() { return head < tail; }

    inline void push(v_type* val) { *(tail++) = *val; }

    inline uint64_t size() { return tail - head; }

    inline int copy_size() { return size(); }
};

template<class T, class Compare = std::less<T>>
inline void merge_sorted_main_loop(buffer_queue<T>& left, buffer_queue<T>& right, buffer_queue<T>& out, const Compare& compare = Compare())
{
    while (left.not_empty() && right.not_empty())
    {
        if (compare(*right.head, *left.head))
            out.push(right.pop());
        else
            out.push(left.pop());
    }
}

template<class T, class Compare = std::less<T>>
void merge_sorted(T* left, int left_size, T* right, int right_size, T* out, const Compare& compare = Compare())
{
    auto left_buffer  = buffer_queue<T>(left,  left_size);
    auto right_buffer = buffer_queue<T>(right, right_size);

    auto out_buffer = buffer_queue<T>(out, 0);

    merge_sorted_main_loop(left_buffer, right_buffer, out_buffer, compare);

    // only one buffer still have items, don't need to shift out_buffer.tail
    std::copy_n(left_buffer.head, left_buffer.copy_size(), out_buffer.tail);

    if (out_buffer.tail != right_buffer.head)
        std::copy_n(right_buffer.head, right_buffer.copy_size(), out_buffer.tail);
}

template<class T, class Compare = std::less<T>>
void merge_sorted_parallel(T* left, int left_size, T* right, int right_size, T* out, const Compare& compare = Compare())
{
    auto split = [](T* first, int f_size, T* second, int s_size, T* out, const Compare& compare = Compare())
    {
        auto f_middle_pos = f_size/2;

        auto first_middle = std::next(first,  f_middle_pos);
        auto second_end   = std::next(second, s_size);

        auto second_middle = std::upper_bound(second, second_end, *first_middle, compare);

        auto s_middle_pos = std::distance(second, second_middle);

        auto out_middle = std::next(out, f_middle_pos + s_middle_pos);

        tbb::parallel_invoke(
            [&] () { merge_sorted_parallel<T>(first, f_middle_pos, second, s_middle_pos, out, compare); },
            [&] () { merge_sorted_parallel<T>(first_middle, f_size - f_middle_pos, second_middle, s_size - s_middle_pos, out_middle, compare); }
        );
    };

    auto constexpr limit = 512;
    if (left_size >= right_size && left_size > limit)
    {
        split(left, left_size, right, right_size, out, compare);
    }
    else if (left_size < right_size && right_size > limit)
    {
        split(right, right_size, left, left_size, out, compare);
    }
    else
    {
        merge_sorted<T>(left, left_size, right, right_size, out, compare);
    }
}

template<class T, class Compare = std::less<T>>
void stable_sort_inner_sort(T* data, int begin, int end, const Compare& compare = Compare())
{
    std::stable_sort(data + begin, data + end, compare);
}


template<class T, class Compare = std::less<T>>
T* stable_sort_impl(T* data, T* temp, int begin, int end, const Compare& compare = Compare())
{
    auto constexpr limit = 512;
    if (end - begin <= limit)
    {
        stable_sort_inner_sort<T>(data, begin, end, compare);

        return data;
    }
    auto middle = begin + (end - begin) / 2;

    T* left = nullptr;
    T* right = nullptr;

    tbb::parallel_invoke(
        [&] () { left  = stable_sort_impl<T>(data, temp, begin,  middle, compare); },
        [&] () { right = stable_sort_impl<T>(data, temp, middle, end,    compare); }
    );

    auto out = data;

    if (left == data)
        out = temp;

    merge_sorted_parallel<T>(std::next(left, begin),
                             middle - begin,
                             std::next(right, middle),
                             end - middle,
                             std::next(out, begin),
                             compare);

    return out;
}

template<class T, class Compare = std::less<T>>
void parallel_stable_sort_(T* data, uint64_t len, const Compare& compare = Compare())
{
    std::unique_ptr<T[]> temp(new T[len]);

    T* result = nullptr;

    get_arena().execute([&]()
    {
        result = stable_sort_impl<T>(data, temp.get(), 0, len, compare);
    });

    if (result == temp.get())
    {
        std::copy_n(result, len, data);
    }
}

template<class I, class T, class Compare = std::less<T>>
void parallel_stable_argsort_(I* index, T* data, uint64_t len, const Compare& compare = Compare())
{
    for (I i = 0; i < len; ++i)
        index[i] = i;

    parallel_stable_sort_(index, len, IndexCompare<T, Compare>(data, compare));
}

template<int ItemSize>
struct parallel_sort_fixed_size
{
    static void call(void* begin, uint64_t len, compare_func cmp)
    {
        auto range = byte_range<ItemSize>(begin, len);
        auto compare = ExternalCompare<compare_func, ItemSize>(cmp);
        parallel_stable_sort_<typename byte_range<ItemSize>::data_type, ExternalCompare<compare_func, ItemSize>>(range.begin(), len, compare);
    }
};

using parallel_sort_call = void(*)(void* begin, uint64_t len, compare_func cmp);

template<int N, template<int> typename function_call>
struct fill_array_body
{
    fill_array_body(std::array<parallel_sort_call, N>& in_arr): arr(in_arr) {}

    template<int K>
    void operator() (index<K>)
    {
        arr[K] = function_call<K>::call;
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

} // namespace

#define declare_single_argsort(index_prefix, type_prefix, ity, ty) \
void parallel_stable_argsort_##index_prefix##type_prefix(void* index, void* begin, uint64_t len) \
{ parallel_stable_argsort_<ity, ty>(reinterpret_cast<ity*>(index), reinterpret_cast<ty*>(begin), len); }

#define declare_argsort(prefix, ty) \
declare_single_argsort(u8,  prefix, uint8_t,  ty) \
declare_single_argsort(u16, prefix, uint16_t, ty) \
declare_single_argsort(u32, prefix, uint32_t, ty) \
declare_single_argsort(u64, prefix, uint64_t, ty)

#define declare_sort(prefix, ty) \
void parallel_stable_sort_##prefix(void* begin, uint64_t len) { parallel_stable_sort_<ty>(reinterpret_cast<ty*>(begin), len); } \
declare_argsort(prefix, ty)

#define declare_int_sort(bits) \
declare_sort(i##bits, int##bits##_t) \
declare_sort(u##bits, uint##bits##_t)

extern "C"
{

declare_int_sort(8)
declare_int_sort(16)
declare_int_sort(32)
declare_int_sort(64)

declare_sort(f32, float)
declare_sort(f64, double)

void parallel_stable_sort(void* begin, uint64_t len, uint64_t size, void* compare)
{
    static const constexpr auto MaxFixSize = 32;
    static const std::array<parallel_sort_call, MaxFixSize> fixed_size_sort = fill_parallel_sort_array<MaxFixSize, parallel_sort_fixed_size>();

    auto cmp = reinterpret_cast<compare_func>(compare);
    if (size <= MaxFixSize)
        return fixed_size_sort[size - 1](begin, len, cmp);
}

}

#undef declare_int_sort
#undef declare_sort

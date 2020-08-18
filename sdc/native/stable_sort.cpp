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
#include <vector>

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

    inline bool not_empty() const { return head < tail; }

    inline void push(v_type* val) { *(tail++) = *val; }

    inline uint64_t size() const { return tail - head; }

    inline int copy_size() const { return size(); }
};

template<class T, class Compare = utils::less<T>>
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

template<class T, class Compare = utils::less<T>>
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

template<class T, class Compare = utils::less<T>>
void merge_sorted_parallel(T* left, int left_size, T* right, int right_size, T* out, const Compare& compare = Compare())
{
    auto split = [](T* first, int f_size, T* second, int s_size, T* out, const Compare& compare = Compare(), bool swap = false)
    {
        auto f_middle_pos = f_size/2;

        auto first_middle = std::next(first,  f_middle_pos);
        auto first_end    = std::next(first,  f_size);
        auto second_end   = std::next(second, s_size);

        const auto& first_middle_value = *first_middle;

        auto equal = [](const T& left, const T& right, const Compare& compare)
        {
            return !compare(left, right) && !compare(right, left);
        };

        if (std::next(first_middle) != first_end && equal(*std::next(first_middle), *first_middle, compare))
        {
            first_middle = std::upper_bound(first,  first_end, first_middle_value, compare);
        }
        else
        {
            first_middle = std::next(first_middle);
        }

        f_middle_pos = std::distance(first, first_middle);

        auto second_middle = std::upper_bound(second, second_end, first_middle_value, compare);

        decltype(f_middle_pos) s_middle_pos = std::distance(second, second_middle);

        auto out_middle = std::next(out, f_middle_pos + s_middle_pos);
        auto out_end    = std::next(out, f_size + s_size);

        // in order to keep order, it is import to pass 'left' buffer as
        // first parameter to merge_sorted_parallel.
        // So, if 'first' is actually 'right' buffer, we must swap them back
        if (swap)
        {
            std::swap(first, second);
            std::swap(f_middle_pos, s_middle_pos);
            std::swap(f_size, s_size);
            std::swap(first_middle, second_middle);
            std::swap(first_end, second_end);
        }

        if (((first_middle == first_end) &&
            (second_middle == second_end)) ||
            (second >= out && second <= out_end))
        {
            merge_sorted(first, f_size, second, s_size, out, compare);
        }
        else
        {
            tbb::parallel_invoke(
                [&] () { merge_sorted_parallel(first, f_middle_pos, second, s_middle_pos, out, compare); },
                [&] () { merge_sorted_parallel(first_middle, f_size - f_middle_pos, second_middle, s_size - s_middle_pos, out_middle, compare); }
            );
        }
    };

    auto constexpr limit = 512;
    if (left_size == 0)
    {
        parallel_copy(right, out, right_size);
    }
    else if (right_size == 0)
    {
        parallel_copy(left, out, left_size);
    }
    else if (left_size >= right_size && left_size > limit)
    {
        split(left, left_size, right, right_size, out, compare);
    }
    else if (left_size < right_size && right_size > limit)
    {
        split(right, right_size, left, left_size, out, compare, true);
    }
    else
    {
        merge_sorted(left, left_size, right, right_size, out, compare);
    }
}

template<class T, class Compare = utils::less<T>>
void stable_sort_inner_sort(T* data, int begin, int end, const Compare& compare = Compare())
{
    std::stable_sort(data + begin, data + end, compare);
}


template<class T, class Compare = utils::less<T>>
T* stable_sort_impl(T* data, T* temp, int begin, int end, const Compare& compare = Compare())
{
    auto constexpr limit = 512;
    if (end - begin <= limit)
    {
        stable_sort_inner_sort(data, begin, end, compare);

        return data;
    }
    auto middle = begin + (end - begin) / 2;

    T* left = nullptr;
    T* right = nullptr;

    tbb::parallel_invoke(
        [&] () { left  = stable_sort_impl(data, temp, begin,  middle, compare); },
        [&] () { right = stable_sort_impl(data, temp, middle, end,    compare); }
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

template<class T, class Compare = utils::less<T>>
void parallel_stable_sort_(T* data, uint64_t len, const Compare& compare = Compare())
{
    std::unique_ptr<T[]> temp(new T[len]);

    T* result = nullptr;

    get_arena().execute([&]()
    {
        result = stable_sort_impl(data, temp.get(), 0, static_cast<int>(len), compare);
    });

    if (result == temp.get())
    {
        parallel_copy(result, data, len);
    }
}

template<class I, class Compare>
void parallel_stable_argsort__(I* index,
                               uint64_t len,
                               const Compare& compare)
{
    fill_index_parallel(index, static_cast<I>(len));
    parallel_stable_sort_(index, len, compare);
}

template<class I, class T, class Compare = utils::less<T>>
void parallel_stable_argsort_(I* index,
                              T* data,
                              uint64_t len,
                              const Compare& compare = Compare())
{
    parallel_stable_argsort__(index, len, IndexCompare<T, Compare>(data, compare));
}

template<class I>
void parallel_stable_argsort_(I* index, void* data, uint64_t len, uint64_t size, compare_func compare)
{
    using comparator_t = IndexCompare<void, compare_func>;
    auto comparator = comparator_t(data, size, compare);

    parallel_stable_argsort__(index, len, comparator);
}

template<int ItemSize>
struct parallel_sort_fixed_size
{
    static void call(void* begin, uint64_t len, compare_func cmp)
    {
        using comparator_t = ExternalCompare<compare_func, ItemSize>;

        auto range      = byte_range<ItemSize>(begin, len);
        auto comparator = comparator_t(cmp);
        parallel_stable_sort_(range.begin(), len, comparator);
    }
};

} // namespace

#define declare_single_argsort(index_prefix, type_prefix, ity, ty) \
void parallel_stable_argsort_##index_prefix##type_prefix(ity* index, void* begin, uint64_t len) \
{ parallel_stable_argsort_(reinterpret_cast<ity*>(index), reinterpret_cast<ty*>(begin), len); }

#define declare_argsort(prefix, ty) \
declare_single_argsort(u8,  prefix, uint8_t,  ty) \
declare_single_argsort(u16, prefix, uint16_t, ty) \
declare_single_argsort(u32, prefix, uint32_t, ty) \
declare_single_argsort(u64, prefix, uint64_t, ty)

#define declare_generic_argsort(prefix, ity) \
void parallel_stable_argsort_##prefix##v(void* index, void* begin, uint64_t len, uint64_t size, void* compare) \
{ \
    auto cmp = reinterpret_cast<compare_func>(compare); \
    parallel_stable_argsort_(reinterpret_cast<ity*>(index), begin, len, size, cmp); \
}

#define declare_sort(prefix, ty) \
void parallel_stable_sort_##prefix(void* begin, uint64_t len) \
{ parallel_stable_sort_(reinterpret_cast<ty*>(begin), len); } \
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

declare_generic_argsort(u8,  uint8_t)
declare_generic_argsort(u16, uint16_t)
declare_generic_argsort(u32, uint32_t)
declare_generic_argsort(u64, uint64_t)

void parallel_stable_sort(void* begin, uint64_t len, uint64_t size, void* compare)
{
    static const constexpr auto MaxFixSize = 32;
    static const std::array<parallel_sort_call, MaxFixSize> fixed_size_sort = fill_parallel_sort_array<MaxFixSize, parallel_sort_fixed_size>();

    auto cmp = reinterpret_cast<compare_func>(compare);
    if (size <= MaxFixSize)
        return fixed_size_sort[size - 1](begin, len, cmp);

    return sort_by_argsort<uint64_t>(begin, len, size, cmp, parallel_stable_argsort_<uint64_t>);
}

}

#undef declare_int_sort
#undef declare_sort
#undef declare_argsort
#undef declare_single_argsort
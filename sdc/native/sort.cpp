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

#include <cstdint>
#include <array>

#include "utils.hpp"
#include "tbb/parallel_sort.h"

#include <iostream>


using namespace utils;

namespace
{

template<typename T, class Compare = utils::less<T>>
void parallel_sort_(T* begin, uint64_t len, const Compare compare = Compare())
{
    tbb::parallel_sort(begin, begin + len, compare);
}

template<int ItemSize>
struct parallel_sort_fixed_size
{
    static void call(void* begin, uint64_t len, compare_func cmp)
    {
        using comparator_t = ExternalCompare<compare_func, ItemSize>;

        auto range      = byte_range<ItemSize>(begin, len);
        auto comparator = comparator_t(cmp);
        parallel_sort_(range.begin(), len, comparator);
    }
};

template<class I, class Compare>
void parallel_argsort__(I* index,
                        uint64_t len,
                        const Compare& compare)
{
    fill_index_parallel(index, static_cast<I>(len));
    parallel_sort_(index, len, compare);
}

template<class I, class T, class Compare = utils::less<T>>
void parallel_argsort_(I* index,
                       T* data,
                       uint64_t len,
                       const Compare& compare = Compare())
{
    parallel_argsort__(index, len, IndexCompare<T, Compare>(data, compare));
}

template<class I>
void parallel_argsort_(I* index, void* data, uint64_t len, uint64_t size, compare_func compare)
{
    using comparator_t = IndexCompare<void, compare_func>;
    auto comparator = comparator_t(data, size, compare);

    parallel_argsort__(index, len, comparator);
}

} // namespace

#define declare_single_argsort(index_prefix, type_prefix, ity, ty) \
void parallel_argsort_##index_prefix##type_prefix(void* index, void* begin, uint64_t len) \
{ parallel_argsort_(reinterpret_cast<ity*>(index), reinterpret_cast<ty*>(begin), len); }

#define declare_argsort(prefix, ty) \
declare_single_argsort(u8,  prefix, uint8_t,  ty) \
declare_single_argsort(u16, prefix, uint16_t, ty) \
declare_single_argsort(u32, prefix, uint32_t, ty) \
declare_single_argsort(u64, prefix, uint64_t, ty)

#define declare_generic_argsort(prefix, ity) \
void parallel_argsort_##prefix##v(void* index, void* begin, uint64_t len, uint64_t size, void* compare) \
{ \
    auto cmp = reinterpret_cast<compare_func>(compare); \
    parallel_argsort_(reinterpret_cast<ity*>(index), begin, len, size, cmp); \
}

#define declare_sort(prefix, ty) \
void parallel_sort_##prefix(void* begin, uint64_t len) \
{ parallel_sort_(reinterpret_cast<ty*>(begin), len); } \
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

void parallel_sort(void* begin, uint64_t len, uint64_t size, void* compare)
{
    static const constexpr auto MaxFixSize = 32;
    static const std::array<parallel_sort_call, MaxFixSize> fixed_size_sort = fill_parallel_sort_array<MaxFixSize, parallel_sort_fixed_size>();

    auto cmp = reinterpret_cast<compare_func>(compare);
    if (size <= MaxFixSize)
        return fixed_size_sort[size - 1](begin, len, cmp);

    return sort_by_argsort<uint64_t>(begin, len, size, cmp, parallel_argsort_<uint64_t>);
}

}

#undef declare_int_sort
#undef declare_sort
#undef declare_argsort
#undef declare_single_argsort

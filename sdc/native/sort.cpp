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
#include "tbb/parallel_sort.h"

#include <iostream>


using namespace utils;

namespace
{

template<typename T>
void parallel_sort_(void* begin, uint64_t len)
{
    auto _begin = reinterpret_cast<T*>(begin);
    auto _end   = _begin + len;

    tbb::parallel_sort(_begin, _end);
}

} // namespace

#define declare_sort(prefix, ty) \
void parallel_sort_##prefix(void* begin, uint64_t len) { parallel_sort_<ty>(begin, len); }

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

void parallel_sort(void* begin, uint64_t len, uint64_t size, void* compare)
{
    auto compare_f = reinterpret_cast<compare_func>(compare);

#define run_sort(range_type) \
{ \
    auto range  = range_type(begin, len, size); \
    auto _begin = range.begin(); \
    auto _end   = range.end(); \
    utils::get_arena().execute([&]() \
    { \
        tbb::parallel_sort(_begin, _end, compare_f); \
    }); \
}

    switch(size)
    {
    case 1:
        run_sort(exact_void_range<1>);
        break;
    case 2:
        run_sort(exact_void_range<2>);
        break;
    case 4:
        run_sort(exact_void_range<4>);
        break;
    case 8:
        run_sort(exact_void_range<8>);
        break;
    default:
        // fallback to c qsort?
        if      (size <= 4)    run_sort(_void_range<4>)
        else if (size <= 8)    run_sort(_void_range<8>)
        else if (size <= 16)   run_sort(_void_range<16>)
        else if (size <= 32)   run_sort(_void_range<32>)
        else if (size <= 64)   run_sort(_void_range<64>)
        else if (size <= 128)  run_sort(_void_range<128>)
        else if (size <= 256)  run_sort(_void_range<256>)
        else if (size <= 512)  run_sort(_void_range<512>)
        else if (size <= 1024) run_sort(_void_range<1024>)
        else
        {
            std::cout << "Unsupported item size " << size << std::endl;
            abort();
        }
        break;
    }

#undef run_sort
}

}

#undef declare_int_sort
#undef declare_sort

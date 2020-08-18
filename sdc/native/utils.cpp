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
#include "tbb/task_arena.h"
#include "tbb/tbb.h"
#include <memory>
#include <iostream>

#define HAS_TASK_SCHEDULER_INIT (TBB_INTERFACE_VERSION < 12002)
#define HAS_TASK_SCHEDULER_HANDLE (TBB_INTERFACE_VERSION >= 12003)

namespace utils
{

using arena_ptr = std::unique_ptr<tbb::task_arena>;

#if HAS_TASK_SCHEDULER_INIT
using tsi_ptr = std::unique_ptr<tbb::task_scheduler_init>;
void ignore_assertion( const char*, int, const char*, const char * ) {}
#elif HAS_TASK_SCHEDULER_HANDLE
using tsh_ptr = tbb::task_scheduler_handle;
#else
        #error("Unsupported version of TBB")
#endif

struct tbb_context
{
#if HAS_TASK_SCHEDULER_INIT
    tsi_ptr   tsi;
#elif HAS_TASK_SCHEDULER_HANDLE
    tsh       tsh;
#else
        #error("Unsupported version of TBB")
#endif

    arena_ptr arena;

    tbb_context()
    {
#if HAS_TASK_SCHEDULER_INIT
        tsi.reset(new tbb::task_scheduler_init(tbb::task_arena::automatic));
#elif HAS_TASK_SCHEDULER_HANDLE
        tsh = tbb::task_scheduler_handle::get();
#else
        #error("Unsupported version of TBB")
#endif
        arena.reset(new tbb::task_arena());
    }

    void set_threads_num(uint64_t threads)
    {
        arena->terminate();
        arena->initialize(threads);
    }

    void finalize()
    {
        if (arena)
            return;

        arena->terminate();
        arena.reset();
#if HAS_TASK_SCHEDULER_INIT
        auto orig = tbb::set_assertion_handler(ignore_assertion);
        tsi->terminate(); // no blocking terminate is needed here
        tsi.reset();
        tbb::set_assertion_handler(orig);
#elif HAS_TASK_SCHEDULER_HANDLE
        (void)tbb::finalize(tsh, std::nothrow);
#else
        #error("Unsupported version of TBB")
#endif
    }

    ~tbb_context()
    {
        finalize();
    }
};

tbb_context& get_tbb_context()
{
    static tbb_context* context = new tbb_context();

    return *context;
}

tbb::task_arena& get_arena()
{
    auto& context = get_tbb_context();
    return *context.arena;
}

void set_threads_num(uint64_t threads)
{
    auto& context = get_tbb_context();
    context.set_threads_num(threads);
}

void finalize_tbb(void*)
{
    auto& context = get_tbb_context();
    context.finalize();
    delete &context;
}

void parallel_copy(void* src, void* dst, uint64_t len, uint64_t size)
{
    using range_t = tbb::blocked_range<uint64_t>;
    tbb::parallel_for(range_t(0,len), [src, dst, size](const range_t& range)
    {
        auto r_src = reinterpret_cast<quant*>(src) + range.begin()*size;
        auto r_dst = reinterpret_cast<quant*>(dst) + range.begin()*size;
        std::copy_n(r_src, range.size()*size, r_dst);
    });
}

template<>
bool nanless<float>(const float& left, const float& right)
{
    return std::less<float>()(left, right) || (isnan(right) && !isnan(left));
}

template<>
bool nanless<double>(const double& left, const double& right)
{
    return std::less<double>()(left, right) || (isnan(right) && !isnan(left));
}

}

//*****************************************************************************
// Copyright (c) 2020, Intel Corporation All rights reserved.
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

#ifndef _DISTRIBUTED_H_INCLUDED
#define _DISTRIBUTED_H_INCLUDED

#include <Python.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdbool.h>
#include <tuple>
#include <vector>

#include "_hpat_common.h"

#define ROOT_PE 0

// XXX same as distributed_api.py:Reduce_Type
struct SDC_ReduceOps
{
    enum ReduceOpsEnum
    {
        SUM = 0,
        PROD = 1,
        MIN = 2,
        MAX = 3,
        ARGMIN = 4,
        ARGMAX = 5,
        OR = 6
    };
};

static int64_t hpat_dist_get_start(int64_t total, int num_pes, int node_id) __UNUSED__;
static int64_t hpat_dist_get_end(int64_t total, int num_pes, int node_id) __UNUSED__;
static int64_t hpat_dist_get_node_portion(int64_t total, int num_pes, int node_id) __UNUSED__;
static int get_elem_size(int type_enum) __UNUSED__;

static int64_t hpat_dist_get_item_pointer(int64_t ind, int64_t start, int64_t count) __UNUSED__;

static int hpat_dummy_ptr[64] __UNUSED__;

/* *********************************************************************
************************************************************************/

static void* hpat_get_dummy_ptr() __UNUSED__;
static void* hpat_get_dummy_ptr()
{
    return hpat_dummy_ptr;
}

static int64_t hpat_dist_get_start(int64_t total, int num_pes, int node_id)
{
    int64_t div_chunk = (int64_t)ceil(total / ((double)num_pes));
    int64_t start = std::min(total, node_id * div_chunk);
    // printf("rank %d start:%lld\n", node_id, start);
    return start;
}

static int64_t hpat_dist_get_end(int64_t total, int num_pes, int node_id)
{
    int64_t div_chunk = (int64_t)ceil(total / ((double)num_pes));
    int64_t end = std::min(total, (node_id + 1) * div_chunk);
    // printf("rank %d end:%lld\n", node_id, end);
    return end;
}

static int64_t hpat_dist_get_node_portion(int64_t total, int num_pes, int node_id)
{
    return hpat_dist_get_end(total, num_pes, node_id) - hpat_dist_get_start(total, num_pes, node_id);
}

static int64_t index_rank(int64_t total, int num_pes, int index)
{
    int64_t div_chunk = (int64_t)ceil(total / ((double)num_pes));
    return index / div_chunk;
}

static int get_elem_size(int type_enum)
{
    if (type_enum < 0 || type_enum > 7)
    {
        std::cerr << "Invalid MPI_Type\n";
        return 8;
    }
    int types_sizes[] = {1, 1, 4, 4, 8, 4, 8, 8};
    return types_sizes[type_enum];
}

static int64_t hpat_dist_get_item_pointer(int64_t ind, int64_t start, int64_t count)
{
    // printf("ind:%lld start:%lld count:%lld\n", ind, start, count);
    if (ind >= start && ind < start + count)
    {
        return ind - start;
    }
    return -1;
}

// Given the permutation index |p| and |rank|, and the number of ranks
// |num_ranks|, finds the destination ranks of indices of the |rank|.  For
// example, if |rank| is 1, |num_ranks| is 3, |p_len| is 12, and |p| is the
// following array [ 9, 8, 6, 4, 11, 7, 2, 3, 5, 0, 1, 10], the function returns
// [0, 2, 0, 1].
static inline std::vector<int64_t> find_dest_ranks(int64_t rank, int64_t num_ranks, int64_t* p, int64_t p_len)
{
    auto chunk_size = hpat_dist_get_node_portion(p_len, num_ranks, rank);
    auto begin = hpat_dist_get_start(p_len, num_ranks, rank);
    std::vector<int64_t> dest_ranks(chunk_size);

    for (auto i = 0; i < p_len; ++i)
        if (rank == index_rank(p_len, num_ranks, p[i]))
            dest_ranks[p[i] - begin] = index_rank(p_len, num_ranks, i);
    return dest_ranks;
}

static inline std::vector<int>
    find_send_counts(const std::vector<int64_t>& dest_ranks, int64_t num_ranks, int64_t elem_size)
{
    std::vector<int> send_counts(num_ranks);
    for (auto dest : dest_ranks)
        ++send_counts[dest];
    return send_counts;
}

static inline std::vector<int> find_disps(const std::vector<int>& counts)
{
    std::vector<int> disps(counts.size());
    for (size_t i = 1; i < disps.size(); ++i)
        disps[i] = disps[i - 1] + counts[i - 1];
    return disps;
}

static inline std::vector<int>
    find_recv_counts(int64_t rank, int64_t num_ranks, int64_t* p, int64_t p_len, int64_t elem_size)
{
    auto begin = hpat_dist_get_start(p_len, num_ranks, rank);
    auto end = hpat_dist_get_end(p_len, num_ranks, rank);
    std::vector<int> recv_counts(num_ranks);
    for (auto i = begin; i < end; ++i)
        ++recv_counts[index_rank(p_len, num_ranks, p[i])];
    return recv_counts;
}

// Returns an |index_array| which would sort the array |v| of size |len| when
// applied to it.  Identical to numpy.argsort.
template <class T>
static std::vector<size_t> arg_sort(T* v, int64_t len)
{
    std::vector<size_t> index_array(len);
    std::iota(index_array.begin(), index_array.end(), 0);
    std::sort(index_array.begin(), index_array.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    return index_array;
}

// |v| is an array of elements of size |elem_size|.  This function swaps
// elements located at indices |i1| and |i2|.
static void elem_swap(unsigned char* v, int64_t elem_size, size_t i1, size_t i2)
{
    std::vector<unsigned char> tmp(elem_size);
    auto i1_offset = v + i1 * elem_size;
    auto i2_offset = v + i2 * elem_size;
    std::copy(i1_offset, i1_offset + elem_size, tmp.data());
    std::copy(i2_offset, i2_offset + elem_size, i1_offset);
    std::copy(std::begin(tmp), std::end(tmp), i2_offset);
}

// Applies the permutation represented by |p| to the array |v| whose elements
// are of size |elem_size| using O(1) space.  See the following URL for the
// details: https://blogs.msdn.microsoft.com/oldnewthing/20170102-00/?p=95095.
static inline void apply_permutation(unsigned char* v, int64_t elem_size, std::vector<size_t>& p)
{
    for (size_t i = 0; i < p.size(); ++i)
    {
        auto current = i;
        while (i != p[current])
        {
            auto next = p[current];
            elem_swap(v, elem_size, next, current);
            p[current] = current;
            current = next;
        }
        p[current] = current;
    }
}

#endif // _DISTRIBUTED_H_INCLUDED

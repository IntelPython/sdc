#ifndef _DISTRIBUTED_H_INCLUDED
#define _DISTRIBUTED_H_INCLUDED

#include <Python.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <stdbool.h>
#include <vector>
#include <tuple>
#include <random>

#include "_hpat_common.h"

#define ROOT_PE 0

// XXX same as distributed_api.py:Reduce_Type
struct HPAT_ReduceOps {
    enum ReduceOpsEnum {
        SUM = 0,
        PROD = 1,
        MIN = 2,
        MAX = 3,
        ARGMIN = 4,
        ARGMAX = 5,
        OR = 6
    };
};


static int hpat_dist_get_rank() __UNUSED__;
static int hpat_dist_get_size() __UNUSED__;
static int64_t hpat_dist_get_start(int64_t total, int num_pes, int node_id) __UNUSED__;
static int64_t hpat_dist_get_end(int64_t total, int num_pes, int node_id) __UNUSED__;
static int64_t hpat_dist_get_node_portion(int64_t total, int num_pes, int node_id) __UNUSED__;
static double hpat_dist_get_time() __UNUSED__;
static double hpat_get_time() __UNUSED__;
static int hpat_barrier() __UNUSED__;
static MPI_Datatype get_MPI_typ(int typ_enum) __UNUSED__;
static MPI_Datatype get_val_rank_MPI_typ(int typ_enum) __UNUSED__;
static MPI_Op get_MPI_op(int op_enum) __UNUSED__;
static int get_elem_size(int type_enum) __UNUSED__;
static void hpat_dist_reduce(char *in_ptr, char *out_ptr, int op, int type_enum) __UNUSED__;

static int hpat_dist_exscan_i4(int value) __UNUSED__;
static int64_t hpat_dist_exscan_i8(int64_t value) __UNUSED__;
static float hpat_dist_exscan_f4(float value) __UNUSED__;
static double hpat_dist_exscan_f8(double value) __UNUSED__;

static int hpat_dist_arr_reduce(void* out, int64_t* shapes, int ndims, int op_enum, int type_enum) __UNUSED__;
static MPI_Request hpat_dist_irecv(void* out, int size, int type_enum, int pe, int tag, bool cond) __UNUSED__;
static MPI_Request hpat_dist_isend(void* out, int size, int type_enum, int pe, int tag, bool cond) __UNUSED__;
static void hpat_dist_recv(void* out, int size, int type_enum, int pe, int tag) __UNUSED__;
static void hpat_dist_send(void* out, int size, int type_enum, int pe, int tag) __UNUSED__;
static int hpat_dist_wait(MPI_Request req, bool cond) __UNUSED__;
static void hpat_dist_waitall(int size, MPI_Request *req) __UNUSED__;

static void c_gather_scalar(void* send_data, void* recv_data, int typ_enum) __UNUSED__;
static void c_gatherv(void* send_data, int sendcount, void* recv_data, int* recv_counts, int* displs, int typ_enum) __UNUSED__;
static void c_bcast(void* send_data, int sendcount, int typ_enum) __UNUSED__;

static void c_alltoallv(void* send_data, void* recv_data, int* send_counts,
                int* recv_counts, int* send_disp, int* recv_disp, int typ_enum) __UNUSED__;
static void c_alltoall(void* send_data, void* recv_data, int count, int typ_enum) __UNUSED__;
static int64_t hpat_dist_get_item_pointer(int64_t ind, int64_t start, int64_t count) __UNUSED__;
static void allgather(void* out_data, int size, void* in_data, int type_enum) __UNUSED__;
static MPI_Request *comm_req_alloc(int size) __UNUSED__;
static void comm_req_dealloc(MPI_Request *req_arr) __UNUSED__;
static void req_array_setitem(MPI_Request * req_arr, int64_t ind, MPI_Request req) __UNUSED__;

static void oneD_reshape_shuffle(char* output,
                          char* input,
                          int64_t new_0dim_global_len,
                          int64_t old_0dim_global_len,
                          int64_t out_lower_dims_size,
                          int64_t in_lower_dims_size) __UNUSED__;

static void permutation_int(int64_t* output, int n) __UNUSED__;
static void permutation_array_index(unsigned char *lhs, int64_t len, int64_t elem_size,
                                    unsigned char *rhs, int64_t *p, int64_t p_len) __UNUSED__;
static int hpat_finalize() __UNUSED__;
static void fix_i_malloc() __UNUSED__;
static int hpat_dummy_ptr[64] __UNUSED__;

/* *********************************************************************
************************************************************************/

static void* hpat_get_dummy_ptr() __UNUSED__;
static void* hpat_get_dummy_ptr() {
    return hpat_dummy_ptr;
}

static size_t get_mpi_req_num_bytes() __UNUSED__;
static size_t get_mpi_req_num_bytes() {
    return sizeof(MPI_Request);
}

static int hpat_dist_get_rank()
{
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized)
        MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // printf("my_rank:%d\n", rank);
    return rank;
}

static int hpat_dist_get_size()
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // printf("r size:%d\n", sizeof(MPI_Request));
    // printf("mpi_size:%d\n", size);
    return size;
}

static int64_t hpat_dist_get_start(int64_t total, int num_pes, int node_id)
{
    int64_t div_chunk = (int64_t)ceil(total/((double)num_pes));
    int64_t start = std::min(total, node_id*div_chunk);
    // printf("rank %d start:%lld\n", node_id, start);
    return start;
}

static int64_t hpat_dist_get_end(int64_t total, int num_pes, int node_id)
{
    int64_t div_chunk = (int64_t)ceil(total/((double)num_pes));
    int64_t end = std::min(total, (node_id+1)*div_chunk);
    // printf("rank %d end:%lld\n", node_id, end);
    return end;
}

static int64_t hpat_dist_get_node_portion(int64_t total, int num_pes, int node_id)
{
    return hpat_dist_get_end(total, num_pes, node_id) -
        hpat_dist_get_start(total, num_pes, node_id);
}

static int64_t index_rank(int64_t total, int num_pes, int index)
{
    int64_t div_chunk = (int64_t)ceil(total/((double)num_pes));
    return index / div_chunk;
}

static double hpat_dist_get_time()
{
    double wtime;
    MPI_Barrier(MPI_COMM_WORLD);
    wtime = MPI_Wtime();
    return wtime;
}

static double hpat_get_time()
{
    double wtime;
    wtime = MPI_Wtime();
    return wtime;
}

static int hpat_barrier()
{
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}

static void hpat_dist_reduce(char *in_ptr, char *out_ptr, int op_enum, int type_enum)
{
    // printf("reduce value: %d\n", value);
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    MPI_Op mpi_op = get_MPI_op(op_enum);

    // argmax and argmin need special handling
    if (mpi_op==MPI_MAXLOC || mpi_op==MPI_MINLOC)
    {
        // since MPI's indexed types use 32 bit integers, we workaround by
        // using rank as index, then broadcasting the actual values from the
        // target rank.
        // TODO: generate user-defined reduce operation to avoid this workaround
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // allreduce struct is value + integer
        int value_size;
        MPI_Type_size(mpi_typ, &value_size);
        // copy input index_value to output
        memcpy(out_ptr, in_ptr, value_size+ sizeof(int64_t));
        // printf("rank:%d index:%lld value:%lf value_size:%d\n", rank,
        //     *(int64_t*)in_ptr, *(double*)(in_ptr+sizeof(int64_t)), value_size);

        // format: value + int (input format is int64+value)
        char *in_val_rank = (char*) malloc(value_size+sizeof(int));
        if (in_val_rank == NULL) return;
        char *out_val_rank = (char*) malloc(value_size+sizeof(int));
        if (out_val_rank == NULL) {
            free(in_val_rank);
            return;
        }

        char *in_val_ptr = in_ptr + sizeof(int64_t);
        memcpy(in_val_rank, in_val_ptr, value_size);
        memcpy(in_val_rank+value_size, &rank, sizeof(int));
        // TODO: support int64_int value on Windows
        MPI_Datatype val_rank_mpi_typ = get_val_rank_MPI_typ(type_enum);
        MPI_Allreduce(in_val_rank, out_val_rank, 1, val_rank_mpi_typ, mpi_op, MPI_COMM_WORLD);

        int target_rank = *((int*)(out_val_rank+value_size));
        // printf("rank:%d allreduce rank:%d val:%lf\n", rank, target_rank, *(double*)out_val_rank);
        MPI_Bcast(out_ptr, value_size+sizeof(int64_t), MPI_BYTE, target_rank, MPI_COMM_WORLD);
        free(in_val_rank);
        free(out_val_rank);
        return;
    }

    MPI_Allreduce(in_ptr, out_ptr, 1, mpi_typ, mpi_op, MPI_COMM_WORLD);
    return;
}

static int hpat_dist_arr_reduce(void* out, int64_t* shapes, int ndims, int op_enum, int type_enum)
{
    int i;
    // printf("ndims:%d shape: ", ndims);
    // for(i=0; i<ndims; i++)
    //     printf("%d ", shapes[i]);
    // printf("\n");
    // fflush(stdout);
    int total_size = (int)shapes[0];
    for(i=1; i<ndims; i++)
        total_size *= (int)shapes[i];
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    MPI_Op mpi_op = get_MPI_op(op_enum);
    int elem_size = get_elem_size(type_enum);
    void* res_buf = malloc(total_size*elem_size);
    MPI_Allreduce(out, res_buf, total_size, mpi_typ, mpi_op, MPI_COMM_WORLD);
    memcpy(out, res_buf, total_size*elem_size);
    free(res_buf);
    return 0;
}


static int hpat_dist_exscan_i4(int value)
{
    // printf("sum value: %d\n", value);
    int out=0;
    MPI_Exscan(&value, &out, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

static int64_t hpat_dist_exscan_i8(int64_t value)
{
    // printf("sum value: %lld\n", value);
    int64_t out=0;
    MPI_Exscan(&value, &out, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

static float hpat_dist_exscan_f4(float value)
{
    // printf("sum value: %f\n", value);
    float out=0;
    MPI_Exscan(&value, &out, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

static double hpat_dist_exscan_f8(double value)
{
    // printf("sum value: %lf\n", value);
    double out=0;
    MPI_Exscan(&value, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

static void hpat_dist_recv(void* out, int size, int type_enum, int pe, int tag)
{
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    MPI_Recv(out, size, mpi_typ, pe, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
 }

static void hpat_dist_send(void* out, int size, int type_enum, int pe, int tag)
{
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    MPI_Send(out, size, mpi_typ, pe, tag, MPI_COMM_WORLD);
}


static MPI_Request hpat_dist_irecv(void* out, int size, int type_enum, int pe, int tag, bool cond)
{
    MPI_Request mpi_req_recv(MPI_REQUEST_NULL);
    // printf("irecv size:%d pe:%d tag:%d, cond:%d\n", size, pe, tag, cond);
    // fflush(stdout);
    if(cond)
    {
        MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
        MPI_Irecv(out, size, mpi_typ, pe, tag, MPI_COMM_WORLD, &mpi_req_recv);
    }
    // printf("after irecv size:%d pe:%d tag:%d, cond:%d\n", size, pe, tag, cond);
    // fflush(stdout);
    return mpi_req_recv;
}

static MPI_Request hpat_dist_isend(void* out, int size, int type_enum, int pe, int tag, bool cond)
{
    MPI_Request mpi_req_recv(MPI_REQUEST_NULL);
    // printf("isend size:%d pe:%d tag:%d, cond:%d\n", size, pe, tag, cond);
    // fflush(stdout);
    if(cond)
    {
        MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
        MPI_Isend(out, size, mpi_typ, pe, tag, MPI_COMM_WORLD, &mpi_req_recv);
    }
    // printf("after isend size:%d pe:%d tag:%d, cond:%d\n", size, pe, tag, cond);
    // fflush(stdout);
    return mpi_req_recv;
}

static int hpat_dist_wait(MPI_Request req, bool cond)
{
    if (cond)
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    return 0;
}

static void allgather(void* out_data, int size, void* in_data, int type_enum)
{
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    MPI_Allgather(in_data, size, mpi_typ, out_data, size, mpi_typ, MPI_COMM_WORLD);
    return;
}

static void req_array_setitem(MPI_Request * req_arr, int64_t ind, MPI_Request req)
{
    req_arr[ind] = req;
    return;
}

static void hpat_dist_waitall(int size, MPI_Request *req_arr)
{
    MPI_Waitall(size, req_arr, MPI_STATUSES_IGNORE);
    return;
}

// _numba_to_c_type_map = {
//     int8:0,
//     uint8:1,
//     int32:2,
//     uint32:3,
//     int64:4,
//     float32:5,
//     float64:6
//     uint64: 7
//     }

static MPI_Datatype get_MPI_typ(int typ_enum)
{
    switch (typ_enum) {
        case HPAT_CTypes::INT8:
            return MPI_CHAR;
        case HPAT_CTypes::UINT8:
            return MPI_UNSIGNED_CHAR;
        case HPAT_CTypes::INT32:
            return MPI_INT;
        case HPAT_CTypes::UINT32:
            return MPI_UNSIGNED;
        case HPAT_CTypes::INT64:
            return MPI_LONG_LONG_INT;
        case HPAT_CTypes::UINT64:
            return MPI_UNSIGNED_LONG_LONG;
        case HPAT_CTypes::FLOAT32:
            return MPI_FLOAT;
        case HPAT_CTypes::FLOAT64:
            return MPI_DOUBLE;
        case HPAT_CTypes::INT16:
            // TODO: use MPI_INT16_T?
            return MPI_SHORT;
        case HPAT_CTypes::UINT16:
            return MPI_UNSIGNED_SHORT;
        default:
            std::cerr << "Invalid MPI_Type" << "\n";
    }
    // dummy value in case of error
    // TODO: raise error properly
    return MPI_LONG_LONG_INT;
}

static MPI_Datatype get_val_rank_MPI_typ(int typ_enum)
{
    // printf("h5 type enum:%d\n", typ_enum);
    // XXX: LONG is used for int64, which doesn't work on Windows
    // XXX: LONG is used for uint64
    if (typ_enum < 0 || typ_enum > 7)
    {
        std::cerr << "Invalid MPI_Type" << "\n";
        return MPI_DATATYPE_NULL;
    }
    MPI_Datatype types_list[] = {MPI_DATATYPE_NULL, MPI_DATATYPE_NULL,
            MPI_2INT, MPI_DATATYPE_NULL, MPI_LONG_INT, MPI_FLOAT_INT, MPI_DOUBLE_INT, MPI_LONG_INT};
    return types_list[typ_enum];
}

// from distributed_api Reduce_Type
static MPI_Op get_MPI_op(int op_enum)
{
    // printf("op type enum:%d\n", op_enum);
    if (op_enum < 0 || op_enum > 6)
    {
        std::cerr << "Invalid MPI_Op" << "\n";
        return MPI_SUM;
    }
    MPI_Op ops_list[] = {MPI_SUM, MPI_PROD, MPI_MIN, MPI_MAX, MPI_MINLOC,
            MPI_MAXLOC, MPI_BOR};

    return ops_list[op_enum];
}

static int get_elem_size(int type_enum)
{
    if (type_enum < 0 || type_enum > 7)
    {
        std::cerr << "Invalid MPI_Type" << "\n";
        return 8;
    }
    int types_sizes[] = {1,1,4,4,8,4,8,8};
    return types_sizes[type_enum];
}

static int64_t hpat_dist_get_item_pointer(int64_t ind, int64_t start, int64_t count)
{
    // printf("ind:%lld start:%lld count:%lld\n", ind, start, count);
    if (ind >= start && ind < start+count)
        return ind-start;
    return -1;
}

static void c_gather_scalar(void* send_data, void* recv_data, int typ_enum)
{
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    MPI_Gather(send_data, 1, mpi_typ, recv_data, 1, mpi_typ, ROOT_PE,
           MPI_COMM_WORLD);
    return;
}

static void c_gatherv(void* send_data, int sendcount, void* recv_data, int* recv_counts, int* displs, int typ_enum)
{
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    MPI_Gatherv(send_data, sendcount, mpi_typ, recv_data, recv_counts, displs, mpi_typ, ROOT_PE,
           MPI_COMM_WORLD);
    return;
}

static void c_bcast(void* send_data, int sendcount, int typ_enum)
{
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    MPI_Bcast(send_data, sendcount, mpi_typ, ROOT_PE, MPI_COMM_WORLD);
    return;
}

static void c_alltoallv(void* send_data, void* recv_data, int* send_counts,
                int* recv_counts, int* send_disp, int* recv_disp, int typ_enum)
{
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    MPI_Alltoallv(send_data, send_counts, send_disp, mpi_typ,
        recv_data, recv_counts, recv_disp, mpi_typ, MPI_COMM_WORLD);
}

static void c_alltoall(void* send_data, void* recv_data, int count, int typ_enum)
{
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    MPI_Alltoall(send_data, count, mpi_typ, recv_data, count, mpi_typ, MPI_COMM_WORLD);
}

MPI_Request *comm_req_alloc(int size)
{
    // printf("req alloc %d\n", size);
    return new MPI_Request[size];
}

static void comm_req_dealloc(MPI_Request *req_arr)
{
    delete[] req_arr;
}

static int hpat_finalize()
{
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) {
        return 0;
    }
    int is_finalized;
    MPI_Finalized(&is_finalized);
    if (!is_finalized) {
        // printf("finalizing\n");
        MPI_Finalize();
    }
    return 0;
}

static void permutation_int(int64_t* output, int n)
{
     MPI_Bcast(output, n, MPI_INT64_T, 0, MPI_COMM_WORLD);
}

// Given the permutation index |p| and |rank|, and the number of ranks
// |num_ranks|, finds the destination ranks of indices of the |rank|.  For
// example, if |rank| is 1, |num_ranks| is 3, |p_len| is 12, and |p| is the
// following array [ 9, 8, 6, 4, 11, 7, 2, 3, 5, 0, 1, 10], the function returns
// [0, 2, 0, 1].
static std::vector<int64_t> find_dest_ranks(int64_t rank, int64_t num_ranks,
                                     int64_t *p, int64_t p_len)
{
    auto chunk_size = hpat_dist_get_node_portion(p_len, num_ranks, rank);
    auto begin = hpat_dist_get_start(p_len, num_ranks, rank);
    std::vector<int64_t> dest_ranks(chunk_size);

    for (auto i = 0; i < p_len; ++i)
        if (rank == index_rank(p_len, num_ranks, p[i]))
            dest_ranks[p[i] - begin] = index_rank(p_len, num_ranks, i);
    return dest_ranks;
}

static std::vector<int> find_send_counts(const std::vector<int64_t>& dest_ranks,
                                  int64_t num_ranks, int64_t elem_size)
{
    std::vector<int> send_counts(num_ranks);
    for (auto dest : dest_ranks)
        ++send_counts[dest];
    return send_counts;
}

static std::vector<int> find_disps(const std::vector<int>& counts) {
    std::vector<int> disps(counts.size());
    for (size_t i = 1; i < disps.size(); ++i)
        disps[i] = disps[i-1] + counts[i-1];
    return disps;
}

static std::vector<int> find_recv_counts(int64_t rank, int64_t num_ranks,
                                  int64_t *p, int64_t p_len,
                                  int64_t elem_size)
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
template<class T>
static std::vector<size_t> arg_sort(T *v, int64_t len) {
    std::vector<size_t> index_array(len);
    std::iota(index_array.begin(), index_array.end(), 0);
    std::sort(index_array.begin(), index_array.end(),
              [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    return index_array;
}

// |v| is an array of elements of size |elem_size|.  This function swaps
// elements located at indices |i1| and |i2|.
static void elem_swap(unsigned char *v, int64_t elem_size, size_t i1, size_t i2)
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
static void apply_permutation(unsigned char *v, int64_t elem_size,
                       std::vector<size_t>& p)
{
    for (size_t i = 0; i < p.size(); ++i) {
        auto current = i;
        while (i != p[current]) {
            auto next = p[current];
            elem_swap(v, elem_size, next, current);
            p[current] = current;
            current = next;
        }
        p[current] = current;
    }
}

// Applies the permutation represented by |p| of size |p_len| to the array |rhs|
// of elements of size |elem_size| and stores the result in |lhs|.
static void permutation_array_index(unsigned char *lhs, int64_t len, int64_t elem_size,
                             unsigned char *rhs, int64_t *p, int64_t p_len)
{
    if (len != p_len) {
        std::cerr << "Array length and permutation index length should match!\n";
        return;
    }

    MPI_Datatype element_t;
    MPI_Type_contiguous(elem_size, MPI_UNSIGNED_CHAR, &element_t);
    MPI_Type_commit(&element_t);

    auto num_ranks = hpat_dist_get_size();
    auto rank = hpat_dist_get_rank();
    auto dest_ranks = find_dest_ranks(rank, num_ranks, p, p_len);
    auto send_counts = find_send_counts(dest_ranks, num_ranks, elem_size);
    auto send_disps = find_disps(send_counts);
    auto recv_counts = find_recv_counts(rank, num_ranks, p, p_len, elem_size);
    auto recv_disps = find_disps(recv_counts);

    auto offsets = send_disps;
    std::vector<unsigned char> send_buf(dest_ranks.size() * elem_size);
    for (size_t i = 0; i < dest_ranks.size(); ++i) {
        auto send_buf_offset = offsets[dest_ranks[i]]++ * elem_size;
        auto *send_buf_begin = send_buf.data() + send_buf_offset;
        auto *rhs_begin = rhs + i * elem_size;
        std::copy(rhs_begin, rhs_begin + elem_size, send_buf_begin);
    }

    MPI_Alltoallv(send_buf.data(), send_counts.data(), send_disps.data(),
                  element_t, lhs, recv_counts.data(), recv_disps.data(),
                  element_t, MPI_COMM_WORLD);

    // Let us assume that the global data array is [a b c d e f g h] and the
    // permutation array that we would like to apply to it is [2 7 5 6 4 3 1 0].
    // Hence, the resultant permutation is [c h f g e d b a].  Assuming that
    // there are two ranks, each receiving 4 data items, and we are rank 0,
    // after MPI_Alltoallv returns, we receive the chunk [c f g h] that
    // corresponds to the sorted chunk of our permutation, which is [2 5 6 7].
    // In order to recover the positions of [c f g h] in the target permutation
    // we first argsort our chunk of permutation array:
    auto begin = p + hpat_dist_get_start(p_len, num_ranks, rank);
    auto p1 = arg_sort(begin, dest_ranks.size());

    // The result of the argsort, stored in p1, is [0 2 3 1].  This tells us how
    // the chunk we have received is different from the target permutation we
    // want to achieve.  Hence, to achieve the target permutation, we need to
    // sort our data chunk based on p1.  One way of sorting array A based on the
    // values of array B, is to argsort array B and apply the permutation to
    // array A.  Therefore, we argsort p1:
    auto p2 = arg_sort(p1.data(), dest_ranks.size());

    // which gives us [0 3 1 2], and apply the resultant permutation to our data
    // chunk to obtain the target permutation.
    apply_permutation(lhs, elem_size, p2);

    MPI_Type_free(&element_t);
}

static void oneD_reshape_shuffle(char* output,
                          char* input,
                          int64_t new_0dim_global_len,
                          int64_t old_0dim_global_len,
                          int64_t out_lower_dims_size,
                          int64_t in_lower_dims_size)
{
    int num_pes = hpat_dist_get_size();
    int rank = hpat_dist_get_rank();

    // get my old and new data interval and convert to byte offsets
    int64_t my_old_start = in_lower_dims_size * hpat_dist_get_start(old_0dim_global_len, num_pes, rank);
    int64_t my_new_start = out_lower_dims_size * hpat_dist_get_start(new_0dim_global_len, num_pes, rank);
    int64_t my_old_end = in_lower_dims_size * hpat_dist_get_end(old_0dim_global_len, num_pes, rank);
    int64_t my_new_end = out_lower_dims_size * hpat_dist_get_end(new_0dim_global_len, num_pes, rank);

    int64_t *send_counts = new int64_t[num_pes];
    int64_t *recv_counts = new int64_t[num_pes];
    int64_t *send_disp = new int64_t[num_pes];
    int64_t *recv_disp = new int64_t[num_pes];

    int64_t curr_send_offset = 0;
    int64_t curr_recv_offset = 0;

    for(int i=0; i<num_pes; i++)
    {
        send_disp[i] = curr_send_offset;
        recv_disp[i] = curr_recv_offset;

        // get pe's old and new data interval and convert to byte offsets
        int64_t pe_old_start = in_lower_dims_size * hpat_dist_get_start(old_0dim_global_len, num_pes, i);
        int64_t pe_new_start = out_lower_dims_size * hpat_dist_get_start(new_0dim_global_len, num_pes, i);
        int64_t pe_old_end = in_lower_dims_size * hpat_dist_get_end(old_0dim_global_len, num_pes, i);
        int64_t pe_new_end = out_lower_dims_size * hpat_dist_get_end(new_0dim_global_len, num_pes, i);

        send_counts[i] = 0;
        recv_counts[i] = 0;

        // if sending to processor (interval overlap)
        if (pe_new_end > my_old_start && pe_new_start < my_old_end)
        {
            send_counts[i] = std::min(my_old_end, pe_new_end) - std::max(my_old_start, pe_new_start);
            curr_send_offset += send_counts[i];
        }

        // if receiving from processor (interval overlap)
        if (my_new_end > pe_old_start && my_new_start < pe_old_end)
        {
            recv_counts[i] = std::min(pe_old_end, my_new_end) - std::max(pe_old_start, my_new_start);
            curr_recv_offset += recv_counts[i];
        }
    }
    // printf("rank:%d send %lld %lld recv %lld %lld\n", rank, send_counts[0], send_counts[1], recv_counts[0], recv_counts[1]);
    // printf("send %d recv %d send_disp %d recv_disp %d\n", send_counts[0], recv_counts[0], send_disp[0], recv_disp[0]);
    // printf("data %lld %lld\n", ((int64_t*)input)[0], ((int64_t*)input)[1]);

    // workaround MPI int limit if necessary
    int *i_send_counts = new int[num_pes];
    int *i_recv_counts = new int[num_pes];
    int *i_send_disp = new int[num_pes];
    int *i_recv_disp = new int[num_pes];
    bool big_shuffle = false;

    for(int i=0; i<num_pes; i++)
    {
        // any value doesn't fit in int
        if (send_counts[i]>=(int64_t)INT_MAX
            || recv_counts[i]>=(int64_t)INT_MAX
            || send_disp[i]>=(int64_t)INT_MAX
            || recv_disp[i]>=(int64_t)INT_MAX)
        {
            big_shuffle = true;
            break;
        }
        i_send_counts[i] = (int) send_counts[i];
        i_recv_counts[i] = (int) recv_counts[i];
        i_send_disp[i] = (int) send_disp[i];
        i_recv_disp[i] = (int) recv_disp[i];
    }

    if (!big_shuffle)
    {
        int ierr = MPI_Alltoallv(input, i_send_counts, i_send_disp, MPI_CHAR,
            output, i_recv_counts, i_recv_disp, MPI_CHAR, MPI_COMM_WORLD);
        if (ierr!=0) std::cerr << "small shuffle error: " << '\n';
    }
    else
    {
        // char err_string[MPI_MAX_ERROR_STRING];
        // err_string[MPI_MAX_ERROR_STRING-1] = '\0';
        // int err_len, err_class;
        // MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

        int *l_send_counts = new int[num_pes];
        int *l_recv_counts = new int[num_pes];
        int *l_send_disp = new int[num_pes];
        int *l_recv_disp = new int[num_pes];

        int64_t *send_offset = new int64_t[num_pes];
        int64_t *recv_offset = new int64_t[num_pes];

        #define LARGE_DTYPE_SIZE 1024
        MPI_Datatype large_dtype;
        MPI_Type_contiguous(LARGE_DTYPE_SIZE, MPI_CHAR, &large_dtype);
        MPI_Type_commit(&large_dtype);

        for(int i=0; i<num_pes; i++)
        {
            // large values
            i_send_counts[i] = (int) (send_counts[i]/LARGE_DTYPE_SIZE);
            i_recv_counts[i] = (int) (recv_counts[i]/LARGE_DTYPE_SIZE);
            i_send_disp[i] = (int) (send_disp[i]/LARGE_DTYPE_SIZE);
            i_recv_disp[i] = (int) (recv_disp[i]/LARGE_DTYPE_SIZE);
            // leftover values
            l_send_counts[i] = (int) (send_counts[i] % LARGE_DTYPE_SIZE);
            l_recv_counts[i] = (int) (recv_counts[i] % LARGE_DTYPE_SIZE);
            l_send_disp[i] = (int) (send_disp[i] % LARGE_DTYPE_SIZE);
            l_recv_disp[i] = (int) (recv_disp[i] % LARGE_DTYPE_SIZE);
            // printf("pe %d rank %d send %d recv %d sdisp %d rdisp %d lsend %d lrecv %d lsdisp %d lrdisp %d\n", i, rank,
            //         i_send_counts[i], i_recv_counts[i], i_send_disp[i], i_recv_disp[i],
            //         l_send_counts[i], l_recv_counts[i], l_send_disp[i], l_recv_disp[i]);
        }

        int64_t curr_send_buff_offset = 0;
        int64_t curr_recv_buff_offset = 0;
        // compute buffer offsets
        for(int i=0; i<num_pes; i++)
        {
            send_offset[i] = curr_send_buff_offset;
            recv_offset[i] = curr_recv_buff_offset;
            curr_send_buff_offset += send_counts[i];
            curr_recv_buff_offset += recv_counts[i];
            // printf("pe %d rank %d send offset %lld recv offset %lld\n", i, rank, send_offset[i], recv_offset[i]);
        }

        // XXX implement alltoallv manually
        for(int i=0; i<num_pes; i++)
        {
            int TAG = 11; // arbitrary
            int dest = (rank+i+num_pes) % num_pes;
            int src = (rank-i+num_pes) % num_pes;
            // printf("rank %d src %d dest %d\n", rank, src, dest);
            // send big type
            int ierr = MPI_Sendrecv(input+send_offset[dest], i_send_counts[dest], large_dtype, dest, TAG,
                        output+recv_offset[src], i_recv_counts[src], large_dtype, src, TAG,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (ierr!=0) std::cerr << "large sendrecv error" << '\n';
            // send leftover
            ierr = MPI_Sendrecv(input+send_offset[dest]+((int64_t)i_send_counts[dest])*LARGE_DTYPE_SIZE, l_send_counts[dest], MPI_CHAR, dest, TAG+1,
                        output+recv_offset[src]+((int64_t)i_recv_counts[dest])*LARGE_DTYPE_SIZE, l_recv_counts[src], MPI_CHAR, src, TAG+1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (ierr!=0) std::cerr << "small sendrecv error" << '\n';
        }

        // cleanup
        MPI_Type_free(&large_dtype);
        delete[] l_send_counts;
        delete[] l_recv_counts;
        delete[] l_send_disp;
        delete[] l_recv_disp;
        delete[] send_offset;
        delete[] recv_offset;
    }

    // cleanup
    delete[] i_send_counts;
    delete[] i_recv_counts;
    delete[] i_send_disp;
    delete[] i_recv_disp;
    delete[] send_counts;
    delete[] recv_counts;
    delete[] send_disp;
    delete[] recv_disp;
}

// fix for tensorflows MKL support that overwrites Intel mallocs,
// which causes Intel MPI to crash.
#ifdef I_MPI_VERSION
#include "i_malloc.h"
static void fix_i_malloc()
{
    i_malloc = malloc;
    i_calloc = calloc;
    i_realloc = realloc;
    i_free = free;
}
#else
static void fix_i_malloc() {}
#endif

#endif // _DISTRIBUTED_H_INCLUDED

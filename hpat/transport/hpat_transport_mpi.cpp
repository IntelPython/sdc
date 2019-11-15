//*****************************************************************************
// Copyright (c) 2019, Intel Corporation All rights reserved.
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

#include <Python.h>
#include <boost/filesystem.hpp>
#include <mpi.h>

#include "../_distributed.h"

using namespace std;

#define ROOT 0
#define LARGE_DTYPE_SIZE 1024

/**
 * Static function to be registered in Python and code helpers
 */

static MPI_Datatype get_MPI_typ(int typ_enum)
{
    switch (typ_enum)
    {
    case SDC_CTypes::INT8:
        return MPI_CHAR;
    case SDC_CTypes::UINT8:
        return MPI_UNSIGNED_CHAR;
    case SDC_CTypes::INT32:
        return MPI_INT;
    case SDC_CTypes::UINT32:
        return MPI_UNSIGNED;
    case SDC_CTypes::INT64:
        return MPI_LONG_LONG_INT;
    case SDC_CTypes::UINT64:
        return MPI_UNSIGNED_LONG_LONG;
    case SDC_CTypes::FLOAT32:
        return MPI_FLOAT;
    case SDC_CTypes::FLOAT64:
        return MPI_DOUBLE;
    case SDC_CTypes::INT16:
        // TODO: use MPI_INT16_T?
        return MPI_SHORT;
    case SDC_CTypes::UINT16:
        return MPI_UNSIGNED_SHORT;
    default:
        cerr << "Invalid MPI_Type\n";
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
        cerr << "Invalid MPI_Type\n";
        return MPI_DATATYPE_NULL;
    }
    MPI_Datatype types_list[] = {MPI_DATATYPE_NULL,
                                 MPI_DATATYPE_NULL,
                                 MPI_2INT,
                                 MPI_DATATYPE_NULL,
                                 MPI_LONG_INT,
                                 MPI_FLOAT_INT,
                                 MPI_DOUBLE_INT,
                                 MPI_LONG_INT};
    return types_list[typ_enum];
}

static double hpat_dist_get_time()
{
    double wtime;
    MPI_Barrier(MPI_COMM_WORLD);
    wtime = MPI_Wtime();
    return wtime;
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

/// return vector of offsets of newlines in first n bytes of given stream
static vector<size_t> count_lines(istream* f, size_t n)
{
    vector<size_t> pos;
    char c;
    size_t i = 0;

    while (i < n && f->get(c))
    {
        if (c == '\n')
            pos.push_back(i);
        ++i;
    }

    if (i < n)
        cerr << "Warning, read only " << i << " bytes out of " << n << "requested\n";

    return pos;
}

/**
 * Code moved from hpat/io/_csv.cpp
 */

static int hpat_dist_wait(MPI_Request req, bool cond)
{
    if (cond)
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    return 0;
}

static void hpat_dist_waitall(int size, MPI_Request* req_arr)
{
    MPI_Waitall(size, req_arr, MPI_STATUSES_IGNORE);
    return;
}

static int hpat_dist_get_size()
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // printf("r size:%d\n", sizeof(MPI_Request));
    // printf("mpi_size:%d\n", size);
    return size;
}

static int64_t hpat_dist_exscan_i8(int64_t value)
{
    // printf("sum value: %lld\n", value);
    int64_t out = 0;
    MPI_Exscan(&value, &out, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

static MPI_Request hpat_dist_irecv(void* out, int size, int type_enum, int pe, int tag, bool cond)
{
    MPI_Request mpi_req_recv(MPI_REQUEST_NULL);
    // printf("irecv size:%d pe:%d tag:%d, cond:%d\n", size, pe, tag, cond);
    // fflush(stdout);
    if (cond)
    {
        MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
        MPI_Irecv(out, size, mpi_typ, pe, tag, MPI_COMM_WORLD, &mpi_req_recv);
    }
    // printf("after irecv size:%d pe:%d tag:%d, cond:%d\n", size, pe, tag, cond);
    // fflush(stdout);
    return mpi_req_recv;
}

// from distributed_api Reduce_Type
static MPI_Op get_MPI_op(int op_enum)
{
    // printf("op type enum:%d\n", op_enum);
    if (op_enum < 0 || op_enum > 6)
    {
        cerr << "Invalid MPI_Op\n";
        return MPI_SUM;
    }
    MPI_Op ops_list[] = {MPI_SUM, MPI_PROD, MPI_MIN, MPI_MAX, MPI_MINLOC, MPI_MAXLOC, MPI_BOR};

    return ops_list[op_enum];
}

static void hpat_dist_reduce(char* in_ptr, char* out_ptr, int op_enum, int type_enum)
{
    // printf("reduce value: %d\n", value);
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    MPI_Op mpi_op = get_MPI_op(op_enum);

    // argmax and argmin need special handling
    if (mpi_op == MPI_MAXLOC || mpi_op == MPI_MINLOC)
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
        memcpy(out_ptr, in_ptr, value_size + sizeof(int64_t));
        // printf("rank:%d index:%lld value:%lf value_size:%d\n", rank,
        //     *(int64_t*)in_ptr, *(double*)(in_ptr+sizeof(int64_t)), value_size);

        // format: value + int (input format is int64+value)
        char* in_val_rank = (char*)malloc(value_size + sizeof(int));
        if (in_val_rank == NULL)
            return;
        char* out_val_rank = (char*)malloc(value_size + sizeof(int));
        if (out_val_rank == NULL)
        {
            free(in_val_rank);
            return;
        }

        char* in_val_ptr = in_ptr + sizeof(int64_t);
        memcpy(in_val_rank, in_val_ptr, value_size);
        memcpy(in_val_rank + value_size, &rank, sizeof(int));
        // TODO: support int64_int value on Windows
        MPI_Datatype val_rank_mpi_typ = get_val_rank_MPI_typ(type_enum);
        MPI_Allreduce(in_val_rank, out_val_rank, 1, val_rank_mpi_typ, mpi_op, MPI_COMM_WORLD);

        int target_rank = *((int*)(out_val_rank + value_size));
        // printf("rank:%d allreduce rank:%d val:%lf\n", rank, target_rank, *(double*)out_val_rank);
        MPI_Bcast(out_ptr, value_size + sizeof(int64_t), MPI_BYTE, target_rank, MPI_COMM_WORLD);
        free(in_val_rank);
        free(out_val_rank);
        return;
    }

    MPI_Allreduce(in_ptr, out_ptr, 1, mpi_typ, mpi_op, MPI_COMM_WORLD);
    return;
}

static MPI_Request hpat_dist_isend(void* out, int size, int type_enum, int pe, int tag, bool cond)
{
    MPI_Request mpi_req_recv(MPI_REQUEST_NULL);
    // printf("isend size:%d pe:%d tag:%d, cond:%d\n", size, pe, tag, cond);
    // fflush(stdout);
    if (cond)
    {
        MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
        MPI_Isend(out, size, mpi_typ, pe, tag, MPI_COMM_WORLD, &mpi_req_recv);
    }
    // printf("after isend size:%d pe:%d tag:%d, cond:%d\n", size, pe, tag, cond);
    // fflush(stdout);
    return mpi_req_recv;
}

static void hpat_mpi_csv_get_offsets(
    istream* f, size_t fsz, bool is_parallel, int64_t skiprows, int64_t nrows, size_t& my_off_start, size_t& my_off_end)
{
    size_t nranks = hpat_dist_get_size();

    if (is_parallel && nranks > 1)
    {
        size_t rank = hpat_dist_get_rank();

        // seek to our chunk
        size_t byte_offset = hpat_dist_get_start(fsz, nranks, rank);
        f->seekg(byte_offset, ios_base::beg);
        if (!f->good() || f->eof())
        {
            cerr << "Could not seek to start position " << hpat_dist_get_start(fsz, nranks, rank) << endl;
            return;
        }
        // We evenly distribute the 'data' byte-wise
        // count number of lines in chunk
        // TODO: count only until nrows
        vector<size_t> line_offset = count_lines(f, hpat_dist_get_node_portion(fsz, nranks, rank));
        size_t no_lines = line_offset.size();
        // get total number of lines using allreduce
        int64_t tot_no_lines = 0;

        hpat_dist_reduce(reinterpret_cast<char*>(&no_lines),
                         reinterpret_cast<char*>(&tot_no_lines),
                         SDC_ReduceOps::SUM,
                         SDC_CTypes::UINT64);

        // Now we need to communicate the distribution as we really want it
        // First determine which is our first line (which is the sum of previous lines)
        int64_t byte_first_line = hpat_dist_exscan_i8(no_lines);
        int64_t byte_last_line = byte_first_line + no_lines;

        // We now determine the chunks of lines that begin and end in our byte-chunk

        // issue IRecv calls, eventually receiving start and end offsets of our line-chunk
        const int START_OFFSET = 47011;
        const int END_OFFSET = 47012;
        vector<MPI_Request> mpi_reqs;

        mpi_reqs.push_back(hpat_dist_irecv(
            &my_off_start, 1, SDC_CTypes::UINT64, MPI_ANY_SOURCE, START_OFFSET, (rank > 0 || skiprows > 0)));
        mpi_reqs.push_back(hpat_dist_irecv(
            &my_off_end, 1, SDC_CTypes::UINT64, MPI_ANY_SOURCE, END_OFFSET, ((rank < (nranks - 1)) || nrows != -1)));

        // check nrows argument
        if (nrows != -1 && (nrows < 0 || nrows > tot_no_lines))
        {
            cerr << "Invalid nrows argument: " << nrows << " for total number of lines: " << tot_no_lines << endl;
            return;
        }

        // number of lines that actually needs to be parsed
        size_t n_lines_to_read = nrows != -1 ? nrows : tot_no_lines - skiprows;

        // TODO skiprows and nrows need testing
        // send start offset of rank 0
        if (skiprows > byte_first_line && skiprows <= byte_last_line)
        {
            size_t i_off = byte_offset + line_offset[skiprows - byte_first_line - 1] +
                           1; // +1 to skip/include leading/trailing newline
            mpi_reqs.push_back(hpat_dist_isend(&i_off, 1, SDC_CTypes::UINT64, 0, START_OFFSET, true));
        }

        // send end offset of rank n-1
        if (nrows > byte_first_line && nrows <= byte_last_line)
        {
            size_t i_off = byte_offset + line_offset[nrows - byte_first_line - 1] +
                           1; // +1 to skip/include leading/trailing newline
            mpi_reqs.push_back(hpat_dist_isend(&i_off, 1, SDC_CTypes::UINT64, nranks - 1, END_OFFSET, true));
        }

        // We iterate through chunk boundaries (defined by line-numbers)
        // we start with boundary 1 as 0 is the beginning of file
        for (size_t i = 1; i < nranks; ++i)
        {
            int64_t i_bndry = skiprows + hpat_dist_get_start(n_lines_to_read, (int)nranks, i);
            // Note our line_offsets mark the end of each line!
            // we check if boundary is on our byte-chunk
            if (i_bndry > byte_first_line && i_bndry <= byte_last_line)
            {
                // if so, send stream-offset to ranks which start/end here
                size_t i_off = byte_offset + line_offset[i_bndry - byte_first_line - 1] +
                               1; // +1 to skip/include leading/trailing newline
                // send to rank that starts at this boundary: i
                mpi_reqs.push_back(hpat_dist_isend(&i_off, 1, SDC_CTypes::UINT64, i, START_OFFSET, true));
                // send to rank that ends at this boundary: i-1
                mpi_reqs.push_back(hpat_dist_isend(&i_off, 1, SDC_CTypes::UINT64, i - 1, END_OFFSET, true));
            }
            else
            {
                // if not and we past our chunk -> we stop
                if (i_bndry > byte_last_line)
                    break;
            } // else we are before our chunk -> continue iteration
        }
        // before reading, make sure we received our start/end offsets
        hpat_dist_waitall(mpi_reqs.size(), mpi_reqs.data());
    } // if is_parallel
    else if (skiprows > 0 || nrows != -1)
    {
        vector<size_t> line_offset = count_lines(f, fsz);
        if (skiprows > 0)
            my_off_start = line_offset[skiprows - 1] + 1;
        if (nrows != -1)
            my_off_end = line_offset[nrows - 1] + 1;
    }
}

/**
 * Code moved from hpat/io/_io.cpp
 */

static uint64_t get_file_size(const char* file_name)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    uint64_t f_size = 0;

    if (rank == ROOT)
    {
        boost::filesystem::path f_path(file_name);
        // TODO: throw FileNotFoundError
        if (!boost::filesystem::exists(f_path))
        {
            cerr << "No such file or directory: " << file_name << '\n';
            return 0;
        }
        f_size = (uint64_t)boost::filesystem::file_size(f_path);
    }

    MPI_Bcast(&f_size, 1, MPI_UNSIGNED_LONG_LONG, ROOT, MPI_COMM_WORLD);

    return f_size;
}

static void file_read_parallel(char* file_name, char* buff, int64_t start, int64_t count)
{
    // printf("MPI READ %lld %lld\n", start, count);
    char err_string[MPI_MAX_ERROR_STRING];
    err_string[MPI_MAX_ERROR_STRING - 1] = '\0';
    int err_len, err_class;
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    MPI_File fh;
    int ierr = MPI_File_open(MPI_COMM_WORLD, (const char*)file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (ierr != 0)
        cerr << "File open error: " << file_name << '\n';

    // work around MPI count limit by using a large dtype
    if (count >= (int64_t)INT_MAX)
    {
        MPI_Datatype large_dtype;
        MPI_Type_contiguous(LARGE_DTYPE_SIZE, MPI_CHAR, &large_dtype);
        MPI_Type_commit(&large_dtype);
        int read_size = (int)(count / LARGE_DTYPE_SIZE);

        ierr = MPI_File_read_at_all(fh, (MPI_Offset)start, buff, read_size, large_dtype, MPI_STATUS_IGNORE);
        if (ierr != 0)
        {
            MPI_Error_class(ierr, &err_class);
            cerr << "File large read error: " << err_class << " " << file_name << '\n';
            MPI_Error_string(ierr, err_string, &err_len);
            printf("Error %s\n", err_string);
            fflush(stdout);
        }
        MPI_Type_free(&large_dtype);
        int64_t left_over = count % LARGE_DTYPE_SIZE;
        int64_t read_byte_size = count - left_over;
        // printf("VAL leftover %lld read %lld\n", left_over, read_byte_size);
        start += read_byte_size;
        buff += read_byte_size;
        count = left_over;
    }
    // printf("MPI leftover READ %lld %lld\n", start, count);

    ierr = MPI_File_read_at_all(fh, (MPI_Offset)start, buff, (int)count, MPI_CHAR, MPI_STATUS_IGNORE);

    // if (ierr!=0) cerr << "File read error: " << file_name << '\n';
    if (ierr != 0)
    {
        MPI_Error_class(ierr, &err_class);
        cerr << "File read error: " << err_class << " " << file_name << '\n';
        MPI_Error_string(ierr, err_string, &err_len);
        printf("Error %s\n", err_string);
        fflush(stdout);
    }

    MPI_File_close(&fh);

    return;
}

static void file_write_parallel(char* file_name, char* buff, int64_t start, int64_t count, int64_t elem_size)
{
    // cout << file_name;
    // printf(" MPI WRITE %lld %lld %lld\n", start, count, elem_size);

    // TODO: handle large write count
    if (count >= (int64_t)INT_MAX)
    {
        cerr << "write count too large " << file_name << '\n';
        return;
    }

    char err_string[MPI_MAX_ERROR_STRING];
    err_string[MPI_MAX_ERROR_STRING - 1] = '\0';
    int err_len, err_class;
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    MPI_File fh;
    int ierr =
        MPI_File_open(MPI_COMM_WORLD, (const char*)file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if (ierr != 0)
        cerr << "File open error (write): " << file_name << '\n';

    MPI_Datatype elem_dtype;
    MPI_Type_contiguous(elem_size, MPI_CHAR, &elem_dtype);
    MPI_Type_commit(&elem_dtype);

    ierr = MPI_File_write_at_all(fh, (MPI_Offset)(start * elem_size), buff, (int)count, elem_dtype, MPI_STATUS_IGNORE);

    MPI_Type_free(&elem_dtype);
    // if (ierr!=0) cerr << "File write error: " << file_name << '\n';
    if (ierr != 0)
    {
        MPI_Error_class(ierr, &err_class);
        cerr << "File write error: " << err_class << " " << file_name << '\n';
        MPI_Error_string(ierr, err_string, &err_len);
        printf("Error %s\n", err_string);
        fflush(stdout);
    }

    MPI_File_close(&fh);

    return;
}

/**
 * Code moved from hpat/_hiframes.cpp
 */

static int64_t get_join_sendrecv_counts(int** p_send_counts,
                                        int** p_recv_counts,
                                        int** p_send_disp,
                                        int** p_recv_disp,
                                        int64_t arr_len,
                                        int type_enum,
                                        void* data)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    // alloc buffers
    int* send_counts = new int[n_pes];
    *p_send_counts = send_counts;
    int* recv_counts = new int[n_pes];
    *p_recv_counts = recv_counts;
    int* send_disp = new int[n_pes];
    *p_send_disp = send_disp;
    int* recv_disp = new int[n_pes];
    *p_recv_disp = recv_disp;

    // TODO: extend to other key types
    int64_t* key_arr = (int64_t*)data;
    memset(send_counts, 0, sizeof(int) * n_pes);
    for (int64_t i = 0; i < arr_len; i++)
    {
        int node_id = key_arr[i] % n_pes;
        send_counts[node_id]++;
    }
    // send displacement
    send_disp[0] = 0;
    for (int64_t i = 1; i < n_pes; i++)
    {
        send_disp[i] = send_disp[i - 1] + send_counts[i - 1];
    }
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
    // recv displacement
    recv_disp[0] = 0;
    for (int64_t i = 1; i < n_pes; i++)
    {
        recv_disp[i] = recv_disp[i - 1] + recv_counts[i - 1];
    }
    // total recv size
    int64_t total_recv_size = 0;
    for (int64_t i = 0; i < n_pes; i++)
    {
        total_recv_size += recv_counts[i];
    }
    return total_recv_size;
}

/**
 * Code moved from hpat/_distibuted.cpp
 */

static MPI_Request* comm_req_alloc(int size)
{
    // printf("req alloc %d\n", size);
    return new MPI_Request[size];
}

static void comm_req_dealloc(MPI_Request* req_arr)
{
    delete[] req_arr;
}

static void req_array_setitem(MPI_Request* req_arr, int64_t ind, MPI_Request req)
{
    req_arr[ind] = req;
    return;
}

static size_t get_mpi_req_num_bytes()
{
    return sizeof(MPI_Request);
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

static int hpat_dist_arr_reduce(void* out, int64_t* shapes, int ndims, int op_enum, int type_enum)
{
    int i;
    // printf("ndims:%d shape: ", ndims);
    // for(i=0; i<ndims; i++)
    //     printf("%d ", shapes[i]);
    // printf("\n");
    // fflush(stdout);
    int total_size = (int)shapes[0];
    for (i = 1; i < ndims; i++)
        total_size *= (int)shapes[i];
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    MPI_Op mpi_op = get_MPI_op(op_enum);
    int elem_size = get_elem_size(type_enum);
    void* res_buf = malloc(total_size * elem_size);
    MPI_Allreduce(out, res_buf, total_size, mpi_typ, mpi_op, MPI_COMM_WORLD);
    memcpy(out, res_buf, total_size * elem_size);
    free(res_buf);
    return 0;
}

static int hpat_dist_exscan_i4(int value)
{
    // printf("sum value: %d\n", value);
    int out = 0;
    MPI_Exscan(&value, &out, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

static float hpat_dist_exscan_f4(float value)
{
    // printf("sum value: %f\n", value);
    float out = 0;
    MPI_Exscan(&value, &out, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

static double hpat_dist_exscan_f8(double value)
{
    // printf("sum value: %lf\n", value);
    double out = 0;
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

static void allgather(void* out_data, int size, void* in_data, int type_enum)
{
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    MPI_Allgather(in_data, size, mpi_typ, out_data, size, mpi_typ, MPI_COMM_WORLD);
    return;
}

static void c_gather_scalar(void* send_data, void* recv_data, int typ_enum)
{
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    MPI_Gather(send_data, 1, mpi_typ, recv_data, 1, mpi_typ, ROOT_PE, MPI_COMM_WORLD);
    return;
}

static void c_gatherv(void* send_data, int sendcount, void* recv_data, int* recv_counts, int* displs, int typ_enum)
{
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    MPI_Gatherv(send_data, sendcount, mpi_typ, recv_data, recv_counts, displs, mpi_typ, ROOT_PE, MPI_COMM_WORLD);
    return;
}

static void c_bcast(void* send_data, int sendcount, int typ_enum)
{
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    MPI_Bcast(send_data, sendcount, mpi_typ, ROOT_PE, MPI_COMM_WORLD);
    return;
}

static void c_alltoall(void* send_data, void* recv_data, int count, int typ_enum)
{
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    MPI_Alltoall(send_data, count, mpi_typ, recv_data, count, mpi_typ, MPI_COMM_WORLD);
}

static void c_alltoallv(
    void* send_data, void* recv_data, int* send_counts, int* recv_counts, int* send_disp, int* recv_disp, int typ_enum)
{
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    MPI_Alltoallv(
        send_data, send_counts, send_disp, mpi_typ, recv_data, recv_counts, recv_disp, mpi_typ, MPI_COMM_WORLD);
}

static int hpat_finalize()
{
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized)
    {
        return 0;
    }
    int is_finalized;
    MPI_Finalized(&is_finalized);
    if (!is_finalized)
    {
        // printf("finalizing\n");
        MPI_Finalize();
    }
    return 0;
}

static void permutation_int(int64_t* output, int n)
{
    MPI_Bcast(output, n, MPI_INT64_T, 0, MPI_COMM_WORLD);
}

// Applies the permutation represented by |p| of size |p_len| to the array |rhs|
// of elements of size |elem_size| and stores the result in |lhs|.
static void permutation_array_index(
    unsigned char* lhs, int64_t len, int64_t elem_size, unsigned char* rhs, int64_t* p, int64_t p_len)
{
    if (len != p_len)
    {
        cerr << "Array length and permutation index length should match!\n";
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
    vector<unsigned char> send_buf(dest_ranks.size() * elem_size);
    for (size_t i = 0; i < dest_ranks.size(); ++i)
    {
        auto send_buf_offset = offsets[dest_ranks[i]]++ * elem_size;
        auto* send_buf_begin = send_buf.data() + send_buf_offset;
        auto* rhs_begin = rhs + i * elem_size;
        copy(rhs_begin, rhs_begin + elem_size, send_buf_begin);
    }

    MPI_Alltoallv(send_buf.data(),
                  send_counts.data(),
                  send_disps.data(),
                  element_t,
                  lhs,
                  recv_counts.data(),
                  recv_disps.data(),
                  element_t,
                  MPI_COMM_WORLD);

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

    int64_t* send_counts = new int64_t[num_pes];
    int64_t* recv_counts = new int64_t[num_pes];
    int64_t* send_disp = new int64_t[num_pes];
    int64_t* recv_disp = new int64_t[num_pes];

    int64_t curr_send_offset = 0;
    int64_t curr_recv_offset = 0;

    for (int i = 0; i < num_pes; i++)
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
            send_counts[i] = min(my_old_end, pe_new_end) - max(my_old_start, pe_new_start);
            curr_send_offset += send_counts[i];
        }

        // if receiving from processor (interval overlap)
        if (my_new_end > pe_old_start && my_new_start < pe_old_end)
        {
            recv_counts[i] = min(pe_old_end, my_new_end) - max(pe_old_start, my_new_start);
            curr_recv_offset += recv_counts[i];
        }
    }
    // printf("rank:%d send %lld %lld recv %lld %lld\n", rank, send_counts[0], send_counts[1], recv_counts[0], recv_counts[1]);
    // printf("send %d recv %d send_disp %d recv_disp %d\n", send_counts[0], recv_counts[0], send_disp[0], recv_disp[0]);
    // printf("data %lld %lld\n", ((int64_t*)input)[0], ((int64_t*)input)[1]);

    // workaround MPI int limit if necessary
    int* i_send_counts = new int[num_pes];
    int* i_recv_counts = new int[num_pes];
    int* i_send_disp = new int[num_pes];
    int* i_recv_disp = new int[num_pes];
    bool big_shuffle = false;

    for (int i = 0; i < num_pes; i++)
    {
        // any value doesn't fit in int
        if (send_counts[i] >= (int64_t)INT_MAX || recv_counts[i] >= (int64_t)INT_MAX ||
            send_disp[i] >= (int64_t)INT_MAX || recv_disp[i] >= (int64_t)INT_MAX)
        {
            big_shuffle = true;
            break;
        }
        i_send_counts[i] = (int)send_counts[i];
        i_recv_counts[i] = (int)recv_counts[i];
        i_send_disp[i] = (int)send_disp[i];
        i_recv_disp[i] = (int)recv_disp[i];
    }

    if (!big_shuffle)
    {
        int ierr = MPI_Alltoallv(
            input, i_send_counts, i_send_disp, MPI_CHAR, output, i_recv_counts, i_recv_disp, MPI_CHAR, MPI_COMM_WORLD);
        if (ierr != 0)
            cerr << "small shuffle error: " << '\n';
    }
    else
    {
        // char err_string[MPI_MAX_ERROR_STRING];
        // err_string[MPI_MAX_ERROR_STRING-1] = '\0';
        // int err_len, err_class;
        // MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

        int* l_send_counts = new int[num_pes];
        int* l_recv_counts = new int[num_pes];
        int* l_send_disp = new int[num_pes];
        int* l_recv_disp = new int[num_pes];

        int64_t* send_offset = new int64_t[num_pes];
        int64_t* recv_offset = new int64_t[num_pes];

        MPI_Datatype large_dtype;
        MPI_Type_contiguous(LARGE_DTYPE_SIZE, MPI_CHAR, &large_dtype);
        MPI_Type_commit(&large_dtype);

        for (int i = 0; i < num_pes; i++)
        {
            // large values
            i_send_counts[i] = (int)(send_counts[i] / LARGE_DTYPE_SIZE);
            i_recv_counts[i] = (int)(recv_counts[i] / LARGE_DTYPE_SIZE);
            i_send_disp[i] = (int)(send_disp[i] / LARGE_DTYPE_SIZE);
            i_recv_disp[i] = (int)(recv_disp[i] / LARGE_DTYPE_SIZE);
            // leftover values
            l_send_counts[i] = (int)(send_counts[i] % LARGE_DTYPE_SIZE);
            l_recv_counts[i] = (int)(recv_counts[i] % LARGE_DTYPE_SIZE);
            l_send_disp[i] = (int)(send_disp[i] % LARGE_DTYPE_SIZE);
            l_recv_disp[i] = (int)(recv_disp[i] % LARGE_DTYPE_SIZE);
            // printf("pe %d rank %d send %d recv %d sdisp %d rdisp %d lsend %d lrecv %d lsdisp %d lrdisp %d\n", i, rank,
            //         i_send_counts[i], i_recv_counts[i], i_send_disp[i], i_recv_disp[i],
            //         l_send_counts[i], l_recv_counts[i], l_send_disp[i], l_recv_disp[i]);
        }

        int64_t curr_send_buff_offset = 0;
        int64_t curr_recv_buff_offset = 0;
        // compute buffer offsets
        for (int i = 0; i < num_pes; i++)
        {
            send_offset[i] = curr_send_buff_offset;
            recv_offset[i] = curr_recv_buff_offset;
            curr_send_buff_offset += send_counts[i];
            curr_recv_buff_offset += recv_counts[i];
            // printf("pe %d rank %d send offset %lld recv offset %lld\n", i, rank, send_offset[i], recv_offset[i]);
        }

        // XXX implement alltoallv manually
        for (int i = 0; i < num_pes; i++)
        {
            int TAG = 11; // arbitrary
            int dest = (rank + i + num_pes) % num_pes;
            int src = (rank - i + num_pes) % num_pes;
            // printf("rank %d src %d dest %d\n", rank, src, dest);
            // send big type
            int ierr = MPI_Sendrecv(input + send_offset[dest],
                                    i_send_counts[dest],
                                    large_dtype,
                                    dest,
                                    TAG,
                                    output + recv_offset[src],
                                    i_recv_counts[src],
                                    large_dtype,
                                    src,
                                    TAG,
                                    MPI_COMM_WORLD,
                                    MPI_STATUS_IGNORE);
            if (ierr != 0)
                cerr << "large sendrecv error" << '\n';
            // send leftover
            ierr = MPI_Sendrecv(input + send_offset[dest] + ((int64_t)i_send_counts[dest]) * LARGE_DTYPE_SIZE,
                                l_send_counts[dest],
                                MPI_CHAR,
                                dest,
                                TAG + 1,
                                output + recv_offset[src] + ((int64_t)i_recv_counts[dest]) * LARGE_DTYPE_SIZE,
                                l_recv_counts[src],
                                MPI_CHAR,
                                src,
                                TAG + 1,
                                MPI_COMM_WORLD,
                                MPI_STATUS_IGNORE);
            if (ierr != 0)
                cerr << "small sendrecv error" << '\n';
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

/**
 * Code moved from hpat/_quantile_alg.cpp
 */

template <class T>
pair<T, T> get_lower_upper_kth_parallel(
    vector<T>& my_array, int64_t total_size, int myrank, int n_pes, int64_t k, int type_enum)
{
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    int64_t local_size = my_array.size();
    default_random_engine r_engine(myrank);
    uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    int64_t sample_size = (int64_t)(pow(10.0, 5.0) / n_pes); // 100000 total
    int64_t my_sample_size = min(sample_size, local_size);

    vector<T> my_sample;
    for (int64_t i = 0; i < my_sample_size; i++)
    {
        int64_t index = (int64_t)(local_size * uniform_dist(r_engine));
        my_sample.push_back(my_array[index]);
    }
    /* select sample */
    // get total sample size;
    vector<T> all_sample_vec;
    int* rcounts = new int[n_pes];
    int* displs = new int[n_pes];
    int total_sample_size = 0;
    // gather the sample sizes
    MPI_Gather(&my_sample_size, 1, MPI_INT, rcounts, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    // calculate size and displacements on root
    if (myrank == ROOT)
    {
        for (int i = 0; i < n_pes; i++)
        {
            // printf("rc %d\n", rcounts[i]);
            displs[i] = total_sample_size;
            total_sample_size += rcounts[i];
        }
        // printf("total sample size: %d\n", total_sample_size);
        all_sample_vec.resize(total_sample_size);
    }
    // gather sample data
    MPI_Gatherv(my_sample.data(),
                my_sample_size,
                mpi_typ,
                all_sample_vec.data(),
                rcounts,
                displs,
                mpi_typ,
                ROOT,
                MPI_COMM_WORLD);
    T k1_val;
    T k2_val;
    if (myrank == ROOT)
    {
        int local_k = (int)(k * (total_sample_size / (T)total_size));
        // printf("k:%ld local_k:%d\n", k, local_k);
        int k1 = (int)(local_k - sqrt(total_sample_size * log(total_size)));
        int k2 = (int)(local_k + sqrt(total_sample_size * log(total_size)));
        k1 = max(k1, 0);
        k2 = min(k2, total_sample_size - 1);
        // printf("k1: %d k2: %d\n", k1, k2);
        nth_element(all_sample_vec.begin(), all_sample_vec.begin() + k1, all_sample_vec.end());
        k1_val = all_sample_vec[k1];
        nth_element(all_sample_vec.begin(), all_sample_vec.begin() + k2, all_sample_vec.end());
        k2_val = all_sample_vec[k2];
        // printf("k1: %d k2: %d k1_val: %lf k2_val:%lf\n", k1, k2, k1_val, k2_val);
    }
    MPI_Bcast(&k1_val, 1, mpi_typ, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&k2_val, 1, mpi_typ, ROOT, MPI_COMM_WORLD);
    // cleanup
    delete[] rcounts;
    delete[] displs;
    return make_pair(k1_val, k2_val);
}

template <class T>
T small_get_nth_parallel(vector<T>& my_array, int64_t total_size, int myrank, int n_pes, int64_t k, int type_enum)
{
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    T res;
    int my_data_size = my_array.size();
    int total_data_size = 0;
    vector<T> all_data_vec;

    // no need to gather data if only 1 processor
    if (n_pes == 1)
    {
        nth_element(my_array.begin(), my_array.begin() + k, my_array.end());
        res = my_array[k];
        return res;
    }

    // gather the data sizes
    int* rcounts = new int[n_pes];
    int* displs = new int[n_pes];
    MPI_Gather(&my_data_size, 1, MPI_INT, rcounts, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    // calculate size and displacements on root
    if (myrank == ROOT)
    {
        for (int i = 0; i < n_pes; i++)
        {
            // printf("rc %d\n", rcounts[i]);
            displs[i] = total_data_size;
            total_data_size += rcounts[i];
        }
        // printf("total small data size: %d\n", total_data_size);
        all_data_vec.resize(total_data_size);
    }
    // gather data
    MPI_Gatherv(
        my_array.data(), my_data_size, mpi_typ, all_data_vec.data(), rcounts, displs, mpi_typ, ROOT, MPI_COMM_WORLD);
    // get nth element on root
    if (myrank == ROOT)
    {
        nth_element(all_data_vec.begin(), all_data_vec.begin() + k, all_data_vec.end());
        res = all_data_vec[k];
    }
    MPI_Bcast(&res, 1, mpi_typ, ROOT, MPI_COMM_WORLD);
    delete[] rcounts;
    delete[] displs;
    return res;
}

template <class T>
T get_nth_parallel(vector<T>& my_array, int64_t k, int myrank, int n_pes, int type_enum)
{
    int64_t local_size = my_array.size();
    int64_t total_size;
    MPI_Allreduce(&local_size, &total_size, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    // printf("total size: %ld k: %ld\n", total_size, k);
    int64_t threshold = (int64_t)pow(10.0, 7.0); // 100 million
    // int64_t threshold = 20;
    if (total_size < threshold || n_pes == 1)
    {
        return small_get_nth_parallel(my_array, total_size, myrank, n_pes, k, type_enum);
    }
    else
    {
        pair<T, T> kths = get_lower_upper_kth_parallel(my_array, total_size, myrank, n_pes, k, type_enum);
        T k1_val = kths.first;
        T k2_val = kths.second;
        // printf("k1_val: %lf  k2_val: %lf\n", k1_val, k2_val);
        int64_t local_l0_num = 0, local_l1_num = 0, local_l2_num = 0;
        int64_t l0_num = 0, l1_num = 0, l2_num = 0;
        for (auto val : my_array)
        {
            if (val < k1_val)
            {
                local_l0_num++;
            }

            if (val >= k1_val && val < k2_val)
            {
                local_l1_num++;
            }

            if (val >= k2_val)
            {
                local_l2_num++;
            }
        }

        MPI_Allreduce(&local_l0_num, &l0_num, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_l1_num, &l1_num, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_l2_num, &l2_num, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
        // printf("set sizes: %ld %ld %ld\n", l0_num, l1_num, l2_num);
        assert(l0_num + l1_num + l2_num == total_size);
        // []----*---o----*-----]
        // if there are more elements in the last set than elemenet k to end,
        // this means k2 is equal to k
        if (l2_num > total_size - k)
            return k2_val;
        assert(l0_num < k);

        vector<T> new_my_array;
        int64_t new_k = k;

        int64_t new_ind = 0;
        if (k < l0_num)
        {
            // first set
            // printf("first set\n");
            new_my_array.resize(local_l0_num);
            // throw away
            for (auto val : my_array)
            {
                if (val < k1_val)
                {
                    new_my_array[new_ind] = val;
                    new_ind++;
                }
            }
            // new_k doesn't change
        }
        else if (k < l0_num + l1_num)
        {
            // middle set
            // printf("second set\n");
            new_my_array.resize(local_l1_num);
            for (auto val : my_array)
            {
                if (val >= k1_val && val < k2_val)
                {
                    new_my_array[new_ind] = val;
                    new_ind++;
                }
            }
            new_k -= l0_num;
        }
        else
        {
            // last set
            // printf("last set\n");
            new_my_array.resize(local_l2_num);
            for (auto val : my_array)
            {
                if (val >= k2_val)
                {
                    new_my_array[new_ind] = val;
                    new_ind++;
                }
            }
            new_k -= (l0_num + l1_num);
        }

        return get_nth_parallel(new_my_array, new_k, myrank, n_pes, type_enum);
    }

    return (T)-1.0;
}

template <class T>
double quantile_parallel_float(T* data, int64_t local_size, double quantile, int type_enum, int myrank, int n_pes)
{
    vector<T> my_array(data, data + local_size);
    // delete NaNs
    my_array.erase(remove_if(begin(my_array), end(my_array), [](T d) { return isnan(d); }), my_array.end());
    local_size = my_array.size();
    // recalculate total size since there could be NaNs
    int64_t total_size;
    MPI_Allreduce(&local_size, &total_size, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    double at = quantile * (total_size - 1);
    int64_t k1 = (int64_t)at;
    int64_t k2 = k1 + 1;
    double fraction = at - (double)k1;

    double res1 = (double)get_nth_parallel(my_array, k1, myrank, n_pes, type_enum);
    double res2 = (double)get_nth_parallel(my_array, k2, myrank, n_pes, type_enum);
    // linear method, TODO: support other methods
    return res1 + (res2 - res1) * fraction;
}

template <class T>
double quantile_parallel_int(T* data, int64_t local_size, double at, int type_enum, int myrank, int n_pes)
{
    int64_t k1 = (int64_t)at;
    int64_t k2 = k1 + 1;
    double fraction = at - (double)k1;
    vector<T> my_array(data, data + local_size);
    double res1 = (double)get_nth_parallel(my_array, k1, myrank, n_pes, type_enum);
    double res2 = (double)get_nth_parallel(my_array, k2, myrank, n_pes, type_enum);
    // linear method, TODO: support other methods
    return res1 + (res2 - res1) * fraction;
}

static double quantile_parallel(void* data, int64_t local_size, int64_t total_size, double quantile, int type_enum)
{
    int myrank, n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (total_size == 0)
        MPI_Allreduce(&local_size, &total_size, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

    // return NA if no elements
    if (total_size == 0)
    {
        return nan("");
    }

    double at = quantile * (total_size - 1);

    switch (type_enum)
    {
    case SDC_CTypes::INT8:
        return quantile_parallel_int((char*)data, local_size, at, type_enum, myrank, n_pes);
    case SDC_CTypes::UINT8:
        return quantile_parallel_int((unsigned char*)data, local_size, at, type_enum, myrank, n_pes);
    case SDC_CTypes::INT32:
        return quantile_parallel_int((int*)data, local_size, at, type_enum, myrank, n_pes);
    case SDC_CTypes::UINT32:
        return quantile_parallel_int((uint32_t*)data, local_size, at, type_enum, myrank, n_pes);
    case SDC_CTypes::INT64:
        return quantile_parallel_int((int64_t*)data, local_size, at, type_enum, myrank, n_pes);
    case SDC_CTypes::UINT64:
        return quantile_parallel_int((uint64_t*)data, local_size, quantile, type_enum, myrank, n_pes);
    case SDC_CTypes::FLOAT32:
        return quantile_parallel_float((float*)data, local_size, quantile, type_enum, myrank, n_pes);
    case SDC_CTypes::FLOAT64:
        return quantile_parallel_float((double*)data, local_size, quantile, type_enum, myrank, n_pes);
    default:
        cerr << "unknown quantile data type\n";
    }

    return -1.0;
}

template <class T>
void get_nth(T* res, T* data, int64_t local_size, int64_t k, int type_enum, int myrank, int n_pes, bool parallel)
{
    // get nth element and store in res pointer
    // assuming NA values of floats are already removed
    vector<T> my_array(data, data + local_size);
    T val;

    if (parallel)
    {
        val = get_nth_parallel(my_array, k, myrank, n_pes, type_enum);
    }
    else
    {
        nth_element(my_array.begin(), my_array.begin() + k, my_array.end());
        val = my_array[k];
    }
    *res = val;
}

static void nth_dispatch(void* res, void* data, int64_t local_size, int64_t k, int type_enum, bool parallel)
{
    int myrank = 0;
    int n_pes = 1;

    if (parallel)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    }

    switch (type_enum)
    {
    case SDC_CTypes::INT8:
        return get_nth((char*)res, (char*)data, local_size, k, type_enum, myrank, n_pes, parallel);
    case SDC_CTypes::UINT8:
        return get_nth((unsigned char*)res, (unsigned char*)data, local_size, k, type_enum, myrank, n_pes, parallel);
    case SDC_CTypes::INT32:
        return get_nth((int*)res, (int*)data, local_size, k, type_enum, myrank, n_pes, parallel);
    case SDC_CTypes::UINT32:
        return get_nth((uint32_t*)res, (uint32_t*)data, local_size, k, type_enum, myrank, n_pes, parallel);
    case SDC_CTypes::INT64:
        return get_nth((int64_t*)res, (int64_t*)data, local_size, k, type_enum, myrank, n_pes, parallel);
    case SDC_CTypes::UINT64:
        return get_nth((uint64_t*)res, (uint64_t*)data, local_size, k, type_enum, myrank, n_pes, parallel);
    case SDC_CTypes::FLOAT32:
        return get_nth((float*)res, (float*)data, local_size, k, type_enum, myrank, n_pes, parallel);
    case SDC_CTypes::FLOAT64:
        return get_nth((double*)res, (double*)data, local_size, k, type_enum, myrank, n_pes, parallel);
    default:
        cerr << "unknown nth data type\n";
    }
}

static void nth_sequential(void* res, void* data, int64_t local_size, int64_t k, int type_enum)
{
    nth_dispatch(res, data, local_size, k, type_enum, false);
}

static void nth_parallel(void* res, void* data, int64_t local_size, int64_t k, int type_enum)
{
    nth_dispatch(res, data, local_size, k, type_enum, true);
}

PyMODINIT_FUNC PyInit_transport_mpi(void)
{
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "transport_mpi",
        "MPI inter-process transport functions",
        -1,
        NULL,
    };

    PyObject* m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "allgather", PyLong_FromVoidPtr((void*)(&allgather)));
    PyObject_SetAttrString(m, "c_alltoall", PyLong_FromVoidPtr((void*)(&c_alltoall)));
    PyObject_SetAttrString(m, "c_alltoallv", PyLong_FromVoidPtr((void*)(&c_alltoallv)));
    PyObject_SetAttrString(m, "c_bcast", PyLong_FromVoidPtr((void*)(&c_bcast)));
    PyObject_SetAttrString(m, "c_gather_scalar", PyLong_FromVoidPtr((void*)(&c_gather_scalar)));
    PyObject_SetAttrString(m, "c_gatherv", PyLong_FromVoidPtr((void*)(&c_gatherv)));
    PyObject_SetAttrString(m, "comm_req_alloc", PyLong_FromVoidPtr((void*)(&comm_req_alloc)));
    PyObject_SetAttrString(m, "comm_req_dealloc", PyLong_FromVoidPtr((void*)(&comm_req_dealloc)));
    PyObject_SetAttrString(m, "file_read_parallel", PyLong_FromVoidPtr((void*)(&file_read_parallel)));
    PyObject_SetAttrString(m, "file_write_parallel", PyLong_FromVoidPtr((void*)(&file_write_parallel)));
    PyObject_SetAttrString(m, "get_file_size", PyLong_FromVoidPtr((void*)(&get_file_size)));
    PyObject_SetAttrString(m, "get_join_sendrecv_counts", PyLong_FromVoidPtr((void*)(&get_join_sendrecv_counts)));
    PyObject_SetAttrString(m, "hpat_barrier", PyLong_FromVoidPtr((void*)(&hpat_barrier)));
    PyObject_SetAttrString(m, "hpat_dist_arr_reduce", PyLong_FromVoidPtr((void*)(&hpat_dist_arr_reduce)));
    PyObject_SetAttrString(m, "hpat_dist_exscan_f4", PyLong_FromVoidPtr((void*)(&hpat_dist_exscan_f4)));
    PyObject_SetAttrString(m, "hpat_dist_exscan_f8", PyLong_FromVoidPtr((void*)(&hpat_dist_exscan_f8)));
    PyObject_SetAttrString(m, "hpat_dist_exscan_i4", PyLong_FromVoidPtr((void*)(&hpat_dist_exscan_i4)));
    PyObject_SetAttrString(m, "hpat_dist_exscan_i8", PyLong_FromVoidPtr((void*)(&hpat_dist_exscan_i8)));
    PyObject_SetAttrString(m, "hpat_dist_get_rank", PyLong_FromVoidPtr((void*)(&hpat_dist_get_rank)));
    PyObject_SetAttrString(m, "hpat_dist_get_size", PyLong_FromVoidPtr((void*)(&hpat_dist_get_size)));
    PyObject_SetAttrString(m, "hpat_dist_get_time", PyLong_FromVoidPtr((void*)(&hpat_dist_get_time)));
    PyObject_SetAttrString(m, "hpat_dist_irecv", PyLong_FromVoidPtr((void*)(&hpat_dist_irecv)));
    PyObject_SetAttrString(m, "hpat_dist_isend", PyLong_FromVoidPtr((void*)(&hpat_dist_isend)));
    PyObject_SetAttrString(m, "hpat_dist_recv", PyLong_FromVoidPtr((void*)(&hpat_dist_recv)));
    PyObject_SetAttrString(m, "hpat_dist_reduce", PyLong_FromVoidPtr((void*)(&hpat_dist_reduce)));
    PyObject_SetAttrString(m, "hpat_dist_send", PyLong_FromVoidPtr((void*)(&hpat_dist_send)));
    PyObject_SetAttrString(m, "hpat_dist_wait", PyLong_FromVoidPtr((void*)(&hpat_dist_wait)));
    PyObject_SetAttrString(m, "hpat_dist_waitall", PyLong_FromVoidPtr((void*)(&hpat_dist_waitall)));
    PyObject_SetAttrString(m, "hpat_finalize", PyLong_FromVoidPtr((void*)(&hpat_finalize)));
    PyObject_SetAttrString(m, "hpat_get_time", PyLong_FromVoidPtr((void*)(&hpat_get_time)));
    PyObject_SetAttrString(m, "hpat_mpi_csv_get_offsets", PyLong_FromVoidPtr((void*)(&hpat_mpi_csv_get_offsets)));
    PyObject_SetAttrString(m, "mpi_req_num_bytes", PyLong_FromSize_t(get_mpi_req_num_bytes()));
    PyObject_SetAttrString(m, "nth_parallel", PyLong_FromVoidPtr((void*)(&nth_parallel)));
    PyObject_SetAttrString(m, "nth_sequential", PyLong_FromVoidPtr((void*)(&nth_sequential)));
    PyObject_SetAttrString(m, "oneD_reshape_shuffle", PyLong_FromVoidPtr((void*)(&oneD_reshape_shuffle)));
    PyObject_SetAttrString(m, "permutation_array_index", PyLong_FromVoidPtr((void*)(&permutation_array_index)));
    PyObject_SetAttrString(m, "permutation_int", PyLong_FromVoidPtr((void*)(&permutation_int)));
    PyObject_SetAttrString(m, "quantile_parallel", PyLong_FromVoidPtr((void*)(&quantile_parallel)));
    PyObject_SetAttrString(m, "req_array_setitem", PyLong_FromVoidPtr((void*)(&req_array_setitem)));

    return m;
}

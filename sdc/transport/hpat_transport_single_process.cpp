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
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32 // MSC_VER
#include <Windows.h>

// no gettimeofday on Win32/Win64
int gettimeofday(struct timeval* tp, struct timezone* tzp)
{
    static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

    SYSTEMTIME nSystemTime;
    FILETIME nFileTime;
    uint64_t nTime;

    GetSystemTime(&nSystemTime);
    SystemTimeToFileTime(&nSystemTime, &nFileTime);
    nTime = ((uint64_t)nFileTime.dwLowDateTime);
    nTime += ((uint64_t)nFileTime.dwHighDateTime) << 32;

    tp->tv_sec = (long)((nTime - EPOCH) / 10000000L);
    tp->tv_usec = (long)(nSystemTime.wMilliseconds * 1000);
    return 0;
}
#else
#include <sys/time.h>
#endif // _WIN32

#include "../_hpat_common.h"

using namespace std;

typedef int MPI_Request;
#define MPI_REQUEST_NULL ((MPI_Request)0x2c000000)

static size_t get_type_size_bytes(int typ_enum)
{
    switch (typ_enum)
    {
    case SDC_CTypes::INT8:
    case SDC_CTypes::UINT8:
    {
        return 1;
    }
    case SDC_CTypes::INT16:
    case SDC_CTypes::UINT16:
    {
        return 2;
    }
    case SDC_CTypes::INT32:
    case SDC_CTypes::UINT32:
    case SDC_CTypes::FLOAT32:
    {
        return 4;
    }
    case SDC_CTypes::INT64:
    case SDC_CTypes::UINT64:
    case SDC_CTypes::FLOAT64:
    {
        return 8;
    }
    default:
    {
        throw out_of_range("Invalid data type in transport_seq::get_type_size_bytes()");
    }
    }

    return 0;
}

static void allgather(void* out_data, int size, void* in_data, int type_enum)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));
}

static void c_alltoall(void* send_data, void* recv_data, int count, int typ_enum)
{
    size_t type_size_bytes = get_type_size_bytes(typ_enum);
    memcpy(recv_data, send_data, type_size_bytes * count);
}

static void c_alltoallv(
    void* send_data, void* recv_data, int* send_counts, int* recv_counts, int* send_disp, int* recv_disp, int typ_enum)
{
    size_t type_size_bytes = get_type_size_bytes(typ_enum);
    memcpy((char*)recv_data + recv_disp[0],
           (char*)send_data + send_disp[0],
           type_size_bytes * min(send_counts[0], recv_counts[0]));
}

static void c_bcast(void* send_data, int sendcount, int typ_enum)
{
    // no work needed
}

static void c_gather_scalar(void* send_data, void* recv_data, int typ_enum)
{
    size_t type_size_bytes = get_type_size_bytes(typ_enum);
    memcpy(recv_data, send_data, type_size_bytes * 1);
}

static void c_gatherv(void* send_data, int sendcount, void* recv_data, int* recv_counts, int* displs, int typ_enum)
{
    size_t type_size_bytes = get_type_size_bytes(typ_enum);
    memcpy((char*)recv_data + displs[0], send_data, type_size_bytes * min(sendcount, recv_counts[0]));
}

static MPI_Request* comm_req_alloc(int size)
{
    return new MPI_Request[size];
}

static void comm_req_dealloc(MPI_Request* req_arr)
{
    delete[] req_arr;
}

static void file_read_parallel(char* file_name, char* buff, int64_t start, int64_t count)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));
}

static void file_write_parallel(char* file_name, char* buff, int64_t start, int64_t count, int64_t elem_size)
{
    ofstream user_file(file_name, ios::binary | ios::out);
    if (!user_file.good())
    {
        throw runtime_error(__FUNCTION__ + string(": Could not open file: ") + file_name);
    }

    user_file.seekp(start * elem_size);
    user_file.write(buff, count * elem_size);

    user_file.close();
}

static uint64_t get_file_size(const char* file_name)
{
    ifstream user_file(file_name, ifstream::binary);
    if (!user_file.good())
    {
        throw runtime_error(__FUNCTION__ + string(": Could not open file: ") + file_name);
    }

    user_file.seekg(0, ios::beg);
    const iostream::pos_type begin = user_file.tellg();
    user_file.seekg(0, ios::end);
    const iostream::pos_type end = user_file.tellg();

    user_file.close();

    if ((begin < 0) || (end < 0))
    {
        throw runtime_error(__FUNCTION__ + string(": Could not read from file: ") + file_name);
    }

    return end - begin;
}

static int64_t get_join_sendrecv_counts(int** p_send_counts,
                                        int** p_recv_counts,
                                        int** p_send_disp,
                                        int** p_recv_disp,
                                        int64_t arr_len,
                                        int type_enum,
                                        void* data)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));
}

static int hpat_barrier()
{
    return 0;
}

static int hpat_dist_arr_reduce(void* out, int64_t* shapes, int ndims, int op_enum, int type_enum)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));
}

static float hpat_dist_exscan_f4(float value)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));
}

static double hpat_dist_exscan_f8(double value)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));
}

static int hpat_dist_exscan_i4(int value)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));
}

static int64_t hpat_dist_exscan_i8(int64_t value)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));
}

static int hpat_dist_get_rank()
{
    return 0;
}

static int hpat_dist_get_size()
{
    return 1;
}

static double hpat_dist_get_time()
{
    timeval result;
    gettimeofday(&result, nullptr);
    double sec = result.tv_sec;
    double usec = result.tv_usec;

    return sec + (usec / 1E6);
}

static MPI_Request hpat_dist_irecv(void* out, int size, int type_enum, int pe, int tag, bool cond)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));
}

static MPI_Request hpat_dist_isend(void* out, int size, int type_enum, int pe, int tag, bool cond)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));
}

static void hpat_dist_recv(void* out, int size, int type_enum, int pe, int tag)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));
}

static void hpat_dist_reduce(char* in_ptr, char* out_ptr, int op_enum, int type_enum)
{
    size_t type_size_bytes = get_type_size_bytes(type_enum);
    memcpy(out_ptr, in_ptr, type_size_bytes * 1);
}

static void hpat_dist_send(void* out, int size, int type_enum, int pe, int tag)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));
}

static int hpat_dist_wait(MPI_Request req, bool cond)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));
}

static void hpat_dist_waitall(int size, MPI_Request* req_arr)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));
}

static int hpat_finalize()
{
    return 0;
}

static double hpat_get_time()
{
    return hpat_dist_get_time();
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
        {
            pos.push_back(i);
        }
        ++i;
    }

    if (i < n)
    {
        cerr << "Warning, read only " << i << " bytes out of " << n << "requested\n";
    }

    return pos;
}

static void hpat_mpi_csv_get_offsets(
    istream* f, size_t fsz, bool is_parallel, int64_t skiprows, int64_t nrows, size_t& my_off_start, size_t& my_off_end)
{
    if (skiprows > 0 || nrows != -1)
    {
        vector<size_t> line_offset = count_lines(f, fsz);

        if (skiprows > 0)
        {
            my_off_start = line_offset[skiprows - 1] + 1;
        }

        if (nrows != -1)
        {
            my_off_end = line_offset[nrows - 1] + 1;
        }
    }

    return;
}

static size_t get_mpi_req_num_bytes()
{
    return sizeof(MPI_Request);
}

static void nth_parallel(void* res, void* data, int64_t local_size, int64_t k, int type_enum)
{
    throw runtime_error(__FUNCTION__ + string(": Should not be called"));
}

template <typename T>
void get_nth(void* result_out, void* data_in, int64_t size, int64_t k)
{
    T* result = reinterpret_cast<T*>(result_out);
    const T* data = reinterpret_cast<T*>(data_in);

    vector<T> my_array(data, data + size);

    nth_element(my_array.begin(), my_array.begin() + k, my_array.end());

    *result = my_array.at(k);
}

static void nth_sequential(void* res, void* data, int64_t local_size, int64_t k, int type_enum)
{
    switch (type_enum)
    {
    case SDC_CTypes::INT8:
    {
        get_nth<char>(res, data, local_size, k);
        break;
    }
    case SDC_CTypes::UINT8:
    {
        get_nth<unsigned char>(res, data, local_size, k);
        break;
    }
    case SDC_CTypes::INT32:
    {
        get_nth<int>(res, data, local_size, k);
        break;
    }
    case SDC_CTypes::UINT32:
    {
        get_nth<uint32_t>(res, data, local_size, k);
        break;
    }
    case SDC_CTypes::INT64:
    {
        get_nth<int64_t>(res, data, local_size, k);
        break;
    }
    case SDC_CTypes::UINT64:
    {
        get_nth<uint64_t>(res, data, local_size, k);
        break;
    }
    case SDC_CTypes::FLOAT32:
    {
        get_nth<float>(res, data, local_size, k);
        break;
    }
    case SDC_CTypes::FLOAT64:
    {
        get_nth<double>(res, data, local_size, k);
        break;
    }
    default:
    {
        throw out_of_range("Invalid data type in transport_seq::nth_sequential()");
    }
    }
}

static void oneD_reshape_shuffle(char* output,
                                 char* input,
                                 int64_t new_0dim_global_len,
                                 int64_t old_0dim_global_len,
                                 int64_t out_lower_dims_size,
                                 int64_t in_lower_dims_size)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));
}

static void permutation_array_index(
    unsigned char* lhs, int64_t len, int64_t elem_size, unsigned char* rhs, int64_t* p, int64_t p_len)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));
}

static void permutation_int(int64_t* output, int n)
{
    // no action needed
}

static double quantile_parallel(void* data, int64_t local_size, int64_t total_size, double quantile, int type_enum)
{
    throw runtime_error(__FUNCTION__ + string(": Is not implemented"));

    return -1.0;
}

static void req_array_setitem(MPI_Request* req_arr, int64_t ind, MPI_Request req)
{
    req_arr[ind] = req;
    return;
}

PyMODINIT_FUNC PyInit_transport_seq(void)
{
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "transport_seq",
        "Single process transport functions as a stub",
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

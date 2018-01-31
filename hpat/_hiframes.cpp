#include "mpi.h"
#include <iostream>
#include <Python.h>
#include "_hpat_sort.h"

int64_t get_join_sendrecv_counts(int **p_send_counts, int **p_recv_counts,
                                int **p_send_disp, int **p_recv_disp,
                                int64_t arr_len, int type_enum, void* data);


PyMODINIT_FUNC PyInit_chiframes(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "chiframes", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "get_join_sendrecv_counts",
                        PyLong_FromVoidPtr((void*)(&get_join_sendrecv_counts)));
    PyObject_SetAttrString(m, "timsort",
                        PyLong_FromVoidPtr((void*)(&__hpat_timsort)));

    return m;
}

int64_t get_join_sendrecv_counts(int **p_send_counts, int **p_recv_counts,
                                int **p_send_disp, int **p_recv_disp,
                                int64_t arr_len, int type_enum, void* data)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    // alloc buffers
    int *send_counts = new int[n_pes];
    *p_send_counts = send_counts;
    int *recv_counts = new int[n_pes];
    *p_recv_counts = recv_counts;
    int *send_disp = new int[n_pes];
    *p_send_disp = send_disp;
    int *recv_disp = new int[n_pes];
    *p_recv_disp = recv_disp;

    // TODO: extend to other key types
    int64_t *key_arr = (int64_t*) data;
    memset(send_counts, 0, sizeof(int)*n_pes);
    for(int64_t i=0; i<arr_len; i++)
    {
        int node_id = key_arr[i] % n_pes;
        send_counts[node_id]++;
    }
    // send displacement
    send_disp[0] = 0;
    for(int64_t i=1; i<n_pes; i++)
    {
        send_disp[i] = send_disp[i-1] + send_counts[i-1];
    }
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
    // recv displacement
    recv_disp[0] = 0;
    for(int64_t i=1; i<n_pes; i++)
    {
        recv_disp[i] = recv_disp[i-1] + recv_counts[i-1];
    }
    // total recv size
    int64_t total_recv_size = 0;
    for(int64_t i=0; i<n_pes; i++)
    {
        total_recv_size += recv_counts[i];
    }
    return total_recv_size;
}

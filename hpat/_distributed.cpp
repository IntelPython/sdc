#include "_distributed.h"

PyMODINIT_FUNC PyInit_hdist(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "hdist", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "hpat_dist_get_rank",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_get_rank)));
    PyObject_SetAttrString(m, "hpat_dist_get_size",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_get_size)));
    PyObject_SetAttrString(m, "hpat_dist_get_start",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_get_start)));
    PyObject_SetAttrString(m, "hpat_dist_get_end",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_get_end)));
    PyObject_SetAttrString(m, "hpat_dist_get_node_portion",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_get_node_portion)));
    PyObject_SetAttrString(m, "hpat_dist_get_time",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_get_time)));
    PyObject_SetAttrString(m, "hpat_get_time",
                            PyLong_FromVoidPtr((void*)(&hpat_get_time)));
    PyObject_SetAttrString(m, "hpat_barrier",
                            PyLong_FromVoidPtr((void*)(&hpat_barrier)));

    PyObject_SetAttrString(m, "hpat_dist_reduce",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_reduce)));

    PyObject_SetAttrString(m, "hpat_dist_exscan_i4",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_exscan_i4)));
    PyObject_SetAttrString(m, "hpat_dist_exscan_i8",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_exscan_i8)));
    PyObject_SetAttrString(m, "hpat_dist_exscan_f4",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_exscan_f4)));
    PyObject_SetAttrString(m, "hpat_dist_exscan_f8",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_exscan_f8)));

    PyObject_SetAttrString(m, "hpat_dist_arr_reduce",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_arr_reduce)));
    PyObject_SetAttrString(m, "hpat_dist_irecv",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_irecv)));
    PyObject_SetAttrString(m, "hpat_dist_isend",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_isend)));
    PyObject_SetAttrString(m, "hpat_dist_recv",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_recv)));
    PyObject_SetAttrString(m, "hpat_dist_send",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_send)));
    PyObject_SetAttrString(m, "hpat_dist_wait",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_wait)));
    PyObject_SetAttrString(m, "hpat_dist_get_item_pointer",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_get_item_pointer)));
    PyObject_SetAttrString(m, "hpat_get_dummy_ptr",
                            PyLong_FromVoidPtr((void*)(&hpat_get_dummy_ptr)));
    PyObject_SetAttrString(m, "c_gather_scalar",
                            PyLong_FromVoidPtr((void*)(&c_gather_scalar)));
    PyObject_SetAttrString(m, "c_gatherv",
                            PyLong_FromVoidPtr((void*)(&c_gatherv)));
    PyObject_SetAttrString(m, "c_bcast",
                            PyLong_FromVoidPtr((void*)(&c_bcast)));
    PyObject_SetAttrString(m, "c_alltoallv",
                            PyLong_FromVoidPtr((void*)(&c_alltoallv)));
    PyObject_SetAttrString(m, "c_alltoall",
                            PyLong_FromVoidPtr((void*)(&c_alltoall)));
    PyObject_SetAttrString(m, "allgather",
                            PyLong_FromVoidPtr((void*)(&allgather)));
    PyObject_SetAttrString(m, "comm_req_alloc",
                            PyLong_FromVoidPtr((void*)(&comm_req_alloc)));
    PyObject_SetAttrString(m, "req_array_setitem",
                            PyLong_FromVoidPtr((void*)(&req_array_setitem)));
    PyObject_SetAttrString(m, "hpat_dist_waitall",
                            PyLong_FromVoidPtr((void*)(&hpat_dist_waitall)));
    PyObject_SetAttrString(m, "comm_req_dealloc",
                            PyLong_FromVoidPtr((void*)(&comm_req_dealloc)));

    PyObject_SetAttrString(m, "hpat_finalize",
                            PyLong_FromVoidPtr((void*)(&hpat_finalize)));
    PyObject_SetAttrString(m, "oneD_reshape_shuffle",
                            PyLong_FromVoidPtr((void*)(&oneD_reshape_shuffle)));
    PyObject_SetAttrString(m, "permutation_int",
                            PyLong_FromVoidPtr((void*)(&permutation_int)));
    PyObject_SetAttrString(m, "permutation_array_index",
                            PyLong_FromVoidPtr((void*)(&permutation_array_index)));

    // add actual int value to module
    PyObject_SetAttrString(m, "mpi_req_num_bytes",
                            PyLong_FromSize_t(get_mpi_req_num_bytes()));
    return m;
}

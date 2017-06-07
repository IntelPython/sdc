#include "mpi.h"
#include <Python.h>

int hpat_dist_get_rank();
int hpat_dist_get_size();
int64_t hpat_dist_get_end(int64_t total, int64_t div_chunk, int num_pes,
                            int node_id);
int64_t hpat_dist_get_node_portion(int64_t total, int64_t div_chunk,
                                    int num_pes, int node_id);
double hpat_dist_get_time();
double hpat_dist_reduce(double value);
int hpat_dist_arr_reduce(void* out, int64_t* shapes, int ndims, int type_enum);

PyMODINIT_FUNC PyInit_hdist(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "hdist", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "hpat_dist_get_rank",
                            PyLong_FromVoidPtr(&hpat_dist_get_rank));
    PyObject_SetAttrString(m, "hpat_dist_get_size",
                            PyLong_FromVoidPtr(&hpat_dist_get_size));
    PyObject_SetAttrString(m, "hpat_dist_get_end",
                            PyLong_FromVoidPtr(&hpat_dist_get_end));
    PyObject_SetAttrString(m, "hpat_dist_get_node_portion",
                            PyLong_FromVoidPtr(&hpat_dist_get_node_portion));
    PyObject_SetAttrString(m, "hpat_dist_get_time",
                            PyLong_FromVoidPtr(&hpat_dist_get_time));
    PyObject_SetAttrString(m, "hpat_dist_reduce",
                            PyLong_FromVoidPtr(&hpat_dist_reduce));
    PyObject_SetAttrString(m, "hpat_dist_arr_reduce",
                            PyLong_FromVoidPtr(&hpat_dist_arr_reduce));
    return m;
}

int hpat_dist_get_rank()
{
    MPI_Init(NULL,NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // printf("my_rank:%d\n", rank);
    return rank;
}

int hpat_dist_get_size()
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // printf("mpi_size:%d\n", size);
    return size;
}

int64_t hpat_dist_get_end(int64_t total, int64_t div_chunk, int num_pes,
                            int node_id)
{
    return ((node_id==num_pes-1) ? total : (node_id+1)*div_chunk);
}

int64_t hpat_dist_get_node_portion(int64_t total, int64_t div_chunk,
                                    int num_pes, int node_id)
{
    int64_t portion = ((node_id==num_pes-1) ? total-node_id*div_chunk : div_chunk);
    // printf("portion:%lld\n", portion);
    return portion;
}

double hpat_dist_get_time()
{
    double wtime;
    MPI_Barrier(MPI_COMM_WORLD);
    wtime = MPI_Wtime();
    return wtime;
}


double hpat_dist_reduce(double value)
{
    // printf("sum value: %lf\n", value);
    double out=0;
    MPI_Allreduce(&value, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

int hpat_dist_arr_reduce(void* out, int64_t* shapes, int ndims, int type_enum)
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
    int elem_size = get_elem_size(type_enum);
    void* res_buf = malloc(total_size*elem_size);
    MPI_Allreduce(out, res_buf, total_size, mpi_typ, MPI_SUM, MPI_COMM_WORLD);
    memcpy(out, res_buf, total_size*elem_size);
    free(res_buf);
    return 0;
}

// _h5_typ_table = {
//     int8:0,
//     uint8:1,
//     int32:2,
//     int64:3,
//     float32:4,
//     float64:5
//     }

MPI_Datatype get_MPI_typ(int typ_enum)
{
    // printf("h5 type enum:%d\n", typ_enum);
    MPI_Datatype types_list[] = {MPI_CHAR, MPI_UNSIGNED_CHAR,
            MPI_INT, MPI_LONG_LONG_INT, MPI_FLOAT, MPI_DOUBLE};
    return types_list[typ_enum];
}

int get_elem_size(int type_enum)
{
    int types_sizes[] = {1,1,4,8,4,8};
    return types_sizes[type_enum];
}

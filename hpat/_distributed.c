#include <stdbool.h>
#include "mpi.h"
#include <Python.h>

int hpat_dist_get_rank();
int hpat_dist_get_size();
int64_t hpat_dist_get_end(int64_t total, int64_t div_chunk, int num_pes,
                            int node_id);
int64_t hpat_dist_get_node_portion(int64_t total, int64_t div_chunk,
                                    int num_pes, int node_id);
double hpat_dist_get_time();

int hpat_dist_reduce_i4(int value);
int64_t hpat_dist_reduce_i8(int64_t value);
float hpat_dist_reduce_f4(float value);
double hpat_dist_reduce_f8(double value);

int hpat_dist_exscan_i4(int value);
int64_t hpat_dist_exscan_i8(int64_t value);
float hpat_dist_exscan_f4(float value);
double hpat_dist_exscan_f8(double value);

int hpat_dist_arr_reduce(void* out, int64_t* shapes, int ndims, int type_enum);
int hpat_dist_irecv(void* out, int size, int type_enum, int pe, int tag, bool cond);
int hpat_dist_isend(void* out, int size, int type_enum, int pe, int tag, bool cond);

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

    PyObject_SetAttrString(m, "hpat_dist_reduce_i4",
                            PyLong_FromVoidPtr(&hpat_dist_reduce_i4));
    PyObject_SetAttrString(m, "hpat_dist_reduce_i8",
                            PyLong_FromVoidPtr(&hpat_dist_reduce_i8));
    PyObject_SetAttrString(m, "hpat_dist_reduce_f4",
                            PyLong_FromVoidPtr(&hpat_dist_reduce_f4));
    PyObject_SetAttrString(m, "hpat_dist_reduce_f8",
                            PyLong_FromVoidPtr(&hpat_dist_reduce_f8));

    PyObject_SetAttrString(m, "hpat_dist_exscan_i4",
                            PyLong_FromVoidPtr(&hpat_dist_exscan_i4));
    PyObject_SetAttrString(m, "hpat_dist_exscan_i8",
                            PyLong_FromVoidPtr(&hpat_dist_exscan_i8));
    PyObject_SetAttrString(m, "hpat_dist_exscan_f4",
                            PyLong_FromVoidPtr(&hpat_dist_exscan_f4));
    PyObject_SetAttrString(m, "hpat_dist_exscan_f8",
                            PyLong_FromVoidPtr(&hpat_dist_exscan_f8));

    PyObject_SetAttrString(m, "hpat_dist_arr_reduce",
                            PyLong_FromVoidPtr(&hpat_dist_arr_reduce));
    PyObject_SetAttrString(m, "hpat_dist_irecv",
                            PyLong_FromVoidPtr(&hpat_dist_irecv));
    PyObject_SetAttrString(m, "hpat_dist_isend",
                            PyLong_FromVoidPtr(&hpat_dist_isend));
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


int hpat_dist_reduce_i4(int value)
{
    // printf("sum value: %d\n", value);
    int out=0;
    MPI_Allreduce(&value, &out, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

int64_t hpat_dist_reduce_i8(int64_t value)
{
    // printf("sum value: %lld\n", value);
    int64_t out=0;
    MPI_Allreduce(&value, &out, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

float hpat_dist_reduce_f4(float value)
{
    // printf("sum value: %f\n", value);
    float out=0;
    MPI_Allreduce(&value, &out, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

double hpat_dist_reduce_f8(double value)
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


int hpat_dist_exscan_i4(int value)
{
    // printf("sum value: %d\n", value);
    int out=0;
    MPI_Exscan(&value, &out, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

int64_t hpat_dist_exscan_i8(int64_t value)
{
    // printf("sum value: %lld\n", value);
    int64_t out=0;
    MPI_Exscan(&value, &out, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

float hpat_dist_exscan_f4(float value)
{
    // printf("sum value: %f\n", value);
    float out=0;
    MPI_Exscan(&value, &out, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

double hpat_dist_exscan_f8(double value)
{
    // printf("sum value: %lf\n", value);
    double out=0;
    MPI_Exscan(&value, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

int hpat_dist_irecv(void* out, int size, int type_enum, int pe, int tag, bool cond)
{
    int i;
    MPI_Request mpi_req_recv = -1;
    printf("irecv size:%d pe:%d tag:%d, cond:%d\n", size, pe, tag, cond);
    fflush(stdout);
    if(cond)
    {
        MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
        MPI_Irecv(out, size, mpi_typ, pe, tag, MPI_COMM_WORLD, mpi_req_recv);
    }
    return mpi_req_recv;
}

int hpat_dist_isend(void* out, int size, int type_enum, int pe, int tag, bool cond)
{
    int i;
    MPI_Request mpi_req_recv = -1;
    printf("isend size:%d pe:%d tag:%d, cond:%d\n", size, pe, tag, cond);
    fflush(stdout);
    if(cond)
    {
        MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
        MPI_Isend(out, size, mpi_typ, pe, tag, MPI_COMM_WORLD, mpi_req_recv);
    }
    return mpi_req_recv;
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

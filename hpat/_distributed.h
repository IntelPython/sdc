#ifndef _DISTRIBUTED_H_INCLUDED
#define _DISTRIBUTED_H_INCLUDED

#define ROOT_PE 0

extern "C" {
    
int hpat_dist_get_rank();
int hpat_dist_get_size();
int64_t hpat_dist_get_start(int64_t total, int num_pes, int node_id);
int64_t hpat_dist_get_end(int64_t total, int num_pes, int node_id);
int64_t hpat_dist_get_node_portion(int64_t total, int num_pes, int node_id);
double hpat_dist_get_time();
double hpat_get_time();
int hpat_barrier();
MPI_Datatype get_MPI_typ(int typ_enum);
MPI_Datatype get_val_rank_MPI_typ(int typ_enum);
MPI_Op get_MPI_op(int op_enum);
int get_elem_size(int type_enum);
void hpat_dist_reduce(char *in_ptr, char *out_ptr, int op, int type_enum);

int hpat_dist_exscan_i4(int value);
int64_t hpat_dist_exscan_i8(int64_t value);
float hpat_dist_exscan_f4(float value);
double hpat_dist_exscan_f8(double value);

int hpat_dist_arr_reduce(void* out, int64_t* shapes, int ndims, int op_enum, int type_enum);
MPI_Request hpat_dist_irecv(void* out, int size, int type_enum, int pe, int tag, bool cond);
MPI_Request hpat_dist_isend(void* out, int size, int type_enum, int pe, int tag, bool cond);
int hpat_dist_wait(MPI_Request req, bool cond);
void hpat_dist_waitall(int size, MPI_Request *req);

void c_gather_scalar(void* send_data, void* recv_data, int typ_enum);
void c_gatherv(void* send_data, int sendcount, void* recv_data, int* recv_counts, int* displs, int typ_enum);
void c_bcast(void* send_data, int sendcount, int typ_enum);

void c_alltoallv(void* send_data, void* recv_data, int* send_counts,
                int* recv_counts, int* send_disp, int* recv_disp, int typ_enum);
void c_alltoall(void* send_data, void* recv_data, int count, int typ_enum);
int64_t hpat_dist_get_item_pointer(int64_t ind, int64_t start, int64_t count);
void allgather(void* out_data, int size, void* in_data, int type_enum);
MPI_Request *comm_req_alloc(int size);
void comm_req_dealloc(MPI_Request *req_arr);
void req_array_setitem(MPI_Request * req_arr, int64_t ind, MPI_Request req);

void oneD_reshape_shuffle(char* output,
                          char* input,
                          int64_t new_0dim_global_len,
                          int64_t old_0dim_global_len,
                          int64_t out_lower_dims_size,
                          int64_t in_lower_dims_size);

void permutation_int(int64_t* output, int n);
void permutation_array_index(unsigned char *lhs, int64_t len, int64_t elem_size,
                             unsigned char *rhs, int64_t *p, int64_t p_len);
int hpat_finalize();

} // extern "C"
#endif // _DISTRIBUTED_H_INCLUDED

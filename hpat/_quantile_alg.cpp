#include "mpi.h"
#include <Python.h>
#include <random>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

#define root 0

template <class T>
std::pair<T, T> get_lower_upper_kth_parallel(std::vector<T> &my_array,
        int64_t total_size, int myrank, int n_pes, int64_t k, int type_enum);

template <class T>
double small_get_nth_parallel(std::vector<T> &my_array, int64_t total_size,
                              int myrank, int n_pes, int64_t k, int type_enum);

template <class T>
double get_nth_parallel(std::vector<T> &my_array, int64_t k, int myrank, int n_pes, int type_enum);

double quantile_parallel(void* data, int64_t local_size, int64_t total_size, double quantile, int type_enum);
template<class T>
double quantile_parallel_int(T* data, int64_t local_size, double at, int type_enum, int myrank, int n_pes);
template<class T>
double quantile_parallel_float(T* data, int64_t local_size, double quantile, int type_enum, int myrank, int n_pes);

PyMODINIT_FUNC PyInit_quantile_alg(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "quantile_alg", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "quantile_parallel",
                            PyLong_FromVoidPtr((void*)(&quantile_parallel)));
    return m;
}

double quantile_parallel(void* data, int64_t local_size, int64_t total_size, double quantile, int type_enum)
{
    int myrank, n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (total_size == 0)
        MPI_Allreduce(&local_size, &total_size, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

    // return NA if no elements
    if (total_size == 0)
    {
        return std::nan("");
    }

    double at = quantile * (total_size-1);

    // FIXME: refactor constants
    if (type_enum == 0)
        return quantile_parallel_int((char *)data, local_size, at, type_enum, myrank, n_pes);
    if (type_enum == 1)
        return quantile_parallel_int((unsigned char *) data, local_size, at, type_enum, myrank, n_pes);
    if (type_enum == 2)
        return quantile_parallel_int((int *)data, local_size, at, type_enum, myrank, n_pes);
    if (type_enum == 3)
        return quantile_parallel_int((int64_t *)data, local_size, at, type_enum, myrank, n_pes);
    if (type_enum == 4)
        return quantile_parallel_float((float*)data, local_size, quantile, type_enum, myrank, n_pes);
    if (type_enum == 5)
        return quantile_parallel_float((double*)data, local_size, quantile, type_enum, myrank, n_pes);

    printf("unknown quantile data type");
    return -1.0;
}

template<class T>
double quantile_parallel_int(T* data, int64_t local_size, double at, int type_enum, int myrank, int n_pes)
{
    int64_t k1 = (int64_t)at;
    int64_t k2 = k1+1;
    double fraction = at - (double)k1;
    std::vector<T> my_array(data, data+local_size);
    double res1 = get_nth_parallel(my_array, k1, myrank, n_pes, type_enum);
    double res2 = get_nth_parallel(my_array, k2, myrank, n_pes, type_enum);
    // linear method, TODO: support other methods
    return res1 + (res2 - res1) * fraction;
}

template<class T>
double quantile_parallel_float(T* data, int64_t local_size, double quantile, int type_enum, int myrank, int n_pes)
{
    std::vector<T> my_array(data, data + local_size);
    // delete NaNs
    my_array.erase(std::remove_if(std::begin(my_array), std::end(my_array),
                                  [](T d) {return std::isnan(d);}), my_array.end());
    local_size = my_array.size();
    // recalculate total size since there could be NaNs
    int64_t total_size;
    MPI_Allreduce(&local_size, &total_size, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    double at = quantile * (total_size-1);
    int64_t k1 = (int64_t)at;
    int64_t k2 = k1+1;
    double fraction = at - (double)k1;

    double res1 = get_nth_parallel(my_array, k1, myrank, n_pes, type_enum);
    double res2 = get_nth_parallel(my_array, k2, myrank, n_pes, type_enum);
    // linear method, TODO: support other methods
    return res1 + (res2 - res1) * fraction;
}

// _h5_typ_table = {
//     int8:0,
//     uint8:1,
//     int32:2,
//     int64:3,
//     float32:4,
//     float64:5
//     }

// TODO: refactor to header
MPI_Datatype get_MPI_typ(int typ_enum)
{
    // printf("h5 type enum:%d\n", typ_enum);
    MPI_Datatype types_list[] = {MPI_CHAR, MPI_UNSIGNED_CHAR,
            MPI_INT, MPI_LONG_LONG_INT, MPI_FLOAT, MPI_DOUBLE};
    return types_list[typ_enum];
}

template <class T>
double get_nth_parallel(std::vector<T> &my_array, int64_t k, int myrank, int n_pes, int type_enum)
{
    int64_t local_size = my_array.size();
    int64_t total_size;
    MPI_Allreduce(&local_size, &total_size, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    // printf("total size: %ld k: %ld\n", total_size, k);
    int64_t threshold = (int64_t) pow(10.0, 7.0); // 100 million
    // int64_t threshold = 20;
    if (total_size < threshold || n_pes==1)
    {
        return small_get_nth_parallel(my_array, total_size, myrank, n_pes, k, type_enum);
    }
    else
    {
        std::pair<T, T> kths = get_lower_upper_kth_parallel(my_array, total_size, myrank, n_pes, k, type_enum);
        T k1_val = kths.first;
        T k2_val = kths.second;
        // printf("k1_val: %lf  k2_val: %lf\n", k1_val, k2_val);
        int64_t local_l0_num = 0, local_l1_num = 0, local_l2_num = 0;
        int64_t l0_num = 0, l1_num = 0, l2_num = 0;
        for(auto val: my_array)
        {
            if (val<k1_val)
                local_l0_num++;
            if (val>=k1_val && val<k2_val)
                local_l1_num++;
            if (val>=k2_val)
                local_l2_num++;
        }
        MPI_Allreduce(&local_l0_num, &l0_num, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_l1_num, &l1_num, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_l2_num, &l2_num, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
        // printf("set sizes: %ld %ld %ld\n", l0_num, l1_num, l2_num);
        assert(l0_num + l1_num + l2_num == total_size);
        // []----*---o----*-----]
        // if there are more elements in the last set than elemenet k to end,
        // this means k2 is equal to k
        if (l2_num > total_size-k)
            return k2_val;
        assert(l0_num < k);

        std::vector<T> new_my_array;
        int64_t new_k = k;

        int64_t new_ind = 0;
        if (k < l0_num)
        {
            // first set
            // printf("first set\n");
            new_my_array.resize(local_l0_num);
            // throw away
            for(auto val: my_array)
            {
                if (val<k1_val)
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
            for(auto val: my_array)
            {
                if (val>=k1_val && val<k2_val)
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
            for(auto val: my_array)
            {
                if (val>=k2_val)
                {
                    new_my_array[new_ind] = val;
                    new_ind++;
                }
            }
            new_k -= (l0_num + l1_num);
        }
        return get_nth_parallel(new_my_array, new_k, myrank, n_pes, type_enum);
    }
    return -1.0;
}

template <class T>
std::pair<T, T> get_lower_upper_kth_parallel(std::vector<T> &my_array,
        int64_t total_size, int myrank, int n_pes, int64_t k, int type_enum)
{
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    int64_t local_size = my_array.size();
    std::default_random_engine r_engine(myrank);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    int64_t sample_size = (int64_t) (pow(10.0, 5.0)/n_pes); // 100000 total
    int64_t my_sample_size = std::min(sample_size, local_size);

    std::vector<T> my_sample;
    for(int64_t i=0; i<my_sample_size; i++)
    {
        int64_t index = (int64_t) (local_size*uniform_dist(r_engine));
        my_sample.push_back(my_array[index]);
    }
    /* select sample */
    // get total sample size;
    std::vector<T> all_sample_vec;
    int *rcounts = new int[n_pes];
    int *displs = new int[n_pes];
    int total_sample_size = 0;
    // gather the sample sizes
    MPI_Gather(&my_sample_size, 1, MPI_INT, rcounts, 1, MPI_INT, root, MPI_COMM_WORLD);
    // calculate size and displacements on root
    if (myrank == root)
    {
        for(int i=0; i<n_pes; i++)
        {
            // printf("rc %d\n", rcounts[i]);
            displs[i] = total_sample_size;
            total_sample_size += rcounts[i];
        }
        // printf("total sample size: %d\n", total_sample_size);
        all_sample_vec.resize(total_sample_size);
    }
    // gather sample data
    MPI_Gatherv(my_sample.data(), my_sample_size, mpi_typ, all_sample_vec.data(), rcounts, displs, mpi_typ, root, MPI_COMM_WORLD);
    T k1_val;
    T k2_val;
    if (myrank == root)
    {
        int local_k = (int) (k*(total_sample_size/(T)total_size));
        // printf("k:%ld local_k:%d\n", k, local_k);
        int k1 = (int) (local_k - sqrt(total_sample_size * log(total_size)));
        int k2 = (int) (local_k + sqrt(total_sample_size * log(total_size)));
        k1 = std::max(k1, 0);
        k2 = std::min(k2, total_sample_size-1);
        // printf("k1: %d k2: %d\n", k1, k2);
        std::nth_element(all_sample_vec.begin(), all_sample_vec.begin()+k1, all_sample_vec.end());
        k1_val = all_sample_vec[k1];
        std::nth_element(all_sample_vec.begin(), all_sample_vec.begin()+k2, all_sample_vec.end());
        k2_val = all_sample_vec[k2];
        // printf("k1: %d k2: %d k1_val: %lf k2_val:%lf\n", k1, k2, k1_val, k2_val);
    }
    MPI_Bcast(&k1_val, 1, mpi_typ, root, MPI_COMM_WORLD);
    MPI_Bcast(&k2_val, 1, mpi_typ, root, MPI_COMM_WORLD);
    // cleanup
    delete[] rcounts;
    delete[] displs;
    return std::make_pair(k1_val, k2_val);
}

template <class T>
double small_get_nth_parallel(std::vector<T> &my_array, int64_t total_size,
                              int myrank, int n_pes, int64_t k, int type_enum)
{
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    T res;
    int my_data_size = my_array.size();
    int total_data_size = 0;
    std::vector<T> all_data_vec;

    // no need to gather data if only 1 processor
    if (n_pes==1)
    {
        std::nth_element(my_array.begin(), my_array.begin() + k, my_array.end());
        res = my_array[k];
        return res;
    }

    // gather the data sizes
    int *rcounts = new int[n_pes];
    int *displs = new int[n_pes];
    MPI_Gather(&my_data_size, 1, MPI_INT, rcounts, 1, MPI_INT, root, MPI_COMM_WORLD);
    // calculate size and displacements on root
    if (myrank == root)
    {
        for(int i=0; i<n_pes; i++)
        {
            // printf("rc %d\n", rcounts[i]);
            displs[i] = total_data_size;
            total_data_size += rcounts[i];
        }
        // printf("total small data size: %d\n", total_data_size);
        all_data_vec.resize(total_data_size);
    }
    // gather data
    MPI_Gatherv(my_array.data(), my_data_size, mpi_typ, all_data_vec.data(), rcounts, displs, mpi_typ, root, MPI_COMM_WORLD);
    // get nth element on root
    if (myrank == root)
    {
        std::nth_element(all_data_vec.begin(), all_data_vec.begin() + k, all_data_vec.end());
        res = all_data_vec[k];
    }
    MPI_Bcast(&res, 1, mpi_typ, root, MPI_COMM_WORLD);
    return res;
}
/*
    // T ep = log(sample_size)/ log(total_size);
    /for (size_t i = 0; i < local_size; i++) {
        if (uniform_dist(r_engine) <= select_probablity)
            my_sample.push_back(my_array[i]);
    }
    int64_t my_sample_size = my_sample.size();

/ T select_probablity =  pow(local_size, -ep); // N^-e
// MPI_Allreduce(&my_sample_size, &total_sample_size, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
*/

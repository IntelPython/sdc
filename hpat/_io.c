#include "mpi.h"
#include "hdf5.h"
#include <Python.h>
#include <string>

int hpat_h5_open(char* file_name, char* mode, int64_t is_parallel);
int64_t hpat_h5_size(hid_t file_id, char* dset_name, int dim);
int hpat_h5_read(hid_t file_id, char* dset_name, int ndims, int64_t* starts,
    int64_t* counts, int64_t is_parallel, void* out, int typ_enum);
int hpat_h5_close(hid_t file_id);
int hpat_h5_create_dset(hid_t file_id, char* dset_name, int ndims,
    int64_t* counts, int typ_enum);
int hpat_h5_create_group(hid_t file_id, char* group_name);
int hpat_h5_write(hid_t file_id, hid_t dataset_id, int ndims, int64_t* starts,
    int64_t* counts, int64_t is_parallel, void* out, int typ_enum);
int hpat_h5_get_type_enum(std::string *s);
hid_t get_h5_typ(int typ_enum);

PyMODINIT_FUNC PyInit_hio(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "hio", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "hpat_h5_open",
                            PyLong_FromVoidPtr((void*)(&hpat_h5_open)));
    PyObject_SetAttrString(m, "hpat_h5_size",
                            PyLong_FromVoidPtr((void*)(&hpat_h5_size)));
    PyObject_SetAttrString(m, "hpat_h5_read",
                            PyLong_FromVoidPtr((void*)(&hpat_h5_read)));
    PyObject_SetAttrString(m, "hpat_h5_close",
                            PyLong_FromVoidPtr((void*)(&hpat_h5_close)));
    PyObject_SetAttrString(m, "hpat_h5_create_dset",
                            PyLong_FromVoidPtr((void*)(&hpat_h5_create_dset)));
    PyObject_SetAttrString(m, "hpat_h5_create_group",
                            PyLong_FromVoidPtr((void*)(&hpat_h5_create_group)));
    PyObject_SetAttrString(m, "hpat_h5_write",
                            PyLong_FromVoidPtr((void*)(&hpat_h5_write)));
    PyObject_SetAttrString(m, "hpat_h5_get_type_enum",
                            PyLong_FromVoidPtr((void*)(&hpat_h5_get_type_enum)));
    return m;
}

int hpat_h5_open(char* file_name, char* mode, int64_t is_parallel)
{
    // printf("h5_open file_name: %s mode:%s\n", file_name, mode);
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    assert(plist_id != -1);
    herr_t ret;
    hid_t file_id = -1;
    unsigned flag = H5F_ACC_RDWR;

    if(is_parallel)
        ret = H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    assert(ret != -1);

    // TODO: handle 'a' mode
    if(strcmp(mode, "r")==0)
    {
        flag = H5F_ACC_RDONLY;
        file_id = H5Fopen((const char*)file_name, flag, plist_id);
    }
    else if(strcmp(mode, "r+")==0)
    {
        flag = H5F_ACC_RDWR;
        file_id = H5Fopen((const char*)file_name, flag, plist_id);
    }
    else if(strcmp(mode, "w")==0)
    {
        flag = H5F_ACC_TRUNC;
        file_id = H5Fcreate((const char*)file_name, flag, H5P_DEFAULT, plist_id);
    }
    else if(strcmp(mode, "w-")==0 || strcmp(mode, "x")==0)
    {
        flag = H5F_ACC_EXCL;
        file_id = H5Fcreate((const char*)file_name, flag, H5P_DEFAULT, plist_id);
        // printf("w- fid:%d\n", file_id);
    }

    assert(file_id != -1);
    ret = H5Pclose(plist_id);
    assert(ret != -1);
    return file_id;
}

int64_t hpat_h5_size(hid_t file_id, char* dset_name, int dim)
{
    hid_t dataset_id;
    dataset_id = H5Dopen2(file_id, dset_name, H5P_DEFAULT);
    assert(dataset_id != -1);
    hid_t space_id = H5Dget_space(dataset_id);
    assert(space_id != -1);
    hsize_t data_ndim = H5Sget_simple_extent_ndims(space_id);
    hsize_t space_dims[data_ndim];
    H5Sget_simple_extent_dims(space_id, space_dims, NULL);
    H5Dclose(dataset_id);
    return space_dims[dim];
}

int hpat_h5_read(hid_t file_id, char* dset_name, int ndims, int64_t* starts,
    int64_t* counts, int64_t is_parallel, void* out, int typ_enum)
{
    //printf("dset_name:%s ndims:%d size:%d typ:%d\n", dset_name, ndims, counts[0], typ_enum);
    // fflush(stdout);
    // printf("start %lld end %lld\n", start_ind, end_ind);
    hid_t dataset_id;
    herr_t ret;
    dataset_id = H5Dopen2(file_id, dset_name, H5P_DEFAULT);
    assert(dataset_id != -1);
    hid_t space_id = H5Dget_space(dataset_id);
    assert(space_id != -1);

    hsize_t* HDF5_start = (hsize_t*)starts;
    hsize_t* HDF5_count = (hsize_t*)counts;

    hid_t xfer_plist_id = H5P_DEFAULT;
    if(is_parallel)
    {
        xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);
    }

    ret = H5Sselect_hyperslab(space_id, H5S_SELECT_SET, HDF5_start, NULL, HDF5_count, NULL);
    assert(ret != -1);
    hid_t mem_dataspace = H5Screate_simple((hsize_t)ndims, HDF5_count, NULL);
    assert (mem_dataspace != -1);
    hid_t h5_typ = get_h5_typ(typ_enum);
    ret = H5Dread(dataset_id, h5_typ, mem_dataspace, space_id, xfer_plist_id, out);
    assert(ret != -1);
    // printf("out: %lf %lf ...\n", ((double*)out)[0], ((double*)out)[1]);
    H5Dclose(dataset_id);
    return ret;
}


// _h5_typ_table = {
//     int8:0,
//     uint8:1,
//     int32:2,
//     int64:3,
//     float32:4,
//     float64:5
//     }

hid_t get_h5_typ(int typ_enum)
{
    // printf("h5 type enum:%d\n", typ_enum);
    hid_t types_list[] = {H5T_NATIVE_CHAR, H5T_NATIVE_UCHAR,
            H5T_NATIVE_INT, H5T_NATIVE_LLONG, H5T_NATIVE_FLOAT, H5T_NATIVE_DOUBLE};
    return types_list[typ_enum];
}

//  _h5_str_typ_table = {
//      'i1':0,
//      'u1':1,
//      'i4':2,
//      'i8':3,
//      'f4':4,
//      'f8':5
//      }

int hpat_h5_get_type_enum(std::string *s)
{
    int typ = -1;
    if ((*s)=="i1")
        typ = 0;
    if ((*s)=="u1")
        typ = 1;
    if ((*s)=="i4")
        typ = 2;
    if ((*s)=="i8")
        typ = 3;
    if ((*s)=="f4")
        typ = 4;
    if ((*s)=="f8")
        typ = 5;

    return typ;
}

int hpat_h5_close(hid_t file_id)
{
    // printf("closing: %d\n", file_id);
    H5Fclose(file_id);
    return 0;
}

int hpat_h5_create_dset(hid_t file_id, char* dset_name, int ndims,
    int64_t* counts, int typ_enum)
{
    // printf("dset_name:%s ndims:%d size:%d typ:%d\n", dset_name, ndims, counts[0], typ_enum);
    // fflush(stdout);

    hid_t dataset_id;
    hid_t  filespace;
    hid_t h5_typ = get_h5_typ(typ_enum);
    filespace = H5Screate_simple(ndims, (const hsize_t *)counts, NULL);
    dataset_id = H5Dcreate(file_id, dset_name, h5_typ, filespace,
                                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);
    return dataset_id;
}

int hpat_h5_create_group(hid_t file_id, char* group_name)
{
    // printf("group_name:%s\n", group_name);
    // fflush(stdout);

    hid_t group_id;
    group_id = H5Gcreate2(file_id, group_name,
                                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    return group_id;
}

int hpat_h5_write(hid_t file_id, hid_t dataset_id, int ndims, int64_t* starts,
    int64_t* counts, int64_t is_parallel, void* out, int typ_enum)
{
    //printf("dset_id:%s ndims:%d size:%d typ:%d\n", dset_id, ndims, counts[0], typ_enum);
    // fflush(stdout);
    herr_t ret;
    assert(dataset_id != -1);
    hid_t space_id = H5Dget_space(dataset_id);
    assert(space_id != -1);

    hsize_t* HDF5_start = (hsize_t*)starts;
    hsize_t* HDF5_count = (hsize_t*)counts;

    hid_t xfer_plist_id = H5P_DEFAULT;
    if(is_parallel)
    {
        xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);
    }

    ret = H5Sselect_hyperslab(space_id, H5S_SELECT_SET, HDF5_start, NULL, HDF5_count, NULL);
    assert(ret != -1);
    hid_t mem_dataspace = H5Screate_simple((hsize_t)ndims, HDF5_count, NULL);
    assert (mem_dataspace != -1);
    hid_t h5_typ = get_h5_typ(typ_enum);
    ret = H5Dwrite(dataset_id, h5_typ, mem_dataspace, space_id, xfer_plist_id, out);
    assert(ret != -1);
    H5Dclose(dataset_id);
    return ret;
}

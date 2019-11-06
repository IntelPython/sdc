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
#include <climits>
#include <cstdio>
#include <iostream>
#include <string>
#include "hdf5.h"

hid_t hpat_h5_open(char* file_name, char* mode, int64_t is_parallel);
hid_t hpat_h5_open_dset_or_group_obj(hid_t file_id, char* obj_name);
int64_t hpat_h5_size(hid_t dataset_id, int dim);
int hpat_h5_read(
    hid_t dataset_id, int ndims, int64_t* starts, int64_t* counts, int64_t is_parallel, void* out, int typ_enum);
int hpat_h5_read_filter(hid_t dataset_id,
                        int ndims,
                        int64_t* starts,
                        int64_t* counts,
                        int64_t is_parallel,
                        void* out,
                        int typ_enum,
                        int64_t* indices,
                        int n_indices);
int hpat_h5_close(hid_t file_id);
hid_t hpat_h5_create_dset(hid_t file_id, char* dset_name, int ndims, int64_t* counts, int typ_enum);
hid_t hpat_h5_create_group(hid_t file_id, char* group_name);
int hpat_h5_write(
    hid_t dataset_id, int ndims, int64_t* starts, int64_t* counts, int64_t is_parallel, void* out, int typ_enum);
int hpat_h5_get_type_enum(char* s);
hid_t get_h5_typ(int typ_enum);
int64_t h5g_get_num_objs(hid_t file_id);
void* h5g_get_objname_by_idx(hid_t file_id, int64_t ind);
void hpat_h5g_close(hid_t group_id);

PyMODINIT_FUNC PyInit__hdf5(void)
{
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_hdf5",
        "No docs",
        -1,
        NULL,
    };

    m = PyModule_Create(&moduledef);
    if (m == NULL)
    {
        return NULL;
    }

    PyObject_SetAttrString(m, "hpat_h5_open", PyLong_FromVoidPtr((void*)(&hpat_h5_open)));
    PyObject_SetAttrString(
        m, "hpat_h5_open_dset_or_group_obj", PyLong_FromVoidPtr((void*)(&hpat_h5_open_dset_or_group_obj)));
    PyObject_SetAttrString(m, "hpat_h5_size", PyLong_FromVoidPtr((void*)(&hpat_h5_size)));
    PyObject_SetAttrString(m, "hpat_h5_read", PyLong_FromVoidPtr((void*)(&hpat_h5_read)));
    PyObject_SetAttrString(m, "hpat_h5_read_filter", PyLong_FromVoidPtr((void*)(&hpat_h5_read_filter)));
    PyObject_SetAttrString(m, "hpat_h5_close", PyLong_FromVoidPtr((void*)(&hpat_h5_close)));
    PyObject_SetAttrString(m, "hpat_h5_create_dset", PyLong_FromVoidPtr((void*)(&hpat_h5_create_dset)));
    PyObject_SetAttrString(m, "hpat_h5_create_group", PyLong_FromVoidPtr((void*)(&hpat_h5_create_group)));
    PyObject_SetAttrString(m, "hpat_h5_write", PyLong_FromVoidPtr((void*)(&hpat_h5_write)));
    PyObject_SetAttrString(m, "hpat_h5_get_type_enum", PyLong_FromVoidPtr((void*)(&hpat_h5_get_type_enum)));
    PyObject_SetAttrString(m, "h5g_get_num_objs", PyLong_FromVoidPtr((void*)(&h5g_get_num_objs)));
    PyObject_SetAttrString(m, "h5g_get_objname_by_idx", PyLong_FromVoidPtr((void*)(&h5g_get_objname_by_idx)));
    PyObject_SetAttrString(m, "hpat_h5g_close", PyLong_FromVoidPtr((void*)(&hpat_h5g_close)));

    return m;
}

// TODO: raise Python error
#define CHECK(expr, msg)                                                                                               \
    if (!(expr))                                                                                                       \
    {                                                                                                                  \
        std::cerr << msg << std::endl;                                                                                 \
        H5Eprint(H5E_DEFAULT, NULL);                                                                                   \
    }

hid_t hpat_h5_open(char* file_name, char* mode, int64_t is_parallel)
{
    // printf("h5_open file_name: %s mode:%s\n", file_name, mode);
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    CHECK(plist_id != -1, "h5 open property create error");
    herr_t ret = 0;
    hid_t file_id = -1;
    unsigned flag = H5F_ACC_RDWR;

    //int num_pes;
    //MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
    // TODO: enable MPIO after fixing address overflow issues
    // if(false && is_parallel && num_pes>1)
    // {
    //     ret = H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    //     CHECK(ret != -1, "h5 open MPI driver set error");
    // }

    // TODO: handle 'a' mode
    if (strcmp(mode, "r") == 0)
    {
        flag = H5F_ACC_RDONLY;
        file_id = H5Fopen((const char*)file_name, flag, plist_id);
    }
    else if (strcmp(mode, "r+") == 0)
    {
        flag = H5F_ACC_RDWR;
        file_id = H5Fopen((const char*)file_name, flag, plist_id);
    }
    else if (strcmp(mode, "w") == 0)
    {
        flag = H5F_ACC_TRUNC;
        file_id = H5Fcreate((const char*)file_name, flag, H5P_DEFAULT, plist_id);
    }
    else if (strcmp(mode, "w-") == 0 || strcmp(mode, "x") == 0)
    {
        flag = H5F_ACC_EXCL;
        file_id = H5Fcreate((const char*)file_name, flag, H5P_DEFAULT, plist_id);
        // printf("w- fid:%d\n", file_id);
    }

    CHECK(file_id != -1, "h5 open file error");
    ret = H5Pclose(plist_id);
    CHECK(ret != -1, "h5 open property close error");
    return file_id;
}

hid_t hpat_h5_open_dset_or_group_obj(hid_t file_or_group_id, char* obj_name)
{
    // handle obj['A'] call, the output can be group or dataset
    // printf("open dset or group: %s\n", obj_name);
    hid_t obj_id = -1;
    H5O_info_t object_info;
    herr_t err = H5Oget_info_by_name(file_or_group_id, obj_name, &object_info, H5P_DEFAULT);
    CHECK(err != -1, "h5 open dset or group get_info_by_name error");
    if (object_info.type == H5O_TYPE_GROUP)
    {
        // printf("open group: %s\n", obj_name);
        obj_id = H5Gopen2(file_or_group_id, obj_name, H5P_DEFAULT);
    }
    if (object_info.type == H5O_TYPE_DATASET)
    {
        // printf("open dset: %s\n", obj_name);
        obj_id = H5Dopen2(file_or_group_id, obj_name, H5P_DEFAULT);
    }
    CHECK(obj_id != -1, "h5 open dset or group error");
    return obj_id;
}

int64_t hpat_h5_size(hid_t dataset_id, int dim)
{
    CHECK(dataset_id != -1, "h5 invalid dataset_id input to size call");
    hid_t space_id = H5Dget_space(dataset_id);
    CHECK(space_id != -1, "h5 size get_space error");
    hsize_t data_ndim = H5Sget_simple_extent_ndims(space_id);
    hsize_t* space_dims = new hsize_t[data_ndim];
    H5Sget_simple_extent_dims(space_id, space_dims, NULL);
    H5Sclose(space_id);
    hsize_t ret = space_dims[dim];
    delete[] space_dims;
    return ret;
}

hid_t get_dset_space_from_range(hid_t dataset_id, int64_t* starts, int64_t* counts)
{
    hid_t space_id = H5Dget_space(dataset_id);
    CHECK(space_id != -1, "h5 read get_space error");

    hsize_t* HDF5_start = (hsize_t*)starts;
    hsize_t* HDF5_count = (hsize_t*)counts;
    herr_t ret = H5Sselect_hyperslab(space_id, H5S_SELECT_SET, HDF5_start, NULL, HDF5_count, NULL);
    CHECK(ret != -1, "h5 read select_hyperslab error");
    return space_id;
}

int hpat_h5_read(
    hid_t dataset_id, int ndims, int64_t* starts, int64_t* counts, int64_t is_parallel, void* out, int typ_enum)
{
    // printf("h5read ndims:%d size:%d typ:%d\n", ndims, counts[0], typ_enum);
    // fflush(stdout);
    // printf("start %lld end %lld\n", start_ind, end_ind);
    herr_t ret;
    CHECK(dataset_id != -1, "h5 read invalid dataset_id");

    hid_t space_id = get_dset_space_from_range(dataset_id, starts, counts);

    //int num_pes;
    //MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
    hid_t xfer_plist_id = H5P_DEFAULT;
    // TODO: enable MPIO after fixing address overflow issues
    // if(false && is_parallel && num_pes>1)
    // {
    //     xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    //     H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);
    // }

    hid_t mem_dataspace = H5Screate_simple((hsize_t)ndims, (hsize_t*)counts, NULL);
    CHECK(mem_dataspace != -1, "h5 read create_simple error");
    hid_t h5_typ = get_h5_typ(typ_enum);
    ret = H5Dread(dataset_id, h5_typ, mem_dataspace, space_id, xfer_plist_id, out);
    CHECK(ret != -1, "h5 read call error");
    // printf("out: %lf %lf ...\n", ((double*)out)[0], ((double*)out)[1]);
    // TODO: close here?
    H5Dclose(dataset_id);
    return ret;
}

hid_t get_dset_space_from_indices(
    hid_t dataset_id, int ndims, int64_t* starts, int64_t* counts, int64_t* indices, int n_indices)
{
    // printf("num ind: %d\n", n_indices);
    hid_t space_id = H5Dget_space(dataset_id);
    CHECK(space_id != -1, "h5 read get_space error");

    hsize_t* HDF5_start = new hsize_t[ndims];
    hsize_t* HDF5_count = new hsize_t[ndims];
    hsize_t* HDF5_block = new hsize_t[ndims];
    for (int i = 1; i < ndims; i++)
    {
        HDF5_start[i] = 0;
        HDF5_count[i] = 1;
        HDF5_block[i] = counts[i];
    }
    // check for empty index list
    if (n_indices <= 0)
    {
        HDF5_start[0] = 0;
        HDF5_count[0] = 0;
        HDF5_block[0] = 0;
        H5Sselect_hyperslab(space_id, H5S_SELECT_SET, HDF5_start, NULL, HDF5_count, HDF5_block);
        return space_id;
    }
    // printf("ind %d\n", indices[0]);
    HDF5_start[0] = indices[0];
    HDF5_count[0] = 1;
    HDF5_block[0] = 1;
    H5Sselect_hyperslab(space_id, H5S_SELECT_SET, HDF5_start, NULL, HDF5_count, HDF5_block);
    for (int i = 1; i < n_indices; i++)
    {
        HDF5_start[0] = indices[i];
        // printf("ind %d\n", indices[i]);
        herr_t ret = H5Sselect_hyperslab(space_id, H5S_SELECT_OR, HDF5_start, NULL, HDF5_count, HDF5_block);
        CHECK(ret != -1, "h5 read select_hyperslab error");
    }

    delete[] HDF5_start;
    delete[] HDF5_count;
    delete[] HDF5_block;
    return space_id;
}

int hpat_h5_read_filter(hid_t dataset_id,
                        int ndims,
                        int64_t* starts,
                        int64_t* counts,
                        int64_t is_parallel,
                        void* out,
                        int typ_enum,
                        int64_t* indices,
                        int n_indices)
{
    //
    // printf("ndim %d starts %d %d %d %d\n", ndims, starts[0], starts[1], starts[2], starts[3]);
    // printf("counts %d %d %d %d\n", counts[0], counts[1], counts[2], counts[3]);
    herr_t ret;
    CHECK(dataset_id != -1, "h5 read invalid dataset_id");

    hid_t space_id = get_dset_space_from_indices(dataset_id, ndims, starts, counts, indices, n_indices);

    //int num_pes;
    //MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
    hid_t xfer_plist_id = H5P_DEFAULT;
    // TODO: enable MPIO after fixing address overflow issues
    // if(false && is_parallel && num_pes>1)
    // {
    //     xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    //     H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);
    // }

    hid_t mem_dataspace = H5Screate_simple((hsize_t)ndims, (hsize_t*)counts, NULL);
    CHECK(mem_dataspace != -1, "h5 read create_simple error");
    hid_t h5_typ = get_h5_typ(typ_enum);
    ret = H5Dread(dataset_id, h5_typ, mem_dataspace, space_id, xfer_plist_id, out);
    CHECK(ret != -1, "h5 read call error");
    // printf("out: %lf %lf ...\n", ((double*)out)[0], ((double*)out)[1]);
    // TODO: close here?
    H5Dclose(dataset_id);
    return ret;
}

// _numba_to_c_type_map = {
//     int8:0,
//     uint8:1,
//     int32:2,
//     uint32:3,
//     int64:4,
//     float32:5,
//     float64:6
//     }

hid_t get_h5_typ(int typ_enum)
{
    // printf("h5 type enum:%d\n", typ_enum);
    hid_t types_list[] = {H5T_NATIVE_CHAR,
                          H5T_NATIVE_UCHAR,
                          H5T_NATIVE_INT,
                          H5T_NATIVE_UINT,
                          H5T_NATIVE_LLONG,
                          H5T_NATIVE_FLOAT,
                          H5T_NATIVE_DOUBLE};
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

// TODO: remove this
int hpat_h5_get_type_enum(char* s)
{
    int typ = -1;
    if (strcmp(s, "i1") == 0)
    {
        typ = 0;
    }
    if (strcmp(s, "u1") == 0)
    {
        typ = 1;
    }
    if (strcmp(s, "i4") == 0)
    {
        typ = 2;
    }
    if (strcmp(s, "u4") == 0)
    {
        typ = 3;
    }
    if (strcmp(s, "i8") == 0)
    {
        typ = 4;
    }
    if (strcmp(s, "f4") == 0)
    {
        typ = 5;
    }
    if (strcmp(s, "f8") == 0)
    {
        typ = 6;
    }

    return typ;
}

void h5_close_object(hid_t obj_id)
{
    H5O_info_t object_info;
    H5Oget_info(obj_id, &object_info);
    if (object_info.type == H5O_TYPE_GROUP)
    {
        // printf("close group %lld\n", obj_id);
        H5Gclose(obj_id);
    }
    if (object_info.type == H5F_OBJ_DATASET)
    {
        // printf("close dset %lld\n", obj_id);
        H5Dclose(obj_id);
    }
}

void h5_close_file_objects(hid_t file_id, unsigned types)
{
    // get object id list
    size_t count = H5Fget_obj_count(file_id, types);
    hid_t* obj_list = (hid_t*)malloc(sizeof(hid_t) * count);
    if (obj_list == NULL)
    {
        return;
    }
    H5Fget_obj_ids(file_id, types, count, obj_list);
    // TODO: check file_id of objects like h5py/files.py:close
    // for(size_t i=0; i<count; i++)
    //     if (H5Iget_file_id(obj_list[i])!=file_id)
    //         obj_list[i] = -1;
    // close objs
    for (size_t i = 0; i < count; i++)
    {
        hid_t obj_id = obj_list[i];
        //if (H5Iget_file_id(obj_id)==file_id)
        if (obj_id != -1)
        {
            h5_close_object(obj_id);
        }
    }
    free(obj_list);
}

int hpat_h5_close(hid_t file_id)
{
    // printf("closing: %d\n", file_id);
    // close file objects similar to h5py/files.py:close
    h5_close_file_objects(file_id, ~H5F_OBJ_FILE);
    h5_close_file_objects(file_id, H5F_OBJ_FILE);
    H5Fclose(file_id);
    return 0;
}

hid_t hpat_h5_create_dset(hid_t file_id, char* dset_name, int ndims, int64_t* counts, int typ_enum)
{
    // printf("dset_name:%s ndims:%d size:%d typ:%d\n", dset_name, ndims, counts[0], typ_enum);
    // fflush(stdout);

    hid_t dataset_id;
    hid_t filespace;
    hid_t h5_typ = get_h5_typ(typ_enum);
    filespace = H5Screate_simple(ndims, (const hsize_t*)counts, NULL);
    dataset_id = H5Dcreate(file_id, dset_name, h5_typ, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);
    return dataset_id;
}

hid_t hpat_h5_create_group(hid_t file_id, char* group_name)
{
    // printf("group_name:%s\n", group_name);
    // fflush(stdout);
    CHECK(file_id != -1, "h5 create_group invalid file_id");
    hid_t group_id;
    group_id = H5Gcreate2(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK(group_id != -1, "h5 create_group error");
    return group_id;
}

int hpat_h5_write(
    hid_t dataset_id, int ndims, int64_t* starts, int64_t* counts, int64_t is_parallel, void* out, int typ_enum)
{
    //printf("dset_id:%s ndims:%d size:%d typ:%d\n", dset_id, ndims, counts[0], typ_enum);
    // fflush(stdout);
    herr_t ret;
    CHECK(dataset_id != -1, "h5 write invalid dataset_id");
    hid_t space_id = H5Dget_space(dataset_id);
    CHECK(space_id != -1, "h5 write get_space error");

    hsize_t* HDF5_start = (hsize_t*)starts;
    hsize_t* HDF5_count = (hsize_t*)counts;

    //int num_pes;
    //MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
    hid_t xfer_plist_id = H5P_DEFAULT;
    // TODO: enable MPIO after fixing address overflow issues
    // if(false && is_parallel && num_pes>1)
    // {
    //     xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    //     H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);
    // }

    ret = H5Sselect_hyperslab(space_id, H5S_SELECT_SET, HDF5_start, NULL, HDF5_count, NULL);
    CHECK(ret != -1, "h5 write select_hyperslab error");
    hid_t mem_dataspace = H5Screate_simple((hsize_t)ndims, HDF5_count, NULL);
    CHECK(mem_dataspace != -1, "h5 write create_simple error");
    hid_t h5_typ = get_h5_typ(typ_enum);
    ret = H5Dwrite(dataset_id, h5_typ, mem_dataspace, space_id, xfer_plist_id, out);
    CHECK(ret != -1, "h5 write call error");
    // XXX fix close properly, refcount dset_id?
    H5Dclose(dataset_id);
    return ret;
}

int64_t h5g_get_num_objs(hid_t file_id)
{
    H5G_info_t group_info;
    herr_t err;
    err = H5Gget_info(file_id, &group_info);
    // printf("num links:%lld\n", group_info.nlinks);
    return (int64_t)group_info.nlinks;
}

void* h5g_get_objname_by_idx(hid_t file_id, int64_t ind)
{
    herr_t err;
    // first call gets size:
    // https://support.hdfgroup.org/HDF5/doc1.8/RM/RM_H5L.html#Link-GetNameByIdx
    int size = H5Lget_name_by_idx(file_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, (hsize_t)ind, NULL, 0, H5P_DEFAULT);
    char* name = (char*)malloc(size + 1);
    if (name == NULL)
    {
        return NULL;
    }
    err = H5Lget_name_by_idx(file_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, (hsize_t)ind, name, size + 1, H5P_DEFAULT);
    // printf("g name:%s\n", name);
    std::string* outstr = new std::string(name);
    free(name);
    // std::cout<<"out: "<<*outstr<<std::endl;
    return outstr;
}

void hpat_h5g_close(hid_t group_id)
{
    herr_t err = H5Gclose(group_id);
}

#undef CHECK

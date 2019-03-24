#include "mpi.h"
#include "_csv.h"
#include <Python.h>
#include <string>
#include <iostream>
#include <cstdio>
#include <climits>
#include <boost/filesystem.hpp>

extern "C" {

uint64_t get_file_size(char* file_name);
void file_read(char* file_name, void* buff, int64_t size);
void file_write(char* file_name, void* buff, int64_t size);
void file_read_parallel(char* file_name, char* buff, int64_t start, int64_t count);
void file_write_parallel(char* file_name, char* buff, int64_t start, int64_t count, int64_t elem_size);

#define ROOT 0
#define LARGE_DTYPE_SIZE 1024

PyMODINIT_FUNC PyInit_hio(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "hio", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    // numpy read
    PyObject_SetAttrString(m, "get_file_size",
                            PyLong_FromVoidPtr((void*)(&get_file_size)));
    PyObject_SetAttrString(m, "file_read",
                            PyLong_FromVoidPtr((void*)(&file_read)));
    PyObject_SetAttrString(m, "file_write",
                            PyLong_FromVoidPtr((void*)(&file_write)));
    PyObject_SetAttrString(m, "file_read_parallel",
                            PyLong_FromVoidPtr((void*)(&file_read_parallel)));
    PyObject_SetAttrString(m, "file_write_parallel",
                            PyLong_FromVoidPtr((void*)(&file_write_parallel)));

    PyInit_csv(m);

    return m;
}

uint64_t get_file_size(char* file_name)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    uint64_t f_size = 0;

    if (rank==ROOT)
    {
        boost::filesystem::path f_path(file_name);
        // TODO: throw FileNotFoundError
        if (!boost::filesystem::exists(f_path))
        {
            std::cerr << "No such file or directory: " << file_name << '\n';
            return 0;
        }
        f_size = (uint64_t)boost::filesystem::file_size(f_path);
    }
    MPI_Bcast(&f_size, 1, MPI_UNSIGNED_LONG_LONG, ROOT, MPI_COMM_WORLD);
    return f_size;
}

void file_read(char* file_name, void* buff, int64_t size)
{
    FILE* fp = fopen(file_name, "rb");
    if (fp == NULL) return;
    size_t ret_code = fread(buff, 1, (size_t)size, fp);
    if (ret_code != (size_t)size)
    {
        std::cerr << "File read error: " << file_name << '\n';
    }
    fclose(fp);
    return;
}

void file_write(char* file_name, void* buff, int64_t size)
{
    FILE* fp = fopen(file_name, "wb");
    if (fp == NULL) return;
    size_t ret_code = fwrite(buff, 1, (size_t)size, fp);
    if (ret_code != (size_t)size)
    {
        std::cerr << "File write error: " << file_name << '\n';
    }
    fclose(fp);
    return;
}

void file_read_parallel(char* file_name, char* buff, int64_t start, int64_t count)
{
    // printf("MPI READ %lld %lld\n", start, count);
    char err_string[MPI_MAX_ERROR_STRING];
    err_string[MPI_MAX_ERROR_STRING-1] = '\0';
    int err_len, err_class;
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    MPI_File fh;
    int ierr = MPI_File_open(MPI_COMM_WORLD, (const char*)file_name,
                             MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (ierr!=0) std::cerr << "File open error: " << file_name << '\n';

    // work around MPI count limit by using a large dtype
    if (count>=(int64_t)INT_MAX)
    {
        MPI_Datatype large_dtype;
        MPI_Type_contiguous(LARGE_DTYPE_SIZE, MPI_CHAR, &large_dtype);
        MPI_Type_commit(&large_dtype);
        int read_size = (int) (count/LARGE_DTYPE_SIZE);

        ierr = MPI_File_read_at_all(fh, (MPI_Offset)start, buff,
                             read_size, large_dtype, MPI_STATUS_IGNORE);
        if (ierr!=0)
        {
            MPI_Error_class(ierr, &err_class);
            std::cerr << "File large read error: " << err_class << " " << file_name << '\n';
            MPI_Error_string(ierr, err_string, &err_len);
            printf("Error %s\n", err_string); fflush(stdout);
        }
        MPI_Type_free(&large_dtype);
        int64_t left_over = count % LARGE_DTYPE_SIZE;
        int64_t read_byte_size = count-left_over;
        // printf("VAL leftover %lld read %lld\n", left_over, read_byte_size);
        start += read_byte_size;
        buff += read_byte_size;
        count = left_over;
    }
    // printf("MPI leftover READ %lld %lld\n", start, count);

    ierr = MPI_File_read_at_all(fh, (MPI_Offset)start, buff,
                         (int)count, MPI_CHAR, MPI_STATUS_IGNORE);

    // if (ierr!=0) std::cerr << "File read error: " << file_name << '\n';
    if (ierr!=0)
    {
        MPI_Error_class(ierr, &err_class);
        std::cerr << "File read error: " << err_class << " " << file_name << '\n';
        MPI_Error_string(ierr, err_string, &err_len);
        printf("Error %s\n", err_string); fflush(stdout);
    }

    MPI_File_close(&fh);
    return;
}


void file_write_parallel(char* file_name, char* buff, int64_t start, int64_t count, int64_t elem_size)
{
    // std::cout << file_name;
    // printf(" MPI WRITE %lld %lld %lld\n", start, count, elem_size);

    // TODO: handle large write count
    if (count>=(int64_t)INT_MAX) {
        std::cerr << "write count too large " << file_name << '\n';
        return;
    }

    char err_string[MPI_MAX_ERROR_STRING];
    err_string[MPI_MAX_ERROR_STRING-1] = '\0';
    int err_len, err_class;
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    MPI_File fh;
    int ierr = MPI_File_open(MPI_COMM_WORLD, (const char*)file_name,
                        MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if (ierr!=0) std::cerr << "File open error (write): " << file_name << '\n';

    MPI_Datatype elem_dtype;
    MPI_Type_contiguous(elem_size, MPI_CHAR, &elem_dtype);
    MPI_Type_commit(&elem_dtype);

    ierr = MPI_File_write_at_all(fh, (MPI_Offset)(start*elem_size), buff,
                         (int)count, elem_dtype, MPI_STATUS_IGNORE);

    MPI_Type_free(&elem_dtype);
    // if (ierr!=0) std::cerr << "File write error: " << file_name << '\n';
    if (ierr!=0)
    {
        MPI_Error_class(ierr, &err_class);
        std::cerr << "File write error: " << err_class << " " << file_name << '\n';
        MPI_Error_string(ierr, err_string, &err_len);
        printf("Error %s\n", err_string); fflush(stdout);
    }

    MPI_File_close(&fh);
    return;
}

} // extern "C"

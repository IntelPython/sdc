#include "mpi.h"
#include <Python.h>
#include <string>
#include <iostream>
#include <cstring>
#include <cmath>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>

// just include parquet reader on Windows since the GCC ABI change issue
// doesn't exist, and VC linker removes unused lib symbols
#ifdef _MSC_VER
#include <parquet_reader/hpat_parquet_reader.cpp>
#else

parquet type sizes (NOT arrow)
boolean, int32, int64, int96, float, double
int pq_type_sizes[] = {1, 4, 8, 12, 4, 8};

extern "C" {

int64_t pq_get_size_single_file(const char* file_name, int64_t column_idx);
int64_t pq_read_single_file(const char* file_name, int64_t column_idx, uint8_t *out,
                int out_dtype);
int pq_read_parallel_single_file(const char* file_name, int64_t column_idx,
                uint8_t* out_data, int out_dtype, int64_t start, int64_t count);
int64_t pq_read_string_single_file(const char* file_name, int64_t column_idx,
                                uint32_t **out_offsets, uint8_t **out_data,
    std::vector<uint32_t> *offset_vec=NULL, std::vector<uint8_t> *data_vec=NULL);
int pq_read_string_parallel_single_file(const char* file_name, int64_t column_idx,
        uint32_t **out_offsets, uint8_t **out_data, int64_t start, int64_t count,
        std::vector<uint32_t> *offset_vec=NULL, std::vector<uint8_t> *data_vec=NULL);

}  // extern "C"

#endif  // _MSC_VER

int64_t pq_get_size(std::string* file_name, int64_t column_idx);
int64_t pq_read(std::string* file_name, int64_t column_idx,
                uint8_t *out_data, int out_dtype);
int pq_read_parallel(std::string* file_name, int64_t column_idx,
                uint8_t* out_data, int out_dtype, int64_t start, int64_t count);
int pq_read_string(std::string* file_name, int64_t column_idx,
                                    uint32_t **out_offsets, uint8_t **out_data);
int pq_read_string_parallel(std::string* file_name, int64_t column_idx,
        uint32_t **out_offsets, uint8_t **out_data, int64_t start, int64_t count);


PyMODINIT_FUNC PyInit_parquet_cpp(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "parquet_cpp", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "read",
                            PyLong_FromVoidPtr((void*)(&pq_read)));
    PyObject_SetAttrString(m, "read_parallel",
                            PyLong_FromVoidPtr((void*)(&pq_read_parallel)));
    PyObject_SetAttrString(m, "get_size",
                            PyLong_FromVoidPtr((void*)(&pq_get_size)));
    PyObject_SetAttrString(m, "read_string",
                            PyLong_FromVoidPtr((void*)(&pq_read_string)));
    PyObject_SetAttrString(m, "read_string_parallel",
                            PyLong_FromVoidPtr((void*)(&pq_read_string_parallel)));

    return m;
}

bool pq_exclude_file(const std::string &file_name)
{
    return ( file_name.compare("_SUCCESS")==0
            || boost::algorithm::ends_with(file_name, "/_SUCCESS")
            || boost::algorithm::ends_with(file_name, "_common_metadata")
            || boost::algorithm::ends_with(file_name, "_metadata")
            || boost::algorithm::ends_with(file_name, ".crc"));
}


std::vector<std::string> get_dir_pq_files(boost::filesystem::path &f_path)
{
    std::vector<std::string> all_files;

    for (boost::filesystem::directory_entry& x : boost::filesystem::directory_iterator(f_path))
    {
        std::string inner_file = x.path().string();
        // std::cout << inner_file << '\n';
        if (!pq_exclude_file(inner_file))
            all_files.push_back(inner_file);
    }
    // sort file names to match pyarrow order
    std::sort(all_files.begin(), all_files.end());
    return all_files;
}

int64_t pq_get_size(std::string* file_name, int64_t column_idx)
{
    // TODO: run on rank 0 and broadcast
    boost::filesystem::path f_path(*file_name);

    if (!boost::filesystem::exists(f_path))
    {
        std::cerr << "pq get size - parquet file path does not exist: " << *file_name << '\n';
        return 0;
    }

    if (boost::filesystem::is_directory(f_path))
    {
        // std::cout << "pq path is dir" << '\n';
        int64_t ret = 0;
        std::vector<std::string> all_files = get_dir_pq_files(f_path);
        for (const auto& inner_file : all_files)
        {
            ret += pq_get_size_single_file(inner_file.c_str(), column_idx);
        }

        // std::cout << "total pq dir size: " << ret << '\n';
        return ret;
    }
    else
    {
        return pq_get_size_single_file(file_name->c_str(), column_idx);
    }
    return 0;
}

int64_t pq_read(std::string* file_name, int64_t column_idx,
                uint8_t *out_data, int out_dtype)
{

    boost::filesystem::path f_path(*file_name);

    if (!boost::filesystem::exists(f_path))
        std::cerr << "parquet file path does not exist: " << *file_name << '\n';

    if (boost::filesystem::is_directory(f_path))
    {
        // std::cout << "pq path is dir" << '\n';
        std::vector<std::string> all_files = get_dir_pq_files(f_path);

        int64_t byte_offset = 0;
        for (const auto& inner_file : all_files)
        {
            byte_offset += pq_read_single_file(inner_file.c_str(), column_idx, out_data+byte_offset, out_dtype);
        }

        // std::cout << "total pq dir size: " << byte_offset << '\n';
        return byte_offset;
    }
    else
    {
        return pq_read_single_file(file_name->c_str(), column_idx, out_data, out_dtype);
    }
    return 0;
}

int pq_read_parallel(std::string* file_name, int64_t column_idx,
                uint8_t* out_data, int out_dtype, int64_t start, int64_t count)
{
    // printf("read parquet parallel column: %lld start: %lld count: %lld\n",
    //                                                 column_idx, start, count);

    if (count==0) {
        return 0;
    }

    boost::filesystem::path f_path(*file_name);

    if (!boost::filesystem::exists(f_path))
        std::cerr << "parquet file path does not exist: " << *file_name << '\n';

    if (boost::filesystem::is_directory(f_path))
    {
        // std::cout << "pq path is dir" << '\n';
        // TODO: get file sizes on root rank only
        std::vector<std::string> all_files = get_dir_pq_files(f_path);

        // skip whole files if no need to read any rows
        int file_ind = 0;
        int64_t file_size = pq_get_size_single_file(all_files[0].c_str(), column_idx);
        while (start >= file_size)
        {
            start -= file_size;
            file_ind++;
            file_size = pq_get_size_single_file(all_files[file_ind].c_str(), column_idx);
        }

        int dtype_size = pq_type_sizes[out_dtype];
        // std::cout << "dtype_size: " << dtype_size << '\n';

        // read data
        int64_t read_rows = 0;
        while (read_rows<count)
        {
            int64_t rows_to_read = std::min(count-read_rows, file_size-start);
            pq_read_parallel_single_file(all_files[file_ind].c_str(), column_idx,
                out_data+read_rows*dtype_size, out_dtype, start, rows_to_read);
            read_rows += rows_to_read;
            start = 0;  // start becomes 0 after reading non-empty first chunk
            file_ind++;
            // std::cout << "next file: " << all_files[file_ind] << '\n';
            if (read_rows<count)
                file_size = pq_get_size_single_file(all_files[file_ind].c_str(), column_idx);
        }
        return 0;
        // std::cout << "total pq dir size: " << byte_offset << '\n';
    }
    else
    {
        return pq_read_parallel_single_file(file_name->c_str(), column_idx,
                                        out_data, out_dtype, start, count);
    }
    return 0;
}

int pq_read_string(std::string* file_name, int64_t column_idx,
                                    uint32_t **out_offsets, uint8_t **out_data)
{
    // std::cout << "string read file" << *file_name << '\n';
    boost::filesystem::path f_path(*file_name);

    if (!boost::filesystem::exists(f_path))
    {
        std::cerr << "parquet file path does not exist: " << *file_name << '\n';
        return 0;
    }

    if (boost::filesystem::is_directory(f_path))
    {
        // std::cout << "pq path is dir" << '\n';
        std::vector<std::string> all_files = get_dir_pq_files(f_path);

        std::vector<uint32_t> offset_vec;
        std::vector<uint8_t> data_vec;
        int32_t last_offset = 0;
        int64_t res = 0;
        for (const auto& inner_file : all_files)
        {
            int64_t n_vals = pq_read_string_single_file(inner_file.c_str(), column_idx, NULL, NULL, &offset_vec, &data_vec);
            if (n_vals==-1)
                continue;

            int size = offset_vec.size();
            for(int64_t i=1; i<=n_vals+1; i++)
                offset_vec[size-i] += last_offset;
            last_offset = offset_vec[size-1];
            offset_vec.pop_back();
            res += n_vals;
        }
        offset_vec.push_back(last_offset);

        *out_offsets = new uint32_t[offset_vec.size()];
        *out_data = new uint8_t[data_vec.size()];

        memcpy(*out_offsets, offset_vec.data(), offset_vec.size()*sizeof(uint32_t));
        memcpy(*out_data, data_vec.data(), data_vec.size());
        // for(int i=0; i<offset_vec.size(); i++)
        //     std::cout << (*out_offsets)[i] << ' ';
        // std::cout << '\n';
        // std::cout << "string dir read done" << '\n';
        return res;
    }
    else
    {
        return pq_read_string_single_file(file_name->c_str(), column_idx, out_offsets, out_data);
    }
    return 0;
}

int pq_read_string_parallel(std::string* file_name, int64_t column_idx,
        uint32_t **out_offsets, uint8_t **out_data, int64_t start, int64_t count)
{
    // printf("read parquet parallel str file: %s column: %lld start: %lld count: %lld\n",
    //                                 file_name->c_str(), column_idx, start, count);

    boost::filesystem::path f_path(*file_name);

    if (!boost::filesystem::exists(f_path))
    {
        std::cerr << "read parquet parallel str column - parquet file path does not exist: " << *file_name << '\n';
        return 0;
    }

    if (boost::filesystem::is_directory(f_path))
    {
        // std::cout << "pq path is dir" << '\n';
        std::vector<std::string> all_files = get_dir_pq_files(f_path);

        // skip whole files if no need to read any rows
        int file_ind = 0;
        int64_t file_size = pq_get_size_single_file(all_files[0].c_str(), column_idx);
        while (start >= file_size)
        {
            start -= file_size;
            file_ind++;
            file_size = pq_get_size_single_file(all_files[file_ind].c_str(), column_idx);
        }

        int64_t res = 0;
        std::vector<uint32_t> offset_vec;
        std::vector<uint8_t> data_vec;

        // read data
        int64_t last_offset = 0;
        int64_t read_rows = 0;
        while (read_rows<count)
        {
            int64_t rows_to_read = std::min(count-read_rows, file_size-start);
            if (rows_to_read>0)
            {
                pq_read_string_parallel_single_file(all_files[file_ind].c_str(), column_idx,
                    NULL, NULL, start, rows_to_read, &offset_vec, &data_vec);

                int size = offset_vec.size();
                for(int64_t i=1; i<=rows_to_read+1; i++)
                    offset_vec[size-i] += last_offset;
                last_offset = offset_vec[size-1];
                offset_vec.pop_back();
                res += rows_to_read;
            }

            read_rows += rows_to_read;
            start = 0;  // start becomes 0 after reading non-empty first chunk
            file_ind++;
            if (read_rows<count)
                file_size = pq_get_size_single_file(all_files[file_ind].c_str(), column_idx);
        }
        offset_vec.push_back(last_offset);

        *out_offsets = new uint32_t[offset_vec.size()];
        *out_data = new uint8_t[data_vec.size()];

        memcpy(*out_offsets, offset_vec.data(), offset_vec.size()*sizeof(uint32_t));
        memcpy(*out_data, data_vec.data(), data_vec.size());
        return res;
    }
    else
    {
        return pq_read_string_parallel_single_file(file_name->c_str(), column_idx,
                out_offsets, out_data, start, count);
    }
    return 0;
}

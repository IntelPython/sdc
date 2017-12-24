#include "mpi.h"
#include <Python.h>
#include <string>
#include <iostream>
#include <cstring>
#include <cmath>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>

#if _MSC_VER >= 1900
  #undef timezone
#endif

#include "parquet/api/reader.h"
#include "parquet/arrow/reader.h"
#include "arrow/table.h"
#include "arrow/io/hdfs.h"

using parquet::arrow::FileReader;
using parquet::ParquetFileReader;

void pq_init_reader(const std::string* file_name,
        std::shared_ptr<FileReader> *a_reader);
int64_t pq_get_size(const std::string* file_name, int64_t column_idx);
int64_t pq_read(const std::string* file_name, int64_t column_idx, uint8_t *out);
int pq_read_parallel(std::string* file_name, int64_t column_idx,
                            uint8_t* out_data, int64_t start, int64_t count);
inline void copy_data(uint8_t* out_data, const uint8_t* buff,
                    int64_t rows_to_skip, int64_t rows_to_read, int dtype,
                    const uint8_t* null_bitmap_buff);
int pq_read_string(std::string* file_name, int64_t column_idx,
                                    uint8_t **out_offsets, uint8_t **out_data);
int pq_read_string_parallel(std::string* file_name, int64_t column_idx,
        uint32_t **out_offsets, uint8_t **out_data, int64_t start, int64_t count);
// parquet type sizes (NOT arrow)
// boolean, int32, int64, int96, float, double
int pq_type_sizes[] = {1, 4, 8, 12, 4, 8};


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

int64_t pq_get_size(const std::string* file_name, int64_t column_idx)
{
    // TODO: run on rank 0 and broadcast
    boost::filesystem::path f_path(*file_name);

    if (!boost::filesystem::exists(f_path))
        std::cerr << "parquet file path does not exist: " << *file_name << '\n';

    if (boost::filesystem::is_directory(f_path))
    {
        // std::cout << "pq path is dir" << '\n';
        int64_t ret = 0;
        for (boost::filesystem::directory_entry& x : boost::filesystem::directory_iterator(f_path))
        {
            std::string inner_file = x.path().string();
            // std::cout << inner_file << '\n';
            if (!pq_exclude_file(inner_file))
                ret += pq_get_size(&inner_file, column_idx);
        }

        // std::cout << "total pq dir size: " << ret << '\n';
        return ret;
    }

    std::shared_ptr<FileReader> arrow_reader;
    pq_init_reader(file_name, &arrow_reader);
    int64_t nrows = arrow_reader->parquet_reader()->metadata()->num_rows();
    // std::cout << nrows << std::endl;
    return nrows;
}

int64_t pq_read(const std::string* file_name, int64_t column_idx, uint8_t *out_data)
{

    boost::filesystem::path f_path(*file_name);

    if (!boost::filesystem::exists(f_path))
        std::cerr << "parquet file path does not exist: " << *file_name << '\n';

    if (boost::filesystem::is_directory(f_path))
    {
        // std::cout << "pq path is dir" << '\n';
        std::vector<std::string> all_files;
        for (boost::filesystem::directory_entry& x : boost::filesystem::directory_iterator(f_path))
        {
            std::string inner_file = x.path().string();
            if (!pq_exclude_file(inner_file))
                all_files.push_back(inner_file);
        }
        // sort file names to match pyarrow order
        std::sort(all_files.begin(), all_files.end());
        int64_t byte_offset = 0;
        for (const auto& inner_file : all_files)
        {
            byte_offset += pq_read(&inner_file, column_idx, out_data+byte_offset);
        }

        // std::cout << "total pq dir size: " << byte_offset << '\n';
        return byte_offset;
    }

    std::shared_ptr<FileReader> arrow_reader;
    pq_init_reader(file_name, &arrow_reader);

    std::shared_ptr< ::arrow::Array > arr;
    arrow_reader->ReadColumn(column_idx, &arr);
    if (arr==NULL)
        return 0;

    int64_t num_values = arr->length();
    // std::cout << "arr: " << arr->ToString() << std::endl;
    int dtype = arrow_reader->parquet_reader()->metadata()->RowGroup(0)->
                                            ColumnChunk(column_idx)->type();
    int dtype_size = pq_type_sizes[dtype];
    // printf("dtype %d\n", dtype);

    auto buffers = arr->data()->buffers;
    // std::cout<<"num buffs: "<< buffers.size()<<std::endl;
    if (buffers.size()!=2) {
        std::cerr << "invalid parquet number of array buffers" << std::endl;
    }
    int64_t buff_size = buffers[1]->size();
    const uint8_t* buff = buffers[1]->data();
    const uint8_t* null_bitmap_buff = buffers[0]->data();

    copy_data(out_data, buff, 0, num_values, dtype, null_bitmap_buff);
    // memcpy(out_data, buffers[1]->data(), buff_size);
    return num_values*dtype_size;
}

int pq_read_parallel(std::string* file_name, int64_t column_idx,
                                uint8_t* out_data, int64_t start, int64_t count)
{
    // printf("read parquet parallel column: %lld start: %lld count: %lld\n",
    //                                                 column_idx, start, count);

    if (count==0) {
        return 0;
    }

    std::shared_ptr<FileReader> arrow_reader;
    pq_init_reader(file_name, &arrow_reader);

    int64_t n_row_groups = arrow_reader->parquet_reader()->metadata()->num_row_groups();
    std::vector<int> column_indices;
    column_indices.push_back(column_idx);

    int row_group_index = 0;
    int64_t skipped_rows = 0;
    int64_t read_rows = 0;


    auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
    int64_t nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
    int dtype = rg_metadata->ColumnChunk(column_idx)->type();
    int dtype_size = pq_type_sizes[dtype];

    // skip whole row groups if no need to read any rows
    while (start-skipped_rows >= nrows_in_group)
    {
        skipped_rows += nrows_in_group;
        row_group_index++;
        auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
        nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
    }

    // printf("first row group: %d skipped_rows: %lld nrows_in_group: %lld\n", row_group_index, skipped_rows, nrows_in_group);

    while (read_rows<count)
    {
        /* -------- read row group ---------- */
        std::shared_ptr<::arrow::Table> table;
        arrow_reader->ReadRowGroup(row_group_index, column_indices, &table);
        std::shared_ptr< ::arrow::Column > column = table->column(0);
        std::shared_ptr< ::arrow::ChunkedArray > chunked_arr = column->data();
        // std::cout << chunked_arr->num_chunks() << std::endl;
        if (chunked_arr->num_chunks()!=1) {
            std::cerr << "invalid parquet number of array chunks" << std::endl;
        }
        std::shared_ptr< ::arrow::Array > arr = chunked_arr->chunk(0);
        // std::cout << arr->ToString() << std::endl;
        auto buffers = arr->data()->buffers;
        // std::cout<<"num buffs: "<< buffers.size()<<std::endl;
        if (buffers.size()!=2) {
            std::cerr << "invalid parquet number of array buffers" << std::endl;
        }
        const uint8_t* buff = buffers[1]->data();
        const uint8_t* null_bitmap_buff = buffers[0]->data();
        /* ----------- read row group ------- */

        int64_t rows_to_skip = start - skipped_rows;
        int64_t rows_to_read = std::min(count-read_rows, nrows_in_group-rows_to_skip);
        // printf("rows_to_skip: %ld rows_to_read: %ld\n", rows_to_skip, rows_to_read);

        copy_data(out_data+read_rows*dtype_size, buff, rows_to_skip, rows_to_read, dtype, null_bitmap_buff);
        // memcpy(out_data+read_rows*dtype_size, buff+rows_to_skip*dtype_size, rows_to_read*dtype_size);

        skipped_rows += rows_to_skip;
        read_rows += rows_to_read;

        row_group_index++;
        if (row_group_index<n_row_groups)
        {
            auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
            nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
        }
        else
            break;
    }
    if (read_rows!=count)
        std::cerr << "parquet read incomplete" << '\n';
    return 0;
}

inline void copy_data(uint8_t* out_data, const uint8_t* buff,
                        int64_t rows_to_skip, int64_t rows_to_read, int dtype,
                        const uint8_t* null_bitmap_buff)
{
    // unpack booleans from bits
    if (dtype==0)
    {
        for(int64_t i=0; i<rows_to_read; i++)
        {
            // std::cout << ::arrow::BitUtil::GetBit(buff, i+rows_to_skip) << std::endl;
            out_data[i] = (uint8_t) ::arrow::BitUtil::GetBit(buff, i+rows_to_skip);
        }
        return;
    }
    int dtype_size = pq_type_sizes[dtype];
    memcpy(out_data, buff+rows_to_skip*dtype_size, rows_to_read*dtype_size);
    // set NaNs for double values
    if (dtype==5)
    {
        double *double_data = (double*)out_data;
        for(int64_t i=0; i<rows_to_read; i++)
        {
            if (::arrow::BitUtil::BitNotSet(null_bitmap_buff, i+rows_to_skip))
            {
                // std::cout << "NULL found" << std::endl;
                // TODO: use NPY_NAN
                double_data[i] = std::nan("");
            }
        }
    }
    // set NaNs for float values
    if (dtype==4)
    {
        float *float_data = (float*)out_data;
        for(int64_t i=0; i<rows_to_read; i++)
        {
            if (::arrow::BitUtil::BitNotSet(null_bitmap_buff, i+rows_to_skip))
            {
                // std::cout << "NULL found" << std::endl;
                // TODO: use NPY_NAN
                float_data[i] = std::nanf("");
            }
        }
    }
    return;
}

int pq_read_string(std::string* file_name, int64_t column_idx,
                                    uint8_t **out_offsets, uint8_t **out_data)
{

    std::shared_ptr<FileReader> arrow_reader;
    pq_init_reader(file_name, &arrow_reader);
    //
    std::shared_ptr< ::arrow::Array > arr;
    arrow_reader->ReadColumn(column_idx, &arr);
    int64_t num_values = arr->length();
    // std::cout << arr->ToString() << std::endl;
    int dtype = arrow_reader->parquet_reader()->metadata()->RowGroup(0)->
                                            ColumnChunk(column_idx)->type();
    if (dtype!=6) // TODO: get constant from parquet-cpp
        std::cerr << "Invalid Parquet string data type" << '\n';


    auto buffers = arr->data()->buffers;
    // std::cout<<"num buffs: "<< buffers.size()<<std::endl;
    if (buffers.size()!=3) {
        std::cerr << "invalid parquet string number of array buffers" << std::endl;
    }

    int64_t offsets_size = buffers[1]->size();
    int64_t data_size = buffers[2]->size();
    // std::cout << "offsets: " << offsets_size << " chars: " << data_size << std::endl;

    *out_offsets = new uint8_t[offsets_size];
    *out_data = new uint8_t[data_size];

    const uint8_t* offsets_buff = buffers[1]->data();
    const uint8_t* data_buff = buffers[2]->data();

    memcpy(*out_offsets, offsets_buff, offsets_size);
    memcpy(*out_data, data_buff, data_size);
    return 0;
}

int pq_read_string_parallel(std::string* file_name, int64_t column_idx,
        uint32_t **out_offsets, uint8_t **out_data, int64_t start, int64_t count)
{
    // printf("read parquet parallel column: %lld start: %lld count: %lld\n",
    //                                                  column_idx, start, count);

    if (count==0) {
        *out_offsets = NULL;
        *out_data = NULL;
        return 0;
    }

    std::shared_ptr<FileReader> arrow_reader;
    pq_init_reader(file_name, &arrow_reader);
    int dtype = arrow_reader->parquet_reader()->metadata()->RowGroup(0)->
                                            ColumnChunk(column_idx)->type();
    if (dtype!=6) // TODO: get constant from parquet-cpp
        std::cerr << "Invalid Parquet string data type" << '\n';


    *out_offsets = new uint32_t[count+1];
    std::vector<uint8_t> tmp_buffer;

    int64_t n_row_groups = arrow_reader->parquet_reader()->metadata()->num_row_groups();
    std::vector<int> column_indices;
    column_indices.push_back(column_idx);

    int row_group_index = 0;
    int64_t skipped_rows = 0;
    int64_t read_rows = 0;

    auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
    int64_t nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();

    // skip whole row groups if no need to read any rows
    while (start-skipped_rows >= nrows_in_group)
    {
        skipped_rows += nrows_in_group;
        row_group_index++;
        auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
        nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
    }

    // printf("first row group: %d skipped_rows: %lld nrows_in_group: %lld\n", row_group_index, skipped_rows, nrows_in_group);

    uint32_t curr_offset = 0;

    /* ------- read offsets and data ------ */
    while (read_rows<count)
    {
        /* -------- read row group ---------- */
        std::shared_ptr<::arrow::Table> table;
        arrow_reader->ReadRowGroup(row_group_index, column_indices, &table);
        std::shared_ptr< ::arrow::Column > column = table->column(0);
        std::shared_ptr< ::arrow::ChunkedArray > chunked_arr = column->data();
        // std::cout << chunked_arr->num_chunks() << std::endl;
        if (chunked_arr->num_chunks()!=1) {
            std::cerr << "invalid parquet number of array chunks" << std::endl;
        }
        std::shared_ptr< ::arrow::Array > arr = chunked_arr->chunk(0);
        // std::cout << arr->ToString() << std::endl;
        auto buffers = arr->data()->buffers;
        // std::cout<<"num buffs: "<< buffers.size()<<std::endl;
        if (buffers.size()!=3) {
            std::cerr << "invalid parquet string number of array buffers" << std::endl;
        }

        const uint32_t* offsets_buff = (const uint32_t*) buffers[1]->data();
        const uint8_t* data_buff = buffers[2]->data();

        /* ----------- read row group ------- */

        int64_t rows_to_skip = start - skipped_rows;
        int64_t rows_to_read = std::min(count-read_rows, nrows_in_group-rows_to_skip);
        // printf("rows_to_skip: %ld rows_to_read: %ld\n", rows_to_skip, rows_to_read);

        for(int64_t i=0; i<rows_to_read; i++) {
            uint32_t str_size = offsets_buff[rows_to_skip+i+1]-offsets_buff[rows_to_skip+i];
            (*out_offsets)[read_rows+i] = curr_offset;
            curr_offset += str_size;
        }

        int data_size = offsets_buff[rows_to_skip+rows_to_read]
                                    - offsets_buff[rows_to_skip];

        tmp_buffer.insert(tmp_buffer.end(),
            data_buff+offsets_buff[rows_to_skip],
            data_buff+offsets_buff[rows_to_skip]+data_size);

        skipped_rows += rows_to_skip;
        read_rows += rows_to_read;

        row_group_index++;
        if (row_group_index<n_row_groups)
        {
            auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
            nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
        }
        else
            break;
    }
    if (read_rows!=count)
        std::cerr << "parquet read incomplete" << '\n';

    (*out_offsets)[count] = curr_offset;
    *out_data = new uint8_t[curr_offset];
    // printf("buffer size:%d curr_offset:%d\n", tmp_buffer.size(), curr_offset);
    memcpy(*out_data, tmp_buffer.data(), curr_offset);

    // printf("offsets: ");
    // for(int i=0; i<=count; i++)
    // {
    //     printf("%d ", (*out_offsets)[i]);
    // }
    // printf("\n");
    return 0;
}

void pq_init_reader(const std::string* file_name,
        std::shared_ptr<FileReader> *a_reader)
{
    auto pool = ::arrow::default_memory_pool();

    // HDFS if starts with hdfs://
    if (file_name->find("hdfs://")==0)
    {
        ::arrow::Status stat = ::arrow::io::HaveLibHdfs();
        if (!stat.ok())
            std::cerr << "libhdfs not found" << '\n';
            ::arrow::io::HdfsConnectionConfig hfs_config;
        // TODO: extract localhost and port from path if available
        hfs_config.host = std::string("default");
        hfs_config.port = 0;
        hfs_config.driver = ::arrow::io::HdfsDriver::LIBHDFS;
        hfs_config.user = std::string("");
        hfs_config.kerb_ticket = std::string("");

        std::shared_ptr<::arrow::io::HadoopFileSystem> fs;
        ::arrow::io::HadoopFileSystem::Connect(&hfs_config, &fs);
        std::shared_ptr<::arrow::io::HdfsReadableFile> file;
        fs->OpenReadable(*file_name, &file);
        a_reader->reset(new FileReader(pool, ParquetFileReader::Open(file)));
    }
    else  // regular file system
    {
        a_reader->reset(new FileReader(pool,
                            ParquetFileReader::OpenFile(*file_name, false)));
    }
    // printf("file open for arrow reader done\n");
    // fflush(stdout);
    return;
}

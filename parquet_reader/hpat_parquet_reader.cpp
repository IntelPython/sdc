#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

#if _MSC_VER >= 1900
#undef timezone
#endif

#include "arrow/io/hdfs.h"
#include "arrow/table.h"
#include "arrow/type.h"
#include "parquet/api/reader.h"
#include "parquet/arrow/reader.h"
#include "parquet/arrow/schema.h"

using arrow::Type;
using parquet::ParquetFileReader;
using parquet::arrow::FileReader;

extern "C"
{
    int64_t pq_get_size_single_file(std::shared_ptr<FileReader> arrow_reader, int64_t column_idx);
    int64_t
        pq_read_single_file(std::shared_ptr<FileReader> arrow_reader, int64_t column_idx, uint8_t* out, int out_dtype);
    int pq_read_parallel_single_file(std::shared_ptr<FileReader> arrow_reader,
                                     int64_t column_idx,
                                     uint8_t* out_data,
                                     int out_dtype,
                                     int64_t start,
                                     int64_t count);

    int64_t pq_read_string_single_file(std::shared_ptr<FileReader> arrow_reader,
                                       int64_t column_idx,
                                       uint32_t** out_offsets,
                                       uint8_t** out_data,
                                       uint8_t** out_nulls,
                                       std::vector<uint32_t>* offset_vec = NULL,
                                       std::vector<uint8_t>* data_vec = NULL,
                                       std::vector<bool>* null_vec = NULL);
    int pq_read_string_parallel_single_file(std::shared_ptr<FileReader> arrow_reader,
                                            int64_t column_idx,
                                            uint32_t** out_offsets,
                                            uint8_t** out_data,
                                            uint8_t** out_nulls,
                                            int64_t start,
                                            int64_t count,
                                            std::vector<uint32_t>* offset_vec = NULL,
                                            std::vector<uint8_t>* data_vec = NULL,
                                            std::vector<bool>* null_vec = NULL);

} // extern "C"

void pack_null_bitmap(uint8_t** out_nulls, std::vector<bool>& null_vec, int64_t n_all_vals);
std::shared_ptr<arrow::DataType> get_arrow_type(std::shared_ptr<FileReader> arrow_reader, int64_t column_idx);
bool arrowPqTypesEqual(std::shared_ptr<arrow::DataType> arrow_type, ::parquet::Type::type pq_type);
inline void copy_data(uint8_t* out_data,
                      const uint8_t* buff,
                      int64_t rows_to_skip,
                      int64_t rows_to_read,
                      std::shared_ptr<arrow::DataType> arrow_type,
                      const uint8_t* null_bitmap_buff,
                      int out_dtype);

template <typename T, int64_t SHIFT>
inline void convertArrowToDT64(const uint8_t* buff, uint8_t* out_data, int64_t rows_to_skip, int64_t rows_to_read);
void append_bits_to_vec(
    std::vector<bool>* null_vec, const uint8_t* null_buff, int64_t null_size, int64_t offset, int64_t num_values);

void pq_init_reader(const char* file_name, std::shared_ptr<FileReader>* a_reader);

// parquet type sizes (NOT arrow), parquet/types.h
// boolean, int32, int64, int96, float, double, byte
// XXX assuming int96 is always converted to int64 since it's timestamp
static int pq_type_sizes[] = {1, 4, 8, 8, 4, 8, 1};
#define PQ_DT64_TYPE 3                     // using INT96 value as dt64, TODO: refactor
#define kNanosecondsInDay 86400000000000LL // TODO: reuse from type_traits.h

int64_t pq_get_size_single_file(std::shared_ptr<FileReader> arrow_reader, int64_t column_idx)
{
    int64_t nrows = arrow_reader->parquet_reader()->metadata()->num_rows();
    // std::cout << nrows << std::endl;
    return nrows;
}

int64_t
    pq_read_single_file(std::shared_ptr<FileReader> arrow_reader, int64_t column_idx, uint8_t* out_data, int out_dtype)
{
    std::shared_ptr<::arrow::ChunkedArray> chunked_array;
    arrow_reader->ReadColumn(column_idx, &chunked_array);
    if (chunked_array == NULL)
        return 0;

    auto arr = chunked_array->chunk(0);

    int64_t num_values = arr->length();
    // std::cout << "arr: " << arr->ToString() << std::endl;
    std::shared_ptr<arrow::DataType> arrow_type = get_arrow_type(arrow_reader, column_idx);
    int dtype_size = pq_type_sizes[out_dtype];
    // printf("out_dtype %d dtype_size %d\n", out_dtype, dtype_size);
    // std::cout << arrow_type->name() << "\n";

    auto buffers = arr->data()->buffers;
    // std::cout<<"num buffs: "<< buffers.size()<<std::endl;
    if (buffers.size() != 2)
    {
        std::cerr << "invalid parquet number of array buffers" << std::endl;
    }
    // int64_t buff_size = buffers[1]->size();
    const uint8_t* buff = buffers[1]->data();
    const uint8_t* null_bitmap_buff = arr->null_count() == 0 ? nullptr : arr->null_bitmap_data();

    copy_data(out_data, buff, 0, num_values, arrow_type, null_bitmap_buff, out_dtype);
    // memcpy(out_data, buffers[1]->data(), buff_size);
    return num_values * dtype_size;
}

int pq_read_parallel_single_file(std::shared_ptr<FileReader> arrow_reader,
                                 int64_t column_idx,
                                 uint8_t* out_data,
                                 int out_dtype,
                                 int64_t start,
                                 int64_t count)
{
    if (count == 0)
    {
        return 0;
    }

    int64_t n_row_groups = arrow_reader->parquet_reader()->metadata()->num_row_groups();
    std::vector<int> column_indices;
    column_indices.push_back(column_idx);

    int row_group_index = 0;
    int64_t skipped_rows = 0;
    int64_t read_rows = 0;

    auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
    int64_t nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
    std::shared_ptr<arrow::DataType> arrow_type = get_arrow_type(arrow_reader, column_idx);
    int dtype_size = pq_type_sizes[out_dtype];

    // skip whole row groups if no need to read any rows
    while (start - skipped_rows >= nrows_in_group)
    {
        skipped_rows += nrows_in_group;
        row_group_index++;
        auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
        nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
    }

    // printf("first row group: %d skipped_rows: %lld nrows_in_group: %lld\n", row_group_index, skipped_rows, nrows_in_group);

    while (read_rows < count)
    {
        /* -------- read row group ---------- */
        std::shared_ptr<::arrow::Table> table;
        arrow_reader->ReadRowGroup(row_group_index, column_indices, &table);
        std::shared_ptr<::arrow::ChunkedArray> chunked_arr = table->column(0);
        // std::cout << chunked_arr->num_chunks() << std::endl;
        if (chunked_arr->num_chunks() != 1)
        {
            std::cerr << "invalid parquet number of array chunks" << std::endl;
        }
        std::shared_ptr<::arrow::Array> arr = chunked_arr->chunk(0);
        // std::cout << arr->ToString() << std::endl;
        auto buffers = arr->data()->buffers;
        // std::cout<<"num buffs: "<< buffers.size()<<std::endl;
        if (buffers.size() != 2)
        {
            std::cerr << "invalid parquet number of array buffers" << std::endl;
        }
        const uint8_t* buff = buffers[1]->data();
        const uint8_t* null_bitmap_buff = arr->null_count() == 0 ? nullptr : arr->null_bitmap_data();
        /* ----------- read row group ------- */

        int64_t rows_to_skip = start - skipped_rows;
        int64_t rows_to_read = std::min(count - read_rows, nrows_in_group - rows_to_skip);
        // printf("rows_to_skip: %ld rows_to_read: %ld\n", rows_to_skip, rows_to_read);

        copy_data(out_data + read_rows * dtype_size,
                  buff,
                  rows_to_skip,
                  rows_to_read,
                  arrow_type,
                  null_bitmap_buff,
                  out_dtype);
        // memcpy(out_data+read_rows*dtype_size, buff+rows_to_skip*dtype_size, rows_to_read*dtype_size);

        skipped_rows += rows_to_skip;
        read_rows += rows_to_read;

        row_group_index++;
        if (row_group_index < n_row_groups)
        {
            auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
            nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
        }
        else
            break;
    }
    if (read_rows != count)
        std::cerr << "parquet read incomplete" << '\n';
    return 0;
}

template <typename T_in, typename T_out>
inline void copy_data_cast(uint8_t* out_data,
                           const uint8_t* buff,
                           int64_t rows_to_skip,
                           int64_t rows_to_read,
                           std::shared_ptr<arrow::DataType> arrow_type,
                           int out_dtype)
{
    T_out* out_data_cast = (T_out*)out_data;
    T_in* in_data_cast = (T_in*)buff;
    for (int64_t i = 0; i < rows_to_read; i++)
    {
        out_data_cast[i] = (T_out)in_data_cast[rows_to_skip + i];
    }
}

inline void copy_data_dispatch(uint8_t* out_data,
                               const uint8_t* buff,
                               int64_t rows_to_skip,
                               int64_t rows_to_read,
                               std::shared_ptr<arrow::DataType> arrow_type,
                               int out_dtype)
{
    // TODO: rewrite in macros?
    // TODO: convert boolean
    // input is int32
    if (arrow_type->id() == Type::INT32)
    {
        if (out_dtype == 2)
            copy_data_cast<int, int64_t>(out_data, buff, rows_to_skip, rows_to_read, arrow_type, out_dtype);
        if (out_dtype == 4)
            copy_data_cast<int, float>(out_data, buff, rows_to_skip, rows_to_read, arrow_type, out_dtype);
        if (out_dtype == 5)
            copy_data_cast<int, double>(out_data, buff, rows_to_skip, rows_to_read, arrow_type, out_dtype);
    }
    // input is int64
    if (arrow_type->id() == Type::INT64)
    {
        if (out_dtype == 1)
            copy_data_cast<int64_t, int>(out_data, buff, rows_to_skip, rows_to_read, arrow_type, out_dtype);
        if (out_dtype == 4)
            copy_data_cast<int64_t, float>(out_data, buff, rows_to_skip, rows_to_read, arrow_type, out_dtype);
        if (out_dtype == 5)
            copy_data_cast<int64_t, double>(out_data, buff, rows_to_skip, rows_to_read, arrow_type, out_dtype);
    }
    // input is float
    if (arrow_type->id() == Type::FLOAT)
    {
        if (out_dtype == 1)
            copy_data_cast<float, int>(out_data, buff, rows_to_skip, rows_to_read, arrow_type, out_dtype);
        if (out_dtype == 2)
            copy_data_cast<float, int64_t>(out_data, buff, rows_to_skip, rows_to_read, arrow_type, out_dtype);
        if (out_dtype == 5)
            copy_data_cast<float, double>(out_data, buff, rows_to_skip, rows_to_read, arrow_type, out_dtype);
    }
    // input is double
    if (arrow_type->id() == Type::DOUBLE)
    {
        if (out_dtype == 1)
            copy_data_cast<double, int>(out_data, buff, rows_to_skip, rows_to_read, arrow_type, out_dtype);
        if (out_dtype == 2)
            copy_data_cast<double, int64_t>(out_data, buff, rows_to_skip, rows_to_read, arrow_type, out_dtype);
        if (out_dtype == 4)
            copy_data_cast<double, float>(out_data, buff, rows_to_skip, rows_to_read, arrow_type, out_dtype);
    }
    // datetime64 cases
    if (out_dtype == PQ_DT64_TYPE)
    {
        // similar to arrow_to_pandas.cc
        if (arrow_type->id() == Type::DATE32)
        {
            // days since epoch
            convertArrowToDT64<int32_t, kNanosecondsInDay>(buff, out_data, rows_to_skip, rows_to_read);
        }
        else if (arrow_type->id() == Type::DATE64)
        {
            // Date64Type is millisecond timestamp stored as int64_t
            convertArrowToDT64<int64_t, 1000000L>(buff, out_data, rows_to_skip, rows_to_read);
        }
        else if (arrow_type->id() == Type::TIMESTAMP)
        {
            const auto& ts_type = static_cast<const arrow::TimestampType&>(*arrow_type);

            if (ts_type.unit() == arrow::TimeUnit::NANO)
            {
                int dtype_size = sizeof(int64_t);
                memcpy(out_data, buff + rows_to_skip * dtype_size, rows_to_read * dtype_size);
            }
            else if (ts_type.unit() == arrow::TimeUnit::MICRO)
            {
                convertArrowToDT64<int64_t, 1000L>(buff, out_data, rows_to_skip, rows_to_read);
            }
            else if (ts_type.unit() == arrow::TimeUnit::MILLI)
            {
                convertArrowToDT64<int64_t, 1000000L>(buff, out_data, rows_to_skip, rows_to_read);
            }
            else if (ts_type.unit() == arrow::TimeUnit::SECOND)
            {
                convertArrowToDT64<int64_t, 1000000000L>(buff, out_data, rows_to_skip, rows_to_read);
            }
            else
            {
                std::cerr << "Invalid datetime timeunit" << out_dtype << " " << arrow_type << std::endl;
            }
        }
        else
        {
            //
            std::cerr << "Invalid datetime conversion" << out_dtype << " " << arrow_type << std::endl;
        }
    }
}

inline void copy_data(uint8_t* out_data,
                      const uint8_t* buff,
                      int64_t rows_to_skip,
                      int64_t rows_to_read,
                      std::shared_ptr<arrow::DataType> arrow_type,
                      const uint8_t* null_bitmap_buff,
                      int out_dtype)
{
    // unpack booleans from bits
    if (out_dtype == 0)
    {
        if (arrow_type->id() != Type::BOOL)
            std::cerr << "boolean type error" << '\n';

        for (int64_t i = 0; i < rows_to_read; i++)
        {
            // std::cout << ::arrow::BitUtil::GetBit(buff, i+rows_to_skip) << std::endl;
            out_data[i] = (uint8_t)::arrow::BitUtil::GetBit(buff, i + rows_to_skip);
        }
        return;
    }

    if (arrowPqTypesEqual(arrow_type, (parquet::Type::type)out_dtype))
    {
        int dtype_size = pq_type_sizes[out_dtype];
        // fast path if no conversion required
        memcpy(out_data, buff + rows_to_skip * dtype_size, rows_to_read * dtype_size);
    }
    else
    {
        copy_data_dispatch(out_data, buff, rows_to_skip, rows_to_read, arrow_type, out_dtype);
    }
    // set NaNs for double values
    if (null_bitmap_buff != nullptr && out_dtype == ::parquet::Type::DOUBLE)
    {
        double* double_data = (double*)out_data;
        for (int64_t i = 0; i < rows_to_read; i++)
        {
            if (!::arrow::BitUtil::GetBit(null_bitmap_buff, i + rows_to_skip))
            {
                // std::cout << "NULL found" << std::endl;
                // TODO: use NPY_NAN
                double_data[i] = std::nan("");
            }
        }
    }
    // set NaNs for float values
    if (null_bitmap_buff != nullptr && out_dtype == ::parquet::Type::FLOAT)
    {
        float* float_data = (float*)out_data;
        for (int64_t i = 0; i < rows_to_read; i++)
        {
            if (!::arrow::BitUtil::GetBit(null_bitmap_buff, i + rows_to_skip))
            {
                // std::cout << "NULL found" << std::endl;
                // TODO: use NPY_NAN
                float_data[i] = std::nanf("");
            }
        }
    }
    return;
}

int64_t pq_read_string_single_file(std::shared_ptr<FileReader> arrow_reader,
                                   int64_t column_idx,
                                   uint32_t** out_offsets,
                                   uint8_t** out_data,
                                   uint8_t** out_nulls,
                                   std::vector<uint32_t>* offset_vec,
                                   std::vector<uint8_t>* data_vec,
                                   std::vector<bool>* null_vec)
{
    // std::cout << "string read file" << '\n';
    //
    std::shared_ptr<::arrow::ChunkedArray> chunked_arr;
    arrow_reader->ReadColumn(column_idx, &chunked_arr);
    if (chunked_arr == NULL)
        return -1;
    auto arr = chunked_arr->chunk(0);
    int64_t num_values = arr->length();
    // std::cout << arr->ToString() << std::endl;
    std::shared_ptr<arrow::DataType> arrow_type = get_arrow_type(arrow_reader, column_idx);
    if (arrow_type->id() != Type::STRING)
        std::cerr << "Invalid Parquet string data type" << '\n';

    auto buffers = arr->data()->buffers;
    // std::cout<<"num buffs: "<< buffers.size()<<std::endl;
    if (buffers.size() != 3)
    {
        std::cerr << "invalid parquet string number of array buffers" << std::endl;
    }

    int64_t null_size = buffers[0]->size();
    int64_t offsets_size = buffers[1]->size();
    int64_t data_size = buffers[2]->size();
    // std::cout << "offsets: " << offsets_size << " chars: " << data_size << std::endl;

    const uint32_t* offsets_buff = (const uint32_t*)buffers[1]->data();
    const uint8_t* data_buff = buffers[2]->data();
    const uint8_t* null_buff = arr->null_bitmap_data();

    if (offset_vec == NULL)
    {
        if (data_vec != NULL)
            std::cerr << "parquet read string input error" << '\n';

        *out_offsets = new uint32_t[offsets_size / sizeof(uint32_t)];
        *out_data = new uint8_t[data_size];

        // printf("null size %p %d\n", null_buff, null_size);
        if (null_buff != nullptr && null_size > 0)
        {
            *out_nulls = new uint8_t[null_size];
            memcpy(*out_nulls, null_buff, null_size);
            // printf("bitmap %d\n", (*out_nulls)[0]);
        }
        else
            *out_nulls = nullptr;

        memcpy(*out_offsets, offsets_buff, offsets_size);
        memcpy(*out_data, data_buff, data_size);
    }
    else
    {
        offset_vec->insert(offset_vec->end(), offsets_buff, offsets_buff + offsets_size / sizeof(uint32_t));
        data_vec->insert(data_vec->end(), data_buff, data_buff + data_size);
        append_bits_to_vec(null_vec, null_buff, null_size, 0, num_values);
    }

    return num_values;
}

int pq_read_string_parallel_single_file(std::shared_ptr<FileReader> arrow_reader,
                                        int64_t column_idx,
                                        uint32_t** out_offsets,
                                        uint8_t** out_data,
                                        uint8_t** out_nulls,
                                        int64_t start,
                                        int64_t count,
                                        std::vector<uint32_t>* offset_vec,
                                        std::vector<uint8_t>* data_vec,
                                        std::vector<bool>* null_vec)
{
    if (count == 0)
    {
        if (offset_vec == NULL)
        {
            *out_offsets = NULL;
            *out_data = NULL;
        }
        return 0;
    }

    std::shared_ptr<arrow::DataType> arrow_type = get_arrow_type(arrow_reader, column_idx);
    if (arrow_type->id() != Type::STRING)
        std::cerr << "Invalid Parquet string data type" << '\n';

    if (offset_vec == NULL)
    {
        *out_offsets = new uint32_t[count + 1];
        data_vec = new std::vector<uint8_t>();
        null_vec = new std::vector<bool>();
    }

    int64_t n_row_groups = arrow_reader->parquet_reader()->metadata()->num_row_groups();
    std::vector<int> column_indices;
    column_indices.push_back(column_idx);

    int row_group_index = 0;
    int64_t skipped_rows = 0;
    int64_t read_rows = 0;

    auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
    int64_t nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();

    // skip whole row groups if no need to read any rows
    while (start - skipped_rows >= nrows_in_group)
    {
        skipped_rows += nrows_in_group;
        row_group_index++;
        auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
        nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
    }

    // printf("first row group: %d skipped_rows: %lld nrows_in_group: %lld\n", row_group_index, skipped_rows, nrows_in_group);

    uint32_t curr_offset = 0;

    /* ------- read offsets and data ------ */
    while (read_rows < count)
    {
        /* -------- read row group ---------- */
        std::shared_ptr<::arrow::Table> table;
        arrow_reader->ReadRowGroup(row_group_index, column_indices, &table);
        std::shared_ptr<::arrow::ChunkedArray> chunked_arr = table->column(0);
        // std::cout << chunked_arr->num_chunks() << std::endl;
        if (chunked_arr->num_chunks() != 1)
        {
            std::cerr << "invalid parquet number of array chunks" << std::endl;
        }
        std::shared_ptr<::arrow::Array> arr = chunked_arr->chunk(0);
        // std::cout << arr->ToString() << std::endl;

        auto buffers = arr->data()->buffers;
        // std::cout<<"num buffs: "<< buffers.size()<<std::endl;
        if (buffers.size() != 3)
        {
            std::cerr << "invalid parquet string number of array buffers" << std::endl;
        }

        int64_t null_size = buffers[0]->size();
        const uint32_t* offsets_buff = (const uint32_t*)buffers[1]->data();
        const uint8_t* data_buff = buffers[2]->data();
        const uint8_t* null_buff = arr->null_bitmap_data();

        /* ----------- read row group ------- */

        int64_t rows_to_skip = start - skipped_rows;
        int64_t rows_to_read = std::min(count - read_rows, nrows_in_group - rows_to_skip);
        // printf("rows_to_skip: %ld rows_to_read: %ld\n", rows_to_skip, rows_to_read);

        for (int64_t i = 0; i < rows_to_read; i++)
        {
            uint32_t str_size = offsets_buff[rows_to_skip + i + 1] - offsets_buff[rows_to_skip + i];
            if (offset_vec == NULL)
                (*out_offsets)[read_rows + i] = curr_offset;
            else
                offset_vec->push_back(curr_offset);
            curr_offset += str_size;
        }

        int data_size = offsets_buff[rows_to_skip + rows_to_read] - offsets_buff[rows_to_skip];

        data_vec->insert(data_vec->end(),
                         data_buff + offsets_buff[rows_to_skip],
                         data_buff + offsets_buff[rows_to_skip] + data_size);
        append_bits_to_vec(null_vec, null_buff, null_size, rows_to_skip, rows_to_read);

        skipped_rows += rows_to_skip;
        read_rows += rows_to_read;

        row_group_index++;
        if (row_group_index < n_row_groups)
        {
            auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
            nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
        }
        else
            break;
    }
    if (read_rows != count)
        std::cerr << "parquet read incomplete" << '\n';

    if (offset_vec == NULL)
    {
        (*out_offsets)[count] = curr_offset;
        *out_data = new uint8_t[curr_offset];
        // printf("buffer size:%d curr_offset:%d\n", data_vec->size(), curr_offset);
        memcpy(*out_data, data_vec->data(), curr_offset);
        pack_null_bitmap(out_nulls, *null_vec, count);
        delete data_vec;
        delete null_vec;
    }
    else
        offset_vec->push_back(curr_offset);

    // printf("offsets: ");
    // for(int i=0; i<=count; i++)
    // {
    //     printf("%d ", (*out_offsets)[i]);
    // }
    // printf("\n");
    return 0;
}

void pq_init_reader(const char* file_name, std::shared_ptr<FileReader>* a_reader)
{
    std::string f_name(file_name);
    auto pool = ::arrow::default_memory_pool();

    // HDFS if starts with hdfs://
    if (f_name.find("hdfs://") == 0)
    {
        ::arrow::Status stat = ::arrow::io::HaveLibHdfs();
        if (!stat.ok())
        {
            std::cerr << "libhdfs not found" << '\n';
            return; // TODO: throw python exception
        }
        ::arrow::io::HdfsConnectionConfig hfs_config;

        // TODO: parse URI properly
        // remove hdfs://
        f_name = f_name.substr(strlen("hdfs://"));
        size_t col_char = f_name.find(':');
        if (col_char != std::string::npos)
        {
            hfs_config.host = f_name.substr(0, col_char);
            size_t slash_char = f_name.find('/');
            hfs_config.port = std::stoi(f_name.substr(col_char + 1, slash_char - col_char - 1));
            f_name = f_name.substr(slash_char);
            // std::cout << "host: " << hfs_config.host << std::endl;
            // std::cout << "port: " << hfs_config.port << std::endl;
            // std::cout << "file_name: " << f_name << std::endl;
        }
        else
        {
            hfs_config.host = std::string("default");
            hfs_config.port = 0;
        }
        hfs_config.driver = ::arrow::io::HdfsDriver::LIBHDFS;
        hfs_config.user = std::string("");
        hfs_config.kerb_ticket = std::string("");

        std::shared_ptr<::arrow::io::HadoopFileSystem> fs;
        ::arrow::io::HadoopFileSystem::Connect(&hfs_config, &fs);
        std::shared_ptr<::arrow::io::HdfsReadableFile> file;
        fs->OpenReadable(f_name, &file);
        std::unique_ptr<FileReader> arrow_reader;
        FileReader::Make(pool, ParquetFileReader::Open(file), &arrow_reader);
        *a_reader = std::move(arrow_reader);
    }
    else // regular file system
    {
        std::unique_ptr<FileReader> arrow_reader;
        FileReader::Make(pool, ParquetFileReader::OpenFile(f_name, false), &arrow_reader);
        *a_reader = std::move(arrow_reader);
    }
    // printf("file open for arrow reader done\n");
    // fflush(stdout);
    return;
}

// get type as enum values defined in arrow/cpp/src/arrow/type.h
// TODO: handle more complex types
std::shared_ptr<arrow::DataType> get_arrow_type(std::shared_ptr<FileReader> arrow_reader, int64_t column_idx)
{
    // TODO: error checking
    // std::vector<int> column_indices;
    // column_indices.push_back(column_idx);

    std::shared_ptr<::arrow::Schema> col_schema;
    // auto descr = arrow_reader->parquet_reader()->metadata()->schema();
    // auto parquet_key_value_metadata = arrow_reader->parquet_reader()->metadata()->key_value_metadata();
    // parquet::arrow::FromParquetSchema(descr, column_indices, parquet_key_value_metadata, &col_schema);
    arrow_reader->GetSchema(&col_schema);
    // std::cout<< col_schema->ToString() << std::endl;
    // std::shared_ptr<::arrow::DataType> arrow_dtype = col_schema->field(0)->type();
    std::shared_ptr<::arrow::DataType> arrow_dtype = col_schema->field(column_idx)->type();

    return arrow_dtype;
}

bool arrowPqTypesEqual(std::shared_ptr<arrow::DataType> arrow_type, ::parquet::Type::type pq_type)
{
    // TODO: remove parquet types, use HPAT Ctypes, handle more types
    if (arrow_type->id() == Type::BOOL && pq_type == ::parquet::Type::BOOLEAN)
        return true;
    if (arrow_type->id() == Type::UINT8 && pq_type == ::parquet::Type::BYTE_ARRAY)
        return true;
    if (arrow_type->id() == Type::INT8 && pq_type == ::parquet::Type::BYTE_ARRAY)
        return true;
    if (arrow_type->id() == Type::INT32 && pq_type == ::parquet::Type::INT32)
        return true;
    if (arrow_type->id() == Type::INT64 && pq_type == ::parquet::Type::INT64)
        return true;
    if (arrow_type->id() == Type::FLOAT && pq_type == ::parquet::Type::FLOAT)
        return true;
    if (arrow_type->id() == Type::DOUBLE && pq_type == ::parquet::Type::DOUBLE)
        return true;
    // XXX byte array is not always string?
    if (arrow_type->id() == Type::STRING && pq_type == ::parquet::Type::BYTE_ARRAY)
        return true;
    // TODO: add timestamp[ns]
    return false;
}

// similar to arrow/python/arrow_to_pandas.cc ConvertDatetimeNanos except with just buffer
// TODO: reuse from arrow
template <typename T, int64_t SHIFT>
inline void convertArrowToDT64(const uint8_t* buff, uint8_t* out_data, int64_t rows_to_skip, int64_t rows_to_read)
{
    int64_t* out_values = (int64_t*)out_data;
    const T* in_values = (const T*)buff;
    for (int64_t i = 0; i < rows_to_read; ++i)
    {
        *out_values++ = (static_cast<int64_t>(in_values[rows_to_skip + i]) * SHIFT);
    }
}

void append_bits_to_vec(
    std::vector<bool>* null_vec, const uint8_t* null_buff, int64_t null_size, int64_t offset, int64_t num_values)
{
    if (null_buff != nullptr && null_size > 0)
    {
        // to make packing portions of data easier, add data to vector in unpacked format then repack
        for (int64_t i = offset; i < offset + num_values; i++)
        {
            bool val = ::arrow::BitUtil::GetBit(null_buff, i);
            // printf("packing %d %d\n", i, (int)val);
            null_vec->push_back(val);
        }
        // null_vec->insert(null_vec->end(), null_buff, null_buff+null_size);
    }
}

void pack_null_bitmap(uint8_t** out_nulls, std::vector<bool>& null_vec, int64_t n_all_vals)
{
    if (null_vec.size() > 0)
    {
        int64_t n_bytes = (null_vec.size() + sizeof(uint8_t) - 1) / sizeof(uint8_t);
        *out_nulls = new uint8_t[n_bytes];
        memset(*out_nulls, 0, n_bytes);
        for (int64_t i = 0; i < n_all_vals; i++)
        {
            // printf("null %d %d\n", i, (int)null_vec[i]);
            if (null_vec[i])
                ::arrow::BitUtil::SetBit(*out_nulls, i);
        }
    }
    else
        *out_nulls = nullptr;
}

#include "mpi.h"
#include <Python.h>
#include <string>
#include <iostream>
#include <cstring>

#include "parquet/api/reader.h"
#include "parquet/arrow/reader.h"
#include "arrow/table.h"

using parquet::arrow::FileReader;
using parquet::ParquetFileReader;

int64_t pq_get_size(std::string* file_name, int64_t column_idx);
int pq_read(std::string* file_name, int64_t column_idx, void* out);
int pq_read_parallel(std::string* file_name, int64_t column_idx, uint8_t* out_data, int64_t start, int64_t count);

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

    return m;
}

int64_t pq_get_size(std::string* file_name, int64_t column_idx)
{
    auto pqreader = ParquetFileReader::OpenFile(*file_name);
    int64_t nrows = pqreader->metadata()->num_rows();
    // std::cout << nrows << std::endl;
    pqreader->Close();
    return nrows;
}

int pq_read(std::string* file_name, int64_t column_idx, void* out_data)
{
    auto pool = ::arrow::default_memory_pool();
    std::unique_ptr<FileReader> arrow_reader;
    arrow_reader.reset(new FileReader(pool,
        ParquetFileReader::OpenFile(*file_name, false)));
    //
    std::shared_ptr< ::arrow::Array > arr;
    arrow_reader->ReadColumn(column_idx, &arr);
    // std::cout << arr->ToString() << std::endl;

    auto buffers = arr->data()->buffers;
    // std::cout<<"num buffs: "<< buffers.size()<<std::endl;
    if (buffers.size()!=2) {
        std::cerr << "invalid parquet number of array buffers" << std::endl;
    }
    int64_t buff_size = buffers[1]->size();
    memcpy(out_data, buffers[1]->data(), buff_size);
    return 0;
}

int pq_read_parallel(std::string* file_name, int64_t column_idx, uint8_t* out_data, int64_t start, int64_t count)
{
    // printf("read parquet parallel column: %lld start: %lld count: %lld\n",
    //                                                 column_idx, start, count);
    // boolean, int32, int64, int96, float, double
    int type_sizes[] = {1, 4, 8, 12, 4, 8};

    auto pool = ::arrow::default_memory_pool();
    std::unique_ptr<FileReader> arrow_reader;
    arrow_reader.reset(new FileReader(pool,
        ParquetFileReader::OpenFile(*file_name, false)));

    int64_t n_row_groups = arrow_reader->parquet_reader()->metadata()->num_row_groups();
    std::vector<int> column_indices;
    column_indices.push_back(column_idx);

    int row_group_index = 0;
    int64_t skipped_rows = 0;
    int64_t read_rows = 0;


    auto rg_metadata = arrow_reader->parquet_reader()->metadata()->RowGroup(row_group_index);
    int64_t nrows_in_group = rg_metadata->ColumnChunk(column_idx)->num_values();
    int  dtype = rg_metadata->ColumnChunk(column_idx)->type();
    int dtype_size = type_sizes[dtype];

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
        /* ----------- read row group ------- */

        int64_t rows_to_skip = start - skipped_rows;
        int64_t rows_to_read = std::min(count-read_rows, nrows_in_group-rows_to_skip);
        // printf("rows_to_skip: %ld rows_to_read: %ld\n", rows_to_skip, rows_to_read);

        memcpy(out_data+read_rows*dtype_size, buff+rows_to_skip*dtype_size, rows_to_read*dtype_size);
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

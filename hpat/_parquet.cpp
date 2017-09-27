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

PyMODINIT_FUNC PyInit_parquet_cpp(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "parquet_cpp", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "read",
                            PyLong_FromVoidPtr((void*)(&pq_read)));
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
    std::shared_ptr<::arrow::Table> table;
    arrow_reader->ReadTable(&table);
    // std::vector<int> column_indices;
    // column_indices.push_back(column_idx);
    // arrow_reader->ReadRowGroup(0, column_indices, &table);
    std::shared_ptr< ::arrow::Column > column = table->column(column_idx);
    int64_t num_values = column->length();
    std::shared_ptr< ::arrow::ChunkedArray > data = column->data();
    for(int i=0; i<data->num_chunks(); i++) {
        std::shared_ptr< ::arrow::Array > arr = data->chunk(i);
        auto buffers = arr->data()->buffers;
        std::cout<<"num buffs: "<< buffers.size()<<std::endl;
        int buff_size = buffers[1]->size();
        memcpy(out_data, buffers[1]->data(), buff_size);
    }

    return 0;
}

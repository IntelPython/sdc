#include "mpi.h"
#include <Python.h>
#include <string>
#include <iostream>

#include "parquet/api/reader.h"
#include "parquet/arrow/reader.h"

using parquet::arrow::FileReader;
using parquet::ParquetFileReader;

void* pq_read(std::string* file_name);

PyMODINIT_FUNC PyInit_parquet_cpp(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "parquet_cpp", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "read",
                            PyLong_FromVoidPtr((void*)(&pq_read)));

    return m;
}

void* pq_read(std::string* file_name)
{
    //std::shared_ptr<::arrow::io::ReadableFile> handle;
    auto pool = ::arrow::default_memory_pool();
    std::unique_ptr<FileReader> arrow_reader;
    // std::unique_ptr<parquet::ParquetFileReader> p_reader =
    //     parquet::ParquetFileReader::OpenFile(*file_name, false);
    arrow_reader.reset(new FileReader(pool, ParquetFileReader::OpenFile(*file_name, false)));
    std::shared_ptr<::arrow::Table> table;
    arrow_reader->ReadTable(&table);
    return file_name;
}

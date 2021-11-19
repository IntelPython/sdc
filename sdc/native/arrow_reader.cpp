// *****************************************************************************
// Copyright (c) 2021, Intel Corporation All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//     Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
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
// *****************************************************************************

#include <Python.h>
#include <iostream>
#include <string_view>
#include <algorithm>
#include <vector>
#include <numeric>
#include <charconv>
#include "arrow/csv/api.h"
#include "arrow/table.h"
#include "arrow/type.h"
#include "arrow/io/file.h"
#include "arrow/array/array_primitive.h"
#include "arrow/array/array_binary.h"
#include "arrow/python/pyarrow.h"
#include "numba/core/runtime/nrt_external.h"


#define REGISTER(func) PyObject_SetAttrString(m, #func, PyLong_FromVoidPtr((void*)(&func)));

enum class ArrowCellReadResult : int8_t {
    CELL_READ_OK = 0,
    CELL_READ_NAN_VALUE,
    CELL_READ_INTERNAL_ERROR,
};

struct ArrowChunkedTable {
    std::shared_ptr<arrow::Table> sp_table;
    size_t n_chunks;
    size_t n_rows;
    size_t n_cols;

    std::vector<size_t> chunk_offsets;
    std::vector<arrow::Array*> column_chunk_ptrs;

    ArrowChunkedTable(const std::shared_ptr<arrow::Table>& _sp_table) : sp_table(_sp_table) {

        n_rows = sp_table->num_rows();
        n_cols = sp_table->num_columns();

        // XXX: this assumes that n_chunks is same for all columns
        n_chunks = sp_table->column(0)->num_chunks();
        chunk_offsets = std::vector<size_t>(n_chunks);
        column_chunk_ptrs = std::vector<arrow::Array*>(n_cols * n_chunks);

        for (int col_idx=0; col_idx < n_cols; ++col_idx) {

            for (int i=0; i < n_chunks; ++i) {
                auto chunk = sp_table->column(col_idx)->chunk(i);
                column_chunk_ptrs[col_idx * n_chunks + i] = chunk.get();
                if (col_idx == 0)
                    chunk_offsets[i] = chunk->length();
            }
        }

        std::partial_sum(chunk_offsets.begin(), chunk_offsets.end(), chunk_offsets.begin());
    }

    ArrowChunkedTable(const ArrowChunkedTable& rhs) {
        n_chunks = rhs.n_chunks;
        n_rows = rhs.n_rows;
        n_cols = rhs.n_cols;
        chunk_offsets = rhs.chunk_offsets;
        sp_table = rhs.sp_table;
        column_chunk_ptrs = rhs.column_chunk_ptrs;
    }

    ~ArrowChunkedTable() = default;
    ArrowChunkedTable& operator=(const ArrowChunkedTable& rhs) = default;
    ArrowChunkedTable(ArrowChunkedTable&& rhs) = delete;
    ArrowChunkedTable& operator=(ArrowChunkedTable&&) = delete;
};

struct PyarrowInit {
    PyarrowInit() {
        auto res = arrow::py::import_pyarrow();
    }
    ~PyarrowInit() = default;
    PyarrowInit(const PyarrowInit& rhs) = default;
};

static PyarrowInit& initialize_pyarrow_once() {
    static PyarrowInit s;
    return s;
}


extern "C"
{

void delete_arrow_chunked_table(void* p_chunked_table)
{
    ArrowChunkedTable* p_table_spec = reinterpret_cast<ArrowChunkedTable*>(p_chunked_table);
    delete p_table_spec;
}

void create_arrow_table(void* pyarrow_table,
                          NRT_MemInfo** meminfo,
                          void* nrt_table)
{
    auto pa_init = initialize_pyarrow_once();
    auto nrt = (NRT_api_functions*)nrt_table;

    auto p_pyarrow_table = (PyObject*)pyarrow_table;
    auto maybe_table = arrow::py::unwrap_table(p_pyarrow_table);
    if (!maybe_table.ok()) {
        std::cerr << "Unwrapping Arrow table from pyobject failed" << std::endl;
    }

    std::shared_ptr<arrow::Table> sp_table = *maybe_table;
    auto p_table = new ArrowChunkedTable(sp_table);
    (*meminfo) = nrt->manage_memory((void*)p_table, delete_arrow_chunked_table);
}

int64_t get_table_len(void* p_table)
{
    auto p_chunked_table = (reinterpret_cast<ArrowChunkedTable*>(p_table));
    auto res = p_chunked_table->n_rows;
    return res;
}

int8_t get_table_cell(void* p_table, int64_t col_idx, int64_t row_idx, void* p_res)
{
    ArrowCellReadResult ret_code;
    auto p_chunked_table = (reinterpret_cast<ArrowChunkedTable*>(p_table));
    auto& p_chunk_offsets = p_chunked_table->chunk_offsets;
    auto n_chunks = p_chunked_table->n_chunks;
    auto col_size = p_chunked_table->n_rows;

    // locates the chunk where row_idx resides, and computes the offset relative to chunk start
    auto chunk_it = std::upper_bound(p_chunk_offsets.begin(), p_chunk_offsets.end(), row_idx);
    size_t target_chunk = std::distance(p_chunk_offsets.begin(), chunk_it);
    size_t prev_bound = target_chunk != 0 ? p_chunk_offsets[target_chunk-1] : 0;
    auto new_row_idx = row_idx - prev_bound;

    auto sp_target_chunk = p_chunked_table->column_chunk_ptrs[col_idx * n_chunks + target_chunk];
    auto arr_type_id = sp_target_chunk->type_id();
    switch (arr_type_id) {
        case arrow::Type::STRING: {
            auto p_column = reinterpret_cast<arrow::StringArray*>(sp_target_chunk); // std::static_pointer_cast<arrow::StringArray>(sp_target_chunk);
            auto arrow_str_view = p_column->GetView(new_row_idx);
            auto p_res_spec = (std::string_view*)p_res;
            *p_res_spec = std::string_view(arrow_str_view.data(), arrow_str_view.size());
            ret_code = ArrowCellReadResult::CELL_READ_OK;
            break;
        }

        // this is not used in converters (they read columns as strings),
        // TO-DO: extend if operating on Arrow Tables with all data types is needed
        case arrow::Type::INT64: {
            auto p_column = reinterpret_cast<arrow::Int64Array*>(sp_target_chunk);  // std::static_pointer_cast<arrow::Int64Array>(sp_target_chunk);
            auto p_res_spec = (int64_t*)p_res;
            if (p_column->IsValid(new_row_idx)) {
                *p_res_spec = p_column->Value(new_row_idx);
                ret_code = ArrowCellReadResult::CELL_READ_OK;
            } else {
                ret_code = ArrowCellReadResult::CELL_READ_NAN_VALUE;
            }
            break;
        }

        default: {
            ret_code = ArrowCellReadResult::CELL_READ_INTERNAL_ERROR;
            break;
        }
    }

    return (int8_t)ret_code;
}


PyMODINIT_FUNC PyInit_harrow_reader()
{
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "harrow_reader",
        "No docs",
        -1,
        NULL,
    };
    PyObject* m = PyModule_Create(&moduledef);
    if (m == NULL)
    {
        return NULL;
    }

    REGISTER(create_arrow_table)
    REGISTER(get_table_len)
    REGISTER(get_table_cell)

    return m;
}

}  // extern "C"

#undef REGISTER

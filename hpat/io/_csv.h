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

#ifndef _CSV_H_INCLUDED
#define _CSV_H_INCLUDED

#include <Python.h>
#include <string>

// CSV exports some stuff to the io module
extern "C" void PyInit_csv(PyObject*);

/**
 * Split file into chunks and return a file-like object per rank. The returned object
 * represents the data to be read on each process.
 *
 * @param[in]  f   the input file name
 * @param[in]  is_parallel   if parallel read of different chunks required
 * @return     HPATIO file-like object to read the owned chunk through pandas.read_csv
 **/
extern "C" PyObject* csv_file_chunk_reader(const char* fname, bool is_parallel, int64_t skiprows, int64_t nrows);

/**
 * Split string into chunks and return a file-like object per rank. The returned object
 * represents the data to be read on each process.
 *
 * @param[in]  f   the input string
 * @param[in]  is_parallel   if parallel read of different chunks required
 * @return     HPATIO file-like object to read the owned chunk through pandas.read_csv
 **/
extern "C" PyObject* csv_string_chunk_reader(const std::string* str, bool is_parallel);

#endif // _CSV_H_INCLUDED

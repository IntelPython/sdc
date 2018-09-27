#ifndef _CSV_H_INCLUDED
#define _CSV_H_INCLUDED

#include <Python.h>
#include <string>

// CSV exports some stuff to the io module
extern "C" void PyInit_csv(PyObject *);

/**
 * Split file into chunks and return a file-like object per rank. The returned object
 * represents the data to be read on each process.
 *
 * @param[in]  f   the input file name
 **/
extern "C" PyObject* csv_file_chunk_reader(const std::string * fname);

/**
 * Split string into chunks and return a file-like object per rank. The returned object
 * represents the data to be read on each process.
 *
 * @param[in]  f   the input string
 **/
extern "C" PyObject* csv_string_chunk_reader(const std::string * str);

#endif // _CSV_H_INCLUDED

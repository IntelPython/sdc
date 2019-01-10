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
 * @param[in]  is_parallel   if parallel read of different chunks required
 * @return     HPATIO file-like object to read the owned chunk through pandas.read_csv
 **/
extern "C" PyObject* csv_file_chunk_reader(const char * fname, bool is_parallel);

/**
 * Split string into chunks and return a file-like object per rank. The returned object
 * represents the data to be read on each process.
 *
 * @param[in]  f   the input string
 * @param[in]  is_parallel   if parallel read of different chunks required
 * @return     HPATIO file-like object to read the owned chunk through pandas.read_csv
 **/
extern "C" PyObject* csv_string_chunk_reader(const std::string * str, bool is_parallel);

#endif // _CSV_H_INCLUDED

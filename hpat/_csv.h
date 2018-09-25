#ifndef _CSV_H_INCLUDED
#define _CSV_H_INCLUDED

#include <Python.h>
#include <string>

// CSV exports some stuff to the io module
extern void PyInit_csv(PyObject *);

extern void ** csv_read_file(const std::string * fname, size_t * cols_to_read, int * dtypes, size_t n_cols_to_read,
                             size_t * first_row, size_t * n_rows,
                             std::string * delimiters, std::string * quotes);

#endif // _CSV_H_INCLUDED

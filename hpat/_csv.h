#ifndef _CSV_H_INCLUDED
#define _CSV_H_INCLUDED

#include <Python.h>
#include <string>

// CSV exports some stuff to the io module
extern void PyInit_csv(PyObject *);

/**
   Read a CSV file.
   Returned pointer must be deallocated by calling csv_delete using the same n_cols_to_read.
   All lines have the same number of tokens/columns.
   Otherwise some lines might be duplicated or skipped in a distributed setup.
   Input parameters must be identical on all ranks, dead-lock can occur otherwise.

   @param[in] fname           Name of the CSV file
   @param[in] cols_to_read    Array of column indices (size_t) to be read
                              If NULL, all columns will be read
   @param[in] dtypes          Array of data type codes (from numpy)
                              Expects array of size n_cols_to_read.
                              Must not be NULL.
   @param[in] n_cols_to_read  Number of columns to read.
   @param[out] first_row_read On success (if data is returned) this will be set to first line number read by this process.
   @param[out] n_rows_read    On success (if data is returned) this will be set to number of lines read by this process.
   @param[in]  delimiters     Any character in the string is considered to be a separator.
   @param[in]  quotes         Any character in the string is considered to be a quote.
   @return 2d-array: array of n_cols_to_read arrays, each of given data type (dtypes)
                     array of NULL pointers if no data was read
                     NULL if an error ocurred.
 **/
extern "C" void * csv_read_file(const std::string * fname, size_t * cols_to_read, int64_t * dtypes, size_t n_cols_to_read,
                                size_t * first_row, size_t * n_rows,
                                std::string * delimiters, std::string * quotes);

#endif // _CSV_H_INCLUDED

/*
  SPMD CSV reader.
*/
#include <mpi.h>
#include <cstdint>
#include <cinttypes>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <boost/tokenizer.hpp>
#include <boost/filesystem/operations.hpp>

#include "_datetime_ext.h"
#include "_distributed.h"
#include "_import_py.h"
#include "_csv.h"

#include <Python.h>
#include "structmember.h"

#if PY_MAJOR_VERSION < 3
#define BUFF_TYPE Py_UNICODE
#define BUFF2UC(_o, _s) PyUnicode_FromUnicode(_o, _s)
#else
#define BUFF_TYPE Py_UCS4
#define BUFF2UC(_o, _s) PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, _o, _s)
#endif

typedef struct {
    PyObject_HEAD
    /* Your internal buffer, size and pos */
    std::istream * ifs;
    size_t chunk_start;
    size_t chunk_size;
    size_t chunk_pos;
    std::vector<char> buf;
} hpatio;

static void
hpatio_dealloc(hpatio* self)
{
    if(self->ifs) delete self->ifs;
    Py_TYPE(self)->tp_free(self);
    /* we do not own the buffer, we only ref to it; nothing else to be done */
}

static PyObject *
hpatio_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    std::cout << "new..." << std::endl;
    hpatio *self;

    self = (hpatio *)type->tp_alloc(type, 0);
    self->ifs = NULL;
    self->chunk_start = 0;
    self->chunk_size = 0;
    self->chunk_pos = 0;

    return (PyObject *)self;
}

static int hpatio_pyinit(PyObject *self, PyObject *args, PyObject *kwds)
{
    std::cout << "init..." << args << std::endl;

    char* str = NULL;
    Py_ssize_t count = 0;

    if(!PyArg_ParseTuple(args, "|z#", &str, &count) || str == NULL) {
        if(PyErr_Occurred()) PyErr_Print();
        return 0;
    }

    ((hpatio*)self)->chunk_start = 0;
    ((hpatio*)self)->chunk_pos = 0;
    ((hpatio*)self)->ifs = new std::istringstream(str);
    if(!((hpatio*)self)->ifs->good()) {
        std::cerr << "Could not create istrstream from string.\n";
        ((hpatio*)self)->chunk_size = 0;
        return -1;
    }
    ((hpatio*)self)->chunk_size = count;

    return 0;
}

static void
hpatio_init(hpatio *self, std::istream * ifs, size_t start, size_t sz)
{
    if(!ifs) {
        std::cerr << "Can't handle NULL pointer as input stream.\n";
        return;
    }
    self->ifs = ifs;
    if(!self->ifs->good() || self->ifs->eof()) {
        std::cerr << "Got bad istream in initializing HPATIO object." << std::endl;
        return;
    }
    self->ifs->seekg(start, std::ios_base::beg);
    if(!self->ifs->good() || self->ifs->eof()) {
        std::cerr << "Could not seek to start position " << start << std::endl;
        return;
    }
    self->chunk_start = start;
    self->chunk_size = sz;
    self->chunk_pos = 0;
}

static PyObject *
hpatio_read(hpatio* self, PyObject *args)
{
    /* taken from CPython stringio.c */
    std::cout << "reading..." << std::endl;

    if(self->ifs == NULL) {
        PyErr_SetString(PyExc_ValueError, "I/O operation on uninitialized HPATIO object");
        return NULL;
    }

    Py_ssize_t size, n;

    PyObject *arg = Py_None;
    if (!PyArg_ParseTuple(args, "|O:read", &arg)) {
        return NULL;
    }
    if (PyNumber_Check(arg)) {
        size = PyNumber_AsSsize_t(arg, PyExc_OverflowError);
        if (size == -1 && PyErr_Occurred()) {
            return NULL;
        }
    }
    else if (arg == Py_None) {
        /* Read until EOF is reached, by default. */
        size = -1;
    }
    else {
        PyErr_Format(PyExc_TypeError, "integer argument expected, got '%s'",
                     Py_TYPE(arg)->tp_name);
        return NULL;
    }

    /* adjust invalid sizes */
    n = self->chunk_size - self->chunk_pos;
    if(size < 0 || size > n) {
        size = n;
        if (size < 0) size = 0;
    }

    self->buf.resize(size);
    self->ifs->read(self->buf.data(), size);
    self->chunk_pos += size;
    if(!*self->ifs) {
        std::cerr << "Failed reading " << size << " bytes";
        return NULL;
    }

    std::cout << "read " << size << " bytes" << std::endl;

    return PyUnicode_FromStringAndSize(self->buf.data(), size);
}

static PyObject *
hpatio_iternext(PyObject *self)
{
    std::cerr << "iternext not implemented";
    return NULL;
};

static PyMethodDef hpatio_methods[] = {
    {"read", (PyCFunction)hpatio_read, METH_VARARGS,
     "Read at most n characters, returned as a string.",
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject hpatio_type = {
    PyObject_HEAD_INIT(NULL)
    "hpat.hio.HPATIO",         /*tp_name*/
    sizeof(hpatio),            /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)hpatio_dealloc,/*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,/*tp_flags*/
    "hpatio objects",          /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    hpatio_iternext,           /* tp_iter */
    hpatio_iternext,           /* tp_iternext */
    hpatio_methods,            /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    hpatio_pyinit,             /* tp_init */
    0,                         /* tp_alloc */
    hpatio_new,                /* tp_new */
};

void PyInit_csv(PyObject * m)
{
    if(PyType_Ready(&hpatio_type) < 0) return;
    Py_INCREF(&hpatio_type);
    PyModule_AddObject(m, "HPATIO", (PyObject *)&hpatio_type);
}

// ***********************************************************************************
// ***********************************************************************************

#define CHECK(expr, msg) if(!(expr)){std::cerr << "Error in csv_read: " << msg << std::endl; return NULL;}

/// return vector of offsets of newlines in first n bytes of given stream
static std::vector<size_t> count_lines(std::istream * f, size_t n)
{
    std::vector<size_t> pos;
    char c;
    size_t i=0;
    while(i<n && f->get(c)) {
        if(c == '\n') pos.push_back(i);
        ++i;
    }
    if(i<n) std::cerr << "Warning, read only " << i << " bytes out of " << n << "requested\n";
    return pos;
}

/** 
 * Split stream into chunks and return a file-like object per rank. The returned object
 * represents the data to be read on each process.
 *
 * We evenly distribute by number of lines by working on byte-chunks in parallel
 *   * counting new-lines and allreducing and exscaning numbers
 *   * computing start/end points of desired chunks-of-lines and sending them to corresponding ranks.
 * If total number is not a multiple of number of ranks the first ranks get an extra line.
 *
 * @param[in]  f   the input stream
 * @param[in]  fsz total number of bytes in stream
 * @param[out] first_row if not NULL will be set to global number of first row in this chunk
 * @param[out] n_rows    if not NULL will be set to number of rows in this chunk
 **/
static PyObject* csv_get_chunk(std::istream * f, size_t fsz, size_t * first_row, size_t * n_rows)
{
    size_t rank = hpat_dist_get_rank();
    size_t nranks = hpat_dist_get_size();

    // We evenly distribute the 'data' byte-wise
    auto chunksize = fsz/nranks;
    // seek to our chunk
    f->seekg(chunksize*rank, std::ios_base::beg);
    if(!f->good() || f->eof()) {
        std::cerr << "Could not seek to start position " << chunksize*rank << std::endl;
        return NULL;
    }
    // count number of lines in chunk
    std::vector<size_t> line_offset = count_lines(f, chunksize);
    size_t no_lines = line_offset.size();
    size_t my_off_start = 0;
    size_t my_off_end = fsz;
    size_t exp_no_lines = no_lines;
    size_t extra_no_lines = 0;

    if(nranks > 1) {
        // get total number of lines using allreduce
        size_t tot_no_lines = no_lines;
        
        hpat_dist_reduce(reinterpret_cast<char *>(no_lines), reinterpret_cast<char *>(tot_no_lines), MPI_SUM, HPAT_CTypes::UINT64);
        // evenly divide
        exp_no_lines = tot_no_lines/nranks;
        // surplus lines added to first ranks
        extra_no_lines = tot_no_lines-(exp_no_lines*nranks);
        
        // Now we need to communicate the distribution as we really want it
        // First determine which is our first line (which is the sum of previous lines)
        size_t byte_first_line = hpat_dist_exscan_i8(no_lines);
        size_t byte_last_line = byte_first_line + no_lines;
        
        // We now determine the chunks of lines that begin and end in our byte-chunk
        
        // issue IRecv calls, eventually receiving start and end offsets of our line-chunk
        const int START_OFFSET = 47011;
        const int END_OFFSET = 47012;
        std::vector<MPI_Request> mpi_reqs;
        mpi_reqs.push_back(hpat_dist_irecv(&my_off_start, 1, HPAT_CTypes::UINT64, MPI_ANY_SOURCE, START_OFFSET, rank>0));
        mpi_reqs.push_back(hpat_dist_irecv(&my_off_end, 1, HPAT_CTypes::UINT64, MPI_ANY_SOURCE, END_OFFSET, rank<(nranks-1)));

        size_t i_start = 0;
        for(size_t i=0; i<nranks; ++i) {
            // if start is on our byte-chunk, send stream-offset to rank i
            // Note our line_offsets mark the end of each line!
            if(i_start > byte_first_line && i_start <= byte_last_line) {
                size_t i_off = line_offset[i_start-byte_first_line-1]+1; // +1 to skip leading newline
                mpi_reqs.push_back(hpat_dist_isend(&i_off, 1, HPAT_CTypes::UINT64, i, START_OFFSET, true));
            }
            // if end is on our byte-chunk, send stream-offset to rank i
            size_t i_end = i_start + exp_no_lines + (i < extra_no_lines ? 1 : 0);
            if(i_end > byte_first_line && i_end <= byte_last_line && i < (nranks-1)) {
                size_t i_off = line_offset[i_end-byte_first_line-1]+1; // +1 to include trailing newline
                mpi_reqs.push_back(hpat_dist_isend(&i_off, 1, HPAT_CTypes::UINT64, i, END_OFFSET, true));
            }
            i_start = i_end;
        }
        // before reading, make sure we received our start/end offsets
        hpat_dist_waitall(mpi_reqs.size(), mpi_reqs.data());
    } // ranks>1

    // Here we now know exactly what chunk to read: [my_off_start,my_off_end[
    // let's create our file-like reader
    auto gilstate = PyGILState_Ensure();
    PyObject * reader = PyObject_CallFunctionObjArgs((PyObject *) &hpatio_type, NULL);
    PyGILState_Release(gilstate);
    if(reader == NULL || PyErr_Occurred()) {
        PyErr_Print();
        std::cerr << "Could not create chunk reader object" << std::endl;
        if(reader) delete reader;
        reader = NULL;
    } else {
        hpatio_init(reinterpret_cast<hpatio*>(reader), f, my_off_start, my_off_end-my_off_start);
        if(first_row) *first_row = rank*exp_no_lines + (rank < extra_no_lines ? 1 : 0);
        if(n_rows) *n_rows = (rank+1)*exp_no_lines + (rank < extra_no_lines ? 1 : 0);
    }

    return reader;
}

/*
  This is a wrapper around pandas.read_csv.
  Doesn't do much except calling pandas.
  Always return NULL for now.

  Comments below are outdated.

  We divide the file into chunks, one for each process.
  Lines generally have different lengths, so we might start int he middle of the line.
  Hence, we determine the number cols in the file and check if our first line is a full line.
  If not, we just throw it away.
  Similary, our chunk might not end at a line boundary.
  Note that we assume all lines have the same numer of tokens/columns.
  We always read full lines until we read at least our chunk-size.
  This makes sure there are no gaps and no data duplication.
 */
static PyObject* csv_read(std::istream * f, size_t fsz, size_t * cols_to_read, int64_t * dtypes, size_t n_cols_to_read,
                          size_t * first_row, size_t * n_rows,
                          std::string * delimiters = NULL, std::string * quotes = NULL)
{
    CHECK(dtypes, "Input parameter dtypes must not be NULL.");
    CHECK(first_row && n_rows, "Output parameters first_row and n_rows must not be NULL.");

    PyObject * reader = csv_get_chunk(f, fsz, first_row, n_rows);
    if(reader == 0) return NULL;

    // Now call pandas.read_csv.
    PyObject * df = NULL;
    auto gilstate = PyGILState_Ensure();
    PyObject * pd_read_csv = import_sym("pandas", "read_csv");
    if (pd_read_csv == NULL || PyErr_Occurred()) {
        PyErr_Print();
        std::cerr << "Could not get pandas.read_csv " << std::endl;
        pd_read_csv = NULL;
    } else {
        df = PyObject_CallFunctionObjArgs(pd_read_csv, reader, NULL);
        Py_XDECREF(pd_read_csv);
    }
    PyGILState_Release(gilstate);

    if (df == NULL || PyErr_Occurred()) {
        PyErr_Print();
        std::cerr << "pandas.read_csv failed." << std::endl;
        return NULL;
    }

    std::cout << "Done!" << std::endl;
    return df;
}


extern "C" void * csv_read_file(const std::string * fname, size_t * cols_to_read, int64_t * dtypes, size_t n_cols_to_read,
                                size_t * first_row, size_t * n_rows,
                                std::string * delimiters, std::string * quotes)
{
    CHECK(fname != NULL, "NULL filename provided.");
    // get total file-size
    auto fsz = boost::filesystem::file_size(*fname);
    std::ifstream * f = new  std::ifstream(*fname);
    CHECK(f->good() && !f->eof() && f->is_open(), "could not open file.");
    return reinterpret_cast<void*>(csv_read(f, fsz, cols_to_read, dtypes, n_cols_to_read, first_row, n_rows, delimiters, quotes));
}


extern "C" PyObject * csv_read_string(const std::string * str, size_t * cols_to_read, int64_t * dtypes, size_t n_cols_to_read,
                                      size_t * first_row, size_t * n_rows,
                                      std::string * delimiters, std::string * quotes)
{
    CHECK(str != NULL, "NULL string provided.");
    // get total file-size
    std::istringstream * f = new std::istringstream(*str);
    CHECK(f->good(), "could not create istrstream from string.");
    return csv_read(f, str->size(), cols_to_read, dtypes, n_cols_to_read, first_row, n_rows, delimiters, quotes);
}

#undef CHECK

// ***********************************************************************************
// ***********************************************************************************

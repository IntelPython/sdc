/*
  SPMD Stream and CSV reader.

  We provide a Python object that is file-like in the Pandas sense
  and can so be used as the input argument to CSV read.
  When called in a parallel/distributed setup, each process owns a
  chunk of the csv file only. The chunks are balanced by number of
  lines (not neessarily number of bytes). The actual file read is
  done lazily in the objects read method.
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
#include "_csv.h"

#include <Python.h>
#include "structmember.h"


// ***********************************************************************************
// Our file-like object for reading chunks in a std::istream
// ***********************************************************************************

typedef struct {
    PyObject_HEAD
    /* Your internal buffer, size and pos */
    std::istream * ifs;    // input stream
    size_t chunk_start;    // start of our chunk
    size_t chunk_size;     // size of our chunk
    size_t chunk_pos;      // current position in our chunk
    std::vector<char> buf; // internal buffer for converting stream input to Unicode object
} stream_reader;


static void stream_reader_dealloc(stream_reader* self)
{
    // we own the stream!
    if(self->ifs) delete self->ifs;
    Py_TYPE(self)->tp_free(self);
}


// alloc a HPTAIO object
static PyObject * stream_reader_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    stream_reader *self = (stream_reader *)type->tp_alloc(type, 0);
    if(PyErr_Occurred()) {
        PyErr_Print();
        return NULL;
    }
    self->ifs = NULL;
    self->chunk_start = 0;
    self->chunk_size = 0;
    self->chunk_pos = 0;

    return (PyObject *)self;
}


// we provide this mostly for testing purposes
// users are not supposed to use this
static int stream_reader_pyinit(PyObject *self, PyObject *args, PyObject *kwds)
{
    char* str = NULL;
    Py_ssize_t count = 0;

    if(!PyArg_ParseTuple(args, "|z#", &str, &count) || str == NULL) {
        if(PyErr_Occurred()) PyErr_Print();
        return 0;
    }

    ((stream_reader*)self)->chunk_start = 0;
    ((stream_reader*)self)->chunk_pos = 0;
    ((stream_reader*)self)->ifs = new std::istringstream(str);
    if(!((stream_reader*)self)->ifs->good()) {
        std::cerr << "Could not create istrstream from string.\n";
        ((stream_reader*)self)->chunk_size = 0;
        return -1;
    }
    ((stream_reader*)self)->chunk_size = count;

    return 0;
}


// We use this (and not the above) from C to init our StreamReader object
// Will seek to chunk beginning
static void stream_reader_init(stream_reader *self, std::istream * ifs, size_t start, size_t sz)
{
    if(!ifs) {
        std::cerr << "Can't handle NULL pointer as input stream.\n";
        return;
    }
    self->ifs = ifs;
    if(!self->ifs->good() || self->ifs->eof()) {
        std::cerr << "Got bad istream in initializing StreamReader object." << std::endl;
        return;
    }
    // seek to our chunk beginning
    self->ifs->seekg(start, std::ios_base::beg);
    if(!self->ifs->good() || self->ifs->eof()) {
        std::cerr << "Could not seek to start position " << start << std::endl;
        return;
    }
    self->chunk_start = start;
    self->chunk_size = sz;
    self->chunk_pos = 0;
}


// read given number of bytes from our chunk and return a Unicode Object
// returns NULL if an error occured.
// does not read beyond end of our chunk (even if file continues)
static PyObject * stream_reader_read(stream_reader* self, PyObject *args)
{
    // partially copied from from CPython's stringio.c

    if(self->ifs == NULL) {
        PyErr_SetString(PyExc_ValueError, "I/O operation on uninitialized StreamReader object");
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

    return PyUnicode_FromStringAndSize(self->buf.data(), size);
}


// Needed to make Pandas accept it, never used
static PyObject * stream_reader_iternext(PyObject *self)
{
    std::cerr << "iternext not implemented";
    return NULL;
};


// our class has only one method
static PyMethodDef stream_reader_methods[] = {
    {"read", (PyCFunction)stream_reader_read, METH_VARARGS,
     "Read at most n characters, returned as a unicode.",
    },
    {NULL}  /* Sentinel */
};


// the actual Python type class
static PyTypeObject stream_reader_type = {
    PyObject_HEAD_INIT(NULL)
    "hpat.hio.StreamReader",   /*tp_name*/
    sizeof(stream_reader),     /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)stream_reader_dealloc,/*tp_dealloc*/
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
    "stream_reader objects",   /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    stream_reader_iternext,    /* tp_iter */
    stream_reader_iternext,    /* tp_iternext */
    stream_reader_methods,     /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    stream_reader_pyinit,      /* tp_init */
    0,                         /* tp_alloc */
    stream_reader_new,                /* tp_new */
};


// at module load time we need to make our type known ot Python
extern "C" void PyInit_csv(PyObject * m)
{
    if(PyType_Ready(&stream_reader_type) < 0) return;
    Py_INCREF(&stream_reader_type);
    PyModule_AddObject(m, "StreamReader", (PyObject *)&stream_reader_type);
    PyObject_SetAttrString(m, "csv_file_chunk_reader",
                           PyLong_FromVoidPtr((void*)(&csv_file_chunk_reader)));
    PyObject_SetAttrString(m, "csv_string_chunk_reader",
                           PyLong_FromVoidPtr((void*)(&csv_string_chunk_reader)));
}


// ***********************************************************************************
// C interface for getting the file-like chunk reader
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
 * Using hpat_dist_get_size and hpat_dist_get_start to compute chunk start/end/size as well as
 * the final chunking of lines.
 *
 * @param[in]  f   the input stream
 * @param[in]  fsz total number of bytes in stream
 * @return     StreamReader file-like object to read the owned chunk through pandas.read_csv
 **/
static PyObject* csv_chunk_reader(std::istream * f, size_t fsz, bool is_parallel)
{
    size_t nranks = hpat_dist_get_size();

    size_t my_off_start = 0;
    size_t my_off_end = fsz;

    if(is_parallel && nranks > 1) {
        size_t rank = hpat_dist_get_rank();

        // seek to our chunk
        size_t byte_offset = hpat_dist_get_start(fsz, nranks, rank);
        f->seekg(byte_offset, std::ios_base::beg);
        if(!f->good() || f->eof()) {
            std::cerr << "Could not seek to start position " << hpat_dist_get_start(fsz, nranks, rank) << std::endl;
            return NULL;
        }
        // We evenly distribute the 'data' byte-wise
        // count number of lines in chunk
        std::vector<size_t> line_offset = count_lines(f, hpat_dist_get_node_portion(fsz, nranks, rank));
        size_t no_lines = line_offset.size();
        // get total number of lines using allreduce
        size_t tot_no_lines(0);

        hpat_dist_reduce(reinterpret_cast<char *>(&no_lines), reinterpret_cast<char *>(&tot_no_lines), HPAT_ReduceOps::SUM, HPAT_CTypes::UINT64);

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

        // We iterate through chunk boundaries (defined by line-numbers)
        // we start with boundary 1 as 0 is the beginning of file
        for(int i=1; i<nranks; ++i) {
            size_t i_bndry = hpat_dist_get_start(tot_no_lines, (int)nranks, i);
            // Note our line_offsets mark the end of each line!
            // we check if boundary is on our byte-chunk
            if(i_bndry > byte_first_line && i_bndry <= byte_last_line) {
                // if so, send stream-offset to ranks which start/end here
                size_t i_off = byte_offset + line_offset[i_bndry-byte_first_line-1]+1; // +1 to skip/include leading/trailing newline
                // send to rank that starts at this boundary: i
                mpi_reqs.push_back(hpat_dist_isend(&i_off, 1, HPAT_CTypes::UINT64, i, START_OFFSET, true));
                // send to rank that ends at this boundary: i-1
                mpi_reqs.push_back(hpat_dist_isend(&i_off, 1, HPAT_CTypes::UINT64, i-1, END_OFFSET, true));
            } else {
                // if not and we past our chunk -> we stop
                if(i_bndry > byte_last_line) break;
            } // else we are before our chunk -> continue iteration
        }
        // before reading, make sure we received our start/end offsets
        hpat_dist_waitall(mpi_reqs.size(), mpi_reqs.data());
    } // if is_parallel

    // Here we now know exactly what chunk to read: [my_off_start,my_off_end[
    // let's create our file-like reader
    auto gilstate = PyGILState_Ensure();
    PyObject * reader = PyObject_CallFunctionObjArgs((PyObject *) &stream_reader_type, NULL);
    PyGILState_Release(gilstate);
    if(reader == NULL || PyErr_Occurred()) {
        PyErr_Print();
        std::cerr << "Could not create chunk reader object" << std::endl;
        if(reader) delete reader;
        reader = NULL;
    } else {
        stream_reader_init(reinterpret_cast<stream_reader*>(reader), f, my_off_start, my_off_end-my_off_start);
    }

    return reader;
}


// taking a file to create a istream and calling csv_chunk_reader
extern "C" PyObject* csv_file_chunk_reader(const char * fname, bool is_parallel)
{
    CHECK(fname != NULL, "NULL filename provided.");
    // get total file-size
    auto fsz = boost::filesystem::file_size(fname);
    std::ifstream * f = new std::ifstream(fname);
    CHECK(f->good() && !f->eof() && f->is_open(), "could not open file.");
    return csv_chunk_reader(f, fsz, is_parallel);
}


// taking a string to create a istream and calling csv_chunk_reader
extern "C" PyObject* csv_string_chunk_reader(const std::string * str, bool is_parallel)
{
    CHECK(str != NULL, "NULL string provided.");
    // get total file-size
    std::istringstream * f = new std::istringstream(*str);
    CHECK(f->good(), "could not create istrstream from string.");
    return csv_chunk_reader(f, str->size(), is_parallel);
}

#undef CHECK

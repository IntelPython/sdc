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
    Py_ssize_t string_size;
    Py_ssize_t pos;
    char *buf;
} hpatio;

static void
hpatio_dealloc(hpatio* self)
{
    Py_TYPE(self)->tp_free(self);
    /* we do not own the buffer, we only ref to it; nothing else to be done */
}

static PyObject *
hpatio_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    std::cout << "new..." << std::endl;
    hpatio *self;

    self = (hpatio *)type->tp_alloc(type, 0);
    self->string_size = 0;
    self->pos = 0;
    self->buf = NULL;

    return (PyObject *)self;
}

static int hpatio_pyinit(PyObject *self, PyObject *args, PyObject *kwds)
{
    std::cout << "init..." << args << std::endl;

    char* str = NULL;
    Py_ssize_t count = 0;
    PyObject *arg = Py_None;

    if(!PyArg_ParseTuple(args, "|z#", &str, &count)) {
        return -1;
    }

    ((hpatio*)self)->buf = str;
    ((hpatio*)self)->string_size = count;

    return 0;
}

static void
hpatio_init(hpatio *self, char * buf, size_t sz)
{
    self->string_size = sz;
    self->pos = 0;
    self->buf = buf;
}

static PyObject *
hpatio_read(hpatio* self, PyObject *args)
{ /* taken from CPython stringio.c */
    std::cout << "reading..." << std::endl;

    Py_ssize_t size, n;
    char * output;

    if(self->buf == NULL) {
        PyErr_SetString(PyExc_ValueError, "I/O operation on uninitialized HPATIO object");
        return NULL;
    }

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
    n = self->string_size - self->pos;
    if (size < 0 || size > n) {
        size = n;
        if (size < 0)
            size = 0;
    }

    output = self->buf + self->pos;
    self->pos += size;

    std::cout << "read " << size << " bytes" << std::endl;

    return PyUnicode_FromStringAndSize(output, size);
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

#include "_meminfo.h"
#define ALIGN 32

extern "C" {
/**
   Like csv_read_file, but reading from a string.
 **/
MemInfo ** csv_read_string(const std::string * fname, size_t * cols_to_read, int64_t * dtypes, size_t n_cols_to_read,
                           size_t * first_row, size_t * n_rows,
                           std::string * delimiters = NULL, std::string * quotes = NULL);

/**
   Delete memory returned by csv_read.
   @param[in] cols    pointer returned by csv_read
   @param[in] release If > 0, release given number of MemInfo pointers
**/
void csv_delete(MemInfo ** cols, size_t release);

}

// ***********************************************************************************
// ***********************************************************************************

static MemInfo_alloc_aligned_type mi_alloc = (MemInfo_alloc_aligned_type) import_meminfo_func("MemInfo_alloc_safe_aligned");
static MemInfo_release_type mi_release = (MemInfo_release_type) import_meminfo_func("MemInfo_release");
#if 1
static MemInfo_data_type mi_data = (MemInfo_data_type)import_meminfo_func("MemInfo_data");
#else
static void * mi_data(MemInfo* mi)
{
    return reinterpret_cast<void*>(((char*)mi)+24);
}
#endif

// Allocating an array of given typesize, optionally copy from org buffer
static MemInfo* alloc_meminfo(int dtype_sz, size_t n, MemInfo * org=NULL, size_t on=0)
{
    auto mi = mi_alloc(n*dtype_sz, ALIGN);
    if(org) memcpy(mi_data(mi), mi_data(org), std::min(n, on) * dtype_sz);
    return mi;
}


// Allocating an array of given type and size, optionally copy from org buffer
static MemInfo* dtype_alloc(int dtype, size_t n, MemInfo * org=NULL, size_t on=0)
{
    int sz = -1;
    switch(dtype) {
    case HPAT_CTypes::INT32:
    case HPAT_CTypes::UINT32:
    case HPAT_CTypes::FLOAT32:
        sz = 4;
        break;
    case HPAT_CTypes::INT64:
    case HPAT_CTypes::UINT64:
    case HPAT_CTypes::FLOAT64:
        sz = 8;
        break;
    case HPAT_CTypes::DATETIME:
        sz = sizeof(pandas_datetimestruct);
        break;
    default:
        std::cerr << "unsupported dtype requested.";
        return NULL;
    }
    return alloc_meminfo(sz, n, org, on);
}


static void dtype_dealloc(MemInfo * mi)
{
    mi_release(mi);
}


// convert string to a dtype
static void dtype_convert(int dtype, const std::string & from, void * to, size_t ln)
{
    switch(dtype) {
    case HPAT_CTypes::INT32:
        (reinterpret_cast<int32_t*>(to))[ln] = std::stoi(from.c_str());
        break;
    case HPAT_CTypes::UINT32:
        (reinterpret_cast<uint32_t*>(to))[ln] = (unsigned int)std::stoul(from.c_str());
        break;
    case HPAT_CTypes::INT64:
        (reinterpret_cast<int64_t*>(to))[ln] = std::stoll(from.c_str());
        break;
    case HPAT_CTypes::UINT64:
        (reinterpret_cast<uint64_t*>(to))[ln] = std::stoull(from.c_str());
        break;
    case HPAT_CTypes::FLOAT32:
        (reinterpret_cast<float*>(to))[ln] = std::stof(from.c_str());
        break;
    case HPAT_CTypes::FLOAT64:
        (reinterpret_cast<double*>(to))[ln] = std::stof(from.c_str());
        break;
    case HPAT_CTypes::DATETIME:
        parse_iso_8601_datetime(const_cast<char*>(from.c_str()), from.size(),
                                reinterpret_cast<pandas_datetimestruct*>(to) + ln,
                                NULL, NULL);
        break;
    default:
        ;
    }
}


// we can use and then delete the extra dtype column
extern "C" void csv_delete(MemInfo ** mi, size_t release)
{
    for(size_t i=0; i<release; ++i) {
        if(mi[i]) {
            dtype_dealloc(mi[i]);
        }
    }
    delete [] mi;
}


#define CHECK(expr, msg) if(!(expr)){std::cerr << "Error in csv_read: " << msg << std::endl; return NULL;}

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
static MemInfo** csv_read(std::istream & f, size_t fsz, size_t * cols_to_read, int64_t * dtypes, size_t n_cols_to_read,
                          size_t * first_row, size_t * n_rows,
                          std::string * delimiters = NULL, std::string * quotes = NULL)
{
    CHECK(dtypes, "Input parameter dtypes must not be NULL.");
    CHECK(first_row && n_rows, "Output parameters first_row and n_rows must not be NULL.");

    // For now we just do single process, so our chunksize is fsz
    size_t chunksize = fsz;
    char * buff = new char[chunksize+1];
    if(! f.read(buff, chunksize)) {
        PyErr_Print();
        std::cerr << "Reading " << chunksize << " bytes failed.";
        return NULL;
    }
    buff[chunksize] = 0;

    // we have our chunk read into buff, now create our 'file-like' input to pandas.read_csv
    auto gilstate = PyGILState_Ensure();
    PyObject * hio = PyObject_CallFunctionObjArgs((PyObject *) &hpatio_type, NULL);
    if (hio == NULL || PyErr_Occurred()) {
        PyErr_Print();
        std::cerr << "Could not create IO buffer object" << std::endl;
        PyGILState_Release(gilstate);
        return NULL;
    }
    hpatio_init(reinterpret_cast<hpatio*>(hio), buff, chunksize);

    // Now call pandas.read_csv.
    PyObject * pd_read_csv = import_sym("pandas", "read_csv");
    if (pd_read_csv == NULL || PyErr_Occurred()) {
        PyErr_Print();
        std::cerr << "Could not get pandas.read_csv " << std::endl;
        PyGILState_Release(gilstate);
        return NULL;
    }
    PyObject * df = PyObject_CallFunctionObjArgs(pd_read_csv, hio, NULL);

    Py_XDECREF(pd_read_csv);
    PyGILState_Release(gilstate);

    delete [] buff;

    if (df == NULL || PyErr_Occurred()) {
        PyErr_Print();
        std::cerr << "pandas.read_csv failed." << std::endl;
        return NULL;
    }

    std::cout << "Done!" << std::endl;
    // return df;
    return NULL;
}



# if 0
    MemInfo ** result = new MemInfo*[n_cols_to_read]();
    if(n_cols_to_read == 0 || fsz == 0) return result;

    int rank = hpat_dist_get_rank();
    int nranks = hpat_dist_get_size();

    // read the first line to determine number cols in file and estimate #lines
    std::string line;
    CHECK(std::getline(f,line), "could not read first line");
    auto linesize = line.size();

    size_t curr_row = 0;
    size_t ncols = 0;
    boost::escaped_list_separator<char> tokfunc("\\", delimiters ? *delimiters : ",", quotes ? *quotes : "\"\'");
    try{
        boost::tokenizer<boost::escaped_list_separator<char> > tk(line, tokfunc);
        for(auto i(tk.begin()); i!=tk.end(); ++ncols, ++i) {}
    } catch (...) {
        // The first line must be correct
        std::cerr << "Error parsing first line; giving up.\n";
        return NULL;
    }
    CHECK(n_cols_to_read <= ncols, "More columns requested than available.");
    if(cols_to_read) for(size_t i=0; i<n_cols_to_read; ++i) CHECK(ncols > cols_to_read[i], "Invalid column index provided.");

    // We evenly distribute the 'data' byte-wise
    auto chunksize = fsz/nranks;
    // and conservatively estimate #lines per chunk
    size_t linesperchunk = (size_t) std::max((1.1 * chunksize) / linesize, 1.0);

    *n_rows = 0;
    // we can skip reading alltogether if our chunk is past eof
    if(chunksize*rank < fsz) {
        // seems there is data for us to read, let's allocate the arrays
        for(size_t i=0; i<n_cols_to_read; ++i) result[i] = dtype_alloc(static_cast<int>(dtypes[i]), linesperchunk);

        // let's prepare an mask fast checking if a column is requested
        std::vector<ssize_t> req(ncols, -1);
        for(size_t i=0; i<n_cols_to_read; ++i) {
            auto idx = cols_to_read ? cols_to_read[i] : i;
            req[idx] = i;
        }

        // seek to our chunk and read it
        f.seekg(chunksize*rank, std::ios_base::beg);
        size_t chunkend = chunksize*(rank+1);
        size_t curr_pos = f.tellg();
        // we stop reading when at least reached boundary of our chunk
        // we always stop on a line-boundary!
        // 2 cases: exact boundary: next rank will have full line
        //          behind boundary: next rank will skip incomplete line
        //          note that the next rank might be after its boundary after skipping the first line!
        while(curr_pos < chunkend && std::getline(f, line)) {
            // if the boundary of the chunk contains enough whitespace the actual data will be entirely in the next line
            // in this case we must not read the line, we are done!
            size_t my_pos = f.tellg();
            if(my_pos > chunkend) {
                int num_whitespaces(0);
                for(char c : line) {
                    if(!std::isspace(c)) break;
                    ++num_whitespaces ;
                }
                // there is no more meaningfull data in our chunk -> we are done
                if(curr_pos + num_whitespaces >= chunkend) break;
            }
            // miss estimated number of lines?
            if(*n_rows >= linesperchunk) {
                size_t new_lpc = std::max(linesperchunk * 1.1, linesperchunk+2.0);
                for(size_t i=0; i<n_cols_to_read; ++i) {
                    MemInfo * tmp = result[i];
                    result[i] = dtype_alloc(static_cast<int>(dtypes[i]), new_lpc, tmp, linesperchunk);
                    dtype_dealloc(reinterpret_cast<MemInfo*>(tmp));
                }
                linesperchunk = new_lpc;
            }
            try{
                size_t c = 0;
                boost::tokenizer<boost::escaped_list_separator<char> > tk(line, tokfunc);
                for(auto i(tk.begin()); i!=tk.end(); ++c, ++i) {
                    if(req[c] >= 0) { // we only care about requested columns
                        dtype_convert(static_cast<int>(dtypes[c]), *i, mi_data(result[req[c]]), *n_rows);
                    }
                }
                // we count/keep only complete lines
                if(c == ncols) ++(*n_rows);
                else throw std::invalid_argument("Too few columns");
            } catch (...) {
                // The first line can easily be incorrect
                if(curr_row > 0 || rank == 0) std::cerr << "Error parsing line " << curr_row << "; skipping.\n";
            }
            curr_pos = my_pos;
            ++curr_row;
        }
    }
    *first_row = hpat_dist_exscan_i8(*n_rows);
    return result;
}
#endif

extern "C" void ** csv_read_file(const std::string * fname, size_t * cols_to_read, int64_t * dtypes, size_t n_cols_to_read,
                                 size_t * first_row, size_t * n_rows,
                                 std::string * delimiters, std::string * quotes)
{
    CHECK(fname != NULL, "NULL filename provided.");
    // get total file-size
    auto fsz = boost::filesystem::file_size(*fname);
    std::ifstream f(*fname);
    CHECK(f.good() && !f.eof() && f.is_open(), "could not open file.");
    return reinterpret_cast<void**>(csv_read(f, fsz, cols_to_read, dtypes, n_cols_to_read, first_row, n_rows, delimiters, quotes));
}


extern "C" MemInfo ** csv_read_string(const std::string * str, size_t * cols_to_read, int64_t * dtypes, size_t n_cols_to_read,
                                      size_t * first_row, size_t * n_rows,
                                      std::string * delimiters, std::string * quotes)
{
    CHECK(str != NULL, "NULL string provided.");
    // get total file-size
    std::istringstream f(*str);
    CHECK(f.good(), "could not create istrstream from string.");
    return csv_read(f, str->size(), cols_to_read, dtypes, n_cols_to_read, first_row, n_rows, delimiters, quotes);
}

#undef CHECK

// ***********************************************************************************
// ***********************************************************************************

#if 0

static void printit(MemInfo ** r, int64_t * dtypes, size_t nr, size_t nc, size_t fr)
{
    for(size_t i=0; i<nr; ++i) {
        std::stringstream f;
        f << "row" << (fr+i) << " ";
        for(size_t j=0; j<nc; ++j) {
            void * data = mi_data(r[j]);
            switch(dtypes[j]) {
            case HPAT_CTypes::INT32:
                f << "i32:" << (reinterpret_cast<int32_t*>(data))[i];
                break;
            case HPAT_CTypes::UINT32:
                f << "ui32:" << (reinterpret_cast<uint32_t*>(data))[i];
                break;
            case HPAT_CTypes::INT64:
                f << "i64:" << (reinterpret_cast<int64_t*>(data))[i];
                break;
            case HPAT_CTypes::UINT64:
                f << "ui64:" << (reinterpret_cast<uint64_t*>(data))[i];
                break;
            case HPAT_CTypes::FLOAT32:
                f << "f32:" << (reinterpret_cast<float*>(data))[i];
                break;
            case HPAT_CTypes::FLOAT64:
                f << "f64:" << (reinterpret_cast<double*>(data))[i];
                break;
            default:
                f << "?";
                ;
            }
            f << ", ";
        }
        f << std::endl;
        std::cout << f.str();
    }
}

int main()
{
    int rank = hpat_dist_get_rank();
    int ncols = 4;

    size_t cols[ncols] = {0,1,2,3};
    int dtypes[ncols] = {HPAT_CTypes::INT, HPAT_CTypes::FLOAT32, HPAT_CTypes::FLOAT64, HPAT_CTypes::UINT64};
    size_t first_row, n_rows;
    std::string delimiters = ",";
    std::string quotes = "\"";

    if(!rank) std::cout << "\na regular case\n";
    std::string csv =
        "0,2.3,4.6,47736\n"
        "1,2.3,4.6,47736\n"
        "2,2.3,4.6,47736\n"
        "4,2.3,4.6,47736\n";
    auto r = csv_read_string(&csv, cols, dtypes, ncols,
                             &first_row, &n_rows,
                             &delimiters, &quotes);
    printit(r, dtypes, n_rows, ncols, first_row);
    csv_delete(r, ncols);

    if(!rank) std::cout << "\nwhite-spaces, mis-predicted line-count, imbalance in the beginning\n";
    MPI_Barrier(MPI_COMM_WORLD);
    csv =
        "0,                       2.3                 ,     4.6                    ,       47736\n"
        "1,\"1.3\",4.6,47736\n"
        "2,2.3,4.6,\"47736\"\n"
        "3,2.3,4.6,47736\n";
    r = csv_read_string(&csv, NULL, dtypes, ncols,
                        &first_row, &n_rows,
                        &delimiters, &quotes);
    printit(r, dtypes, n_rows, ncols, first_row);
    csv_delete(r, ncols);

    if(!rank) std::cout << "\nwhite-spaces, imbalance in the end\n";
    MPI_Barrier(MPI_COMM_WORLD);
    csv =
        "0,2.3,4.6,47736\n"
        "1,\"1.3\",4.6,47736\n"
        "2,2.3,4.6,\"47736\"\n"
        "3,         2.3                 ,     4.6                    ,       47736\n";
    r = csv_read_string(&csv, NULL, dtypes, ncols,
                        &first_row, &n_rows,
                        &delimiters, &quotes);
    printit(r, dtypes, n_rows, ncols, first_row);
    csv_delete(r, ncols);


    if(!rank) std::cout << "\nwhite-spaces, quotes, imbalance in the middle\n";
    MPI_Barrier(MPI_COMM_WORLD);
    csv =
        "0,2.3,4.6,47736\n"
        "1,         2.3                          ,     4.6                    ,       47736\n"
        "2,\"1.3\",4.6,47736\n"
        "3,2.3,4.6,\"47736\"\n";
    r = csv_read_string(&csv, NULL, dtypes, ncols,
                        &first_row, &n_rows,
                        &delimiters, &quotes);
    printit(r, dtypes, n_rows, ncols, first_row);
    csv_delete(r, ncols);

    if(!rank) std::cout << "\nsyntax errors, no explicit quotes/delimiters\n";
    MPI_Barrier(MPI_COMM_WORLD);
    csv =
        "0,2.3,4.6,47736\n"
        "1,2.3,4.6,error\n"
        "2,\"2.3\",4.6,47736\n"
        "3,2.3,4.6,\"47736\"\n";
    r = csv_read_string(&csv, NULL, dtypes, ncols,
                        &first_row, &n_rows);
    printit(r, dtypes, n_rows, ncols, first_row);
    csv_delete(r, ncols);

    csv_read_string(NULL, NULL, NULL, 4, NULL, NULL, NULL, NULL);
    csv_read_string(&csv, NULL, NULL, 4, NULL, NULL, NULL, NULL);


    MPI_Finalize();
}
#endif

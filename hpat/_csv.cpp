/*
  SPMD CSV reader. 
*/
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
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
#include <numpy/ndarraytypes.h>

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
                     NULL if no data was read
 **/
void ** csv_read_file(const std::string * fname, size_t * cols_to_read, int * dtypes, size_t n_cols_to_read,
                      size_t * first_row, size_t * n_rows,
                      std::string * delimiters = NULL, std::string * quotes = NULL);
/**
   Like csv_read_file, but reading from a string.
 **/
void ** csv_read_string(const std::string * fname, size_t * cols_to_read, int * dtypes, size_t n_cols_to_read,
                        size_t * first_row, size_t * n_rows,
                        std::string * delimiters = NULL, std::string * quotes = NULL);

/**
   Delete memory returned by csv_read.
   @param[in] cols   pointer returned by csv_read
   @param[in] n_cols number of cols, must be identical to n_cols_to_read when calling read_csv
**/
void csv_delete(void ** cols, size_t n_cols);


// ***********************************************************************************
// ***********************************************************************************

int64_t hpat_dist_exscan_i8(int64_t value)
{
    // printf("sum value: %lld\n", value);
    int64_t out=0;
    MPI_Exscan(&value, &out, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

int hpat_dist_get_rank()
{
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized)
        MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // printf("my_rank:%d\n", rank);
    return rank;
}

int hpat_dist_get_size()
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // printf("r size:%d\n", sizeof(MPI_Request));
    // printf("mpi_size:%d\n", size);
    return size;
}

template<int T> struct DTYPE;

template<>
struct DTYPE<NPY_BOOL>
{
    typedef bool dtype;
};

template<>
struct DTYPE<NPY_BYTE> // also NPY_INT8
{
    typedef int8_t dtype;
};

template<>
struct DTYPE<NPY_SHORT> // also NPY_INT16
{
    typedef int16_t dtype;
};

template<>
struct DTYPE<NPY_INT> // also NPY_INT32
{
    typedef int32_t dtype;
};

template<>
struct DTYPE<NPY_INT64> // also NPY_INTP
{
    typedef int64_t dtype;
};

template<>
struct DTYPE<NPY_UBYTE> // also NPY_UINT8
{
    typedef int8_t dtype;
};

template<>
struct DTYPE<NPY_USHORT> // also NPY_UINT16
{
    typedef uint16_t dtype;
};

template<>
struct DTYPE<NPY_UINT> // also NPY_UINT32
{
    typedef uint32_t dtype;
};

template<>
struct DTYPE<NPY_UINT64> // also NPY_INTUP
{
    typedef int64_t dtype;
};

template<>
struct DTYPE<NPY_FLOAT> // also NPY_FLOAT32
{
    typedef float dtype;
};

template<>
struct DTYPE<NPY_DOUBLE> // also NPY_FLOAT64
{
    typedef double dtype;
};


// Allocating an array of given type and size, optionally copy from org buffer
template< typename T >
static T * dtype_alloc(size_t sz, void * org=NULL, size_t osz=0)
{
    T * tmp = new T[sz];
    if(org) memcpy(tmp, org, std::min(sz, osz) * sizeof(T));
    return tmp;
}
    
static void * dtype_alloc(int dtype, size_t sz, void * org=NULL, size_t osz=0)
{
    switch(dtype) {
    case NPY_INT32:
    case NPY_UINT32: // also NPY_UINT NPY_INT
        return dtype_alloc<DTYPE<NPY_INT32>::dtype>(sz, org, osz);
    case NPY_INT64:
    case NPY_UINT64: // also NPY_UINTP NPY_INTP
        return dtype_alloc<DTYPE<NPY_INT64>::dtype>(sz, org, osz);
    case NPY_FLOAT32: // also NPY_FLOAT
        return dtype_alloc<DTYPE<NPY_FLOAT32>::dtype>(sz, org, osz);
    case NPY_FLOAT64:
        return dtype_alloc<DTYPE<NPY_FLOAT64>::dtype>(sz, org, osz);
    default:
        std::cerr << "Error in CSV: unsupported dtype\n";
        return NULL;
    }
}


// De-Allocating given array, requires its type
template< typename T >
static void dtype_dealloc(void * org)
{
    delete [] reinterpret_cast<T*>(org);
}

static void dtype_dealloc(int dtype, void * ptr)
{
    switch(dtype) {
    case NPY_INT32:
    case NPY_UINT32: // also NPY_UINT NPY_INT
        dtype_dealloc<DTYPE<NPY_INT32>::dtype>(ptr);
        break;
    case NPY_INT64:
    case NPY_UINT64: // also NPY_UINTP NPY_INTP
        dtype_dealloc<DTYPE<NPY_INT64>::dtype>(ptr);
        break;
    case NPY_FLOAT32: // also NPY_FLOAT
        dtype_dealloc<DTYPE<NPY_FLOAT32>::dtype>(ptr);
        break;
    case NPY_FLOAT64:
        dtype_dealloc<DTYPE<NPY_FLOAT64>::dtype>(ptr);
        break;
    default:
        std::cerr << "Error in CSV: unsupported dtype\n";
    }
}

// convert string to a dtype
static void dtype_convert(int dtype, const std::string & from, void * to, size_t ln)
{
    switch(dtype) {
    case NPY_INT32: // also NPY_INT
        (reinterpret_cast<int32_t*>(to))[ln] = std::stoi(from.c_str());
        break;
    case NPY_UINT32: // also NPY_UINT
        (reinterpret_cast<uint32_t*>(to))[ln] = (unsigned int)std::stoul(from.c_str());
        break;
    case NPY_INT64: // also NPY_INTP
        (reinterpret_cast<int64_t*>(to))[ln] = std::stoll(from.c_str());
        break;
    case NPY_UINT64: // also NPY_UINTP
        (reinterpret_cast<uint64_t*>(to))[ln] = std::stoull(from.c_str());
        break;
    case NPY_FLOAT32: // also NPY_FLOAT
        (reinterpret_cast<float*>(to))[ln] = std::stof(from.c_str());
        break;
    case NPY_FLOAT64: // also NPY_DOUBLE
        (reinterpret_cast<double*>(to))[ln] = std::stof(from.c_str());
        break;
    default:
        ;
    }
}


// we can use and then delete the extra dtype column
void csv_delete(void ** cols, size_t n_cols)
{
    if(cols == NULL) return;
    for(auto i=0; i<n_cols; ++i) {
        if(cols[i]) {
            dtype_dealloc(reinterpret_cast<int*>(cols[n_cols])[i], cols[i]);
        }
    }
    if(cols[n_cols]) delete [] reinterpret_cast<int*>(cols[n_cols]);
    delete [] cols;
}


#define CHECK(expr, msg) if(!(expr)){std::cerr << "Error in csv_read: " << msg << std::endl; return NULL;}

/* 
  We divide the file into chunks, one for each process.
  Lines generally have different lengths, so we might start int he middle of the line.
  Hence, we determine the number cols in the file and check if our first line is a full line.
  If not, we just throw it away.
  Similary, our chunk might not end at a line boundary. 
  Note that we assume all lines have the same numer of tokens/columns.
  We always read full lines until we read at least our chunk-size.
  This makes sure there are no gaps and no data duplication.

  We store the dtypes in an extra array at the end of the returned result.
 */
static void ** csv_read(std::istream & f, size_t fsz, size_t * cols_to_read, int * dtypes, size_t n_cols_to_read,
                        size_t * first_row, size_t * n_rows,
                        std::string * delimiters = NULL, std::string * quotes = NULL)
{
    CHECK(dtypes, "Input parameter dtypes must not be NULL.");
    CHECK(first_row && n_rows, "Output parameters first_row and n_rows must not be NULL.");

    void ** result = new void*[n_cols_to_read+1]();
    if(n_cols_to_read == 0 || fsz == 0) return result;

    int rank = hpat_dist_get_rank();
    int nranks = hpat_dist_get_size();

    // read the first line to determine number cols in file and estimate #lines
    std::string line;
    CHECK(std::getline(f,line), "could not read first line");
    auto linesize = line.size();

    size_t ncols = 0;
    boost::escaped_list_separator<char> tokfunc("\\", delimiters ? *delimiters : ",", quotes ? *quotes : "\"\'");
    {
        boost::tokenizer<boost::escaped_list_separator<char> > tk(line, tokfunc);
        for(auto i(tk.begin()); i!=tk.end(); ++ncols, ++i) {}
    }
    CHECK(n_cols_to_read <= ncols, "More columns requested than available.");
    if(cols_to_read) for(auto i=0; i<n_cols_to_read; ++i) CHECK(ncols > cols_to_read[i], "Invalid column index provided.");
    
    // We evenly distribute the 'data' byte-wise
    auto chunksize = fsz/nranks;
    // and conservatively estimate #lines per chunk
    size_t linesperchunk = (size_t) std::max((1.1 * chunksize) / linesize, 1.0);
    
    *n_rows = 0;
    // we can skip reading alltogether if our chunk is past eof
    if(chunksize*rank < fsz) {
        // seems there is data for us to read, let's allocate the arrays
        for(auto i=0; i<n_cols_to_read; ++i) result[i] = dtype_alloc(dtypes[i], linesperchunk);
        // we append the dtypes array at the end
        result[n_cols_to_read] = new int[n_cols_to_read];
        memcpy(result[n_cols_to_read], dtypes, n_cols_to_read * sizeof(int));
 
        // let's prepare an mask fast checking if a column is requested
        std::vector<ssize_t> req(ncols, -1);
        for(auto i=0; i<n_cols_to_read; ++i) {
            auto idx = cols_to_read ? cols_to_read[i] : i;
            req[idx] = i;
        }
        
        // seek to our chunk and read it
        f.seekg(chunksize*rank, std::ios_base::beg);
        size_t chunkend = chunksize*(rank+1);
        // we stop reading when at least reached boundary of our chunk
        // we always stop on a line-boundary!
        // 2 cases: exact boundary: next rank will have full line
        //          behind boundary: next rank will skip incomplete line
        //          note that the next rank might be after its boundary after skipping the first line!
        while(f.tellg() < chunkend && std::getline(f, line)) {
            // miss estimated number of lines?
            if(*n_rows >= linesperchunk) {
                size_t new_lpc = std::max(linesperchunk * 1.1, linesperchunk+2.0);
                for(auto i=0; i<n_cols_to_read; ++i) {
                    void * tmp = result[i];
                    result[i] = dtype_alloc(dtypes[i], new_lpc, tmp, linesperchunk);
                    dtype_dealloc(dtypes[i], tmp);
                }
                linesperchunk = new_lpc;
            }
            // first line needs special attention
            size_t c = 0;
            boost::tokenizer<boost::escaped_list_separator<char> > tk(line, tokfunc);
            try{
                for(auto i(tk.begin()); i!=tk.end(); ++c, ++i) {
                    if(req[c] >= 0) { // we only care about requested columns
                        dtype_convert(dtypes[c], *i, result[req[c]], *n_rows);
                    }
                }
                // we count/keep all lines except the first if it's incomplete
                if(*n_rows > 0 || c == ncols) ++(*n_rows);
            } catch (...) {
                // The first line can easily be incorrect
                if(*n_rows > 0 || rank == 0) std::cerr << "Error parsing line " << (*n_rows) << "; skipping.\n";
            }
        }
    }
    *first_row = hpat_dist_exscan_i8(*n_rows);
    return result;
}


void ** csv_read_file(const std::string * fname, size_t * cols_to_read, int * dtypes, size_t n_cols_to_read,
                      size_t * first_row, size_t * n_rows,
                      std::string * delimiters, std::string * quotes)
{
    CHECK(fname != NULL, "NULL filename provided.");
    // get total file-size
    auto fsz = boost::filesystem::file_size(*fname);
    std::ifstream f(*fname);
    CHECK(f.good() && !f.eof() && f.is_open(), "could not open file.");
    return csv_read(f, fsz, cols_to_read, dtypes, n_cols_to_read, first_row, n_rows, delimiters, quotes);
}


void ** csv_read_string(const std::string * str, size_t * cols_to_read, int * dtypes, size_t n_cols_to_read,
                        size_t * first_row, size_t * n_rows,
                        std::string * delimiters, std::string * quotes)
{
    CHECK(str != NULL, "NULL string provided.");
    // get total file-size
    std::istringstream f(*str);
    CHECK(f.good(), "could not create istrstream from string.");
    return csv_read(f, str->size(), cols_to_read, dtypes, n_cols_to_read, first_row, n_rows, delimiters, quotes);
}


// ***********************************************************************************
// ***********************************************************************************

static void printit(void ** r, size_t nr, size_t nc, size_t fr)
{
    for(auto i=0; i<nr; ++i) {
        std::stringstream f;
        f << "row" << (fr+i) << " ";
        for(auto j=0; j<nc; ++j) {
            switch(reinterpret_cast<int*>(r[nc])[j]) {
            case NPY_INT32: // also NPY_INT
                f << (reinterpret_cast<int32_t*>(r[j]))[i];
                break;
            case NPY_UINT32: // also NPY_UINT
                f << (reinterpret_cast<uint32_t*>(r[j]))[i];
                break;
            case NPY_INT64: // also NPY_INTP
                f << (reinterpret_cast<int64_t*>(r[j]))[i];
                break;
            case NPY_UINT64: // also NPY_UINTP
                f << (reinterpret_cast<uint64_t*>(r[j]))[i];
                break;
            case NPY_FLOAT32: // also NPY_FLOAT
                f << (reinterpret_cast<float*>(r[j]))[i];
                break;
            case NPY_FLOAT64: // also NPY_DOUBLE
                f << (reinterpret_cast<double*>(r[j]))[i];
                break;
            default:
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
    int dtypes[ncols] = {NPY_INT, NPY_FLOAT, NPY_DOUBLE, NPY_UINT64};
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
    printit(r, n_rows, ncols, first_row);
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
    printit(r, n_rows, ncols, first_row);
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
    printit(r, n_rows, ncols, first_row);
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
    printit(r, n_rows, ncols, first_row);
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
    printit(r, n_rows, ncols, first_row);
    csv_delete(r, ncols);

    csv_read_string(NULL, NULL, NULL, 4, NULL, NULL, NULL, NULL);
    csv_read_string(&csv, NULL, NULL, 4, NULL, NULL, NULL, NULL);
    

    MPI_Finalize();
}

// NPY_LONGDOUBLE
// The enumeration value for a platform-specific floating point type which is at least as large as NPY_DOUBLE, but larger on many platforms.

// NPY_CFLOAT
// NPY_COMPLEX64
// The enumeration value for a 64-bit/8-byte complex type made up of two NPY_FLOAT values.

// NPY_CDOUBLE
// NPY_COMPLEX128
// The enumeration value for a 128-bit/16-byte complex type made up of two NPY_DOUBLE values.

// NPY_CLONGDOUBLE
// The enumeration value for a platform-specific complex floating point type which is made up of two NPY_LONGDOUBLE values.

// NPY_DATETIME
// The enumeration value for a data type which holds dates or datetimes with a precision based on selectable date or time units.

// NPY_TIMEDELTA
// The enumeration value for a data type which holds lengths of times in integers of selectable date or time units.

// NPY_STRING
// The enumeration value for ASCII strings of a selectable size. The strings have a fixed maximum size within a given array.

// NPY_UNICODE
// The enumeration value for UCS4 strings of a selectable size. The strings have a fixed maximum size within a given array.

// NPY_OBJECT
// The enumeration value for references to arbitrary Python objects.

// NPY_VOID
// Primarily used to hold struct dtypes, but can contain arbitrary binary data.

// Some useful aliases of the above types are

// NPY_MASK
// The enumeration value of the type used for masks, such as with the NPY_ITER_ARRAYMASK iterator flag. This is equivalent to NPY_UINT8.

// NPY_DEFAULT_TYPE
// The default type to use when no dtype is explicitly specified, for example when calling np.zero(shape). This is equivalent to NPY_DOUBLE.

// Other useful related constants are

// NPY_NTYPES
// The total number of built-in NumPy types. The enumeration covers the range from 0 to NPY_NTYPES-1.

// NPY_NOTYPE
// A signal value guaranteed not to be a valid type enumeration number.

// NPY_USERDEF
// The start of type numbers used for Custom Data types.

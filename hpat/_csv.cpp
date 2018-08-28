/*
  SPMD CSV reader. 
*/
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
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
//#include <numpy/ndarraytypes.h>
#include "_hpat_common.h"

// #include "_distributed.h"

static int hpat_dist_get_rank()
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

static int hpat_dist_get_size()
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // printf("r size:%d\n", sizeof(MPI_Request));
    // printf("mpi_size:%d\n", size);
    return size;
}

static int64_t hpat_dist_exscan_i8(int64_t value)
{
    // printf("sum value: %lld\n", value);
    int64_t out=0;
    MPI_Exscan(&value, &out, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

#include "_meminfo.h"

extern "C" {

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
MemInfo ** csv_read_file(const std::string * fname, size_t * cols_to_read, int64_t * dtypes, size_t n_cols_to_read,
                         size_t * first_row, size_t * n_rows,
                         std::string * delimiters = NULL, std::string * quotes = NULL);
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

template<int T> struct DTYPE;

template<>
struct DTYPE<HPAT_CTypes::INT32>
{
    typedef int32_t dtype;
};

template<>
struct DTYPE<HPAT_CTypes::INT64>
{
    typedef int64_t dtype;
};

template<>
struct DTYPE<HPAT_CTypes::UINT32>
{
    typedef uint32_t dtype;
};

template<>
struct DTYPE<HPAT_CTypes::UINT64>
{
    typedef int64_t dtype;
};

template<>
struct DTYPE<HPAT_CTypes::FLOAT32>
{
    typedef float dtype;
};

template<>
struct DTYPE<HPAT_CTypes::FLOAT64>
{
    typedef double dtype;
};


static MemInfo_alloc_aligned_type mi_alloc = (MemInfo_alloc_aligned_type) import_meminfo_func("MemInfo_alloc_aligned");
static MemInfo_release_type mi_release = (MemInfo_release_type) import_meminfo_func("MemInfo_release");
#if 0
static MemInfo_data_type mi_data = (MemInfo_data_type)import_meminfo_func("MemInfo_data");
#else
static void * mi_data(MemInfo* mi)
{
    return reinterpret_cast<void*>(((char*)mi)+24);
}
#endif

// Allocating an array of given typesize, optionally copy from org buffer
static MemInfo* alloc_meminfo(int dtype_sz, size_t n, void * org=NULL, size_t on=0)
{
    auto mi = mi_alloc(n*dtype_sz, dtype_sz);
    std::cout << "kk " << mi << std::endl;
    if(org) memcpy(mi_data(mi), org, std::min(n, on) * dtype_sz);
    return mi;
}


// Allocating an array of given type and size, optionally copy from org buffer
static MemInfo* dtype_alloc(int dtype, size_t n, void * org=NULL, size_t on=0)
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
    default:
        ;
    }
}


// we can use and then delete the extra dtype column
extern "C" void csv_delete(MemInfo ** mi, size_t release)
{
    for(auto i=0; i<release; ++i) {
        if(mi[i]) {
            dtype_dealloc(mi[i]);
        }
    }
    delete [] mi;
}


#define CHECK(expr, msg) if(!(expr)){std::cerr << "Error in csv_read: " << msg << std::endl; return NULL;}

static void printit(MemInfo ** r, int64_t * dtypes, size_t nr, size_t nc, size_t fr);

/* 
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

    MemInfo ** result = new MemInfo*[n_cols_to_read]();
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
        for(auto i=0; i<n_cols_to_read; ++i) result[i] = dtype_alloc(static_cast<int>(dtypes[i]), linesperchunk);
 
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
                    result[i] = dtype_alloc(static_cast<int>(dtypes[i]), new_lpc, tmp, linesperchunk);
                    dtype_dealloc(reinterpret_cast<MemInfo*>(tmp));
                }
                linesperchunk = new_lpc;
            }
            // first line needs special attention
            size_t c = 0;
            boost::tokenizer<boost::escaped_list_separator<char> > tk(line, tokfunc);
            try{
                for(auto i(tk.begin()); i!=tk.end(); ++c, ++i) {
                    if(req[c] >= 0) { // we only care about requested columns
                        dtype_convert(static_cast<int>(dtypes[c]), *i, mi_data(result[req[c]]), *n_rows);
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
    printit(result, dtypes, *n_rows, n_cols_to_read, *first_row);
    return result;
}


extern "C" MemInfo ** csv_read_file(const std::string * fname, size_t * cols_to_read, int64_t * dtypes, size_t n_cols_to_read,
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


// ***********************************************************************************
// ***********************************************************************************

static void printit(MemInfo ** r, int64_t * dtypes, size_t nr, size_t nc, size_t fr)
{
    for(auto i=0; i<nr; ++i) {
        std::stringstream f;
        f << "row" << (fr+i) << " ";
        for(auto j=0; j<nc; ++j) {
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

#if 0
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

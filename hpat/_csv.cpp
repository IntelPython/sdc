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
#include "_hpat_common.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// #include "_distributed.h"

// FIXME: USE HPAT/Pandas from lib, not copy&paste

typedef struct {
    npy_int64 year;
    npy_int32 month, day, hour, min, sec, us, ps, as;
} pandas_datetimestruct;

static int is_leapyear(npy_int64 year) {
    return (year & 0x3) == 0 && /* year % 4 == 0 */
        ((year % 100) != 0 || (year % 400) == 0);
}

static const int days_per_month_table[2][12] = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}};

static int parse_iso_8601_datetime(char *str, int len,
                                   pandas_datetimestruct *out,
                                   int *out_local, int *out_tzoffset) {
    int year_leap = 0;
    int i, numdigits;
    char *substr, sublen;

    /* If year-month-day are separated by a valid separator,
     * months/days without leading zeroes will be parsed
     * (though not iso8601). If the components aren't separated,
     * 4 (YYYY) or 8 (YYYYMMDD) digits are expected. 6 digits are
     * forbidden here (but parsed as YYMMDD elsewhere).
    */
    int has_ymd_sep = 0;
    char ymd_sep = '\0';
    char valid_ymd_sep[] = {'-', '.', '/', '\\', ' '};
    int valid_ymd_sep_len = sizeof(valid_ymd_sep);

    /* hour-minute-second may or may not separated by ':'. If not, then
     * each component must be 2 digits. */
    int has_hms_sep = 0;
    int hour_was_2_digits = 0;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(pandas_datetimestruct));
    out->month = 1;
    out->day = 1;

    substr = str;
    sublen = len;

    /* Skip leading whitespace */
    while (sublen > 0 && isspace(*substr)) {
        ++substr;
        --sublen;
    }

    /* Leading '-' sign for negative year */
    if (*substr == '-') {
        ++substr;
        --sublen;
    }

    if (sublen == 0) {
        goto parse_error;
    }

    /* PARSE THE YEAR (4 digits) */
    out->year = 0;
    if (sublen >= 4 && isdigit(substr[0]) && isdigit(substr[1]) &&
        isdigit(substr[2]) && isdigit(substr[3])) {
        out->year = 1000 * (substr[0] - '0') + 100 * (substr[1] - '0') +
                    10 * (substr[2] - '0') + (substr[3] - '0');

        substr += 4;
        sublen -= 4;
    }

    /* Negate the year if necessary */
    if (str[0] == '-') {
        out->year = -out->year;
    }
    /* Check whether it's a leap-year */
    year_leap = is_leapyear(out->year);

    /* Next character must be a separator, start of month, or end of string */
    if (sublen == 0) {
        if (out_local != NULL) {
            *out_local = 0;
        }
        goto finish;
    }

    if (!isdigit(*substr)) {
        for (i = 0; i < valid_ymd_sep_len; ++i) {
            if (*substr == valid_ymd_sep[i]) {
                break;
            }
        }
        if (i == valid_ymd_sep_len) {
            goto parse_error;
        }
        has_ymd_sep = 1;
        ymd_sep = valid_ymd_sep[i];
        ++substr;
        --sublen;
        /* Cannot have trailing separator */
        if (sublen == 0 || !isdigit(*substr)) {
            goto parse_error;
        }
    }

    /* PARSE THE MONTH */
    /* First digit required */
    out->month = (*substr - '0');
    ++substr;
    --sublen;
    /* Second digit optional if there was a separator */
    if (isdigit(*substr)) {
        out->month = 10 * out->month + (*substr - '0');
        ++substr;
        --sublen;
    } else if (!has_ymd_sep) {
        goto parse_error;
    }
    if (out->month < 1 || out->month > 12) {
        printf("Month out of range in datetime string \"%s\"", str);
        goto error;
    }

    /* Next character must be the separator, start of day, or end of string */
    if (sublen == 0) {
        /* Forbid YYYYMM. Parsed instead as YYMMDD by someone else. */
        if (!has_ymd_sep) {
            goto parse_error;
        }
        if (out_local != NULL) {
            *out_local = 0;
        }
        goto finish;
    }

    if (has_ymd_sep) {
        /* Must have separator, but cannot be trailing */
        if (*substr != ymd_sep || sublen == 1) {
            goto parse_error;
        }
        ++substr;
        --sublen;
    }

    /* PARSE THE DAY */
    /* First digit required */
    if (!isdigit(*substr)) {
        goto parse_error;
    }
    out->day = (*substr - '0');
    ++substr;
    --sublen;
    /* Second digit optional if there was a separator */
    if (isdigit(*substr)) {
        out->day = 10 * out->day + (*substr - '0');
        ++substr;
        --sublen;
    } else if (!has_ymd_sep) {
        goto parse_error;
    }
    if (out->day < 1 ||
        out->day > days_per_month_table[year_leap][out->month - 1]) {
        printf("Day out of range in datetime string \"%s\"", str);
        goto error;
    }

    /* Next character must be a 'T', ' ', or end of string */
    if (sublen == 0) {
        if (out_local != NULL) {
            *out_local = 0;
        }
        goto finish;
    }

    if ((*substr != 'T' && *substr != ' ') || sublen == 1) {
        goto parse_error;
    }
    ++substr;
    --sublen;

    /* PARSE THE HOURS */
    /* First digit required */
    if (!isdigit(*substr)) {
        goto parse_error;
    }
    out->hour = (*substr - '0');
    ++substr;
    --sublen;
    /* Second digit optional */
    if (isdigit(*substr)) {
        hour_was_2_digits = 1;
        out->hour = 10 * out->hour + (*substr - '0');
        ++substr;
        --sublen;
        if (out->hour >= 24) {
            printf("Hours out of range in datetime string \"%s\"", str);
            goto error;
        }
    }

    /* Next character must be a ':' or the end of the string */
    if (sublen == 0) {
        if (!hour_was_2_digits) {
            goto parse_error;
        }
        goto finish;
    }

    if (*substr == ':') {
        has_hms_sep = 1;
        ++substr;
        --sublen;
        /* Cannot have a trailing separator */
        if (sublen == 0 || !isdigit(*substr)) {
            goto parse_error;
        }
    } else if (!isdigit(*substr)) {
        if (!hour_was_2_digits) {
            goto parse_error;
        }
        goto parse_timezone;
    }

    /* PARSE THE MINUTES */
    /* First digit required */
    out->min = (*substr - '0');
    ++substr;
    --sublen;
    /* Second digit optional if there was a separator */
    if (isdigit(*substr)) {
        out->min = 10 * out->min + (*substr - '0');
        ++substr;
        --sublen;
        if (out->min >= 60) {
            printf("Minutes out of range in datetime string \"%s\"", str);
            goto error;
        }
    } else if (!has_hms_sep) {
        goto parse_error;
    }

    if (sublen == 0) {
        goto finish;
    }

    /* If we make it through this condition block, then the next
     * character is a digit. */
    if (has_hms_sep && *substr == ':') {
        ++substr;
        --sublen;
        /* Cannot have a trailing ':' */
        if (sublen == 0 || !isdigit(*substr)) {
            goto parse_error;
        }
    } else if (!has_hms_sep && isdigit(*substr)) {
    } else {
        goto parse_timezone;
    }

    /* PARSE THE SECONDS */
    /* First digit required */
    out->sec = (*substr - '0');
    ++substr;
    --sublen;
    /* Second digit optional if there was a separator */
    if (isdigit(*substr)) {
        out->sec = 10 * out->sec + (*substr - '0');
        ++substr;
        --sublen;
        if (out->sec >= 60) {
            printf("Seconds out of range in datetime string \"%s\"", str);
            goto error;
        }
    } else if (!has_hms_sep) {
        goto parse_error;
    }

    /* Next character may be a '.' indicating fractional seconds */
    if (sublen > 0 && *substr == '.') {
        ++substr;
        --sublen;
    } else {
        goto parse_timezone;
    }

    /* PARSE THE MICROSECONDS (0 to 6 digits) */
    numdigits = 0;
    for (i = 0; i < 6; ++i) {
        out->us *= 10;
        if (sublen > 0 && isdigit(*substr)) {
            out->us += (*substr - '0');
            ++substr;
            --sublen;
            ++numdigits;
        }
    }

    if (sublen == 0 || !isdigit(*substr)) {
        goto parse_timezone;
    }

    /* PARSE THE PICOSECONDS (0 to 6 digits) */
    numdigits = 0;
    for (i = 0; i < 6; ++i) {
        out->ps *= 10;
        if (sublen > 0 && isdigit(*substr)) {
            out->ps += (*substr - '0');
            ++substr;
            --sublen;
            ++numdigits;
        }
    }

    if (sublen == 0 || !isdigit(*substr)) {
        goto parse_timezone;
    }

    /* PARSE THE ATTOSECONDS (0 to 6 digits) */
    numdigits = 0;
    for (i = 0; i < 6; ++i) {
        out->as *= 10;
        if (sublen > 0 && isdigit(*substr)) {
            out->as += (*substr - '0');
            ++substr;
            --sublen;
            ++numdigits;
        }
    }

parse_timezone:
    /* trim any whitepsace between time/timeezone */
    while (sublen > 0 && isspace(*substr)) {
        ++substr;
        --sublen;
    }

    if (sublen == 0) {
        // Unlike NumPy, treating no time zone as naive
        goto finish;
    }

    /* UTC specifier */
    if (*substr == 'Z') {
        /* "Z" should be equivalent to tz offset "+00:00" */
        if (out_local != NULL) {
            *out_local = 1;
        }

        if (out_tzoffset != NULL) {
            *out_tzoffset = 0;
        }

        if (sublen == 1) {
            goto finish;
        } else {
            ++substr;
            --sublen;
        }
    } else if (*substr == '-' || *substr == '+') {
        /* Time zone offset */
        int offset_neg = 0, offset_hour = 0, offset_minute = 0;

        /*
         * Since "local" means local with respect to the current
         * machine, we say this is non-local.
         */

        if (*substr == '-') {
            offset_neg = 1;
        }
        ++substr;
        --sublen;

        /* The hours offset */
        if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
            offset_hour = 10 * (substr[0] - '0') + (substr[1] - '0');
            substr += 2;
            sublen -= 2;
            if (offset_hour >= 24) {
                printf("Timezone hours offset out of range "
                             "in datetime string \"%s\"",
                             str);
                goto error;
            }
        } else if (sublen >= 1 && isdigit(substr[0])) {
            offset_hour = substr[0] - '0';
            ++substr;
            --sublen;
        } else {
            goto parse_error;
        }

        /* The minutes offset is optional */
        if (sublen > 0) {
            /* Optional ':' */
            if (*substr == ':') {
                ++substr;
                --sublen;
            }

            /* The minutes offset (at the end of the string) */
            if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
                offset_minute = 10 * (substr[0] - '0') + (substr[1] - '0');
                substr += 2;
                sublen -= 2;
                if (offset_minute >= 60) {
                    printf("Timezone minutes offset out of range "
                                 "in datetime string \"%s\"",
                                 str);
                    goto error;
                }
            } else if (sublen >= 1 && isdigit(substr[0])) {
                offset_minute = substr[0] - '0';
                ++substr;
                --sublen;
            } else {
                goto parse_error;
            }
        }

        /* Apply the time zone offset */
        if (offset_neg) {
            offset_hour = -offset_hour;
            offset_minute = -offset_minute;
        }
        if (out_local != NULL) {
            *out_local = 1;
            // Unlike NumPy, do not change internal value to local time
            *out_tzoffset = 60 * offset_hour + offset_minute;
        }
    }

    /* Skip trailing whitespace */
    while (sublen > 0 && isspace(*substr)) {
        ++substr;
        --sublen;
    }

    if (sublen != 0) {
        goto parse_error;
    }

finish:
    return 0;

parse_error:
    printf("Error parsing datetime string \"%s\" at position %d", str,
                 (int)(substr - str));
    return -1;

error:
    return -1;
}

// FIXME use HPAT dist libs
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
#define ALIGN 32

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

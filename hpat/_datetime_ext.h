#ifndef _DATETIME_EXT_H_INCLUDED
#define _DATETIME_EXT_H_INCLUDED

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <iostream>
#include <numpy/arrayobject.h>

#include "_hpat_common.h"

extern "C"
{
    // #include "np_datetime.h"
    //
    // struct pd_timestamp {
    //     int64_t year;
    //     int64_t month;
    //     int64_t day;
    //     int64_t hour;
    //     int64_t minute;
    //     int64_t second;
    //     int64_t microsecond;
    //     int64_t nanosecond;
    // };
    //
    // void dt_to_timestamp(int64_t val, pd_timestamp *ts);

    // TODO: call Pandas C libs directly and remove copy paste

    typedef struct
    {
        npy_int64 year;
        npy_int32 month, day, hour, min, sec, us, ps, as;
    } pandas_datetimestruct;

    static int parse_iso_8601_datetime(
        char* str, int len, pandas_datetimestruct* out, int* out_local, int* out_tzoffset) __UNUSED__;

    typedef enum
    {
        PANDAS_FR_Y = 0, // Years
        PANDAS_FR_M = 1, // Months
        PANDAS_FR_W = 2, // Weeks
        // Gap where NPY_FR_B was
        PANDAS_FR_D = 4,       // Days
        PANDAS_FR_h = 5,       // hours
        PANDAS_FR_m = 6,       // minutes
        PANDAS_FR_s = 7,       // seconds
        PANDAS_FR_ms = 8,      // milliseconds
        PANDAS_FR_us = 9,      // microseconds
        PANDAS_FR_ns = 10,     // nanoseconds
        PANDAS_FR_ps = 11,     // picoseconds
        PANDAS_FR_fs = 12,     // femtoseconds
        PANDAS_FR_as = 13,     // attoseconds
        PANDAS_FR_GENERIC = 14 // Generic, unbound units, can
                               // convert to anything
    } PANDAS_DATETIMEUNIT;

    static int convert_datetimestruct_to_datetime(PANDAS_DATETIMEUNIT base,
                                                  const pandas_datetimestruct* dts,
                                                  npy_datetime* out) __UNUSED__;

    static void*
        np_datetime_date_array_from_packed_ints(uint64_t* dt_data, int64_t n_elems, PyObject* dt_date_class) __UNUSED__;

    // void dt_to_timestamp(int64_t val, pd_timestamp *ts) {
    //     pandas_datetimestruct out;
    //     pandas_datetime_to_datetimestruct(val, PANDAS_FR_ns, &out);
    //     ts->year = out.year;
    //     ts->month = out.month;
    //     ts->day = out.day;
    //     ts->hour = out.hour;
    //     ts->minute = out.min;
    //     ts->second = out.sec;
    //     ts->microsecond = out.us;
    //     ts->nanosecond = out.ps * 1000;
    // }

    static inline PyObject* py_datetime_date_from_packed_int(uint64_t dt, PyObject* dt_date_class)
    {
        uint64_t year = dt >> 32;
        uint64_t month = (dt >> 16) & 0xFFFF;
        uint64_t day = dt & 0xFFFF;
        return PyObject_CallFunction(dt_date_class, "iii", year, month, day);
    }

    // given an array of packed integers for datetime.date (pd_timestamp_ext format),
    // create and return a pd.Series of datetime.date() objects
    static void* np_datetime_date_array_from_packed_ints(uint64_t* dt_data, int64_t n_elems, PyObject* dt_date_class)
    {
#define CHECK(expr, msg)                                                                                               \
    if (!(expr))                                                                                                       \
    {                                                                                                                  \
        std::cerr << msg << std::endl;                                                                                 \
        PyGILState_Release(gilstate);                                                                                  \
        return NULL;                                                                                                   \
    }
        auto gilstate = PyGILState_Ensure();

        npy_intp dims[] = {n_elems};
        PyObject* ret = PyArray_SimpleNew(1, dims, NPY_OBJECT);
        CHECK(ret, "allocating numpy array failed");

        for (int64_t i = 0; i < n_elems; ++i)
        {
            PyObject* s = py_datetime_date_from_packed_int(dt_data[i], dt_date_class);
            CHECK(s, "creating Python datetime.date object failed");
            auto p = PyArray_GETPTR1((PyArrayObject*)ret, i);
            CHECK(p, "getting offset in numpy array failed");
            int err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, s);
            CHECK(err == 0, "setting item in numpy array failed");
            Py_DECREF(s);
        }

        PyGILState_Release(gilstate);
        return ret;
#undef CHECK
    }

    // XXX copy paste from Pandas for parse_iso_8601_datetime

    static int is_leapyear(npy_int64 year)
    {
        return (year & 0x3) == 0 && /* year % 4 == 0 */
               ((year % 100) != 0 || (year % 400) == 0);
    }

    static const int days_per_month_table[2][12] = {{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
                                                    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}};

    static int
        parse_iso_8601_datetime(char* str, int len, pandas_datetimestruct* out, int* out_local, int* out_tzoffset)
    {
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
        while (sublen > 0 && isspace(*substr))
        {
            ++substr;
            --sublen;
        }

        /* Leading '-' sign for negative year */
        if (*substr == '-')
        {
            ++substr;
            --sublen;
        }

        if (sublen == 0)
        {
            goto parse_error;
        }

        /* PARSE THE YEAR (4 digits) */
        out->year = 0;
        if (sublen >= 4 && isdigit(substr[0]) && isdigit(substr[1]) && isdigit(substr[2]) && isdigit(substr[3]))
        {
            out->year = 1000 * (substr[0] - '0') + 100 * (substr[1] - '0') + 10 * (substr[2] - '0') + (substr[3] - '0');

            substr += 4;
            sublen -= 4;
        }

        /* Negate the year if necessary */
        if (str[0] == '-')
        {
            out->year = -out->year;
        }
        /* Check whether it's a leap-year */
        year_leap = is_leapyear(out->year);

        /* Next character must be a separator, start of month, or end of string */
        if (sublen == 0)
        {
            if (out_local != NULL)
            {
                *out_local = 0;
            }
            goto finish;
        }

        if (!isdigit(*substr))
        {
            for (i = 0; i < valid_ymd_sep_len; ++i)
            {
                if (*substr == valid_ymd_sep[i])
                {
                    break;
                }
            }
            if (i == valid_ymd_sep_len)
            {
                goto parse_error;
            }
            has_ymd_sep = 1;
            ymd_sep = valid_ymd_sep[i];
            ++substr;
            --sublen;
            /* Cannot have trailing separator */
            if (sublen == 0 || !isdigit(*substr))
            {
                goto parse_error;
            }
        }

        /* PARSE THE MONTH */
        /* First digit required */
        out->month = (*substr - '0');
        ++substr;
        --sublen;
        /* Second digit optional if there was a separator */
        if (isdigit(*substr))
        {
            out->month = 10 * out->month + (*substr - '0');
            ++substr;
            --sublen;
        }
        else if (!has_ymd_sep)
        {
            goto parse_error;
        }
        if (out->month < 1 || out->month > 12)
        {
            printf("Month out of range in datetime string \"%s\"", str);
            goto error;
        }

        /* Next character must be the separator, start of day, or end of string */
        if (sublen == 0)
        {
            /* Forbid YYYYMM. Parsed instead as YYMMDD by someone else. */
            if (!has_ymd_sep)
            {
                goto parse_error;
            }
            if (out_local != NULL)
            {
                *out_local = 0;
            }
            goto finish;
        }

        if (has_ymd_sep)
        {
            /* Must have separator, but cannot be trailing */
            if (*substr != ymd_sep || sublen == 1)
            {
                goto parse_error;
            }
            ++substr;
            --sublen;
        }

        /* PARSE THE DAY */
        /* First digit required */
        if (!isdigit(*substr))
        {
            goto parse_error;
        }
        out->day = (*substr - '0');
        ++substr;
        --sublen;
        /* Second digit optional if there was a separator */
        if (isdigit(*substr))
        {
            out->day = 10 * out->day + (*substr - '0');
            ++substr;
            --sublen;
        }
        else if (!has_ymd_sep)
        {
            goto parse_error;
        }
        if (out->day < 1 || out->day > days_per_month_table[year_leap][out->month - 1])
        {
            printf("Day out of range in datetime string \"%s\"", str);
            goto error;
        }

        /* Next character must be a 'T', ' ', or end of string */
        if (sublen == 0)
        {
            if (out_local != NULL)
            {
                *out_local = 0;
            }
            goto finish;
        }

        if ((*substr != 'T' && *substr != ' ') || sublen == 1)
        {
            goto parse_error;
        }
        ++substr;
        --sublen;

        /* PARSE THE HOURS */
        /* First digit required */
        if (!isdigit(*substr))
        {
            goto parse_error;
        }
        out->hour = (*substr - '0');
        ++substr;
        --sublen;
        /* Second digit optional */
        if (isdigit(*substr))
        {
            hour_was_2_digits = 1;
            out->hour = 10 * out->hour + (*substr - '0');
            ++substr;
            --sublen;
            if (out->hour >= 24)
            {
                printf("Hours out of range in datetime string \"%s\"", str);
                goto error;
            }
        }

        /* Next character must be a ':' or the end of the string */
        if (sublen == 0)
        {
            if (!hour_was_2_digits)
            {
                goto parse_error;
            }
            goto finish;
        }

        if (*substr == ':')
        {
            has_hms_sep = 1;
            ++substr;
            --sublen;
            /* Cannot have a trailing separator */
            if (sublen == 0 || !isdigit(*substr))
            {
                goto parse_error;
            }
        }
        else if (!isdigit(*substr))
        {
            if (!hour_was_2_digits)
            {
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
        if (isdigit(*substr))
        {
            out->min = 10 * out->min + (*substr - '0');
            ++substr;
            --sublen;
            if (out->min >= 60)
            {
                printf("Minutes out of range in datetime string \"%s\"", str);
                goto error;
            }
        }
        else if (!has_hms_sep)
        {
            goto parse_error;
        }

        if (sublen == 0)
        {
            goto finish;
        }

        /* If we make it through this condition block, then the next
         * character is a digit. */
        if (has_hms_sep && *substr == ':')
        {
            ++substr;
            --sublen;
            /* Cannot have a trailing ':' */
            if (sublen == 0 || !isdigit(*substr))
            {
                goto parse_error;
            }
        }
        else if (!has_hms_sep && isdigit(*substr))
        {
        }
        else
        {
            goto parse_timezone;
        }

        /* PARSE THE SECONDS */
        /* First digit required */
        out->sec = (*substr - '0');
        ++substr;
        --sublen;
        /* Second digit optional if there was a separator */
        if (isdigit(*substr))
        {
            out->sec = 10 * out->sec + (*substr - '0');
            ++substr;
            --sublen;
            if (out->sec >= 60)
            {
                printf("Seconds out of range in datetime string \"%s\"", str);
                goto error;
            }
        }
        else if (!has_hms_sep)
        {
            goto parse_error;
        }

        /* Next character may be a '.' indicating fractional seconds */
        if (sublen > 0 && *substr == '.')
        {
            ++substr;
            --sublen;
        }
        else
        {
            goto parse_timezone;
        }

        /* PARSE THE MICROSECONDS (0 to 6 digits) */
        numdigits = 0;
        for (i = 0; i < 6; ++i)
        {
            out->us *= 10;
            if (sublen > 0 && isdigit(*substr))
            {
                out->us += (*substr - '0');
                ++substr;
                --sublen;
                ++numdigits;
            }
        }

        if (sublen == 0 || !isdigit(*substr))
        {
            goto parse_timezone;
        }

        /* PARSE THE PICOSECONDS (0 to 6 digits) */
        numdigits = 0;
        for (i = 0; i < 6; ++i)
        {
            out->ps *= 10;
            if (sublen > 0 && isdigit(*substr))
            {
                out->ps += (*substr - '0');
                ++substr;
                --sublen;
                ++numdigits;
            }
        }

        if (sublen == 0 || !isdigit(*substr))
        {
            goto parse_timezone;
        }

        /* PARSE THE ATTOSECONDS (0 to 6 digits) */
        numdigits = 0;
        for (i = 0; i < 6; ++i)
        {
            out->as *= 10;
            if (sublen > 0 && isdigit(*substr))
            {
                out->as += (*substr - '0');
                ++substr;
                --sublen;
                ++numdigits;
            }
        }

    parse_timezone:
        /* trim any whitepsace between time/timeezone */
        while (sublen > 0 && isspace(*substr))
        {
            ++substr;
            --sublen;
        }

        if (sublen == 0)
        {
            // Unlike NumPy, treating no time zone as naive
            goto finish;
        }

        /* UTC specifier */
        if (*substr == 'Z')
        {
            /* "Z" should be equivalent to tz offset "+00:00" */
            if (out_local != NULL)
            {
                *out_local = 1;
            }

            if (out_tzoffset != NULL)
            {
                *out_tzoffset = 0;
            }

            if (sublen == 1)
            {
                goto finish;
            }
            else
            {
                ++substr;
                --sublen;
            }
        }
        else if (*substr == '-' || *substr == '+')
        {
            /* Time zone offset */
            int offset_neg = 0, offset_hour = 0, offset_minute = 0;

            /*
             * Since "local" means local with respect to the current
             * machine, we say this is non-local.
             */

            if (*substr == '-')
            {
                offset_neg = 1;
            }
            ++substr;
            --sublen;

            /* The hours offset */
            if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1]))
            {
                offset_hour = 10 * (substr[0] - '0') + (substr[1] - '0');
                substr += 2;
                sublen -= 2;
                if (offset_hour >= 24)
                {
                    printf(
                        "Timezone hours offset out of range "
                        "in datetime string \"%s\"",
                        str);
                    goto error;
                }
            }
            else if (sublen >= 1 && isdigit(substr[0]))
            {
                offset_hour = substr[0] - '0';
                ++substr;
                --sublen;
            }
            else
            {
                goto parse_error;
            }

            /* The minutes offset is optional */
            if (sublen > 0)
            {
                /* Optional ':' */
                if (*substr == ':')
                {
                    ++substr;
                    --sublen;
                }

                /* The minutes offset (at the end of the string) */
                if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1]))
                {
                    offset_minute = 10 * (substr[0] - '0') + (substr[1] - '0');
                    substr += 2;
                    sublen -= 2;
                    if (offset_minute >= 60)
                    {
                        printf(
                            "Timezone minutes offset out of range "
                            "in datetime string \"%s\"",
                            str);
                        goto error;
                    }
                }
                else if (sublen >= 1 && isdigit(substr[0]))
                {
                    offset_minute = substr[0] - '0';
                    ++substr;
                    --sublen;
                }
                else
                {
                    goto parse_error;
                }
            }

            /* Apply the time zone offset */
            if (offset_neg)
            {
                offset_hour = -offset_hour;
                offset_minute = -offset_minute;
            }
            if (out_local != NULL)
            {
                *out_local = 1;
                // Unlike NumPy, do not change internal value to local time
                *out_tzoffset = 60 * offset_hour + offset_minute;
            }
        }

        /* Skip trailing whitespace */
        while (sublen > 0 && isspace(*substr))
        {
            ++substr;
            --sublen;
        }

        if (sublen != 0)
        {
            goto parse_error;
        }

    finish:
        return 0;

    parse_error:
        printf("Error parsing datetime string \"%s\" at position %d", str, (int)(substr - str));
        return -1;

    error:
        return -1;
    }

    // XXX copy paste from Pandas for convert_datetimestruct_to_datetime

    static npy_int64 get_datetimestruct_days(const pandas_datetimestruct* dts)
    {
        int i, month;
        npy_int64 year, days = 0;
        const int* month_lengths;

        year = dts->year - 1970;
        days = year * 365;

        /* Adjust for leap years */
        if (days >= 0)
        {
            /*
             * 1968 is the closest leap year before 1970.
             * Exclude the current year, so add 1.
             */
            year += 1;
            /* Add one day for each 4 years */
            days += year / 4;
            /* 1900 is the closest previous year divisible by 100 */
            year += 68;
            /* Subtract one day for each 100 years */
            days -= year / 100;
            /* 1600 is the closest previous year divisible by 400 */
            year += 300;
            /* Add one day for each 400 years */
            days += year / 400;
        }
        else
        {
            /*
             * 1972 is the closest later year after 1970.
             * Include the current year, so subtract 2.
             */
            year -= 2;
            /* Subtract one day for each 4 years */
            days += year / 4;
            /* 2000 is the closest later year divisible by 100 */
            year -= 28;
            /* Add one day for each 100 years */
            days -= year / 100;
            /* 2000 is also the closest later year divisible by 400 */
            /* Subtract one day for each 400 years */
            days += year / 400;
        }

        month_lengths = days_per_month_table[is_leapyear(dts->year)];
        month = dts->month - 1;

        /* Add the months */
        for (i = 0; i < month; ++i)
        {
            days += month_lengths[i];
        }

        /* Add the days */
        days += dts->day - 1;

        return days;
    }

    static int convert_datetimestruct_to_datetime(PANDAS_DATETIMEUNIT base,
                                                  const pandas_datetimestruct* dts,
                                                  npy_datetime* out)
    {
        npy_datetime ret;

        if (base == PANDAS_FR_Y)
        {
            /* Truncate to the year */
            ret = dts->year - 1970;
        }
        else if (base == PANDAS_FR_M)
        {
            /* Truncate to the month */
            ret = 12 * (dts->year - 1970) + (dts->month - 1);
        }
        else
        {
            /* Otherwise calculate the number of days to start */
            npy_int64 days = get_datetimestruct_days(dts);

            switch (base)
            {
            case PANDAS_FR_W:
                /* Truncate to weeks */
                if (days >= 0)
                {
                    ret = days / 7;
                }
                else
                {
                    ret = (days - 6) / 7;
                }
                break;
            case PANDAS_FR_D:
                ret = days;
                break;
            case PANDAS_FR_h:
                ret = days * 24 + dts->hour;
                break;
            case PANDAS_FR_m:
                ret = (days * 24 + dts->hour) * 60 + dts->min;
                break;
            case PANDAS_FR_s:
                ret = ((days * 24 + dts->hour) * 60 + dts->min) * 60 + dts->sec;
                break;
            case PANDAS_FR_ms:
                ret = (((days * 24 + dts->hour) * 60 + dts->min) * 60 + dts->sec) * 1000 + dts->us / 1000;
                break;
            case PANDAS_FR_us:
                ret = (((days * 24 + dts->hour) * 60 + dts->min) * 60 + dts->sec) * 1000000 + dts->us;
                break;
            case PANDAS_FR_ns:
                ret = ((((days * 24 + dts->hour) * 60 + dts->min) * 60 + dts->sec) * 1000000 + dts->us) * 1000 +
                      dts->ps / 1000;
                break;
            case PANDAS_FR_ps:
                ret = ((((days * 24 + dts->hour) * 60 + dts->min) * 60 + dts->sec) * 1000000 + dts->us) * 1000000 +
                      dts->ps;
                break;
            case PANDAS_FR_fs:
                /* only 2.6 hours */
                ret = (((((days * 24 + dts->hour) * 60 + dts->min) * 60 + dts->sec) * 1000000 + dts->us) * 1000000 +
                       dts->ps) *
                          1000 +
                      dts->as / 1000;
                break;
            case PANDAS_FR_as:
                /* only 9.2 secs */
                ret = (((((days * 24 + dts->hour) * 60 + dts->min) * 60 + dts->sec) * 1000000 + dts->us) * 1000000 +
                       dts->ps) *
                          1000000 +
                      dts->as;
                break;
            default:
                /* Something got corrupted */
                PyErr_SetString(PyExc_ValueError, "NumPy datetime metadata with corrupt unit value");
                return -1;
            }
        }

        *out = ret;

        return 0;
    }

} // extern "C"

#endif // _DATETIME_EXT_H_INCLUDED

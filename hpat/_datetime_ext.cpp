#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <string>
#include <iostream>
#include <vector>

#ifndef _WIN32
#include <glob.h>
#endif

#include "np_datetime.h"

extern "C" {

struct pd_timestamp {
    int64_t year;
    int64_t month;
    int64_t day;
    int64_t hour;
    int64_t minute;
    int64_t second;
    int64_t microsecond;
    int64_t nanosecond;
};

void dt_to_timestamp(int64_t val, pd_timestamp *ts);

PyMODINIT_FUNC PyInit_hdatetime_ext(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "hdatetime_ext", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    // init numpy
    import_array();

    PyObject_SetAttrString(m, "dt_to_timestamp",
                            PyLong_FromVoidPtr((void*)(&dt_to_timestamp)));

    return m;
}

void dt_to_timestamp(int64_t val, pd_timestamp *ts) {
    pandas_datetimestruct out;
    pandas_datetime_to_datetimestruct(val, PANDAS_FR_ns, &out);
    ts->year = out.year;
    ts->month = out.month;
    ts->day = out.day;
    ts->hour = out.hour;
    ts->minute = out.min;
    ts->second = out.sec;
    ts->microsecond = out.us;
    ts->nanosecond = out.ps * 1000;
}

} // extern "C"

//*****************************************************************************
// Copyright (c) 2019, Intel Corporation All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

#include "_datetime_ext.h"

extern "C"
{
    PyMODINIT_FUNC PyInit_hdatetime_ext(void)
    {
        PyObject* m;
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "hdatetime_ext",
            "No docs",
            -1,
            NULL,
        };
        m = PyModule_Create(&moduledef);
        if (m == NULL)
        {
            return NULL;
        }

        // init numpy
        import_array();
        //
        // PyObject_SetAttrString(m, "dt_to_timestamp",
        //                         PyLong_FromVoidPtr((void*)(&dt_to_timestamp)));

        PyObject_SetAttrString(m,
                               "np_datetime_date_array_from_packed_ints",
                               PyLong_FromVoidPtr((void*)(&np_datetime_date_array_from_packed_ints)));

        PyObject_SetAttrString(m, "parse_iso_8601_datetime", PyLong_FromVoidPtr((void*)(&parse_iso_8601_datetime)));
        PyObject_SetAttrString(
            m, "convert_datetimestruct_to_datetime", PyLong_FromVoidPtr((void*)(&convert_datetimestruct_to_datetime)));

        return m;
    }

} // extern "C"

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

#include <Python.h>
#include <climits>
#include <cstdio>
#include <iostream>
#include <string>

#include "_csv.h"

void file_read(char* file_name, void* buff, int64_t size)
{
    FILE* fp = fopen(file_name, "rb");
    if (fp == NULL)
    {
        return;
    }

    size_t ret_code = fread(buff, 1, (size_t)size, fp);
    if (ret_code != (size_t)size)
    {
        std::cerr << "File read error: " << file_name << '\n';
    }
    fclose(fp);
    return;
}

void file_write(char* file_name, void* buff, int64_t size)
{
    FILE* fp = fopen(file_name, "wb");
    if (fp == NULL)
    {
        return;
    }

    size_t ret_code = fwrite(buff, 1, (size_t)size, fp);
    if (ret_code != (size_t)size)
    {
        std::cerr << "File write error: " << file_name << '\n';
    }
    fclose(fp);

    return;
}

PyMODINIT_FUNC PyInit_hio(void)
{
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "hio",
        "No docs",
        -1,
        NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
    {
        return NULL;
    }

    // numpy read
    PyObject_SetAttrString(m, "file_read", PyLong_FromVoidPtr((void*)(&file_read)));
    PyObject_SetAttrString(m, "file_write", PyLong_FromVoidPtr((void*)(&file_write)));

    PyInit_csv(m);

    return m;
}

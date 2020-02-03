//*****************************************************************************
// Copyright (c) 2020, Intel Corporation All rights reserved.
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

#ifndef _IMPORT_PY_INCLUDED
#define _IMPORT_PY_INCLUDED

#include <Python.h>

/* Import 'sym' from module 'module'.
 */
static PyObject* import_sym(const char* module, const char* sym) __UNUSED__
{
#define CHECK(expr, msg)                                                                                               \
    if (!(expr))                                                                                                       \
    {                                                                                                                  \
        std::cerr << msg << std::endl;                                                                                 \
        PyGILState_Release(gilstate);                                                                                  \
        return NULL;                                                                                                   \
    }
    PyObject* mod = NULL;
    PyObject* func = NULL;
    auto gilstate = PyGILState_Ensure();

    mod = PyImport_ImportModule(module);
    CHECK(mod, "importing failed");
    func = PyObject_GetAttrString(mod, sym);
    CHECK(func, "getting symbol from module failed");

    Py_XDECREF(mod);
    PyGILState_Release(gilstate);

    return func;
#undef CHECK
}

#endif // _IMPORT_PY_INCLUDED

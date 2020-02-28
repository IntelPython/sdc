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

#include <Python.h>
#include <daal.h>


using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::services;


extern "C"
{

int test(int x)
{
    return x + 42;
}

double sum(double *p, int c)
{
    double result = 0.0;
    for (int i = 0; i < c; ++i)
    {
        result += p[i];
    }
    return result;
}

double median(int nRows, double *ptr)
{
    quantiles::Batch<double> algorithm;
    algorithm.input.set(quantiles::data, HomogenNumericTable<double>::create(ptr, 1, nRows));
    algorithm.compute();
    return algorithm.getResult()->get(quantiles::quantiles)->getValue<double>(0, 0);
}

void quantile(const double * data, const __int64_t nFeatures, const __int64_t nVectors,
              const __int64_t quantOrderN, const double * quantOrder,
              double * quants)
{
    Environment::getInstance()->setNumberOfThreads(4);  // does not affect

    // quantiles::Batch<> algorithm;
    quantiles::Batch<double> algorithm;  // 2 times faster
    algorithm.parameter.quantileOrders = HomogenNumericTable<double>::create(quantOrder, 1, quantOrderN);

    algorithm.input.set(quantiles::data, HomogenNumericTable<double>::create(data, nFeatures, nVectors));

    algorithm.compute();

    auto out_table = algorithm.getResult()->get(quantiles::quantiles);
    for (int i = 0; i < quantOrderN; ++i)
        quants[i] = out_table->getValue<double>(0, i);
}

PyMODINIT_FUNC PyInit_daal()
{
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "daal",
        "No docs",
        -1,
        NULL,
    };
    PyObject* m = PyModule_Create(&moduledef);
    if (m == NULL)
    {
        return NULL;
    }

#define REGISTER(func) PyObject_SetAttrString(m, #func, PyLong_FromVoidPtr((void*)(&func)));
    REGISTER(test)
    REGISTER(sum)
    REGISTER(median)
    REGISTER(quantile)
#undef REGISTER
    return m;
}

}  // extern "C"

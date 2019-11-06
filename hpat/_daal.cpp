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
#include <unordered_set>

#include "daal.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
#define mpi_root 0

struct svc_payload
{
    services::SharedPtr<multi_class_classifier::training::Result>* trainingResultPtr;
    int64_t n_classes;
};

struct mnb_payload
{
    services::SharedPtr<multinomial_naive_bayes::training::Result>* trainingResultPtr;
    int64_t n_classes;
};

void* svc_train(int64_t num_features, int64_t num_samples, double* X, double* y, int64_t* n_classes_ptr);
void svc_predict(void* model_ptr, int64_t num_features, int64_t num_samples, double* p, double* res, int64_t n_classes);
void dtor_svc(void* model_ptr, int64_t size, void* in);

void* mnb_train(int64_t num_features, int64_t num_samples, int* X, int* y, int64_t* n_classes_ptr);
void mnb_predict(void* model_ptr, int64_t num_features, int64_t num_samples, int* p, int* res, int64_t n_classes);
void dtor_mnb(void* model_ptr, int64_t size, void* in);

PyMODINIT_FUNC PyInit_daal_wrapper(void)
{
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "daal_wrapper",
        "No docs",
        -1,
        NULL,
    };

    m = PyModule_Create(&moduledef);
    if (m == NULL)
    {
        return NULL;
    }

    PyObject_SetAttrString(m, "svc_train", PyLong_FromVoidPtr((void*)(&svc_train)));
    PyObject_SetAttrString(m, "svc_predict", PyLong_FromVoidPtr((void*)(&svc_predict)));
    PyObject_SetAttrString(m, "dtor_svc", PyLong_FromVoidPtr((void*)(&dtor_svc)));
    PyObject_SetAttrString(m, "mnb_train", PyLong_FromVoidPtr((void*)(&mnb_train)));
    PyObject_SetAttrString(m, "mnb_predict", PyLong_FromVoidPtr((void*)(&mnb_predict)));
    PyObject_SetAttrString(m, "dtor_mnb", PyLong_FromVoidPtr((void*)(&dtor_mnb)));

    return m;
}

template <class T>
int64_t get_num_classes(T* y, int64_t num_samples)
{
    std::unordered_set<T> vals;
    for (int64_t i = 0; i < num_samples; i++)
    {
        vals.insert(y[i]);
    }

    return vals.size();
}

void* svc_train(int64_t num_features, int64_t num_samples, double* X, double* y, int64_t* n_classes_ptr)
{
    int64_t n_classes = *n_classes_ptr;
    // if number of classes is not known, count label values and assign to ptr
    // to update SVC data
    if (n_classes == -1)
    {
        n_classes = get_num_classes(y, num_samples);
    }
    *n_classes_ptr = n_classes;
    // printf("svn_train nFeatures:%ld nSamples:%ld X[0]:%lf y[0]:%lf\n", num_features, num_samples, X[0], y[0]);
    // printf("train classes: %lld\n", n_classes);
    services::SharedPtr<svm::training::Batch<>> training(new svm::training::Batch<>());
    services::SharedPtr<multi_class_classifier::training::Result> trainingResult;
    services::SharedPtr<svm::prediction::Batch<>> prediction(new svm::prediction::Batch<>());
    services::SharedPtr<kernel_function::KernelIface> kernel(new kernel_function::linear::Batch<>());
    training->parameter.cacheSize = 100000000;
    training->parameter.kernel = kernel;
    prediction->parameter.kernel = kernel;

    services::SharedPtr<HomogenNumericTable<double>> trainData =
        HomogenNumericTable<double>::create(X, num_features, num_samples);
    services::SharedPtr<HomogenNumericTable<double>> trainGroundTruth =
        HomogenNumericTable<double>::create(y, 1, num_samples);
    // printf("label rows: %ld\n", trainGroundTruth->getNumberOfRows());

    multi_class_classifier::training::Batch<> algorithm;

    algorithm.parameter.nClasses = n_classes;
    algorithm.parameter.training = training;
    algorithm.parameter.prediction = prediction;

    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);

    algorithm.compute();

    trainingResult = algorithm.getResult();
    // FIXME: return pointer to SharedPtr since get/set functions don't work
    services::SharedPtr<multi_class_classifier::training::Result>* ptres =
        new services::SharedPtr<multi_class_classifier::training::Result>();
    *ptres = trainingResult;
    return ptres;
}

void svc_predict(void* model_ptr, int64_t num_features, int64_t num_samples, double* p, double* res, int64_t n_classes)
{
    // printf("predict classes: %lld\n", n_classes);
    services::SharedPtr<multi_class_classifier::training::Result>* trainingResultPtr =
        (services::SharedPtr<multi_class_classifier::training::Result>*)(model_ptr);
    services::SharedPtr<classifier::prediction::Result> predictionResult;
    services::SharedPtr<svm::training::Batch<>> training(new svm::training::Batch<>());
    services::SharedPtr<svm::prediction::Batch<>> prediction(new svm::prediction::Batch<>());
    services::SharedPtr<kernel_function::KernelIface> kernel(new kernel_function::linear::Batch<>());

    training->parameter.cacheSize = 100000000;
    training->parameter.kernel = kernel;
    prediction->parameter.kernel = kernel;

    services::SharedPtr<HomogenNumericTable<double>> testData =
        HomogenNumericTable<double>::create(p, num_features, num_samples);

    multi_class_classifier::prediction::Batch<> algorithm;

    algorithm.parameter.nClasses = n_classes;
    algorithm.parameter.training = training;
    algorithm.parameter.prediction = prediction;

    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model, (*trainingResultPtr)->get(classifier::training::model));

    algorithm.compute();

    predictionResult = algorithm.getResult();
    NumericTablePtr res_table = predictionResult->get(classifier::prediction::prediction);
    BlockDescriptor<double> block1;
    res_table->getBlockOfRows(0, num_samples, readOnly, block1);
    double* data_ptr = block1.getBlockPtr();
    // printf("%lf %lf\n", data_ptr[0], data_ptr[1]);
    memcpy(res, data_ptr, num_samples * sizeof(double));
    res_table->releaseBlockOfRows(block1);
    return;
}

void dtor_svc(void* model_ptr, int64_t size, void* in)
{
    svc_payload* st = (svc_payload*)model_ptr;
    delete st->trainingResultPtr;
    return;
}

void* mnb_train(int64_t num_features, int64_t num_samples, int* X, int* y, int64_t* n_classes_ptr)
{
    int rankId, num_pes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
    size_t nBlocks = num_pes;

    int64_t n_classes = *n_classes_ptr;
    // if number of classes is not known, count label values and assign to ptr
    // to update MNB data
    if (n_classes == -1)
        n_classes = get_num_classes(y, num_samples);
    *n_classes_ptr = n_classes;

    // printf("mnb_train nClasses:%ld nFeatures:%ld nSamples:%ld X[0]:%ld y[0]:%ld\n",
    //         n_classes, num_features, num_samples, X[0], y[0]);

    services::SharedPtr<HomogenNumericTable<int>> trainData =
        HomogenNumericTable<int>::create(X, num_features, num_samples);
    services::SharedPtr<HomogenNumericTable<int>> trainGroundTruth =
        HomogenNumericTable<int>::create(y, 1, num_samples);

    multinomial_naive_bayes::training::ResultPtr trainingResult;
    multinomial_naive_bayes::training::Distributed<step1Local> localAlgorithm(n_classes);

    localAlgorithm.input.set(classifier::training::data, trainData);
    localAlgorithm.input.set(classifier::training::labels, trainGroundTruth);
    localAlgorithm.compute();

    services::SharedPtr<byte> serializedData;
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();

    if (rankId == mpi_root)
    {
        serializedData.reset(new byte[perNodeArchLength * nBlocks]);
    }
    {
        services::SharedPtr<byte> nodeResults(new byte[perNodeArchLength]);
        dataArch.copyArchiveToArray(nodeResults.get(), perNodeArchLength);

        MPI_Gather(nodeResults.get(),
                   perNodeArchLength,
                   MPI_CHAR,
                   serializedData.get(),
                   perNodeArchLength,
                   MPI_CHAR,
                   mpi_root,
                   MPI_COMM_WORLD);
    }

    if (rankId == mpi_root)
    {
        multinomial_naive_bayes::training::Distributed<step2Master> masterAlgorithm(n_classes);

        for (size_t i = 0; i < nBlocks; i++)
        {
            OutputDataArchive dataArch(serializedData.get() + perNodeArchLength * i, perNodeArchLength);

            multinomial_naive_bayes::training::PartialResultPtr dataForStep2FromStep1(
                new multinomial_naive_bayes::training::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            masterAlgorithm.input.add(multinomial_naive_bayes::training::partialModels, dataForStep2FromStep1);
        }

        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        trainingResult = masterAlgorithm.getResult();
    }

    // broadcast model
    size_t modelArchLength = 0;
    services::SharedPtr<byte> serializedModel;
    if (rankId == mpi_root)
    {
        InputDataArchive modelArch;
        trainingResult->serialize(modelArch);
        modelArchLength = modelArch.getSizeOfArchive();
        serializedModel.reset(new byte[modelArchLength]);
        modelArch.copyArchiveToArray(serializedModel.get(), modelArchLength);
    }
    // broadcast model size to enable memory allocation
    MPI_Bcast(&modelArchLength, sizeof(size_t), MPI_CHAR, mpi_root, MPI_COMM_WORLD);
    if (rankId != mpi_root)
        serializedModel.reset(new byte[modelArchLength]);
    MPI_Bcast(serializedModel.get(), modelArchLength, MPI_CHAR, mpi_root, MPI_COMM_WORLD);
    if (rankId != mpi_root)
    {
        OutputDataArchive modelArch(serializedModel.get(), modelArchLength);
        trainingResult.reset(new multinomial_naive_bayes::training::Result());
        trainingResult->deserialize(modelArch);
    }

    // FIXME: return pointer to SharedPtr since get/set functions don't work
    services::SharedPtr<multinomial_naive_bayes::training::Result>* ptres =
        new services::SharedPtr<multinomial_naive_bayes::training::Result>();
    *ptres = trainingResult;
    return ptres;
}

void mnb_predict(void* model_ptr, int64_t num_features, int64_t num_samples, int* p, int* res, int64_t n_classes)
{
    // printf("predict mnb classes: %lld\n", n_classes);
    services::SharedPtr<multinomial_naive_bayes::training::Result>* trainingResult =
        (services::SharedPtr<multinomial_naive_bayes::training::Result>*)(model_ptr);
    services::SharedPtr<classifier::prediction::Result> predictionResult;

    services::SharedPtr<HomogenNumericTable<int>> testData =
        HomogenNumericTable<int>::create(p, num_features, num_samples);

    multinomial_naive_bayes::prediction::Batch<> algorithm(n_classes);

    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model, (*trainingResult)->get(classifier::training::model));

    algorithm.compute();

    predictionResult = algorithm.getResult();
    NumericTablePtr res_table = predictionResult->get(classifier::prediction::prediction);
    BlockDescriptor<int> block1;
    res_table->getBlockOfRows(0, num_samples, readOnly, block1);
    int* data_ptr = block1.getBlockPtr();
    // printf("%lf %lf\n", data_ptr[0], data_ptr[1]);
    memcpy(res, data_ptr, num_samples * sizeof(int));
    res_table->releaseBlockOfRows(block1);
    return;
}
void dtor_mnb(void* model_ptr, int64_t size, void* in)
{
    mnb_payload* st = (mnb_payload*)model_ptr;
    delete st->trainingResultPtr;
    return;
}

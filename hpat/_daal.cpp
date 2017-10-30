#include <Python.h>
#include "daal.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

void svc_train(int64_t num_features, int64_t num_samples, double* X, double *y);

PyMODINIT_FUNC PyInit_daal_wrapper(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "daal_wrapper", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "svc_train",
                            PyLong_FromVoidPtr((void*)(&svc_train)));

    return m;
}

void svc_train(int64_t num_features, int64_t num_samples, double* X, double *y)
{
    // printf("svn_train nFeatures:%ld nSamples:%ld X[0]:%lf y[0]:%lf\n", num_features, num_samples, X[0], y[0]);
    services::SharedPtr<svm::training::Batch<> > training(new svm::training::Batch<>());
    services::SharedPtr<multi_class_classifier::training::Result> trainingResult;
    services::SharedPtr<svm::prediction::Batch<> > prediction(new svm::prediction::Batch<>());
    services::SharedPtr<kernel_function::KernelIface> kernel(new kernel_function::linear::Batch<>());
    training->parameter.cacheSize = 100000000;
    training->parameter.kernel = kernel;
    prediction->parameter.kernel = kernel;

    services::SharedPtr< HomogenNumericTable< double > > trainData = HomogenNumericTable<double>::create(X, num_features, num_samples);
    services::SharedPtr< HomogenNumericTable< double > > trainGroundTruth = HomogenNumericTable<double>::create(y, 1, num_samples);
    // printf("label rows: %ld\n", trainGroundTruth->getNumberOfRows());

    multi_class_classifier::training::Batch<> algorithm;

    algorithm.parameter.nClasses = 2; // FIXME
    algorithm.parameter.training = training;
    algorithm.parameter.prediction = prediction;

    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);

    algorithm.compute();


    trainingResult = algorithm.getResult();
    return;
}

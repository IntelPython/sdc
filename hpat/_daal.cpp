#include <Python.h>
#include "daal.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

struct svc_payload {
    services::SharedPtr<multi_class_classifier::training::Result>* trainingResultPtr;
    int64_t n_classes;
};

void* svc_train(int64_t num_features, int64_t num_samples, double* X, double *y, int64_t n_classes);
void svc_predict(void* model_ptr, int64_t num_features, int64_t num_samples, double* p, double *res, int64_t n_classes);
void dtor_svc(void* model_ptr, int64_t size, void* in);

PyMODINIT_FUNC PyInit_daal_wrapper(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "daal_wrapper", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "svc_train",
                            PyLong_FromVoidPtr((void*)(&svc_train)));
    PyObject_SetAttrString(m, "svc_predict",
                            PyLong_FromVoidPtr((void*)(&svc_predict)));
    PyObject_SetAttrString(m, "dtor_svc",
                            PyLong_FromVoidPtr((void*)(&dtor_svc)));

    return m;
}


void* svc_train(int64_t num_features, int64_t num_samples, double* X, double *y, int64_t n_classes)
{
    // printf("svn_train nFeatures:%ld nSamples:%ld X[0]:%lf y[0]:%lf\n", num_features, num_samples, X[0], y[0]);
    printf("train classes: %lld\n", n_classes);
    services::SharedPtr<svm::training::Batch<> > training(new svm::training::Batch<>());
    services::SharedPtr<multi_class_classifier::training::Result> trainingResult;
    services::SharedPtr<svm::prediction::Batch<> > prediction(new svm::prediction::Batch<>());
    services::SharedPtr<kernel_function::KernelIface> kernel(new kernel_function::linear::Batch<>());
    training->parameter.cacheSize = 100000000;
    training->parameter.kernel = kernel;
    prediction->parameter.kernel = kernel;

    services::SharedPtr< HomogenNumericTable< double > > trainData =
            HomogenNumericTable<double>::create(X, num_features, num_samples);
    services::SharedPtr< HomogenNumericTable< double > > trainGroundTruth =
                        HomogenNumericTable<double>::create(y, 1, num_samples);
    // printf("label rows: %ld\n", trainGroundTruth->getNumberOfRows());

    multi_class_classifier::training::Batch<> algorithm;

    algorithm.parameter.nClasses = 2; // FIXME
    algorithm.parameter.training = training;
    algorithm.parameter.prediction = prediction;

    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);

    algorithm.compute();


    trainingResult = algorithm.getResult();
    // FIXME: return pointer to SharedPtr since get/set functions don't work
    services::SharedPtr<multi_class_classifier::training::Result> * ptres =
        new services::SharedPtr<multi_class_classifier::training::Result>();
    *ptres = trainingResult;
    return ptres;
}

void svc_predict(void* model_ptr, int64_t num_features, int64_t num_samples, double* p, double *res, int64_t n_classes)
{
    printf("predict classes: %lld\n", n_classes);
    services::SharedPtr<multi_class_classifier::training::Result>* trainingResultPtr =
        (services::SharedPtr<multi_class_classifier::training::Result>*)(model_ptr);
    services::SharedPtr<classifier::prediction::Result> predictionResult;
    services::SharedPtr<svm::training::Batch<> > training(new svm::training::Batch<>());
    services::SharedPtr<svm::prediction::Batch<> > prediction(new svm::prediction::Batch<>());
    services::SharedPtr<kernel_function::KernelIface> kernel(new kernel_function::linear::Batch<>());

    training->parameter.cacheSize = 100000000;
    training->parameter.kernel = kernel;
    prediction->parameter.kernel = kernel;

    services::SharedPtr< HomogenNumericTable< double > > testData =
            HomogenNumericTable<double>::create(p, num_features, num_samples);

    multi_class_classifier::prediction::Batch<> algorithm;

    algorithm.parameter.nClasses = 2; // FIXME
    algorithm.parameter.training = training;
    algorithm.parameter.prediction = prediction;

    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model,
                        (*trainingResultPtr)->get(classifier::training::model));

    algorithm.compute();

    predictionResult = algorithm.getResult();
    NumericTablePtr res_table = predictionResult->get(classifier::prediction::prediction);
    BlockDescriptor<double> block1;
    res_table->getBlockOfRows(0, num_samples, readOnly, block1);
    double *data_ptr = block1.getBlockPtr();
    // printf("%lf %lf\n", data_ptr[0], data_ptr[1]);
    memcpy(res, data_ptr, num_samples*sizeof(double));
    res_table->releaseBlockOfRows(block1);
    return;
}

void dtor_svc(void* model_ptr, int64_t size, void* in)
{
    svc_payload* st = (svc_payload*) model_ptr;
    delete st->trainingResultPtr;
    return;
}

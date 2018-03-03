#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <Python.h>
#include <string>

using namespace cv;

void* cv_imread(int64_t *shapes, uint8_t **data, std::string* file_name);
void cv_resize(int64_t new_rows, int64_t new_cols, uint8_t *data,
                uint8_t *in_data, int64_t rows, int64_t cols);
void cv_mat_release(cv::Mat *img);

PyMODINIT_FUNC PyInit_cv_wrapper(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "cv_wrapper", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "cv_imread",
                            PyLong_FromVoidPtr((void*)(&cv_imread)));
    PyObject_SetAttrString(m, "cv_mat_release",
                            PyLong_FromVoidPtr((void*)(&cv_mat_release)));
    PyObject_SetAttrString(m, "cv_resize",
                            PyLong_FromVoidPtr((void*)(&cv_resize)));
    return m;
}


void* cv_imread(int64_t *shapes, uint8_t **data, std::string* file_name)
{

    cv::Mat image;
    image = cv::imread(*file_name);
    if(!image.data )
    {
        std::cerr << "no image found" << '\n';
        return 0;
    }
    if (image.type()!=CV_8UC3)
    {
        std::cerr << "image is not uint8 (CV_8UC3)" << '\n';
        return 0;
    }
    if (!image.isContinuous())
    {
        std::cerr << "image is not continuous" << '\n';
        return 0;
    }
    if (image.channels()!=3)
    {
        std::cerr << "image is not 3 channels" << '\n';
        return 0;
    }
    // printf("size %d %d\n", image.size[0], image.size[1]);
    // image.size is MatSize, returns rows (height) first!
    shapes[0] = (int64_t) image.size[0];
    shapes[1] = (int64_t) image.size[1];
    shapes[2] = 3; // XXX: assuming 3 channels

    *data = image.data;
    return new cv::Mat(image);
}

void cv_resize(int64_t new_rows, int64_t new_cols, uint8_t *data,
                uint8_t *in_data, int64_t rows, int64_t cols)
{
    // printf("%lld %lld %lld %lld\n", rows, cols, new_rows, new_cols);
    // cv::Size takes columns (width) first!
    cv::Mat in_image(cv::Size((int)cols, (int)rows), CV_8UC3, (void*)in_data);
    // CV_MAT_CONT_FLAG
    cv::Mat out_image(cv::Size((int)new_cols, (int)new_rows), CV_8UC3, (void*)data);

    cv::resize(in_image, out_image, cv::Size((int)new_cols, (int)new_rows));
    return;
}

void cv_mat_release(cv::Mat *img)
{
    delete img;
    return;
}

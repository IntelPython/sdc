#include <opencv2/opencv.hpp>
#include <Python.h>

void* cv_imread(int64_t *shapes, uint8_t **data, std::string* file_name);
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
    if (image.depth()!=0)//cv::CV_8U)
    {
        std::cerr << "image is not uint8 (CV_8U)" << '\n';
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
    shapes[0] = (int64_t) image.size[0];
    shapes[1] = (int64_t) image.size[1];
    shapes[2] = 3; // XXX: assuming 3 channels

    *data = image.data;
    return new cv::Mat(image);
}

void cv_mat_release(cv::Mat *img)
{
    delete img;
    return;
}

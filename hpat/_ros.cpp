#include <Python.h>
#include <string>
#include <rosbag/bag.h>
#include <sensor_msgs/Image.h>

void* open_bag(std::string* fname);

PyMODINIT_FUNC PyInit_ros_cpp(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "ros_cpp", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "open_bag",
                            PyLong_FromVoidPtr((void*)(&open_bag)));


    return m;
}

void* open_bag(std::string* fname)
{
    rosbag::Bag* bag = new rosbag::Bag;
    bag->open(*fname);
    return bag;
}

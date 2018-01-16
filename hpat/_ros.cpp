#include <Python.h>
#include <string>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>

void* open_bag(std::string* fname);
int64_t get_msg_count(rosbag::Bag* bag);
void get_image_dims(int64_t* out_dims, rosbag::Bag* bag);

PyMODINIT_FUNC PyInit_ros_cpp(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "ros_cpp", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "open_bag",
                            PyLong_FromVoidPtr((void*)(&open_bag)));
    PyObject_SetAttrString(m, "get_msg_count",
                            PyLong_FromVoidPtr((void*)(&get_msg_count)));
    PyObject_SetAttrString(m, "get_image_dims",
                            PyLong_FromVoidPtr((void*)(&get_image_dims)));

    return m;
}

void* open_bag(std::string* fname)
{
    rosbag::Bag* bag = new rosbag::Bag;
    bag->open(*fname);
    return bag;
}

int64_t get_msg_count(rosbag::Bag* bag)
{
    rosbag::View view(*bag);
    // XXX: assuming size() always returns number of messages (undocumented)
    uint32_t num_msgs = view.size();
    return (int64_t)num_msgs;
}

void get_image_dims(int64_t* out_dims, rosbag::Bag* bag)
{
    rosbag::View view(*bag);
    rosbag::MessageInstance const msg = *view.begin();
    sensor_msgs::Image::ConstPtr im_msg = msg.instantiate<sensor_msgs::Image>();

    out_dims[0] = im_msg->height;
    out_dims[1] = im_msg->width;
    return;
}

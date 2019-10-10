#include <Python.h>
#include <boost/foreach.hpp>
#include <cstring>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <string>
#define foreach BOOST_FOREACH

void* open_bag(std::string* fname);
int64_t get_msg_count(rosbag::Bag* bag);
void get_image_dims(int64_t* out_dims, rosbag::Bag* bag);
int read_images(uint8_t* buff, rosbag::Bag* bag);
int read_images_parallel(uint8_t* buff, rosbag::Bag* bag, int64_t start, int64_t cout);

PyMODINIT_FUNC PyInit_ros_cpp(void)
{
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "ros_cpp",
        "No docs",
        -1,
        NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    PyObject_SetAttrString(m, "open_bag", PyLong_FromVoidPtr((void*)(&open_bag)));
    PyObject_SetAttrString(m, "get_msg_count", PyLong_FromVoidPtr((void*)(&get_msg_count)));
    PyObject_SetAttrString(m, "get_image_dims", PyLong_FromVoidPtr((void*)(&get_image_dims)));
    PyObject_SetAttrString(m, "read_images", PyLong_FromVoidPtr((void*)(&read_images)));
    PyObject_SetAttrString(m, "read_images_parallel", PyLong_FromVoidPtr((void*)(&read_images_parallel)));
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

int read_images(uint8_t* buff, rosbag::Bag* bag)
{
    // std::cout << "ros read images" << "\n";
    rosbag::View view(*bag);
    uint32_t height;
    uint32_t width;
    int channels = 3;
    int msg_no = 0;
    foreach (rosbag::MessageInstance const msg, view)
    {
        sensor_msgs::Image::ConstPtr im_msg = msg.instantiate<sensor_msgs::Image>();
        if (im_msg != NULL)
        {
            if (msg_no == 0)
            {
                height = im_msg->height;
                width = im_msg->width;
            }
            else
            {
                if (height != im_msg->height || width != im_msg->width)
                {
                    std::cerr << "ROS image height/width not consistent"
                              << "\n";
                    return -1;
                }
            }
            int img_size = height * width * channels;
            // std::cout << img_size << " " << im_msg->step << "\n";
            uint8_t* curr_buff = buff + img_size * msg_no;
            memcpy(curr_buff, im_msg->data.data(), img_size);
            msg_no++;
        }
    }
    return 0;
}

int read_images_parallel(uint8_t* buff, rosbag::Bag* bag, int64_t start, int64_t count)
{
    // std::cout << "ros read images" << "\n";
    rosbag::View view(*bag);
    uint32_t height = 0;
    uint32_t width = 0;
    int channels = 3;
    int64_t msg_no = 0;
    foreach (rosbag::MessageInstance const msg, view)
    {
        if (msg_no < start)
        {
            msg_no++;
            continue;
        }
        if (msg_no >= start + count)
            break;
        sensor_msgs::Image::ConstPtr im_msg = msg.instantiate<sensor_msgs::Image>();
        if (im_msg != NULL)
        {
            if (height == 0)
            {
                height = im_msg->height;
                width = im_msg->width;
            }
            else
            {
                if (height != im_msg->height || width != im_msg->width)
                {
                    std::cerr << "ROS image height/width not consistent"
                              << "\n";
                    return -1;
                }
            }
            int img_size = height * width * channels;
            // std::cout << img_size << " " << im_msg->step << "\n";
            uint8_t* curr_buff = buff + img_size * (msg_no - start);
            memcpy(curr_buff, im_msg->data.data(), img_size);
            msg_no++;
        }
    }
    return 0;
}

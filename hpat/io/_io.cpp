#include "_csv.h"
#include <Python.h>
#include <string>
#include <iostream>
#include <cstdio>
#include <climits>

extern "C" {

void file_read(char* file_name, void* buff, int64_t size)
{
    FILE* fp = fopen(file_name, "rb");
    if (fp == NULL) return;
    size_t ret_code = fread(buff, 1, (size_t)size, fp);
    if (ret_code != (size_t)size)
    {
        std::cerr << "File read error: " << file_name << '\n';
    }
    fclose(fp);
    return;
}

void file_write(char* file_name, void* buff, int64_t size)
{
    FILE* fp = fopen(file_name, "wb");
    if (fp == NULL) return;
    size_t ret_code = fwrite(buff, 1, (size_t)size, fp);
    if (ret_code != (size_t)size)
    {
        std::cerr << "File write error: " << file_name << '\n';
    }
    fclose(fp);
    return;
}

PyMODINIT_FUNC PyInit_hio(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "hio", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    // numpy read
    PyObject_SetAttrString(m, "file_read",
                            PyLong_FromVoidPtr((void*)(&file_read)));
    PyObject_SetAttrString(m, "file_write",
                            PyLong_FromVoidPtr((void*)(&file_write)));

    PyInit_csv(m);

    return m;
}
} // extern "C"

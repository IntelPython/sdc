#include <Python.h>

extern "C"
{
    void parallel_sort(void* begin, uint64_t len, uint64_t size, void* compare);

    void parallel_sort_i8(void* begin, uint64_t len);
    void parallel_sort_u8(void* begin, uint64_t len);
    void parallel_sort_i16(void* begin, uint64_t len);
    void parallel_sort_u16(void* begin, uint64_t len);
    void parallel_sort_i32(void* begin, uint64_t len);
    void parallel_sort_u32(void* begin, uint64_t len);
    void parallel_sort_i64(void* begin, uint64_t len);
    void parallel_sort_u64(void* begin, uint64_t len);

    void parallel_sort_f32(void* begin, uint64_t len);
    void parallel_sort_f64(void* begin, uint64_t len);

    void parallel_stable_sort(void* begin, uint64_t len, uint64_t size, void* compare);

    void parallel_stable_sort_i8(void* begin, uint64_t len);
    void parallel_stable_sort_u8(void* begin, uint64_t len);
    void parallel_stable_sort_i16(void* begin, uint64_t len);
    void parallel_stable_sort_u16(void* begin, uint64_t len);
    void parallel_stable_sort_i32(void* begin, uint64_t len);
    void parallel_stable_sort_u32(void* begin, uint64_t len);
    void parallel_stable_sort_i64(void* begin, uint64_t len);
    void parallel_stable_sort_u64(void* begin, uint64_t len);

    void parallel_stable_sort_f32(void* begin, uint64_t len);
    void parallel_stable_sort_f64(void* begin, uint64_t len);
}

PyMODINIT_FUNC PyInit_sort()
{
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "sort",
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
    REGISTER(parallel_sort)

    REGISTER(parallel_sort_i8)
    REGISTER(parallel_sort_u8)
    REGISTER(parallel_sort_i16)
    REGISTER(parallel_sort_u16)
    REGISTER(parallel_sort_i32)
    REGISTER(parallel_sort_u32)
    REGISTER(parallel_sort_i64)
    REGISTER(parallel_sort_u64)

    REGISTER(parallel_sort_f32)
    REGISTER(parallel_sort_f64)

    REGISTER(parallel_stable_sort)

    REGISTER(parallel_stable_sort_i8)
    REGISTER(parallel_stable_sort_u8)
    REGISTER(parallel_stable_sort_i16)
    REGISTER(parallel_stable_sort_u16)
    REGISTER(parallel_stable_sort_i32)
    REGISTER(parallel_stable_sort_u32)
    REGISTER(parallel_stable_sort_i64)
    REGISTER(parallel_stable_sort_u64)

    REGISTER(parallel_stable_sort_f32)
    REGISTER(parallel_stable_sort_f64)
#undef REGISTER
    return m;
}
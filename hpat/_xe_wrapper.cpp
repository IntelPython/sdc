
#include <Python.h>
#include "xe.h"
#include <iostream>
#include <vector>

inline void read_xe_row(uint8_t* &buf, uint8_t* &curr_arr, uint64_t tp_enum, bool do_read, int &len);

extern "C" {


static PyObject* get_schema(PyObject *self, PyObject *args);
int64_t get_column_size_xenon(std::string* dset, uint64_t col_id);
void read_xenon_col(std::string* dset, uint64_t col_id, uint8_t* arr, uint64_t* xe_typ_enums);
void read_xenon_col_str(std::string* dset, uint64_t col_id, uint32_t **out_offsets,
                                    uint8_t **out_data, uint64_t* xe_typ_enums);

inline int16_t get_2byte_val(uint8_t* buf);
inline int get_4byte_val(uint8_t* buf);
inline int64_t get_8byte_val(uint8_t* buf);

static PyMethodDef xe_wrapper_methods[] = {
    {
        "get_schema", get_schema, METH_VARARGS, // METH_STATIC
        "get the xenon dataset schema"
    },
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC PyInit_hxe_ext(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "hxe_ext", "No docs", -1, xe_wrapper_methods, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;


    PyObject_SetAttrString(m, "get_column_size_xenon",
                            PyLong_FromVoidPtr((void*)(&get_column_size_xenon)));
    PyObject_SetAttrString(m, "read_xenon_col",
                            PyLong_FromVoidPtr((void*)(&read_xenon_col)));
    PyObject_SetAttrString(m, "read_xenon_col_str",
                            PyLong_FromVoidPtr((void*)(&read_xenon_col_str)));

    return m;
}

#define MAX_SCHEMA_LEN 200

static PyObject* get_schema(PyObject *self, PyObject *args) {
#define CHECK(expr, msg) if(!(expr)){std::cerr << msg << std::endl; return NULL;}

    const char* dset_name;
    char *read_schema = (char *) malloc(MAX_SCHEMA_LEN * sizeof(uint8_t));
    uint64_t fanout;

    CHECK(PyArg_ParseTuple(args, "s", &dset_name), "xenon dataset name expected");

    xe_connection_t xe_connection = xe_connect("localhost:41000");

	if (!xe_connection) {
		printf ("Fail to connect to Xenon.\n");
		return NULL;
	}

    xe_dataset_t xe_dataset = xe_open(xe_connection, dset_name, 0, 0, XE_O_READONLY);

    if (!xe_dataset) {
        printf("Fail to open dataset.\n");
        return NULL;
    }

    if (xe_exists(xe_connection, dset_name, (const char **)&read_schema, &fanout)) {
        printf("Fail to check exist.\n");
        return NULL;
    }

    xe_close(xe_connection, xe_dataset);
    xe_disconnect(xe_connection);

    PyObject* schema = PyUnicode_FromString(read_schema);
    return schema;
#undef CHECK
}

int64_t get_column_size_xenon(std::string* dset, uint64_t col_id)
{
#define CHECK(expr, msg) if(!(expr)){std::cerr << msg << std::endl; return -1;}

    const char* dset_name = dset->c_str();
	xe_connection_t xe_connection;
	xe_dataset_t	xe_dataset;
	struct xe_status status;

    xe_connection = xe_connect("localhost:41000");
    CHECK(xe_connection, "Fail to connect to Xenon");

    xe_dataset = xe_open(xe_connection, dset_name, 0, 0, XE_O_READONLY);
	CHECK(xe_dataset, "Fail to open dataset");

	int err = xe_status(xe_connection, xe_dataset, 0, &status);
	CHECK(!err, "Fail to stat dataset");
    CHECK(col_id < status.ncols, "invalid column number");

    xe_close(xe_connection, xe_dataset);
    xe_disconnect(xe_connection);

    return status.nrows;
#undef CHECK
}

void read_xenon_col(std::string* dset, uint64_t col_id, uint8_t* arr, uint64_t* xe_typ_enums)
{
#define CHECK(expr, msg) if(!(expr)){std::cerr << msg << std::endl; return;}

    // printf("read column %lld\n", col_id);
    // for (uint64_t i = 0; i<6; i++)
    //     printf("%lld ", xe_typ_enums[i]);
    // printf("\n");

    const char* dset_name = dset->c_str();
	xe_connection_t xe_connection;
	xe_dataset_t	xe_dataset;
	struct xe_status status;

    xe_connection = xe_connect("localhost:41000");
    CHECK(xe_connection, "Fail to connect to Xenon");

    xe_dataset = xe_open(xe_connection, dset_name, 0, 0, XE_O_READONLY);
	CHECK(xe_dataset, "Fail to open dataset");

	int err = xe_status(xe_connection, xe_dataset, 0, &status);
	CHECK(!err, "Fail to stat dataset");
    CHECK(col_id < status.ncols, "invalid column number");

    // _type_to_xe_dtype_number = {'int8': 0, 'int16': 1, 'int32': 2, 'int64': 3,
    //                             'float32': 4, 'float64': 5, 'DECIMAL': 6,
    //                              'bool_': 7, 'string': 8, 'BLOB': 9}

    const int read_buf_size = 2000000;
    uint8_t *curr_arr = arr;
    uint8_t *read_buf = (uint8_t *) malloc(read_buf_size * sizeof(uint8_t));
    uint64_t nrows = 0;
    int len = 0;

    for (uint64_t sid = 0; sid < status.fanout; sid++) {
        xe_rewind(xe_connection, xe_dataset, sid);

        do {
            uint8_t *buf = read_buf;
            xe_get(xe_connection, xe_dataset, sid, read_buf, read_buf_size, &nrows);
            for (uint64_t r = 0; r < nrows; r++) {
                for (uint64_t c = 0; c < status.ncols; c++) {
                    uint64_t tp_enum = xe_typ_enums[c];
                    read_xe_row(buf, curr_arr, tp_enum, col_id == c, len);
                }
            }
        } while (nrows);
    }

    delete[] read_buf;
    xe_close(xe_connection, xe_dataset);
    xe_disconnect(xe_connection);

    return;
#undef CHECK
}


void read_xenon_col_str(std::string* dset, uint64_t col_id, uint32_t **out_offsets,
                                    uint8_t **out_data, uint64_t* xe_typ_enums)
{
#define CHECK(expr, msg) if(!(expr)){std::cerr << msg << std::endl; return;}

    // printf("read column %lld\n", col_id);
    // for (uint64_t i = 0; i<6; i++)
    //     printf("%lld ", xe_typ_enums[i]);
    // printf("\n");

    const char* dset_name = dset->c_str();
	xe_connection_t xe_connection;
	xe_dataset_t	xe_dataset;
	struct xe_status status;

    xe_connection = xe_connect("localhost:41000");
    CHECK(xe_connection, "Fail to connect to Xenon");

    xe_dataset = xe_open(xe_connection, dset_name, 0, 0, XE_O_READONLY);
	CHECK(xe_dataset, "Fail to open dataset");

	int err = xe_status(xe_connection, xe_dataset, 0, &status);
	CHECK(!err, "Fail to stat dataset");
    CHECK(col_id < status.ncols, "invalid column number");

    // _type_to_xe_dtype_number = {'int8': 0, 'int16': 1, 'int32': 2, 'int64': 3,
    //                             'float32': 4, 'float64': 5, 'DECIMAL': 6,
    //                              'bool_': 7, 'string': 8, 'BLOB': 9}

    *out_offsets = new uint32_t[status.nrows+1];
    uint32_t* curr_offset = *out_offsets;
    uint32_t curr_len = 0;
    std::vector<uint8_t> data_vec;

    const int read_buf_size = 2000000;
    uint8_t *data_arr = (uint8_t *) malloc(read_buf_size * sizeof(uint8_t));
    uint8_t *read_buf = (uint8_t *) malloc(read_buf_size * sizeof(uint8_t));
    uint64_t nrows = 0;
    int len = 0;

    for (uint64_t sid = 0; sid < status.fanout; sid++) {
        xe_rewind(xe_connection, xe_dataset, sid);

        do {
            uint8_t *buf = read_buf;
            uint8_t * curr_arr = data_arr;
            xe_get(xe_connection, xe_dataset, sid, read_buf, read_buf_size, &nrows);
            for (uint64_t r = 0; r < nrows; r++) {
                for (uint64_t c = 0; c < status.ncols; c++) {
                    uint64_t tp_enum = xe_typ_enums[c];
                    read_xe_row(buf, curr_arr, tp_enum, col_id == c, len);
                    if (col_id == c) {
                        *curr_offset = curr_len;
                        curr_offset++;
                        curr_len += len;
                        data_vec.insert(data_vec.end(), curr_arr-len, curr_arr);
                    }
                }
            }
        } while (nrows);
    }
    *curr_offset = curr_len;

    *out_data = new uint8_t[data_vec.size()];
    memcpy(*out_data, data_vec.data(), data_vec.size());

    delete[] data_arr;

    delete[] read_buf;
    xe_close(xe_connection, xe_dataset);
    xe_disconnect(xe_connection);

    return;
#undef CHECK
}

inline int16_t get_2byte_val(uint8_t* buf)
{
    int16_t val = 0;
    for (int i = 0; i < 2; i++) {
        val = (val << 8) + *(buf+i);
    }
    return val;
}

inline int get_4byte_val(uint8_t* buf)
{
    int val_i32 = 0;
    for (int i = 0; i < 4; i++) {
        val_i32 = (val_i32 << 8) + *(buf+i);
    }
    return val_i32;
}

inline int64_t get_8byte_val(uint8_t* buf)
{
    int64_t val_i64 = 0;
    for (int i = 0; i < 8; i++) {
        val_i64 = (val_i64 << 8) + *(buf+i);
    }
    return val_i64;
}

} // extern "C"

inline void read_xe_row(uint8_t* &buf, uint8_t* &curr_arr, uint64_t tp_enum, bool do_read, int &len)
{
#define CHECK(expr, msg) if(!(expr)){std::cerr << msg << std::endl; return;}
    if (*buf) {
        buf++;
        switch (tp_enum) {
            case 0:  // int8
                if (do_read) {
                    *curr_arr = *buf;
                    curr_arr++;
                }
                buf++;
                break;
            case 1:  // int16
                if (do_read) {
                    *(int16_t*)curr_arr = get_2byte_val(buf);
                    curr_arr += 2;
                }
                buf += 2;
                break;
            case 2:  // int32
                if (do_read) {
                    *(int*)curr_arr = get_4byte_val(buf);
                    curr_arr += 4;
                }
                buf += 4;
                break;
            case 3:  // int64
                if (do_read) {
                    *(int64_t*)curr_arr = get_8byte_val(buf);
                    curr_arr += 8;
                }
                buf += 8;
                break;
            case 4:  // float32
                if (do_read) {
                    *(int*)curr_arr = get_4byte_val(buf);
                    curr_arr += 4;
                }
                buf += 4;
                break;
            case 5:  // float64
                if (do_read) {
                    *(int64_t*)curr_arr = get_8byte_val(buf);
                    curr_arr += 8;
                }
                buf += 8;
                break;
            case 6:  // decimal
                CHECK(false, "Decimal values not supported yet");
                break;
            case 7:  // bool
                if (do_read) {
                    *curr_arr = *buf;
                    curr_arr++;
                }
                buf++;
                break;
            case 8:  // string
                len = get_2byte_val(buf);
                buf += 2;
                if (do_read) {
                    memcpy(curr_arr, buf, len);
                    curr_arr += len;
                }
                buf += len;
                // printf("%s,", str);
                break;
            case 9:  // blob
                CHECK(false, "Blob values not supported yet");
                break;
            default:
                CHECK(false, "Unknown type");
        }
    } else {
        // printf("null,");
        // TODO: null values supported for float32 and float64 only
        if (do_read) {
            // float32
            if (tp_enum==4)
            {
                // TODO: use NPY_NAN
                *(float*)curr_arr = std::nanf("");
                curr_arr += 4;
            }
            // float64
            if (tp_enum==5)
            {
                // TODO: use NPY_NAN
                *(double*)curr_arr = std::nan("");
                curr_arr += 8;
            }
        }
        buf++;
    }
    return;
#undef CHECK
}

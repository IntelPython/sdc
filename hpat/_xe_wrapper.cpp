#include <Python.h>
#include "xe.h"
#include <iostream>

extern "C" {


static PyObject* get_schema(PyObject *self, PyObject *args);
int64_t get_column_size_xenon(std::string* dset, uint64_t col_id);
void read_xenon_col(std::string* dset, uint64_t col_id, char* arr, int xe_typ);

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

    return m;
}

#define MAX_SCHEMA_LEN 200

static PyObject* get_schema(PyObject *self, PyObject *args) {
#define CHECK(expr, msg) if(!(expr)){std::cerr << msg << std::endl; return NULL;}

    const char* dset_name;
    char *read_schema = (char *) malloc(MAX_SCHEMA_LEN * sizeof(char));
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

void read_xenon_col(std::string* dset, uint64_t col_id, char* arr, int xe_typ)
{
#define CHECK(expr, msg) if(!(expr)){std::cerr << msg << std::endl; return;}

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

    return;
#undef CHECK
}

} // extern "C"

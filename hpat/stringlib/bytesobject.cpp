#include <Python.h>
#include <iostream>

/* The _C_BytesWriter structure is big: it contains an embedded "stack buffer".
   A _C_BytesWriter variable must be declared at the end of variables in a
   function to optimize the memory allocation on the stack. */
typedef struct {
    /* bytes, bytearray or NULL (when the small buffer is used) */
    char *buffer;

    /* Number of allocated size. */
    Py_ssize_t allocated;

    /* Minimum number of allocated bytes,
       incremented by _C_BytesWriter_Prepare() */
    Py_ssize_t min_size;

    /* If non-zero, use a bytearray instead of a bytes object for buffer. */
    int use_bytearray;

    /* If non-zero, overallocate the buffer (default: 0).
       This flag must be zero if use_bytearray is non-zero. */
    int overallocate;

    /* Stack buffer */
    int use_small_buffer;
    char small_buffer[512];
} _C_BytesWriter;


#ifdef MS_WINDOWS
   /* On Windows, overallocate by 50% is the best factor */
#  define OVERALLOCATE_FACTOR 2
#else
   /* On Linux, overallocate by 25% is the best factor */
#  define OVERALLOCATE_FACTOR 4
#endif

void _C_BytesWriter_Init(_C_BytesWriter *writer)
{
    /* Set all attributes before small_buffer to 0 */
    memset(writer, 0, offsetof(_C_BytesWriter, small_buffer));
}


void
_C_BytesWriter_Dealloc(_C_BytesWriter *writer)
{
    if (writer->buffer != NULL)
        delete[] writer->buffer;
}

Py_LOCAL_INLINE(char*)
_C_BytesWriter_AsString(_C_BytesWriter *writer)
{
    if (writer->use_small_buffer) {
        assert(writer->buffer == NULL);
        return writer->small_buffer;
    }
    else if (writer->use_bytearray) {
        assert(writer->buffer != NULL);
        return writer->buffer;
    }
    else {
        assert(writer->buffer != NULL);
        return writer->buffer;
    }
}

Py_LOCAL_INLINE(Py_ssize_t)
_C_BytesWriter_GetSize(_C_BytesWriter *writer, char *str)
{
    char *start = _C_BytesWriter_AsString(writer);
    assert(str != NULL);
    assert(str >= start);
    assert(str - start <= writer->allocated);
    return str - start;
}

void*
_C_BytesWriter_Resize(_C_BytesWriter *writer, void *str, Py_ssize_t size)
{
    Py_ssize_t allocated, pos;

    // _C_BytesWriter_CheckConsistency(writer, str);
    assert(writer->allocated < size);

    allocated = size;
    if (writer->overallocate
        && allocated <= (PY_SSIZE_T_MAX - allocated / OVERALLOCATE_FACTOR)) {
        /* overallocate to limit the number of realloc() */
        allocated += allocated / OVERALLOCATE_FACTOR;
    }

    pos = _C_BytesWriter_GetSize(writer, (char*)str);
    if (!writer->use_small_buffer) {
        char *new_buff = new char[allocated];
        memcpy(new_buff, writer->buffer, pos);
        delete[] writer->buffer;
        writer->buffer = new_buff;
        if (writer->buffer == NULL)
            goto error;
    }
    else {
        /* convert from stack buffer to bytes object buffer */
        assert(writer->buffer == NULL);

        writer->buffer = new char[allocated];

        if (writer->buffer == NULL)
            goto error;

        if (pos != 0) {
            char *dest;
            if (writer->use_bytearray)
                dest = writer->buffer;
            else
                dest = writer->buffer;
            memcpy(dest,
                      writer->small_buffer,
                      pos);
        }

        writer->use_small_buffer = 0;

    }
    writer->allocated = allocated;

    str = _C_BytesWriter_AsString(writer) + pos;
    // _C_BytesWriter_CheckConsistency(writer, str);
    return str;

error:
    _C_BytesWriter_Dealloc(writer);
    return NULL;
}

void*
_C_BytesWriter_Prepare(_C_BytesWriter *writer, void *str, Py_ssize_t size)
{
    Py_ssize_t new_min_size;

    // _C_BytesWriter_CheckConsistency(writer, str);
    assert(size >= 0);

    if (size == 0) {
        /* nothing to do */
        return str;
    }

    if (writer->min_size > PY_SSIZE_T_MAX - size) {
        // TODO: memory error
        std::cerr << "invalid maximum character" << std::endl;
        _C_BytesWriter_Dealloc(writer);
        return NULL;
    }
    new_min_size = writer->min_size + size;

    if (new_min_size > writer->allocated)
        str = _C_BytesWriter_Resize(writer, str, new_min_size);

    writer->min_size = new_min_size;
    return str;
}

/* Allocate the buffer to write size bytes.
   Return the pointer to the beginning of buffer data.
   Raise an exception and return NULL on error. */
void*
_C_BytesWriter_Alloc(_C_BytesWriter *writer, Py_ssize_t size)
{
    /* ensure that _C_BytesWriter_Alloc() is only called once */
    assert(writer->min_size == 0 && writer->buffer == NULL);
    assert(size >= 0);

    writer->use_small_buffer = 1;
    writer->allocated = sizeof(writer->small_buffer);
    return _C_BytesWriter_Prepare(writer, writer->small_buffer, size);
}

int64_t _C_BytesWriter_Finish(char *out_data, _C_BytesWriter *writer, void *str)
{
    Py_ssize_t size;

    // _C_BytesWriter_CheckConsistency(writer, str);

    size = _C_BytesWriter_GetSize(writer, (char*)str);
    if (size == 0 && !writer->use_bytearray) {
        if (writer->buffer != NULL)
            delete[] writer->buffer;
    }
    else if (writer->use_small_buffer) {
        if (out_data != NULL)
            memcpy(out_data, writer->small_buffer, size);
    }
    else {
        if (out_data != NULL)
            memcpy(out_data, writer->buffer, size);
        delete[] writer->buffer;
    }
    return size;
}

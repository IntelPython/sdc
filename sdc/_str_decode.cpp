//*****************************************************************************
// Copyright (c) 2020, Intel Corporation All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

#include <Python.h>
#include <iostream>
#include <stdlib.h>

#include "numba/core/runtime/nrt_external.h"

#ifndef Py_UNREACHABLE
#define Py_UNREACHABLE() abort()
#endif

// ******** ported from CPython 31e8d69bfe7cf5d4ffe0967cb225d2a8a229cc97

typedef struct
{
    NRT_api_functions* nrt;
    NRT_MemInfo* buffer;
    void* data;
    enum PyUnicode_Kind kind;
    int is_ascii;
    Py_UCS4 maxchar;
    Py_ssize_t size;
    Py_ssize_t pos;

    /* minimum number of allocated characters (default: 0) */
    Py_ssize_t min_length;

    /* minimum character (default: 127, ASCII) */
    Py_UCS4 min_char;

    /* If non-zero, overallocate the buffer (default: 0). */
    unsigned char overallocate;

    /* If readonly is 1, buffer is a shared string (cannot be modified)
       and size is set to 0. */
    unsigned char readonly;
} _C_UnicodeWriter;

void _C_UnicodeWriter_Init(_C_UnicodeWriter* writer)
{
    memset(writer, 0, sizeof(*writer));

    /* ASCII is the bare minimum */
    writer->min_char = 127;

    /* use a value smaller than PyUnicode_1BYTE_KIND() so
       _C_UnicodeWriter_PrepareKind() will copy the buffer. */
    writer->kind = PyUnicode_WCHAR_KIND;
    writer->is_ascii = 0;
    assert(writer->kind <= PyUnicode_1BYTE_KIND);
}

#ifdef MS_WINDOWS
/* On Windows, overallocate by 50% is the best factor */
#define OVERALLOCATE_FACTOR 2
#else
/* On Linux, overallocate by 25% is the best factor */
#define OVERALLOCATE_FACTOR 4
#endif

/* Maximum code point of Unicode 6.0: 0x10ffff (1,114,111) */
#define MAX_UNICODE 0x10ffff

/* Prepare the buffer to write 'length' characters
   with the specified maximum character.

   Return 0 on success, raise an exception and return -1 on error. */
#define _C_UnicodeWriter_Prepare(WRITER, LENGTH, MAXCHAR)                                                              \
    (((MAXCHAR) <= (WRITER)->maxchar && (LENGTH) <= (WRITER)->size - (WRITER)->pos)                                    \
         ? 0                                                                                                           \
         : (((LENGTH) == 0) ? 0 : _C_UnicodeWriter_PrepareInternal((WRITER), (LENGTH), (MAXCHAR))))

#define KIND_MAX_CHAR_VALUE(kind)                                                                                      \
    (kind == PyUnicode_1BYTE_KIND ? (0xffU) : (kind == PyUnicode_2BYTE_KIND ? (0xffffU) : (0x10ffffU)))

#include "stringlib/bytesobject.cpp"

#include "stringlib/asciilib.h"

#include "stringlib/codecs.h"
#include "stringlib/undef.h"

// TODO needs to be redesigned. This is not acceptable *.h handling.
#include "stringlib/ucs1lib.h"

#include "stringlib/codecs.h"
#include "stringlib/undef.h"

#include "stringlib/ucs2lib.h"

#include "stringlib/codecs.h"
#include "stringlib/undef.h"

#include "stringlib/ucs4lib.h"

#include "stringlib/codecs.h"
#include "stringlib/undef.h"

static inline int _C_UnicodeWriter_WriteCharInline(_C_UnicodeWriter* writer, Py_UCS4 ch);
static int _copy_characters(NRT_api_functions* nrt,
                            NRT_MemInfo* to,
                            Py_ssize_t to_start,
                            NRT_MemInfo* from,
                            Py_ssize_t from_start,
                            Py_ssize_t how_many,
                            unsigned int from_kind,
                            unsigned int to_kind);


static void str_data_dtor(void* data_ptr)
{
    free(data_ptr);
}

// similar to PyUnicode_New()
NRT_MemInfo* alloc_writer(_C_UnicodeWriter* writer, Py_ssize_t newlen, Py_UCS4 maxchar)
{
    enum PyUnicode_Kind kind;
    int is_ascii = 0;
    Py_ssize_t char_size;
    auto nrt = writer->nrt;

    if (maxchar < 128)
    {
        kind = PyUnicode_1BYTE_KIND;
        is_ascii = 1;
        char_size = 1;
    }
    else if (maxchar < 256)
    {
        kind = PyUnicode_1BYTE_KIND;
        char_size = 1;
    }
    else if (maxchar < 65536)
    {
        kind = PyUnicode_2BYTE_KIND;
        char_size = 2;
    }
    else
    {
        if (maxchar > MAX_UNICODE)
        {
            std::cerr << "invalid maximum character" << std::endl;
            return NULL;
        }
        kind = PyUnicode_4BYTE_KIND;
        char_size = 4;
    }

    char* str_data = (char*)malloc((newlen + 1) * char_size);
    if (str_data == NULL)
    {
        return NULL;
    }

    auto newbuffer = nrt->manage_memory(str_data, str_data_dtor);
    if (writer->buffer != NULL)
    {
        _copy_characters(nrt, newbuffer, 0, writer->buffer, 0, writer->pos, writer->kind, kind);
        nrt->release(writer->buffer);
    }
    writer->buffer = newbuffer;
    writer->maxchar = KIND_MAX_CHAR_VALUE(kind);
    writer->data = nrt->get_data(writer->buffer);

    if (!writer->readonly)
    {
        writer->kind = kind;
        writer->is_ascii = is_ascii;
        writer->size = newlen;
    }
    else
    {
        /* use a value smaller than PyUnicode_1BYTE_KIND() so
           _PyUnicodeWriter_PrepareKind() will copy the buffer. */
        writer->kind = PyUnicode_WCHAR_KIND;
        assert(writer->kind <= PyUnicode_1BYTE_KIND);
        writer->is_ascii = 0;

        /* Copy-on-write mode: set buffer size to 0 so
         * _PyUnicodeWriter_Prepare() will copy (and enlarge) the buffer on
         * next write. */
        writer->size = 0;
    }
    return newbuffer;
}

int _C_UnicodeWriter_PrepareInternal(_C_UnicodeWriter* writer, Py_ssize_t length, Py_UCS4 maxchar)
{
    Py_ssize_t newlen;
    NRT_MemInfo* newbuffer;

    assert(maxchar <= MAX_UNICODE);

    /* ensure that the _C_UnicodeWriter_Prepare macro was used */
    assert((maxchar > writer->maxchar && length >= 0) || length > 0);

    if (length > PY_SSIZE_T_MAX - writer->pos)
    {
        // TODO: proper memory error
        std::cerr << "memory error" << std::endl;
        return -1;
    }
    newlen = writer->pos + length;

    maxchar = Py_MAX(maxchar, writer->min_char);

    if (writer->buffer == NULL)
    {
        assert(!writer->readonly);
        if (writer->overallocate && newlen <= (PY_SSIZE_T_MAX - newlen / OVERALLOCATE_FACTOR))
        {
            /* overallocate to limit the number of realloc() */
            newlen += newlen / OVERALLOCATE_FACTOR;
        }
        if (newlen < writer->min_length)
            newlen = writer->min_length;

        writer->buffer = alloc_writer(writer, newlen, maxchar);
        if (writer->buffer == NULL)
        {
            return -1;
        }
    }
    else if (newlen > writer->size)
    {
        if (writer->overallocate && newlen <= (PY_SSIZE_T_MAX - newlen / OVERALLOCATE_FACTOR))
        {
            /* overallocate to limit the number of realloc() */
            newlen += newlen / OVERALLOCATE_FACTOR;
        }
        if (newlen < writer->min_length)
            newlen = writer->min_length;

        if (maxchar > writer->maxchar || writer->readonly)
        {
            /* resize + widen */
            maxchar = Py_MAX(maxchar, writer->maxchar);
            newbuffer = alloc_writer(writer, newlen, maxchar);
            if (newbuffer == NULL)
            {
                return -1;
            }
            writer->readonly = 0;
        }
        else
        {
            newbuffer = alloc_writer(writer, newlen, writer->maxchar);
            if (newbuffer == NULL)
            {
                return -1;
            }
        }
        writer->buffer = newbuffer;
    }
    else if (maxchar > writer->maxchar)
    {
        assert(!writer->readonly);
        newbuffer = alloc_writer(writer, writer->size, maxchar);
        if (newbuffer == NULL)
        {
            return -1;
        }
    }
    return 0;

#undef OVERALLOCATE_FACTOR
}

/* Mask to quickly check whether a C 'long' contains a
   non-ASCII, UTF8-encoded char. */
#if (SIZEOF_LONG == 8)
#define ASCII_CHAR_MASK 0x8080808080808080UL
#elif (SIZEOF_LONG == 4)
#define ASCII_CHAR_MASK 0x80808080UL
#else
#error C 'long' size should be either 4 or 8!
#endif

static Py_ssize_t ascii_decode(const char* start, const char* end, Py_UCS1* dest)
{
    const char* p = start;
    const char* aligned_end = (const char*)_Py_ALIGN_DOWN(end, SIZEOF_LONG);

#if SIZEOF_LONG <= SIZEOF_VOID_P
    assert(_Py_IS_ALIGNED(dest, SIZEOF_LONG));
    if (_Py_IS_ALIGNED(p, SIZEOF_LONG))
    {
        /* Fast path, see in STRINGLIB(utf8_decode) for
           an explanation. */
        /* Help allocation */
        const char* _p = p;
        Py_UCS1* q = dest;
        while (_p < aligned_end)
        {
            unsigned long value = *(const unsigned long*)_p;
            if (value & ASCII_CHAR_MASK)
                break;
            *((unsigned long*)q) = value;
            _p += SIZEOF_LONG;
            q += SIZEOF_LONG;
        }
        p = _p;
        while (p < end)
        {
            if ((unsigned char)*p & 0x80)
            {
                break;
            }
            *q++ = *p++;
        }
        return p - start;
    }
#endif
    while (p < end)
    {
        /* Fast path, see in STRINGLIB(utf8_decode) in stringlib/codecs.h
           for an explanation. */
        if (_Py_IS_ALIGNED(p, SIZEOF_LONG))
        {
            /* Help allocation */
            const char* _p = p;
            while (_p < aligned_end)
            {
                unsigned long value = *(unsigned long*)_p;
                if (value & ASCII_CHAR_MASK)
                    break;
                _p += SIZEOF_LONG;
            }
            p = _p;
            if (_p == end)
            {
                break;
            }
        }
        if ((unsigned char)*p & 0x80)
        {
            break;
        }
        ++p;
    }
    memcpy(dest, start, p - start);
    return p - start;
}


// ported from CPython PyUnicode_DecodeUTF8Stateful: https://github.com/python/cpython/blob/31e8d69bfe7cf5d4ffe0967cb225d2a8a229cc97/Objects/unicodeobject.c#L4813
void decode_utf8(const char* s, Py_ssize_t size, int* kind, int* is_ascii, int* length, NRT_MemInfo** meminfo, void* nrt_table)
{
    _C_UnicodeWriter writer;
    const char* end = s + size;
    auto nrt = (NRT_api_functions*)nrt_table;

    const char* errmsg = "";
    *is_ascii = 0;

    if (size == 0)
    {
        char* str_data = (char*)malloc(1);
        (*meminfo) = nrt->manage_memory(str_data, str_data_dtor);
        ((char*)(nrt->get_data(*meminfo)))[0] = 0;
        *kind = PyUnicode_1BYTE_KIND;
        *is_ascii = 1;
        *length = 0;
        return;
    }

    /* ASCII is equivalent to the first 128 ordinals in Unicode. */
    if (size == 1 && (unsigned char)s[0] < 128)
    {
        // TODO interning
        char* str_data = (char*)malloc(2);
        (*meminfo) = nrt->manage_memory(str_data, str_data_dtor);
        ((char*)(nrt->get_data(*meminfo)))[0] = s[0];
        ((char*)(nrt->get_data(*meminfo)))[1] = 0;
        *kind = PyUnicode_1BYTE_KIND;
        *is_ascii = 1;
        *length = 1;
        return;
    }

    _C_UnicodeWriter_Init(&writer);
    writer.min_length = size;
    writer.nrt = nrt;
    if (_C_UnicodeWriter_Prepare(&writer, writer.min_length, 127) == -1)
        goto onError;

    writer.pos = ascii_decode(s, end, (Py_UCS1*)writer.data);
    s += writer.pos;
    while (s < end)
    {
        Py_UCS4 ch;
        int kind = writer.kind;

        if (kind == PyUnicode_1BYTE_KIND)
        {
            if (writer.is_ascii == 1)
                ch = asciilib_utf8_decode(&s, end, (Py_UCS1*)writer.data, &writer.pos);
            else
                ch = ucs1lib_utf8_decode(&s, end, (Py_UCS1*)writer.data, &writer.pos);
        }
        else if (kind == PyUnicode_2BYTE_KIND)
        {
            ch = ucs2lib_utf8_decode(&s, end, (Py_UCS2*)writer.data, &writer.pos);
        }
        else
        {
            assert(kind == PyUnicode_4BYTE_KIND);
            ch = ucs4lib_utf8_decode(&s, end, (Py_UCS4*)writer.data, &writer.pos);
        }

        switch (ch)
        {
        case 0:
            if (s == end)
                goto End;
            errmsg = "unexpected end of data";
            break;
        case 1:
            errmsg = "invalid start byte";
            break;
        case 2:
        case 3:
        case 4:
            if (s == end)
            {
                goto End;
            }
            errmsg = "invalid continuation byte";
            break;
        default:
            if (_C_UnicodeWriter_WriteCharInline(&writer, ch) < 0)
                goto onError;
            continue;
        }

        // TODO: error handlers
        goto onError;
    }

End:
    (*meminfo) = writer.buffer;
    *kind = writer.kind;
    *is_ascii = writer.is_ascii;
    *length = writer.pos;
    // set null
    if (writer.kind == PyUnicode_1BYTE_KIND)
    {
        ((char*)writer.data)[writer.pos] = 0;
    }
    else if (writer.kind == PyUnicode_2BYTE_KIND)
    {
        ((Py_UCS2*)writer.data)[writer.pos] = 0;
    }
    else
    {
        assert(writer.kind == PyUnicode_4BYTE_KIND);
        ((Py_UCS4*)writer.data)[writer.pos] = 0;
    }
    return;

onError:
    std::cerr << "utf8 decode error:" << errmsg << std::endl;
    nrt->release(*meminfo);
    return;
}

/* Generic helper macro to convert characters of different types.
   from_type and to_type have to be valid type names, begin and end
   are pointers to the source characters which should be of type
   "from_type *".  to is a pointer of type "to_type *" and points to the
   buffer where the result characters are written to. */
#define _PyUnicode_CONVERT_BYTES(from_type, to_type, begin, end, to)                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        to_type* _to = (to_type*)(to);                                                                                 \
        const from_type* _iter = (from_type*)(begin);                                                                  \
        const from_type* _end = (from_type*)(end);                                                                     \
        Py_ssize_t n = (_end) - (_iter);                                                                               \
        const from_type* _unrolled_end = _iter + _Py_SIZE_ROUND_DOWN(n, 4);                                            \
        while (_iter < (_unrolled_end))                                                                                \
        {                                                                                                              \
            _to[0] = (to_type)_iter[0];                                                                                \
            _to[1] = (to_type)_iter[1];                                                                                \
            _to[2] = (to_type)_iter[2];                                                                                \
            _to[3] = (to_type)_iter[3];                                                                                \
            _iter += 4;                                                                                                \
            _to += 4;                                                                                                  \
        }                                                                                                              \
        while (_iter < (_end))                                                                                         \
            *_to++ = (to_type)*_iter++;                                                                                \
    } while (0)

static int _copy_characters(NRT_api_functions* nrt,
                            NRT_MemInfo* to,
                            Py_ssize_t to_start,
                            NRT_MemInfo* from,
                            Py_ssize_t from_start,
                            Py_ssize_t how_many,
                            unsigned int from_kind,
                            unsigned int to_kind)
{
    void *from_data, *to_data;

    assert(0 <= how_many);
    assert(0 <= from_start);
    assert(0 <= to_start);

    if (how_many == 0)
        return 0;

    from_data = nrt->get_data(from);
    to_data = nrt->get_data(to);

    if (from_kind == to_kind)
    {
        memcpy((char*)to_data + to_kind * to_start, (char*)from_data + from_kind * from_start, to_kind * how_many);
    }
    else if (from_kind == PyUnicode_1BYTE_KIND && to_kind == PyUnicode_2BYTE_KIND)
    {
        _PyUnicode_CONVERT_BYTES(Py_UCS1,
                                 Py_UCS2,
                                 ((Py_UCS1*)(from_data)) + from_start,
                                 ((Py_UCS1*)(from_data)) + from_start + how_many,
                                 ((Py_UCS2*)(to_data)) + to_start);
    }
    else if (from_kind == PyUnicode_1BYTE_KIND && to_kind == PyUnicode_4BYTE_KIND)
    {
        _PyUnicode_CONVERT_BYTES(Py_UCS1,
                                 Py_UCS4,
                                 ((Py_UCS1*)(from_data)) + from_start,
                                 ((Py_UCS1*)(from_data)) + from_start + how_many,
                                 ((Py_UCS4*)(to_data)) + to_start);
    }
    else if (from_kind == PyUnicode_2BYTE_KIND && to_kind == PyUnicode_4BYTE_KIND)
    {
        _PyUnicode_CONVERT_BYTES(Py_UCS2,
                                 Py_UCS4,
                                 ((Py_UCS2*)(from_data)) + from_start,
                                 ((Py_UCS2*)(from_data)) + from_start + how_many,
                                 ((Py_UCS4*)(to_data)) + to_start);
    }
    else
    {
        if (1)
        {
            if (from_kind == PyUnicode_2BYTE_KIND && to_kind == PyUnicode_1BYTE_KIND)
            {
                _PyUnicode_CONVERT_BYTES(Py_UCS2,
                                         Py_UCS1,
                                         ((Py_UCS2*)(from_data)) + from_start,
                                         ((Py_UCS2*)(from_data)) + from_start + how_many,
                                         ((Py_UCS1*)(to_data)) + to_start);
            }
            else if (from_kind == PyUnicode_4BYTE_KIND && to_kind == PyUnicode_1BYTE_KIND)
            {
                _PyUnicode_CONVERT_BYTES(Py_UCS4,
                                         Py_UCS1,
                                         ((Py_UCS4*)(from_data)) + from_start,
                                         ((Py_UCS4*)(from_data)) + from_start + how_many,
                                         ((Py_UCS1*)(to_data)) + to_start);
            }
            else if (from_kind == PyUnicode_4BYTE_KIND && to_kind == PyUnicode_2BYTE_KIND)
            {
                _PyUnicode_CONVERT_BYTES(Py_UCS4,
                                         Py_UCS2,
                                         ((Py_UCS4*)(from_data)) + from_start,
                                         ((Py_UCS4*)(from_data)) + from_start + how_many,
                                         ((Py_UCS2*)(to_data)) + to_start);
            }
            else
            {
                abort();
            }
        }
    }
    return 0;
}

static inline int _C_UnicodeWriter_WriteCharInline(_C_UnicodeWriter* writer, Py_UCS4 ch)
{
    assert(ch <= MAX_UNICODE);
    if (_C_UnicodeWriter_Prepare(writer, 1, ch) < 0)
        return -1;
    PyUnicode_WRITE(writer->kind, writer->data, writer->pos, ch);
    writer->pos++;
    return 0;
}

int64_t unicode_to_utf8(char* out_data, char* data, int64_t size, int kind)
{
    //
    switch (kind)
    {
    default:
        Py_UNREACHABLE();
    case PyUnicode_1BYTE_KIND:
        return ucs1lib_utf8_encoder(out_data, (Py_UCS1*)data, size);
    case PyUnicode_2BYTE_KIND:
        return ucs2lib_utf8_encoder(out_data, (Py_UCS2*)data, size);
    case PyUnicode_4BYTE_KIND:
        return ucs4lib_utf8_encoder(out_data, (Py_UCS4*)data, size);
    }
}

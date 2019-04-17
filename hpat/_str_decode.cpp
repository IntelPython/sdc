#include <Python.h>
#include <iostream>

// ******** copied from Numba
// TODO: make Numba C library
typedef void (*NRT_dtor_function)(void *ptr, size_t size, void *info);
struct MemInfo {
    size_t            refct;
    NRT_dtor_function dtor;
    void              *dtor_info;
    void              *data;
    size_t            size;    /* only used for NRT allocated memory */
};

typedef struct MemInfo NRT_MemInfo;


NRT_MemInfo* (*NRT_MemInfo_alloc_safe)(size_t);
void (*NRT_MemInfo_call_dtor)(NRT_MemInfo*);

// initialize allocator/deallocator from Numba pointers. need to be set before calling decode
void init_memsys(NRT_MemInfo* (*_NRT_MemInfo_alloc_safe)(size_t), void (*_NRT_MemInfo_call_dtor)(NRT_MemInfo*))
{
    NRT_MemInfo_alloc_safe = _NRT_MemInfo_alloc_safe;
    NRT_MemInfo_call_dtor = _NRT_MemInfo_call_dtor;
}


// ******** ported from CPython 31e8d69bfe7cf5d4ffe0967cb225d2a8a229cc97

typedef struct {
    NRT_MemInfo *buffer;
    void *data;
    enum PyUnicode_Kind kind;
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
} _C_UnicodeWriter ;


void
_C_UnicodeWriter_Init(_C_UnicodeWriter *writer)
{
    memset(writer, 0, sizeof(*writer));

    /* ASCII is the bare minimum */
    writer->min_char = 127;

    /* use a value smaller than PyUnicode_1BYTE_KIND() so
       _C_UnicodeWriter_PrepareKind() will copy the buffer. */
    writer->kind = PyUnicode_WCHAR_KIND;
    assert(writer->kind <= PyUnicode_1BYTE_KIND);
}

/* Mask to quickly check whether a C 'long' contains a
   non-ASCII, UTF8-encoded char. */
#if (SIZEOF_LONG == 8)
# define ASCII_CHAR_MASK 0x8080808080808080UL
#elif (SIZEOF_LONG == 4)
# define ASCII_CHAR_MASK 0x80808080UL
#else
# error C 'long' size should be either 4 or 8!
#endif

/* Round pointer "p" down to the closest "a"-aligned address <= "p". */
#define _Py_ALIGN_DOWN(p, a) ((void *)((uintptr_t)(p) & ~(uintptr_t)((a) - 1)))
/* Round pointer "p" up to the closest "a"-aligned address >= "p". */
#define _Py_ALIGN_UP(p, a) ((void *)(((uintptr_t)(p) + \
        (uintptr_t)((a) - 1)) & ~(uintptr_t)((a) - 1)))
/* Check if pointer "p" is aligned to "a"-bytes boundary. */
#define _Py_IS_ALIGNED(p, a) (!((uintptr_t)(p) & (uintptr_t)((a) - 1)))

static Py_ssize_t
ascii_decode(const char *start, const char *end, Py_UCS1 *dest)
{
    const char *p = start;
    const char *aligned_end = (const char *) _Py_ALIGN_DOWN(end, SIZEOF_LONG);

#if SIZEOF_LONG <= SIZEOF_VOID_P
    assert(_Py_IS_ALIGNED(dest, SIZEOF_LONG));
    if (_Py_IS_ALIGNED(p, SIZEOF_LONG)) {
        /* Fast path, see in STRINGLIB(utf8_decode) for
           an explanation. */
        /* Help allocation */
        const char *_p = p;
        Py_UCS1 * q = dest;
        while (_p < aligned_end) {
            unsigned long value = *(const unsigned long *) _p;
            if (value & ASCII_CHAR_MASK)
                break;
            *((unsigned long *)q) = value;
            _p += SIZEOF_LONG;
            q += SIZEOF_LONG;
        }
        p = _p;
        while (p < end) {
            if ((unsigned char)*p & 0x80)
                break;
            *q++ = *p++;
        }
        return p - start;
    }
#endif
    while (p < end) {
        /* Fast path, see in STRINGLIB(utf8_decode) in stringlib/codecs.h
           for an explanation. */
        if (_Py_IS_ALIGNED(p, SIZEOF_LONG)) {
            /* Help allocation */
            const char *_p = p;
            while (_p < aligned_end) {
                unsigned long value = *(unsigned long *) _p;
                if (value & ASCII_CHAR_MASK)
                    break;
                _p += SIZEOF_LONG;
            }
            p = _p;
            if (_p == end)
                break;
        }
        if ((unsigned char)*p & 0x80)
            break;
        ++p;
    }
    memcpy(dest, start, p - start);
    return p - start;
}

// ported from CPython PyUnicode_DecodeUTF8Stateful: https://github.com/python/cpython/blob/31e8d69bfe7cf5d4ffe0967cb225d2a8a229cc97/Objects/unicodeobject.c#L4813
void decode_utf8(const char *s, Py_ssize_t size, int* kind, int* length, NRT_MemInfo** meminfo)
{
    _C_UnicodeWriter writer;
    const char *starts = s;
    const char *end = s + size;

    Py_ssize_t startinpos;
    Py_ssize_t endinpos;
    const char *errmsg = "";

    if (size == 0) {
        (*meminfo) = NRT_MemInfo_alloc_safe(1);
        ((char*)((*meminfo)->data))[0] = 0;
        *kind = PyUnicode_1BYTE_KIND;
        *length = 0;
        return;
    }

    /* ASCII is equivalent to the first 128 ordinals in Unicode. */
    if (size == 1 && (unsigned char)s[0] < 128) {
        // TODO interning
        (*meminfo) = NRT_MemInfo_alloc_safe(1);
        ((char*)((*meminfo)->data))[0] = s[0];
        ((char*)((*meminfo)->data))[1] = 0;
        *kind = PyUnicode_1BYTE_KIND;
        *length = 1;
        return;
    }

    _C_UnicodeWriter_Init(&writer);
    writer.min_length = size;
    if (_C_UnicodeWriter_Prepare(&writer, writer.min_length, 127) == -1)
        goto onError;

    writer.pos = ascii_decode(s, end, writer.data);
    s += writer.pos;
    while (s < end) {
        Py_UCS4 ch;
        int kind = writer.kind;

        if (kind == PyUnicode_1BYTE_KIND) {
            if (PyUnicode_IS_ASCII(writer.buffer))
                ch = asciilib_utf8_decode(&s, end, writer.data, &writer.pos);
            else
                ch = ucs1lib_utf8_decode(&s, end, writer.data, &writer.pos);
        } else if (kind == PyUnicode_2BYTE_KIND) {
            ch = ucs2lib_utf8_decode(&s, end, writer.data, &writer.pos);
        } else {
            assert(kind == PyUnicode_4BYTE_KIND);
            ch = ucs4lib_utf8_decode(&s, end, writer.data, &writer.pos);
        }

        switch (ch) {
        case 0:
            if (s == end)
                goto End;
            errmsg = "unexpected end of data";
            startinpos = s - starts;
            endinpos = end - starts;
            break;
        case 1:
            errmsg = "invalid start byte";
            startinpos = s - starts;
            endinpos = startinpos + 1;
            break;
        case 2:
        case 3:
        case 4:
            if (s == end) {
                goto End;
            }
            errmsg = "invalid continuation byte";
            startinpos = s - starts;
            endinpos = startinpos + ch - 1;
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
    return _C_UnicodeWriter_Finish(&writer);

onError:
    std::cerr << errmsg << std::endl;
    _C_UnicodeWriter_Dealloc(&writer);
    return;
}


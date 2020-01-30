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

/* stringlib: codec implementations */

// copied from CPython 31e8d69bfe7cf5d4ffe0967cb225d2a8a229cc97

/* Mask to quickly check whether a C 'long' contains a
   non-ASCII, UTF8-encoded char. */
#if (SIZEOF_LONG == 8)
#define ASCII_CHAR_MASK 0x8080808080808080UL
#elif (SIZEOF_LONG == 4)
#define ASCII_CHAR_MASK 0x80808080UL
#else
#error C 'long' size should be either 4 or 8!
#endif

/* 10xxxxxx */
#define IS_CONTINUATION_BYTE(ch) ((ch) >= 0x80 && (ch) < 0xC0)

Py_LOCAL_INLINE(Py_UCS4)
    STRINGLIB(utf8_decode)(const char** inptr, const char* end, STRINGLIB_CHAR* dest, Py_ssize_t* outpos)
{
    Py_UCS4 ch;
    const char* s = *inptr;
    const char* aligned_end = (const char*)_Py_ALIGN_DOWN(end, SIZEOF_LONG);
    STRINGLIB_CHAR* p = dest + *outpos;

    while (s < end)
    {
        ch = (unsigned char)*s;

        if (ch < 0x80)
        {
            /* Fast path for runs of ASCII characters. Given that common UTF-8
               input will consist of an overwhelming majority of ASCII
               characters, we try to optimize for this case by checking
               as many characters as a C 'long' can contain.
               First, check if we can do an aligned read, as most CPUs have
               a penalty for unaligned reads.
            */
            if (_Py_IS_ALIGNED(s, SIZEOF_LONG))
            {
                /* Help register allocation */
                const char* _s = s;
                STRINGLIB_CHAR* _p = p;
                while (_s < aligned_end)
                {
                    /* Read a whole long at a time (either 4 or 8 bytes),
                       and do a fast unrolled copy if it only contains ASCII
                       characters. */
                    unsigned long value = *(unsigned long*)_s;
                    if (value & ASCII_CHAR_MASK)
                        break;
#if PY_LITTLE_ENDIAN
                    _p[0] = (STRINGLIB_CHAR)(value & 0xFFu);
                    _p[1] = (STRINGLIB_CHAR)((value >> 8) & 0xFFu);
                    _p[2] = (STRINGLIB_CHAR)((value >> 16) & 0xFFu);
                    _p[3] = (STRINGLIB_CHAR)((value >> 24) & 0xFFu);
#if SIZEOF_LONG == 8
                    _p[4] = (STRINGLIB_CHAR)((value >> 32) & 0xFFu);
                    _p[5] = (STRINGLIB_CHAR)((value >> 40) & 0xFFu);
                    _p[6] = (STRINGLIB_CHAR)((value >> 48) & 0xFFu);
                    _p[7] = (STRINGLIB_CHAR)((value >> 56) & 0xFFu);
#endif
#else
#if SIZEOF_LONG == 8
                    _p[0] = (STRINGLIB_CHAR)((value >> 56) & 0xFFu);
                    _p[1] = (STRINGLIB_CHAR)((value >> 48) & 0xFFu);
                    _p[2] = (STRINGLIB_CHAR)((value >> 40) & 0xFFu);
                    _p[3] = (STRINGLIB_CHAR)((value >> 32) & 0xFFu);
                    _p[4] = (STRINGLIB_CHAR)((value >> 24) & 0xFFu);
                    _p[5] = (STRINGLIB_CHAR)((value >> 16) & 0xFFu);
                    _p[6] = (STRINGLIB_CHAR)((value >> 8) & 0xFFu);
                    _p[7] = (STRINGLIB_CHAR)(value & 0xFFu);
#else
                    _p[0] = (STRINGLIB_CHAR)((value >> 24) & 0xFFu);
                    _p[1] = (STRINGLIB_CHAR)((value >> 16) & 0xFFu);
                    _p[2] = (STRINGLIB_CHAR)((value >> 8) & 0xFFu);
                    _p[3] = (STRINGLIB_CHAR)(value & 0xFFu);
#endif
#endif
                    _s += SIZEOF_LONG;
                    _p += SIZEOF_LONG;
                }
                s = _s;
                p = _p;
                if (s == end)
                    break;
                ch = (unsigned char)*s;
            }
            if (ch < 0x80)
            {
                s++;
                *p++ = ch;
                continue;
            }
        }

        if (ch < 0xE0)
        {
            /* \xC2\x80-\xDF\xBF -- 0080-07FF */
            Py_UCS4 ch2;
            if (ch < 0xC2)
            {
                /* invalid sequence
                \x80-\xBF -- continuation byte
                \xC0-\xC1 -- fake 0000-007F */
                goto InvalidStart;
            }
            if (end - s < 2)
            {
                /* unexpected end of data: the caller will decide whether
                   it's an error or not */
                break;
            }
            ch2 = (unsigned char)s[1];
            if (!IS_CONTINUATION_BYTE(ch2))
                /* invalid continuation byte */
                goto InvalidContinuation1;
            ch = (ch << 6) + ch2 - ((0xC0 << 6) + 0x80);
            assert((ch > 0x007F) && (ch <= 0x07FF));
            s += 2;
            if (STRINGLIB_MAX_CHAR <= 0x007F || (STRINGLIB_MAX_CHAR < 0x07FF && ch > STRINGLIB_MAX_CHAR))
                /* Out-of-range */
                goto Return;
            *p++ = ch;
            continue;
        }

        if (ch < 0xF0)
        {
            /* \xE0\xA0\x80-\xEF\xBF\xBF -- 0800-FFFF */
            Py_UCS4 ch2, ch3;
            if (end - s < 3)
            {
                /* unexpected end of data: the caller will decide whether
                   it's an error or not */
                if (end - s < 2)
                    break;
                ch2 = (unsigned char)s[1];
                if (!IS_CONTINUATION_BYTE(ch2) || (ch2 < 0xA0 ? ch == 0xE0 : ch == 0xED))
                    /* for clarification see comments below */
                    goto InvalidContinuation1;
                break;
            }
            ch2 = (unsigned char)s[1];
            ch3 = (unsigned char)s[2];
            if (!IS_CONTINUATION_BYTE(ch2))
            {
                /* invalid continuation byte */
                goto InvalidContinuation1;
            }
            if (ch == 0xE0)
            {
                if (ch2 < 0xA0)
                    /* invalid sequence
                       \xE0\x80\x80-\xE0\x9F\xBF -- fake 0000-0800 */
                    goto InvalidContinuation1;
            }
            else if (ch == 0xED && ch2 >= 0xA0)
            {
                /* Decoding UTF-8 sequences in range \xED\xA0\x80-\xED\xBF\xBF
                   will result in surrogates in range D800-DFFF. Surrogates are
                   not valid UTF-8 so they are rejected.
                   See http://www.unicode.org/versions/Unicode5.2.0/ch03.pdf
                   (table 3-7) and http://www.rfc-editor.org/rfc/rfc3629.txt */
                goto InvalidContinuation1;
            }
            if (!IS_CONTINUATION_BYTE(ch3))
            {
                /* invalid continuation byte */
                goto InvalidContinuation2;
            }
            ch = (ch << 12) + (ch2 << 6) + ch3 - ((0xE0 << 12) + (0x80 << 6) + 0x80);
            assert((ch > 0x07FF) && (ch <= 0xFFFF));
            s += 3;
            if (STRINGLIB_MAX_CHAR <= 0x07FF || (STRINGLIB_MAX_CHAR < 0xFFFF && ch > STRINGLIB_MAX_CHAR))
                /* Out-of-range */
                goto Return;
            *p++ = ch;
            continue;
        }

        if (ch < 0xF5)
        {
            /* \xF0\x90\x80\x80-\xF4\x8F\xBF\xBF -- 10000-10FFFF */
            Py_UCS4 ch2, ch3, ch4;
            if (end - s < 4)
            {
                /* unexpected end of data: the caller will decide whether
                   it's an error or not */
                if (end - s < 2)
                    break;
                ch2 = (unsigned char)s[1];
                if (!IS_CONTINUATION_BYTE(ch2) || (ch2 < 0x90 ? ch == 0xF0 : ch == 0xF4))
                    /* for clarification see comments below */
                    goto InvalidContinuation1;
                if (end - s < 3)
                    break;
                ch3 = (unsigned char)s[2];
                if (!IS_CONTINUATION_BYTE(ch3))
                    goto InvalidContinuation2;
                break;
            }
            ch2 = (unsigned char)s[1];
            ch3 = (unsigned char)s[2];
            ch4 = (unsigned char)s[3];
            if (!IS_CONTINUATION_BYTE(ch2))
            {
                /* invalid continuation byte */
                goto InvalidContinuation1;
            }
            if (ch == 0xF0)
            {
                if (ch2 < 0x90)
                    /* invalid sequence
                       \xF0\x80\x80\x80-\xF0\x8F\xBF\xBF -- fake 0000-FFFF */
                    goto InvalidContinuation1;
            }
            else if (ch == 0xF4 && ch2 >= 0x90)
            {
                /* invalid sequence
                   \xF4\x90\x80\80- -- 110000- overflow */
                goto InvalidContinuation1;
            }
            if (!IS_CONTINUATION_BYTE(ch3))
            {
                /* invalid continuation byte */
                goto InvalidContinuation2;
            }
            if (!IS_CONTINUATION_BYTE(ch4))
            {
                /* invalid continuation byte */
                goto InvalidContinuation3;
            }
            ch = (ch << 18) + (ch2 << 12) + (ch3 << 6) + ch4 - ((0xF0 << 18) + (0x80 << 12) + (0x80 << 6) + 0x80);
            assert((ch > 0xFFFF) && (ch <= 0x10FFFF));
            s += 4;
            if (STRINGLIB_MAX_CHAR <= 0xFFFF || (STRINGLIB_MAX_CHAR < 0x10FFFF && ch > STRINGLIB_MAX_CHAR))
                /* Out-of-range */
                goto Return;
            *p++ = ch;
            continue;
        }
        goto InvalidStart;
    }
    ch = 0;
Return:
    *inptr = s;
    *outpos = p - dest;
    return ch;
InvalidStart:
    ch = 1;
    goto Return;
InvalidContinuation1:
    ch = 2;
    goto Return;
InvalidContinuation2:
    ch = 3;
    goto Return;
InvalidContinuation3:
    ch = 4;
    goto Return;
}

#undef ASCII_CHAR_MASK

/* UTF-8 encoder specialized for a Unicode kind to avoid the slow
   PyUnicode_READ() macro. Delete some parts of the code depending on the kind:
   UCS-1 strings don't need to handle surrogates for example. */
int64_t STRINGLIB(utf8_encoder)(char* out_data, STRINGLIB_CHAR* data, Py_ssize_t size)
{
    Py_ssize_t i; /* index into data of next input character */
    char* p;      /* next free byte in output buffer */

#if STRINGLIB_SIZEOF_CHAR == 1
    const Py_ssize_t max_char_size = 2;
#elif STRINGLIB_SIZEOF_CHAR == 2
    const Py_ssize_t max_char_size = 3;
#else /*  STRINGLIB_SIZEOF_CHAR == 4 */
    const Py_ssize_t max_char_size = 4;
#endif
    _C_BytesWriter writer;

    assert(size >= 0);
    _C_BytesWriter_Init(&writer);

    if (size > PY_SSIZE_T_MAX / max_char_size)
    {
        /* integer overflow */
        // TODO: proper memory error
        std::cerr << "memory error in utf8 encoder" << std::endl;
        return 0;
    }

    p = (char*)_C_BytesWriter_Alloc(&writer, size * max_char_size);
    if (p == NULL)
        return 0;

    for (i = 0; i < size;)
    {
        Py_UCS4 ch = data[i++];

        if (ch < 0x80)
        {
            /* Encode ASCII */
            *p++ = (char)ch;
        }
        else
#if STRINGLIB_SIZEOF_CHAR > 1
            if (ch < 0x0800)
#endif
        {
            /* Encode Latin-1 */
            *p++ = (char)(0xc0 | (ch >> 6));
            *p++ = (char)(0x80 | (ch & 0x3f));
        }
#if STRINGLIB_SIZEOF_CHAR > 1
        else if (Py_UNICODE_IS_SURROGATE(ch))
        {
            Py_ssize_t startpos, endpos;

            startpos = i - 1;
            endpos = startpos + 1;

            while ((endpos < size) && Py_UNICODE_IS_SURROGATE(data[endpos]))
                endpos++;

            /* Only overallocate the buffer if it's not the last write */
            writer.overallocate = (endpos < size);

            // TODO: error handlers
            goto error;

            /* If overallocation was disabled, ensure that it was the last
               write. Otherwise, we missed an optimization */
            assert(writer.overallocate || i == size);
        }
        else
#if STRINGLIB_SIZEOF_CHAR > 2
            if (ch < 0x10000)
#endif
        {
            *p++ = (char)(0xe0 | (ch >> 12));
            *p++ = (char)(0x80 | ((ch >> 6) & 0x3f));
            *p++ = (char)(0x80 | (ch & 0x3f));
        }
#if STRINGLIB_SIZEOF_CHAR > 2
        else /* ch >= 0x10000 */
        {
            assert(ch <= MAX_UNICODE);
            /* Encode UCS4 Unicode ordinals */
            *p++ = (char)(0xf0 | (ch >> 18));
            *p++ = (char)(0x80 | ((ch >> 12) & 0x3f));
            *p++ = (char)(0x80 | ((ch >> 6) & 0x3f));
            *p++ = (char)(0x80 | (ch & 0x3f));
        }
#endif /* STRINGLIB_SIZEOF_CHAR > 2 */
#endif /* STRINGLIB_SIZEOF_CHAR > 1 */
    }

    return _C_BytesWriter_Finish(out_data, &writer, p);

#if STRINGLIB_SIZEOF_CHAR > 1
error:
    _C_BytesWriter_Dealloc(&writer);
    return 0;
#endif
}

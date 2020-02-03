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

/* this is sort of a hack.  there's at least one place (formatting
   floats) where some stringlib code takes a different path if it's
   compiled as unicode. */
#define STRINGLIB_IS_UNICODE 1

#define FASTSEARCH ucs4lib_fastsearch
#define STRINGLIB(F) ucs4lib_##F
#define STRINGLIB_OBJECT PyUnicodeObject
#define STRINGLIB_SIZEOF_CHAR 4
#define STRINGLIB_MAX_CHAR 0x10FFFFu
#define STRINGLIB_CHAR Py_UCS4
#define STRINGLIB_TYPE_NAME "unicode"
#define STRINGLIB_PARSE_CODE "U"
#define STRINGLIB_EMPTY unicode_empty
#define STRINGLIB_ISSPACE Py_UNICODE_ISSPACE
#define STRINGLIB_ISLINEBREAK BLOOM_LINEBREAK
#define STRINGLIB_ISDECIMAL Py_UNICODE_ISDECIMAL
#define STRINGLIB_TODECIMAL Py_UNICODE_TODECIMAL
#define STRINGLIB_STR PyUnicode_4BYTE_DATA
#define STRINGLIB_LEN PyUnicode_GET_LENGTH
#define STRINGLIB_NEW _PyUnicode_FromUCS4
#define STRINGLIB_CHECK PyUnicode_Check
#define STRINGLIB_CHECK_EXACT PyUnicode_CheckExact

#define STRINGLIB_TOSTR PyObject_Str
#define STRINGLIB_TOASCII PyObject_ASCII

#define _Py_InsertThousandsGrouping _PyUnicode_ucs4_InsertThousandsGrouping

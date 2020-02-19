# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

from sdc.tests.test_base import TestCase

from sdc.utilities.prange_utils import get_chunks, Chunk


class ChunkTest(TestCase):

    def _get_chunks_data(self):
        yield (5, 5), [
            Chunk(start=0, stop=1),
            Chunk(start=1, stop=2),
            Chunk(start=2, stop=3),
            Chunk(start=3, stop=4),
            Chunk(start=4, stop=5),
        ]
        yield (5, 6), [
            Chunk(start=0, stop=1),
            Chunk(start=1, stop=2),
            Chunk(start=2, stop=3),
            Chunk(start=3, stop=4),
            Chunk(start=4, stop=5),
        ]
        yield (9, 5), [
            Chunk(start=0, stop=2),
            Chunk(start=2, stop=4),
            Chunk(start=4, stop=6),
            Chunk(start=6, stop=8),
            Chunk(start=8, stop=9),
        ]
        yield (9, 4), [
            Chunk(start=0, stop=3),
            Chunk(start=3, stop=5),
            Chunk(start=5, stop=7),
            Chunk(start=7, stop=9),
        ]
        yield (9, 2), [
            Chunk(start=0, stop=5),
            Chunk(start=5, stop=9),
        ]
        yield (9, 3), [
            Chunk(start=0, stop=3),
            Chunk(start=3, stop=6),
            Chunk(start=6, stop=9),
        ]

    def _check_get_chunks(self, args, expected_chunks):
        pyfunc = get_chunks
        cfunc = self.jit(pyfunc)

        self.assertEqual(pyfunc(*args), expected_chunks)
        self.assertEqual(cfunc(*args), expected_chunks)

    def test_get_chunks(self):
        for args, expected_chunks in self._get_chunks_data():
            with self.subTest(args=args):
                self._check_get_chunks(args, expected_chunks)

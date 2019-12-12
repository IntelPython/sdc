# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
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

import csv


def generate(rows, headers, providers, file_name):
    """Generate CSV file.
    rows: rows count
    headers: list of column names
    providers: list of functions whic provide values for corresponding column
    file_name:
    """

    assert len(headers) == len(providers)

    with open(file_name, 'wt') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for i in range(rows):
            writer.writerow({k: p() for k, p in zip(headers, providers)})


def md5(filename):
    """Return MD5 sum of the file."""
    import hashlib
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def csv_file_name(rows=10**5, columns=10, seed=0):
    """Return file name for given parameters."""
    return f"data_{rows}_{columns}_{seed}.csv"


def generate_csv(rows=10**5, columns=10, seed=0):
    """Generate CSV file and return file name."""
    import random

    md5_sums = {
        (10**6, 10, 0): "6fa2a115dfeaee4f574106b513ad79e6"
    }

    file_name = csv_file_name(rows, columns, seed)

    try:
        if md5_sums.get((rows, columns, seed)) == md5(file_name):
            return file_name
    except:
        pass

    r = random.Random(seed)
    generate(rows,
        [f"f{c}" for c in range(columns)],
        [lambda: r.uniform(-1.0, 1.0) for _ in range(columns)],
        file_name
    )

    md5_sums[(rows, columns, seed)] = md5(file_name)

    return file_name

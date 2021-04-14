# *****************************************************************************
# Copyright (c) 2021, Intel Corporation All rights reserved.
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

import numba
import numpy as np
import unittest

from itertools import product, chain, filterfalse, groupby
from numba.core import types
from numba.core.errors import TypingError
from numba import prange
from numba.np.numpy_support import as_dtype
from numba.tests.support import MemoryLeakMixin
from sdc.extensions.sdc_hashmap_type import ConcurrentDict
from sdc.tests.test_base import TestCase
from sdc.tests.test_series import test_global_input_data_float64
from sdc.tests.test_utils import gen_strlist

from sdc.extensions.sdc_hashmap_ext import (
    supported_numeric_key_types,
    supported_numeric_value_types,
    )
from numba.core.extending import register_jitable


int_limits_min = list(map(lambda x: np.iinfo(x).min, ['int32', 'int64', 'uint32', 'uint64']))
int_limits_max = list(map(lambda x: np.iinfo(x).max, ['int32', 'int64', 'uint32', 'uint64']))


global_test_cdict_values = {
    types.Integer: [0, -5, 17] + int_limits_min + int_limits_max,
    types.Float: list(chain.from_iterable(test_global_input_data_float64)),
    types.UnicodeType: ['a1', 'a2', 'b', 'sdf', 'q', 're', 'fde', '']
}


def assert_dict_correct(self, result, fromdata):
    """ This function checks that result's keys and values match data from which it was created,
    i.e. keys match strictly and all values are associated with the same key in fromdata """

    self.assertEqual(set(result.keys()), set(dict(fromdata).keys()))

    def key_func(x):
        return x[0]

    fromdata = sorted(fromdata, key=key_func)
    for k, g in groupby(fromdata, key_func):
        v = result[k]
        group_values_arr = np.array(list(zip(*g))[1])
        group_values_set = set(group_values_arr)
        if isinstance(v, float) and np.isnan(v):
            self.assertTrue(any(np.isnan(group_values_arr)),
                            f"result[{k}] == {v} not found in source values")
        else:
            self.assertIn(v, group_values_set)


class TestHashmapNumeric(MemoryLeakMixin, TestCase):  # FIXME: use unified names (i.e. TBB hashmap of ConcDict)
    """ Verifies correctness of single following specialization of numeric hashmap,
        i.e. when both keys and values are numeric. """

    key_types = supported_numeric_key_types
    value_types = supported_numeric_value_types

    _default_key = 7
    _default_value = 25

    def get_default_key(self, nbtype):
        return as_dtype(nbtype).type(self._default_key)

    def get_default_value(self, nbtype):
        return as_dtype(nbtype).type(self._default_value)

    def get_random_sequence(self, nbtype, n=10):
        assert isinstance(nbtype, types.Number), "Non-numeric type used in TestHashmapNumeric"
        values = global_test_cdict_values[type(nbtype)]
        return np.random.choice(values, n).astype(str(nbtype))

    # **************************** Tests start **************************** #

    def test_hashmap_numeric_create_empty(self):

        @self.jit
        def test_impl(key_type, value_type):
            a_dict = ConcurrentDict.empty(key_type, value_type)
            return len(a_dict)

        expected_res = 0
        for key_type, value_type in product(self.key_types, self.value_types):
            with self.subTest(key_type=key_type, value_type=value_type):
                self.assertEqual(test_impl(key_type, value_type), expected_res)

    def test_hashmap_numeric_create_from_arrays(self):

        @self.jit
        def test_impl(keys, values):
            a_dict = ConcurrentDict.from_arrays(keys, values)
            res = list(a_dict.items())      # this relies on working iterator
            return res

        n = 47
        np.random.seed(0)

        for key_type, value_type in product(self.key_types, self.value_types):
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            source_kv_pairs = list(zip(keys, values))
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, values=values):
                result = test_impl(keys, values)
                assert_dict_correct(self, dict(result), source_kv_pairs)

    def test_hashmap_numeric_create_from_typed_dict(self):

        # FIXME_Numba#XXXX: iterating through typed.Dict fails memleak checks!
        self.disable_leak_check()

        from numba.typed import Dict

        @self.jit
        # FIXME: we still need to implement key_type and value_type properties??
        def test_impl(tdict, key_type, value_type):
            a_dict = ConcurrentDict.empty(key_type, value_type)
            for k, v in tdict.items():
                a_dict[k] = v

            res = list(a_dict.items())      # this relies on working iterator
            return res

        n = 47
        np.random.seed(0)

        for key_type, value_type in product(self.key_types, self.value_types):
            tdict = Dict.empty(key_type, value_type)
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            source_kv_pairs = list(zip(keys, values))
            for k, v in source_kv_pairs:
                tdict[k] = v
            with self.subTest(key_type=key_type, value_type=value_type, tdict=tdict):
                result = test_impl(tdict, key_type, value_type)
                assert_dict_correct(self, dict(result), source_kv_pairs)

    def test_hashmap_numeric_insert(self):

        @self.jit
        def test_impl(key_type, value_type, key, value):
            a_dict = ConcurrentDict.empty(key_type, value_type)
            a_dict[key] = value
            return len(a_dict), a_dict[key]

        for key_type, value_type in product(self.key_types, self.value_types):

            _key = self.get_default_key(key_type)
            _value = self.get_default_value(value_type)
            with self.subTest(key_type=key_type, value_type=value_type):
                self.assertEqual(
                    test_impl(key_type, value_type, _key, _value),
                    (1, _value)
                )

    def test_hashmap_numeric_set_value(self):

        @self.jit
        def test_impl(key, value, new_value):
            a_dict = ConcurrentDict.from_arrays(
                np.array([key, ]),
                np.array([value, ]),
            )

            a_dict[key] = new_value
            return a_dict[key]

        new_value = 11
        for key_type, value_type in product(self.key_types, self.value_types):

            _key = self.get_default_key(key_type)
            _value = self.get_default_value(value_type)
            _new_value = as_dtype(value_type).type(new_value)
            with self.subTest(key_type=key_type, value_type=value_type):
                self.assertEqual(test_impl(_key, _value, _new_value), _new_value)

    def test_hashmap_numeric_lookup_positive(self):

        @self.jit
        def test_impl(key, value):
            a_dict = ConcurrentDict.from_arrays(
                np.array([key, ]),
                np.array([value, ]),
            )
            return a_dict[key]

        for key_type, value_type in product(self.key_types, self.value_types):
            _key = self.get_default_key(key_type)
            _value = self.get_default_value(value_type)
            with self.subTest(key_type=key_type, value_type=value_type):
                self.assertEqual(test_impl(_key, _value), _value)

    def test_hashmap_numeric_lookup_negative(self):

        # this is common for all Numba tests that check exceptions are raised
        self.disable_leak_check()

        @self.jit
        def test_impl(key, value):
            a_dict = ConcurrentDict.from_arrays(
                np.array([key, ]),
                np.array([value, ]),
            )

            return a_dict[2*key]

        for key_type, value_type in product(self.key_types, self.value_types):
            _key = self.get_default_key(key_type)
            _value = self.get_default_value(value_type)
            with self.subTest(key_type=key_type, value_type=value_type):
                with self.assertRaises(KeyError) as raises:
                    test_impl(_key, _value)
                msg = 'ConcurrentDict key not found'
                self.assertIn(msg, str(raises.exception))

    def test_hashmap_numeric_contains(self):

        @self.jit
        def test_impl(key, value):
            a_dict = ConcurrentDict.from_arrays(
                np.array([key, ]),
                np.array([value, ]),
            )
            return key in a_dict, 2*key in a_dict

        expected_res = (True, False)
        for key_type, value_type in product(self.key_types, self.value_types):
            _key = self.get_default_key(key_type)
            _value = self.get_default_value(value_type)
            with self.subTest(key_type=key_type, value_type=value_type):
                self.assertEqual(test_impl(_key, _value), expected_res)

    def test_hashmap_numeric_pop(self):

        @self.jit
        def test_impl(key, value):
            a_dict = ConcurrentDict.from_arrays(
                np.array([key, ]),
                np.array([value, ]),
            )
            a_dict.pop(key)
            return len(a_dict), a_dict.get(key, None)

        expected_res = (0, None)
        for key_type, value_type in product(self.key_types, self.value_types):
            _key = self.get_default_key(key_type)
            _value = self.get_default_value(value_type)
            with self.subTest(key_type=key_type, value_type=value_type):
                self.assertEqual(test_impl(_key, _value), expected_res)

    def test_hashmap_numeric_clear(self):

        @self.jit
        def test_impl(keys, values):
            a_dict = ConcurrentDict.from_arrays(keys, values)
            r1 = len(a_dict)
            a_dict.clear()
            r2 = len(a_dict)
            return r1, r2

        n = 47
        np.random.seed(0)

        for key_type, value_type in product(self.key_types, self.value_types):
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            expected_res = (len(set(keys)), 0)
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, values=values):
                self.assertEqual(test_impl(keys, values), expected_res)

    def test_hashmap_numeric_get(self):

        @self.jit
        def test_impl(key, value, default):
            a_dict = ConcurrentDict.from_arrays(
                np.array([key, ]),
                np.array([value, ]),
            )
            r1 = a_dict.get(key, None)
            r2 = a_dict.get(2*key, default)
            r3 = a_dict.get(2*key)
            return r1, r2, r3

        default_value = 0
        for key_type, value_type in product(self.key_types, self.value_types):
            _key = self.get_default_key(key_type)
            _value = self.get_default_value(value_type)
            _default = value_type(default_value)
            expected_res = (_value, _default, None)
            with self.subTest(key_type=key_type, value_type=value_type):
                self.assertEqual(test_impl(_key, _value, _default), expected_res)

    def test_hashmap_numeric_insert_implicit_cast(self):

        @self.jit
        def test_impl(key_type, value_type, key, value):
            a_dict = ConcurrentDict.empty(key_type, value_type)
            a_dict[key] = value
            return len(a_dict), key in a_dict

        key_type, value_type = types.int64, types.int64
        _key = np.dtype('int16').type(self._default_key)
        _value = np.dtype('uint16').type(self._default_value)
        expected_res = (1, True)
        result = test_impl(key_type, value_type, _key, _value)
        self.assertEqual(result, expected_res)

    def test_hashmap_numeric_insert_cast_fails(self):

        @self.jit
        def test_impl(key_type, value_type, key, value):
            a_dict = ConcurrentDict.empty(key_type, value_type)
            a_dict[key] = value
            return len(a_dict), key in a_dict

        key_type, value_type = types.int64, types.int64
        _key = np.dtype('float32').type(self._default_key)
        _value = np.dtype('uint16').type(self._default_value)
        with self.subTest(subtest='first', key_type=key_type, value_type=value_type):
            with self.assertRaises(TypingError) as raises:
                test_impl(key_type, value_type, _key, _value)
            msg = 'TypingError: cannot safely cast'
            self.assertIn(msg, str(raises.exception))

        _key = np.dtype('uint16').type(self._default_key)
        _value = np.dtype('float64').type(self._default_value)
        with self.subTest(subtest='second', key_type=key_type, value_type=value_type):
            with self.assertRaises(TypingError) as raises:
                test_impl(key_type, value_type, _key, _value)
            msg = 'TypingError: cannot safely cast'
            self.assertIn(msg, str(raises.exception))

    def test_hashmap_numeric_use_prange(self):

        @self.jit
        def test_impl(key_type, value_type, keys, values):
            a_dict = ConcurrentDict.empty(key_type, value_type)
            for i in prange(len(keys)):
                a_dict[keys[i]] = values[i]

            res = list(a_dict.items())      # this relies on working iterator
            return res

        n = 47
        np.random.seed(0)

        for key_type, value_type in product(self.key_types, self.value_types):
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            source_kv_pairs = list(zip(keys, values))
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, values=values):
                result = test_impl(key_type, value_type, keys, values)
                assert_dict_correct(self, dict(result), source_kv_pairs)

    def test_hashmap_numeric_fromkeys_class(self):

        @self.jit
        def test_impl(keys, value):
            a_dict = ConcurrentDict.fromkeys(keys, value)
            check_keys = np.array([k in a_dict for k in keys])
            return len(a_dict), np.all(check_keys)

        n = 47
        np.random.seed(0)

        for key_type, value_type in product(self.key_types, self.value_types):
            keys = self.get_random_sequence(key_type, n)
            value = self.get_default_value(value_type)
            expected_res = (len(set(keys)), True)
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, value=value):
                self.assertEqual(test_impl(keys, value), expected_res)

    def test_hashmap_numeric_fromkeys_dictobject(self):

        @self.jit
        def test_impl(keys, value):
            a_dict = ConcurrentDict.empty(types.int64, types.float64)
            res = a_dict.fromkeys(keys, value)
            check_keys = np.array([k in res for k in keys])
            return len(res), np.all(check_keys), len(a_dict)

        n = 47
        np.random.seed(0)

        for key_type, value_type in product(self.key_types, self.value_types):
            keys = self.get_random_sequence(key_type, n)
            value = self.get_default_value(value_type)
            expected_res = (len(set(keys)), True, 0)
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, value=value):
                self.assertEqual(test_impl(keys, value), expected_res)

    def test_hashmap_numeric_update(self):

        @self.jit
        def test_impl(keys1, values1, keys2, values2):
            a_dict = ConcurrentDict.from_arrays(keys1, values1)
            other_dict = ConcurrentDict.from_arrays(keys2, values2)
            r1 = len(a_dict)
            a_dict.update(other_dict)
            r2 = len(a_dict)
            check_keys = np.array([k in a_dict for k in keys2])
            return r1, r2, np.all(check_keys)

        n = 47
        np.random.seed(0)

        for key_type, value_type in product(self.key_types, self.value_types):
            keys1 = self.get_random_sequence(key_type, n)
            keys2 = self.get_random_sequence(key_type, 2 * n)
            values1 = self.get_random_sequence(value_type, n)
            values2 = self.get_random_sequence(value_type, 2 * n)
            before_size = len(set(keys1))
            after_size = len(set(keys1).union(set(keys2)))
            expected_res = (before_size, after_size, True)
            with self.subTest(key_type=key_type, value_type=value_type,
                              keys1=keys1, values1=values1,
                              keys2=keys2, values2=values2):
                result = test_impl(keys1, values1, keys2, values2)
                self.assertEqual(result, expected_res)

    def test_hashmap_numeric_iterator(self):

        @self.jit
        def test_impl(keys, values):
            a_dict = ConcurrentDict.from_arrays(keys, values)
            res = []
            for k in a_dict:
                res.append(k)
            return res

        n = 47
        np.random.seed(0)

        for key_type, value_type in product(self.key_types, self.value_types):
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, values=values):
                # expect a list of keys returned in some (i.e. non-fixed) order
                result = test_impl(keys, values)
                self.assertEqual(set(result), set(keys))

    def test_hashmap_numeric_iterator_freed(self):

        @self.jit
        def test_impl(keys, values):
            a_dict = ConcurrentDict.from_arrays(keys, values)
            dict_iter = iter(a_dict)
            r1 = next(dict_iter)
            r2 = next(dict_iter)
            r3 = next(dict_iter)
            return r1, r2, r3

        n = 47
        np.random.seed(0)

        for key_type, value_type in product(self.key_types, self.value_types):
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, values=values):
                result = test_impl(keys, values)
                self.assertTrue(set(result).issubset(set(keys)),
                                f"Some key ({result}) is not found in source keys: {keys}")

    def test_hashmap_numeric_keys(self):

        @self.jit
        def test_impl(keys, values):
            a_dict = ConcurrentDict.from_arrays(keys, values)
            res = []
            for k in a_dict.keys():
                res.append(k)
            return res

        n = 47
        np.random.seed(0)

        for key_type, value_type in product(self.key_types, self.value_types):
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, values=values):
                # expect a list of keys returned in some (i.e. non-fixed) order
                result = test_impl(keys, values)
                self.assertEqual(set(result), set(keys))

    def test_hashmap_numeric_items(self):

        @self.jit
        def test_impl(keys, values):
            a_dict = ConcurrentDict.from_arrays(keys, values)
            res = []
            for k, v in a_dict.items():
                res.append((k, v))
            return res

        n = 47
        np.random.seed(0)

        for key_type, value_type in product(self.key_types, self.value_types):
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            source_kv_pairs = list(zip(keys, values))
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, values=values):
                result = test_impl(keys, values)
                assert_dict_correct(self, dict(result), source_kv_pairs)

    def test_hashmap_numeric_values(self):

        @self.jit
        def test_impl(keys, values):
            a_dict = ConcurrentDict.from_arrays(keys, values)
            res = []
            for k, v in a_dict.items():
                res.append((k, v))
            return res

        n = 47
        np.random.seed(0)

        for key_type, value_type in product(self.key_types, self.value_types):
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            source_kv_pairs = list(zip(keys, values))
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, values=values):
                result = test_impl(keys, values)
                assert_dict_correct(self, dict(result), source_kv_pairs)


class TestHashmapGeneric(MemoryLeakMixin, TestCase):
    """ Verifies correctness of following specializations:
        generic-key hashmap, generic-value hashmap and generic-key-and-value.
        Generic means objects are passed as void*. """

    @classmethod
    def key_value_combinations(cls):
        res = filterfalse(
            lambda x: isinstance(x[0], types.Number) and isinstance(x[1], types.Number),
            product(cls.key_types, cls.value_types)
        )
        return res

    key_types = [
        types.int32,
        types.uint32,
        types.int64,
        types.uint64,
        types.unicode_type,
    ]

    value_types = [
        types.int32,
        types.uint32,
        types.int64,
        types.uint64,
        types.float32,
        types.float64,
        types.unicode_type,
    ]

    _default_data = {
        types.Integer: 11,
        types.Float: 42.3,
        types.UnicodeType: 'sdf',
    }

    def get_default_scalar(self, nbtype):
        meta_type = type(nbtype)
        if isinstance(nbtype, types.Number):
            res = as_dtype(nbtype).type(self._default_data[meta_type])
        elif isinstance(nbtype, types.UnicodeType):
            res = self._default_data[meta_type]
        return res

    # TO-DO: this looks too similar to gen_arr_of_dtype in perf_tests, re-use?
    def get_random_sequence(self, nbtype, n=10):
        if isinstance(nbtype, types.Number):
            values = np.arange(n // 2, dtype=as_dtype(nbtype))
            res = np.random.choice(values, n)
        elif isinstance(nbtype, types.UnicodeType):
            values = gen_strlist(n // 2)
            res = list(np.random.choice(values, n))
        return res

    # **************************** Tests start **************************** #

    def test_hashmap_generic_create_empty(self):

        @self.jit
        def test_impl(key_type, value_type):
            a_dict = ConcurrentDict.empty(key_type, value_type)
            return len(a_dict)

        expected_res = 0
        for key_type, value_type in self.key_value_combinations():
            with self.subTest(key_type=key_type, value_type=value_type):
                self.assertEqual(test_impl(key_type, value_type), expected_res)

    def test_hashmap_generic_create_from_typed_dict(self):

        # FIXME_Numba#XXXX: iterating through typed.Dict fails memleak checks!
        self.disable_leak_check()

        from numba.typed import Dict

        @self.jit
        def test_impl(tdict, key_type, value_type):
            a_dict = ConcurrentDict.empty(key_type, value_type)
            for k, v in tdict.items():
                a_dict[k] = v

            res = list(a_dict.items())      # this relies on working iterator
            return res

        n = 47
        np.random.seed(0)

        for key_type, value_type in self.key_value_combinations():
            tdict = Dict.empty(key_type, value_type)
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            source_kv_pairs = list(zip(keys, values))
            for k, v in source_kv_pairs:
                tdict[k] = v
            with self.subTest(key_type=key_type, value_type=value_type, tdict=tdict):
                result = test_impl(tdict, key_type, value_type)
                assert_dict_correct(self, dict(result), source_kv_pairs)

    def test_hashmap_generic_insert(self):

        @self.jit
        def test_impl(key_type, value_type, key, value):
            a_dict = ConcurrentDict.empty(key_type, value_type)
            a_dict[key] = value
            return len(a_dict), a_dict[key]

        for key_type, value_type in self.key_value_combinations():

            _key = self.get_default_scalar(key_type)
            _value = self.get_default_scalar(value_type)
            with self.subTest(key_type=key_type, value_type=value_type):
                self.assertEqual(
                    test_impl(key_type, value_type, _key, _value),
                    (1, _value)
                )

    def test_hashmap_generic_set_value(self):

        @self.jit
        def test_impl(key, value, new_value):
            a_dict = ConcurrentDict.fromkeys([key], value)

            a_dict[key] = new_value
            return a_dict[key]

        np.random.seed(0)

        for key_type, value_type in self.key_value_combinations():
            _key = self.get_default_scalar(key_type)
            _value = self.get_default_scalar(value_type)
            _new_value = self.get_random_sequence(value_type)[0]
            with self.subTest(key_type=key_type, value_type=value_type):
                self.assertEqual(test_impl(_key, _value, _new_value), _new_value)

    def test_hashmap_generic_lookup_positive(self):

        @self.jit
        def test_impl(key, value):
            a_dict = ConcurrentDict.fromkeys([key], value)
            return a_dict[key]

        for key_type, value_type in self.key_value_combinations():
            _key = self.get_default_scalar(key_type)
            _value = self.get_default_scalar(value_type)
            with self.subTest(key_type=key_type, value_type=value_type):
                self.assertEqual(test_impl(_key, _value), _value)

    def test_hashmap_generic_lookup_negative(self):

        # this is common for all Numba tests that check exceptions are raised
        self.disable_leak_check()

        @self.jit
        def test_impl(key, value):
            a_dict = ConcurrentDict.fromkeys([key], value)

            return a_dict[2*key]

        for key_type, value_type in self.key_value_combinations():
            _key = self.get_default_scalar(key_type)
            _value = self.get_default_scalar(value_type)
            with self.subTest(key_type=key_type, value_type=value_type):
                with self.assertRaises(KeyError) as raises:
                    test_impl(_key, _value)
                msg = 'ConcurrentDict key not found'
                self.assertIn(msg, str(raises.exception))

    def test_hashmap_generic_contains(self):

        @self.jit
        def test_impl(key, value):
            a_dict = ConcurrentDict.fromkeys([key], value)
            return key in a_dict, 2*key in a_dict

        expected_res = (True, False)
        for key_type, value_type in self.key_value_combinations():
            _key = self.get_default_scalar(key_type)
            _value = self.get_default_scalar(value_type)
            with self.subTest(key_type=key_type, value_type=value_type):
                self.assertEqual(test_impl(_key, _value), expected_res)

    def test_hashmap_generic_pop_positive(self):

        @self.jit
        def test_impl(key, value):
            a_dict = ConcurrentDict.fromkeys([key], value)
            r1 = a_dict.pop(key)
            return r1, len(a_dict), a_dict.get(key, None)

        for key_type, value_type in self.key_value_combinations():
            _key = self.get_default_scalar(key_type)
            _value = self.get_default_scalar(value_type)
            expected_res = (_value, 0, None)
            with self.subTest(key_type=key_type, value_type=value_type):
                self.assertEqual(test_impl(_key, _value), expected_res)

    def test_hashmap_generic_pop_negative(self):

        @self.jit
        def test_impl(key, value):
            a_dict = ConcurrentDict.fromkeys([key], value)
            r1 = a_dict.pop(2*key)
            r2 = a_dict.pop(2*key, 2*value)
            return r1, r2

        for key_type, value_type in self.key_value_combinations():
            _key = self.get_default_scalar(key_type)
            _value = self.get_default_scalar(value_type)
            expected_res = (None, 2*_value)
            with self.subTest(key_type=key_type, value_type=value_type):
                self.assertEqual(test_impl(_key, _value), expected_res)

    def test_hashmap_generic_clear(self):

        @self.jit
        def test_impl(keys, values):
            a_dict = ConcurrentDict.fromkeys(keys, values[0])
            r1 = len(a_dict)
            a_dict.clear()
            r2 = len(a_dict)
            return r1, r2

        n = 47
        np.random.seed(0)

        for key_type, value_type in self.key_value_combinations():
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            expected_res = (len(set(keys)), 0)
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, values=values):
                self.assertEqual(test_impl(keys, values), expected_res)

    def test_hashmap_generic_get(self):

        @self.jit
        def test_impl(key, value, default):
            a_dict = ConcurrentDict.fromkeys([key], value)
            r1 = a_dict.get(key, None)
            r2 = a_dict.get(2*key, default)
            r3 = a_dict.get(2*key)
            return r1, r2, r3

        np.random.seed(0)

        for key_type, value_type in self.key_value_combinations():
            _key = self.get_default_scalar(key_type)
            _value = self.get_default_scalar(value_type)
            _default = self.get_random_sequence(value_type)[0]
            expected_res = (_value, _default, None)
            with self.subTest(key_type=key_type, value_type=value_type):
                self.assertEqual(test_impl(_key, _value, _default), expected_res)

    def test_hashmap_generic_use_prange(self):

        @self.jit
        def test_impl(key_type, value_type, keys, values):
            a_dict = ConcurrentDict.empty(key_type, value_type)
            for i in prange(len(keys)):
                a_dict[keys[i]] = values[i]

            res = list(a_dict.items())      # this relies on working iterator
            return res

        n = 47
        np.random.seed(0)

        for key_type, value_type in self.key_value_combinations():
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            source_kv_pairs = list(zip(keys, values))
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, values=values):
                result = test_impl(key_type, value_type, keys, values)
                assert_dict_correct(self, dict(result), source_kv_pairs)

    def test_hashmap_generic_fromkeys_class(self):

        @self.jit
        def test_impl(keys, value):
            a_dict = ConcurrentDict.fromkeys(keys, value)
            check_keys = np.array([k in a_dict for k in keys])
            return len(a_dict), np.all(check_keys)

        n = 47
        np.random.seed(0)

        for key_type, value_type in self.key_value_combinations():
            keys = self.get_random_sequence(key_type, n)
            value = self.get_default_scalar(value_type)
            expected_res = (len(set(keys)), True)
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys):
                self.assertEqual(test_impl(keys, value), expected_res)

    def test_hashmap_generic_fromkeys_dictobject(self):

        @self.jit
        def test_impl(keys, value):
            a_dict = ConcurrentDict.empty(types.int64, types.float64)
            res = a_dict.fromkeys(keys, value)
            check_keys = np.array([k in res for k in keys])
            return len(res), np.all(check_keys), len(a_dict)

        n = 47
        np.random.seed(0)

        for key_type, value_type in self.key_value_combinations():
            keys = self.get_random_sequence(key_type, n)
            value = self.get_default_scalar(value_type)
            expected_res = (len(set(keys)), True, 0)
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys):
                self.assertEqual(test_impl(keys, value), expected_res)

    def test_hashmap_generic_update(self):

        @self.jit
        def test_impl(keys1, values1, keys2, values2):
            a_dict = ConcurrentDict.fromkeys(keys1, values1[0])
            for k, v in zip(keys1, values1):
                a_dict[k] = v

            other_dict = ConcurrentDict.fromkeys(keys2, values2[0])
            for k, v in zip(keys2, values2):
                other_dict[k] = v

            r1 = len(a_dict)
            a_dict.update(other_dict)
            r2 = len(a_dict)
            check_keys = np.array([k in a_dict for k in keys2])

            return r1, r2, np.all(check_keys)

        n = 47
        np.random.seed(0)

        for key_type, value_type in self.key_value_combinations():
            keys1 = self.get_random_sequence(key_type, n)
            keys2 = self.get_random_sequence(key_type, 2 * n)
            values1 = self.get_random_sequence(value_type, n)
            values2 = self.get_random_sequence(value_type, 2 * n)
            before_size = len(set(keys1))
            after_size = len(set(keys1).union(set(keys2)))
            expected_res = (before_size, after_size, True)
            with self.subTest(key_type=key_type, value_type=value_type,
                              keys1=keys1, values1=values1,
                              keys2=keys2, values2=values2):
                result = test_impl(keys1, values1, keys2, values2)
                self.assertEqual(result, expected_res)

    def test_hashmap_generic_iterator(self):

        @self.jit
        def test_impl(keys, values):
            a_dict = ConcurrentDict.fromkeys(keys, values[0])
            for k, v in zip(keys, values):
                a_dict[k] = v

            res = []
            for k in a_dict:
                res.append(k)
            return res

        n = 47
        np.random.seed(0)

        for key_type, value_type in self.key_value_combinations():
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, values=values):
                # expect a list of keys returned in some (i.e. non-fixed) order
                result = test_impl(keys, values)
                self.assertEqual(set(result), set(keys))

    def test_hashmap_generic_iterator_freed(self):

        @self.jit
        def test_impl(keys, values):
            a_dict = ConcurrentDict.fromkeys(keys, values[0])
            for k, v in zip(keys, values):
                a_dict[k] = v

            dict_iter = iter(a_dict)
            r1 = next(dict_iter)
            r2 = next(dict_iter)
            r3 = next(dict_iter)
            return r1, r2, r3

        n = 47
        np.random.seed(0)

        for key_type, value_type in self.key_value_combinations():
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, values=values):
                result = test_impl(keys, values)
                self.assertTrue(set(result).issubset(set(keys)),
                                f"Some key ({result}) is not found in source keys: {keys}")

    def test_hashmap_generic_keys(self):

        @self.jit
        def test_impl(keys, values):
            a_dict = ConcurrentDict.fromkeys(keys, values[0])
            for k, v in zip(keys, values):
                a_dict[k] = v

            res = []
            for k in a_dict.keys():
                res.append(k)
            return res

        n = 47
        np.random.seed(0)

        for key_type, value_type in self.key_value_combinations():
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, values=values):
                # expect a list of keys returned in some (i.e. non-fixed) order
                result = test_impl(keys, values)
                self.assertEqual(set(result), set(keys))

    def test_hashmap_generic_items(self):

        @self.jit
        def test_impl(keys, values):
            a_dict = ConcurrentDict.fromkeys(keys, values[0])
            for k, v in zip(keys, values):
                a_dict[k] = v

            res = []
            for k, v in a_dict.items():
                res.append((k, v))
            return res

        n = 47
        np.random.seed(0)

        for key_type, value_type in self.key_value_combinations():
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            source_kv_pairs = list(zip(keys, values))
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, values=values):
                result = test_impl(keys, values)
                assert_dict_correct(self, dict(result), source_kv_pairs)

    def test_hashmap_generic_values(self):

        @self.jit
        def test_impl(keys, values):
            a_dict = ConcurrentDict.fromkeys(keys, values[0])
            for k, v in zip(keys, values):
                a_dict[k] = v

            res = []
            for k, v in a_dict.items():
                res.append((k, v))
            return res

        n = 47
        np.random.seed(0)

        for key_type, value_type in self.key_value_combinations():
            keys = self.get_random_sequence(key_type, n)
            values = self.get_random_sequence(value_type, n)
            source_kv_pairs = list(zip(keys, values))
            with self.subTest(key_type=key_type, value_type=value_type, keys=keys, values=values):
                result = test_impl(keys, values)
                assert_dict_correct(self, dict(result), source_kv_pairs)

    def test_hashmap_generic_tuple_keys(self):

        @self.jit
        def test_impl(key_type, value_type, keys, values):
            a_dict = ConcurrentDict.empty(key_type, value_type)
            for k, v in zip(keys, values):
                a_dict[k] = v
            return len(a_dict)

        n = 47
        np.random.seed(0)

        key_values = list(product([0, 1], repeat=5))
        key_type, value_type = numba.typeof(key_values[0]), types.int64

        keys = [key_values[i] for i in np.random.randint(0, len(key_values), n)]
        values = self.get_random_sequence(types.int64, n)
        expected_res = len(set(keys))
        self.assertEqual(
            test_impl(key_type, value_type, keys, values),
            expected_res
        )


if __name__ == "__main__":
    unittest.main()

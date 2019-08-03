from collections import namedtuple
import numpy as np

from numba import types
from numba.extending import overload

import hpat
from hpat.utils import get_ctypes_ptr, _numba_to_c_type_map
from hpat.timsort import getitem_arr_tup
from hpat.str_ext import string_type
from hpat.str_arr_ext import (string_array_type, to_string_list,
                              get_offset_ptr, get_data_ptr, convert_len_arr_to_offset,
                              pre_alloc_string_array, num_total_chars)


# metadata required for shuffle
# send_counts -> pre, single
# recv_counts -> single
# send_buff
# out_arr
# n_send  -> single
# n_out  -> single
# send_disp -> single
# recv_disp -> single
# tmp_offset -> single
# string arrays
# send_counts_char -> pre
# recv_counts_char
# send_arr_lens -> pre
# send_arr_chars
# send_disp_char
# recv_disp_char
# tmp_offset_char
# dummy array to key reference count alive, since ArrayCTypes can't be
# passed to jitclass TODO: update
# send_arr_chars_arr


PreShuffleMeta = namedtuple('PreShuffleMeta',
                            'send_counts, send_counts_char_tup, send_arr_lens_tup')

ShuffleMeta = namedtuple('ShuffleMeta',
                         ('send_counts, recv_counts, n_send, n_out, send_disp, recv_disp, '
                          'tmp_offset, send_buff_tup, out_arr_tup, send_counts_char_tup, '
                          'recv_counts_char_tup, send_arr_lens_tup, send_arr_chars_tup, '
                          'send_disp_char_tup, recv_disp_char_tup, tmp_offset_char_tup, '
                          'send_arr_chars_arr_tup'))


# before shuffle, 'send_counts' is needed as well as
# 'send_counts_char' and 'send_arr_lens' for every string type
def alloc_pre_shuffle_metadata(arr, data, n_pes, is_contig):
    return PreShuffleMeta(np.zeros(n_pes, np.int32), ())


@overload(alloc_pre_shuffle_metadata)
def alloc_pre_shuffle_metadata_overload(key_arrs, data, n_pes, is_contig):

    func_text = "def f(key_arrs, data, n_pes, is_contig):\n"
    # send_counts
    func_text += "  send_counts = np.zeros(n_pes, np.int32)\n"

    # send_counts_char, send_arr_lens for strings
    n_keys = len(key_arrs.types)
    n_str = 0
    for i, typ in enumerate(key_arrs.types + data.types):
        if typ == string_array_type:
            func_text += ("  arr = key_arrs[{}]\n".format(i) if i < n_keys
                else "  arr = data[{}]\n".format(i - n_keys))
            func_text += "  send_counts_char_{} = np.zeros(n_pes, np.int32)\n".format(n_str)
            func_text += "  send_arr_lens_{} = np.empty(1, np.uint32)\n".format(n_str)
            # needs allocation since written in update before finalize
            func_text += "  if is_contig:\n"
            func_text += "    send_arr_lens_{} = np.empty(len(arr), np.uint32)\n".format(n_str)
            n_str += 1

    count_char_tup = ", ".join("send_counts_char_{}".format(i)
                                                        for i in range(n_str))
    lens_tup = ", ".join("send_arr_lens_{}".format(i) for i in range(n_str))
    extra_comma = "," if n_str == 1 else ""
    func_text += "  return PreShuffleMeta(send_counts, ({}{}), ({}{}))\n".format(
        count_char_tup, extra_comma, lens_tup, extra_comma)

    # print(func_text)

    loc_vars = {}
    exec(func_text, {'np': np, 'PreShuffleMeta': PreShuffleMeta}, loc_vars)
    alloc_impl = loc_vars['f']
    return alloc_impl


# 'send_counts' is updated, and 'send_counts_char' and 'send_arr_lens'
# for every string type
def update_shuffle_meta(pre_shuffle_meta, node_id, ind, val, data, is_contig=True):
    pre_shuffle_meta.send_counts[node_id] += 1


@overload(update_shuffle_meta)
def update_shuffle_meta_overload(pre_shuffle_meta, node_id, ind, val, data, is_contig=True):
    func_text = "def f(pre_shuffle_meta, node_id, ind, val, data, is_contig=True):\n"
    func_text += "  pre_shuffle_meta.send_counts[node_id] += 1\n"
    n_keys = len(val.types)
    n_str = 0
    for i, typ in enumerate(val.types + data.types):
        if typ in (string_type, string_array_type):
            val_or_data = 'val[{}]'.format(i) if i < n_keys else 'data[{}]'.format(i - n_keys)
            func_text += "  n_chars = len({})\n".format(val_or_data)
            func_text += "  pre_shuffle_meta.send_counts_char_tup[{}][node_id] += n_chars\n".format(n_str)
            func_text += "  if is_contig:\n"
            func_text += "    pre_shuffle_meta.send_arr_lens_tup[{}][ind] = n_chars\n".format(n_str)
            n_str += 1

    # print(func_text)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    update_impl = loc_vars['f']
    return update_impl


def finalize_shuffle_meta(arrs, data, pre_shuffle_meta, n_pes, is_contig, init_vals=()):
    return ShuffleMeta()


@overload(finalize_shuffle_meta)
def finalize_shuffle_meta_overload(key_arrs, data, pre_shuffle_meta, n_pes, is_contig, init_vals=()):

    func_text = "def f(key_arrs, data, pre_shuffle_meta, n_pes, is_contig, init_vals=()):\n"
    # common metas: send_counts, recv_counts, tmp_offset, n_out, n_send, send_disp, recv_disp
    func_text += "  send_counts = pre_shuffle_meta.send_counts\n"
    func_text += "  recv_counts = np.empty(n_pes, np.int32)\n"
    func_text += "  tmp_offset = np.zeros(n_pes, np.int32)\n"  # for non-contig
    func_text += "  hpat.distributed_api.alltoall(send_counts, recv_counts, 1)\n"
    func_text += "  n_out = recv_counts.sum()\n"
    func_text += "  n_send = send_counts.sum()\n"
    func_text += "  send_disp = hpat.hiframes.join.calc_disp(send_counts)\n"
    func_text += "  recv_disp = hpat.hiframes.join.calc_disp(recv_counts)\n"

    n_keys = len(key_arrs.types)
    n_all = len(key_arrs.types + data.types)
    n_str = 0

    for i, typ in enumerate(key_arrs.types + data.types):
        func_text += ("  arr = key_arrs[{}]\n".format(i) if i < n_keys
                      else "  arr = data[{}]\n".format(i - n_keys))
        if isinstance(typ, types.Array):
            func_text += "  out_arr_{} = fix_cat_array_type(np.empty(n_out, arr.dtype))\n".format(i)
            func_text += "  send_buff_{} = arr\n".format(i)
            func_text += "  if not is_contig:\n"
            if i >= n_keys and init_vals != ():
                func_text += "    send_buff_{} = fix_cat_array_type(np.full(n_send, init_vals[{}], arr.dtype))\n".format(i, i - n_keys)
            else:
                func_text += "    send_buff_{} = fix_cat_array_type(np.empty(n_send, arr.dtype))\n".format(i)
        else:
            assert typ == string_array_type
            # send_buff is None for strings
            func_text += "  send_buff_{} = None\n".format(i)
            # send/recv counts
            func_text += "  send_counts_char_{} = pre_shuffle_meta.send_counts_char_tup[{}]\n".format(n_str, n_str)
            func_text += "  recv_counts_char_{} = np.empty(n_pes, np.int32)\n".format(n_str)
            func_text += ("  hpat.distributed_api.alltoall("
                          "send_counts_char_{}, recv_counts_char_{}, 1)\n").format(n_str, n_str)
            # alloc output
            func_text += "  n_all_chars = recv_counts_char_{}.sum()\n".format(n_str)
            func_text += "  out_arr_{} = pre_alloc_string_array(n_out, n_all_chars)\n".format(i)
            # send/recv disp
            func_text += ("  send_disp_char_{} = hpat.hiframes.join."
                          "calc_disp(send_counts_char_{})\n").format(n_str, n_str)
            func_text += ("  recv_disp_char_{} = hpat.hiframes.join."
                          "calc_disp(recv_counts_char_{})\n").format(n_str, n_str)

            # tmp_offset_char, send_arr_lens
            func_text += "  tmp_offset_char_{} = np.zeros(n_pes, np.int32)\n".format(n_str)
            func_text += "  send_arr_lens_{} = pre_shuffle_meta.send_arr_lens_tup[{}]\n".format(n_str, n_str)
            # send char arr
            # TODO: arr refcount if arr is not stored somewhere?
            func_text += "  send_arr_chars_arr_{} = np.empty(1, np.uint8)\n".format(n_str)
            func_text += "  send_arr_chars_{} = get_ctypes_ptr(get_data_ptr(arr))\n".format(n_str)
            func_text += "  if not is_contig:\n"
            func_text += "    send_arr_lens_{} = np.empty(n_send, np.uint32)\n".format(n_str)
            func_text += "    s_n_all_chars = send_counts_char_{}.sum()\n".format(n_str)
            func_text += "    send_arr_chars_arr_{} = np.empty(s_n_all_chars, np.uint8)\n".format(n_str)
            func_text += "    send_arr_chars_{} = get_ctypes_ptr(send_arr_chars_arr_{}.ctypes)\n".format(n_str, n_str)
            n_str += 1

    send_buffs = ", ".join("send_buff_{}".format(i) for i in range(n_all))
    out_arrs = ", ".join("out_arr_{}".format(i) for i in range(n_all))
    all_comma = "," if n_all == 1 else ""
    send_counts_chars = ", ".join("send_counts_char_{}".format(i) for i in range(n_str))
    recv_counts_chars = ", ".join("recv_counts_char_{}".format(i) for i in range(n_str))
    send_arr_lens = ", ".join("send_arr_lens_{}".format(i) for i in range(n_str))
    send_arr_chars = ", ".join("send_arr_chars_{}".format(i) for i in range(n_str))
    send_disp_chars = ", ".join("send_disp_char_{}".format(i) for i in range(n_str))
    recv_disp_chars = ", ".join("recv_disp_char_{}".format(i) for i in range(n_str))
    tmp_offset_chars = ", ".join("tmp_offset_char_{}".format(i) for i in range(n_str))
    send_arr_chars_arrs = ", ".join("send_arr_chars_arr_{}".format(i) for i in range(n_str))
    str_comma = "," if n_str == 1 else ""


    func_text += ('  return ShuffleMeta(send_counts, recv_counts, n_send, '
        'n_out, send_disp, recv_disp, tmp_offset, ({}{}), ({}{}), ({}{}), ({}{}), ({}{}), ({}{}), ({}{}), ({}{}), ({}{}), ({}{}), )\n').format(
            send_buffs, all_comma, out_arrs, all_comma, send_counts_chars, str_comma, recv_counts_chars, str_comma,
            send_arr_lens, str_comma, send_arr_chars, str_comma, send_disp_chars, str_comma, recv_disp_chars, str_comma,
            tmp_offset_chars, str_comma, send_arr_chars_arrs, str_comma
        )

    # print(func_text)

    loc_vars = {}
    exec(func_text, {'np': np, 'hpat': hpat,
                     'pre_alloc_string_array': pre_alloc_string_array,
                     'num_total_chars': num_total_chars,
                     'get_data_ptr': get_data_ptr,
                     'ShuffleMeta': ShuffleMeta,
                     'get_ctypes_ptr': get_ctypes_ptr,
                     'fix_cat_array_type':
                     hpat.hiframes.pd_categorical_ext.fix_cat_array_type}, loc_vars)
    finalize_impl = loc_vars['f']
    return finalize_impl


def alltoallv(arr, m):
    return

@overload(alltoallv)
def alltoallv_impl(arr, metadata):
    if isinstance(arr, types.Array):
        def a2av_impl(arr, metadata):
            hpat.distributed_api.alltoallv(
                metadata.send_buff, metadata.out_arr, metadata.send_counts,
                metadata.recv_counts, metadata.send_disp, metadata.recv_disp)
        return a2av_impl

    assert arr == string_array_type
    int32_typ_enum = np.int32(_numba_to_c_type_map[types.int32])
    char_typ_enum = np.int32(_numba_to_c_type_map[types.uint8])

    def a2av_str_impl(arr, metadata):
        # TODO: increate refcount?
        offset_ptr = get_offset_ptr(metadata.out_arr)
        hpat.distributed_api.c_alltoallv(
            metadata.send_arr_lens.ctypes, offset_ptr, metadata.send_counts.ctypes,
            metadata.recv_counts.ctypes, metadata.send_disp.ctypes, metadata.recv_disp.ctypes, int32_typ_enum)
        hpat.distributed_api.c_alltoallv(
            metadata.send_arr_chars, get_data_ptr(metadata.out_arr), metadata.send_counts_char.ctypes,
            metadata.recv_counts_char.ctypes, metadata.send_disp_char.ctypes, metadata.recv_disp_char.ctypes, char_typ_enum)
        convert_len_arr_to_offset(offset_ptr, metadata.n_out)
    return a2av_str_impl


def alltoallv_tup(arrs, shuffle_meta):
    return arrs


@overload(alltoallv_tup)
def alltoallv_tup_overload(arrs, meta):
    func_text = "def f(arrs, meta):\n"
    n_str = 0
    for i, typ in enumerate(arrs.types):
        if isinstance(typ, types.Array):
            func_text += ("  hpat.distributed_api.alltoallv("
                          "meta.send_buff_tup[{}], meta.out_arr_tup[{}], meta.send_counts,"
                          "meta.recv_counts, meta.send_disp, meta.recv_disp)\n").format(i, i)
        else:
            assert typ == string_array_type
            func_text += "  offset_ptr_{} = get_offset_ptr(meta.out_arr_tup[{}])\n".format(i, i)

            func_text += ("  hpat.distributed_api.c_alltoallv("
                          "meta.send_arr_lens_tup[{}].ctypes, offset_ptr_{}, meta.send_counts.ctypes, "
                          "meta.recv_counts.ctypes, meta.send_disp.ctypes, "
                          "meta.recv_disp.ctypes, int32_typ_enum)\n").format(n_str, i)

            func_text += ("  hpat.distributed_api.c_alltoallv("
                          "meta.send_arr_chars_tup[{}], get_data_ptr(meta.out_arr_tup[{}]),"
                          "meta.send_counts_char_tup[{}].ctypes, meta.recv_counts_char_tup[{}].ctypes,"
                          "meta.send_disp_char_tup[{}].ctypes, meta.recv_disp_char_tup[{}].ctypes,"
                          "char_typ_enum)\n").format(n_str, i, n_str, n_str, n_str, n_str)

            func_text += "  convert_len_arr_to_offset(offset_ptr_{}, meta.n_out)\n".format(i)
            n_str += 1

    func_text += "  return ({}{})\n".format(
        ','.join(['meta.out_arr_tup[{}]'.format(i) for i in range(arrs.count)]),
        "," if arrs.count == 1 else "")

    int32_typ_enum = np.int32(_numba_to_c_type_map[types.int32])
    char_typ_enum = np.int32(_numba_to_c_type_map[types.uint8])
    loc_vars = {}
    exec(func_text, {'hpat': hpat, 'get_offset_ptr': get_offset_ptr,
                     'get_data_ptr': get_data_ptr, 'int32_typ_enum': int32_typ_enum,
                     'char_typ_enum': char_typ_enum,
                     'convert_len_arr_to_offset': convert_len_arr_to_offset}, loc_vars)
    a2a_impl = loc_vars['f']
    return a2a_impl


def _get_keys_tup(recvs, key_arrs):
    return recvs[:len(key_arrs)]


@overload(_get_keys_tup)
def _get_keys_tup_overload(recvs, key_arrs):
    n_keys = len(key_arrs.types)
    func_text = "def f(recvs, key_arrs):\n"
    res = ",".join("recvs[{}]".format(i) for i in range(n_keys))
    func_text += "  return ({}{})\n".format(res, "," if n_keys == 1 else "")
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars['f']
    return impl


def _get_data_tup(recvs, key_arrs):
    return recvs[len(key_arrs):]


@overload(_get_data_tup)
def _get_data_tup_overload(recvs, key_arrs):
    n_keys = len(key_arrs.types)
    n_all = len(recvs.types)
    n_data = n_all - n_keys
    func_text = "def f(recvs, key_arrs):\n"
    res = ",".join("recvs[{}]".format(i) for i in range(n_keys, n_all))
    func_text += "  return ({}{})\n".format(res, "," if n_data == 1 else "")
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars['f']
    return impl


# returns scalar instead of tuple if only one array
def getitem_arr_tup_single(arrs, i):
    return arrs[0][i]


@overload(getitem_arr_tup_single)
def getitem_arr_tup_single_overload(arrs, i):
    if len(arrs.types) == 1:
        return lambda arrs, i: arrs[0][i]
    return lambda arrs, i: getitem_arr_tup(arrs, i)


def val_to_tup(val):
    return (val,)


@overload(val_to_tup)
def val_to_tup_overload(val):
    if isinstance(val, types.BaseTuple):
        return lambda val: val
    return lambda val: (val,)

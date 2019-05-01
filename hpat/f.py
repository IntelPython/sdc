def f(key_arrs, data, pre_shuffle_meta, n_pes, is_contig, init_vals=()):
  send_counts = pre_shuffle_meta.send_counts
  recv_counts = np.empty(n_pes, np.int32)
  tmp_offset = np.zeros(n_pes, np.int32)
  hpat.distributed_api.alltoall(send_counts, recv_counts, 1)
  n_out = recv_counts.sum()
  n_send = send_counts.sum()
  send_disp = hpat.hiframes.join.calc_disp(send_counts)
  recv_disp = hpat.hiframes.join.calc_disp(recv_counts)
  arr = key_arrs[0]
  out_arr_0 = fix_cat_array_type(np.empty(n_out, arr.dtype))
  send_buff_0 = arr
  if not is_contig:
    send_buff_0 = fix_cat_array_type(np.empty(n_send, arr.dtype))
  arr = data[0]
  send_buff_1 = None
  send_counts_char_0 = pre_shuffle_meta.send_counts_char_tup[0]
  recv_counts_char_0 = np.empty(n_pes, np.int32)
  hpat.distributed_api.alltoall(send_counts_char_0, recv_counts_char_0, 1)
  n_all_chars = recv_counts_char_0.sum()
  out_arr_1 = pre_alloc_string_array(n_out, n_all_chars)
  send_disp_char_0 = hpat.hiframes.join.calc_disp(send_counts_char_0)
  recv_disp_char_0 = hpat.hiframes.join.calc_disp(recv_counts_char_0)
  tmp_offset_char_0 = np.zeros(n_pes, np.int32)
  send_arr_lens_0 = pre_shuffle_meta.send_arr_lens_tup[0]
  send_arr_chars_arr_1 = np.empty(1, np.uint8)
  send_arr_chars_1 = get_ctypes_ptr(get_data_ptr(arr))
  if not is_contig:
    send_arr_lens_0 = np.empty(n_send, np.uint32)
    s_n_all_chars = send_counts_char_0.sum()
    send_arr_chars_arr_0 = np.empty(s_n_all_chars, np.uint8)
    send_arr_chars_0 = get_ctypes_ptr(send_arr_chars_arr_0.ctypes)
  return ShuffleMeta(send_counts, recv_counts, n_send, n_out, send_disp,
    recv_disp, tmp_offset, (send_buff_0, send_buff_1), (out_arr_0, out_arr_1),
    (send_counts_char_0,), (recv_counts_char_0,), (send_arr_lens_0,),
    (send_arr_chars_0,), (send_disp_char_0,), (recv_disp_char_0,),
    (tmp_offset_char_0,), (send_arr_chars_arr_0,), )

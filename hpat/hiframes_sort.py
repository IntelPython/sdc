import numpy as np
import math
import numba
from numba import types
import hpat

#MIN_SAMPLES = 1000000
MIN_SAMPLES = 100
samplePointsPerPartitionHint = 20
MPI_ROOT = 0

key_typ = numba.int64[:]
data_tup_typ = types.Tuple([])

sort_state_spec = [
    ('key_arr', key_typ),
    ('aLength', numba.intp),
    ('minGallop', numba.intp),
    ('tmpLength', numba.intp),
    ('tmp', key_typ),
    ('stackSize', numba.intp),
    ('runBase', numba.int64[:]),
    ('runLen', numba.int64[:]),
    ('data', data_tup_typ),
    ('tmp_data', data_tup_typ),
]

SortStateCL = numba.jitclass(sort_state_spec)(hpat.timsort.SortState)

@hpat.jit
def parallel_sort(key_arr, n_total):
    # local sort
    n_local = len(key_arr)
    sort_state = SortStateCL(key_arr, n_local, ())
    hpat.timsort.sort(sort_state, key_arr, 0, n_local, ())

    n_pes = hpat.distributed_api.get_size()
    my_rank = hpat.distributed_api.get_rank()

    # similar to Spark's sample computation Partitioner.scala
    sampleSize = min(samplePointsPerPartitionHint * n_pes, MIN_SAMPLES)

    fraction = min(sampleSize / max(n_total, 1), 1.0)
    n_loc_samples = min(math.ceil(fraction * n_local), n_local)
    inds = np.random.randint(0, n_local, n_loc_samples)
    samples = key_arr[inds]
    # print(sampleSize, fraction, n_local, n_loc_samples, len(samples))

    all_samples = hpat.distributed_api.gatherv(samples)
    bounds = np.empty(n_pes-1, key_arr.dtype)

    if my_rank == MPI_ROOT:
        all_samples.sort()
        n_samples = len(all_samples)
        step = math.ceil(n_samples / n_pes)
        for i in range(n_pes - 1):
            bounds[i] = all_samples[min((i + 1) * step, n_samples - 1)]
        # print(bounds)

    hpat.distributed_api.bcast(bounds)

    # calc send/recv counts
    send_counts = np.zeros(n_pes, np.int32)
    recv_counts = np.empty(n_pes, np.int32)
    node_id = 0
    for i in range(n_local):
        if node_id < (n_pes - 1) and key_arr[i] >= bounds[node_id]:
            node_id += 1
        send_counts[node_id] += 1
    hpat.distributed_api.alltoall(send_counts, recv_counts, 1)

    # shuffle
    n_out = recv_counts.sum()
    out_data = np.empty(n_out, key_arr.dtype)
    send_disp = hpat.hiframes_join.calc_disp(send_counts)
    recv_disp = hpat.hiframes_join.calc_disp(recv_counts)
    hpat.distributed_api.alltoallv(key_arr, out_data, send_counts, recv_counts, send_disp, recv_disp)

    # TODO: use k-way merge instead of sort
    # sort output
    sort_state_o = SortStateCL(out_data, n_out, ())
    hpat.timsort.sort(sort_state_o, out_data, 0, n_out, ())

    return my_rank, out_data

n = 100
A = np.arange(n)
b = parallel_sort(A, n * 5)
print(b)

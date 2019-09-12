import pyarrow.parquet as pq
import hpat
from hpat import prange
import numpy as np
import argparse
import time


@hpat.jit
def kde():
    t = pq.read_table('kde.parquet')
    df = t.to_pandas()
    X = df['points'].values
    b = 0.5
    points = np.array([-1.0, 2.0, 5.0])
    N = points.shape[0]
    n = X.shape[0]
    exps = 0
    t1 = time.time()
    for i in prange(n):
        p = X[i]
        d = (-(p - points)**2) / (2 * b**2)
        m = np.min(d)
        exps += m - np.log(b * N) + np.log(np.sum(np.exp(d - m)))
    t = time.time() - t1
    print("Execution time:", t, "\nresult:", exps)
    return exps


def main():
    parser = argparse.ArgumentParser(description='Kernel-Density')
    args = parser.parse_args()

    res = kde()


if __name__ == '__main__':
    main()

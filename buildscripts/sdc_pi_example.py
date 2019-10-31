import hpat
import numpy as np
import time

@hpat.jit
def calc_pi(n):
    t1 = time.time()
    x = 2 * np.random.ranf(n) - 1
    y = 2 * np.random.ranf(n) - 1
    pi = 4 * np.sum(x**2 + y**2 < 1) / n
    print("Execution time:", time.time()-t1, "\nresult:", pi)
    return pi

calc_pi(2 * 10**8)
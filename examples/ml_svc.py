import numpy as np
import hpat
from hpat import prange
import hpat.ml
import time

hpat.multithread_mode = True

@hpat.jit
def f(N, D, M):
    X = np.random.ranf((N, D))
    y = np.empty(N)
    for i in prange(N):
        y[i] = i%4
    p = np.random.ranf((M, D))
    clf = hpat.ml.SVC(n_classes=4)
    t1 = time.time()
    clf.train(X, y)
    res = clf.predict(p)
    print("Exec time:", time.time()-t1)
    return res.sum()

N = 1024*16
D = 20
M = 128

print(f(N, D, M))

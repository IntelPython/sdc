import numpy as np
import hpat
from hpat import prange
import hpat.ml
import time

#hpat.multithread_mode = True

@hpat.jit
def f(N, D, M):
    X = np.random.randint(0, 5, size=(N, D)).astype(np.int32)
    y = np.empty(N, dtype=np.int32)
    for i in prange(N):
        y[i] = i%4
    p = np.random.randint(0, 5, size=(M, D)).astype(np.int32)
    clf = hpat.ml.MultinomialNB(n_classes=4)
    t1 = time.time()
    clf.train(X, y)
    res = clf.predict(p)
    print("Exec time:", time.time()-t1)
    return res.sum()

N = 1024*128
D = 20
M = 128

print(f(N, D, M))

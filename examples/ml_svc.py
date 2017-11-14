import numpy as np
import hpat
from hpat import prange
import hpat.ml

hpat.multithread_mode = True

@hpat.jit
def f(N, D, M):
    X = np.random.ranf((N, D))
    y = np.empty(N)
    for i in prange(N):
        y[i] = i%2
    p = np.random.ranf((M, D))
    clf = hpat.ml.SVC(n_classes=2)
    clf.train(X, y)
    res = clf.predict(p)
    return res.sum()

N = 1024
D = 20
M = 128

print(f(N, D, M))

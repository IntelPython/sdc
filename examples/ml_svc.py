import numpy as np
import hpat
import hpat.ml

@hpat.jit
def f(X, y, p):
    clf = hpat.ml.SVC(n_classes=2)
    clf.train(X, y)
    return clf.predict(p)

N = 1000
D = 20
np.random.seed(10)
X = np.random.ranf((N, D))
y = np.empty(N)
for i in range(N):
    y[i] = i%2
p = np.random.ranf((10, D))
print(f(X, y, p))

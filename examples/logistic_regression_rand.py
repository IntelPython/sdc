import numpy as np
import hpat
import time


@hpat.jit
def logistic_regression(iterations):
    t1 = time.time()
    N = 10**8
    D = 10
    g = 2 * np.random.ranf(D) - 1
    X = 2 * np.random.ranf((N, D)) - 1
    Y = ((np.dot(X, g) > 0.0) == (np.random.ranf(N) > .90)) + .0

    w = np.random.ranf(D) - .5
    for i in range(iterations):
        w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X, w))) - 1.0) * Y), X)
        R = np.dot(X, w) > 0.0
        accuracy = np.sum(R == Y) / N

    print(accuracy, w)
    print("Execution time:", time.time() - t1)
    return w


w = logistic_regression(20)

import numpy as np
import hpat

@hpat.jit
def logistic_regression(iterations):
    print("generating random data...")
    N = 10**8
    D = 10
    g = 2*np.random.ranf(D)-1
    X = 2*np.random.ranf((N,D))-1
    Y = (np.dot(X,g)>0.0)==(np.random.ranf(N)>.90)

    w = 2*np.random.ranf(D)-1
    for i in range(iterations):
        w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X,w))) - 1.0) * Y), X)
        R = np.dot(X,w)>0.0
        accuracy = np.sum(R==Y)/N
        print(accuracy, w)

w = logistic_regression(2000)

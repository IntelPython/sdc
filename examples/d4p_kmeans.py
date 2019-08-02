import daal4py
import daal4py.hpat
import hpat
import numpy as np


@hpat.jit(nopython=True)
def kmeans(N, D, nClusters, maxit):
    a = np.random.ranf((N, D))  # doesn't make much sense, but ok for now
    kmi = daal4py.kmeans_init(nClusters, method='plusPlusDense')
    km = daal4py.kmeans(nClusters, maxit)
    kmr = km.compute(a, kmi.compute(a).centroids)
    return (kmr.centroids, kmr.assignments, kmr.objectiveFunction, kmr.goalFunction, kmr.nIterations)


print(kmeans(10000, 20, 2, 30))

hpat.distribution_report()

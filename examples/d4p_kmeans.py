import daal4py
import hpat
import numpy as np

daal4py.daalinit(spmd=True)

@hpat.jit
def kmeans(N, D, nClusters, fptype, initmethod, seed, oversamplingFactor, nRounds, maxIterations, method, accuracyThreshold, gamma, assignFlag):
    a = np.random.ranf((N,D)) # doesn't make much sense, but ok for now
    kmi = daal4py.kmeans_init(nClusters, fptype, initmethod, seed, oversamplingFactor, nRounds)
    km = daal4py.kmeans(nClusters, maxIterations, fptype, method, accuracyThreshold, gamma, assignFlag)
    kmr = km.compute(a, kmi.compute(a).centroids)
    return (kmr.centroids, kmr.assignments, kmr.objectiveFunction, kmr.goalFunction, kmr.nIterations)

print(kmeans(10000, 20, 2, "double", "defaultDense", 111, 2.2, 33, 300, "lloydDense", 0.00001, 0.1, "True"))

hpat.distribution_report()

daal4py.daalfini()

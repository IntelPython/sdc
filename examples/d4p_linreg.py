import daal4py
import hpat
import numpy as np

daal4py.daalinit(spmd=True)

@hpat.jit
def linreg(N, D):
    data = np.random.ranf((N,D))
    gt = np.random.ranf((N,2))
    tres = daal4py.linear_regression_training(interceptFlag=True, method='qrDense').compute(data, gt)
    #FIXME res.model.InterceptFlag
    pres = daal4py.linear_regression_prediction().compute(data, tres.model)
    return (pres.prediction[0], tres.model.NumberOfBetas, tres.model.NumberOfResponses, tres.model.Beta, tres.model.NumberOfFeatures)

print(linreg(1000, 10))

hpat.distribution_report()

daal4py.daalfini()

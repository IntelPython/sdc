import daal4py
import hpat
import numpy as np

daal4py.daalinit(spmd=True)

@hpat.jit
def linreg(N, D, fptype, method, intercept):
    data = np.random.ranf((N,D))
    gt = np.random.ranf((N,2))
    tres = daal4py.linear_regression_training(fptype, method, intercept).compute(data, gt)
    #FIXME res.model.InterceptFlag
    pres = daal4py.linear_regression_prediction(fptype, method).compute(data, tres.model)
    return (pres.prediction[0], tres.model.NumberOfBetas, tres.model.NumberOfResponses, tres.model.Beta, tres.model.NumberOfFeatures)

print(linreg(1000, 10, "double", "defaultDense", "true"))

hpat.distribution_report()

daal4py.daalfini()

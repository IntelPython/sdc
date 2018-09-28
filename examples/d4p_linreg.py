import daal4py
import hpat
import numpy as np

daal4py.daalinit(spmd=True)

#@hpat.jit
#def tunbox(r):
#    print(r.model.NumberOfBetas)

@hpat.jit
def linreg(N, D):
    data = np.random.ranf((N,D))
    gt = np.random.ranf((N,2))
    tres = daal4py.linear_regression_training(interceptFlag=True, method='qrDense').compute(data, gt)
    data = np.random.ranf((N/2,D))
    pres = daal4py.linear_regression_prediction().compute(data, tres.model)
    #tunbox(pres)
    return tres, pres  #(pres.prediction[0], tres.model.NumberOfBetas, tres.model.NumberOfResponses, tres.model.Beta, tres.model.NumberOfFeatures)

t,p = linreg(1000, 10)
print(p.prediction[0], t.model.NumberOfBetas)

hpat.distribution_report()

daal4py.daalfini()

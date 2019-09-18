import daal4py
import daal4py.hpat
import hpat
import numpy as np


@hpat.jit
def lr_predict(N, D, model):
    data = np.random.ranf((N / 2, D))
    return daal4py.linear_regression_prediction().compute(data, model)


@hpat.jit
def lr_train(N, D):
    data = np.random.ranf((N, D))
    gt = np.random.ranf((N, 2))
    return daal4py.linear_regression_training(interceptFlag=True, method='qrDense').compute(data, gt)


t_res = lr_train(1000, 10)
p_res = lr_predict(1000, 10, t_res.model)

print(p_res.prediction[0], t_res.model.NumberOfBetas)

hpat.distribution_report()

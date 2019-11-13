# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************


try:
    import daal4py as d4p
except ImportError:
    print('Ignoring daal4py tests.')
else:
    import unittest
    import pandas as pd
    import numpy as np
    from math import sqrt
    import numba
    import hpat
    from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                                       count_parfor_OneDs, count_array_OneDs,
                                       count_parfor_OneD_Vars, count_array_OneD_Vars,
                                       dist_IR_contains)

    class TestD4P(unittest.TestCase):
        def test_logistic_regression(self):
            '''
            Testing logistic regression including
               * result and model boxing/unboxing
               * optional and required arguments passing
            '''
            def train_impl(n, d):
                X = np.ones((n, d), dtype=np.double) + .5
                Y = np.ones((n, 1), dtype=np.double)
                algo = d4p.logistic_regression_training(2,
                                                        penaltyL1=0.1,
                                                        penaltyL2=0.1,
                                                        interceptFlag=True)
                return algo.compute(X, Y)

            def prdct_impl(n, d, model):
                w = np.ones((n, d), dtype=np.double) - 22.5
                algo = d4p.logistic_regression_prediction(
                    2,
                    resultsToCompute="computeClassesLabels|computeClassesProbabilities|computeClassesLogProbabilities"
                )
                return algo.compute(w, model)

            train_hpat = hpat.jit(train_impl)
            prdct_hpat = hpat.jit(prdct_impl)
            n = 11
            d = 4
            pred_impl = prdct_impl(n, d, train_impl(n, d).model).prediction
            pred_hpat = prdct_hpat(n, d, train_hpat(n, d).model).prediction

            np.testing.assert_allclose(pred_impl, pred_hpat)

    if __name__ == "__main__":
        unittest.main()

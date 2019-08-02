import numpy as np
import time
import pandas as pd
import hpat


@hpat.jit
def accel_infer(n):

    t1 = time.time()
    X = np.random.ranf(n)
    Y = np.random.ranf(n)
    Z = np.random.ranf(n)

    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

    g = 9.81
    df['accel'] = np.sqrt(df.X**2 + df.Y**2 + (df.Z - g)**2)
    threshold = df.accel.mean() + 5 * df.accel.std()
    df['is_brake'] = (df.rolling(10)['accel'].mean() > threshold)

    df.is_brake.fillna(False, inplace=True)
    checksum = df.is_brake.sum()
    t2 = time.time()
    print("exec time:", t2 - t1)
    return checksum


n = 10**8
accel_infer(n)

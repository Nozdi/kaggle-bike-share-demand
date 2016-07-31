import numpy as np


def rmsle(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))

    # numpy.expm1 inverse of log1p

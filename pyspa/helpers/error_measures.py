import numpy as np

def calc_mae(observed, predicted):
    diff = observed - predicted
    return np.mean(np.abs(diff))

def calc_rmse(observed, predicted):
    diff = observed - predicted
    return np.sqrt(np.mean(np.power(diff, 2.0)))

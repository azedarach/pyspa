import numpy as np

def calc_projection_shifts(sorted_data):
    (T, n) = sorted_data.shape
    t_hat = np.zeros(T)
    i = n - 2
    idxs = i * np.ones(T)
    num_to_resolve = T
    while num_to_resolve > 0 and i >= 0:
        t_hat[idxs == i] = ((np.sum(sorted_data[idxs == i, i+1:], axis=1) - 1)
                            / (n - i - 1))
        idxs[(t_hat >= sorted_data[:, i]) & (idxs == i)] = -2
        idxs[idxs == i] = i - 1
        i -= 1
        num_to_resolve = np.sum(idxs == i)

    t_hat[idxs == -1] = (np.sum(sorted_data[idxs == -1, :], axis=1) - 1) / n

    return t_hat

def simplex_projection(data):
    ndims = data.ndim
    if ndims == 1:
        T = 1
        n = data.shape[0]
        sorted_data = np.sort(np.reshape(data, (1, n)))
    elif ndims == 2:
        (T, n) = data.shape
        sorted_data = np.sort(data, axis=1)
    else:
        raise ValueError("input data must be a vector or matrix")

    t_hat = calc_projection_shifts(sorted_data)
    projections = np.fmax(data - np.reshape(t_hat, (T, 1)), 0)

    if ndims == 1:
        return np.ravel(projections)
    else:
        return projections

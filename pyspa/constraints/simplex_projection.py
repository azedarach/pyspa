import numpy as np

def simplex_projection(data):
    shape = data.shape
    if np.size(shape) == 1:
        T = 1
        n = shape[0]
        data_sorted = np.sort(np.reshape(data, (1, n)))
    elif np.size(shape) == 2:
        (T, n) = shape
        data_sorted = np.sort(data, axis=1)
    else:
        raise ValueError("input data must be a vector or matrix")

    t_hat = np.zeros(T)
    i = n - 2
    idxs = i * np.ones(T)
    num_to_resolve = T
    while num_to_resolve > 0 and i >= 0:
        t_hat[idxs == i] = ((np.sum(data_sorted[idxs == i, i+1:], axis=1) - 1)
                            / (n - i - 1))
        idxs[(t_hat >= data_sorted[:, i]) & (idxs == i)] = -2
        idxs[idxs == i] = i - 1
        i -= 1
        num_to_resolve = np.sum(idxs == i)

    t_hat[idxs == -1] = (np.sum(data_sorted[idxs == -1, :], axis=1) - 1) / n

    projections = np.fmax(data - np.reshape(t_hat, (T, 1)), 0)

    if np.size(shape) == 1:
        return np.ravel(projections)
    else:
        return projections

import numpy as np

def rectangle_projection(data, lower_bounds=None, upper_bounds=None):
    shape = data.shape
    if np.size(shape) == 1:
        T = 1
        n = shape[0]
        projections = np.reshape(data, (1, n))
    else:
        (T, n) = shape
        projections = data

    if lower_bounds is not None:
        if np.size(lower_bounds) != n:
            raise ValueError("lower bounds must be provided for all dimensions")
        projections = np.fmax(projections,
                              np.stack([lower_bounds for rows in range(T)],
                                       axis=0))

    if upper_bounds is not None:
        if np.size(upper_bounds) != n:
            raise ValueError("upper bounds must be provided for all dimensions")
        projections = np.fmin(projections,
                              np.stack([upper_bounds for rows in range(T)],
                                       axis=0))

    if np.size(shape) == 1:
        return np.ravel(projections)
    else:
        return projections

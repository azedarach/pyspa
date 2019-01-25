import numpy as np

def read_data(input_file, delimiter=",", names=True, verbose=False):
    data = np.genfromtxt(input_file, names=names, delimiter=delimiter)
    fields = [f for f in data.dtype.fields]

    if "t" not in fields:
        raise RuntimeError("no time data found in dataset")
    t = data["t"]

    if verbose:
        print("Reading " + str(np.size(t)) + " points")

    n = np.sum(np.array([f.find("xt") for f in fields]) == 0)

    if n == 0:
        raise RuntimeError("no data found for xt variables in dataset")
    elif verbose:
        print("Found number of slow variables n = " + str(n))

    m = int(np.sum(np.array([f.find("yt") for f in fields]) == 0) / n)
    if m == 0:
        raise RuntimeError("no data found for yt variables in dataset")
    elif verbose:
        print("Found number of fast variables m = " + str(m))

    x = np.vstack([data["xt" + str(i)] for i in range(1, n + 1)])
    y = np.zeros((n, m, np.size(t)))
    for i in range(n):
        for j in range(m):
            y[i,j,:] = data["yt" + str(i + 1) + str(j+1)]

    return (t, x, y)

def build_dataset(x, y, discard_fraction=0.0, verbose=False):
    if discard_fraction < 0.0 or discard_fraction > 1.0:
        raise ValueError(
            "fraction of points to discard must be between 0 and 1")

    T = x.shape[1]
    n = x.shape[0]
    m = y.shape[1]

    y_flat = np.reshape(np.moveaxis(y, -1, 0), (T, n * m))
    data = np.hstack([np.transpose(x), y_flat])

    row_skip = int(np.floor(discard_fraction * T))

    if verbose:
        print("Skipping initial " + str(row_skip) + " points")

    return data[row_skip:,:]

def get_random_affiliations(T, K):
    gamma = np.random.random((T, K))
    normalizations = np.sum(gamma, axis=1)
    return np.divide(gamma, normalizations[:, np.newaxis])

import numpy as np

from sklearn.cluster import KMeans

def positive_section(A):
    return (A >= 0) * A

def negative_section(A):
    return (A < 0) * (-A)

def nndsvd(A, k):
    (m, n) = A.shape

    if k >= min(m, n):
        raise ValueError(
            "approximate rank k must be less than maximum rank of A")

    (U, S, Vh) = np.linalg.svd(A)

    if U[0, 0] < 0:
        U = -U
        Vh = -Vh

    W = np.zeros((m, k))
    H = np.zeros((k, n))

    W[:, 0] = np.sqrt(S[0]) * U[:, 0]
    H[0, :] = np.sqrt(S[0]) * Vh[0, :]

    for j in range(1, k):
        x = U[:, j]
        y = Vh[j, :]
        x_plus = positive_section(x)
        x_minus = negative_section(x)
        y_plus = positive_section(y)
        y_minus = negative_section(y)

        xp_norm = np.linalg.norm(x_plus)
        xm_norm = np.linalg.norm(x_minus)
        yp_norm = np.linalg.norm(y_plus)
        ym_norm = np.linalg.norm(y_minus)

        mu_plus = xp_norm * yp_norm
        mu_minus = xm_norm * ym_norm

        if mu_plus > mu_minus:
            u = x_plus / xp_norm
            v = y_plus / yp_norm
            sigma = mu_plus
        else:
            u = x_minus / xm_norm
            v = y_minus / ym_norm
            sigma = mu_minus

        W[:, j] = np.sqrt(S[j] * sigma) * u
        H[j, :] = np.sqrt(S[j] * sigma) * np.transpose(v)

    return (W, H)

def nndsvda(A, k):
    mu = np.mean(A)
    (W, H) = nndsvd(A, k)

    W[W == 0] = mu
    H[H == 0] = mu

    return (W, H)

def nndsvdar(A, k, scale=0.01):
    mu = np.mean(A)
    (W, H) = nndsvd(A, k)

    W_mask = W < 0
    W[W_mask] = np.random.uniform(0., scale * mu, size=np.sum(W_mask))

    H_mask = H < 0
    H[H_mask] = np.random.uniform(0., scale * mu, size=np.sum(H_mask))

    return (W, H)

def semi_nmf_update_dictionary(X, W, H):
    wtw = np.transpose(W) @ W
    wtx = np.transpose(W) @ X
    return np.linalg.pinv(wtw) @ wtx

def semi_nmf_update_representation(X, W, H):
    hht = H @ np.transpose(H)
    whhtm = W @ negative_section(hht)
    whhtp = W @ positive_section(hht)
    xht = X @ np.transpose(H)
    xhtp = positive_section(xht)
    xhtm = negative_section(xht)
    epsilon = 1e-16
    return W * np.sqrt(np.divide(whhtm + xhtp, whhtp + xhtm + epsilon))

def semi_nmf(X, n_clusters, W_init=None, delta=0.2, tolerance=1.e-3,
             max_iterations=1000):
    (n_samples, feature_dim) = X.shape

    if W_init is None:
        labels = KMeans(n_clusters=n_clusters).fit(X).labels_
        W_init = np.full((n_samples, n_clusters), delta)
        for j in range(n_clusters):
            mask = labels == j
            W_init[mask, j] += 1
    else:
        if W_init.shape != (n_samples, n_clusters):
            raise ValueError("incompatible shape for initial guess for W")

    convergence_check = lambda old_W, new_W : (np.max(np.abs(old_W - new_W))
                                               < tolerance)
    is_converged = False
    new_W = W_init
    new_H = np.zeros((n_clusters, feature_dim))

    iterations = 0
    while not is_converged and iterations < max_iterations:
        old_W = new_W
        old_H = new_H
        new_H = semi_nmf_update_dictionary(X, old_W, old_H)
        new_W = semi_nmf_update_representation(X, old_W, new_H)
        is_converged = convergence_check(old_W, new_W)
        iterations += 1

    return (new_W, new_H)

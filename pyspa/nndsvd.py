import numpy as np


def _positive_section(A):
    return (A >= 0) * A


def _negative_section(A):
    return (A < 0) * (-A)


def nndsvd(A, k):
    (m, n) = A.shape

    if k >= min(m, n):
        raise ValueError(
            'approximate rank k must be less than maximum rank of A')

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
        x_plus = _positive_section(x)
        x_minus = _negative_section(x)
        y_plus = _positive_section(y)
        y_minus = _negative_section(y)

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

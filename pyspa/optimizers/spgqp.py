import numpy as np

# assumes QP problem of form q^T.x + (1/2) x^T.P.x
# where x is required to lie in a d-dimensional simplex
# and x0 is a d-dimensional vector
def spgqp(p, q, x0, projector=None,
          tol=1.e-3, max_iterations=10000,
          m=7, gamma=1.e-4, sigma=0.99, alpha_max=1e4):

    if projector is None:
        projector = lambda x : x

    if m < 1:
        raise ValueError("number of points retained must be at least one")

    if sigma <= 0 or sigma >= 1:
        raise ValueError("sigma must be in the interval (0, 1)")

    if gamma <= 0 or gamma >= 1:
        raise ValueError("gamma must be in the interval (0, 1)")

    if max_iterations < 1:
        raise ValueError("maximum number of iterations must be at least one")

    xk = projector(x0)
    gk = p @ xk + q
    fk = 0.5 * xk.dot(gk + q)

    dinf = projector(xk - gk) - xk
    converged = np.sqrt(np.linalg.norm(dinf)) < tol

    alpha_k = 1.0
    if np.max(np.abs(dinf)) != 0:
        alpha_k = 1.0 / np.max(np.abs(dinf))

    f_prev = fk * np.ones(m)

    iterations = 0
    while not converged and iterations < max_iterations:
        dk = projector(xk - alpha_k * gk) - xk
        pdk = p @ dk
        dkdk = dk.dot(dk)
        dkpdk = dk.dot(pdk)
        dkgk = dk.dot(gk)

        converged = np.sqrt(np.linalg.norm(projector(xk - gk) - xk)) < tol
#        converged = np.sqrt(np.linalg.norm(dk)) < tol

        f_max = np.max(f_prev)
        xi = (f_max - fk) / dkpdk
        beta_bar = -dkgk / dkpdk
        beta_hat = gamma * beta_bar + np.sqrt(gamma ** 2 * beta_bar ** 2
                                              + 2 * xi)
        beta_k = min(sigma, beta_hat)

        xk = xk + beta_k * dk
        gk = gk + beta_k * pdk
        fk = fk + beta_k * dkgk + 0.5 * beta_k ** 2 * dkpdk

        f_prev[:-1] = f_prev[1:]
        f_prev[-1] = fk

        alpha_k = dkdk / dkpdk

        iterations += 1

    if iterations == max_iterations and not converged:
        return None
    else:
        return xk

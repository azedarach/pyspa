import numpy as np

def spg(f, df, x0, projector=None,
        tol=1.e-3, max_iterations=5000, m=7, gamma=1e-4,
        alpha_min=1e-16, alpha_max=1e3,
        sigma_1=0.1, sigma_2=0.9):

    if projector is None:
        projector = lambda x : x

    if m < 1:
        raise ValueError("number of points retained must be at least one")

    if alpha_min <= 0:
        raise ValueError("alpha_min must be positive")

    if alpha_max <= 0:
        raise ValueError("alpha_max must be positive")

    if sigma_1 <= 0 or sigma_1 >= 1:
        raise ValueError("sigma_1 must be in the interval (0, 1)")

    if sigma_2 <= 0 or sigma_1 >= 1:
        raise ValueError("sigma_2 must be in the interval (0, 1)")

    if gamma <= 0 or gamma >= 1:
        raise ValueError("gamma must be in the interval (0, 1)")

    if max_iterations < 1:
        raise ValueError("maximum number of iterations must be at least one")

    if alpha_max < alpha_min:
        tmp = alpha_max
        alpha_max = alpha_min
        alpha_min = tmp

    if sigma_2 < sigma_1:
        tmp = sigma_2
        sigma_2 = sigma_1
        sigma_1 = tmp

    xk = projector(x0)
    fk = f(xk)
    gk = df(xk)

    dinf = projector(xk - gk) - xk
    converged = np.sqrt(np.linalg.norm(dinf)) < tol

    alpha_k = 1.0 / np.max(np.abs(dinf))
    f_prev = fk * np.ones(m)

    iterations = 0
    while not converged and iterations < max_iterations:
        dk = projector(xk - alpha_k * gk) - xk
        dkgk = dk.dot(gk)
        lmda = 1
        backtracking = True
        while backtracking:
            xp = xk + lmda * dk
            fp = f(xp)
            backtracking = fp > np.max(f_prev + gamma * lmda * dkgk)
            # @todo implement quadratic interpolation
            lmda = 0.5 * (sigma_1 + sigma_2) * lmda

        f_prev[:-1] = f_prev[1:]
        f_prev[-1] = fp

        gkp1 = df(xp)
        sk = xp - xk
        yk = gkp1 - gk
        bk = np.dot(sk, yk)
        if bk <= 0:
            alpha_k = alpha_max
        else:
            ak = np.dot(sk, sk)
            alpha_k = min(alpha_max, max(alpha_min, ak / bk))

        xk = xp
        gk = gkp1

        dinf = projector(xk - gk) - xk
        converged = np.sqrt(np.linalg.norm(dinf)) < tol
        iterations += 1

    if iterations == max_iterations and not converged:
        return None
    else:
        return xk

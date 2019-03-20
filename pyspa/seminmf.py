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
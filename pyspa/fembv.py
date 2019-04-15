import numbers
import warnings

import numpy as np
import os
import pickle
import scipy.sparse as sps

from scipy.optimize import (Bounds, LinearConstraint, linprog, minimize)
from scipy.spatial import ConvexHull

from sklearn.cluster import KMeans
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

INTEGER_TYPES = (numbers.Integral, np.integer)

FEMBV_KMEANS_INITIALIZATION_METHODS = (None, 'random', 'kmeans')
FEMBV_BINX_INITIALIZATION_METHODS = (None, 'random')


class PiecewiseConstantFEMBasis(object):
    def __init__(self, width=1, value=1):
        self.width = width
        self.value = value

    def _width(self):
        width = int(np.floor(self.width))
        if width < 1:
            width == 1
        return width

    def _basis_func(self, i, grid_points):
        value = np.zeros(grid_points.shape)
        value[grid_points - i < self._width()] = self.value
        value[grid_points < i] = 0
        return value

    def __call__(self, n_grid_points):
        grid_points = np.arange(n_grid_points)
        width = self._width()

        i_max = int(np.floor((n_grid_points - 1) / width))
        left_end_points = width * np.arange(i_max + 1)

        n_elements = np.size(left_end_points)
        V = np.zeros((n_grid_points, n_elements))
        for i, v in enumerate(left_end_points):
            V[:, i] = self._basis_func(v, grid_points)

        return V


class TriangleFEMBasis(object):
    def __init__(self, n_points=3, value=1):
        self.n_points = n_points
        self.value = value

    def _basis_func(self, i, grid_points):
        half_width = int((self.n_points - 1) / 2)
        slope = self.value / half_width
        value = np.zeros(grid_points.shape)
        value[grid_points <= i] = (
            self.value + slope * (grid_points[grid_points <= i] - i))
        value[grid_points > i] = (
            self.value - slope * (grid_points[grid_points > i] - i))
        value[np.abs(grid_points - i) >= half_width] = 0
        return value

    def __call__(self, n_grid_points):
        if self.n_points % 2 == 0:
            raise ValueError('number of element points must be an odd number')

        half_width = int((self.n_points - 1) / 2)
        if n_grid_points < half_width:
            raise ValueError('too few grid points')

        grid_points = np.arange(n_grid_points)
        i_max = int(np.floor((n_grid_points - 1) / half_width))
        midpoints = half_width * np.arange(i_max + 1)
        n_elements = np.size(midpoints)
        V = np.zeros((n_grid_points, n_elements))
        for i, v in enumerate(midpoints):
            V[:, i] = self._basis_func(v, grid_points)

        return V


def _create_fembv_checkpoint(Gamma, Theta, checkpoint_file):
    """Serialize model components for checkpointing."""
    temp_file = checkpoint_file + '.old'
    if os.path.exists(checkpoint_file):
        os.rename(checkpoint_file, temp_file)

    data = {'Theta': Theta, 'Gamma': Gamma}
    with open(checkpoint_file, 'wb') as cpf:
        pickle.dump(data, cpf, pickle.HIGHEST_PROTOCOL)

    if os.path.exists(temp_file):
        os.remove(temp_file)


def _check_unit_axis_sums(A, whom, axis=0):
    axis_sums = np.sum(A, axis=axis)
    if not np.all(np.isclose(axis_sums, 1)):
        raise ValueError(
            'Array with incorrect axis sums passed to %s. '
            'Expected sums along axis %d to be 1.'
            % (whom, axis))


def _check_array_shape(A, shape, whom):
    if np.shape(A) != shape:
        raise ValueError(
            'Array with wrong shape passed to %s. '
            'Expected %s, but got %s' % (whom, shape, np.shape(A)))


def _check_init_fembv_Gamma(Gamma, shape, whom):
    Gamma = check_array(Gamma)
    _check_array_shape(Gamma, shape, whom)
    _check_unit_axis_sums(Gamma, whom, axis=1)


def _random_affiliations(shape, random_state=None):
    """Return random matrix with unit row sums."""
    rng = check_random_state(random_state)

    Gamma = rng.uniform(size=shape)
    row_sums = np.sum(Gamma, axis=1)
    Gamma = Gamma / row_sums[:, np.newaxis]

    return Gamma


def _fembv_generic_cost(X, Gamma, Theta, distance_matrix,
                        distance_matrix_pars=None,
                        epsilon_Theta=0, regularization_Theta=None):
    G = distance_matrix(X, Theta, distance_matrix_pars)
    cost = np.trace(np.dot(np.transpose(Gamma), G))
    if regularization_Theta is not None and epsilon_Theta != 0:
        cost += epsilon_Theta * regularization_Theta(Theta)
    return cost


def _fembv_convergence_check(old_cost, new_cost, tol):
    """Return True if change in cost function is below desired tolerance."""
    delta_cost = np.abs(old_cost - new_cost)

    if np.abs(old_cost) > np.abs(new_cost):
        min_cost = new_cost
        max_cost = old_cost
    else:
        min_cost = old_cost
        max_cost = new_cost

    rel_cost = 1 - np.abs(min_cost / max_cost)

    return delta_cost < tol or rel_cost < tol


def _fembv_Gamma_equality_constraints(n_components, V):
    """Return vector form of equality constraints for affiliations."""
    n_samples, n_elem = V.shape
    n_coeffs = n_components * n_elem

    A_eq = sps.lil_matrix((n_samples, 2 * n_coeffs))
    A_eq[:, :n_coeffs] = np.repeat(V, n_components, axis=1)

    b_eq = np.ones((n_samples,))

    A_csr = A_eq.tocsr()
    A_csr.eliminate_zeros()

    return A_csr, b_eq


def _fembv_Gamma_upper_bound_constraints(n_components, V, max_tv_norm=None):
    """Return vector form of upper bound constraints for affiliations."""
    n_samples, n_elem = V.shape
    n_gamma_points = n_components * n_samples
    n_eta_points = n_components * (n_samples - 1)
    n_coeffs = n_components * n_elem

    DV = np.diff(V, axis=0)

    V_expanded = sps.lil_matrix((n_gamma_points, n_coeffs))
    DV_expanded = sps.lil_matrix((n_eta_points, n_coeffs))
    for i in range(n_samples):
        for k in range(n_components):
            row = k + i * n_components
            V_expanded[row, k:k + n_coeffs:n_components] = V[i, :]
            if i != n_samples - 1:
                DV_expanded[row, k:k + n_coeffs:n_components] = DV[i, :]

    A_pos = sps.bmat([[-V_expanded, None],
                      [None, -V_expanded[:-n_components]]])
    b_pos = np.zeros(n_gamma_points + n_eta_points)

    A_aux = sps.bmat([[DV_expanded, -V_expanded[:-n_components]],
                      [-DV_expanded, -V_expanded[:-n_components]]])
    b_aux = np.zeros(2 * n_eta_points)

    if max_tv_norm is None:
        A_ub = sps.vstack([A_pos, A_aux])
        b_ub = np.concatenate([b_pos, b_aux])
        return A_ub, b_ub

    A_bv = sps.lil_matrix((n_components, 2 * n_coeffs))
    for i in range(n_components):
        for k in range(n_elem):
            idx = n_coeffs + i + k * n_components
            A_bv[i, idx] = np.sum(V[:-1, k])

    if np.isscalar(max_tv_norm):
        b_bv = np.full(n_components, max_tv_norm)
    else:
        b_bv = max_tv_norm

    A_ub = sps.vstack([A_pos, A_aux, A_bv])
    b_ub = np.concatenate([b_pos, b_aux, b_bv])

    A_csr = A_ub.tocsr()
    A_csr.eliminate_zeros()

    return A_csr, b_ub


def _subspace_update_fembv_Gamma(G, basis_values, A_ub, b_ub, A_eq, b_eq,
                                 tol=1e-5, max_iter=500, verbose=0,
                                 bounds=(None, None),
                                 method='interior-point'):
    """Update affiliations in FEM-BV using linear programming.

    Parameters
    ----------
    G : array-like, shape (n_samples, n_components)
        Matrix of model distances.


    max_iter : integer, default: 500
        Maximum number of iterations before stopping.

    verbose : integer, default: 0
        The verbosity level.

    bounds : sequence, optional
        If given, additional lower and upper bounds to place on
        finite-element coefficients of the solution.

    method : None | 'interioir-point' : 'simplex'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls back to 'interior-point'

        - 'interior-point': uses interior point method

        - 'simplex': uses simplex method

    Returns
    -------
    Gamma : array-like, shape (n_samples, n_components)
        Solution for affiliations in FEM-BV discretization.

    Eta : array-like, shape (n_samples - 1, n_components)
        Auxiliary variables used to enforce bounded variation constraint.
    """
    n_samples, n_components = G.shape
    n_elem = basis_values.shape[1]
    n_coeffs = n_components * n_elem

    vtg = np.dot(np.transpose(basis_values), G)
    g_vec = np.concatenate([np.ravel(vtg), np.zeros(n_coeffs)])

    if sps.issparse(A_eq) or sps.issparse(A_ub):
        is_sparse = True
    else:
        is_sparse = False
    options = {'disp': verbose > 0, 'maxiter': max_iter,
               'tol': tol, 'sparse': is_sparse, 'presolve': True}
    res = linprog(g_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method=method, options=options)

    if not res['success']:
        raise RuntimeError('failed to solve linear optimization problem')

    sol = res['x']

    alpha = np.reshape(sol[:n_coeffs], (n_elem, n_components))
    beta = np.reshape(sol[n_coeffs:], (n_elem, n_components))

    Gamma = np.dot(basis_values, alpha)
    Eta = np.dot(basis_values[:n_samples, ...], beta)

    return Gamma, Eta


def _fit_generic_fembv_subspace(X, Gamma, Theta, distance_matrix, theta_update,
                                distance_matrix_pars=None,
                                theta_update_pars=None,
                                max_tv_norm=None,
                                epsilon_Theta=0, regularization_Theta=None,
                                tol=1e-4, max_iter=200, fem_basis='constant',
                                method='interior-point',
                                update_Theta=True, verbose=0,
                                checkpoint=False, checkpoint_file=None,
                                checkpoint_iter=None):
    """Compute FEM-BV fit using subspace algorithm.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix to be fitted.

    Gamma : array-like, shape (n_samples, n_components)
        If init='custom', used as initial guess for the solution.

    Theta :
        If init='custom', used as initial guess for the solution.
        If update_Theta=False, used as a constant to solve for Gamma only.

    distance_matrix : callable
        Function callable with the signature distance_matrix(X, Theta, pars)
        where pars is a tuple of additional parameters. Should return
        an array of shape (n_samples, n_components) containing the local
        model distances at each data point.

    theta_update : callable
        Function callable with the signature
        theta_update(X, Gamma, Theta, pars). Should return an updated
        guess for the optimal parameters Theta.

    distance_matrix_pars : optional
        Additional parameters to be passed as the last argument to the
        distance matrix calculation.

    theta_update_pars : optional
        Additional parameters to be passed as the last argument to the
        update method for the parameters.

    max_tv_norm : None, scalar or array-like, shape (n_components,)
        If a scalar, a common maximum TV norm for all cluster
        affiliations. If array-like, the maximum TV norm for
        each of the separate affiliation sequences. If not given,
        no upper bound is imposed.

    epsilon_Theta : float, default: 0
        Regularization parameter for model parameters.

    regularization_Theta : optional, callable
        If given, must be callable with the signature
        regularization_Theta(Theta) and return a scalar value corresponding
        to a penalty on the parameters.

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations before stopping.

    fem_basis : None, string, or callable
        The choice of finite-element basis functions to use. If given
        as a string, valid options are:

        - 'constant' : use piecewise constant basis functions

        - 'triangle' : use triangle basis functions with a width of 3
            grid points

        If given as a callable object, must be callable with the
        signature fem_basis(n_samples), and should return an
        array of shape (n_samples, n_elements) where n_elements is
        the number of finite-element basis functions, containing the
        values of the basis functions at each grid point. If None,
        defaults to piecewise constant basis functions.

    method : None | 'interior-point' | 'simplex'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls back to 'interior-point'

        - 'interior-point': uses interior point method

        - 'simplex': uses simplex method

    update_Theta : boolean, default: True
        If True, both Gamma and Theta will be estimated from initial guesses.
        If False, only Gamma will be computed.

    verbose : integer, default: 0
        The verbosity level.

    checkpoint : boolean, default: False
        If True, write current solution to specified checkpoint file.

    checkpoint_file : string or None, default: None
        If `checkpoint` is True, filename for the checkpoint file.

    checkpoint_iter : integer or None, default: None
        If `checkpoint` is True, number of iterations to checkpoint after.

    Returns
    -------
    Gamma : array-like, shape (n_samples, n_components)
        Solution for affiliations in FEM-BV discretization.

    Theta : array-like, shape (n_parameters, n_components)
        Solution for local model parameters in FEM-BV discretization.

    n_iter : integer
        The number of iterations done in the algorithm.
    """
    if fem_basis is None or fem_basis == 'constant':
        fem_basis = PiecewiseConstantFEMBasis()
    elif fem_basis == 'triangle':
        fem_basis = TriangleFEMBasis()
    elif not callable(fem_basis):
        raise ValueError('finite-element basis must be callable')

    n_samples, n_components = Gamma.shape
    V = fem_basis(n_samples)

    A_eq, b_eq = _fembv_Gamma_equality_constraints(n_components, V)
    A_ub, b_ub = _fembv_Gamma_upper_bound_constraints(
        n_components, V, max_tv_norm=max_tv_norm)

    for n_iter in range(max_iter):
        initial_cost = _fembv_generic_cost(
            X, Gamma, Theta, distance_matrix,
            distance_matrix_pars=distance_matrix_pars,
            epsilon_Theta=epsilon_Theta,
            regularization_Theta=regularization_Theta)

        if n_iter == 0:
            start_cost = initial_cost

        if update_Theta:
            Theta = theta_update(X, Gamma, Theta, *theta_update_pars)

        G = distance_matrix(X, Theta, distance_matrix_pars)
        Gamma, _ = _subspace_update_fembv_Gamma(
            G, V, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            max_iter=max_iter, method=method, verbose=verbose)

        final_cost = _fembv_generic_cost(
            X, Gamma, Theta, distance_matrix,
            distance_matrix_pars=distance_matrix_pars,
            epsilon_Theta=epsilon_Theta,
            regularization_Theta=regularization_Theta)

        if verbose:
            print('Iteration %d:' % (n_iter + 1))
            print('--------------')
            print('Cost function = %.5e' % final_cost)
            print('Cost delta = %.5e' % (final_cost - initial_cost))
            print('Cost ratio = %.5e' % (final_cost / start_cost))

        if checkpoint and n_iter % checkpoint_iter == 0:
            _create_fembv_checkpoint(Gamma, Theta, checkpoint_file)

        converged = _fembv_convergence_check(initial_cost, final_cost, tol)

        if converged:
            if verbose:
                print('*** Converged at iteration %d ***' % (n_iter + 1))
            break

    return Gamma, Theta, n_iter


def _check_init_fembv_kmeans_Theta(Theta, shape, whom):
    Theta = check_array(Theta)
    _check_array_shape(Theta, shape, whom)


def _initialize_fembv_kmeans_random(X, n_components, random_state=None):
    """Return random initial affiliations and cluster parameters.

    The affiliations matrix is first initialized with uniformly distributed
    random numbers on [0, 1), before the normalization requirement is imposed.
    The cluster parameters (centroids) are chosen as random rows of the
    data matrix X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be fitted

    n_components : integer
        The number of clusters desired

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    Returns
    -------
    Gamma : array-like, shape (n_samples, n_components)
        Random initial guess for affiliations

    Theta : array-like, shape (n_components, n_features)
        Random initial guess for cluster centroids
    """
    n_samples, n_features = X.shape
    rng = check_random_state(random_state)

    rows = rng.choice(n_samples, n_components, replace=False)
    Theta = X[rows]

    Gamma = _random_affiliations((n_samples, n_components), random_state=rng)

    return Gamma, Theta


def _initialize_fembv_kmeans_kmeans(X, n_components, random_state=None):
    """Return initial affiliations and cluster parameters from k-means clustering.

    An initial k-means clustering is performed on the data. The initial
    affiliations are set to the obtained cluster affiliations, and the
    cluster parameters to the calculated centroids.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be fitted

    n_components : integer
        The number of clusters desired

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    Returns
    -------
    Gamma : array-like, shape (n_samples, n_components)
        Initial guess for affiliations from k-means affiliations

    Theta : array-like, shape (n_components, n_features)
        Initial guess for cluster centroids from k-means centroids
    """
    kmeans = KMeans(n_clusters=n_components, random_state=random_state).fit(X)
    Theta = kmeans.cluster_centers_

    labels = kmeans.labels_
    n_samples = X.shape[0]
    Gamma = np.zeros((n_samples, n_components))
    for i in range(n_samples):
        Gamma[i, labels[i]] = 1

    return Gamma, Theta


def _initialize_fembv_kmeans(X, n_components, init='random',
                             random_state=None):
    """Return initial guesses for affiliations and cluster parameters.

    Initial values for the affiliations Gamma and cluster parameters Theta
    are calculated using the given initialization method.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be fitted.

    n_components : integer
        The number of clusters.

    init : None | 'random' | 'kmeans'
        Method used for initialization.
        Default: 'random'
        Valid options:

        - None: falls back to 'random'.

        - 'random': random initialization of the cluster affiliations
            and centroids.

        - 'kmeans': perform an initial k-means clustering of the data

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the random number
        generator. If a RandomState instance, random_state is the
        random number generator. If None, the random number generator is
        the RandomState instance used by `np.random`.

    Returns
    -------
    Gamma : array-like, shape (n_samples, n_components)
        Initial guess for affiliations

    Theta : array-like, shape (n_components, n_features)
        Initial guess for cluster parameters
    """
    if init is None:
        init = 'random'

    if init == 'random':
        return _initialize_fembv_kmeans_random(
            X, n_components, random_state=random_state)
    elif init == 'kmeans':
        return _initialize_fembv_kmeans_kmeans(
            X, n_components, random_state=random_state)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, FEMBV_KMEANS_INITIALIZATION_METHODS))


def _fembv_kmeans_Theta_update(X, Gamma, Theta, *pars):
    Theta = np.dot(np.transpose(Gamma), X)
    normalizations = np.sum(Gamma, axis=0)
    return Theta / normalizations[:, np.newaxis]


def _fembv_kmeans_distance_matrix(X, Theta, *pars):
    n_samples, n_features = X.shape
    n_components = Theta.shape[0]

    G = np.zeros((n_samples, n_components))
    for i in range(n_samples):
        for j in range(n_components):
            G[i, j] = np.linalg.norm(X[i] - Theta[j]) ** 2

    return G


def _fembv_kmeans_cost(X, Gamma, Theta):
    return _fembv_generic_cost(X, Gamma, Theta, _fembv_kmeans_distance_matrix)


def fembv_kmeans(X, Gamma=None, Theta=None, n_components=None,
                 init=None, update_Theta=True, solver='subspace',
                 tol=1e-4, max_iter=200, random_state=None, max_tv_norm=None,
                 fem_basis='constant', method='interior-point',
                 verbose=0, checkpoint=False,
                 checkpoint_file=None, checkpoint_iter=None):
    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    if not isinstance(n_components, INTEGER_TYPES) or n_components <= 0:
        raise ValueError('Number of components must be a positive integer;'
                         ' got (n_components=%r)' % n_components)
    if not isinstance(max_iter, INTEGER_TYPES) or max_iter <= 0:
        raise ValueError('Maximum number of iterations must be a positive '
                         'integer; got (max_iter=%r)' % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError('Tolerance for stopping criteria must be '
                         'positive; got (tol=%r)' % tol)

    if checkpoint:
        if checkpoint_file is None:
            raise ValueError('Name of checkpoint file must be given if '
                             'checkpointing is enabled.')

        if (not isinstance(checkpoint_iter, INTEGER_TYPES) or
            checkpoint_iter <= 0):
            raise ValueError(
                'Number of iterations to checkpoint after must be a '
                'positive integer; got (checkpoint_iter=%r)' % checkpoint_iter)

    if init == 'custom' and update_Theta:
        _check_init_fembv_Gamma(Gamma, (n_samples, n_components),
                                'FEM-BV-KMeans (input Gamma)')
        _check_init_fembv_kmeans_Theta(
            Theta, (n_components, n_features), 'FEM-BV-KMeans (input Theta)')
    elif not update_Theta:
        _check_init_fembv_kmeans_Theta(Theta, (n_components, n_features),
                                       'FEM-BV-KMeans (input Theta)')
        Gamma, _ = _initialize_fembv_kmeans(X, n_components, init=init,
                                            random_state=random_state)
    else:
        Gamma, Theta = _initialize_fembv_kmeans(X, n_components, init=init,
                                                random_state=random_state)

    if solver == 'subspace':
        Gamma, Theta, n_iter = _fit_generic_fembv_subspace(
            X, Gamma, Theta, _fembv_kmeans_distance_matrix,
            _fembv_kmeans_Theta_update,
            theta_update_pars=None, distance_matrix_pars=None,
            epsilon_Theta=0, regularization_Theta=None,
            tol=tol, max_iter=max_iter, update_Theta=update_Theta,
            max_tv_norm=max_tv_norm, fem_basis=fem_basis,
            method=method,
            verbose=verbose, checkpoint=checkpoint,
            checkpoint_file=checkpoint_file,
            checkpoint_iter=checkpoint_iter)
    else:
        raise ValueError("Invalid solver parameter '%s'." % solver)

    if n_iter == max_iter and tol > 0:
        warnings.warn('Maximum number of iterations %d reached.' % max_iter,
                      warnings.UserWarning)

    return Gamma, Theta, n_iter


def _fembv_binx_lambda_vector(i, Theta, u=None):
    if u is None:
        return Theta[i, :]

    if u.ndim == 1:
        n_samples = 1
        n_external = u.shape[0]
        u_mat = np.reshape(u, (n_samples, n_external))
    else:
        n_samples, n_external = u.shape
        u_mat = u

    n_features = int(Theta.shape[1] / (n_external + 1))

    lam = np.broadcast_to(
        Theta[i, :n_features], (n_samples, n_features)).copy()

    for k in range(n_external):
        lk = Theta[i, (k + 1) * n_features:(k + 2) * n_features]
        lam += lk[np.newaxis, :] * np.broadcast_to(
            np.reshape(u_mat[:, k], (n_samples, 1)), (n_samples, n_features))

    if n_samples == 1:
        return np.ravel(lam)
    else:
        return lam


def _check_init_fembv_binx_Theta(Theta, shape, whom, u=None):
    Theta = check_array(Theta)
    _check_array_shape(Theta, shape, whom)

    if u is not None:
        n_components = Theta.shape[0]
        for i in range(n_components):
            lx = _fembv_binx_lambda_vector(i, Theta, u=u)
            row_sums = np.sum(lx, axis=1)
            if np.any(np.logical_or(row_sums < 0, row_sums > 1)):
                warnings.warn(
                    'sum of transition probabilities passed to %s '
                    'is not between 0 and 1 for cluster %d' %
                    (whom, i), warnings.UserWarning)
            if np.any(np.logical_or(lx < 0, lx > 1)):
                warnings.warn(
                    'transition probabilities passed to %s '
                    'do not lie between 0 and 1 for cluster %d' %
                    (whom, i), warnings.UserWarning)


def _initialize_fembv_binx_random(YX, n_components, u=None, random_state=None):
    """Return random initial affiliations and cluster parameters.

    The affiliations matrix is first initialized with uniformly distributed
    random numbers on [0, 1), before the normalization requirement is imposed.

    If no external factors are given, the cluster parameters Theta are
    chosen to be uniformly distributed numbers on [0, 1) subject to the
    constraint that each row sums to unity.

    Parameters
    ----------
    YX : array-like, shape (n_samples, n_features + 1)
        The merged data matrix to be fitted

    n_components : integer
        The number of clusters desired

    u : optional, array-like, shape (n_samples, n_external)
        External factors to be used in fitting the data.

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    Returns
    -------
    Gamma : array-like, shape (n_samples, n_components)
        Random initial guess for affiliations

    Theta : array-like, shape (n_components, n_features)
        Random initial guess for cluster centroids
    """
    n_samples = YX.shape[0]
    rng = check_random_state(random_state)

    Gamma = _random_affiliations((n_samples, n_components), random_state=rng)

    n_features = YX.shape[1] - 1
    if u is None:
        Theta = _random_affiliations(
            (n_components, n_features), random_state=rng)
    else:
        # @todo ensure initial guess is in feasible region
        if u.ndim == 1:
            n_external = 1
        else:
            n_external = u.shape[1]
        n_component_pars = n_features * (n_external + 1)
        Theta = _random_affiliations(
            (n_components, n_component_pars), random_state=rng)

        # if u.ndim == 1:
        #     n_external = 1
        #     vertices = np.array([[np.min(u)], [np.max(u)]])
        # else:
        #     n_external = u.shape[1]
        #     hull = ConvexHull(u)
        #     vertices = u[hull.vertices]

        # theta_0  = _random_affiliations(
        #     (n_components, n_features), random_state=rng)

    return Gamma, Theta


def _initialize_fembv_binx(YX, n_components, init='random',
                           u=None, random_state=None):
    """Return initial guesses for affiliations and cluster parameters.

    Initial values for the affiliations Gamma and cluster parameters Theta
    are calculated using the given initialization method.

    Note that the provided data matrix is assumed to contain as its first
    column the binary outcome variable,  YX = [Y, X]

    Parameters
    ----------
    YX : array-like, shape (n_samples, n_features + 1)
        The merged data matrix to be fitted.

    n_components : integer
        The number of clusters.

    init : None | 'random'
        Method used for initialization.
        Default: 'random'
        Valid options:

        - None: falls back to 'random'.

        - 'random': random initialization of the cluster affiliations
            and centroids.

    u : optional, array-like, shape (n_samples, n_external)
        External factors to be used in fitting the data.

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the random number
        generator. If a RandomState instance, random_state is the
        random number generator. If None, the random number generator is
        the RandomState instance used by `np.random`.

    Returns
    -------
    Gamma : array-like, shape (n_samples, n_components)
        Initial guess for affiliations

    Theta : array-like, shape (n_components, n_features)
        Initial guess for cluster parameters
    """
    if init is None:
        init = 'random'

    if init == 'random':
        return _initialize_fembv_binx_random(
            YX, n_components, u=u, random_state=random_state)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, FEMBV_BINX_INITIALIZATION_METHODS))


def _fembv_binx_distance_matrix(YX, Theta, u=None):
    Y = YX[:, 0]
    X = YX[:, 1:]

    n_samples, n_features = X.shape
    n_components = Theta.shape[0]

    G = np.zeros((n_samples, n_components))
    for i in range(n_samples):
        for j in range(n_components):
            if u is None:
                lam = _fembv_binx_lambda_vector(j, Theta)
            else:
                lam = _fembv_binx_lambda_vector(j, Theta, u=u[i, :])
            lxp = np.dot(lam, X[i, :])
            G[i, j] = -((1 - Y[i]) * np.log(1 - lxp) + Y[i] * np.log(lxp))

    return G


def _fembv_binx_Theta_regularization(Theta, u=None, epsilon_Theta=0, *pars):
    if u is None:
        return np.sum(Theta)
    else:
        return np.sum(Theta ** 2)


def _fembv_binx_distance_grad(YX, Gamma, Theta, u=None):
    Y = YX[:, 0]
    X = YX[:, 1:]

    n_samples, n_features = X.shape
    n_components = Theta.shape[0]

    if u is None:
        n_external = 0
    else:
        n_external = u.shape[1]

    n_pars = n_features * (n_external + 1)

    yp = Y[:, np.newaxis] * X
    yp_comp = (1 - Y[:, np.newaxis]) * X
    DG = np.zeros((n_components, n_pars))
    for i in range(n_components):
        gyp = Gamma[:, i][:, np.newaxis] * yp
        gyp_comp = Gamma[:, i][:, np.newaxis] * yp_comp
        if u is None:
            ldotp = np.sum(Theta[i, :][np.newaxis, :] * X, axis=1)
            for j in range(n_pars):
                DG[i, j] = -np.sum(
                    np.divide(gyp[:, j], ldotp) -
                    np.divide(gyp_comp[:, j], 1 - ldotp))
        else:
            lx = _fembv_binx_lambda_vector(i, Theta, u=u)
            lxdotp = np.sum(lx * X, axis=1)
            for j in range(n_features):
                DG[i, j] = -np.sum(
                    np.divide(gyp[:, j], lxdotp) -
                    np.divide(gyp_comp[:, j], 1 - lxdotp))
            for k in range(n_external):
                for j in range(n_features):
                    par_idx = j + (k + 1) * n_features
                    DG[i, par_idx] = np.sum(
                        np.divide(gyp[:, j] * u[:, k], lxdotp) -
                        np.divide(gyp_comp[:, j] * u[:, k], 1 - lxdotp))

    return DG


def _fembv_binx_regularization_grad(Theta, u=None, epsilon_Theta=0):
    return epsilon_Theta * np.ones(Theta.shape)


def _fembv_binx_cost_grad(YX, Gamma, Theta, u=None, epsilon_Theta=0):
    return (_fembv_binx_distance_grad(YX, Gamma, Theta, u=u) +
            _fembv_binx_regularization_grad(
                Theta, u=u, epsilon_Theta=epsilon_Theta))


def _fembv_binx_distance_hess(YX, Gamma, Theta, u=None):
    Y = YX[:, 0]
    X = YX[:, 1:]

    n_samples, n_features = X.shape
    n_components = Theta.shape[0]

    if u is None:
        n_external = 0
    else:
        n_external = u.shape[1]

    n_component_pars = n_features * (n_external + 1)
    n_pars = n_components * n_component_pars

    yp = Y[:, np.newaxis] * X
    yp_comp = (1 - Y[:, np.newaxis]) * X
    H = np.zeros((n_pars, n_pars))

    def hp(i, n_features, u=None):
        if u is None or i == 0:
            return 1
        else:
            return u[:, p]

    yp = Y[:, np.newaxis] * X
    yp_comp = (1 - Y[:, np.newaxis]) * X
    for i in range(n_components):
        lx = _fembv_binx_lambda_vector(i, Theta, u=u)
        lxdotp = np.sum(lx * X, axis=1)
        for j in range(n_component_pars):
            for k in range(j, n_component_pars):
                p = int(j / n_features)
                q = int(k / n_features)
                r = j - p * n_features
                s = k - q * n_features

                row_idx = j + i * n_component_pars
                col_idx = k + i * n_component_pars

                coeff = hp(p, n_features, u=u) * hp(q, n_features, u=u)
                ypp = yp[:, r] * X[:, s]
                ypp_comp = yp_comp[:, r] * X[:, s]
                gypp = Gamma[:, i] * ypp
                gypp_comp = Gamma[:, i] * ypp_comp
                val = np.sum(
                    np.divide(coeff * gypp, lxdotp ** 2) +
                    np.divide(coeff * gypp_comp, (1 - lxdotp) ** 2))

                H[row_idx, col_idx] = val
                if j != k:
                    H[col_idx, row_idx] = val

    return H


def _fembv_binx_regularization_hess(Theta, u=None, epsilon_Theta=0):
    n_pars = np.size(Theta)
    if u is None:
        return np.zeros((n_pars, n_pars))
    else:
        return 2 * epsilon_Theta * np.identity(n_pars)


def _fembv_binx_cost_hess(YX, Gamma, Theta, u=None, epsilon_Theta=0):
    return (_fembv_binx_distance_hess(YX, Gamma, Theta, u=u) +
            _fembv_binx_regularization_hess(
                Theta, epsilon_Theta=epsilon_Theta))


def _fembv_binx_cost(YX, Gamma, Theta, u=None, epsilon_Theta=0):
    return _fembv_generic_cost(
        YX, Gamma, Theta, _fembv_binx_distance_matrix,
        distance_matrix_pars=u, epsilon_Theta=epsilon_Theta,
        regularization_Theta=_fembv_binx_Theta_regularization)


def _fembv_binx_Theta_bounds(n_components, n_features, u=None):
    if u is None:
        return Bounds(0, 1)

    n_external = u.shape[1]
    n_component_pars = n_features * (n_external + 1)
    n_pars = n_components * n_component_pars
    lb = np.full(n_pars, -np.inf)
    ub = np.full(n_pars, np.inf)
    for i in range(n_components):
        lb[i * n_component_pars:i * n_component_pars + n_features] = 0
        ub[i * n_component_pars:i * n_component_pars + n_features] = 1
    return Bounds(lb, ub)


def _fembv_binx_Theta_constraints(n_components, n_features, u=None):
    if u is None:
        n_pars = n_components * n_features
        A_constr = np.zeros((n_components, n_pars))
        for i in range(n_components):
            A_constr[i, i * n_features:(i + 1) * n_features] = 1
        return LinearConstraint(A_constr, 0, 1)

    n_samples, n_external = u.shape
    n_component_pars = n_features * (n_external + 1)
    n_pars = n_components * n_component_pars

    if n_samples > 2:
        convex_hull = ConvexHull(u)
        vertices = u[convex_hull.vertices]
    else:
        vertices = u
    n_vertices = vertices.shape[0]
    n_elem_constraints = n_components * n_features * n_vertices
    A_elem = np.zeros((n_elem_constraints, n_pars))
    for k in range(n_vertices):
        vertex = vertices[k]
        for i in range(n_components):
            base_offset = i * n_component_pars
            ext_start = base_offset + n_features
            ext_end = ext_start + n_external * n_features
            for j in range(n_features):
                idx = j + i * n_features + k * n_components * n_features
                A_elem[idx, j + base_offset] = 1
                A_elem[idx, j + ext_start:j + ext_end:n_features] = vertex

    n_sum_constraints = n_components * (n_vertices + 1)
    A_sum = np.zeros((n_sum_constraints, n_pars))

    for i in range(n_components):
        A_sum[i, i * n_component_pars:i * n_component_pars + n_features] = 1

    for k in range(n_vertices):
        vertex = vertices[k]
        for i in range(n_components):
            idx = i + (k + 1) * n_components
            base_offset = i * n_component_pars
            ext_start = base_offset + n_features
            A_sum[idx, i * n_component_pars:
                  i * n_component_pars + n_features] = 1
            for j in range(n_external):
                A_sum[idx, j * n_features + ext_start:
                      ext_start + (j + 1) * n_features] = vertex[j]

    A_bounds = np.vstack([A_elem, A_sum])

    return LinearConstraint(A_bounds, 0, 1)


def _fembv_binx_Theta_update(YX, Gamma, Theta, *pars, **kwargs):
    u = pars[0]
    epsilon_Theta = pars[1]
    bounds = pars[2]
    constraints = pars[3]
    verbose = pars[4]

    n_components, n_pars = Theta.shape

    x0 = np.ravel(Theta)

    args = (YX, Gamma, u, epsilon_Theta)

    def f(x, *args):
        yx = args[0]
        gamma = args[1]
        u = args[2]
        epsilon_Theta = args[3]
        theta = np.reshape(x, (n_components, n_pars))
        return _fembv_binx_cost(
            yx, gamma, theta, u=u, epsilon_Theta=epsilon_Theta)

    def jac(x, *args):
        yx = args[0]
        gamma = args[1]
        u = args[2]
        epsilon_Theta = args[3]
        theta = np.reshape(x, (n_components, n_pars))
        jac_mat = _fembv_binx_cost_grad(
            yx, gamma, theta, u=u, epsilon_Theta=epsilon_Theta)
        return np.ravel(jac_mat)

    def hess(x, *args):
        yx = args[0]
        gamma = args[1]
        u = args[2]
        epsilon_Theta = args[3]
        theta = np.reshape(x, (n_components, n_pars))
        hess_mat = _fembv_binx_cost_hess(
            yx, gamma, theta, u=u, epsilon_Theta=epsilon_Theta)
        return hess_mat

    if 'method' in kwargs:
        method = kwargs['method']
    else:
        method = 'trust-krylov'

    res = minimize(f, x0, args=args, jac=jac, hess=hess,
                   bounds=bounds, constraints=constraints,
                   method=method, options={'disp': verbose > 0},
                   **kwargs)

    if not res['success']:
        raise RuntimeError('minimization of FEM-BV-BINX cost function failed')

    sol = res['x']

    return np.reshape(sol, (n_components, n_pars))


def fembv_binx(X, Y, Gamma=None, Theta=None, u=None, n_components=None,
               init=None, update_Theta=True, epsilon_Theta=0,
               solver='subspace', tol=1e-4, max_iter=200, random_state=None,
               max_tv_norm=None, fem_basis='constant',
               method='interior-point', verbose=0,
               checkpoint=False, checkpoint_file=None, checkpoint_iter=None):
    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    if u is None:
        n_component_pars = n_features
    else:
        if u.ndim == 1:
            n_component_pars = 2 * n_features
        else:
            n_component_pars = n_features * (1 + u.shape[1])

    if not isinstance(n_components, INTEGER_TYPES) or n_components <= 0:
        raise ValueError('Number of components must be a positive integer;'
                         ' got (n_components=%r)' % n_components)
    if not isinstance(max_iter, INTEGER_TYPES) or max_iter <= 0:
        raise ValueError('Maximum number of iterations must be a positive '
                         'integer; got (max_iter=%r)' % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError('Tolerance for stopping criteria must be '
                         'positive; got (tol=%r)' % tol)

    if checkpoint:
        if checkpoint_file is None:
            raise ValueError('Name of checkpoint file must be given if '
                             'checkpointing is enabled.')

        if (not isinstance(checkpoint_iter, INTEGER_TYPES) or
            checkpoint_iter <= 0):
            raise ValueError(
                'Number of iterations to checkpoint after must be a '
                'positive integer; got (checkpoint_iter=%r)' % checkpoint_iter)

    invalid_y_vals = np.any(np.logical_and(Y != 0, Y != 1))
    if invalid_y_vals:
        raise ValueError(
            'data matrix Y must be a binary variable (values 0 or 1)')

    if Y.ndim == 1:
        yx = np.concatenate([np.reshape(Y, (n_samples, 1)), X], axis=-1)
    else:
        yx = np.hstack([Y, X])

    theta_update_bounds = _fembv_binx_Theta_bounds(
        n_components, n_features, u=u)
    theta_update_constraints = _fembv_binx_Theta_constraints(
        n_components, n_features, u=u)
    theta_update_pars = (u, epsilon_Theta,
                         theta_update_bounds, theta_update_constraints,
                         verbose)

    if init == 'custom' and update_Theta:
        _check_init_fembv_Gamma(Gamma, (n_samples, n_components),
                                'FEM-BV-BINX (input Gamma)')
        _check_init_fembv_binx_Theta(
            Theta, (n_components, n_component_pars),
            'FEM-BV-BINX (input Theta)', u=u)
    elif not update_Theta:
        _check_init_fembv_binx_Theta(
            Theta, (n_components, n_component_pars),
            'FEM-BV-BINX (input Theta)', u=u)
        Gamma, _ = _initialize_fembv_binx(yx, n_components, init=init,
                                          u=u, random_state=random_state)
    else:
        Gamma, Theta = _initialize_fembv_binx(yx, n_components, init=init,
                                              u=u, random_state=random_state)

    if solver == 'subspace':
        Gamma, Theta, n_iter = _fit_generic_fembv_subspace(
            yx, Gamma, Theta, _fembv_binx_distance_matrix,
            _fembv_binx_Theta_update,
            theta_update_pars=theta_update_pars, distance_matrix_pars=u,
            epsilon_Theta=epsilon_Theta,
            regularization_Theta=_fembv_binx_Theta_regularization,
            tol=tol, max_iter=max_iter, update_Theta=update_Theta,
            max_tv_norm=max_tv_norm, fem_basis=fem_basis,
            method=method,
            verbose=verbose, checkpoint=checkpoint,
            checkpoint_file=checkpoint_file,
            checkpoint_iter=checkpoint_iter)
    else:
        raise ValueError("Invalid solver parameter '%s'." % solver)

    if n_iter == max_iter and tol > 0:
        warnings.warn('Maximum number of iterations %d reached.' % max_iter,
                      warnings.UserWarning)

    return Gamma, Theta, n_iter


class FEMBVKMeans(object):
    r"""FEM-BV clustering with k-means distance function.

    The local distance function is::

        || x_t - \theta_i||_2^2

    where x_t is row t of the data matrix, and \theta_t is the centroid
    of cluster i.

    The total objective function is minimized with an alternating
    minimization with respect to the cluster centroids \theta_i and
    the affiliations Gamma.

    Parameters
    ----------
    n_components : integer or None
        If an integer, the number of clusters. If None, then all features
        are kept.

    init : None | 'random' | 'kmeans' | 'custom'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls back to 'random'

        - 'random': random initialization of the cluster affiliations
            and centroids.

        - 'kmeans': perform an initial k-means clustering of the data

        - 'custom': use custom matrices for Gamma and Theta

    solver : 'subspace'
        Numerical solver to use:
        'subspace' is the subspace algorithm.

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations before stopping.

    random_state : integer, RandomState, or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If
        None, the random number generator is the
        RandomState instance used by `np.random`.

    max_tv_norm : None, scalar or array-like, shape (n_components,)
        If a scalar, a common maximum TV norm for all cluster
        affiliations. If array-like, the maximum TV norm for
        each of the separate affiliation sequences. If not given,
        no upper bound is imposed.

    verbose : integer, default: 0
        The verbosity level.

    fem_basis : None, string, or callable
        The choice of finite-element basis functions to use. If given
        as a string, valid options are:

        - 'constant' : use piecewise constant basis functions

        - 'triangle' : use triangle basis functions with a width of 3
            grid points

        If given as a callable object, must be callable with the
        signature fem_basis(n_samples), and should return an
        array of shape (n_samples, n_elements) where n_elements is
        the number of finite-element basis functions, containing the
        values of the basis functions at each grid point. If None,
        defaults to piecewise constant basis functions.

    Attributes
    ----------
    components_ : array-like, shape (n_components, n_features)
        The cluster parameters.

    cost_ : number
        Value of the FEM-BV cost function for the obtained fit.

    n_iter_ : integer
        Actual number of iterations.

    Examples
    --------
    import numpy as np
    X = np.random.rand(100, 4)
    from pyspa.fembv import FEMBVKMeans
    model = FEMBVKMeans(n_components=2, init='random', random_state=0)
    Gamma = model.fit_transform(X)
    Theta = model.components_

    References
    ----------
    P. Metzner, L. Putzig, and I. Horenko, "Analysis of Persistent
    Nonstationary Time Series and Applications", Communications in
    Applied Mathematics and Computational Science 7, 2 (2012), 175 - 229
    """

    def __init__(self, n_components, init=None, solver='subspace',
                 tol=1e-4, max_iter=200, random_state=None,
                 max_tv_norm=None, verbose=0, fem_basis='constant',
                 **params):
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.max_tv_norm = max_tv_norm
        self.verbose = verbose
        self.fem_basis = fem_basis
        self.params = params

    def fit_transform(self, X, Gamma=None, Theta=None):
        """Calculate FEM-BV-k-means fit for data and return affiliations.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be fitted.

        Gamma : array-like, shape (n_samples, n_components)
            If init='custom', used as initial guess for solution.

        Theta : array-like, shape (n_components, n_features)
            If init='custom', used as initial guess for solution.

        Returns
        -------
        Gamma : array-like, shape (n_samples, n_components)
            Affiliation sequence for the data.
        """
        Gamma, Theta, n_iter_ = fembv_kmeans(
            X, Gamma=Gamma, Theta=Theta,
            n_components=self.n_components,
            init=self.init, update_Theta=True, solver=self.solver,
            tol=self.tol, max_iter=self.max_iter,
            random_state=self.random_state, max_tv_norm=self.max_tv_norm,
            fem_basis=self.fem_basis,
            verbose=self.verbose, **self.params)

        self.cost_ = _fembv_kmeans_cost(X, Gamma, Theta)

        self.n_components_ = Theta.shape[0]
        self.components_ = Theta
        self.n_iter_ = n_iter_

        return Gamma

    def fit(self, X, **params):
        """Calculate FEM-BV-k-means fit for the data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be fitted.

        Returns
        -------
        self
        """
        self.fit_transform(X, **params)
        return self

    def transform(self, X):
        """Transform the data according to the fitted model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to compute representation for.

        Returns
        -------
        Gamma : array-like, shape (n_samples, n_components)
            Affiliation sequence for the data.
        """
        check_is_fitted(self, 'n_components_')

        Gamma, _, n_iter_ = fembv_kmeans(
            X=X, Gamma=None, Theta=self.components_,
            n_components=self.n_components_, init=self.init,
            solver=self.solver, tol=self.tol, max_iter=self.max_iter,
            fem_basis=self.fem_basis, max_tv_norm=self.max_tv_norm,
            verbose=self.verbose)

        return Gamma


class FEMBVBINX(object):
    r"""FEM-BV clustering for binary data with external factors.

    The local distance function is given by the negative log-likelihood

        -y_t log(\Theta_i^T P_x(t)) - (1 - y_t) log(1 - \Theta_i^T P_x(t))

    where the binary variable y_t is assumed to take the values 0 or
    1, P_x is a vector of probabilities at time t for the predictor
    values x, and \Theta_i is a vector of conditional probabilities
    for cluster i.

    The total objective function is minimized with an alternating
    minimization with respect to the cluster probabilities \Theta_i and
    the affiliations Gamma.

    Parameters
    ----------
    n_components : integer or None
        If an integer, the number of clusters. If None, then all features
        are kept.

    init : None | 'random' | 'custom'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls back to 'random'

        - 'random': random initialization of the cluster affiliations
            and probabilities.

        - 'custom': use custom matrices for Gamma and Theta

    solver : 'subspace'
        Numerical solver to use:
        'subspace' is the subspace algorithm.

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations before stopping.

    random_state : integer, RandomState, or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If
        None, the random number generator is the
        RandomState instance used by `np.random`.

    max_tv_norm : None, scalar or array-like, shape (n_components,)
        If a scalar, a common maximum TV norm for all cluster
        affiliations. If array-like, the maximum TV norm for
        each of the separate affiliation sequences. If not given,
        no upper bound is imposed.

    epsilon_Theta : scalar, default: 0
        Regularization parameter for probability vectors.

    verbose : integer, default: 0
        The verbosity level.

    fem_basis : None, string, or callable
        The choice of finite-element basis functions to use. If given
        as a string, valid options are:

        - 'constant' : use piecewise constant basis functions

        - 'triangle' : use triangle basis functions with a width of 3
            grid points

        If given as a callable object, must be callable with the
        signature fem_basis(n_samples), and should return an
        array of shape (n_samples, n_elements) where n_elements is
        the number of finite-element basis functions, containing the
        values of the basis functions at each grid point. If None,
        defaults to piecewise constant basis functions.

    Attributes
    ----------
    components_ : array-like, shape (n_components, n_features)
        The cluster parameters.

    cost_ : number
        Value of the FEM-BV cost function for the obtained fit.

    n_iter_ : integer
        Actual number of iterations.

    Examples
    --------
    import numpy as np
    Y = np.random.rand(100, 1)
    Y[Y < 0.5] = 0
    Y[Y >= 0.5] = 1
    X = np.random.rand(100, 4)
    from pyspa.fembv import FEMBVKMeans
    model = FEMBVBINX(n_components=2, init='random', random_state=0)
    Gamma = model.fit_transform(X, Y)
    Theta = model.components_

    References
    ----------
    Gerber, S. and Horenko, I., "On Inference of Causality for Discrete
    State Models in a Multiscale Context", Proceedings of the National
    Academy of Sciences 111, 41 (2014), 14651 - 14656
    """

    def __init__(self, n_components, init=None, solver='subspace',
                 tol=1e-4, max_iter=200, random_state=None,
                 max_tv_norm=None, epsilon_Theta=0,
                 verbose=0, fem_basis='constant',
                 **params):
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.max_tv_norm = max_tv_norm
        self.epsilon_Theta = epsilon_Theta
        self.verbose = verbose
        self.fem_basis = fem_basis
        self.params = params

    def _merge_data(self, X, Y):
        n_samples = Y.shape[0]
        if Y.ndim == 1:
            return np.concatenate([np.reshape(Y, (n_samples, 1)), X], axis=-1)
        else:
            return np.hstack([Y, X])

    def fit_transform(self, X, Y, Gamma=None, Theta=None, u=None):
        """Calculate FEM-BV-BINX fit for data and return affiliations.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Predictors matrix to be fitted.

        Y : array-like, shape (n_samples,)
            Binary outcome data to be fitted.

        Gamma : array-like, shape (n_samples, n_components)
            If init='custom', used as initial guess for solution.

        Theta : array-like, shape (n_components, n_features)
            If init='custom', used as initial guess for solution.

        u : optional, array-like, shape (n_samples, n_exteranl)
            Array of external factor values.

        Returns
        -------
        Gamma : array-like, shape (n_samples, n_components)
            Affiliation sequence for the data.
        """
        Gamma, Theta, n_iter_ = fembv_binx(
            X, Y, Gamma=Gamma, Theta=Theta, u=u,
            n_components=self.n_components,
            init=self.init, update_Theta=True,
            epsilon_Theta=self.epsilon_Theta,
            solver=self.solver, tol=self.tol,
            max_iter=self.max_iter,
            random_state=self.random_state,
            max_tv_norm=self.max_tv_norm,
            fem_basis=self.fem_basis, verbose=self.verbose,
            **self.params)

        YX = self._merge_data(X, Y)

        self.cost_ = _fembv_binx_cost(
            YX, Gamma, Theta, u=u, epsilon_Theta=self.epsilon_Theta)

        self.n_components_ = Theta.shape[0]
        self.components_ = Theta
        self.n_iter_ = n_iter_

        return Gamma

    def fit(self, X, Y, **params):
        """Calculate FEM-BV-BINX fit for the predictors X and outcome Y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Predictors matrix to be fitted.

        Y : array-like, shape (n_samples,))
            Binary outcome data to be fitted.

        Returns
        -------
        self
        """
        self.fit_transform(X, Y, **params)
        return self

    def transform(self, X, Y, u=None):
        """Transform the data according to the fitted model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Predictors matrix to compute representation for.

        Y : array-like, shape (n_samples,)
            Binary outcome data to compute representation for.

        u : optional, array-like, shape (n_samples, n_external)
            Array of external factor values.

        Returns
        -------
        Gamma : array-like, shape (n_samples, n_components)
            Affiliation sequence for the data.
        """
        check_is_fitted(self, 'n_components_')

        Gamma, _, n_iter_ = fembv_binx(
            X=X, Y=Y, Gamma=None, Theta=self.components_, u=u,
            n_components=self.n_components,
            init=self.init, update_Theta=True,
            epsilon_Theta=self.epsilon_Theta,
            solver=self.solver, tol=self.tol,
            max_iter=self.max_iter,
            max_tv_norm=self.max_tv_norm,
            fem_basis=self.fem_basis, verbose=self.verbose)

        return Gamma

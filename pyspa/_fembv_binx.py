from __future__ import division

import numbers
import numpy as np
import warnings

from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.spatial import ConvexHull

from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from ._fembv_generic import (_check_array_shape,
                             _check_init_fembv_Gamma,
                             _fembv_generic_cost,
                             _fit_generic_fembv_subspace)
from .utils import right_stochastic_matrix

INTEGER_TYPES = (numbers.Integral, np.integer)
FEMBV_BINX_INITIALIZATION_METHODS = (None, 'random')


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
    constraint that each row sums to 0.5, corresponding to the initial
    guess P(y_t = 1) <= 0.5.

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

    Gamma = right_stochastic_matrix(
        (n_samples, n_components), random_state=rng)

    n_features = YX.shape[1] - 1
    if u is None:
        Theta = 0.5 * right_stochastic_matrix(
            (n_components, n_features), random_state=rng)
    else:
        # @todo ensure initial guess is in feasible region
        if u.ndim == 1:
            n_external = 1
        else:
            n_external = u.shape[1]
        n_component_pars = n_features * (n_external + 1)
        Theta = 0.5 * right_stochastic_matrix(
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
        return Bounds(0, 1, keep_feasible=True)

    n_external = u.shape[1]
    n_component_pars = n_features * (n_external + 1)
    n_pars = n_components * n_component_pars
    lb = np.full(n_pars, -np.inf)
    ub = np.full(n_pars, np.inf)
    for i in range(n_components):
        lb[i * n_component_pars:i * n_component_pars + n_features] = 0
        ub[i * n_component_pars:i * n_component_pars + n_features] = 1
    return Bounds(lb, ub, keep_feasible=True)


def _fembv_binx_Theta_constraints(n_components, n_features, u=None):
    if u is None:
        n_pars = n_components * n_features
        A_elem = np.identity(n_pars)
        A_sum = np.zeros((n_components, n_pars))
        for i in range(n_components):
            A_sum[i, i * n_features:(i + 1) * n_features] = 1
        A_constr = np.vstack([A_elem, A_sum])
        return LinearConstraint(A_constr, 0, 1, keep_feasible=True)

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

    return LinearConstraint(A_bounds, 0, 1, keep_feasible=True)


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
        method = 'slsqp'

    options = {'disp': verbose > 0}
    if 'max_iter' in kwargs:
        options['maxiter'] = kwargs['max_iter']
    else:
        options['maxiter'] = 500

    res = minimize(f, x0, args=args, jac=jac, hess=hess,
                   bounds=bounds, constraints=constraints,
                   method=method, options=options,
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
    if X.ndim == 1:
        n_samples = X.shape[0]
        n_features = 1
    else:
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

    if Y.ndim == 1 and X.ndim == 1:
        yx = np.hstack([np.reshape(Y, (n_samples, 1)),
                        np.reshape(X, (n_samples, 1))])
    elif Y.ndim == 1:
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

    log_likelihood_bound_ : number
        Value of the FEM-BV lower-bound on the log likelihood.

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
        if Y.ndim == 1 and X.ndim == 1:
            return np.hstack([np.reshape(Y, (n_samples, 1)),
                              np.reshape(X, (n_samples, 1))])
        elif Y.ndim == 1:
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
        self.log_likelihood_bound_ = -_fembv_binx_cost(
            YX, Gamma, Theta, u=u, epsilon_Theta=0)

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

from __future__ import division

import numbers
import numpy as np
import warnings

from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from ._fembv_generic import (_check_array_shape,
                             _check_init_fembv_Gamma,
                             _fembv_generic_cost,
                             _fit_generic_fembv_subspace)
from .utils import right_stochastic_matrix

INTEGER_TYPES = (numbers.Integral, np.integer)
FEMBV_VARX_INITIALIZATION_METHODS = (None, 'random')


def _check_init_fembv_varx_Theta(Theta, n_components, n_features,
                                 memory, whom, u=None):
    if len(Theta) != n_components:
        raise ValueError('Wrong number of elements passed to %s. '
                         'Expected %s, but got %s' %
                         (whom, n_components, len(Theta)))

    for k in range(n_components):
        Theta[k]['mu'] = check_array(Theta[k]['mu'])
        _check_array_shape(Theta[k]['mu'], (n_features,), whom)
        if memory > 0:
            Theta[k]['At'] = check_array(Theta[k]['At'])
            At_shape = (memory * n_features, n_features)
            _check_array_shape(Theta[k]['At'], At_shape, whom)
        if u is not None:
            Theta[k]['Bt'] = check_array(Theta[k]['Bt'])
            n_external = u.shape[1]
            Bt_shape = (n_external, n_features)
            _check_array_shape(Theta[k]['Bt'], Bt_shape, whom)
        Theta[k]['C'] = check_array(Theta[k]['Sigma'])
        _check_array_shape(
            Theta[k]['Sigma'], (n_features, n_features), whom)


def _fembv_varx_max_memory(Theta):
    return np.max(np.array([t['memory'] for t in Theta]))


def _initialize_fembv_varx_random(X, n_components, memory=0,
                                  u=None, random_state=None,
                                  random_Theta=True):
    """Return random initial affiliations and model parameters.

    The affiliations matrix is first initialized with uniformly
    distributed random numbers on [0, 1), before the normalization
    requirement is imposed.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be fitted.

    n_components : integer
        The number of clusters desired.

    u : optional, array-like, shape (n_samples, n_external)
        External factors to be used in fitting the data.

    memory : integer, default: 0
        Maximum lag used in AR model.

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
    rng = check_random_state(random_state)

    n_samples, n_features = X.shape
    max_memory = np.max(memory)
    n_terms = n_samples - max_memory
    if u is None:
        n_external = 0
    else:
        n_external = u.shape[1]

    Gamma = right_stochastic_matrix(
        (n_terms, n_components), random_state=rng)

    xb = X.mean(axis=0)
    Theta = [{'mu': xb.copy(), 'memory': memory,
              'At': None, 'Bt': None,
              'Sigma': np.identity(n_features)}] * n_components

    for k in range(n_components):
        if memory > 0:
            Theta[k]['At'] = np.empty((memory * n_features, n_features))
            for i in range(memory):
                if random_Theta:
                    Theta[k]['At'][i * n_features:
                                   (i + 1) * n_features] = rng.uniform(
                        size=(n_features, n_features))
                else:
                    Theta[k]['At'][i * n_features:
                                   (i + 1) * n_features] = np.identity(
                        n_features)
        if u is not None:
            if random_Theta:
                Theta[k]['Bt'] = rng.uniform(size=(n_external, n_features))
            else:
                Theta[k]['Bt'] = np.zeros((n_external, n_features))

    return Gamma, Theta


def _initialize_fembv_varx(X, n_components, init='random',
                           u=None, memory=0, random_state=None):
    """Return initial guesses for affiliations and AR model parameters.

    Initial values for the affiliations Gamma and local model parameters
    Theta are calculated using the given initialization method.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be fitted.

    n_components : integer
        The number of clusters.

    init : None | 'random'
        Method used for initialization.
        Default: 'random'
        Valid options:

        - None: falls back to 'random'

        - 'random': random initialization of the cluster affiliations
            and model parameters.

    u : optional, array-like, shape (n_samples, n_external)
        External factors to be used in fitting the data.

    memory : integer, default: 0
        Maximum lag used in AR model.

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
        Initial guess for model parameters
    """
    if init is None:
        init = 'random'

    if init == 'random':
        return _initialize_fembv_varx_random(
            X, n_components, u=u, memory=memory, random_state=random_state)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, FEMBV_VARX_INITIALIZATION_METHODS))


def _fembv_varx_residuals(X, Theta, u=None):
    n_samples, n_features = X.shape
    n_components = len(Theta)
    max_memory = _fembv_varx_max_memory(Theta)
    n_terms = n_samples - max_memory

    res = np.empty((n_components, n_terms, n_features))
    for i in range(n_terms):
        t = i + max_memory
        for j in range(n_components):
            m = Theta[j]['memory']
            res[j, i] = X[t] - Theta[j]['mu']
            if m > 0:
                x_lag = np.ravel(X[t - m:t])
                res[j, i] -= np.dot(x_lag, Theta[j]['At'])
            if u is not None:
                res[j, i] -= np.dot(u[t], Theta[j]['Bt'])

    return res


def _fembv_varx_residual_norms(X, Theta, u=None):
    residuals = _fembv_varx_residuals(X, Theta, u=u)
    n_components, n_residuals = residuals.shape[:2]

    norms = np.zeros((n_components, n_residuals))
    for j in range(n_components):
        norms[j] = np.linalg.norm(residuals[j], axis=1)

    return norms


def _fembv_varx_distance_matrix(X, Theta, *pars):
    u = pars[0]
    return np.transpose(_fembv_varx_residual_norms(X, Theta, u=u) ** 2)


def _fembv_varx_cost(X, Gamma, Theta, u=None):
    return _fembv_generic_cost(
        X, Gamma, Theta, _fembv_varx_distance_matrix,
        distance_matrix_pars=[u])


def _fembv_varx_Sigma_estimate(X, Gamma, Theta, u=None):
    n_features = X.shape[1]
    n_components = Gamma.shape[1]

    normalizations = np.sum(Gamma, axis=0)

    residuals = _fembv_varx_residuals(X, Theta, u=u)
    res_means = np.mean(residuals, axis=1)
    res_shift = residuals - res_means[:, np.newaxis, :]

    Sigma = np.empty((n_components, n_features, n_features))
    for i in range(n_components):
        weighted_res = res_shift[i] * np.sqrt(Gamma[:, i][:, np.newaxis])
        Sigma[i] = (np.dot(np.transpose(weighted_res), weighted_res) /
                    (normalizations[i] - 1))

    return Sigma


def _fembv_varx_Theta_update(X, Gamma, Theta, *pars):
    u = pars[0]

    n_samples, n_features = X.shape
    n_components = Gamma.shape[1]

    if u is None:
        n_external = 0
    else:
        n_external = u.shape[1]

    max_memory = _fembv_varx_max_memory(Theta)
    n_terms = n_samples - max_memory

    Wl = [np.zeros((1 + n_features * Theta[k]['memory'] + n_external,
                    n_features)) for k in range(n_components)]
    Zl = [np.zeros((1 + n_features * Theta[k]['memory'] + n_external,
                    1 + n_features * Theta[k]['memory'] + n_external))
          for k in range(n_components)]

    for i in range(n_terms):
        t = i + max_memory
        y = X[t]
        x_lag = np.ones(1 + n_features * max_memory + n_external)
        for j in range(max_memory):
            x_lag[j * n_features:(j + 1) * n_features] = X[t - max_memory + j]
        if u is not None:
            x_lag[max_memory * n_features + 1:] = u[t]

        for k in range(n_components):
            m = Theta[k]['memory']
            xy = np.outer(x_lag[(max_memory - m) * n_features:], y)
            xx = np.outer(x_lag[(max_memory - m) * n_features:],
                          x_lag[(max_memory - m) * n_features:])
            Wl[k] += Gamma[i, k] * xy
            Zl[k] += Gamma[i, k] * xx

    for k in range(n_components):
        sol, _, _, _ = np.linalg.lstsq(Zl[k], Wl[k], rcond=None)
        m = Theta[k]['memory']
        if m > 0:
            Theta[k]['At'] = sol[:m * n_features]
        Theta[k]['mu'] = sol[m * n_features]
        if u is not None:
            Theta[k]['Bt'] = sol[m * n_features + 1:]

    sigma_est = _fembv_varx_Sigma_estimate(X, Gamma, Theta, u=u)
    for i in range(n_components):
        Theta[i]['Sigma'] = sigma_est[i]

    return Theta


def fembv_varx(X, Gamma=None, Theta=None, u=None, n_components=None, memory=0,
               init=None, update_Theta=True, solver='subspace', tol=1e-4,
               max_iter=200, random_state=None, max_tv_norm=None,
               fem_basis='constant', method='interior-point', verbose=0,
               checkpoint=False, checkpoint_file=None, checkpoint_iter=None):
    if X.ndim == 1:
        n_samples = X.shape[0]
        n_features = 1
    else:
        n_samples, n_features = X.shape

    if n_components is None:
        n_components = n_features

    if not isinstance(n_components, INTEGER_TYPES) or n_components <= 0:
        raise ValueError('Number of components must be a positive integer;'
                         ' got (n_components=%r)' % n_components)
    if not isinstance(memory, INTEGER_TYPES) or memory < 0:
        raise ValueError('Memory must be a non-negative integer;'
                         ' got (memory=%r)' % memory)
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
                                'FEM-BV-VARX (input Gamma)')
        _check_init_fembv_varx_Theta(
            Theta, n_components, n_features, memory,
            'FEM-BV-VARX (input Theta)', u=u)
    elif not update_Theta:
        _check_init_fembv_varx_Theta(
            Theta, n_components, n_features, memory,
            'FEM-BV-VARX (input Theta)', u=u)
        Gamma, _ = _initialize_fembv_varx(X, n_components, init=init,
                                          u=u, memory=memory,
                                          random_state=random_state)
    else:
        Gamma, Theta = _initialize_fembv_varx(X, n_components, init=init,
                                              u=u, memory=memory,
                                              random_state=random_state)

    if solver == 'subspace':
        Gamma, Theta, n_iter = _fit_generic_fembv_subspace(
            X, Gamma, Theta, _fembv_varx_distance_matrix,
            _fembv_varx_Theta_update,
            theta_update_pars=[u], distance_matrix_pars=[u],
            epsilon_Theta=0, regularization_Theta=None,
            tol=tol, max_iter=max_iter, update_Theta=update_Theta,
            max_tv_norm=max_tv_norm, fem_basis=fem_basis,
            method=method, verbose=verbose, checkpoint=checkpoint,
            checkpoint_file=checkpoint_file, checkpoint_iter=checkpoint_iter)
    else:
        raise ValueError("Invalid solver parameter '%s'." % solver)

    if n_iter == max_iter and tol > 0:
        warnings.warn('Maximum number of iterations %d reached.' % max_iter,
                      warnings.UserWarning)

    return Gamma, Theta, n_iter


class FEMBVVARX(object):
    r"""FEM-BV clustering with (linear) AR model and external factors.


    The total objective function is minimized with an alternating
    minimization with respect to the model parameters \Theta_i and
    the affiliations Gamma.

    Parameters
    ----------
    n_components : integer or None
        If an integer, the number of clusters. If None, then the
        number of clusters is set equal to the number of features.

    memory : integer, default: 0
        Maximum lag used in AR model.

    init : None | 'random' | 'custom'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls backs to 'random'

        - 'random': random initialization of the cluster
            affiliations and model parameters.

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
    components_ :
        The local model parameters.

    cost_ : number
        Value of the FEM-BV cost function for the obtained fit.

    n_iter_ : integer
        Actual number of iterations.

    Examples
    --------
    import numpy as np
    X = np.random.rand(100, 3)
    from pyspa.fembv import FEMBVVARX
    model = FEMBVVARX(n_components=2, memory=1, init='random',
                      random_state=0)
    Gamma = model.fit_transform(X)
    Theta = model.components_

    References
    ----------
    """

    def __init__(self, n_components, memory=0, init=None, solver='subspace',
                 tol=1e-4, max_iter=200, random_state=None,
                 max_tv_norm=None, verbose=0, fem_basis='constant',
                 **params):
        self.n_components = n_components
        self.memory = memory
        self.init = init
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.max_tv_norm = max_tv_norm
        self.verbose = verbose
        self.fem_basis = fem_basis
        self.params = params

    def fit_transform(self, X, Gamma=None, Theta=None, u=None):
        """Calculate FEM-BV-VARX fit for data and return affiliations.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be fitted.

        Gamma : array-like, shape (n_samples, n_components):
            If init='custom', used as initial guess for solution.

        Theta :
            If init='custom', used as initial guess for solution.

        u : optional, array-like, shape (n_samples, n_external)
            Array of external factor values.

        Returns
        -------
        Gamma : array-like, shape (n_samples, n_components)
            Affiliation sequence for the data.
        """
        Gamma, Theta, n_iter_ = fembv_varx(
            X, Gamma=Gamma, Theta=Theta, u=u,
            n_components=self.n_components, memory=self.memory,
            init=self.init, update_Theta=True,
            solver=self.solver, tol=self.tol, max_iter=self.max_iter,
            random_state=self.random_state,
            max_tv_norm=self.max_tv_norm,
            fem_basis=self.fem_basis, verbose=self.verbose,
            **self.params)

        self.cost_ = _fembv_varx_cost(X, Gamma, Theta, u)

        self.n_components_ = len(Theta)
        self.components_ = Theta
        self.n_iter_ = n_iter_

        return Gamma

    def fit(self, X, **params):
        """Calculate FEM-BV-VARX for the data X.

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

    def transform(self, X, u=None):
        """Transform the data according to the fitted model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to compute representation for.

        u : optional, array-like, shape (n_samples, n_external)
            Array of external factor values.

        Returns
        -------
        Gamma : array-like, shape (n_samples, n_components)
            Affiliation sequence for the data.
        """
        check_is_fitted(self, 'n_components_')

        Gamma, _, n_iter_ = fembv_varx(
            X=X, Gamma=None, Theta=self.components_, u=u,
            n_components=self.n_components_, memory=self.memory,
            init=self.init, update_Theta=True,
            solver=self.solver, tol=self.tol, max_iter=self.max_iter,
            max_tv_norm=self.max_tv_norm,
            fem_basis=self.fem_basis, verbose=self.verbose)

        return Gamma

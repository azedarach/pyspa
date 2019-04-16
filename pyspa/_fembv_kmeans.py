import numbers
import numpy as np
import warnings

from sklearn.cluster import KMeans
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from ._fembv_generic import (_check_array_shape,
                             _check_init_fembv_Gamma,
                             _fembv_generic_cost,
                             _fit_generic_fembv_subspace)
from .utils import right_stochastic_matrix

INTEGER_TYPES = (numbers.Integral, np.integer)
FEMBV_KMEANS_INITIALIZATION_METHODS = (None, 'random', 'kmeans')


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

    Gamma = right_stochastic_matrix(
        (n_samples, n_components), random_state=rng)

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

import numbers
import warnings

import numpy as np
import os
import pickle

from joblib import Parallel, delayed

from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from .constraints import simplex_projection
from .optimizers import solve_qp

INTEGER_TYPES = (numbers.Integral, np.integer)

INITIALIZATION_METHODS = (None, "random")


def _create_spa_checkpoint(Gamma, S, checkpoint_file):
    """Serialize model components for checkpointing."""
    temp_file = checkpoint_file + ".old"
    if os.path.exists(checkpoint_file):
        os.rename(checkpoint_file, temp_file)

    data = {"S": S, "Gamma": Gamma}
    with open(checkpoint_file, "wb") as cpf:
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
            'Expected %s, but got %s ' % (whom, shape, np.shape(A)))


def _check_init_Gamma(Gamma, shape, whom):
    Gamma = check_array(Gamma)
    _check_array_shape(Gamma, shape, whom)
    _check_unit_axis_sums(Gamma, whom, axis=1)


def _check_init_S(S, shape, whom):
    S = check_array(S)
    _check_array_shape(S, shape, whom)


def _random_affiliations(shape, random_state=None):
    """Return random matrix with unit row sums."""
    rng = check_random_state(random_state)

    Gamma = rng.uniform(size=shape)
    row_sums = np.sum(Gamma, axis=1)
    Gamma = Gamma / row_sums[:, np.newaxis]

    return Gamma


def _euclidean_spa_dist(X, Gamma, S, normalize=True):
    """Calculate Euclidean SPA cost function without regularization.

    The cost function is given by ||X - Gamma S||_Fro^2 / normalization,
    where the normalization is n_samples * n_features if normalizing
    by the size of the data matrix, or 1 otherwise.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.

    Gamma : array-like, shape (n_samples, n_components)
        The affiliations in the solution X ~= Gamma S

    S : array-like, shape (n_components, n_features)
        The states in the solution X ~= Gamma S

    normalize : boolean, default: True
        If true, normalize the cost function by the size of the data matrix.

    Returns
    -------
    dist : number
        The Frobenius norm of the difference between the original data ``X``
        and the reconstructed data ``Gamma S`` with the desired normalization.
    """
    dist = np.linalg.norm(X - Gamma @ S, 'fro') ** 2
    if normalize:
        return dist / np.size(X)
    else:
        return dist


def _euclidean_spa_regularization_S(S):
    """Return value of regularization function for components."""
    n_components, n_features = S.shape

    if n_components == 1:
        return 0

    reg = (n_components * np.trace(np.transpose(S) @ S) -
           np.sum(S @ np.transpose(S)))

    prefactor = 2.0 / (n_components * n_features * (n_components - 1.0))

    return prefactor * reg


def _euclidean_spa_cost(X, Gamma, S, epsilon_states=0,
                        regularization=None, normalize=True):
    """Return value of cost function for Euclidean SPA method.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.

    Gamma : array-like, shape (n_samples, n_components)
        The affiliations in the solution X ~= Gamma S

    S : array-like, shape (n_components, n_features)
        The states in the solution X ~= Gamma S

    epsilon_states : float, default: 0
        Regularization parameter for states.

    regularization : None | 'components'
        Select whether only affiliations Gamma or states S should be
        regularized, both should be, or neither.

    normalize : boolean, default: True
        If true, normalize the cost function by the size of the data matrix.

    Returns
    -------
    cost : number
        The value of the cost function in the Euclidean SPA algorithm.
    """
    cost = _euclidean_spa_dist(X, Gamma, S, normalize=normalize)

    if regularization == 'components':
        cost += epsilon_states * _euclidean_spa_regularization_S(S)

    return cost


def _euclidean_spa_convergence_check(old_cost, new_cost, tol):
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


def _initialize_euclidean_spa_random(X, n_components, random_state=None):
    """Return random initial affiliations and states.

    The affiliations and states are initialized to random values.
    The affiliations matrix is first initialized with uniformly
    distributed random numbers on [0, 1), before the requirement
    that the result be a stochastic matrix is imposed. The states
    are initialized with normally distributed random numbers
    scaled by sqrt(X.mean() / n_components).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : integer
        The number of states desired in the SPA discretization.

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If
        None, the random number generator is the
        RandomState instance used by `np.random`.

    Returns
    -------
    Gamma : array-like, shape (n_samples, n_components)
        Random initial guess for affiliations in X ~= Gamma S

    S : array-like, shape (n_components, n_features)
        Random initial guess for states in X ~= Gamma S
    """
    n_samples, n_features = X.shape

    avg = np.sqrt(np.abs(X).mean() / n_components)
    rng = check_random_state(random_state)

    S = avg * rng.randn(n_components, n_features)
    Gamma = _random_affiliations(
        (n_samples, n_components), random_state=random_state)

    return Gamma, S


def _initialize_euclidean_spa(X, n_components,
                              init="random", random_state=None):
    """Return initial guesses for affiliations and states.

    Initial values for the affiliations Gamma and states S
    in the SPA decomposition X ~= Gamma S are calculated
    using the given initialization method.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : integer
        The number of states desired in the SPA discretization.

    init : None | 'random'
        Method used for initialization.
        Default: 'random'
        Valid options:

        - None: falls back to 'random'.

        - 'random': random matrix of states scaled by
            sqrt(X.mean() / n_components), and a random
            stochastic matrix of affiliations.

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If
        None, the random number generator is the
        RandomState instance used by `np.random`. Used
        when ``init`` == "random".

    Returns
    -------
    Gamma : array-like, shape (n_samples, n_components)
        Initial guess for affiliations in X ~= Gamma S

    S : array-like, shape (n_components, n_features)
        Initial guess for states in X ~= Gamma S
    """
    if init is None:
        init = 'random'

    if init == 'random':
        return _initialize_euclidean_spa_random(
            X, n_components, random_state=random_state)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, INITIALIZATION_METHODS))


def _subspace_update_euclidean_spa_S(X, Gamma, S, epsilon_states=0):
    """Update states in Euclidean SPA method using exact minimizer."""
    n_samples, n_components = Gamma.shape

    gtx = ((np.transpose(Gamma) @ X) / n_samples)
    gtg = ((np.transpose(Gamma) @ Gamma) / n_samples)

    prefactor = 2.0 * epsilon_states / n_components
    if n_components > 1:
        prefactor *= 1.0 / (n_components - 1.0)

    reg = prefactor * (n_components * np.identity(n_components) -
                       np.ones((n_components, n_components)))

    H_eps = gtg + reg

    (sol, _, _, _) = np.linalg.lstsq(H_eps, gtx, rcond=None)

    return sol


def _solve_euclidean_spa_Gamma_qp_vec(args):
    if args['solver'] == 'spgqp':
        return solve_qp(
            args['P'], args['q'], x0=args['x0'],
            tol=args['tol'], qpsolver='spgqp',
            projector=simplex_projection)
    elif args['solver'] == 'cvxopt':
        n_components = np.size(args['x0'])
        return solve_qp(args['P'], args['q'], tol=args['tol'],
                        qpsolver='cvxopt', A=np.ones((1, n_components)),
                        b=np.ones((1, 1)), G=-np.identity(n_components),
                        h=np.zeros((n_components, 1)))
    else:
        raise RuntimeError("Unrecognized QP solver '%s'" % args['solver'])


def _solve_euclidean_spa_Gamma_qp(Gamma, P, q, tol=1e-4, solver='cvxopt'):
    """Return optimal solution for Gamma by solving QP optimization problem."""
    n_samples = Gamma.shape[0]

    optim_args = ({'P': P, 'q': q[i, :], 'x0': Gamma[i, :],
                   'tol': tol, 'solver': solver}
                  for i in range(n_samples))

    result = Parallel(n_jobs=-1)(
        delayed(_solve_euclidean_spa_Gamma_qp_vec)(a)
        for a in optim_args)

    sol = np.zeros(Gamma.shape)
    for i in range(n_samples):
        sol[i, :] = np.ravel(result[i])

    return sol


def _subspace_update_euclidean_spa_Gamma(X, Gamma, S, tol=1e-4,
                                         min_cost_improvement=1e-10):
    """Update affiliations in Euclidean SPA using descent step."""
    n_samples, n_features = X.shape

    normalization = n_samples * n_features
    q = (-2 * (X @ np.transpose(S)) / normalization)
    P = (2 * (S @ np.transpose(S)) / normalization)

    initial_cost = _euclidean_spa_dist(X, Gamma, S, normalize=True)

    # @todo better implementation of descent step
    evals = np.linalg.eigvalsh(P)
    alpha_try = 1.0 / np.max(np.abs(evals))
    grad = (Gamma @ P + q)
    sol = simplex_projection(Gamma - alpha_try * grad)

    trial_cost = _euclidean_spa_dist(X, sol, S, normalize=True)

    if np.abs(trial_cost - initial_cost) < min_cost_improvement:
        sol = _solve_euclidean_spa_Gamma_qp(Gamma, P, q, tol=tol)

    return sol


def _fit_euclidean_spa_subspace(X, Gamma, S, tol=1e-4, max_iter=200,
                                epsilon_states=0, update_S=True, verbose=0,
                                checkpoint=False, checkpoint_file=None,
                                checkpoint_iter=None):
    """Compute SPA discretization using subspace algorithm.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.

    Gamma : array-like, shape (n_samples, n_components)
        Initial guess for the solution.

    S : array-like, shape (n_components, n_features)
        Initial guess for the solution.

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations before stopping.

    epsilon_states : float, default: 0
        Regularization parameter for states.

    update_S : boolean, default: True
        If True, both Gamma and S will be estimated from initial guesses.
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
        Solution for affiliations in X ~= Gamma S

    S : array-like, shape (n_components, n_features)
        Solution for states in X ~= Gamma S

    n_iter : integer
        The number of iterations done in the algorithm.
    """
    if update_S:
        regularization = 'components'
    else:
        regularization = None

    for n_iter in range(max_iter):
        initial_cost = _euclidean_spa_cost(
            X, Gamma, S, epsilon_states=epsilon_states,
            regularization=regularization, normalize=True)

        if n_iter == 0:
            start_cost = initial_cost

        if update_S:
            S = _subspace_update_euclidean_spa_S(
                X, Gamma, S, epsilon_states=epsilon_states)

        Gamma = _subspace_update_euclidean_spa_Gamma(X, Gamma, S, tol=tol)

        final_cost = _euclidean_spa_cost(
            X, Gamma, S, epsilon_states=epsilon_states,
            regularization=regularization, normalize=True)

        if verbose:
            print('Iteration %d:' % (n_iter + 1))
            print('--------------')
            print('Cost function = %.5e' % final_cost)
            print('Cost delta = %.5e' % (final_cost - initial_cost))
            print('Cost ratio = %.5e' % (final_cost / start_cost))

        if checkpoint and n_iter % checkpoint_iter == 0:
            _create_spa_checkpoint(Gamma, S, checkpoint_file)

        converged = _euclidean_spa_convergence_check(
            initial_cost, final_cost, tol)

        if converged:
            if verbose:
                print("*** Converged at iteration %d ***" % (n_iter + 1))
            break

    return Gamma, S, n_iter


def euclidean_spa(X, Gamma=None, S=None, n_components=None,
                  init=None, update_S=True, solver='subspace', tol=1e-4,
                  max_iter=200, random_state=None, epsilon_states=0,
                  verbose=0, checkpoint=False, checkpoint_file=None,
                  checkpoint_iter=None):
    r"""Compute SPA discretization using Euclidean cost function.

    The objective function is::

        ||X - Gamma S||_Fro^2 / (n_samples * n_features)
        + epsilon_states * \Phi_S(S)

    where the data matrix has shape (n_samples, n_features) and::

        \Phi_S(S) = \sum_{i, j = 1}^{n_components} ||s_i - s_j||_2^2
                    / (n_features * n_components * (n_components - 1))

    The objective function is minimized with an alternating minimization
    with respect to S and Gamma. If S is given and update_S=False,
    it solves for Gamma only.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix to be decomposed.

    Gamma : array-like, shape (n_samples, n_components)
        If init='custom', used as initial guess for the solution.

    S : array-like, shape (n_components, n_features)
        If init='custom', used as initial guess for the solution.
        If update_S=False, used as a constant to solve for Gamma only.

    n_components : integer or None
        If an integer, the number of components. If None, then all
        features are kept.

    init : None | 'random' | 'custom'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls back to 'random'

        - 'random': random matrix of states scaled by
            sqrt(X.mean() / n_components), and a random
            stochastic matrix of affiliations.

        - 'custom': use custom matrices for Gamma and S

    update_S : boolean, default: True
        If True, both Gamma and S will be estimated from initial guesses.
        If False, only Gamma will be computed.

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

    epsilon_states : float, default: 0
        Regularization parameter for states.

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
        Solution for affiliations in X ~= Gamma S

    S : array-like, shape (n_components, n_features)
        Solution for states in X ~= Gamma S

    n_iter : integer
        The number of iterations done in the algorithm.
    """
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

    if init == 'custom' and update_S:
        _check_init_Gamma(Gamma, (n_samples, n_components),
                          'Euclidean SPA (input Gamma)')
        _check_init_S(S, (n_components, n_features),
                      'Euclidean SPA (input S)')
    elif not update_S:
        _check_init_S(S, (n_components, n_features),
                      'Euclidean SPA (input S)')
        Gamma = _random_affiliations(
            (n_samples, n_components), random_state=random_state)
    else:
        Gamma, S = _initialize_euclidean_spa(
            X, n_components, init=init, random_state=random_state)

    if solver == 'subspace':
        Gamma, S, n_iter = _fit_euclidean_spa_subspace(
            X, Gamma, S, tol, max_iter,
            epsilon_states, update_S=update_S,
            verbose=verbose,
            checkpoint=checkpoint, checkpoint_file=checkpoint_file,
            checkpoint_iter=checkpoint_iter)
    else:
        raise ValueError("Invalid solver parameter '%s'." % solver)

    if n_iter == max_iter and tol > 0:
        warnings.warn('Maximum number of iterations %d reached.' % max_iter,
                      warnings.UserWarning)

    return Gamma, S, n_iter


class EuclideanSPA(object):
    r"""SPA discretization with Euclidean cost function.

    The objective function is::

        ||X - Gamma S||_Fro^2 / (n_samples * n_features)
        + epsilon_states * \Phi_S(S)

    where the data matrix has shape (n_samples, n_features) and::

        \Phi_S(S) = \sum_{i, j = 1}^{n_components} ||s_i - s_j||_2^2
                    / (n_features * n_components * (n_components - 1))

    The objective function is minimized with an alternating minimization
    with respect to S and Gamma.

    Parameters
    ----------
    n_components : integer or None
        If an integer, the number of components. If None, then all
        features are kept.

    init : None | 'random' | 'custom'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls back to 'random'

        - 'random': random matrix of states scaled by
            sqrt(X.mean() / n_components), and a random
            stochastic matrix of affiliations.

        - 'custom': use custom matrices for Gamma and S

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

    epsilon_states : float, default: 0
        Regularization parameter for states.

    verbose : integer, default: 0
        The verbosity level.

    Attributes
    ----------
    components_ : array-like, shape (n_components, n_features)
        The dictionary of states or atoms.

    reconstruction_err_ : number
        Frobenius norm of the matrix difference between the
        training data ``X`` and the reconstructed data
        ``Gamma S`` from the SPA algorithm.

    n_iter_ : integer
        Actual number of iterations.

    Examples
    --------
    import numpy as np
    X = np.random.rand(10, 4)
    from pyspa.spa import EuclideanSPA
    model = EuclideanSPA(n_components=2, init='random', random_state=0)
    Gamma = model.fit_transform(X)
    S = model.components_

    References
    ----------
    S. Gerber, L. Pospisil, M. Navandar, and I. Horenko,
    "Low-cost scalable discretization, prediction and feature selection
    for complex systems" (2018)
    """

    def __init__(self, n_components, init=None, solver='subspace',
                 tol=1e-4, max_iter=200, random_state=None,
                 epsilon_states=0, verbose=0):
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.epsilon_states = epsilon_states
        self.verbose = verbose

    def fit_transform(self, X, Gamma=None, S=None):
        """Calculate Euclidean SPA discretization and return transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be discretized.

        Gamma : array-like, shape (n_samples, n_components)
            If init='custom', used as initial guess for solution.

        S : array-like, shape (n_components, n_features)
            If init='custom', used as initial guess for solution.

        Returns
        -------
        Gamma : array-like, shape (n_samples, n_components)
            Represention of data.
        """
        Gamma, S, n_iter_ = euclidean_spa(
            X=X, Gamma=Gamma, S=S, n_components=self.n_components,
            init=self.init, solver=self.solver, tol=self.tol,
            max_iter=self.max_iter, epsilon_states=self.epsilon_states,
            verbose=self.verbose)

        self.reconstruction_err_ = _euclidean_spa_dist(
            X, Gamma, S, normalize=False)

        self.n_components_ = S.shape[0]
        self.components_ = S
        self.n_iter_ = n_iter_

        return Gamma

    def fit(self, X, **params):
        """Calculate Euclidean SPA discretization for the data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be discretized.

        Returns
        -------
        self
        """
        self.fit_transform(X, **params)
        return self

    def transform(self, X):
        """Transform the data according to the fitted SPA discretization.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to compute representation for.

        Returns
        -------
        Gamma : array-like, shape (n_samples, n_components)
            Representation of data.
        """
        check_is_fitted(self, 'n_components_')

        Gamma, _, n_iter_ = euclidean_spa(
            X=X, Gamma=None, S=self.components_,
            n_components=self.n_components_,
            init=self.init, solver=self.solver, tol=self.tol,
            max_iter=self.max_iter,
            epsilon_states=self.epsilon_states, verbose=self.verbose)

        return Gamma

    def inverse_transform(self, Gamma):
        """Transform data back into its original space.

        Parameters
        ----------
        Gamma : array-like, shape (n_samples, n_components)
            Representation of data matrix.

        Returns
        -------
        X : array-like, shape (n_samples, n_features)
            Data matrix with original shape.
        """
        check_is_fitted(self, 'n_components_')
        return np.dot(Gamma, self.components_)


class JointEuclideanSPA(object):
    r"""Joint SPA discretization with Euclidean cost function.

    Convenience class for performing joint discretization of
    two datasets by constructing a merged dataset. As for a
    single dataset, the objective function is::

        ||X_\epsilon - Gamma_x S_\epsilon||_Fro^2 /
        (n_samples * n_total_features)
        + epsilon_states * \Phi_S(S_\epsilon)

    where the augmented data matrix X_\epsilon is constructed
    from the given data matrices X and Y as X_\epsilon = [Y, epsilon_weight X]
    and has shape (n_samples, n_total_features), with
    n_total_features = n_y_features + n_x_features, and::

        \Phi_S(S) = \sum_{i, j = 1}^{n_components} ||s_i - s_j||_2^2
                    / (n_features * n_components * (n_components - 1))

    where S_\epsilon is the augmented dictionary. The objective function is
    minimized with an alternating minimization with respect to S and Gamma.

    Parameters
    ----------
    n_components : integer or None
        If an integer, the number of components. If None, then all
        features are kept.

    init : None | 'random' | 'custom'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls back to 'random'

        - 'random': random matrix of states scaled by
            sqrt(X.mean() / n_components), and a random
            stochastic matrix of affiliations.

        - 'custom': use custom matrices for Gamma and S

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

    epsilon_states : float, default: 0
        Regularization parameter for states.

    epsilon_weight : float, default: 1
        Regularization parameter controlling the weight assigned
        to each discretization.

    verbose : integer, default: 0
        The verbosity level.

    Attributes
    ----------
    components_ : array-like, shape (n_components, n_features)
        The dictionary of states or atoms.

    reconstruction_err_ : number
        Frobenius norm of the matrix difference between the
        training data ``X_\epsilon`` and the reconstructed data
        ``Gamma S_\epsilon`` from the SPA algorithm.

    n_iter_ : integer
        Actual number of iterations.

    Examples
    --------
    import numpy as np
    X = np.random.rand(10, 4)
    Y = np.random.rand(10, 3)
    from pyspa.spa import EuclideanSPA
    model = EuclideanSPA(n_components=2, init='random', random_state=0)
    Gamma = model.fit_transform(X, Y)
    S = model.components_

    References
    ----------
    S. Gerber, L. Pospisil, M. Navandar, and I. Horenko,
    "Low-cost scalable discretization, prediction and feature selection
    for complex systems" (2018)
    """

    def __init__(self, n_components, init=None, solver='subspace',
                 tol=1e-4, max_iter=200, random_state=None,
                 epsilon_states=0, epsilon_weight=1, verbose=0):
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.epsilon_states = epsilon_states
        self.epsilon_weight = epsilon_weight
        self.verbose = verbose

    def _merge_datasets(self, X, Y):
        """Merge two datasets for performing joint discretization.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_x_features)
            First data matrix to be discretized.

        Y : array-like, shape (n_samples, n_y_features)
            Second data matrix to be discretized.

        Returns
        -------
        X_eps : array-like, shape (n_samples, n_x_features + n_y_features)
            Merged dataset.
        """
        n_x_samples = X.shape[0]
        n_y_samples = Y.shape[0]
        if n_x_samples != n_y_samples:
            raise ValueError(
                'Number of samples in X and Y must be the same, '
                'but got (n_x_samples = %r, n_y_samples = %r)' %
                (n_x_samples, n_y_samples))

        return np.hstack([Y, self.epsilon_weight * X])

    def fit_transform(self, X, Y, Gamma=None, S=None):
        """Calculate Euclidean SPA discretization and return transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_x_features)
            Data matrix to be discretized.

        Y : array-like, shape (n_samples, n_y_features)
            Data matrix to be discretized.

        Gamma : array-like, shape (n_samples, n_components)
            If init='custom', used as initial guess for solution.

        S : array-like, shape (n_components, n_x_features + n_y_features)
            If init='custom', used as initial guess for solution.

        Returns
        -------
        Gamma : array-like, shape (n_samples, n_components)
            Represention of data.
        """
        X_eps = self._merge_datasets(X, Y)

        Gamma, S, n_iter_ = euclidean_spa(
            X=X_eps, Gamma=Gamma, S=S, n_components=self.n_components,
            init=self.init, solver=self.solver, tol=self.tol,
            max_iter=self.max_iter, epsilon_states=self.epsilon_states,
            verbose=self.verbose)

        self.reconstruction_err_ = _euclidean_spa_dist(
            X_eps, Gamma, S, normalize=False)

        self.n_components_ = S.shape[0]
        self.n_y_features_ = Y.shape[1]
        self.n_x_features_ = X.shape[1]
        self.components_ = S
        self.n_iter_ = n_iter_

        return Gamma

    def fit(self, X, Y, **params):
        """Calculate Euclidean SPA discretization for the datasets X and Y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_x_features)
            Data matrix to be discretized.

        Y : array-like, shape (n_samples, n_y_features)
            Data matrix to be discretized.

        Returns
        -------
        self
        """
        self.fit_transform(X, Y, **params)
        return self

    def transform(self, X, Y):
        """Transform the data according to the fitted SPA discretization.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_x_features)
            Data matrix to compute representation for.

        Y : array-like, shape (n_samples, n_y_features)

        Returns
        -------
        Gamma : array-like, shape (n_samples, n_components)
            Representation of data.
        """
        check_is_fitted(self, 'n_components_')

        X_eps = self._merge_datasets(X, Y)

        Gamma, _, n_iter_ = euclidean_spa(
            X=X_eps, Gamma=None, S=self.components_,
            n_components=self.n_components_,
            init=self.init, solver=self.solver, tol=self.tol,
            max_iter=self.max_iter,
            epsilon_states=self.epsilon_states, verbose=self.verbose)

        return Gamma

    def inverse_transform(self, Gamma):
        """Transform data back into its original space.

        Parameters
        ----------
        Gamma : array-like, shape (n_samples, n_components)
            Representation of data matrices.

        Returns
        -------
        X : array-like, shape (n_samples, n_x_features)
            Data matrix with original shape.

        Y : array-like, shape (n_samples, n_y_features)
            Data matrix with original shape.
        """
        check_is_fitted(self, 'n_components_')

        X_eps = np.dot(Gamma, self.components_)
        Y = X_eps[:, :self.n_y_features_]
        X = X_eps[:, self.n_y_features:] / self.epsilon_weight

        return X, Y

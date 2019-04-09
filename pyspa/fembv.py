import numbers
import warnings

import numpy as np
import os
import pickle

from scipy.optimize import linprog


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


def _fembv_generic_cost(X, Gamma, Theta, distance_matrix,
                        epsilon_Theta=0, regularization_Theta=None):
    G = distance_matrix(X, Theta)
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


def _subspace_update_fembv_Gamma(G, c, fem_basis=None, max_iter=500, verbose=0,
    method='interior-point'):
    """Update affiliations in FEM-BV using linear programming.

    Parameters
    ----------
    G : array-like, shape (n_samples, n_components)
        Matrix of model distances.

    c : scalar or array-like, shape (n_components,)
        If scalar, a single value corresponding to an upper bound on the
        total variation norm of all affiliations. If array-like, separate
        upper bounds on the total variation norm for each set of
        affiliations.

    fem_basis : callable, optional
        If given, should return an array of size (n_samples, n_elements)
        containing the value of the finite elements at each sample point.

    Returns
    -------
    Gamma : array-like, shape (n_samples, n_components)
        Solution for affiliations in FEM-BV discretization.

    Eta : array-like, shape (n_samples - 1, n_components)
        Auxiliary variables used to enforce bounded variation constraint.
    """
    n_samples, n_components = G.shape

    V = fem_basis(n_samples)
    n_elem = V.shape[1]
    n_coeffs = n_components * n_elem

    vtg = np.dot(np.transpose(V), G)
    g_vec = np.ravel(vtg)

    A_eq = np.zeros((n_samples, 2 * n_coeffs))

    b_eq = np.ones((n_samples,))

    b_pos = np.zeros(n_components * (2 * n_samples - 1))
    if np.isscalar(c):
        b_bv = np.full(n_components, c)
    else:
        b_bv = c
    b_aux = np.zeros(2 * n_components * (n_samples - 1))

    b_ub = np.concatenate([b_pos, b_bv, b_aux])

    options = {'disp': verbose > 0, 'maxiter': max_iter}
    res = linprog(g_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  method=method, options=options)

    if not res['success']:
        raise RuntimeError('failed to solve linear optimization problem')

    sol = res['x']

    alpha = np.reshape(sol[:n_coeffs], (n_elem, n_components))
    beta = np.reshape(sol[n_coeffs:], (n_elem, n_components))

    Gamma = np.dot(V, alpha)
    Eta = np.dot(V[:n_samples, ...], beta)

    return Gamma, Eta


def _fit_generic_fembv_subspace(X, Gamma, Theta, distance_matrix,
                                epsilon_Theta=0, regularization_Theta=None,
                                tol=1e-4, max_iter=200,
                                update_Theta=True, verbose=0,
                                checkpoint=False, checkpoint_file=None,
                                checkpoint_iter=None):
    """Compute FEM-BV fit using subspace algorithm.

    Parameters
    ----------
    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations before stopping.

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
    for n_iter in range(max_iter):
        initial_cost = _fembv_generic_cost(
            X, Gamma, Theta, distance_matrix,
            epsilon_Theta=epsilon_Theta,
            regularization_Theta=regularization_Theta)

        if n_iter == 0:
            start_cost = initial_cost

        if update_Theta:
            Theta = theta_update(X, Gamma)

        G = distance_matrix(X, Theta)
        Gamma = _subspace_update_fembv_Gamma(
            G, c, fem_basis=fem_basis, max_iter=max_iter,
            method=method, verbose=verbose)

        final_cost = _fembv_generic_cost(
            X, Gamma, Theta, distance_matrix,
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


def fembv_discrete(X, Gamma=None, pars=None, n_components=None,
                 init=None, update_pars=True, solver='subspace', tol=1e-4,
                 max_iter=200, random_state=None, fem_basis='hat',
                 epsilon_pars=0, verbose=0, checkpoint=False,
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

    if init == 'custom' and update_pars:
        _check_init_Gamma(Gamma, (n_samples, n_components),
                          'FEMBV (input Gamma)')
        _check_init_pars(pars, (n_components, n_features),
                         'FEMBV (input parameters)')
    elif not update_pars:
        _check_init_pars(pars, (n_components, n_features),
                         'FEMBV (input parameters)')
        Gamma = _random_affiliations(
            (n_samples, n_components), random_state=random_state)
    else:
        Gamma, pars = _initialize_fembv(
            X, n_components, init=init, random_state=random_state)

    if solver == 'subspace':
        Gamma, pars, n_iter = _fit_fembv_binary_subspace()
    else:
        raise ValueError("Invalid solver parameter '%s'." % solver)

    if n_iter == max_iter and tol > 0:
        warnings.warn('Maximum number of iterations %d reached.' % max_iter,
                      warnings.UserWarning)

    return Gamma, pars, n_iter

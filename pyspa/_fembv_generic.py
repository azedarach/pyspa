from __future__ import (division, print_function)

import numpy as np
import os
import pickle
import scipy.sparse as sps

from scipy.optimize import linprog

from sklearn.utils import check_array, check_random_state

from ._fembv_fe_basis import (PiecewiseConstantFEMBasis, TriangleFEMBasis)


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


def _check_fem_basis(V, n_samples, whom):
    if V.shape[0] != n_samples:
        raise ValueError(
            'Array with wrong number of rows passed to %s. '
            'Expected %d, but got %d' % (whom, n_samples, V.shape[0]))
    if np.any(V < 0):
        raise ValueError(
            'Basis elements with negative values passed to %s. '
            'FEM basis functions must be non-negative' % whom)


def _check_init_fembv_Gamma(Gamma, shape, whom):
    Gamma = check_array(Gamma)
    _check_array_shape(Gamma, shape, whom)
    _check_unit_axis_sums(Gamma, whom, axis=1)


def _fembv_generic_cost(X, Gamma, Theta, distance_matrix,
                        distance_matrix_pars=(),
                        epsilon_Theta=0, regularization_Theta=None):
    G = distance_matrix(X, Theta, *distance_matrix_pars)
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

    bounds = [(None, None)] * 2 * n_coeffs

    trivial_bounds_rows = np.array(np.sum(A_pos != 0, axis=1) == 1).flatten()
    if np.any(trivial_bounds_rows):
        A_pos = A_pos.tolil()
        trivial_cols = A_pos[trivial_bounds_rows, :].nonzero()[1]
        for c in trivial_cols:
            bounds[c] = (0, None)
        A_pos = A_pos[np.logical_not(trivial_bounds_rows), :]

    b_pos = np.zeros(A_pos.shape[0])

    A_aux = sps.bmat([[DV_expanded, -V_expanded[:-n_components]],
                      [-DV_expanded, -V_expanded[:-n_components]]])
    b_aux = np.zeros(2 * n_eta_points)

    if max_tv_norm is None:
        A_ub = sps.vstack([A_pos, A_aux])
        b_ub = np.concatenate([b_pos, b_aux])
        return A_ub, b_ub, bounds

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

    return A_csr, b_ub, bounds


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
                                distance_matrix_pars=(),
                                theta_update_pars=(),
                                theta_update_kwargs={},
                                max_tv_norm=None,
                                epsilon_Theta=0, regularization_Theta=None,
                                tol=1e-4, max_iter=200, fem_basis='triangle',
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
        values of the basis functions at each grid point. The basis
        functions must take non-negative values at each grid point.

        If None, defaults to triangular basis functions.

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
    if fem_basis is None or fem_basis == 'triangle':
        fem_basis = TriangleFEMBasis()
    elif fem_basis == 'constant':
        fem_basis = PiecewiseConstantFEMBasis()
    elif not callable(fem_basis):
        raise ValueError('finite-element basis must be callable')

    n_samples, n_components = Gamma.shape

    V = fem_basis(n_samples)
    _check_fem_basis(V, n_samples, '_fit_generic_fembv_subspace')

    A_eq, b_eq = _fembv_Gamma_equality_constraints(n_components, V)
    A_ub, b_ub, gamma_bounds = _fembv_Gamma_upper_bound_constraints(
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
            Theta = theta_update(X, Gamma, Theta, *theta_update_pars,
                                 **theta_update_kwargs)

        G = distance_matrix(X, Theta, *distance_matrix_pars)
        Gamma, _ = _subspace_update_fembv_Gamma(
            G, V, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            max_iter=max_iter, method=method, verbose=verbose,
            bounds=gamma_bounds)

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

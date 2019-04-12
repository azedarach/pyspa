import unittest

import numpy as np

from pyspa.fembv import (_fembv_Gamma_equality_constraints,
                         _fembv_Gamma_upper_bound_constraints,
                         _fembv_binx_cost, _fembv_binx_cost_grad)


def _piecewise_constant_fem_basis(n_samples, n_elem=None, value=1):
    if n_elem is None:
        n_elem = n_samples - 1
    elif n_elem >= n_samples:
        raise ValueError(
            'number of elements must be less than number of samples')

    support = int(np.floor((n_samples - 1) / n_elem))

    V = np.zeros((n_samples, n_elem))
    for i in range(n_elem - 1):
        V[i * support:(i + 1) * support, i] = value
    V[(n_elem - 1) * support:, n_elem - 1] = value

    return V


def _random_fem_coefficients(n_elem, n_components):
    return np.random.uniform(low=0.0, high=1.0, size=(n_elem, n_components))


def _random_affiliations(n_samples, n_components):
    Gamma = np.random.uniform(low=0.0, high=1.0,
                              size=(n_samples, n_components))
    row_sums = np.sum(Gamma, axis=1)
    return Gamma / row_sums[:, np.newaxis]


class TestFEMBVGammaDiscretization(unittest.TestCase):

    def test_vector_form_constraints(self):
        n_samples = 500
        n_components = 10
        n_elem = 250

        V = _piecewise_constant_fem_basis(n_samples, n_elem=n_elem)
        Gamma_coeffs = _random_fem_coefficients(n_elem, n_components)
        Eta_coeffs = _random_fem_coefficients(n_elem, n_components)

        Gamma = np.dot(V, Gamma_coeffs)
        Eta = np.dot(V[:-1], Eta_coeffs)

        c = np.random.uniform(low=1.0, high=10, size=(n_components,))

        coeffs_vec = np.concatenate([np.ravel(Gamma_coeffs),
                                     np.ravel(Eta_coeffs)])
        A_eq, b_eq = _fembv_Gamma_equality_constraints(n_components, V)
        A_ub, b_ub = _fembv_Gamma_upper_bound_constraints(n_components, V, c)

        vec_eq_lhs = np.dot(A_eq.toarray(), coeffs_vec)
        vec_ub_lhs = np.dot(A_ub.toarray(), coeffs_vec)

        expected_eq_lhs = np.sum(Gamma, axis=1)

        expected_pos_lhs = np.concatenate([-np.ravel(Gamma), -np.ravel(Eta)])
        expected_bv_lhs = np.sum(Eta, axis=0)
        expected_aux_lhs = np.concatenate(
            [np.ravel(np.diff(Gamma, axis=0) - Eta),
             np.ravel(-np.diff(Gamma, axis=0) - Eta)])
        expected_ub_lhs = np.concatenate(
            [expected_pos_lhs, expected_aux_lhs, expected_bv_lhs])

        self.assertTrue(np.allclose(vec_eq_lhs, expected_eq_lhs))
        self.assertTrue(np.allclose(vec_ub_lhs, expected_ub_lhs))


def _central_derivative(f, x, args=(), h=1e-6):
    return (f(x + h, *args) - f(x - h, *args)) / (2 * h)


def _random_binary_sequence(n_samples):
    seq = np.random.uniform(low=0.0, high=1.0, size=(n_samples,))
    seq[seq < 0.5] = 0
    seq[seq >= 0.5] = 1
    return seq


class TestFEMBVBINX(unittest.TestCase):

    def test_cost_gradient_no_u_no_epsilon(self):
        n_samples = 500
        n_features = 3
        n_components = 4

        Y = _random_binary_sequence(n_samples)
        X = _random_affiliations(n_samples, n_features)
        YX = np.concatenate([np.reshape(Y, (n_samples, 1)), X], axis=-1)

        Theta = _random_affiliations(n_components, n_features)
        Gamma = _random_affiliations(n_samples, n_components)

        def f(x, *args):
            yx = args[0]
            gamma = args[1]
            theta = args[2].copy()
            row_idx = args[3]
            col_idx = args[4]

            theta[row_idx, col_idx] = x

            return _fembv_binx_cost(yx, gamma, theta)

        numeric = np.zeros((n_components, n_features))
        for i in range(n_components):
            for j in range(n_features):
                args = (YX, Gamma, Theta, i, j)
                x = Theta[i, j]
                numeric[i, j] = _central_derivative(f, x, args=args)

        analytic = _fembv_binx_cost_grad(YX, Gamma, Theta)

        self.assertTrue(np.allclose(numeric, analytic))

    def test_cost_gradient_no_u_epsilon(self):
        n_samples = 500
        n_features = 3
        n_components = 4

        epsilon_Theta = np.random.uniform(low=2, high=10)

        Y = _random_binary_sequence(n_samples)
        X = _random_affiliations(n_samples, n_features)
        YX = np.concatenate([np.reshape(Y, (n_samples, 1)), X], axis=-1)

        Theta = _random_affiliations(n_components, n_features)
        Gamma = _random_affiliations(n_samples, n_components)

        def f(x, *args):
            yx = args[0]
            gamma = args[1]
            theta = args[2].copy()
            eps = args[3]
            row_idx = args[4]
            col_idx = args[5]

            theta[row_idx, col_idx] = x

            return _fembv_binx_cost(yx, gamma, theta, epsilon_Theta=eps)

        numeric = np.zeros((n_components, n_features))
        for i in range(n_components):
            for j in range(n_features):
                args = (YX, Gamma, Theta, epsilon_Theta, i, j)
                x = Theta[i, j]
                numeric[i, j] = _central_derivative(f, x, args=args)

        analytic = _fembv_binx_cost_grad(YX, Gamma, Theta,
                                         epsilon_Theta=epsilon_Theta)

        self.assertTrue(np.allclose(numeric, analytic))

    def test_cost_gradient_u_no_epsilon(self):
        n_samples = 500
        n_features = 3
        n_components = 4
        n_external = 4
        n_pars = n_features * (n_external + 1)

        Y = _random_binary_sequence(n_samples)
        X = _random_affiliations(n_samples, n_features)
        YX = np.concatenate([np.reshape(Y, (n_samples, 1)), X], axis=-1)

        u = np.random.uniform(low=-10, high=10,
                              size=(n_samples, n_external))

        Theta = _random_affiliations(n_components, n_pars)
        Gamma = _random_affiliations(n_samples, n_components)

        def f(x, *args):
            yx = args[0]
            gamma = args[1]
            theta = args[2].copy()
            ext = args[3]
            row_idx = args[4]
            col_idx = args[5]

            theta[row_idx, col_idx] = x

            return _fembv_binx_cost(yx, gamma, theta, u=ext)

        numeric = np.zeros((n_components, n_pars))
        for i in range(n_components):
            for k in range(n_external):
                for j in range(n_features):
                    col_idx = j + k * n_features
                    args = (YX, Gamma, Theta, u, i, col_idx)
                    x = Theta[i, col_idx]
                    numeric[i, col_idx] = _central_derivative(f, x, args=args)

        analytic = _fembv_binx_cost_grad(YX, Gamma, Theta, u=u)

        self.assertTrue(np.allclose(numeric, analytic))

    def test_cost_gradient_u_epsilon(self):
        n_samples = 500
        n_features = 3
        n_components = 4
        n_external = 4
        n_pars = n_features * (n_external + 1)

        Y = _random_binary_sequence(n_samples)
        X = _random_affiliations(n_samples, n_features)
        YX = np.concatenate([np.reshape(Y, (n_samples, 1)), X], axis=-1)

        u = np.random.uniform(low=-10, high=10,
                              size=(n_samples, n_external))
        epsilon_Theta = np.random.uniform(low=2, high=10)

        Theta = _random_affiliations(n_components, n_pars)
        Gamma = _random_affiliations(n_samples, n_components)

        def f(x, *args):
            yx = args[0]
            gamma = args[1]
            theta = args[2].copy()
            ext = args[3]
            eps = args[4]
            row_idx = args[5]
            col_idx = args[6]

            theta[row_idx, col_idx] = x

            return _fembv_binx_cost(yx, gamma, theta, u=ext,
                                    epsilon_Theta=eps)

        numeric = np.zeros((n_components, n_pars))
        for i in range(n_components):
            for k in range(n_external):
                for j in range(n_features):
                    col_idx = j + k * n_features
                    args = (YX, Gamma, Theta, u, epsilon_Theta, i, col_idx)
                    x = Theta[i, col_idx]
                    numeric[i, col_idx] = _central_derivative(f, x, args=args)

        analytic = _fembv_binx_cost_grad(YX, Gamma, Theta,
                                         u=u, epsilon_Theta=epsilon_Theta)

        self.assertTrue(np.allclose(numeric, analytic))

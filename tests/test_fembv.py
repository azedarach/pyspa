import unittest

import numpy as np

from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from scipy.spatial import ConvexHull

from pyspa.fembv import (_fembv_Gamma_equality_constraints,
                         _fembv_Gamma_upper_bound_constraints,
                         _fembv_binx_cost, _fembv_binx_cost_grad,
                         _fembv_binx_cost_hess,
                         _fembv_binx_lambda_vector,
                         _fembv_binx_Theta_bounds,
                         _fembv_binx_Theta_constraints,
                         fembv_binx)


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


def _second_partial_derivative(f, x, i, j, args=(), h=1e-5):
    delta_i = np.zeros(x.shape)
    delta_j = np.zeros(x.shape)

    delta_i[i] = h
    delta_j[j] = h

    xphph = x + delta_i + delta_j
    xphmh = x + delta_i - delta_j
    xmhph = x - delta_i + delta_j
    xmhmh = x - delta_i - delta_j

    return (f(xphph, *args) - f(xphmh, *args) -
            f(xmhph, *args) + f(xmhmh, *args)) / (4 * h * h)


def _random_binary_sequence(n_samples):
    seq = np.random.uniform(low=0.0, high=1.0, size=(n_samples,))
    seq[seq < 0.5] = 0
    seq[seq >= 0.5] = 1
    return seq


def _random_probability_vectors(n_components, n_pars, u=None):
    if u is None:
        return _random_affiliations(n_components, n_pars)

    n_external = u.shape[1]
    n_features = int(n_pars / (n_external + 1))

    x0 = _random_affiliations(n_components, n_pars)
    x0[:, n_features:] = np.random.uniform(low=0, high=0.01,
        size=(n_components, n_pars - n_features))
    x0 = np.reshape(x0, (n_components * n_pars,))

    convex_hull = ConvexHull(u)
    vertices = u[convex_hull.vertices]
    n_vertices = vertices.shape[0]

    def objective(x, *args):
        u = args[0]
        vertices = args[1]
        n_components = args[2]

        n_component_pars = int(np.size(x) / n_components)
        n_vertices = vertices.shape[0]
        n_external = u.shape[1]
        n_features = int(n_component_pars / (n_external + 1))

        x_mat = np.reshape(x, (n_components, n_pars))
        penalty = 1
        value = 0
        for i in range(n_components):
            for j in range(n_vertices):
                lam = x_mat[i, :n_features]
                for k in range(n_external):
                    start = (k + 1) * n_features
                    end = start + n_features
                    lam += vertices[j, k] * x_mat[i, start:end]
                for k in range(n_features):
                    if lam[k] < 0:
                        value += penalty
                if np.sum(lam) < 0 or np.sum(lam) > 1:
                    value += penalty

        return value

    bounds = _fembv_binx_Theta_bounds(n_components, n_features, u=u)
    constraints = _fembv_binx_Theta_constraints(n_components, n_features, u=u)
    args = (u, vertices, n_components)
    res = minimize(objective, x0, args=args,# bounds=bounds,
        constraints=constraints)

    sol = res['x']
    print(res)
    print('f = ', res['fun'])

    theta = np.reshape(sol, (n_components, n_pars))
    for i in range(n_components):
        for j in range(n_vertices):
            lam = theta[i, :n_features]
            for k in range(n_external):
                start = (k + 1) * n_features
                end = start + n_features
                lam += vertices[j, k] * theta[i, start:end]
#            for k in range(n_features):
#                if lam[k] < 0:
#                    print(lam)
#            if np.sum(lam) < 0 or np.sum(lam) > 1:
#                print(lam)

    return theta


class TestFEMBVBINX(unittest.TestCase):

    def test_lambda_vector_no_u(self):
        n_components = 2
        n_component_pars = 2
        theta = _random_affiliations(n_components, n_component_pars)
        for i in range(n_components):
            lam = _fembv_binx_lambda_vector(i, theta)
            self.assertTrue(np.allclose(lam, theta[i, :]))

    def test_lambda_vector_u(self):
        n_samples = 2
        n_components = 2
        n_component_pars = 6

        u = np.random.uniform(
            low=-10, high=-10, size=(2, 2))

        theta = _random_affiliations(n_components, n_component_pars)

        for i in range(n_samples):
            for j in range(n_components):
                lam = _fembv_binx_lambda_vector(j, theta, u=u)
                expected = (theta[j, :2] +
                            u[i, 0] * theta[j, 2:4] + u[i, 1] * theta[j, 4:6])
                self.assertTrue(np.allclose(lam, expected))

    def test_theta_bounds_no_u(self):
        n_components = 2
        n_features = 2
        bounds = _fembv_binx_Theta_bounds(n_components, n_features)
        expected_bounds = Bounds(0, 1)

        self.assertTrue(bounds.lb == expected_bounds.lb)
        self.assertTrue(bounds.ub == expected_bounds.ub)

    def test_theta_bounds_u(self):
        n_components = 2
        n_features = 2
        u = np.random.rand(2, 1)
        bounds = _fembv_binx_Theta_bounds(n_components, n_features, u=u)
        expected_bounds = Bounds(
            np.array([0, 0, -np.inf, -np.inf, 0, 0, -np.inf, -np.inf]),
            np.array([1, 1, np.inf, np.inf, 1, 1, np.inf, np.inf]))

        self.assertTrue(np.allclose(bounds.lb, expected_bounds.lb))
        self.assertTrue(np.allclose(bounds.ub, expected_bounds.ub))

    def test_theta_constraints_no_u(self):
        n_components = 2
        n_features = 2
        constraints = _fembv_binx_Theta_constraints(n_components, n_features)
        A_constr = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
        expected_constraints = LinearConstraint(A_constr, 0, 1)

        self.assertTrue(np.allclose(constraints.A, A_constr))
        self.assertTrue(constraints.lb == expected_constraints.lb)
        self.assertTrue(constraints.ub == expected_constraints.ub)

    def test_theta_constraints_u(self):
        n_components = 2
        n_features = 2
        u = np.random.rand(2, 2)
        constraints = _fembv_binx_Theta_constraints(
            n_components, n_features, u=u)
        A_pos = np.array(
            [[1, 0, u[0, 0], 0, u[0, 1], 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, u[0, 0], 0, u[0, 1], 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, u[0, 0], 0, u[0, 1], 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, u[0, 0], 0, u[0, 1]],
             [1, 0, u[1, 0], 0, u[1, 1], 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, u[1, 0], 0, u[1, 1], 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, u[1, 0], 0, u[1, 1], 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, u[1, 0], 0, u[1, 1]]])
        A_sum = np.array(
            [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
             [1, 1, u[0, 0], u[0, 0], u[0, 1], u[0, 1], 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 1, u[0, 0], u[0, 0], u[0, 1], u[0, 1]],
             [1, 1, u[1, 0], u[1, 0], u[1, 1], u[1, 1], 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 1, u[1, 0], u[1, 0], u[1, 1], u[1, 1]]])
        A_constr = np.vstack([A_pos, A_sum])
        expected_constraints = LinearConstraint(A_constr, 0, 1)

        self.assertTrue(np.allclose(constraints.A, A_constr))
        self.assertTrue(constraints.lb == expected_constraints.lb)
        self.assertTrue(constraints.ub == expected_constraints.ub)

    def test_cost_gradient_no_u_no_epsilon(self):
        n_samples = 500
        n_features = 3
        n_components = 4

        Y = _random_binary_sequence(n_samples)
        X = _random_affiliations(n_samples, n_features)
        YX = np.concatenate([np.reshape(Y, (n_samples, 1)), X], axis=-1)

        Theta = _random_probability_vectors(n_components, n_features)
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

        Theta = _random_probability_vectors(n_components, n_features)
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

    # def test_cost_gradient_u_no_epsilon(self):
    #     n_samples = 500
    #     n_features = 3
    #     n_components = 4
    #     n_external = 4
    #     n_pars = n_features * (n_external + 1)

    #     Y = _random_binary_sequence(n_samples)
    #     X = _random_affiliations(n_samples, n_features)
    #     YX = np.concatenate([np.reshape(Y, (n_samples, 1)), X], axis=-1)

    #     u = np.random.uniform(low=0, high=10,
    #                           size=(n_samples, n_external))

    #     Theta = _random_probability_vectors(n_components, n_pars, u=u)
    #     Gamma = _random_affiliations(n_samples, n_components)

    #     def f(x, *args):
    #         yx = args[0]
    #         gamma = args[1]
    #         theta = args[2].copy()
    #         ext = args[3]
    #         row_idx = args[4]
    #         col_idx = args[5]

    #         theta[row_idx, col_idx] = x

    #         return _fembv_binx_cost(yx, gamma, theta, u=ext)

    #     numeric = np.zeros((n_components, n_pars))
    #     for i in range(n_components):
    #         for k in range(n_external):
    #             for j in range(n_features):
    #                 col_idx = j + k * n_features
    #                 args = (YX, Gamma, Theta, u, i, col_idx)
    #                 x = Theta[i, col_idx]
    #                 numeric[i, col_idx] = _central_derivative(f, x, args=args)

    #     analytic = _fembv_binx_cost_grad(YX, Gamma, Theta, u=u)

    #     self.assertTrue(np.allclose(numeric, analytic))

    # def test_cost_gradient_u_epsilon(self):
    #     n_samples = 500
    #     n_features = 3
    #     n_components = 4
    #     n_external = 4
    #     n_pars = n_features * (n_external + 1)

    #     Y = _random_binary_sequence(n_samples)
    #     X = _random_affiliations(n_samples, n_features)
    #     YX = np.concatenate([np.reshape(Y, (n_samples, 1)), X], axis=-1)

    #     u = np.random.uniform(low=0, high=1,
    #                           size=(n_samples, n_external))
    #     epsilon_Theta = np.random.uniform(low=2, high=10)

    #     Theta = _random_probability_vectors(n_components, n_pars, u=u)
    #     Gamma = _random_affiliations(n_samples, n_components)

    #     def f(x, *args):
    #         yx = args[0]
    #         gamma = args[1]
    #         theta = args[2].copy()
    #         ext = args[3]
    #         eps = args[4]
    #         row_idx = args[5]
    #         col_idx = args[6]

    #         theta[row_idx, col_idx] = x

    #         return _fembv_binx_cost(yx, gamma, theta, u=ext,
    #                                 epsilon_Theta=eps)

    #     numeric = np.zeros((n_components, n_pars))
    #     for i in range(n_components):
    #         for k in range(n_external):
    #             for j in range(n_features):
    #                 col_idx = j + k * n_features
    #                 args = (YX, Gamma, Theta, u, epsilon_Theta, i, col_idx)
    #                 x = Theta[i, col_idx]
    #                 numeric[i, col_idx] = _central_derivative(f, x, args=args)

    #     analytic = _fembv_binx_cost_grad(YX, Gamma, Theta,
    #                                      u=u, epsilon_Theta=epsilon_Theta)

    #     self.assertTrue(np.allclose(numeric, analytic))

    def test_cost_hess_no_u_no_epsilon(self):
        n_samples = 500
        n_features = 2
        n_components = 4
        n_pars = n_components * n_features

        Y = _random_binary_sequence(n_samples)
        X = _random_affiliations(n_samples, n_features)
        YX = np.concatenate([np.reshape(Y, (n_samples, 1)), X], axis=-1)

        Theta = _random_probability_vectors(n_components, n_features)
        Gamma = _random_affiliations(n_samples, n_components)

        def f(x, *args):
            yx = args[0]
            gamma = args[1]

            n_components = gamma.shape[1]
            n_component_pars = yx.shape[1] - 1
            theta = np.reshape(x, (n_components, n_component_pars))

            return _fembv_binx_cost(yx, gamma, theta)

        numeric = np.zeros((n_pars, n_pars))
        for i in range(n_components):
            for j in range(n_features):
                for k in range(n_features):
                    args = (YX, Gamma)
                    row_idx = j + i * n_features
                    col_idx = k + i * n_features
                    x = np.ravel(Theta).copy()
                    numeric[row_idx, col_idx] = _second_partial_derivative(
                        f, x, row_idx, col_idx, args=args)

        analytic = _fembv_binx_cost_hess(YX, Gamma, Theta)

        self.assertTrue(np.allclose(numeric, analytic))

    def test_cost_hess_no_u_epsilon(self):
        n_samples = 500
        n_features = 2
        n_components = 4
        n_pars = n_components * n_features

        Y = _random_binary_sequence(n_samples)
        X = _random_affiliations(n_samples, n_features)
        YX = np.concatenate([np.reshape(Y, (n_samples, 1)), X], axis=-1)

        epsilon_Theta = np.random.uniform(low=2, high=10)

        Theta = _random_probability_vectors(n_components, n_features)
        Gamma = _random_affiliations(n_samples, n_components)

        def f(x, *args):
            yx = args[0]
            gamma = args[1]
            eps = args[2]

            n_components = gamma.shape[1]
            n_component_pars = yx.shape[1] - 1
            theta = np.reshape(x, (n_components, n_component_pars))

            return _fembv_binx_cost(yx, gamma, theta, epsilon_Theta=eps)

        numeric = np.zeros((n_pars, n_pars))
        for i in range(n_components):
            for j in range(n_features):
                for k in range(n_features):
                    args = (YX, Gamma, epsilon_Theta)
                    row_idx = j + i * n_features
                    col_idx = k + i * n_features
                    x = np.ravel(Theta).copy()
                    numeric[row_idx, col_idx] = _second_partial_derivative(
                        f, x, row_idx, col_idx, args=args)

        analytic = _fembv_binx_cost_hess(
            YX, Gamma, Theta, epsilon_Theta=epsilon_Theta)

        self.assertTrue(np.allclose(numeric, analytic))

    def test_two_state_independent_outcomes(self):
        n_samples = 500
        n_components = 1
        tol = 1e-6

        Y = _random_binary_sequence(n_samples)
        X = np.zeros((n_samples - 1, 2))
        X[:, 0][Y[:-1] == 1] = 1
        X[:, 1][Y[:-1] == 0] = 1
        Y = Y[1:]

        Gamma, Theta, n_iter = fembv_binx(
            X, Y, n_components=n_components, tol=tol)

        expected_Theta = np.array([0.5, 0.5])

        self.assertTrue(np.allclose(Theta, expected_Theta, atol=0.05))

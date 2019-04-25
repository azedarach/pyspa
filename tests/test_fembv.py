import unittest

import numpy as np

from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from scipy.spatial import ConvexHull

from sklearn.utils.validation import check_random_state

from pyspa import fembv_binx
from pyspa._fembv_binx import (_fembv_binx_cost, _fembv_binx_cost_grad,
                               _fembv_binx_cost_hess,
                               _fembv_binx_lambda_vector,
                               _fembv_binx_Theta_bounds,
                               _fembv_binx_Theta_constraints,
                               _fembv_binx_Theta_update)
from pyspa._fembv_generic import (_fembv_Gamma_equality_constraints,
                                  _fembv_Gamma_upper_bound_constraints)
from pyspa._fembv_varx import (_fembv_varx_cost, _fembv_varx_Theta_update)


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


def _random_fem_coefficients(n_elem, n_components, random_state=None):
    rng = check_random_state(random_state)
    return rng.uniform(low=0.0, high=1.0, size=(n_elem, n_components))


def _random_affiliations(n_samples, n_components, random_state=None):
    rng = check_random_state(random_state)
    Gamma = rng.uniform(low=0.0, high=1.0,
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


def _second_partial_derivative(f, x, i, j, args=(), h=1e-4):
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


def _ref_fembv_varx_cost(X, Gamma, mu, A=None, B=None, u=None):
    n_terms, n_components = Gamma.shape

    if len(mu) != n_components:
        raise ValueError('incorrect number of mu vectors provided')

    if A is not None:
        if len(A) != n_components:
            raise ValueError('incorrect number of A matrices provided')
    if B is not None:
        if len(B) != n_components:
            raise ValueError('incorrect number of B matrices provided')

    if A is None:
        max_memory = 0
    else:
        max_memory = np.max(np.array([A[j].shape[0]
                                      if A[j] is not None else 0
                                      for j in range(n_components)]))

    cost = 0
    for i in range(n_terms):
        t = i + max_memory
        xt = X[t]
        for j in range(n_components):
            muj = mu[j]
            r = xt - muj

            if A is not None:
                Aj = A[j]
                if Aj is not None:
                    m = Aj.shape[0]
                    for k in range(m):
                        xtmk = X[t - k - 1]
                        r -= np.dot(Aj[k], xtmk)

            if B is not None and u is not None:
                r -= np.dot(B[j], u[t])

            cost += Gamma[i, j] * np.linalg.norm(r) ** 2

    return cost


def _fembv_varx_pars_to_vec(mu, A=None, B=None):
    x = np.concatenate(mu)
    if A is not None:
        A_vecs = [np.reshape(a, np.size(a)) for a in A if a is not None]
        x = np.concatenate([x] + A_vecs)

    if B is not None:
        B_vecs = [np.reshape(b, np.size(b)) for b in B]
        x = np.concatenate([x] + B_vecs)

    return x


def _fembv_varx_vec_to_pars(x, n_components, n_features, memory=0, u=None):
    n_pars = np.size(x)

    if np.isscalar(memory):
        memory = np.full((n_components,), memory, dtype='i8')
    else:
        memory = np.asarray(memory, dtype='i8')

    mu_vec = x[:n_components * n_features]
    mu = [mu_vec[i * n_features:(i + 1) * n_features]
          for i in range(n_components)]

    A = None
    if n_pars > n_components * n_features:
        offset = n_components * n_features
        A = [None] * n_components
        for i in range(n_components):
            if memory[i] > 0:
                n_A_pars = memory[i] * n_features * n_features
                A[i] = np.reshape(x[offset:offset + n_A_pars],
                                  (memory[i], n_features, n_features))
                offset += n_A_pars

    B = None
    if u is not None:
        n_external = u.shape[1]
        n_ext_pars = n_features * n_external
        B = [np.reshape(x[offset + i * n_ext_pars:
                          offset + (i + 1) * n_ext_pars],
                        (n_features, n_external))
             for i in range(n_components)]

    return mu, A, B


def _fembv_varx_theta_solution(X, Gamma, mu, A=None, B=None, u=None):
    def objective(x, *args):
        data = args[0]
        gamma = args[1]
        ext = args[2]
        memory = args[3]

        n_features = data.shape[1]
        n_components = gamma.shape[1]

        mu, A, B = _fembv_varx_vec_to_pars(x, n_components, n_features,
                                           memory=memory, u=ext)

        return _ref_fembv_varx_cost(data, gamma, mu, A=A, B=B, u=ext)

    n_features = X.shape[1]
    n_components = Gamma.shape[1]
    memory = 0
    if A is not None:
        memory = [a.shape[0] if a is not None else 0 for a in A]
    x0 = _fembv_varx_pars_to_vec(mu, A=A, B=B)
    args = (X, Gamma, u, memory)

    res = minimize(objective, x0, args=args)

    if not res['success']:
        raise RuntimeError('failed to solve for VARX model parameters')

    mu_sol, A_sol, B_sol = _fembv_varx_vec_to_pars(
        res['x'], n_components, n_features, memory=memory, u=u)

    return {'mu': mu_sol, 'A': A_sol, 'B': B_sol}


def _weighted_mean(X, weights):
    return np.sum(weights[:, np.newaxis] * X, axis=0) / np.sum(weights)


class TestFEMBVVARX(unittest.TestCase):

    def test_varx_cost_no_u_no_memory(self):
        rng = np.random.RandomState(seed=0)

        n_samples = 300
        n_features = 5
        n_components = 2
        memory = 0
        n_terms = n_samples - np.max(memory)

        X = rng.rand(n_samples, n_features)
        Gamma = _random_affiliations(
            n_terms, n_components, random_state=rng)
        mu = [rng.uniform(size=(n_features,))
              for i in range(n_components)]

        Theta = [{'mu': mu[i], 'memory': memory, 'At': None,
                  'Bt': None, 'Sigma': np.identity(n_features)}
                 for i in range(n_components)]

        cost = _fembv_varx_cost(X, Gamma, Theta)
        expected_cost = _ref_fembv_varx_cost(X, Gamma, mu)

        tol = 1e-9
        self.assertTrue(np.abs(cost - expected_cost) < tol)

    def test_varx_cost_no_u_uniform_memory(self):
        rng = np.random.RandomState(seed=0)

        n_samples = 400
        n_features = 4
        n_components = 3
        memory = 2
        n_terms = n_samples - np.max(memory)

        X = rng.rand(n_samples, n_features)
        Gamma = _random_affiliations(
            n_terms, n_components, random_state=rng)

        mu = [rng.uniform(size=(n_features,))
              for i in range(n_components)]
        A = [rng.uniform(size=(memory, n_features, n_features))
             for i in range(n_components)]

        Theta = [{'mu': mu[i], 'memory': memory,
                  'At': np.vstack([np.transpose(A[i][j])
                                   for j in range(memory - 1, -1, -1)]),
                  'Bt': None, 'Sigma': np.identity(n_features)}
                 for i in range(n_components)]

        cost = _fembv_varx_cost(X, Gamma, Theta)
        expected_cost = _ref_fembv_varx_cost(X, Gamma, mu, A)

        tol = 1e-9
        self.assertTrue(np.abs(cost - expected_cost) < tol)

    def test_varx_cost_no_u_variable_memory(self):
        rng = np.random.RandomState(seed=0)

        n_samples = 300
        n_features = 5
        n_components = 3
        memory = [4, 0, 1]
        n_terms = n_samples - np.max(memory)

        X = rng.rand(n_samples, n_features)
        Gamma = _random_affiliations(
            n_terms, n_components, random_state=rng)

        mu = [rng.uniform(size=(n_features,))
              for i in range(n_components)]
        A = [rng.uniform(size=(memory[i], n_features, n_features))
             if memory[i] > 0 else None
             for i in range(n_components)]

        Theta = [{'mu': mu[i], 'memory': memory[i],
                  'At': np.vstack([np.transpose(A[i][j])
                                   for j in range(memory[i] - 1, -1, -1)])
                  if memory[i] > 0 else None,
                  'Bt': None, 'Sigma': np.identity(n_features)}
                 for i in range(n_components)]

        cost = _fembv_varx_cost(X, Gamma, Theta)
        expected_cost = _ref_fembv_varx_cost(X, Gamma, mu, A)

        tol = 1e-9
        self.assertTrue(np.abs(cost - expected_cost) < tol)

    def test_varx_cost_u_no_memory(self):
        rng = np.random.RandomState(seed=0)

        n_samples = 500
        n_features = 2
        n_components = 10
        memory = 0
        n_terms = n_samples - np.max(memory)
        n_external = 5

        X = rng.rand(n_samples, n_features)
        u = rng.rand(n_samples, n_external)
        Gamma = _random_affiliations(
            n_terms, n_components, random_state=rng)

        mu = [rng.uniform(size=(n_features,))
              for i in range(n_components)]
        B = [rng.uniform(size=(n_features, n_external))
             for i in range(n_components)]

        Theta = [{'mu': mu[i], 'memory': memory, 'At': None,
                  'Bt': np.transpose(B[i]),
                  'Sigma': np.identity(n_features)}
                 for i in range(n_components)]

        cost = _fembv_varx_cost(X, Gamma, Theta, u=u)
        expected_cost = _ref_fembv_varx_cost(X, Gamma, mu, B=B, u=u)

        tol = 1e-9
        self.assertTrue(np.abs(cost - expected_cost) < tol)

    def test_varx_cost_u_uniform_memory(self):
        rng = np.random.RandomState(seed=0)

        n_samples = 179
        n_features = 7
        n_components = 2
        memory = 6
        n_terms = n_samples - np.max(memory)
        n_external = 3

        X = rng.rand(n_samples, n_features)
        u = rng.rand(n_samples, n_external)
        Gamma = _random_affiliations(
            n_terms, n_components, random_state=rng)

        mu = [rng.uniform(size=(n_features,))
              for i in range(n_components)]
        A = [rng.uniform(size=(memory, n_features, n_features))
             for i in range(n_components)]
        B = [rng.uniform(size=(n_features, n_external))
             for i in range(n_components)]

        Theta = [{'mu': mu[i], 'memory': memory,
                  'At': np.vstack([np.transpose(A[i][j])
                                   for j in range(memory - 1, -1, -1)]),
                  'Bt': np.transpose(B[i]),
                  'Sigma': np.identity(n_features)}
                 for i in range(n_components)]

        cost = _fembv_varx_cost(X, Gamma, Theta, u=u)
        expected_cost = _ref_fembv_varx_cost(X, Gamma, mu, A=A, B=B, u=u)

        tol = 1e-9
        self.assertTrue(np.abs(cost - expected_cost) < tol)

    def test_varx_cost_u_variable_memory(self):
        rng = np.random.RandomState(seed=0)

        n_samples = 786
        n_features = 5
        n_components = 8
        memory = [1, 0, 0, 3, 4, 2, 8, 0]
        n_terms = n_samples - np.max(memory)
        n_external = 6

        X = rng.rand(n_samples, n_features)
        u = rng.rand(n_samples, n_external)
        Gamma = _random_affiliations(
            n_terms, n_components, random_state=rng)

        mu = [rng.uniform(size=(n_features,))
              for i in range(n_components)]
        A = [rng.uniform(size=(memory[i], n_features, n_features))
             if memory[i] > 0 else None
             for i in range(n_components)]
        B = [rng.uniform(size=(n_features, n_external))
             for i in range(n_components)]

        Theta = [{'mu': mu[i], 'memory': memory[i],
                  'At': np.vstack([np.transpose(A[i][j])
                                   for j in range(memory[i] - 1, -1, -1)])
                  if memory[i] > 0 else None,
                  'Bt': np.transpose(B[i]), 'Sigma': np.identity(n_features)}
                 for i in range(n_components)]

        cost = _fembv_varx_cost(X, Gamma, Theta, u=u)
        expected_cost = _ref_fembv_varx_cost(X, Gamma, mu, A=A, B=B, u=u)

        tol = 1e-9
        self.assertTrue(np.abs(cost - expected_cost) < tol)

    def test_varx_theta_update_no_u_no_memory(self):
        rng = np.random.RandomState(seed=0)

        n_samples = 100
        n_features = 4
        n_components = 3
        memory = 0
        max_memory = np.max(memory)
        n_terms = n_samples - max_memory

        X = rng.rand(n_samples, n_features)
        Gamma = _random_affiliations(
            n_terms, n_components, random_state=rng)

        mu = [rng.uniform(size=(n_features,))
              for i in range(n_components)]

        Theta = [{'mu': mu[i], 'memory': memory,
                  'At': None, 'Bt': None,
                  'Sigma': np.identity(n_features)}
                 for i in range(n_components)]

        Theta_sol = _fembv_varx_Theta_update(X, Gamma, Theta, None)
        numerical_sol = _fembv_varx_theta_solution(X, Gamma, mu)
        analytic_sol = {'mu': [_weighted_mean(X[max_memory:], Gamma[:, i])
                               for i in range(n_components)],
                        'A': None, 'B': None}

        for j in range(n_components):
            self.assertTrue(np.allclose(numerical_sol['mu'][j],
                                        analytic_sol['mu'][j]))
            self.assertTrue(np.allclose(numerical_sol['mu'][j],
                                        Theta_sol[j]['mu']))

    def test_varx_theta_update_no_u_uniform_memory(self):
        rng = np.random.RandomState(seed=0)

        n_samples = 200
        n_features = 3
        n_components = 2
        memory = 2
        max_memory = np.max(memory)
        n_terms = n_samples - max_memory

        X = rng.rand(n_samples, n_features)
        Gamma = _random_affiliations(
            n_terms, n_components, random_state=rng)

        mu = [rng.uniform(size=(n_features,))
              for i in range(n_components)]
        A = [rng.uniform(size=(memory, n_features, n_features))
             for i in range(n_components)]

        Theta = [{'mu': mu[i], 'memory': memory,
                  'At': np.vstack([np.transpose(A[i][j])
                                   for j in range(memory - 1, -1, -1)]),
                  'Bt': None,
                  'Sigma': np.identity(n_features)}
                 for i in range(n_components)]

        Theta_sol = _fembv_varx_Theta_update(X, Gamma, Theta, None)
        numerical_sol = _fembv_varx_theta_solution(X, Gamma, mu, A)

        for j in range(n_components):
            self.assertTrue(np.allclose(numerical_sol['mu'][j],
                                        Theta_sol[j]['mu']))
            offset = (memory - 1) * n_features
            for m in range(memory):
                A_sol = np.transpose(Theta_sol[j]['At'][
                    offset:offset + n_features])
                self.assertTrue(
                    np.allclose(
                        numerical_sol['A'][j][m], A_sol, 1e-4))
                offset -= n_features

    def test_varx_theta_update_no_u_variable_memory(self):
        rng = np.random.RandomState(seed=0)

        n_samples = 150
        n_features = 2
        n_components = 3
        memory = [2, 0, 1]
        max_memory = np.max(memory)
        n_terms = n_samples - max_memory

        X = rng.rand(n_samples, n_features)
        Gamma = _random_affiliations(
            n_terms, n_components, random_state=rng)

        mu = [rng.uniform(size=(n_features,))
              for i in range(n_components)]
        A = [rng.uniform(size=(memory[i], n_features, n_features))
             if memory[i] > 0 else None
             for i in range(n_components)]

        Theta = [{'mu': mu[i], 'memory': memory[i],
                  'At': np.vstack([np.transpose(A[i][j])
                                   for j in range(memory[i] - 1, -1, -1)])
                  if memory[i] > 0 else None,
                  'Bt': None, 'Sigma': np.identity(n_features)}
                 for i in range(n_components)]

        Theta_sol = _fembv_varx_Theta_update(X, Gamma, Theta, None)
        numerical_sol = _fembv_varx_theta_solution(X, Gamma, mu, A)

        for j in range(n_components):
            self.assertTrue(np.allclose(numerical_sol['mu'][j],
                                        Theta_sol[j]['mu']))

            if memory[j] > 0:
                offset = (memory[j] - 1) * n_features
                for m in range(memory[j]):
                    A_sol = np.transpose(Theta_sol[j]['At'][
                        offset:offset + n_features])
                    self.assertTrue(
                        np.allclose(
                            numerical_sol['A'][j][m], A_sol, 1e-4))
                    offset -= n_features

    def test_varx_theta_update_u_no_memory(self):
        rng = np.random.RandomState(seed=0)

        n_samples = 200
        n_features = 4
        n_components = 3
        memory = 0
        max_memory = np.max(memory)
        n_terms = n_samples - max_memory
        n_external = 3

        X = rng.rand(n_samples, n_features)
        u = rng.rand(n_samples, n_external)
        Gamma = _random_affiliations(
            n_terms, n_components, random_state=rng)

        mu = [rng.uniform(size=(n_features,))
              for i in range(n_components)]
        B = [rng.uniform(size=(n_features, n_external))
             for i in range(n_components)]

        Theta = [{'mu': mu[i], 'memory': memory, 'At': None,
                  'Bt': np.transpose(B[i]),
                  'Sigma': np.identity(n_features)}
                 for i in range(n_components)]

        Theta_sol = _fembv_varx_Theta_update(X, Gamma, Theta, u)
        numerical_sol = _fembv_varx_theta_solution(
            X, Gamma, mu, A=None, B=B, u=u)

        for j in range(n_components):
            self.assertTrue(np.allclose(numerical_sol['mu'][j],
                                        Theta_sol[j]['mu']))

            self.assertTrue(np.allclose(
                numerical_sol['B'][j],
                np.transpose(Theta_sol[j]['Bt']), 1e-4))

    def test_varx_theta_update_u_uniform_memory(self):
        rng = np.random.RandomState(seed=0)

        n_samples = 100
        n_features = 3
        n_components = 3
        memory = 3
        max_memory = np.max(memory)
        n_terms = n_samples - max_memory
        n_external = 2

        X = rng.rand(n_samples, n_features)
        u = rng.rand(n_samples, n_external)
        Gamma = _random_affiliations(
            n_terms, n_components, random_state=rng)

        mu = [rng.uniform(size=(n_features,))
              for i in range(n_components)]
        A = [rng.uniform(size=(memory, n_features, n_features))
             for i in range(n_components)]
        B = [rng.uniform(size=(n_features, n_external))
             for i in range(n_components)]

        Theta = [{'mu': mu[i], 'memory': memory,
                  'At': np.vstack([np.transpose(A[i][j])
                                   for j in range(memory - 1, -1, -1)]),
                  'Bt': np.transpose(B[i]),
                  'Sigma': np.identity(n_features)}
                 for i in range(n_components)]

        Theta_sol = _fembv_varx_Theta_update(X, Gamma, Theta, u)
        numerical_sol = _fembv_varx_theta_solution(
            X, Gamma, mu, A=A, B=B, u=u)

        for j in range(n_components):
            self.assertTrue(np.allclose(numerical_sol['mu'][j],
                                        Theta_sol[j]['mu']))

            offset = (memory - 1) * n_features
            for m in range(memory):
                A_sol = np.transpose(Theta_sol[j]['At'][
                    offset:offset + n_features])
                self.assertTrue(
                    np.allclose(
                        numerical_sol['A'][j][m], A_sol,
                        rtol=1e-3, atol=1e-5))
                offset -= n_features

            self.assertTrue(np.allclose(
                numerical_sol['B'][j],
                np.transpose(Theta_sol[j]['Bt']), 1e-3))

    def test_varx_theta_update_u_variable_memory(self):
        rng = np.random.RandomState(seed=0)

        n_samples = 150
        n_features = 2
        n_components = 3
        memory = [3, 2, 0]
        max_memory = np.max(memory)
        n_terms = n_samples - max_memory
        n_external = 3

        X = rng.rand(n_samples, n_features)
        u = rng.rand(n_samples, n_external)
        Gamma = _random_affiliations(
            n_terms, n_components, random_state=rng)

        mu = [rng.uniform(size=(n_features,))
              for i in range(n_components)]
        A = [rng.uniform(size=(memory[i], n_features, n_features))
             if memory[i] > 0 else None
             for i in range(n_components)]
        B = [rng.uniform(size=(n_features, n_external))
             for i in range(n_components)]

        Theta = [{'mu': mu[i], 'memory': memory[i],
                  'At': np.vstack([np.transpose(A[i][j])
                                   for j in range(memory[i] - 1, -1, -1)])
                  if memory[i] > 0 else None,
                  'Bt': np.transpose(B[i]),
                  'Sigma': np.identity(n_features)}
                 for i in range(n_components)]

        Theta_sol = _fembv_varx_Theta_update(X, Gamma, Theta, u)
        numerical_sol = _fembv_varx_theta_solution(
            X, Gamma, mu, A=A, B=B, u=u)

        for j in range(n_components):
            self.assertTrue(np.allclose(numerical_sol['mu'][j],
                                        Theta_sol[j]['mu']))

            if memory[j] > 0:
                offset = (memory[j] - 1) * n_features
                for m in range(memory[j]):
                    A_sol = np.transpose(Theta_sol[j]['At'][
                        offset:offset + n_features])

                    self.assertTrue(
                        np.allclose(
                            numerical_sol['A'][j][m], A_sol,
                            rtol=1e-3, atol=1e-5))
                    offset -= n_features

            self.assertTrue(np.allclose(
                numerical_sol['B'][j],
                np.transpose(Theta_sol[j]['Bt']), 1e-3))

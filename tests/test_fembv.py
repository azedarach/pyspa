import unittest

import numpy as np

from pyspa.fembv import (_fembv_Gamma_equality_constraints,
                         _fembv_Gamma_upper_bound_constraints)


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

        vec_eq_lhs = np.dot(A_eq, coeffs_vec)
        vec_ub_lhs = np.dot(A_ub, coeffs_vec)

        expected_eq_lhs = np.sum(Gamma, axis=1)

        expected_pos_lhs = np.concatenate([-np.ravel(Gamma), -np.ravel(Eta)])
        expected_bv_lhs = np.sum(Eta, axis=0)
        expected_aux_lhs = np.concatenate(
            [np.ravel(np.diff(Gamma, axis=0) - Eta),
             np.ravel(-np.diff(Gamma, axis=0) - Eta)])
        expected_ub_lhs = np.concatenate(
            [expected_pos_lhs, expected_bv_lhs, expected_aux_lhs])

        self.assertTrue(np.allclose(vec_eq_lhs, expected_eq_lhs))
        self.assertTrue(np.allclose(vec_ub_lhs, expected_ub_lhs))

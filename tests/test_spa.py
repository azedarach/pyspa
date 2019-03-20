import unittest

import numpy as np

from pyspa.spa import (EuclideanSPA,
                       _subspace_update_euclidean_spa_S,
                       _subspace_update_euclidean_spa_Gamma)


def generate_random_dataset(feature_dim, statistics_size):
    return np.random.random((statistics_size, feature_dim))


def generate_random_affiliations(clusters, statistics_size):
    affs = np.random.random((statistics_size, clusters))
    row_sums = np.sum(affs, axis=1)
    return affs / row_sums[:, np.newaxis]


def get_unregularized_states(dataset, affiliations, normalize=True):
    normalization = 1.0
    if normalize:
        T = dataset.shape[0]
        d = dataset.shape[1]
        normalization = 1.0 * T * d

    gtg = np.matmul(np.transpose(affiliations), affiliations) / normalization
    gtgpinv = np.linalg.pinv(gtg)
    gtx = np.matmul(np.transpose(affiliations), dataset) / normalization

    return np.matmul(gtgpinv, gtx)


def get_regularized_states(dataset, affiliations, eps_s_sq, normalize=True):
    k = affiliations.shape[1]
    d = dataset.shape[1]
    normalization = 1.0
    if normalize:
        T = dataset.shape[0]
        normalization = 1.0 * T * d
    if k == 1:
        return get_unregularized_states(dataset, affiliations)
    gtg = np.matmul(np.transpose(affiliations), affiliations) / normalization
    reg = (2 * eps_s_sq * (k * np.identity(k) - np.ones((k, k))) /
           (d * k * (k - 1.0)))
    H_eps = gtg + reg
    H_eps_inv = np.linalg.inv(H_eps)
    gtx = np.matmul(np.transpose(affiliations), dataset) / normalization
    return np.matmul(H_eps_inv, gtx)


def get_two_cluster_affiliations(x, states):
    d = x.shape[1]
    if states.shape[0] != 2:
        raise ValueError("exact expression only holds for K = 2 clusters")
    if states.shape[1] != d:
        raise ValueError("dimension of states must match dimension of data")
    diff = states[0, :] - states[1, :]
    denom = np.dot(diff, diff)
    alpha_1 = np.dot(x - states[1, :], diff) / denom
    alpha_2 = -np.dot(x - states[0, :], diff) / denom
    return np.fmax(0, np.hstack([np.fmin(1, alpha_1[:, np.newaxis]),
                                 np.fmin(1, alpha_2[:, np.newaxis])]))


class TestEuclideanSPA(unittest.TestCase):

    def test_s_subproblem_solution_no_reg(self):
        n = 7
        T = 200
        dataset = generate_random_dataset(n, T)
        max_clusters = 30
        eps_s_sq = 0

        for k in range(1, max_clusters + 1):
            random_affs = generate_random_affiliations(k, T)
            output_states = np.zeros((k, n))
            output_states = _subspace_update_euclidean_spa_S(
                dataset, random_affs, output_states, eps_s_sq)
            expected_states = get_unregularized_states(
                dataset, random_affs)
            self.assertTrue(np.allclose(output_states, expected_states,
                                        rtol=1.e-4, atol=1.e-5))

    def test_s_subproblem_solution_with_reg(self):
        n = 5
        T = 100
        dataset = generate_random_dataset(n, T)
        max_clusters = 30
        eps_s_sq_vals = 10 * np.random.random((5,))

        for k in range(1, max_clusters + 1):
            for eps_s_sq in eps_s_sq_vals:
                random_affs = generate_random_affiliations(k, T)
                output_states = np.zeros((k, n))
                output_states = _subspace_update_euclidean_spa_S(
                    dataset, random_affs, output_states, eps_s_sq)
                expected_states = get_regularized_states(
                    dataset, random_affs, eps_s_sq,
                    normalize=True)
                self.assertTrue(np.allclose(output_states, expected_states,
                                            rtol=1.e-4, atol=1.e-5))

    def test_gamma_subproblem_two_clusters(self):
        n = 5
        T = 10
        np.random.seed(1)
        dataset = generate_random_dataset(n, T)
        eps_s_sq_vals = np.random.random((5,))

        for eps_s_sq in eps_s_sq_vals:
            model = EuclideanSPA(n_components=2, tol=1e-8, max_iter=10000,
                                 epsilon_states=eps_s_sq, verbose=1)
            output_states = np.random.random((2, n))
            output_aff = generate_random_affiliations(2, T)
            output_aff = _subspace_update_euclidean_spa_Gamma(
                dataset, output_aff, output_states,
                tol=1e-6, min_cost_improvement=1)
            exact_aff = get_two_cluster_affiliations(dataset, output_states)
            self.assertTrue(np.allclose(output_aff, exact_aff,
                                        rtol=1.e-2, atol=1.e-2))


if __name__ == "__main__":
    unittest.main()

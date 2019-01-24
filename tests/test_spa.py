import unittest

import numpy as np

from pyspa.spa import SPA2Model

def generate_random_dataset(feature_dim, statistics_size):
    return np.random.random((statistics_size, feature_dim))

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
    reg = (2 * eps_s_sq * (k * np.identity(k) - np.ones((k,k)))
           / (d * k * (k - 1.0)))
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
    diff = states[0,:] - states[1,:]
    denom = np.dot(diff, diff)
    alpha_1 = np.dot(x - states[1,:], diff) / denom
    alpha_2 = -np.dot(x - states[0,:], diff) / denom
    return np.fmax(0, np.hstack([np.fmin(1, alpha_1[:,np.newaxis]),
                                 np.fmin(1, alpha_2[:,np.newaxis])]))

class TestEuclideanSPAModel(unittest.TestCase):

    def test_s_subproblem_solution_no_reg(self):
        n = 7
        T = 200
        dataset = generate_random_dataset(n, T)
        max_clusters = 30
        eps_s_sq = 0

        for k in range(1, max_clusters + 1):
            model = SPA2Model(dataset, k, eps_s_sq=eps_s_sq)
            model.solve_subproblem_s()
            output_states = model.states
            expected_states = get_unregularized_states(
                model.dataset, model.affiliations)
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
                model = SPA2Model(dataset, k, eps_s_sq=eps_s_sq)
                model.solve_subproblem_s()
                output_states = model.states
                expected_states = get_regularized_states(
                    model.dataset, model.affiliations, eps_s_sq,
                    model.normalize)
                self.assertTrue(np.allclose(output_states, expected_states,
                                            rtol=1.e-4, atol=1.e-5))

    def test_gamma_subproblem_two_clusters(self):
        k = 2
        n = 5
        T = 10
        np.random.seed(1)
        dataset = generate_random_dataset(n, T)
        eps_s_sq_vals = np.random.random((5,))

        for eps_s_sq in eps_s_sq_vals:
            model = SPA2Model(dataset, k, eps_s_sq=eps_s_sq)
            model.states = np.random.random((k, n))
            model.solve_subproblem_gamma()
            output_aff = model.affiliations
            exact_aff = get_two_cluster_affiliations(dataset, model.states)
            output_qf =  model.eval_quality_function()
            model.affiliations = exact_aff
            exact_qf =  model.eval_quality_function()
            self.assertTrue(np.allclose(output_aff, exact_aff,
                                        rtol=1.e-5, atol=1.e-5))

if __name__ == "__main__":
    unittest.main()

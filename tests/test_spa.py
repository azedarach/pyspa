import unittest

import numpy as np

from pyspa.spa import SPA2Model

def generate_random_dataset(feature_dim, statistics_size):
    return np.random.random((statistics_size, feature_dim))

def get_unregularized_states(dataset, affiliations):
    gtg = np.matmul(np.transpose(affiliations), affiliations)
    gtgpinv = np.linalg.pinv(gtg)
    gtx = np.matmul(np.transpose(affiliations), dataset)
    return np.matmul(gtgpinv, gtx)

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

    def test_s_subproblem_solution(self):
        n = 5
        T = 100
        dataset = generate_random_dataset(n, T)
        max_clusters = 30
        eps_s_sq_vals = np.random.random((5,))

        for k in range(1, max_clusters + 1):
            for eps_s_sq in eps_s_sq_vals:
                model = SPA2Model(dataset, k, eps_s_sq=eps_s_sq)
                model.solve_subproblem_s()

if __name__ == "__main__":
    unittest.main()

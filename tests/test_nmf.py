import unittest

import numpy as np

from pyspa.nmf import semi_nmf

def random_non_negative_matrix(rows, cols, scale=1.0, normalize_rows=False):
    x = scale * np.random.rand(rows, cols)
    if normalize_rows:
        row_sums = np.sum(x, axis=1)
        x = x / row_sum[:, np.newaxis]
    return x

class TestSemiNMF(unittest.TestCase):

    def test_factorization_non_negative(self):
        n_clusters = [2, 3, 4, 5,]
        feature_dims = [2, 3, 4, 5,]
        n_samples = [5, 10,]

        for d in feature_dims:
            for T in n_samples:
                test_data = np.random.randn(T, d)
                for k in n_clusters:
                    (W, H) = semi_nmf(test_data, k)
                    print("W = ", W)
                    self.assertTrue(np.all(W >= 0))

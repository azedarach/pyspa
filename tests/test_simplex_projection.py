import unittest

import numpy as np

from pyspa.constraints import simplex_projection

class TestSimplexProjection(unittest.TestCase):

    def test_vector_in_simplex_unchanged(self):
        max_dims = 100
        for i in range(1, max_dims + 1):
            test_vec = np.random.random_sample((i,))
            test_vec = test_vec / np.sum(test_vec)
            projected = simplex_projection(test_vec)
            self.assertTrue(np.allclose(test_vec, projected,
                                        rtol=1.e-10, atol=1.e-10))

    def test_projected_vector_in_simplex(self):
        max_dims = 100
        min_elem = -100
        max_elem = 100
        for i in range(1, max_dims + 1):
            test_vec = ((max_elem - min_elem) * np.random.random_sample((i,))
                        + min_elem)
            projected = simplex_projection(test_vec)
            self.assertTrue(np.all(projected >= 0))
            self.assertTrue(np.abs(np.sum(projected) - 1.) < 1.e-12)

if __name__ == "__main__":
    unittest.main()

import unittest

from math import pi

import numpy as np

from pyspa.constraints import rectangle_projection
from pyspa.optimizers import spg, spgqp, spg_qp, quadprog_solvers

def extended_rosenbrock_f(x):
    return (100 * np.sum((x[1:] - x[:-1]**2) ** 2) + np.sum((1 - x[:-1]) ** 2))

def extended_rosenbrock_df(x):
    grad = np.zeros(np.size(x))
    grad[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    grad[1:-1] = (200 * (x[1:-1] - x[:-2] ** 2)
                   - 400 * x[1:-1] * (x[2:] - x[1:-1] ** 2)
                   - 2 * (1 - x[1:-1]))
    grad[-1] = 200 * (x[-1] - x[-2] ** 2)
    return grad

def wood_f(x):
    if np.size(x) != 4:
        raise ValueError("input vector must be 4-dimensional")
    return (100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2
            + (x[2] - 1) ** 2 + 90 * (x[2] ** 2 - x[3]) ** 2
            + 10.1 * ((x[1] - 1) ** 2 + (x[3] - 1)**2)
            + 19.8 * (x[1] - 1) * (x[3] - 1))

def wood_df(x):
    if np.size(x) != 4:
        raise ValueError("input vector must be 4-dimensional")
    gradient = np.zeros(4)
    gradient[0] = 400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1)
    gradient[1] = (-200 * (x[0] ** 2 - x[1]) + 20.2 * (x[1] - 1)
                   + 19.8 * (x[3] - 1))
    gradient[2] = 2 * (x[2] - 1) + 360 * (x[2] ** 2 - x[3]) * x[2]
    gradient[3] = (-180 * (x[2] ** 2 - x[3]) + 20.2 * (x[3] - 1)
                   + 19.8 * (x[1] - 1))
    return gradient

def powell_singular_f(x):
    if np.size(x) != 4:
        raise ValueError("input vector must be 4-dimensional")
    return ((x[0] + 10 * x[1]) ** 2 + 5 * (x[2] - x[3]) ** 2
            + (x[1] - 2 * x[2]) ** 4 + 10 * (x[0] - x[3]) ** 4)

def powell_singular_df(x):
    if np.size(x) != 4:
        raise ValueError("input vector must be 4-dimensional")
    gradient = np.zeros(4)
    gradient[0] = 2 * (x[0] + 10 * x[1]) + 40 * (x[0] - x[3]) ** 3
    gradient[1] = 20 * (x[0] + 10 * x[1]) + 4 * (x[1] - 2 * x[2]) ** 3
    gradient[2] = 10 * (x[2] - x[3]) - 8 * ( x[1] - 2 * x[2]) ** 3
    gradient[3] = -10 * (x[2] - x[3]) - 40 * (x[0] - x[3]) ** 3
    return gradient

def cube_f(x):
    if np.size(x) != 2:
        raise ValueError("input vector must be 2-dimensional")
    return 100 * (x[1] - x[0] ** 3) ** 2 + (1 - x[0]) ** 2

def cube_df(x):
    if np.size(x) != 2:
        raise ValueError("input vector must be 2-dimensional")
    gradient = np.zeros(2)
    gradient[0] = -600 * x[0] ** 2 * (x[1] - x[0] ** 3) - 2 * (1 - x[0])
    gradient[1] = 200 * (x[1] - x[0] ** 3)
    return gradient

def trig_f(x):
    n = np.size(x)
    sum_cos = np.sum(np.cos(x))
    return np.sum((n - sum_cos - np.sin(x)
                   + np.linspace(1, n, n) * (1 - np.cos(x))) ** 2)

def trig_df(x):
    n = np.size(x)
    sum_cos = np.sum(np.cos(x))
    terms = n + np.linspace(1, n, n) * (1 - np.cos(x)) - np.sin(x) - sum_cos
    sum_terms = np.sum(terms)
    gradient = (2 * (np.linspace(1, n, n) * np.sin(x) - np.cos(x)) * terms
                + 2 * sum_terms * np.sin(x))
    return gradient

def helical_valley_f(x):
    if np.size(x) != 3:
        raise ValueError("input vector must be 3-dimensional")
    if x[0] > 0:
        theta = np.arctan(x[1] / x[0]) / (2 * pi)
    else:
        theta = (pi + np.arctan(x[1] / x[0])) / (2 * pi)
    return (100 * ((x[2] - 10 * theta) ** 2
                   + (np.sqrt(x[0] ** 2 + x[1] ** 2) - 1) ** 2)
            + x[2] ** 2)

def helical_valley_df(x):
    if np.size(x) != 3:
        raise ValueError("input vector must be 3-dimensional")
    if x[0] > 0:
        theta = np.arctan(x[1] / x[0]) / (2 * pi)
    else:
        theta = (pi + np.arctan(x[1] / x[0])) / (2 * pi)
    dthetadx0 = -x[0] * x[1] / (2 * pi * (x[0] ** 2 + x[1] ** 2))
    dthetadx1 = x[0] / (2 * pi * (x[0] ** 2 + x[1] ** 2))
    gradient = np.zeros(3)
    gradient[0] = (-2000 * (x[2] - 10 * theta) * dthetadx0
                   + 200 * (np.sqrt(x[0] ** 2 + x[1] ** 2) - 1)
                    * x[0] / np.sqrt(x[0] ** 2 + x[1] ** 2))
    gradient[1] = (-2000 * (x[2] - 10 * theta) * dthetadx1
                   + 200 * (np.sqrt(x[0] ** 2 + x[1] ** 2) - 1)
                   * x[1] / np.sqrt(x[0] ** 2 + x[1] ** 2))
    gradient[2] = 200 * (x[2] - 10 * theta) + 2 * x[2]
    return gradient

class TestSPG(unittest.TestCase):

    def test_extended_rosenbrock(self):
        max_dim = 15
        for n in range(2, max_dim + 1):
            lower_bounds = -2 * np.ones(n)
            upper_bounds = 2 * np.ones(n)
            proj = lambda x : rectangle_projection(x, lower_bounds,
                                                   upper_bounds)
            f = extended_rosenbrock_f
            df = extended_rosenbrock_df
            x0 = np.ones(n)
            x0[::2] = -1.2
            expected = np.ones(n)
            result = spg(f, df, x0, projector=proj)
            self.assertTrue(np.allclose(result, expected,
                                        rtol=1.e-5, atol=1.e-5))

    def test_wood_function(self):
        lower_bounds = -5 * np.ones(4)
        upper_bounds = 7 * np.ones(4)
        proj = lambda x : rectangle_projection(x, lower_bounds,
                                               upper_bounds)
        f = wood_f
        df = wood_df
        x0 = np.array([3., -1., -3., -1.])
        expected = np.ones(4)
        result = spg(f, df, x0, projector=proj)
        self.assertTrue(np.allclose(result, expected,
                                    rtol=1.e-5, atol=1.e-5))

    def test_powell_singular(self):
        lower_bounds = -10 * np.ones(4)
        upper_bounds = 10 * np.ones(4)
        proj = lambda x : rectangle_projection(x, lower_bounds,
                                               upper_bounds)
        f = powell_singular_f
        df = powell_singular_df
        x0 = np.array([3., -1., 0., 1.])
        expected = np.zeros(4)
        result = spg(f, df, x0, projector=proj)
        self.assertTrue(np.allclose(result, expected,
                                    rtol=1.e-2, atol=1.e-2))

    def test_cube_function(self):
        lower_bounds = -20 * np.ones(2)
        upper_bounds = 40 * np.ones(2)
        proj = lambda x : rectangle_projection(x, lower_bounds,
                                               upper_bounds)
        f = cube_f
        df = cube_df
        x0 = np.array([-1.2, -1.])
        expected = np.ones(2)
        result = spg(f, df, x0, projector=proj)
        self.assertTrue(np.allclose(result, expected,
                                    rtol=1.e-5, atol=1.e-5))

    def test_trig_f(self):
        max_dim = 15
        for n in range(2, max_dim + 1):
            lower_bounds = -3 * np.ones(n)
            upper_bounds = 3 * np.ones(n)
            proj = lambda x : rectangle_projection(x, lower_bounds,
                                                   upper_bounds)
            f = trig_f
            df = trig_df
            x0 = np.ones(n) / (5.0 * n)
            expected = np.zeros(n)
            result = spg(f, df, x0, projector=proj)
            self.assertTrue(np.allclose(result, expected,
                                        rtol=1.e-5, atol=1.e-5))

class TestSPGQP(unittest.TestCase):

    def test_simple_quadratic(self):
        max_dims = 30
        for n in range(1, max_dims):
            lower_bounds = -100 * np.ones(n)
            upper_bounds = 100 * np.ones(n)
            proj = lambda x : rectangle_projection(x, lower_bounds,
                                                   upper_bounds)
            p = np.identity(n)
            q = np.zeros(n)
            evals = np.linalg.eigvals(p)
            alpha_max = 1.0 / np.max(evals)

            x0 = np.ones(n)
            expected = np.zeros(n)
            result = spgqp(p, q, x0, alpha_max=alpha_max,
                           projector=proj)
            self.assertTrue(np.allclose(result, expected,
                                        rtol=1.e-5, atol=1.e-5))

class TestSPGQPWrapper(unittest.TestCase):

    def test_simple_quadratic(self):
        max_dims = 30
        for n in range(1, max_dims):
            lower_bounds = -100 * np.ones(n)
            upper_bounds = 100 * np.ones(n)
            proj = lambda x : rectangle_projection(x, lower_bounds,
                                                   upper_bounds)
            p = np.identity(n)
            q = np.zeros(n)
            x0 = np.ones(n)
            expected = np.zeros(n)
            result = spg_qp(p, q, x0, projector=proj)
            self.assertTrue(np.allclose(result, expected,
                                        rtol=1.e-5, atol=1.e-5))

if "cvxopt_qp" in quadprog_solvers:
    from pyspa.optimizers import cvxopt_qp

@unittest.skipIf("cvxopt_qp" not in quadprog_solvers, "CVXOPT not available")
class TestCVXOPTQP(unittest.TestCase):

    def test_simple_quadratic(self):
        max_dims = 30
        for n in range(1, max_dims):
            p = np.identity(n)
            q = np.zeros(n)
            x0 = np.ones(n)
            expected = np.zeros(n)
            result = cvxopt_qp(p, q)
            self.assertTrue(np.allclose(result, expected,
                                        rtol=1.e-5, atol=1.e-5))

if __name__ == "__main__":
    unittest.main()

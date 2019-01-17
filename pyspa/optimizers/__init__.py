import numpy as np

from .spg import spg
from .spgqp import spgqp

quadprog_solvers = ["spg", "spgqp"]

def spg_qp(p, q, x0, **kwargs):
    objective = lambda x : (0.5 * np.matmul(np.transpose(x),
                                            np.matmul(p, x))
                            + np.dot(q, x))
    gradient = lambda x : np.matmul(p, x) + q
    return spg(objective, gradient, x0, **kwargs)

try:
    from .cvxopt_interface import cvxopt_qp
    quadprog_solvers.append("cvxopt_qp")
except ImportError:
    def cvxopt_qp(p, q):
        raise ImportError("CVXOPT could not be imported")

def solve_qp(p, q, qpsolver="spgqp", **kwargs):
    if qpsolver == "spgqp":
        if "x0" not in kwargs:
            raise ValueError("spgqp solver requires initial guess x0")
        return spgqp(p, q, kwargs.get("x0"), **kwargs)
    elif qpsolver == "spg":
        if "x0" not in kwargs:
            raise ValueError("spg solver requires initial guess x0")
        return spg_qp(p, q, kwargs.get("x0"), **kwargs)
    elif qpsolver == "cvxopt":
        return cvxopt_qp(p, q, **kwargs)
    else:
        raise ValueError("unrecognized solver '" + solver + "'")

__all__ = ["spg", "spgqp", "solve_qp", "cvxopt_qp", "quadprog_solvers"]

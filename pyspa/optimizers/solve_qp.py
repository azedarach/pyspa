from .spg import spg
from .spgqp import spgqp

def solve_qp(p, q, solver="spgqp", **kwargs):
    if solver == "spgqp":
        if "x0" not in kwargs:
            raise ValueError("spgqp solver requires initial guess 'x0'")
        return spgqp(p, q, kwargs.get("x0"), **kwargs)
    elif solver == "spg":
        if "x0" not in kwargs:
            raise ValueError("spg solver requires initial guess 'x0'")
        objective = lambda x : return (0.5 * np.matmul(np.transpose(x),
                                                       np.matmul(p, x))
                                       + np.dot(q, x))
        gradient = lambda x : return np.matmul(p, x) + q
        return spg(objective, gradient, kwargs.get("x0"), **kwargs)
    elif solver == "cvxopt_qp":
        import cvxopt.solvers
        return cvxopt.solvers.qp(p, q, **kwargs)
    else:
        raise ValueError("unrecognized solver name '" + solver + "'")

from cvxopt import matrix, spmatrix
from cvxopt.solvers import options, qp

from numpy import asarray, ndarray, ravel

options["show_progress"] = False

def to_cvxopt_matrix(m):
    if type(m) is ndarray:
        return matrix(m)
    elif type(m) is matrix or type(m) is spmatrix:
        return m

def cvxopt_qp(p, q, G=None, h=None, A=None, b=None,
              solver=None, kktsolver=None, initvals=None, **kwargs):
    p_cvxopt = to_cvxopt_matrix(p)
    q_cvxopt = to_cvxopt_matrix(q)
    if G is not None:
        G_cvxopt = to_cvxopt_matrix(G)
    else:
        G_cvxopt = G
    if h is not None:
        h_cvxopt = to_cvxopt_matrix(h)
    else:
        h_cvxopt = h
    if A is not None:
        A_cvxopt = to_cvxopt_matrix(A)
    else:
        A_cvxopt = A
    if b is not None:
        b_cvxopt = to_cvxopt_matrix(b)
    else:
        b_cvxopt = b

    sol = qp(p_cvxopt, q_cvxopt, G_cvxopt, h_cvxopt, A_cvxopt, b_cvxopt,
              solver, kktsolver, initvals, **kwargs)

    if sol["status"] != "optimal":
        return None
    else:
        return sol["x"]

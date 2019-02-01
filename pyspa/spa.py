import numpy as np

from multiprocessing import Pool
from scipy.optimize import minimize

from .constraints import simplex_projection
from .optimizers import spg, solve_qp

def delta_convergence(old_qf, new_qf, abs_tol=1.e-5):
    if abs_tol <= 0.:
        raise ValueError("absolute tolerance must be positive")
    if old_qf is None:
        return False
    delta_qf = np.abs(old_qf - new_qf)
    return delta_qf < abs_tol

def euclidean_spa_states_reg(states):
    (k, d) = states.shape

    if k == 1:
        return 0

    reg = (k * np.trace(np.transpose(states) @ states)
           - np.sum(states @ np.transpose(states)))

    prefactor = 2.0 / (k * d * (k - 1.0))

    return prefactor * reg

def euclidean_spa_dist(dataset, affiliations, states, normalization=1.0):
    errs = dataset - affiliations @ states
    return np.linalg.norm(errs) ** 2 / normalization

def exact_euclidean_spa_states(gtx, gtg, eps_s_sq):
    (k, d) = gtx.shape

    prefactor = 2.0 * eps_s_sq / (k * d)
    if k > 1:
        prefactor *= 1.0 / (k - 1.0)

    reg = prefactor * (k * np.identity(k) - np.ones((k, k)))

    H_eps = gtg + reg
    H_eps_inv = np.linalg.inv(H_eps)

    return H_eps_inv @ gtx

def build_euclidean_states_f(dataset, affiliations, eps_s_sq,
                             normalization):
    d = dataset.shape[1]
    k = affiliations.shape[1]

    gtx = ((np.transpose(affiliations) @ dataset) / normalization)
    gtg = ((np.transpose(affiliations) @ affiliations) / normalization)

    s_vec_dim = k * d

    q = -2 * np.reshape(gtx, (s_vec_dim,))

    reg_prefactor = 0.0
    if eps_s_sq > 0.0 and k > 1:
        reg_prefactor = 4.0 * eps_s_sq / (k * d * (k - 1.0))

    P = np.zeros((s_vec_dim, s_vec_dim))
    for i in range(k):
        for j in range(k):
            diag_reg = 0.0
            if i == j:
                diag_reg = k
            for diag in range(d):
                reg = reg_prefactor * (diag_reg - 1.0)
                P[i * d + diag, j * d + diag] = 2.0 * gtg[i,j] + reg

    return (P, q)

def eval_euclidean_states_f(P, q, states_vec):
    return (0.5 * (np.transpose(states_vec) @ (P @ states_vec))
            + q.dot(states_vec))

def eval_euclidean_states_df(P, q, states_vec):
    return P @ states_vec + q

def solve_euclidean_states_subproblem(dataset, affiliations, states, eps_s_sq,
                                      normalization=1.0,
                                      use_exact_states=False, solver="BFGS",
                                      solution_tol=1.e-5):
    (k, d) = states.shape

    s_guess = np.ravel(states)
    (P, q) = build_euclidean_states_f(dataset, affiliations, eps_s_sq,
                                      normalization)

    f = lambda s : eval_euclidean_states_f(P, q, s)
    df = lambda s : eval_euclidean_states_df(P, q, s)

    if solver == "spg":
        s_soln = spg(f, df, x0=s_guess, tol=solution_tol)
    elif solver == "cvxopt":
        s_soln = solve_qp(P, q, tol=solution_tol, qpsolver="cvxopt")
    elif solver in ("BFGS"):
        res = minimize(f, s_guess, method=solver,
                       jac=df, tol=solution_tol)
        s_soln = res.x
    else:
        raise ValueError("unrecognized solver")

    if s_soln is None:
        raise RuntimeError("failed to solve S subproblem")

    return np.reshape(s_soln, ((k, d)))

def calculate_affiliation_vector(args):
    if args["solver"] == "spgqp":
        return solve_qp(
            args["P"], args["q"], x0=args["x0"],
            tol=args["tol"], qpsolver="spgqp",
            projector=simplex_projection)
    elif args["solver"] == "cvxopt":
        clusters = np.size(args["x0"])
        return solve_qp(args["P"], args["q"], tol=args["tol"],
                        qpsolver="cvxopt", A=np.ones((1,clusters)),
                        b=np.ones((1,1)), G=-np.identity(clusters),
                        h=np.zeros((clusters,1)))
    else:
        raise RuntimeError("unrecognized solver")

def euclidean_spa_affiliations_trial_step(P, q, affiliations,
                                          normalization=1.0):
    (evals, _) = np.linalg.eig(normalization * P)
    alpha_try = 1.0 / np.max(np.abs(evals))
    grad = normalization * (affiliations @ P + q)
    return simplex_projection(affiliations - alpha_try * grad)

def solve_euclidean_gamma_subproblem(dataset, affiliations, states,
                                     normalization=1.0, use_trial_step=False,
                                     trial_step_tol=1.e-10, solver="spgqp",
                                     solution_tol=1.e-5,
                                     max_processes=8):
    T = dataset.shape[0]

    q = (-2 * (dataset @ np.transpose(states)) / normalization)
    P = (2 * (states @ np.transpose(states)) / normalization)

    gamma = np.zeros(affiliations.shape)
    if use_trial_step:
        initial_qf = euclidean_spa_dist(dataset, affiliations, states,
                                        normalization)
        gamma = euclidean_spa_affiliations_trial_step(P, q, affiliations,
                                                      normalization)
        trial_qf = euclidean_spa_dist(dataset, gamma, states, normalization)

        if np.abs(trial_qf - initial_qf) > trial_step_tol:
            return gamma

    optim_args = ({"P": P, "q": q[i,:], "x0": affiliations[i,:],
                   "tol": solution_tol, "solver": solver}
                  for i in range(T))
    with Pool(processes=max_processes) as pool:
        res = pool.imap(calculate_affiliation_vector, optim_args)
        for i in range(T):
            r = next(res)
            if r is None:
                raise RuntimeError("failed to solve Gamma subproblem")
            else:
                gamma[i,:] = np.ravel(r)

    return gamma

def run_euclidean_spa(dataset, n_clusters, eps_s_sq, initial_affiliations,
                      initial_states=None, normalize=True,
                      max_iterations=500,
                      convergence_checker=delta_convergence,
                      use_exact_states_solution=False,
                      states_solver="cvxopt",
                      use_affiliations_trial_step=False,
                      affiliations_solver="spgqp"):
    if dataset.ndim != 2:
        raise ValueError(
            "the input dataset must be two dimensional")

    if eps_s_sq < 0:
        raise ValueError(
            "regularization parameter must be non-negative")

    if n_clusters < 1:
        raise ValueError(
            "the number of clusters must be at least one")

    (T, d) = dataset.shape

    (gamma_T, k) = initial_affiliations.shape

    if gamma_T != T:
        raise ValueError(
            "initial affiliations must have same number of rows as input data")

    normalization = 1.0
    if normalize:
        normalization *= T * d

    dist = lambda g, s : euclidean_spa_dist(dataset, g, s, normalization)

    if eps_s_sq == 0.:
        reg = lambda s : 0
    else:
        reg = lambda s : eps_s_sq * euclidean_spa_states_reg(s)

    qf = lambda g, s : dist(g, s) + reg(s)

    affiliations = initial_affiliations
    if initial_states is not None:
        if initial_states.shape != (k, d):
            raise ValueError("incorrect shape for initial states")
        states = initial_states
    else:
        states = np.zeros((k, d))

    qf_old = None
    qf_new = qf(affiliations, states)

    iters = 0
    is_converged = convergence_checker(qf_old, qf_new)
    while (not is_converged and iters < max_iterations):
        qf_old = qf_new

        states = solve_euclidean_states_subproblem(
            dataset, affiliations, states,
            eps_s_sq, normalization=normalization,
            use_exact_states=use_exact_states_solution,
            solver=states_solver)

        affiliations = solve_euclidean_gamma_subproblem(
            dataset, affiliations, states,
            normalization=normalization,
            use_trial_step=use_affiliations_trial_step,
            solver=affiliations_solver)

        qf_new = qf(affiliations, states)
        iters += 1
        is_converged = convergence_checker(qf_old, qf_new)

    if (iters == max_iterations and not is_converged):
        raise RuntimeError("failed to converge")

    return {"affiliations": affiliations, "states": states,
            "quality_func": qf_new, "iterations": iters}

class EuclideanSPAModel(object):
    def __init__(self, dataset, clusters,
                 eps_s_sq=0, normalize=True,
                 stopping_tol=1.e-5, max_iterations=500,
                 gamma_solver="spgqp",
                 verbose=False, use_exact_states=False,
                 use_trial_step=False):

        self.dataset = dataset.copy()
        (self.statistics_size, self.feature_dim) = self.dataset.shape

        self.clusters = clusters
        self.eps_s_sq = eps_s_sq

        self.reg_norm = eps_s_sq / (self.feature_dim * self.clusters)
        if self.clusters > 1:
            self.reg_norm = self.reg_norm / (self.clusters - 1.0)

        self.normalize = normalize
        if self.normalize:
            self.normalization = 1.0 * self.statistics_size * self.feature_dim
        else:
            self.normalization = 1.0

        self.stopping_tol = stopping_tol
        self.max_iterations = max_iterations
        self.gamma_solver = gamma_solver.lower()

        self.verbose = verbose
        self.use_exact_states = use_exact_states
        self.use_trial_step = use_trial_step

        self.affiliations = np.random.random(
            (self.statistics_size, self.clusters))
        row_sums = np.sum(self.affiliations, axis=1)
        self.affiliations = np.divide(self.affiliations,
                                      row_sums[:, np.newaxis])

        self.states = np.zeros((self.clusters, self.feature_dim))

    def distance(self):
        return euclidean_spa_dist(self.dataset, self.affiliations,
                                  self.states, self.normalization)

    def states_regularization(self):
        return self.eps_s_sq * euclidean_spa_states_reg(self.states)

    def eval_quality_function(self):
        return self.distance() + self.states_regularization()

    def solve_subproblem_s(self):
        if self.verbose:
            initial_qf = self.eval_quality_function()
            print("\tInitial L = " + str(self.eval_quality_function()))

        self.states = solve_euclidean_states_subproblem(
            self.dataset, self.affiliations, self.states, self.eps_s_sq,
            self.normalization, self.use_exact_states)

        if self.verbose:
            updated_qf = self.eval_quality_function()
            print("\tFinal L = " + str(updated_qf))
            if updated_qf > initial_qf:
                print("\tWARNING: quality function increased")
            print("Successfully solved S subproblem")

    def solve_subproblem_gamma(self):
        if self.verbose:
            initial_qf = self.eval_quality_function()
            print("\tInitial L = " + str(self.eval_quality_function()))

        self.affiliations = solve_euclidean_gamma_subproblem(
            self.dataset, self.affiliations, self.states,
            normalization=self.normalization,
            use_trial_step=self.use_trial_step,
            solver=self.gamma_solver)

        if self.verbose:
            updated_qf = self.eval_quality_function()
            print("\tFinal L = " + str(updated_qf))
            if updated_qf > initial_qf:
                print("\tWARNING: quality function increased")
            print("Successfully solved Gamma subproblem")

    def is_converged(self, old_qf, new_qf):
        if old_qf is None:
            return False
        delta_qf = np.abs(old_qf - new_qf)

        if np.abs(old_qf) > np.abs(new_qf):
            min_qf = new_qf
            max_qf = old_qf
        else:
            min_qf = old_qf
            max_qf = new_qf

        r_qf = 1.0 - np.abs(min_qf / max_qf)

        return delta_qf < self.stopping_tol or r_qf < self.stopping_tol

    def find_optimal_approx(self, initial_affs, initial_states=None):
        self.affiliations = initial_affs

        if initial_states is not None:
            self.states = initial_states

        check_convergence = lambda o, n : self.is_converged(o, n)

        result = run_euclidean_spa(
            self.dataset, self.clusters, self.eps_s_sq,
            self.affiliations,
            initial_states=self.states,
            normalize=self.normalize,
            max_iterations=self.max_iterations,
            convergence_checker=check_convergence,
            use_exact_states_solution=self.use_exact_states,
            use_affiliations_trial_step=self.use_trial_step,
            affiliations_solver=self.gamma_solver)

        self.affiliations = result["affiliations"]
        self.states = result["states"]

class SimEuclideanSPAModel(object):
    def __init__(self, x_dataset, y_dataset, x_clusters, y_clusters,
                 rel_weight=1.0, stopping_tol=1.e-5, max_iterations=500,
                 gamma_solver="spgqp",
                 verbose=False, use_trial_step=False, use_exact_probs=False):

        self.x_clusters = x_clusters
        self.y_clusters = y_clusters
        self.rel_weight = rel_weight

        self.x_dataset = x_dataset.copy()
        self.y_dataset = y_dataset.copy()

        (x_statistics_size, self.x_feature_dim) = self.x_dataset.shape
        (y_statistics_size, self.y_feature_dim) = self.y_dataset.shape

        if x_statistics_size != y_statistics_size:
            raise ValueError(
                "number of points in each dataset must be the same")

        self.statistics_size = x_statistics_size

        self.stopping_tol = stopping_tol
        self.max_iterations = max_iterations
        self.gamma_solver = gamma_solver.lower()

        self.verbose = verbose
        self.use_trial_step = use_trial_step
        self.use_exact_probs = use_exact_probs

        self.x_affiliations = np.zeros((self.statistics_size,
                                        self.x_clusters))
        self.y_affiliations = np.zeros((self.statistics_size,
                                        self.y_clusters))

        self.x_states = np.zeros((self.x_clusters, self.x_feature_dim))
        self.y_states = np.zeros((self.y_clusters, self.y_feature_dim))

    def distance(self):
        y_dist = euclidean_spa_dist(self.y_dataset, self.y_affiliations,
                                   self.y_states)
        x_dist = euclidean_spa_dist(self.x_dataset, self.x_affiliations,
                                    self.x_states)
        return y_dist + self.rel_weight ** 2.0 * x_dist

    def eval_quality_function(self):
        return self.distance()

    def solve_subproblem_s(self):
        if self.verbose:
            initial_qf = self.eval_quality_function()
            print("\tInitial L = " + str(self.eval_quality_function()))

        self.y_states = solve_euclidean_states_subproblem(
            self.y_dataset, self.y_affiliations, self.y_states, eps_s_sq=0.0,
            solver="BFGS")
        self.x_states = solve_euclidean_states_subproblem(
            self.x_dataset, self.x_affiliations, self.x_states, eps_s_sq=0.0,
            solver="BFGS")

        if self.verbose:
            updated_qf = self.eval_quality_function()
            print("\tFinal L = " + str(updated_qf))
            if updated_qf > initial_qf:
                print("\tWARNING: quality function increased")
            print("Successfully solved S subproblem")

    def solve_subproblem_gamma(self):
        if self.verbose:
            initial_qf = self.eval_quality_function()
            print("\tInitial L = " + str(self.eval_quality_function()))

        self.y_affiliations = solve_euclidean_gamma_subproblem(
            self.y_dataset, self.y_affiliations, self.y_states,
            solver=self.gamma_solver)
        self.x_affiliations = solve_euclidean_gamma_subproblem(
            self.rel_weight * self.x_dataset, self.x_affiliations,
            self.rel_weight * self.x_states, solver=self.gamma_solver)

        if self.verbose:
            updated_qf = self.eval_quality_function()
            print("\tFinal L = " + str(updated_qf))
            if updated_qf > initial_qf:
                print("\tWARNING: quality function increased")
            print("Successfully solved Gamma subproblem")

    def find_optimal_approx(self, initial_x_affs, initial_y_affs,
                            initial_x_states=None, initial_y_states=None):
        if initial_x_affs.shape != self.x_affiliations.shape:
            raise ValueError(
                "initial guess for X affiliations has incorrect shape")

        if initial_y_affs.shape != self.y_affiliations.shape:
            raise ValueError(
                "initial guess for Y affiliations has incorrect shape")

        self.x_affiliations = initial_x_affs
        self.y_affiliations = initial_y_affs

        if initial_x_states is not None:
            if initial_x_states.shape != self.x_states.shape:
                raise ValueError(
                    "initial guess for X states has incorrect shape")
            self.x_states = initial_x_states

        if initial_y_states is not None:
            if initial_y_states.shape != self.y_states.shape:
                raise ValueError(
                    "initial guess for Y states has incorrect shape")
            self.y_states = initial_y_states

        qf_old = None
        qf_new = self.eval_quality_function()
        delta_qf = 1e10 + self.stopping_tol
        iters = 0
        if self.verbose:
            print("Iterating with stopping tolerance = "
                  + str(self.stopping_tol)
                  + " and max. iterations = " + str(self.max_iterations))
        while (not self.is_converged(qf_old, qf_new)
               and iters < self.max_iterations):
            qf_old = qf_new

            if self.verbose:
                print("Iteration = " + str(iters))
                print("Solving S subproblem ...")
            self.solve_subproblem_s()

            if self.verbose:
                print("Solving Gamma subproblem ...")
            self.solve_subproblem_gamma()

            qf_new = self.eval_quality_function()
            iters += 1

        if (iters == self.max_iterations
            and not self.is_converged(qf_old, qf_new)):
            raise RuntimeError(
                "failed to converge")

    def is_converged(self, old_qf, new_qf):
        if old_qf is None:
            return False
        delta_qf = np.abs(old_qf - new_qf)

        if np.abs(old_qf) > np.abs(new_qf):
            min_qf = new_qf
            max_qf = old_qf
        else:
            min_qf = old_qf
            max_qf = new_qf

        r_qf = 1.0 - np.abs(min_qf / max_qf)

        return delta_qf < self.stopping_tol or r_qf < self.stopping_tol

    def get_transition_probabilities(self):
        gxtgy = np.matmul(np.transpose(self.x_affiliations),
                          self.y_affiliations)
        gxtgx = np.matmul(np.transpose(self.x_affiliations),
                          self.x_affiliations)

        if self.use_exact_probs:
            return np.matmul(np.linalg.pinv(gxtgx), gxtgy)

        lambda_vec_dim = self.x_clusters * self.y_clusters
        lambda_guess = np.ravel(gxtgy)

        q = -2 * np.reshape(gxtgy, (lambda_vec_dim,))
        p_blocks = [[2 * gxtgx[i,j] * np.identity(self.y_clusters)
                     for j in range(self.x_clusters)]
                    for i in range(self.x_clusters)]
        P = np.block(p_blocks)

        if self.gamma_solver == "spgqp":
            lambda_sol = solve_qp(
                P, q, x0=lambda_guess, tol=1.e-4,
                qpsolver="spgqp", projector=simplex_projection)
        elif self.gamma_solver == "cvxopt":
            A = np.zeros((self.x_clusters, lambda_vec_dim))
            for i in range(self.x_clusters):
                A[i,i * self.y_clusters:(i+1)*self.y_clusters] = np.ones(
                    (1, self.y_clusters))
            lambda_sol = solve_qp(
                P, q, tol=1.e-5,
                qpsolver="cvxopt", A=A,
                b=np.ones((self.x_clusters,1)), G=-np.identity(lambda_vec_dim),
                h=np.zeros((lambda_vec_dim,1)))
        else:
            raise RuntimeError("unrecognized solver for Gamma subproblem")

        if lambda_sol is None:
            raise RuntimeError("unable to solve for transition probabilities")

        return np.reshape(lambda_sol, ((self.x_clusters, self.y_clusters)))

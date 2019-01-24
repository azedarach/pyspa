import numpy as np

from .constraints import simplex_projection
from .optimizers import solve_qp

class EuclideanSPAModel(object):
    def __init__(self, dataset, clusters, affiliations=None,
                 eps_s_sq=0, normalize=True,
                 stopping_tol=1.e-5, max_iterations=500,
                 gamma_solver="spgqp",
                 verbose=False, use_exact_states=False,
                 use_trial_step=False):

        if dataset.ndim != 2:
            raise ValueError(
                "the input dataset must be two dimensional")

        if eps_s_sq < 0:
            raise ValueError(
                "regularization parameter must be non-negative")

        if clusters < 1:
            raise ValueError(
                "the number of clusters must be at least one")

        self.dataset = dataset.copy()
        self.statistics_size = self.dataset.shape[0]
        self.feature_dim = self.dataset.shape[1]

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

        if affiliations is not None:
            if affiliations.ndim != 2:
                raise ValueError(
                    "the initial affiliations must be two-dimensional")
            if affiliations.shape[0] != self.statistics_size:
                raise ValueError(
                    "affiliations must be provided for all points")
            if not np.all(affiliations >= 0):
                raise ValueError(
                    "affiliations must be non-negative")
            aff_sums = np.sum(affiliations, axis=1)
            if not np.allclose(aff_sums, np.ones(self.feature_dim)):
                raise ValueError(
                    "affiliations at each time must sum to unity")
            self.affiliations = affiliations
        else:
            self.affiliations = np.random.random(
                (self.statistics_size, self.clusters))
            row_sums = np.sum(self.affiliations, axis=1)
            self.affiliations = np.divide(self.affiliations,
                                          row_sums[:, np.newaxis])

        self.states = np.zeros((self.clusters, self.feature_dim))

    def distance(self):
        errs = self.dataset - np.matmul(self.affiliations, self.states)
        return np.linalg.norm(errs) ** 2 / self.normalization

    def states_regularization(self):
        if self.clusters == 1:
            return 0

        reg = (self.clusters * np.trace(np.matmul(
            np.transpose(self.states), self.states))
               - np.sum(np.matmul(self.states, np.transpose(self.states))))

        return 2 * self.reg_norm * reg

    def eval_quality_function(self):
        return self.distance() + self.states_regularization()

    def solve_subproblem_s(self):
        if self.verbose:
            initial_qf = self.eval_quality_function()
            print("\tInitial L = " + str(self.eval_quality_function()))

        gtx = (np.matmul(np.transpose(self.affiliations), self.dataset)
               / self.normalization)
        gtg = (np.matmul(np.transpose(self.affiliations), self.affiliations)
               / self.normalization)

        if self.use_exact_states:
            reg = (2 * self.reg_norm *
                   (self.clusters * np.identity(self.clusters)
                    - np.ones((self.clusters, self.clusters))))
            H_eps = gtg + reg
            H_eps_inv = np.linalg.inv(H_eps)
            self.states = np.matmul(H_eps_inv, gtx)
            return

        s_vec_dim = self.clusters * self.feature_dim
        s_guess = np.ravel(self.states)

        q = -2 * np.reshape(gtx, (s_vec_dim,))

        h1_blocks = [[2 * gtg[i,j] * np.identity(self.feature_dim)
                      for j in range(self.clusters)]
                     for i in range(self.clusters)]
        H1 = np.block(h1_blocks)
        P = H1
        if self.clusters > 1:
            H2 = (self.clusters * np.identity(s_vec_dim)
                  - np.block([[np.identity(self.feature_dim)
                               for j in range(self.clusters)]
                              for i in range(self.clusters)]))
            P += 4 * self.reg_norm * H2

        s_soln = solve_qp(P, q, x0=s_guess, qpsolver="spg", tol=1.e-5)
        if s_soln is None:
            raise RuntimeError("failed to solve S subproblem")

        self.states = np.reshape(s_soln, ((self.clusters, self.feature_dim)))

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

        q_vecs = (-2 * np.matmul(self.dataset, np.transpose(self.states))
                  / self.normalization)
        P = (2 * np.matmul(self.states, np.transpose(self.states))
             / self.normalization)

        # try single step
        if self.use_trial_step:
            (evals, _) = np.linalg.eig(self.normalization * P)
            alpha_try = 1.0 / np.max(np.abs(evals))
            initial_qf = self.eval_quality_function()
            grad = self.normalization * (np.matmul(self.affiliations, P)
                                         + q_vecs)
            self.affiliations = simplex_projection(self.affiliations
                                                   - alpha_try * grad)
            trial_qf = self.eval_quality_function()

            if np.abs(trial_qf - initial_qf) > 1.0e-10:
                return

        # @todo replace with parallel equivalent
        for i in range(self.statistics_size):
            if self.gamma_solver == "spgqp":
                gamma_sol = solve_qp(
                    P, q_vecs[i,:], x0=self.affiliations[i,:], tol=1.e-4,
                    qpsolver="spgqp", projector=simplex_projection)
            elif self.gamma_solver == "cvxopt":
                gamma_sol = solve_qp(
                    P, q_vecs[i,:], tol=1.e-5,
                    qpsolver="cvxopt", A=np.ones((1,self.clusters)),
                    b=np.ones((1,1)), G=-np.identity(self.clusters),
                    h=np.zeros((self.clusters,1)))
            else:
                raise RuntimeError("unrecognized solver for Gamma subproblem")

            if gamma_sol is None:
                raise RuntimeError("failed to solve Gamma subproblem")

            self.affiliations[i,:] = np.ravel(gamma_sol)

        if self.verbose:
            updated_qf = self.eval_quality_function()
            print("\tFinal L = " + str(updated_qf))
            if updated_qf > initial_qf:
                print("\tWARNING: quality function increased")
            print("Successfully solved Gamma subproblem")

    def find_optimal_approx(self, initial_affs, initial_states=None):
        if initial_affs.shape != self.affiliations.shape:
            raise ValueError(
                "initial guess for affiliations has incorrect shape")

        self.affiliations = initial_affs

        if initial_states is not None:
            if initial_states.shape != self.states:
                raise ValueError(
                    "initial guess for states has incorrect shape")
            self.states = initial_states

        qf_old = self.eval_quality_function()
        delta_qf = 1e10 + self.stopping_tol
        iters = 0
        if self.verbose:
            print("Iterating with stopping tolerance = "
                  + str(self.stopping_tol)
                  + " and max. iterations = " + str(self.max_iterations))
        while delta_qf > self.stopping_tol and iters < self.max_iterations:
            if self.verbose:
                print("Iteration = " + str(iters))
                print("Solving S subproblem ...")
            self.solve_subproblem_s()

            if self.verbose:
                print("Solving Gamma subproblem ...")
            self.solve_subproblem_gamma()

            qf_new = self.eval_quality_function()
            delta_qf = np.abs(qf_old - qf_new)

            if self.verbose:
                print("Quality function at end of iteration:")
                print("\tOld L = " + str(qf_old))
                print("\tNew L = " + str(qf_new))
                print("\tDelta L = " + str(delta_qf))

            qf_old = qf_new

            iters += 1

        if iters == self.max_iterations and delta_qf > self.stopping_tol:
            raise RuntimeError(
                "failed to converge")

import numpy as np
from scipy.optimize import minimize

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
            if initial_states.shape != self.states.shape:
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

class SimEuclideanSPAModel(object):
    def __init__(self, x_dataset, y_dataset, x_clusters, y_clusters,
                 rel_weight=1.0, stopping_tol=1.e-5, max_iterations=500,
                 gamma_solver="spgqp",
                 verbose=False, use_trial_step=False, use_exact_probs=False):

        if x_dataset.ndim != 2:
            raise ValueError(
                "the input X dataset must be two-dimensional")

        if y_dataset.ndim != 2:
            raise ValueError(
                "the input Y dataset must be two-dimensional")

        if x_clusters < 1:
            raise ValueError(
                "the number of X clusters must be at least one")

        if y_clusters < 1:
            raise ValueError(
                "the number of Y clusters must be at least one")

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
        x_rec = np.matmul(self.x_affiliations, self.x_states)
        y_rec = np.matmul(self.y_affiliations, self.y_states)
        return (np.linalg.norm(self.y_dataset - y_rec) ** 2
                + self.rel_weight ** 2.0 *
                np.linalg.norm(self.x_dataset - x_rec) ** 2)

    def eval_quality_function(self):
        return self.distance()

    def solve_subproblem_s_y(self):
        gyty = np.matmul(np.transpose(self.y_affiliations), self.y_dataset)
        gytgy = np.matmul(np.transpose(self.y_affiliations),
                          self.y_affiliations)

        s_y_dim = self.y_clusters * self.y_feature_dim
        s_y_guess = np.ravel(self.y_states)

        q = -2 * np.reshape(gyty, (s_y_dim,))
        h1_y_blocks = [[2 * gytgy[i, j] * np.identity(self.y_feature_dim)
                        for j in range(self.y_clusters)]
                       for i in range(self.y_clusters)]
        P = np.block(h1_y_blocks)
        f = lambda s : (0.5 * np.matmul(np.transpose(s),
                                        np.matmul(P, s))
                        + np.dot(q, s))
        df = lambda s : np.matmul(P, s) + q
        y_res = minimize(f, s_y_guess, method="BFGS", jac=df)
        s_y_soln = y_res.x
        if s_y_soln is None:
            raise RuntimeError("failed to solve S subproblem for Y states")

        self.y_states = np.reshape(s_y_soln,
                                   ((self.y_clusters, self.y_feature_dim)))

    def solve_subproblem_s_x(self):
        gxtx = np.matmul(np.transpose(self.x_affiliations), self.x_dataset)
        gxtgx = np.matmul(np.transpose(self.x_affiliations),
                          self.x_affiliations)

        s_x_dim = self.x_clusters * self.x_feature_dim
        s_x_guess = np.ravel(self.x_states)

        eps_sq = self.rel_weight ** 2.0
        q = -2 * eps_sq * np.reshape(gxtx, (s_x_dim,))

        h1_x_blocks = [[2 * eps_sq * gxtgx[i, j]
                        * np.identity(self.x_feature_dim)
                        for j in range(self.x_clusters)]
                       for i in range(self.x_clusters)]
        P = np.block(h1_x_blocks)

        f = lambda s : (0.5 * np.matmul(np.transpose(s),
                                        np.matmul(P, s))
                        + np.dot(q, s))
        df = lambda s : np.matmul(P, s) + q
        x_res = minimize(f, s_x_guess, method="BFGS", jac=df)
        s_x_soln = x_res.x
        if s_x_soln is None:
            raise RuntimeError("failed to solve S subproblem for X states")

        self.x_states = np.reshape(s_x_soln,
                                   ((self.x_clusters, self.x_feature_dim)))

    def solve_subproblem_s(self):
        if self.verbose:
            initial_qf = self.eval_quality_function()
            print("\tInitial L = " + str(self.eval_quality_function()))

        self.solve_subproblem_s_y()
        self.solve_subproblem_s_x()

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

        q_y_vecs = (-2 * np.matmul(self.y_dataset, np.transpose(self.y_states)))
        q_x_vecs = (-2 * self.rel_weight ** 2.0 *
                    np.matmul(self.x_dataset, np.transpose(self.x_states)))

        P_y = (2 * np.matmul(self.y_states, np.transpose(self.y_states)))
        P_x = (2 * self.rel_weight ** 2.0 *
               np.matmul(self.x_states, np.transpose(self.x_states)))

        tol=1.e-5
        for i in range(self.statistics_size):
            if self.gamma_solver == "spgqp":
                gamma_x_sol = solve_qp(
                    P_x, q_x_vecs[i,:], x0=self.x_affiliations[i,:], tol=tol,
                    qpsolver="spgqp", projector=simplex_projection)
                gamma_y_sol = solve_qp(
                    P_y, q_y_vecs[i,:], x0=self.y_affiliations[i,:], tol=tol,
                    qpsolver="spgqp", projector=simplex_projection)
            elif self.gamma_solver == "cvxopt":
                gamma_x_sol = solve_qp(
                    P_x, q_x_vecs[i,:], tol=tol,
                    qpsolver="cvxopt", A=np.ones((1, self.x_clusters)),
                    b=np.ones((1,1)), G=-np.identity(self.x_clusters),
                    h=np.zeros((self.x_clusters,1)))
                gamma_y_sol = solve_qp(
                    P_y, q_y_vecs[i,:], tol=tol,
                    qpsolver="cvxopt", A=np.ones((1, self.y_clusters)),
                    b=np.ones((1,1)), G=-np.identity(self.y_clusters),
                    h=np.zeros((self.y_clusters,1)))
            else:
                raise RuntimeError("unrecognized solver for Gamma subproblem")

            if gamma_x_sol is None or gamma_y_sol is None:
                raise RuntimeError("failed to solve Gamma subproblem")

            print("solved row ", i)
            print("init qf = ", self.eval_quality_function())
            self.x_affiliations[i,:] = np.ravel(gamma_x_sol)
            self.y_affiliations[i,:] = np.ravel(gamma_y_sol)
            print("final qf = ", self.eval_quality_function())

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
            qf_old = qf_new
            iters += 1

        if iters == self.max_iterations and delta_qf > self.stopping_tol:
            raise RuntimeError(
                "failed to converge")

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

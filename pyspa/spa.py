import numpy as np

from .optimizers import solve_qp

class SPA2Model(object):
    def __init__(self, dataset, clusters, affiliations=None,
                 eps_s_sq=0, normalize=True,
                 stopping_tol=1.e-2, max_iterations=100):
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
        if normalize:
            self.dataset = self.normalize_dataset(self.dataset)
        self.trxtx = np.sum(np.power(self.dataset, 2))

        self.clusters = clusters
        self.eps_s_sq = eps_s_sq
        self.reg_norm = eps_s_sq / (self.feature_dim * self.clusters)
        if self.clusters > 1:
            self.reg_norm = self.reg_norm / (self.clusters - 1.0)

        self.stopping_tol = stopping_tol
        self.max_iterations = max_iterations

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
            normalisations = np.sum(self.affiliations, axis=1)
            self.affiliations = np.divide(self.affiliations,
                                          normalisations[:, np.newaxis])

        self.states = np.zeros((self.clusters, self.feature_dim))

    def normalize_dataset(self, dataset):
        return np.divide(dataset, np.amax(np.abs(dataset), axis=0))

    def distance(self):
        errs = self.dataset - np.matmul(self.affiliations, self.states)
        return np.sum(np.power(errs, 2))

    def states_regularization(self):
        if self.clusters == 1:
            return 0
        reg = (self.clusters * np.trace(np.matmul(
            np.transpose(self.states), self.states))
               - np.sum(np.matmul(self.states, np.transpose(self.states))))
        return (2 * self.eps_s_sq * reg /
                (self.feature_dim * self.clusters * (self.clusters - 1)))

    def eval_quality_function(self):
        return self.distance() + self.states_regularization()

    def solve_subproblem_s(self):
        # construct matrices for QP subproblem for states S
        gtx = np.matmul(np.transpose(self.affiliations), self.dataset)
        gtg = np.matmul(np.transpose(self.affiliations), self.affiliations)

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
        self.states = np.reshape(s_soln, ((self.clusters, self.feature_dim)))

    def solve_subproblem_gamma(self):
        # construct matrices for QP subproblem for affiliations
        # N.B. no imposed regularization on affiliations currently
        gamma_vec_dim = self.clusters * self.statistics_size
        q_vecs = -2 * np.matmul(self.dataset, np.transpose(self.states))
        P = 2 * np.matmul(self.states, np.transpose(self.states))
        # @todo replace with parallel equivalent
        for i in range(self.statistics_size):
            self.affiliations[i,:] = solve_qp(P, q_vecs[i,:],
                                              solver="spgqp")

    def find_optimal_approx(self, initial_affs, initial_states=None):
        if initial_affs.shape != self.affiliations:
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
        while delta_qf > self.stopping_tol and iters < self.max_iterations:
            self.solve_subproblem_s()
            self.solve_subproblem_gamma()
            qf_new = self.eval_quality_function()
            delta_qf = np.abs(qf_old - qf_new)
            qf_old = qf_new
            iters += 1

        if iters == max_iterations and delta_qf > self.stopping_tol:
            raise RuntimeError(
                "failed to converge")

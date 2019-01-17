import numpy as np

class SPA2Model(object):
    def __init__(self, dataset, clusters, eps_s_sq=0, normalize=True,
                 stopping_tol=1.e-2, max_iterations=100):
        if dataset.ndim != 2:
            raise ValueError(
                "the input dataset must be at least two dimensional")

        if eps_s_sq < 0:
            raise ValueError(
                "regularization parameter must be non-negative")

        self.dataset = dataset.copy()
        self.statistics_size = self.dataset.shape[0]
        self.feature_dim = self.dataset.shape[1]
        if normalize:
            self.dataset = self.normalize_dataset(self.dataset)
        self.trxtx = np.sum(np.power(self.dataset, 2))

        self.clusters = clusters
        self.eps_s_sq = eps_s_sq

        self.stopping_tol = stopping_tol
        self.max_iterations = max_iterations

        self.affiliations = np.zeros((self.statistics_size, self.clusters))
        self.states = np.zeros((self.clusters, self.feature_dim))

    def normalize_dataset(self, dataset):
        return np.divide(dataset, np.amax(np.abs(dataset), axis=0))

    def distance(self):
        errs = self.dataset - np.matmul(self.affiliations, self.states)
        return np.sum(np.power(errs, 2))

    def states_regularization(self):
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
        c0 = self.trxtx
        q = -2 * np.reshape(gtx, (s_vec_dim, 1))
        h1_blocks = [[2 * gtg[i,j] * np.eye(self.feature_dim)
                      for j in range(self.clusters)]
                     for i in range(self.clusters)]
        H1 = np.block(h1_blocks)
        H2 = self.clusters * np.eye(s_vec_dim) - np.ones(s_vec_dim)
        P = H1 + 4 * self.eps_s_sq * H2 / (s_vec_dim * (self.clusters - 1))
        s_soln = solve_qp(c0, q, P)
        self.states = np.reshape(s_soln, ((self.clusters, self.feature_dim)))

    def solve_subproblem_gamma(self):
        # construct matrices for QP subproblem for affiliations
        # N.B. no imposed regularization on affiliations currently
        gamma_vec_dim = self.clusters * self.statistics_size
        c0 = self.trxtx
        q_vecs = -2 * np.matmul(self.dataset, np.transpose(self.states))
        P = 2 * np.matmul(self.states, np.transpose(self.states))
        # @todo replace with parallel equivalent
        for i in range(self.statistics_size):
            self.affiliations[i,:] = solve_qp(0, q_vecs[i,:], P,
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

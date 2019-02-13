import unittest

import numpy as np
from pyspa.helpers import assign_persistent_state_ids

class TestPersistenceHelpers(unittest.TestCase):
    def test_assign_persistent_state_ids_no_persistence(self):
        persistence_threshold = 1
        lengths = np.logspace(0, 6, 7, dtype="int")
        n_states = 6
        states = np.arange(n_states)
        n_tries = 5
        for l in lengths:
            for i in range(n_tries):
                test_path = np.random.randint(0, n_states, size=l)
                state_ids = assign_persistent_state_ids(
                states, test_path,
                persistence_threshold=persistence_threshold)
                self.assertTrue(np.allclose(test_path, state_ids))

    def test_assign_persistent_state_ids_persistence_2(self):
        persistence_threshold = 2
        n_states = 5
        states = np.arange(n_states)

        vp1 = np.array([1,2,3,2,4,5])
        ids1 = assign_persistent_state_ids(
            states, vp1, persistence_threshold=persistence_threshold)
        expected = n_states * np.ones(np.size(vp1))
        self.assertTrue(np.allclose(ids1, expected))

        vp2 = np.array([1,1,2,3,2,4,2])
        ids2 = assign_persistent_state_ids(
            states, vp2, persistence_threshold=persistence_threshold)
        expected = np.array([1, 1, n_states, n_states,
                             n_states, n_states, n_states])
        self.assertTrue(np.allclose(ids2, expected))

        vp3 = np.array([4,4])
        ids3 = assign_persistent_state_ids(
            states, vp3, persistence_threshold=persistence_threshold)
        expected = np.array([4, 4])
        self.assertTrue(np.allclose(ids3, expected))

        vp4 = np.array([3])
        ids4 = assign_persistent_state_ids(
            states, vp4, persistence_threshold=persistence_threshold)
        expected = np.array([n_states])
        self.assertTrue(np.allclose(ids4, expected))

        vp5 = np.array([4, 4, 4, 1, 3, 3, 3, 2, 2])
        ids5 = assign_persistent_state_ids(
            states, vp5, persistence_threshold=persistence_threshold)
        expected = np.array([4, 4, 4, n_states, 3, 3, 3, 2, 2])
        self.assertTrue(np.allclose(ids5, expected))

    def test_assign_persistent_state_ids_persistence_5(self):
        persistence_threshold = 5
        n_states = 5
        states = np.arange(n_states)

        vp1 = np.array([1,2,3,2,4,5])
        ids1 = assign_persistent_state_ids(
            states, vp1, persistence_threshold=persistence_threshold)
        expected = n_states * np.ones(np.size(vp1))
        self.assertTrue(np.allclose(ids1, expected))

        vp2 = np.array([1])
        ids2 = assign_persistent_state_ids(
            states, vp2, persistence_threshold=persistence_threshold)
        expected = np.array([n_states])
        self.assertTrue(np.allclose(ids2, expected))

        vp3 = np.array([2, 2, 2, 2])
        ids3 = assign_persistent_state_ids(
            states, vp3, persistence_threshold=persistence_threshold)
        expected = np.array([n_states, n_states, n_states, n_states])
        self.assertTrue(np.allclose(ids3, expected))

        vp4 = np.array([1, 2, 3, 3, 3, 3, 3, 4, 4, 1, 1, 1, 1, 1, 1, 2])
        ids4 = assign_persistent_state_ids(
            states, vp4, persistence_threshold=persistence_threshold)
        expected = np.array([n_states, n_states, 3, 3, 3, 3, 3, n_states,
                             n_states, 1, 1, 1, 1, 1, 1, n_states])
        self.assertTrue(np.allclose(ids4, expected))

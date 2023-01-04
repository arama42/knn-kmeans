import unittest

import numpy as np
from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import cosine_distances

import distance_metrics


class TestClass(unittest.TestCase):

    def test_euclidean_distance(self):
        n_examples_a = 20
        n_examples_b = 2
        n_features = 784
        a = np.random.rand(n_examples_a, n_features)
        b = np.random.rand(n_examples_b, n_features)
        sklearn_metric = DistanceMetric.get_metric('euclidean')
        implemented_distances = distance_metrics.euclidean_distances(a, b)
        sklearn_distances = sklearn_metric.pairwise(a, b)
        abs_diff = np.abs(implemented_distances - sklearn_distances)
        epsilon = 1e-7
        self.assertTrue(np.all(abs_diff < epsilon))

    def test_cosine_distance(self):
        n_examples_a = 20
        n_examples_b = 2
        n_features = 784
        a = np.random.rand(n_examples_a, n_features)
        b = np.random.rand(n_examples_b, n_features)
        implemented_distances = distance_metrics.cosine_distances(a, b)
        sklearn_distances = cosine_distances(a, b)
        abs_diff = np.abs(implemented_distances - sklearn_distances)
        epsilon = 1e-7
        self.assertTrue(np.all(abs_diff < epsilon))


if __name__ == '__main__':
    unittest.main()

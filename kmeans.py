import numpy as np
from distance_metrics import euclidean_distance


class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        n_samples = features.shape[0]
        self.means = features[np.random.randint(0, n_samples, self.n_clusters)]
        assignments = {i: None for i in range(n_samples)}
        assignment_updated = True
        while assignment_updated:
            assignment_updated = self.update_assignments(features, assignments)
            self.update_means(features, assignments)

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        n_samples = features.shape[0]
        predictions = []
        for i in range(n_samples):
            min_dist_cluster = self.predict_single(features[i])
            predictions.append(min_dist_cluster)
        return np.array(predictions)

    def predict_single(self, features):
        min_dist = 10 ** 10
        min_dist_cluster = None
        for j in range(self.n_clusters):
            dist = euclidean_distance(features, self.means[j])
            if dist < min_dist:
                min_dist = dist
                min_dist_cluster = j
        return min_dist_cluster

    def update_assignments(self, features, assignments):
        n_samples = features.shape[0]
        assignment_updated = False
        for i in range(n_samples):
            min_dist_cluster = self.predict_single(features[i])
            if assignments[i] != min_dist_cluster:
                assignments[i] = min_dist_cluster
                assignment_updated = True
        return assignment_updated

    def update_means(self, features, assignments):
        n_samples = features.shape[0]

        cluster_members = {i: [] for i in range(self.n_clusters)}
        for i in range(n_samples):
            cluster_members[assignments[i]].append(features[i])
        new_means = []
        for i in range(self.n_clusters):
            new_means.append(np.mean(cluster_members[i], axis=0))
        self.means = np.array(new_means)
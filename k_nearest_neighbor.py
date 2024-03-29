import statistics

import numpy as np
from distance_metrics import euclidean_distances, cosine_distances
import math

class KNearestNeighbor():
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure (ex. eucliden cosim) you will
        use to compare data points. 

        ```aggregator``` lets you alter how a label is predicted for a data point based 
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3 
        closest neighbors are:
            [
                [1, 2, 3], 
                [2, 3, 4], 
                [3, 4, 5]
            ] 
        And the aggregator is 'mean', applied along each dimension, this will return for 
        that point:
            [
                [2, 3, 4]
            ]

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean' or 'cosim'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        if distance_measure == 'euclidean':
            self.distance_measure = euclidean_distances
        elif distance_measure == 'cosim':
            self.distance_measure = cosine_distances
        else:
            raise NotImplementedError('Distance measure "{}" has not been implemented.'.format(distance_measure))
        self.aggregator = aggregator
        self.train_features = None
        self.train_targets = None
        self.train_example_count = None
        self.train_feature_count = None

    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """
        self.train_features = features
        self.train_targets = np.array(targets)
        self.train_example_count = len(features)
        self.train_feature_count = len(features[0])

    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the 
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """

        distances = self.distance_measure(features, self.train_features)
        sorted_distance_indices = np.argsort(distances)

        labels = []
        test_example_count = len(features)
        for ix_test in range(test_example_count):
            window_left, window_right = 0, self.n_neighbors
            if ignore_first:
                window_left += 1
                window_right += 1
            nearest_k_targets = self.train_targets[sorted_distance_indices[ix_test, window_left:window_right]]
            if self.aggregator == 'mean':
                labels.append(str(statistics.mean(nearest_k_targets)))
            elif self.aggregator == 'median':
                labels.append(str(statistics.median(nearest_k_targets)))
            elif self.aggregator == 'mode':
                labels.append(str(statistics.mode(nearest_k_targets)))
            else:
                raise NotImplementedError('Aggregator "{}" has not been implemented.'.format(self.aggregator))
        return labels

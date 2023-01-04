import numpy as np

class SoftKMeans():
    '''
    :::params:::
    n_clusters = number of clusters
    beta: temperature (controls the sharpness of the probability distribution; beta ---> inf approaches hard k-means)
    :::returns:::
    None - initializes an object of the class SoftKMeans with the given parameters
    '''
    def __init__(self, n_clusters, beta):
        self.n_clusters = n_clusters
        self.means = None
        self.beta = beta
        self.loss = 0

    '''
    :::params:::
    features (np.ndarray): array containing inputs of size (n_samples, n_features).
    :::returns:::
    None - updates the mean values of different clusters
    '''    
    def fit(self, features):
        n_samples = features.shape[0]
        self.means = features[np.random.randint(0, n_samples, self.n_clusters)]

        while True: 
            prob = self.get_cluster_probabilities(features)
            self.update_means(features, prob)
            prev_loss = self.loss
            self.loss = self.calculate_loss(prob, features)
            if np.abs(self.loss - prev_loss) < 1e-5:
                break
    
    '''
    :::params:::
    features (np.ndarray): array containing inputs of size (n_samples, n_features).
    :::returns:::
    an array of array containing probabilities of every input sample belonging to each of the clusters
    '''
    def predict(self, features):
        cluster_probabilities = self.get_cluster_probabilities(features)
        return np.array(cluster_probabilities)

            
    def update_means(self, features, resp):
        feature_dim = features[0].shape[0]
        means = np.zeros((self.n_clusters, feature_dim))

        for idx in range(self.n_clusters):
            means[idx] = resp[:, idx].dot(features) / resp[:, idx].sum()
        
        self.means = means


    def get_cluster_probabilities(self, features):
        n_samples = features.shape[0]
        p = np.zeros((n_samples, self.n_clusters))

        for idx in range(n_samples):        
            p[idx] = np.exp(-self.beta * np.linalg.norm(features[idx] - self.means, 2, axis=1)) 
        p /= p.sum(axis=1, keepdims=True)

        return p
    
    def calculate_loss(self, p, features):
        loss = 0
        for idx in range(self.n_clusters):
            norm = np.linalg.norm(features - self.means[idx], 2)
            loss += (norm * np.expand_dims(p[:, idx], axis=1)).sum()
        return loss
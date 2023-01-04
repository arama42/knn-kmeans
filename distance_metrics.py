import numpy as np


def euclidean_distances(a, b):
    """
    :param a: 2-D numpy array of shape (n_vectors_a, n_dims)
    :param b: 2-D numpy array of shape (n_vectors_b, n_dims)
    :return: 2-D numpy array of shape (n_vectors_a, n_vectors_b),
             containing Euclidean distances between all vectors in a and b.
    """
    distances = []
    for i in range(a.shape[0]):
        distances_b = []
        for j in range(b.shape[0]):
            distances_b.append(euclidean_distance(a[i], b[j]))
        distances.append(distances_b)
    return np.array(distances)


def euclidean_distance(a, b):
    """
    :param a: Numpy vector
    :param b: Numpy vector
    :return: Euclidean distance between a and b
    """
    return np.sqrt(((a - b) ** 2).sum())


def cosine_distances(a, b):
    """
    :param a: 2-D numpy array of shape (n_vectors_a, n_dims)
    :param b: 2-D numpy array of shape (n_vectors_b, n_dims)
    :return: 2-D numpy array of shape (n_vectors_a, n_vectors_b),
             containing Euclidean distances between all vectors in a and b.
             where cosine distance = 1 - cosine similarity
    """
    distances = []
    for i in range(a.shape[0]):
        distances_b = []
        for j in range(b.shape[0]):
            distances_b.append(cosine_distance(a[i], b[j]))
        distances.append(distances_b)
    return np.array(distances)


def cosine_distance(a, b):
    """
    :param a: Numpy vector
    :param b: Numpy vector
    :return: Cosine Distance between a and b
    """
    return 1 - cosine_similarity(a, b)


def cosine_similarity(a, b):
    """
    :param a: Numpy vector
    :param b: Numpy vector
    :return: Cosine Similarity between a and b
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

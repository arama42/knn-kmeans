#imports
import numpy as np
from sklearn.decomposition import PCA
import k_nearest_neighbor
import kmeans as km
import matplotlib.pyplot as plt    
import soft_kmeans as skm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, completeness_score, homogeneity_score, silhouette_score


# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train, query, metric, aggregator='mode'):

    features_train = train[0]
    targets_train = train[1]
    
    knn_model = k_nearest_neighbor.KNearestNeighbor(4, metric, aggregator)
    knn_model.fit(features_train,targets_train)

    labels = knn_model.predict(query)

    return labels


def kmeans(train, query):
    kMeans_model = km.KMeans(10)
    kMeans_model.fit(train)

    labels = kMeans_model.predict(query)

    return labels

def soft_kmeans(train, query, beta):
    kMeans_model = skm.SoftKMeans(10, beta)
    kMeans_model.fit(train)
    cluster_probabilities = kMeans_model.predict(query)

    return(cluster_probabilities)



def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')

def split_features_targets(train_data, validation_data, test_data):
    features_train = []
    targets_train = []
    for ix in range(len(train_data)):
        ft_train = np.array(train_data[ix][1]).astype('float32')
        features_train.append(ft_train)
        targets_train.append(train_data[ix][0])
    
    features_validation = []
    targets_validation = []
    for ix in range(len(validation_data)):
        val = np.array(validation_data[ix][1]).astype('float32')
        features_validation.append(val)
        targets_validation.append(validation_data[ix][0])
    
    features_test = []
    targets_test= []
    for ix in range(len(test_data)):
        ft_test = np.array(test_data[ix][1]).astype('float32')
        features_test.append(ft_test)
        targets_test.append(test_data[ix][0])

    return [features_train, targets_train, features_validation, targets_validation, features_test, targets_test]

def evaluate_knn(true_labels, pred_labels):
    correctly_classified = 0
    for ix in range(len(pred_labels)):
        if pred_labels[ix] == true_labels[ix] :
            correctly_classified += 1 

    return (correctly_classified/len(pred_labels)) *100

def create_confusion_matrix(targets_test, knn_test_pred):
    labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    cm = confusion_matrix(targets_test, knn_test_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot()
    plt.show()

def evaluate_clusters(true_labels, pred_labels, features):
    cmplt_score = completeness_score(true_labels, pred_labels) #1.0 = perfectly complete labeling
    hmgns_score = homogeneity_score(true_labels, pred_labels) #1.0 stands for perfectly homogeneous labeling)
    slht_score = silhouette_score(features, pred_labels)
    '''
    The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. 
    Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.
    '''
    print('Cluster Completeness Score: ', cmplt_score)
    print('Cluster Homogeneity Score: ', hmgns_score)
    print('Cluster Silhoutte Score: ', slht_score)



def main():
    #show('valid.csv','pixels')

    #read data
    train_data = read_data('train.csv')
    validation_data = read_data('valid.csv')
    test_data = read_data('test.csv')

    #separate the features and targets
    features_train, targets_train, features_validation, targets_validation, features_test, targets_test = split_features_targets(train_data, validation_data, test_data)

    #dimensionality reduction using PCA
    pca = PCA(n_components = 0.95)
    features_train_reduced = pca.fit_transform(features_train)
    features_validation_reduced = pca.transform(features_validation)
    features_test_reduced = pca.transform(features_test)

    # KNN - validation
    knn_validation_pred = knn([features_train_reduced, targets_train], features_validation_reduced, 'euclidean')
    knn_validation_accuracy = evaluate_knn(targets_validation, knn_validation_pred)
    print('knn_validation_accuracy with Euclidean distance:', knn_validation_accuracy)

    knn_validation_pred = knn([features_train_reduced, targets_train], features_validation_reduced, 'cosim')
    knn_validation_accuracy = evaluate_knn(targets_validation, knn_validation_pred)
    print('knn_validation_accuracy with cosine distance:', knn_validation_accuracy)

    # KNN - test
    knn_test_pred = knn([features_train_reduced, targets_train], features_test_reduced, 'euclidean')
    knn_test_accuracy = evaluate_knn(targets_test, knn_test_pred)
    print('knn_test_accuracy with Euclidean distance:', knn_test_accuracy)

    create_confusion_matrix(targets_test, knn_test_pred)

    knn_test_pred = knn([features_train_reduced, targets_train], features_test_reduced, 'cosim')
    knn_test_accuracy = evaluate_knn(targets_test, knn_test_pred)
    print('knn_test_accuracy with cosine distance:', knn_test_accuracy)

    create_confusion_matrix(targets_test, knn_test_pred)

    kmeans_pred = kmeans(features_train_reduced, features_test_reduced)

    evaluate_clusters(targets_test, kmeans_pred, features_test_reduced)

    normalized_train_data = []
    for f in features_train_reduced:
        normalized_train_data.append(np.array([i/255.0 for i in f]))
    
    normalized_test_data = []
    for f in features_test_reduced:
        normalized_test_data.append(np.array([i/255.0 for i in f]))

    cluster_probabilities = soft_kmeans(np.array(normalized_train_data), np.array(normalized_test_data), 10)
    print(cluster_probabilities)


if __name__ == "__main__":
    main()

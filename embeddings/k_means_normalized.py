import numpy as np


def k_means(X, k, initializations, max_iter=20):
    ''' Computes the k-means algorithm with normalized centroids

    Parameters
    ----------
    X: nxm array
        Array containing the dataset to be clusterd, n samples with m dimensions

    k: int
        Number of clusters

    initializations: int
        Number of random intializations of the algorithm

    max_iter: int
        Maximum iterations in each intialization of the algorithm
    '''

    n_samples, n_features = X.shape

    Wdistances = []  # we will store the within-cluster distance on this list
    labels_list = []  # we will store on this list the labels returned by the different initalizations of the algorithm

    Wdistances = []
    labels_list = []
    centroids_list = []

    for m in range(initializations):  # this loop runs the algorithms with different initializations

        unique_values = 0
        while unique_values != k:  # This loop guarantees that each class has at least one point, otherwise the code cannot be executed without errors
            labels = np.random.randint(low=0, high=k, size=n_samples)
            unique_values = len(np.unique(labels))
        X_labels = np.append(X, labels.reshape(-1, 1), axis=1)

        # we compute the centroids of each of the k clusters
        centroids = np.zeros((k, n_features))
        for i in range(k):
            centroids[i] = np.mean([x for x in X_labels if x[-1] == i], axis=0)[0:n_features]
            centroids[i] /= np.linalg.norm(centroids[i])

        new_labels = np.array(range(k))
        difference = 0
        distances = np.zeros((k, n_samples))

        # k-means algorithm
        for i in range(max_iter):
            # distances: between data points and centroids

            for l in np.unique(new_labels):  # we only update the non empty classes
                distances[l, :] = np.linalg.norm(X - centroids[l], axis=1) ** 2

            # new_labels: computed by finding centroid with minimal distance
            new_labels = np.argmin(distances, axis=0)

            if (labels == new_labels).all() or i == max_iter - 1:
                W = 0  # we will store the within-cluster distance of this initialization in this variable
                for l in np.unique(new_labels):  # we only compute the distance for the nonempty labels
                    W += np.sum(distances[
                                    l, new_labels == l])  # sum the distance to the centroid for the points that are included in class l
                Wdistances.append(W)
                labels_list.append(new_labels)
                centroids_list.append(centroids)
                break
            else:
                # labels changed
                labels = new_labels
                for l in np.unique(new_labels):
                    # we compute the centroids by taking the mean over associated data points
                    centroids[l] = np.mean(X[labels == l], axis=0)
                    centroids[l] /= np.linalg.norm(centroids[l])  # we normalize the centroids

    optimum_clustering = np.argmin(Wdistances)
    return labels_list[optimum_clustering], Wdistances[optimum_clustering], centroids_list[optimum_clustering]


def predict_k_means(y, centroids):
    ''' Predict the clusters to which the clusters belong

    Parameters
    ----------
    y: nxm array
        Array containing the dataset for the prediction, n samples with m dimensions

    centroids: lxm array
        Array containing the centroids of the clusters, l centroids with m dimensions
    '''

    n, _ = np.shape(y)
    l, _ = np.shape(centroids)
    distances = np.zeros((l, n))
    for i in range(l):
        distances[i, :] = np.linalg.norm(y - centroids[i], axis=1) ** 2
    return np.argmin(distances, axis=0)

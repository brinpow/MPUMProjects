import numpy as np
from scipy.spatial.distance import cdist

DISTS = []

class HierarchicalClustering:
    def cluster_min(self, mins, distances, **kwargs):
        index = 0
        for i in range(mins.shape[0]):
            if distances[i][mins[i]] < distances[index][mins[index]]:
                index = i
        index2 = mins[index]
        DISTS.append(distances[index][index2])

        for i in range(mins.shape[0]):
            if distances[index2][i] < distances[index][i]:
                distances[index][i] = distances[index2][i]
                distances[i][index] = distances[index2][i]
        distances[index][index] = np.inf

        for i in range(mins.shape[0]):
            distances[index2][i] = np.inf
            distances[i][index2] = np.inf

        for i in range(mins.shape[0]):
            if mins[i] == index2:
                mins[i] = index
            if distances[index][i] < distances[index][mins[index]]:
                mins[index] = i

        self.clusters[index2] = index
        for i in range(mins.shape[0]):
            if self.clusters[i] == index2:
                self.clusters[i] = index

    @staticmethod
    def min_init(distances):
        return np.argmin(distances, axis=1)

    def cluster_max(self, maxs, distances, **kwargs):
        index = 0
        maxs = np.argmin(distances, axis=1)
        for i in range(maxs.shape[0]):
            if distances[i][maxs[i]] < distances[index][maxs[index]]:
                index = i
        index2 = maxs[index]
        DISTS.append(distances[index][index2])

        for i in range(maxs.shape[0]):
            if distances[index2][i] > distances[index][i]:
                distances[index][i] = distances[index2][i]
                distances[i][index] = distances[index2][i]

        for i in range(maxs.shape[0]):
            distances[index2][i] = np.inf
            distances[i][index2] = np.inf

        self.clusters[index2] = index
        for i in range(maxs.shape[0]):
            if self.clusters[i] == index2:
                self.clusters[i] = index

    def cluster_centroid(self, distances, **kwargs):
        X = kwargs['X']
        index = 0
        centroids = np.argmin(distances, axis=1)
        for i in range(centroids.shape[0]):
            if distances[i][centroids[i]] < distances[index][centroids[index]]:
                index = i
        index2 = centroids[index]
        DISTS.append(distances[index][index2])

        for i in range(centroids.shape[0]):
            if self.clusters[i] == index2:
                self.clusters[i] = index

        cluster = [X[i] for i in range(X.shape[0]) if self.clusters[i] == index]
        cluster = np.array(cluster)
        self.centroids[index] = cluster.mean(axis=0)

        for i in range(centroids.shape[0]):
            distances[index2][i] = np.inf
            distances[i][index2] = np.inf

        for i in range(centroids.shape[0]):
            val = np.linalg.norm(self.centroids[index] - self.centroids[i])
            if distances[index][i] != np.inf:
                distances[index][i] = val
                distances[i][index] = val

    def cluster_mean(self, distances, **kwargs):
        sums = kwargs['sums']
        index = 0
        means = np.argmin(distances, axis=1)
        for i in range(means.shape[0]):
            if distances[i][means[i]] < distances[index][means[index]]:
                index = i
        index2 = means[index]
        DISTS.append(distances[index][index2])

        for i in range(means.shape[0]):
            if self.clusters[i] == index2:
                self.clusters[i] = index

        for i in range(means.shape[0]):
            if self.clusters[i] == i:
                dist = (distances[index][i]*sums[index] + distances[index2][i]*sums[index2])/(sums[index]+sums[index2])
                distances[index][i] = dist
                distances[i][index] = dist
        distances[index][index] = np.inf
        sums[index] += sums[index2]

        for i in range(means.shape[0]):
            distances[index2][i] = np.inf
            distances[i][index2] = np.inf

    def __init__(self):
        self.methods = {"SINGLE": self.cluster_min, "COMPLETE": self.cluster_max,
                        "CENTROID": self.cluster_centroid, "AVERAGE": self.cluster_mean}

    def fit(self, X, method, classes):
        DISTS = []
        distances = cdist(X, X, 'euclidean')
        self.clusters = np.zeros(X.shape[0])
        self.centroids = []
        for i in range(distances.shape[0]):
            distances[i][i] = np.inf
            self.clusters[i] = i
            self.centroids.append(X[i])

        self.centroids = np.array(self.centroids)
        helper = self.min_init(distances)
        sums = np.ones(distances.shape[0])

        for i in range(X.shape[0] - classes):
            if method == "CENTROID":
                self.methods[method](distances, X=X)
            elif method == "AVERAGE":
                self.methods[method](distances, sums=sums)
            else:
                self.methods[method](helper, distances=distances)

    def aclusters(self):
        return self.clusters
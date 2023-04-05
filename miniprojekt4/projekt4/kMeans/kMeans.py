import numpy as np
from scipy.spatial.distance import cdist


class Kmeans:
    def inertia(self, X):
        distances = cdist(X, self.centroids, 'euclidean')
        inertia = np.array([np.min(dist)*np.min(dist) for dist in distances])
        return np.sum(inertia)

    def choose_centroids(self, X):
        centroids = [X[np.random.randint(0, X.shape[0])]]
        for i in range(self.k-1):
            arr = np.array([min([np.square(np.linalg.norm(x-centr)) for centr in centroids]) for x in X])
            arr = arr / arr.sum()
            arr = arr.cumsum()
            rand = np.random.random()
            index = np.where(arr >= rand)[0][0]
            centroids.append(X[index])
        return np.array(centroids)

    def fit(self, X, k, iterations, algo):
        self.k = k

        if algo == 'Kmeans':
            pos = np.random.choice(X.shape[0], k, replace=False)
            self.centroids = X[pos, :]
        else:
            self.centroids = self.choose_centroids(X)
        distances = cdist(X, self.centroids, 'euclidean')

        self.cluster_assign = np.array([np.argmin(dist) for dist in distances])

        for _ in range(iterations):
            self.centroids = []
            for i in range(k):
                new_centroid = X[self.cluster_assign == i].mean(axis=0)
                self.centroids.append(new_centroid)

            self.centroids = np.vstack(self.centroids)

            distances = cdist(X, self.centroids, 'euclidean')
            new_cluster_assign = np.array([np.argmin(dist) for dist in distances])

            if np.sum(self.cluster_assign == new_cluster_assign) == self.cluster_assign.shape[0]:
                inertia = self.inertia(X)
                return _, inertia

            self.cluster_assign = new_cluster_assign
        return np.inf, self.inertia(X)
    def clusters(self):
        return self.cluster_assign

    def check(self, X, y):
        pass

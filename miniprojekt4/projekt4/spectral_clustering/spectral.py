import numpy as np
from scipy.spatial.distance import cdist
from projekt4.kMeans import kMeans
from projekt4.hierarchical_clustering import hierarchical_clustering

class SpectralClustering:
    def Gaussian(self, X):
        W = cdist(X, X, metric='sqeuclidean')
        W = np.exp(-W*W / (2 * self.sigma ** 2))
        return W

    def reverse(self, X):
        rev = np.vectorize(lambda x,y: 1/np.linalg.norm(X[x]-X[y]) if x != y else 1.0)
        return np.fromfunction(rev, shape=(X.shape[0],X.shape[0]), dtype=int)

    def epsilon_func(self, X):
        rev = np.vectorize(lambda x, y: 1.0 if np.linalg.norm(X[x]-X[y]) < self.epsilon else 0.0)
        return np.fromfunction(rev, shape=(X.shape[0], X.shape[0]), dtype=int)

    def __init__(self, sigma=None, k=None, epsilon=None):
        self.sigma = sigma
        self.k = k
        self.epsilon = epsilon
        self.methods = {"Gaussian": self.Gaussian, "Rev": self.reverse, "Epsilon": self.epsilon_func}

    def fit(self, X, type):
        W = self.methods[type](X)
        D = np.diag(np.sum(W, axis=1))
        L = D - W

        if self.k == 2:
            eigval, eigvec = np.linalg.eig(L)
            indexes = np.argsort(eigval)
            self.cluster_assign = eigvec[:, indexes[1]]
            self.cluster_assign[self.cluster_assign > 0] = 1
            self.cluster_assign[self.cluster_assign < 0] = 0
        else:
            D_srev = np.diag(1/np.diag(np.sqrt(D)))
            L = D_srev@L@D_srev
            eigval, eigvec = np.linalg.eig(L)
            indexes = np.argsort(eigval)
            V = eigvec[:,indexes[:self.k]]
            sums = np.sqrt(np.sum(np.square(V), axis=1)).reshape(-1, 1)
            U = V / sums
            Ur = []
            for i in range(U.shape[0]):
                Ur.append([])
                for ii in range(U.shape[1]):
                    Ur[i].append(U[i][ii].real)
            U = np.array(Ur)
            km = kMeans.Kmeans()
            iter, inertia = km.fit(U, self.k, 300, 'Kmeans++')
            self.cluster_assign = km.clusters()
            """hc = hierarchical_clustering.HierarchicalClustering()
            hc.fit(U,"CENTROID", self.k)
            self.cluster_assign = hc.aclusters()"""
            return inertia

    def aclusters(self):
        return self.cluster_assign
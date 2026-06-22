import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin


class FCMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=2, n_init = 100, m = 2, tol = 1e-7, max_iter = 100):
        if not isinstance(n_clusters,int):
            raise TypeError("Argument 'n_clusters' must be an integer")
        if not isinstance(n_init,int):
            raise TypeError("Argument 'n_init' must be an integer")
        if m <= 1:
            raise ValueError("Fuzziness coefficient m must be greater than 1")
        
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.m = m
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, x):
        x = np.array(x)
        best_inertia = np.inf
        for _ in range(self.n_init):
            centroids, membership, inertia = self._fit_single(x)
            if inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centroids
                self.membership_ = membership
                self.labels_ = np.argmax(membership, axis=1)
        self.inertia_ = best_inertia
        return self
    
    def _fit_single(self, x):
        n = x.shape[0]
        membership = np.random.dirichlet(np.ones(self.n_clusters), size=n)
        for i in range(self.max_iter):
            centroids = self._compute_centroids(x, membership)
            membership_new = self._compute_membership(x, centroids)
            if np.linalg.norm(membership_new - membership) < self.tol:
                break
            membership = membership_new
        inertia = self._compute_inertia(x, centroids, membership)
        return centroids, membership, inertia

    def _compute_centroids(self, x, membership):
        mu_m = membership ** self.m         
        return (mu_m.T @ x) / mu_m.sum(axis=0, keepdims=True).T

    def _compute_membership(self, x, centroids):
        dists = cdist(x, centroids, metric='euclidean')
        dists = np.fmax(dists, 1e-10)
        power = 2 / (self.m - 1)
        inverted_dists = 1/dists
        numerator = inverted_dists**power
        denominator = np.sum(numerator, axis=1, keepdims=True)
        return numerator/denominator
    
    def _compute_inertia(self, x, centroids, membership):
        mu_m = membership ** self.m  
        dists = cdist(x, centroids, metric='sqeuclidean')
        return np.sum(mu_m*dists)
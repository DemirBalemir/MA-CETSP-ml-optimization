import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNNSurvival:
    """KNN survival model. Risk = -mean(neighbor survival times).
    Short-lived neighbors → less negative score → higher risk → rejected."""

    def __init__(self, n_neighbors=15):
        self.n_neighbors = n_neighbors

    def fit(self, X, survival_times):
        self.survival_times_ = np.array(survival_times, dtype=float)
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm="ball_tree")
        self.nn_.fit(X)
        return self

    def predict(self, X):
        _, indices = self.nn_.kneighbors(X)
        return -self.survival_times_[indices].mean(axis=1)

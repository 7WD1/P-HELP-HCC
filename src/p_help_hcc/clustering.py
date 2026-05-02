"""Phenotype-aware PCA + K-means branch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


@dataclass
class PhenotypeClusterer:
    k: int = 4
    pca_variance: float = 0.95
    n_init: int = 20
    random_state: int = 42
    pca: PCA | None = None
    kmeans: KMeans | None = None

    def fit(self, x: np.ndarray) -> "PhenotypeClusterer":
        x = np.asarray(x, dtype=np.float64)
        n_components = self.pca_variance if x.shape[0] > 2 and x.shape[1] > 1 else 1
        self.pca = PCA(n_components=n_components, svd_solver="full", random_state=self.random_state)
        z = self.pca.fit_transform(x)
        k = max(1, min(self.k, x.shape[0] - 1))
        self.kmeans = KMeans(n_clusters=k, init="k-means++", n_init=self.n_init, random_state=self.random_state)
        self.kmeans.fit(z)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.pca is None or self.kmeans is None:
            raise RuntimeError("PhenotypeClusterer is not fitted")
        return self.kmeans.predict(self.pca.transform(np.asarray(x, dtype=np.float64))).astype(int)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PhenotypeClusterer is not fitted")
        return self.pca.transform(np.asarray(x, dtype=np.float64))

    def one_hot(self, x: np.ndarray) -> np.ndarray:
        labels = self.predict(x)
        n_clusters = int(max(labels.max() + 1, self.kmeans.n_clusters if self.kmeans is not None else self.k))
        out = np.zeros((len(labels), n_clusters), dtype=float)
        out[np.arange(len(labels)), labels] = 1.0
        return out

    def quality(self, x: np.ndarray) -> dict[str, float]:
        z = self.transform(x)
        labels = self.predict(x)
        if len(np.unique(labels)) < 2 or len(labels) <= len(np.unique(labels)):
            return {"silhouette": float("nan"), "davies_bouldin": float("nan"), "calinski_harabasz": float("nan")}
        return {
            "silhouette": float(silhouette_score(z, labels)),
            "davies_bouldin": float(davies_bouldin_score(z, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(z, labels)),
        }

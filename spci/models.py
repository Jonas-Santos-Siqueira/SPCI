# SPDX-License-Identifier: MIT
import numpy as np

# Try to import a quantile RF implementation compatible with authors' code
_QRF_BACKEND = None
try:
    from sklearn_quantile import RandomForestQuantileRegressor as _SkQRF
    _QRF_BACKEND = "sklearn_quantile"
except Exception:
    try:
        # skranger exposes quantile predictions via predict with quantiles=True
        from skranger.ensemble import RangerForestRegressor as _RangerQRF
        _QRF_BACKEND = "skranger"
    except Exception:
        _QRF_BACKEND = None

from sklearn.neighbors import NearestNeighbors

class QRFQuantileRegressor:
    """
    Wrapper offering a unified API:
    - fit(X, y, sample_weight=None)
    - predict_quantiles(X_new, quantiles=[...]) -> (n, len(quantiles))
    Prefers sklearn-quantile, falls back to skranger if available.
    """
    def __init__(self, n_estimators=200, random_state=0, **kwargs):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.kwargs = kwargs
        if _QRF_BACKEND is None:
            raise ImportError("No QRF backend found. Install 'sklearn-quantile' or 'skranger'.")
        if _QRF_BACKEND == "sklearn_quantile":
            self.model = _SkQRF(n_estimators=n_estimators, random_state=random_state, **kwargs)
        elif _QRF_BACKEND == "skranger":
            self.model = _RangerQRF(num_trees=n_estimators, random_state=random_state, quantiles=True, **kwargs)

    def fit(self, X, y, sample_weight=None):
        if _QRF_BACKEND == "sklearn_quantile":
            return self.model.fit(X, y, sample_weight=sample_weight)
        elif _QRF_BACKEND == "skranger":
            # skranger doesn't support sample_weight for quantiles
            return self.model.fit(X, y)

    def predict_quantiles(self, X, quantiles):
        qs = np.asarray(quantiles, dtype=float)
        if _QRF_BACKEND == "sklearn_quantile":
            return self.model.predict(X, quantiles=qs)
        elif _QRF_BACKEND == "skranger":
            # skranger returns (n, n_quantiles) when predict with specified quantiles
            return self.model.predict(X, quantiles=qs)

class KNNQuantileRegressor:
    """
    Simple KNN-based quantile predictor used as a fallback when QRF is unavailable.
    Not in the original paper, but useful for environments without QRF deps.
    API mirrors QRFQuantileRegressor.
    """
    def __init__(self, n_neighbors=50):
        self.k = n_neighbors
        self.nn = NearestNeighbors(n_neighbors=self.k, algorithm="auto")
        self.y = None
        self.X = None

    def fit(self, X, y, sample_weight=None):
        self.X = np.asarray(X)
        self.y = np.asarray(y).reshape(-1)
        self.nn.fit(self.X)
        return self

    def predict_quantiles(self, X, quantiles):
        X = np.asarray(X)
        idx = self.nn.kneighbors(X, return_distance=False)
        # compute empirical quantiles of neighbors' targets
        out = np.zeros((X.shape[0], len(quantiles)), dtype=float)
        for i, q in enumerate(quantiles):
            out[:, i] = np.quantile(self.y[idx], q, axis=1, method="linear")
        return out

import numpy as np
_BACKEND = None
_SkQRF = None
_RangerQRF = None
try:
    from sklearn_quantile import RandomForestQuantileRegressor as _SkQRF
    _BACKEND = "sklearn_quantile"
except Exception:
    try:
        from skranger.ensemble import RangerForestRegressor as _RangerQRF
        _BACKEND = "skranger"
    except Exception:
        _BACKEND = None

class QRFQuantileRegressor:
    def __init__(self, n_estimators=200, random_state=0, **kwargs):
        if _BACKEND is None:
            raise ImportError("No QRF backend available. Install 'sklearn-quantile' or 'skranger'.")
        self.backend = _BACKEND
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.kwargs = kwargs
        if self.backend == "sklearn_quantile":
            self.model = _SkQRF(n_estimators=self.n_estimators, random_state=self.random_state, **kwargs)
        elif self.backend == "skranger":
            self.model = _RangerQRF(num_trees=self.n_estimators, random_state=self.random_state, quantiles=True, **kwargs)
    def fit(self, X, y, sample_weight=None):
        if self.backend == "sklearn_quantile":
            return self.model.fit(X, y, sample_weight=sample_weight)
        else:
            return self.model.fit(X, y)
    def predict_quantiles(self, X, quantiles):
        q = np.asarray(quantiles, float)
        return self.model.predict(X, quantiles=q)

class KNNQuantileRegressor:
    def __init__(self, n_neighbors=50):
        self.k = int(n_neighbors)
        self.X_ = None; self.y_ = None
    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, float); y = np.asarray(y, float).ravel()
        if X.ndim == 1: X = X.reshape(-1, 1)
        self.X_, self.y_ = X, y
        return self
    def predict_quantiles(self, X, quantiles):
        X = np.asarray(X, float)
        if X.ndim == 1: X = X.reshape(1, -1)
        qs = np.atleast_1d(quantiles)
        out = np.zeros((X.shape[0], len(qs)), dtype=float)
        for i, x in enumerate(X):
            d = np.linalg.norm(self.X_ - x, axis=1)
            idx = np.argsort(d)[:max(1, self.k)]
            vals = self.y_[idx]
            for j, q in enumerate(qs):
                out[i, j] = np.quantile(vals, q)
        return out

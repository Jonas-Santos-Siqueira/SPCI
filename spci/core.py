import numpy as np
from .utils import generate_bootstrap_samples, pick_beta, pick_beta_horizon
from .models import QRFQuantileRegressor, KNNQuantileRegressor

class SPCI:
    def __init__(self, base_model="rf", B=30, alpha=0.1, w=20, bins=5, qrf_backend="auto", random_state=0):
        self.base_model = base_model
        self.B = int(B); self.alpha = float(alpha)
        self.w = int(w); self.bins = int(bins)
        self.qrf_backend = qrf_backend
        self.random_state = int(random_state)
        self.models_ = None; self.in_boot_ = None; self.residuals_ = None
    def _make_qregr(self):
        if self.qrf_backend == "knn":
            return KNNQuantileRegressor(n_neighbors=min(50, max(5, self.w)))
        try:
            return QRFQuantileRegressor(n_estimators=200, random_state=self.random_state)
        except Exception:
            return KNNQuantileRegressor(n_neighbors=min(50, max(5, self.w)))
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float).ravel(); n = len(y)
        boot = generate_bootstrap_samples(n, n, self.B, rng=self.random_state)
        self.models_ = []; self.in_boot_ = np.zeros((self.B, n), dtype=bool)
        from sklearn.ensemble import RandomForestRegressor
        if self.base_model == "rf" or self.base_model is None:
            def ctor(seed): return RandomForestRegressor(n_estimators=200, random_state=seed)
        else:
            from copy import deepcopy
            def ctor(seed):
                try:
                    from sklearn.base import clone
                    return clone(self.base_model)
                except Exception:
                    return deepcopy(self.base_model)
        preds_train = np.zeros((self.B, n))
        for b in range(self.B):
            model = ctor(self.random_state + b + 1)
            idx = boot[b]; self.in_boot_[b, idx] = True
            model.fit(X[idx], y[idx]); self.models_.append(model)
            preds_train[b] = model.predict(X)
        center = np.zeros(n)
        for i in range(n):
            mask = ~self.in_boot_[:, i]
            if not np.any(mask): mask[:] = True
            center[i] = preds_train[mask, i].mean()
        self.residuals_ = y - center
        return self
    def _center_predict(self, X_new):
        preds = np.column_stack([m.predict(X_new) for m in self.models_])
        return preds.mean(axis=1)
    def predict_interval(self, X_new, y_true=None):
        X_new = np.asarray(X_new, float)
        H = X_new.shape[0]
        center = self._center_predict(X_new)
        lower = np.empty(H); upper = np.empty(H)
        resid_series = list(self.residuals_.ravel())
        if y_true is None:
            for h in range(1, H+1):
                qregr = self._make_qregr()
                ql, qh, b = pick_beta_horizon(qregr, resid_series, self.w, self.alpha, self.bins, horizon=h)
                lower[h-1] = center[h-1] + ql
                upper[h-1] = center[h-1] + qh
            return {"lower": lower, "upper": upper, "center": center}
        y_true = np.asarray(y_true, float).ravel()
        if y_true.size != H: raise ValueError("y_true must have same length as X_new")
        for t in range(H):
            qregr = self._make_qregr()
            ql, qh, b = pick_beta(qregr, resid_series, self.w, self.alpha, self.bins)
            lower[t] = center[t] + ql
            upper[t] = center[t] + qh
            resid_series.append(float(y_true[t] - center[t]))
        return {"lower": lower, "upper": upper, "center": center}

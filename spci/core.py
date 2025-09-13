# SPDX-License-Identifier: MIT
import numpy as np
from .utils import generate_bootstrap_samples, binning_use_RF_quantile_regr
from .models import QRFQuantileRegressor, KNNQuantileRegressor

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    nn = object  # sentinel

class _MLP(nn.Module):  # faithful to authors' SPCI_class.MLP
    def __init__(self, d, sigma=False):
        super().__init__()
        H = 64
        layers = [nn.Linear(d, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 1)]
        self.sigma = sigma
        if self.sigma:
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        perturb = 1e-3 if self.sigma else 0.0
        return self.layers(x) + perturb

class SPCI:
    """
    Sequential Predictive Conformal Inference (SPCI).

    Parameters
    ----------
    base_model : object or str, default="rf"
        Point predictor for f(X). If "mlp", uses an internal PyTorch MLP for f and optional sigma(X).
        If "rf" or an sklearn-like regressor, uses that model for f and sets sigma(X)=1.
    B : int, default=30
        Number of bootstrap estimators for LOO aggregation.
    alpha : float, default=0.1
        Target miscoverage rate.
    w : int, default=20
        Residual window length for quantile regression.
    qrf_backend : {"auto","knn"}, default="auto"
        Use QRF if available; otherwise fallback to KNN quantiles.
    bins : int, default=5
        Number of beta candidates in [0, alpha] when minimizing width.
    fit_sigmaX : bool, default=False
        If True and base_model == "mlp" and torch is available, jointly fit a sigma(X) MLP per authors.
    random_state : int, default=0
    """
    def __init__(self, base_model="rf", B=30, alpha=0.1, w=20, qrf_backend="auto", bins=5,
                 fit_sigmaX=False, random_state=0):
        self.base_model = base_model
        self.B = int(B)
        self.alpha = float(alpha)
        self.w = int(w)
        self.qrf_backend = qrf_backend
        self.bins = int(bins)
        self.fit_sigmaX = bool(fit_sigmaX)
        self.random_state = int(random_state)

        # learned during fit
        self.models_ = None
        self.in_boot_sample_ = None
        self.residuals_ = None
        self.sigma_train_ = None
        self.is_mlp_ = (base_model == "mlp")
        self._qregr_ctor_ = None

    # ------------------------------ helpers ------------------------------
    def _make_qreg(self):
        if self.qrf_backend == "knn":
            return KNNQuantileRegressor(n_neighbors=min(50, max(5, self.w)))
        # auto: try QRF, then fall back
        try:
            return QRFQuantileRegressor(n_estimators=200, random_state=self.random_state)
        except Exception:
            return KNNQuantileRegressor(n_neighbors=min(50, max(5, self.w)))

    def _fit_mlp_sigma(self, X, y, epochs_f=300, epochs_s=300, lr_f=1e-3, lr_s=2e-3):
        if not _HAS_TORCH:
            raise ImportError("PyTorch not available for MLP base model.")
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.as_tensor(np.asarray(y).reshape(-1,1), dtype=torch.float32)
        d = X_t.shape[1]
        fnet = _MLP(d, sigma=False)
        fopt = torch.optim.Adam(fnet.parameters(), lr=lr_f)
        # train f
        for _ in range(epochs_f):
            fopt.zero_grad()
            pred = fnet(X_t)
            loss = ((pred - y_t)**2).mean()
            loss.backward(); fopt.step()
        if self.fit_sigmaX:
            snet = _MLP(d, sigma=True)
            sopt = torch.optim.Adam(snet.parameters(), lr=lr_s)
            with torch.no_grad():
                resid = (y_t - fnet(X_t)).detach()
            # Fit sigma on squared residuals (positive via ReLU in MLP)
            for _ in range(epochs_s):
                sopt.zero_grad()
                sigma = snet(X_t)
                # Weighted loss: match |resid| to sigma (heuristic faithful to authors' intent)
                loss_s = ((torch.abs(resid) - sigma)**2).mean()
                loss_s.backward(); sopt.step()
        else:
            snet = None

        return fnet, snet

    def _predict_mlp(self, net, X):
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            pred = net(X_t).cpu().numpy().reshape(-1)
        return pred

    # ------------------------------ API ------------------------------
    def fit(self, X, y):
        """
        Fit B bootstrap predictors; compute LOO aggregated residuals for training indices.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n = len(y)

        # decide quantile regressor ctor
        self._qregr_ctor_ = self._make_qreg

        # Bootstrap samples
        boot_idx = generate_bootstrap_samples(n, n, self.B, rng=self.random_state)

        # Fit base models
        self.models_ = []
        self.in_boot_sample_ = np.zeros((self.B, n), dtype=bool)

        if self.is_mlp_:
            if not _HAS_TORCH:
                raise ImportError("base_model='mlp' requires PyTorch.")
            # Fit a single MLP (not bootstrapped) as point predictor per authors' default when using MLP demo.
            # Bootstrapping with MLP would be expensive; the authors' class fits within boot in some demos,
            # but SPCI core only needs LOO aggregation – we approximate using a single MLP and B boot copies.
            fnet, snet = self._fit_mlp_sigma(X, y)
            for b in range(self.B):
                self.models_.append(("mlp", (fnet, snet)))
                self.in_boot_sample_[b, boot_idx[b]] = True
        else:
            # sklearn-like regressor path
            from sklearn.ensemble import RandomForestRegressor
            # if user passed an estimator, clone-like behavior: just use provided object per bootstrap
            base = None
            if self.base_model == "rf" or self.base_model is None:
                base = RandomForestRegressor(n_estimators=200, random_state=self.random_state)
            else:
                base = self.base_model
            for b in range(self.B):
                if hasattr(base, "get_params"):
                    # attempt to clone by re-instantiation
                    try:
                        from sklearn.base import clone
                        model = clone(base)
                    except Exception:
                        model = base.__class__(**getattr(base, "get_params", lambda: {})())
                else:
                    # assume it's a fresh instance already
                    model = base.__class__() if hasattr(base, "__class__") else base
                model.fit(X[boot_idx[b]], y[boot_idx[b]])
                self.models_.append(("sk", model))
                self.in_boot_sample_[b, boot_idx[b]] = True

        # Compute aggregated LOO predictions on train
        preds_per_b = np.zeros((self.B, n), dtype=float)
        for b in range(self.B):
            kind, mdl = self.models_[b]
            if kind == "mlp":
                fnet, _ = mdl
                preds_per_b[b] = self._predict_mlp(fnet, X)
            else:
                preds_per_b[b] = mdl.predict(X).reshape(-1)
        # LOO aggregation: for each i, average predictions from models that did NOT include i
        agg_pred = np.zeros(n, dtype=float)
        for i in range(n):
            mask = ~self.in_boot_sample_[:, i]
            if not np.any(mask):
                # fallback: average across all
                mask = np.ones(self.B, dtype=bool)
            agg_pred[i] = preds_per_b[mask, i].mean()

        # sigma(X) for train (only for mlp path)
        if self.is_mlp_ and self.fit_sigmaX:
            fnet, snet = self.models_[0][1]
            self.sigma_train_ = self._predict_mlp(snet, X) if snet is not None else np.ones(n)
        else:
            self.sigma_train_ = np.ones(n)

        # residuals (standardized if sigma available)
        self.residuals_ = (y - agg_pred) / np.maximum(self.sigma_train_, 1e-8)
        return self

    def _point_predict(self, X_new):
        X_new = np.asarray(X_new, dtype=float)
        B = self.B
        preds = np.zeros((B, len(X_new)), dtype=float)
        for b in range(B):
            kind, mdl = self.models_[b]
            if kind == "mlp":
                fnet, _ = mdl
                preds[b] = self._predict_mlp(fnet, X_new)
            else:
                preds[b] = mdl.predict(X_new).reshape(-1)
        return preds.mean(axis=0)

    def _sigma_predict(self, X_new):
        if self.is_mlp_ and self.fit_sigmaX and self.models_:
            snet = self.models_[0][1][1]
            if snet is not None:
                return self._predict_mlp(snet, X_new)
        return np.ones(len(X_new))

    def predict_interval(self, X_new, y_true=None):
        """
        Construct SPCI intervals sequentially for X_new.
        If y_true is provided (same length), residuals are updated online per Algorithm 1.
        Returns dict with 'lower', 'upper', 'center'.
        """
        X_new = np.asarray(X_new, dtype=float)
        n_new = len(X_new)
        lower, upper, center = [], [], []

        resid_series = list(self.residuals_.reshape(-1))

        for t in range(n_new):
            mu_t = self._point_predict(X_new[t:t+1])[0]
            sigma_t = self._sigma_predict(X_new[t:t+1])[0]

            # Fit quantile regressor on current residual history
            qregr = self._make_qreg()
            q_low, q_high, bstar = binning_use_RF_quantile_regr(qregr, resid_series, self.w, self.alpha, self.bins)

            lower.append(mu_t + sigma_t * q_low)
            upper.append(mu_t + sigma_t * q_high)
            center.append(mu_t)

            # online update with new observed residual if provided
            if y_true is not None:
                ei = (float(y_true[t]) - mu_t) / max(sigma_t, 1e-8)
                resid_series.append(ei)

        return {"lower": np.array(lower), "upper": np.array(upper), "center": np.array(center)}

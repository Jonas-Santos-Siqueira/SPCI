import numpy as np
def generate_bootstrap_samples(n, m, B, rng=None):
    rng = np.random.default_rng(rng)
    return rng.integers(0, n, size=(B, m))
def lag_windows(resid, w, horizon=1):
    r = np.asarray(resid, float).ravel()
    T = len(r)
    if T <= w + (horizon - 1):
        return np.empty((0, w)), np.empty((0,)), None
    X = np.lib.stride_tricks.sliding_window_view(r, w)[:-1 - (horizon - 1)]
    y = r[w + (horizon - 1):]
    x_next = r[-w:].reshape(1, -1)
    return X, y, x_next
def beta_grid(alpha, bins=5, eps=1e-8):
    if bins <= 1: return np.array([alpha/2.0])
    return np.linspace(eps, alpha - eps, bins)
def pick_beta(qregr, resid_series, w, alpha, bins=5):
    X, y, x_next = lag_windows(resid_series, w, horizon=1)
    if y.size == 0:
        ql = float(np.quantile(resid_series, alpha/2.0))
        qh = float(np.quantile(resid_series, 1 - alpha/2.0))
        return ql, qh, alpha/2.0
    qregr.fit(X, y)
    betas = beta_grid(alpha, bins=bins)
    Qs = np.unique(np.r_[betas, 1 - alpha + betas])
    pred = qregr.predict_quantiles(x_next, quantiles=Qs).ravel()
    qmap = {q: pred[i] for i, q in enumerate(Qs)}
    widths = np.array([qmap[1 - alpha + b] - qmap[b] for b in betas])
    j = int(np.argmin(widths))
    return float(qmap[betas[j]]), float(qmap[1 - alpha + betas[j]]), float(betas[j])
def pick_beta_horizon(qregr, resid_series, w, alpha, bins, horizon):
    X, y, x_next = lag_windows(resid_series, w, horizon=horizon)
    if y.size == 0:
        ql = float(np.quantile(resid_series, alpha/2.0))
        qh = float(np.quantile(resid_series, 1 - alpha/2.0))
        return ql, qh, alpha/2.0
    qregr.fit(X, y)
    betas = beta_grid(alpha, bins=bins)
    Qs = np.unique(np.r_[betas, 1 - alpha + betas])
    pred = qregr.predict_quantiles(x_next, quantiles=Qs).ravel()
    qmap = {q: pred[i] for i, q in enumerate(Qs)}
    widths = np.array([qmap[1 - alpha + b] - qmap[b] for b in betas])
    j = int(np.argmin(widths))
    return float(qmap[betas[j]]), float(qmap[1 - alpha + betas[j]]), float(betas[j])

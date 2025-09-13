# SPDX-License-Identifier: MIT
import numpy as np

def generate_bootstrap_samples(n, m, B, rng=None):
    """
    Return a (B, m) array of indices: each row is a bootstrap sample of size m from range(n).
    """
    rng = np.random.default_rng(None if rng is None else rng)
    return rng.integers(0, n, size=(B, m), endpoint=False)

def strided_windows(arr, w):
    """
    Build (T-w) x w lagged windows of a 1D array.
    X_t = [arr[t+w-1], ..., arr[t]]
    y_t = arr[t+w]
    """
    arr = np.asarray(arr).astype(float).reshape(-1)
    T = len(arr)
    if T <= w:
        return np.empty((0, w)), np.empty((0,))
    X = np.stack([arr[i:(i+T-w)] for i in range(w)][::-1], axis=1)
    y = arr[w:]
    return X, y

def beta_grid(alpha, bins=5, eps=1e-6):
    """
    Candidate beta in [0, alpha], excluding endpoints to avoid degeneracy.
    """
    if bins <= 1:
        return np.array([alpha/2.0])
    return np.linspace(eps, alpha-eps, bins)

def binning_use_RF_quantile_regr(regr, resid_series, w, alpha, bins=5):
    """
    Train quantile regressor on residual windows and choose beta minimizing interval width, per authors' utils_SPCI.py.
    Returns (q_low, q_high, beta_star).
    """
    resid_series = np.asarray(resid_series).reshape(-1)
    X, y = strided_windows(resid_series, w)
    if len(y) == 0:
        # Not enough residuals yet: fall back to symmetric quantiles from empirical residuals
        low_q = np.quantile(resid_series, alpha/2.0)
        high_q = np.quantile(resid_series, 1 - alpha/2.0)
        return float(low_q), float(high_q), alpha/2.0

    # Fit the quantile regressor
    regr.fit(X, y)

    # feature for next-step residual quantile
    x_next = resid_series[-w:].reshape(1, -1)

    beta_ls = beta_grid(alpha, bins=bins)
    # to avoid duplicate evaluations, query all quantiles at once
    quantiles = np.unique(np.r_[beta_ls, 1 - alpha + beta_ls])
    pred = regr.predict_quantiles(x_next, quantiles=quantiles).reshape(1, -1)
    # map quantile -> value
    qmap = {q: pred[0, i] for i, q in enumerate(quantiles)}

    widths = (np.array([qmap[1 - alpha + b] - qmap[b] for b in beta_ls]))
    i_star = int(np.argmin(widths))
    bstar = float(beta_ls[i_star])
    return float(qmap[bstar]), float(qmap[1 - alpha + bstar]), bstar

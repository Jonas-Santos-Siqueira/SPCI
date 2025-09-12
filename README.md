# SPCI (Fidelity, QRF-first)
Implements SPCI faithful to authors: LOO residuals (bagging), conditional QR on residual windows (QRF preferred), beta search in [0,alpha], online update, and multi-step.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel setuptools numpy scipy scikit-learn
pip install sklearn-quantile   # or: pip install skranger
pip install -e .
```

## Quick start
```python
import numpy as np
from spci import SPCI

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5))
beta = rng.normal(size=(5,)); y = X @ beta + 0.3 * rng.standard_t(df=5, size=200)

m = SPCI(base_model="rf", B=30, alpha=0.1, w=20, bins=7, qrf_backend="auto", random_state=0)
m.fit(X[:160], y[:160])
res = m.predict_interval(X[160:], y_true=y[160:])  # online
print(res["lower"].shape, res["upper"].shape)
```

## Multi-step
```python
H = 12
res = m.predict_interval(X[-H:])  # no y_true -> multi-step (h=1..H)
```

## Smoke tests
```bash
python smoke_test_qrf.py   # requires sklearn-quantile or skranger installed
python smoke_test_knn.py   # fallback KNN (no extra deps)
```

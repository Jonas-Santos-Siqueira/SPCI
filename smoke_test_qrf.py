import numpy as np
from spci import SPCI
from spci.models import QRFQuantileRegressor

# Ensure QRF available
_ = QRFQuantileRegressor()

rng = np.random.default_rng(0)
n = 240
X = rng.normal(size=(n, 5))
beta = rng.normal(size=(5,))
y = X @ beta + 0.3 * rng.standard_t(df=5, size=n)

m = SPCI(base_model='rf', B=30, alpha=0.1, w=20, bins=7, qrf_backend='auto', fit_sigmaX=False, random_state=42)
m.fit(X[:180], y[:180])

res = m.predict_interval(X[180:], y_true=y[180:])
inside = (y[180:] >= res["lower"]) & (y[180:] <= res["upper"])

print("Coverage (QRF, n={}): {:.3f}".format(len(inside), inside.mean()))
print("Avg width:", np.mean(res["upper"] - res["lower"]))
print("First 3 intervals:", list(zip(res["lower"][:3], res["center"][:3], res["upper"][:3])))

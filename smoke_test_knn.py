import numpy as np
from spci import SPCI

rng = np.random.default_rng(1)
n = 200
X = rng.normal(size=(n, 5))
beta = rng.normal(size=(5,))
y = X @ beta + 0.35 * rng.standard_t(df=5, size=n)

m = SPCI(base_model='rf', B=20, alpha=0.1, w=15, bins=5, qrf_backend='knn', fit_sigmaX=False, random_state=42)
m.fit(X[:150], y[:150])

res = m.predict_interval(X[150:], y_true=y[150:])
inside = (y[150:] >= res["lower"]) & (y[150:] <= res["upper"])

print("Coverage (KNN fallback, n={}): {:.3f}".format(len(inside), inside.mean()))
print("Avg width:", float(np.mean(res["upper"] - res["lower"])))
print("First 3 centers:", list(np.round(res["center"][:3], 4)))
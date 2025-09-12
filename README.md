# SPCI (Python)
Sequential Predictive Conformal Inference (time series) — API.

- Bagging preditivo para $$f̂(x)$$; resíduos OOB.
- Largura de intervalo aprendida online via regressão de quantis dos resíduos passados (k-NN por padrão).

Quick start:
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from spci import SPCIModel

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3)); y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.5, 200)
m = SPCIModel(base_model=LinearRegression(), B=20, alpha=0.1, lag=30)
m.fit(X[:150], y[:150])
res = m.get_prediction(X[150:], y_true=y[150:])
print(res.summary())
```

# SPDX-License-Identifier: MIT
"""
spci: Sequential Predictive Conformal Inference (SPCI)

Implements SPCI faithful to the authors' reference code:
- LOO residuals from B bootstrap base models (EnbPI-style)
- Conditional quantile regression on residual windows via QRF (default) or KNN fallback
- Beta search to minimize interval width (Algorithm 1, eq. (10)-(11))
- Optional MLP-based sigma(X) when using internal MLP point predictor (as in authors' SPCI_class.py)
"""
from .core import SPCI
from .models import QRFQuantileRegressor, KNNQuantileRegressor
from . import utils

__all__ = ["SPCI", "QRFQuantileRegressor", "KNNQuantileRegressor", "utils"]
__version__ = "0.2.0"

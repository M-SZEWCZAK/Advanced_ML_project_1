"""Smoke tests for the FISTA logistic-lasso model."""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from project1.models.fista_logistic_lasso import FistaLogisticLassoRegressionClassifierFamily


def test_fista_predict_and_predict_proba_return_valid_outputs():
    """The FISTA model should emit probabilities in [0, 1] and class labels in {0, 1}."""
    X = np.array(
        [
            [1.0, -1.0, 0.0],
            [1.0, -0.5, 0.2],
            [1.0, 0.1, 0.3],
            [1.0, 0.5, 0.4],
            [1.0, 1.0, 0.8],
            [1.0, 1.2, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    model = FistaLogisticLassoRegressionClassifierFamily(lambdas=[0.01], n_iter=50)
    model.fit(X, y)

    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    assert probabilities.shape == (len(X), 2)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert set(np.unique(predictions)).issubset({0, 1})

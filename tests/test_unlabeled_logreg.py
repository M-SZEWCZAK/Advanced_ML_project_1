"""Smoke tests for the unlabeled logistic-regression model."""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from project1.models.unlabeled_logreg import UnlabeledLogReg


def test_unlabeled_logreg_completes_labels_and_predicts_probabilities():
    """The model should fill missing labels and expose predict_proba after fitting."""
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
    y_obs = np.array([0, -1, 0, 1, -1, 1], dtype=int)
    y_full = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    model = UnlabeledLogReg(imputation_type="logistic", lambdas=[0.01], n_iter=50)

    completed_labels = model.complete_labels(X, y_obs)
    assert -1 not in completed_labels

    model.fit(X, y_obs)
    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert set(np.unique(model.predict(X))).issubset({0, 1})
    assert set(np.unique(y_full)).issubset({0, 1})

"""Smoke tests for missing-label generators."""

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from project1.missingness.generators import MAR1, MAR2, MCAR, MNAR


@pytest.fixture
def sample_data():
    """Provide a small binary dataset for missingness smoke tests."""
    X = np.array(
        [
            [1.0, -0.5, 0.2],
            [0.5, 0.1, -0.4],
            [1.2, 0.3, 0.8],
            [-0.8, -0.2, 0.5],
            [0.1, 0.6, -0.7],
            [-0.3, 0.4, 0.9],
            [0.9, -0.1, -0.2],
            [-0.5, 0.2, 0.3],
        ]
    )
    y = np.array([0, 1, 1, 0, 1, 0, 1, 0], dtype=int)
    return X, y


@pytest.mark.parametrize(
    ("generator_name", "generator"),
    [
        ("MCAR", lambda X, y: MCAR(X, y, missing_proba=0.3, seed=0)),
        ("MAR1", lambda X, y: MAR1(X, y, missing_proba=0.3, missing_influence_col_index=0, seed=0)),
        ("MAR2", lambda X, y: MAR2(X, y, missing_proba=0.3, seed=0)),
        ("MNAR", lambda X, y: MNAR(X, y, missing_proba=0.3, seed=0)),
    ],
)
def test_missingness_generators_return_valid_observed_labels(sample_data, generator_name, generator):
    """Each generator should preserve length and use only labels from {0, 1, -1}."""
    X, y = sample_data

    y_obs = generator(X, y)

    assert len(y_obs) == len(y) == X.shape[0], f"{generator_name} changed the label length."
    assert set(np.unique(y_obs)).issubset({-1, 0, 1}), f"{generator_name} produced invalid observed labels."

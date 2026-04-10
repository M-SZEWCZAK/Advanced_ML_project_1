"""Minimal end-to-end smoke test for the experiment pipeline."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from project1.experiments.runner import run_single_experiment


def test_single_experiment_pipeline_smoke():
    """Run one lightweight experiment configuration end-to-end."""
    result = run_single_experiment(
        dataset="breast_cancer",
        scheme="mcar",
        method="naive",
        missing_rate=0.2,
        seed=42,
    )

    assert result["dataset"] == "breast_cancer"
    assert result["scheme"] == "mcar"
    assert result["method"] == "naive"
    for metric_name in ["accuracy", "balanced_accuracy", "f1", "roc_auc"]:
        assert 0.0 <= result[metric_name] <= 1.0

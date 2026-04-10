"""Central experiment configuration values for Project 1."""

# Datasets currently supported by the local ARFF-based data layer.
EXPERIMENT_DATASETS = [
    "breast_cancer",
    "ionosphere",
    "spambase",
    "madelon",
]

# Missing-label generation schemes supported by the experiment runner.
MISSINGNESS_SCHEMES = [
    "mcar",
    "mar1",
    "mar2",
    "mnar",
]

# Modeling approaches available in the single-experiment runner.
EXPERIMENT_METHODS = [
    "naive",
    "oracle",
    "unlabeled",
]

# Random seeds used to repeat experiments across different train/test splits
# and missing-label realizations.
EXPERIMENT_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]

# Default missing-label rates that can be swept in experiments.
EXPERIMENT_MISSING_RATES = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
]

# Explicit MCAR-focused sweep values kept separate so bulk runners can use
# a named configuration for MCAR analyses without hardcoding rates elsewhere.
MCAR_ANALYSIS_MISSING_RATES = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
]


__all__ = [
    "EXPERIMENT_DATASETS",
    "MISSINGNESS_SCHEMES",
    "EXPERIMENT_METHODS",
    "EXPERIMENT_SEEDS",
    "EXPERIMENT_MISSING_RATES",
    "MCAR_ANALYSIS_MISSING_RATES",
]

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

# Label-completion strategies evaluated for the unlabeled method in the full
# experiment runner.
UNLABELED_LABEL_COMPLETION_METHODS = [
    "logistic",
    "knn",
]

# Default label-completion strategy used when a single unlabeled experiment is
# launched without overriding the completion method explicitly.
DEFAULT_UNLABELED_LABEL_COMPLETION_METHOD = UNLABELED_LABEL_COMPLETION_METHODS[0]

# Random seeds used to repeat experiments across different train/test splits
# and missing-label realizations.
EXPERIMENT_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]

# Default missing-label rates available for experiment sweeps.
EXPERIMENT_MISSING_RATES = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
]

# Baseline missing-label rate used for non-MCAR schemes in the full runner.
DEFAULT_MISSING_RATE = 0.3

# Explicit MCAR-focused sweep values kept separate so bulk runners can use
# a named configuration for MCAR analyses without hardcoding rates elsewhere.
MCAR_ANALYSIS_MISSING_RATES = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
]

# Central mapping from missingness scheme to the rates used by the full runner.
# MCAR gets a full sweep for separate sensitivity analysis, while the other
# schemes currently run at one shared default rate.
MISSING_RATES_BY_SCHEME = {
    "mcar": MCAR_ANALYSIS_MISSING_RATES,
    "mar1": [DEFAULT_MISSING_RATE],
    "mar2": [DEFAULT_MISSING_RATE],
    "mnar": [DEFAULT_MISSING_RATE],
}


__all__ = [
    "EXPERIMENT_DATASETS",
    "MISSINGNESS_SCHEMES",
    "EXPERIMENT_METHODS",
    "UNLABELED_LABEL_COMPLETION_METHODS",
    "DEFAULT_UNLABELED_LABEL_COMPLETION_METHOD",
    "EXPERIMENT_SEEDS",
    "EXPERIMENT_MISSING_RATES",
    "DEFAULT_MISSING_RATE",
    "MCAR_ANALYSIS_MISSING_RATES",
    "MISSING_RATES_BY_SCHEME",
]

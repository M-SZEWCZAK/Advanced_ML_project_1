# Logistic Regression with Missing Labels

## Project Title

Logistic Regression with Missing Labels

## Project Description

This project investigates binary classification in the presence of **missing labels**, with a particular focus on logistic-regression-based methods and experimental comparison under controlled missingness mechanisms. In many practical settings, feature vectors are available for all observations, while class labels are observed only for a subset of the training data. This creates a learning problem in which standard supervised methods cannot be applied directly without additional assumptions or preprocessing.

The project considers several missing-label mechanisms:

- **MCAR (Missing Completely At Random)**: label missingness is independent of both the observed features and the true label.
- **MAR (Missing At Random)**: label missingness depends on observed covariates. In this repository, this setting is represented by `mar1` and `mar2`.
- **MNAR (Missing Not At Random)**: label missingness depends on the label itself or on latent information related to the label.

The repository includes a modular data pipeline, baseline models, a custom FISTA-based logistic lasso implementation, an `UnlabeledLogReg` approach for label completion, and experiment runners for single and full experimental sweeps.

## Project Structure

- `src/`
  - Core project code.
  - `src/project1/data/` contains dataset loading, splitting, and preprocessing logic.
  - `src/project1/models/` contains baseline models, FISTA logistic lasso, and `UnlabeledLogReg`.
  - `src/project1/missingness/` contains missing-label generators.
  - `src/project1/experiments/` contains experiment configuration, aggregation, and runner utilities.
- `scripts/`
  - Command-line entry points for data preparation and experiment execution.
- `data/`
  - Local input datasets used by the project.
  - The current implementation expects local ARFF files under `data/raw/`.
- `outputs/`
  - Generated outputs, including CSV result tables and auxiliary experiment artifacts.
- `tests/`
  - Smoke tests covering missingness generation, FISTA, unlabeled learning, and the end-to-end pipeline.
- `notebooks/`
  - Auxiliary notebook material retained for reference and exploratory work.

## Installation

Install the required dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Prepare a dataset using the local data pipeline:

```bash
python3 scripts/prepare_datasets.py breast_cancer
```

Run a single experiment:

```bash
python3 scripts/run_single_experiment.py \
  --dataset breast_cancer \
  --scheme mcar \
  --method naive \
  --missing-rate 0.2 \
  --seed 42 \
  --label-completion-method logistic
```

Run the full configured experiment grid:

```bash
python3 scripts/run_all_experiments.py
```

## Experiments

The experimental layer currently supports the following elements.

### Datasets

- `breast_cancer`
- `ionosphere`
- `spambase`
- `madelon`

### Missingness Schemes

- `mcar`
- `mar1`
- `mar2`
- `mnar`

### Methods

- `naive`
  - Trains only on observations with available labels.
- `oracle`
  - Trains on the full label vector and serves as a reference upper bound.
- `unlabeled`
  - First completes missing labels in the training set, then trains the downstream classifier.

## Outputs

Experimental results are saved as CSV files in:

- `outputs/tables/`

Typical output files include:

- raw experiment tables, for example `all_experiments_results_<timestamp>.csv`
- aggregated summaries, for example `all_experiments_summary_<timestamp>.csv`

These files are intended to support later quantitative analysis and report preparation.

## Testing

Run the test suite with `pytest`:

```bash
pytest
```

To execute only the smoke tests added for the current pipeline:

```bash
pytest tests/test_missingness.py tests/test_fista.py tests/test_unlabeled_logreg.py tests/test_pipeline_smoke.py
```

## Notes

- The notebook in `notebooks/` is retained as auxiliary material only and should be treated as a reference artifact rather than the main execution path.
- The principal project logic is implemented in `src/`, and all scripts are designed to rely on that modular codebase.

# Results Tables Description

This directory contains the final CSV files used for analysis and reporting after the experiment pipeline has been completed.

## Files

### `final_results.csv`

This file contains the full set of experiment runs.

Each row corresponds to one concrete configuration of:

- dataset
- missingness scheme
- method
- label-completion method
- random seed
- missing rate

The most important columns are:

- `dataset`
- `scheme`
- `method`
- `seed`
- `missing_rate`
- `label_completion_method`
- `status`
- `accuracy`
- `balanced_accuracy`
- `f1`
- `roc_auc`
- `train_size`
- `valid_size`
- `test_size`
- `n_features_before_preprocessing`
- `n_features_after_preprocessing`
- `observed_label_fraction_train`

Use this file for:

- full quantitative analyses
- filtering specific configurations
- inspecting individual runs
- reproducibility checks

### `final_summary.csv`

This file contains aggregated experiment results.

Each row summarizes results grouped by:

- `dataset`
- `scheme`
- `method`
- `label_completion_method`
- `missing_rate`

The main columns are:

- `dataset`
- `scheme`
- `method`
- `missing_rate`
- `label_completion_method`
- `accuracy_mean`
- `accuracy_std`
- `balanced_accuracy_mean`
- `balanced_accuracy_std`
- `f1_mean`
- `f1_std`
- `roc_auc_mean`
- `roc_auc_std`

Use this file for:

- the final report
- summary tables
- high-level comparisons between methods
- comparisons across datasets and missingness settings

## Recommended Usage

- `final_results.csv` is the primary file for complete analyses and traceability.
- `final_summary.csv` is the primary file for the written report and presentation of aggregated results, including comparisons between `unlabeled` variants such as `logistic` and `knn`.

# Report Outline

## Purpose

This file organizes the material for the final report and points to the result tables that should be used during writing and figure preparation.

## Suggested Report Structure

### 1. Project Goal

- Describe the binary-classification task with missing labels.
- State the comparison goal between baseline methods, oracle reference, and label-completion approaches.
- Summarize the role of the implemented experimental pipeline.

### 2. Data Description

- List the datasets used in the experiments:
  - `breast_cancer`
  - `ionosphere`
  - `spambase`
  - `madelon`
- Describe the number of observations, number of features, binary target, and preprocessing pipeline.
- Mention that all datasets are loaded from local ARFF files.

### 3. Missingness Generators

- Explain the four missing-label schemes:
  - `MCAR`
  - `MAR1`
  - `MAR2`
  - `MNAR`
- Describe how missing labels are generated only on the training split.
- Note the role of `missing_rate` and, for MCAR, the sweep over multiple values.

### 4. FISTA Logistic Lasso

- Summarize the model formulation.
- Describe the FISTA optimization procedure.
- Mention the treatment of the intercept and the validation-based lambda selection.
- Explain how predictions and probabilities are produced.

### 5. UnlabeledLogReg

- Explain the two-stage procedure:
  - complete missing labels in train,
  - fit the downstream logistic-lasso classifier.
- Describe the currently supported label-completion strategies.
- Clarify how this differs from `NaiveLogReg` and `OracleLogReg`.

### 6. Experimental Methodology

- Describe the sequence:
  - dataset loading,
  - train/validation/test split,
  - preprocessing without leakage,
  - missing-label generation on train only,
  - model fitting,
  - evaluation on test.
- List the reported metrics:
  - accuracy
  - balanced accuracy
  - f1
  - roc auc
- Mention repeated runs over multiple seeds.

### 7. Results Analysis

- Compare methods across datasets.
- Compare methods across missingness schemes.
- For MCAR, analyze performance as a function of `missing_rate`.
- Use aggregated mean/std tables and, if needed, the raw tables for per-seed inspection.

### 8. Conclusions

- Summarize which methods are strongest under each setting.
- Discuss robustness to increasing missing-label rates.
- Note limitations and possible future extensions.

## Result Files To Use In The Report

Primary source tables in `outputs/tables/`:

- [all_experiments_results_20260410_192450.csv](/Users/igorrudolf/DataspellProjects/Advanced_ML_project_1/outputs/tables/all_experiments_results_20260410_192450.csv)
- [all_experiments_results_20260410_193958.csv](/Users/igorrudolf/DataspellProjects/Advanced_ML_project_1/outputs/tables/all_experiments_results_20260410_193958.csv)
- [all_experiments_summary_20260410_192450.csv](/Users/igorrudolf/DataspellProjects/Advanced_ML_project_1/outputs/tables/all_experiments_summary_20260410_192450.csv)
- [all_experiments_summary_20260410_193958.csv](/Users/igorrudolf/DataspellProjects/Advanced_ML_project_1/outputs/tables/all_experiments_summary_20260410_193958.csv)

Auxiliary / smoke-run tables that can be ignored in the final report unless needed for debugging:

- [run_all_smoke_results.csv](/Users/igorrudolf/DataspellProjects/Advanced_ML_project_1/outputs/tables/run_all_smoke_results.csv)
- [aggregation_smoke.csv](/Users/igorrudolf/DataspellProjects/Advanced_ML_project_1/outputs/tmp/aggregation_smoke.csv)

## Reproducibility Notes

- The report should cite the exact CSV files used for each table or figure.
- Aggregated analysis should prefer the `all_experiments_summary_*.csv` files.
- When discussing specific runs, refer back to the matching `all_experiments_results_*.csv` raw file.

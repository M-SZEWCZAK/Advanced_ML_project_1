# Advanced_ML_project_1

Projekt dotyczy eksperymentów z klasyfikacją binarną przy brakujących etykietach. Repozytorium zawiera warstwę danych opartą o lokalne pliki ARFF, implementacje modeli bazowych i metod uczących się z niepełnych etykiet oraz skrypty do uruchamiania pojedynczych i zbiorczych eksperymentów.

## Cel projektu

Celem projektu jest porównanie kilku podejść do uczenia w obecności brakujących etykiet:
- baseline `NaiveLogReg`, który uczy się tylko na obserwowanych etykietach,
- baseline `OracleLogReg`, który korzysta z pełnych etykiet referencyjnych,
- `UnlabeledLogReg`, który najpierw uzupełnia brakujące etykiety, a potem trenuje model końcowy,
- własna implementacja logistic lasso z FISTA.

## Struktura repo

- `src/project1/data/` – loadery danych, split i preprocessing.
- `src/project1/models/` – baseline'y, FISTA logistic lasso i `UnlabeledLogReg`.
- `src/project1/missingness/` – generatory brakujących etykiet.
- `src/project1/experiments/` – runner pojedynczych i zbiorczych eksperymentów oraz agregacja wyników.
- `scripts/` – skrypty uruchomieniowe do przygotowania danych i eksperymentów.
- `tests/` – lekkie smoke testy najważniejszych elementów pipeline'u.
- `data/raw/` – lokalne pliki ARFF używane przez warstwę danych.
- `outputs/` – wygenerowane tabele, logi i pliki wynikowe.
- `notebooks/` – notebook prototypowy zachowany do referencji.

## Instalacja zależności

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Przygotowanie danych

Skrypt ładuje lokalny dataset, wykonuje split oraz preprocessing i wypisuje podsumowanie finalnych macierzy:

```bash
python scripts/prepare_datasets.py breast_cancer
```

## Uruchomienie pojedynczego eksperymentu

Przykład pojedynczego eksperymentu dla `breast_cancer`, schematu `mcar` i metody `naive`:

```bash
python scripts/run_single_experiment.py \
  --dataset breast_cancer \
  --scheme mcar \
  --method naive \
  --missing-rate 0.2 \
  --seed 42 \
  --label-completion-method logistic
```

## Uruchomienie pełnej siatki eksperymentów

Pełny runner iteruje po konfiguracjach z `src/project1/experiments/configs.py` i zapisuje wyniki do CSV:

```bash
python scripts/run_all_experiments.py
```

## Gdzie zapisują się wyniki

- surowe wyniki pełnego biegu zapisują się w `outputs/tables/all_experiments_results_<timestamp>.csv`,
- agregacje mean/std zapisują się w `outputs/tables/all_experiments_summary_<timestamp>.csv`,
- pomocnicze pliki wynikowe i smoke-output mogą pojawiać się również w `outputs/tables/`.

## Uruchomienie smoke testów

Aby uruchomić dodane lekkie testy:

```bash
pytest tests/test_missingness.py tests/test_fista.py tests/test_unlabeled_logreg.py tests/test_pipeline_smoke.py
```

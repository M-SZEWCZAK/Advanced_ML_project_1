from __future__ import annotations

import ast
import glob
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(".").resolve()

EXPECTED_DATASETS = {"breast_cancer", "ionosphere", "spambase", "madelon"}
EXPECTED_SCHEMES = {"mcar", "mar1", "mar2", "mnar"}
EXPECTED_METHODS = {"naive", "oracle", "unlabeled"}
MIN_MCAR_RATES = 2  # co najmniej 2 różne wartości missing_rate dla MCAR


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _find_preferred_output_file(final_path: str, compatibility_pattern: str) -> Path | None:
    """Return the preferred final output file, with a compatibility fallback.

    The checker treats the stable final filenames as the primary source of truth:
    `final_results.csv` and `final_summary.csv`. If such a file is missing, it
    falls back to the newest timestamped compatibility file matching the legacy
    `all_experiments_*` naming scheme.
    """
    preferred_path = ROOT / final_path
    if preferred_path.exists():
        return preferred_path

    candidates = sorted(
        glob.glob(compatibility_pattern),
        key=os.path.getmtime,
        reverse=True,
    )
    return Path(candidates[0]) if candidates else None


def find_results_csv() -> Path | None:
    """Find the final results CSV, with fallback to legacy timestamped files."""
    return _find_preferred_output_file(
        final_path="outputs/tables/final_results.csv",
        compatibility_pattern="outputs/tables/all_experiments_results_*.csv",
    )


def find_summary_csv() -> Path | None:
    """Find the final summary CSV, with fallback to legacy timestamped files."""
    return _find_preferred_output_file(
        final_path="outputs/tables/final_summary.csv",
        compatibility_pattern="outputs/tables/all_experiments_summary_*.csv",
    )


def parse_python_file(path: Path) -> ast.AST | None:
    try:
        return ast.parse(read_text(path))
    except Exception:
        return None


def find_class_methods(path: Path, class_name_contains: str) -> set[str]:
    tree = parse_python_file(path)
    methods: set[str] = set()
    if tree is None:
        return methods

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and class_name_contains.lower() in node.name.lower():
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.add(item.name)
    return methods


def find_string_in_repo(targets: list[str], include_dirs=("src", "scripts", "notebooks")) -> list[Path]:
    matches: list[Path] = []
    for include_dir in include_dirs:
        base = ROOT / include_dir
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".ipynb", ".md"}:
                try:
                    text = read_text(path)
                except Exception:
                    continue
                if all(t in text for t in targets):
                    matches.append(path)
    return matches


def check_fista() -> bool:
    path = ROOT / "src/project1/models/fista_logistic_lasso.py"
    if not path.exists():
        fail("Brakuje pliku src/project1/models/fista_logistic_lasso.py")
        return False

    methods = find_class_methods(path, "fista")
    needed = {"fit", "validate", "predict_proba", "predict", "plot", "plot_coefficients"}
    missing = needed - methods

    if missing:
        fail(f"FISTA: brakuje metod: {sorted(missing)}")
        return False

    ok("FISTA zawiera fit, validate, predict_proba, predict, plot, plot_coefficients")
    return True


def check_unlabeled_methods() -> bool:
    path = ROOT / "src/project1/models/unlabeled_logreg.py"
    if not path.exists():
        fail("Brakuje pliku src/project1/models/unlabeled_logreg.py")
        return False

    text = read_text(path)

    class_methods = find_class_methods(path, "unlabeled")
    needed = {"complete_labels", "fit", "predict", "predict_proba"}
    missing = needed - class_methods

    if missing:
        fail(f"UnlabeledLogReg: brakuje metod: {sorted(missing)}")
        return False

    # heurystyka: szukamy co najmniej dwóch strategii
    strategy_hits = set()
    for token in ["logistic", "knn", "prior", "rf", "tree"]:
        if token in text.lower():
            strategy_hits.add(token)

    if len(strategy_hits) < 2:
        fail(f"UnlabeledLogReg: wykryto mniej niż 2 strategie uzupełniania etykiet: {sorted(strategy_hits)}")
        return False

    ok(f"UnlabeledLogReg ma wymagane metody i co najmniej 2 strategie: {sorted(strategy_hits)}")
    return True


def check_sklearn_comparison() -> bool:
    # szukamy śladów porównania do sklearn L1 poza samymi baseline'ami
    matches = []
    for path in find_string_in_repo(["LogisticRegression"], include_dirs=("src", "scripts", "notebooks")):
        if "baselines.py" in str(path):
            continue
        text = read_text(path)
        if "penalty='l1'" in text or 'penalty="l1"' in text or "sklearn.linear_model" in text:
            matches.append(path)

    if not matches:
        fail("Nie znaleziono w repo śladu porównania do sklearn LogisticRegression z L1 poza baseline'ami")
        return False

    ok("Znaleziono ślad porównania do sklearn L1 w plikach:")
    for m in matches:
        print("   -", m.as_posix())
    return True


def check_results_csv() -> bool:
    latest = find_results_csv()
    if latest is None:
        fail(
            "Nie znaleziono pliku outputs/tables/final_results.csv "
            "ani kompatybilnościowego outputs/tables/all_experiments_results_*.csv"
        )
        return False

    print(f"[INFO] Analizuję najnowszy plik wynikowy: {latest.as_posix()}")

    df = pd.read_csv(latest)

    needed_cols = {"dataset", "scheme", "method", "seed", "missing_rate", "accuracy", "balanced_accuracy", "f1", "roc_auc"}
    missing_cols = needed_cols - set(df.columns)
    if missing_cols:
        fail(f"Brakuje kolumn w wynikach: {sorted(missing_cols)}")
        return False

    success = True

    datasets = set(df["dataset"].dropna().astype(str).unique())
    schemes = set(df["scheme"].dropna().astype(str).unique())
    methods = set(df["method"].dropna().astype(str).unique())

    missing_datasets = EXPECTED_DATASETS - datasets
    missing_schemes = EXPECTED_SCHEMES - schemes
    missing_methods = EXPECTED_METHODS - methods

    if missing_datasets:
        fail(f"W wynikach brakuje datasetów: {sorted(missing_datasets)}")
        success = False
    else:
        ok(f"Wyniki zawierają wszystkie datasety: {sorted(EXPECTED_DATASETS)}")

    if missing_schemes:
        fail(f"W wynikach brakuje schemes: {sorted(missing_schemes)}")
        success = False
    else:
        ok(f"Wyniki zawierają wszystkie schemes: {sorted(EXPECTED_SCHEMES)}")

    if missing_methods:
        fail(f"W wynikach brakuje metod: {sorted(missing_methods)}")
        success = False
    else:
        ok(f"Wyniki zawierają wszystkie metody: {sorted(EXPECTED_METHODS)}")

    mcar = df[df["scheme"].astype(str) == "mcar"].copy()
    if mcar.empty:
        fail("Brak rekordów dla scheme = mcar")
        success = False
    else:
        rates = sorted(set(float(x) for x in mcar["missing_rate"].dropna().unique()))
        print(f"[INFO] Wykryte missing_rate dla MCAR: {rates}")
        if len(rates) < MIN_MCAR_RATES:
            fail(f"Za mało różnych missing_rate dla MCAR: {rates}")
            success = False
        else:
            ok("Analiza MCAR zawiera wiele wartości missing_rate")

    return success


def check_summary_csv() -> bool:
    """Check whether the aggregated summary CSV is present and structurally valid."""
    summary_path = find_summary_csv()
    if summary_path is None:
        fail(
            "Nie znaleziono pliku outputs/tables/final_summary.csv "
            "ani kompatybilnościowego outputs/tables/all_experiments_summary_*.csv"
        )
        return False

    print(f"[INFO] Analizuję plik summary: {summary_path.as_posix()}")

    df = pd.read_csv(summary_path)
    needed_cols = {
        "dataset",
        "scheme",
        "method",
        "missing_rate",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
    }
    missing_cols = needed_cols - set(df.columns)
    if missing_cols:
        fail(f"Brakuje kolumn w summary: {sorted(missing_cols)}")
        return False

    ok("Plik summary zawiera wymagane kolumny agregatów")
    return True


def main() -> None:
    print("=== CHECK PRE-MAIN START ===")
    results = {
        "fista": check_fista(),
        "unlabeled": check_unlabeled_methods(),
        "sklearn_comparison": check_sklearn_comparison(),
        "results_csv": check_results_csv(),
        "summary_csv": check_summary_csv(),
    }

    print("\n=== PODSUMOWANIE ===")
    print(json.dumps(results, indent=2))

    if all(results.values()):
        print("\n[FINAL OK] Repo wygląda na gotowe do merge develop -> main od strony podstawowych wymagań.")
    else:
        print("\n[FINAL FAIL] Przed merge do main trzeba jeszcze domknąć elementy oznaczone jako FAIL.")


if __name__ == "__main__":
    main()

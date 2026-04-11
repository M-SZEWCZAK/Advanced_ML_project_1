"""Create report-style plots from outputs/tables/final_results.csv."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "outputs" / "tables" / "final_results.csv"
DEFAULT_SUMMARY_INPUT_PATH = PROJECT_ROOT / "outputs" / "tables" / "final_summary.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "plots"
VALID_METRICS = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
METHOD_ORDER = ["naive", "oracle", "unlabeled-logreg", "unlabeled-knn"]
SCHEME_ORDER = ["mcar", "mar1", "mar2", "mnar"]


def load_results(csv_path):
    """Load experiment results and keep only successful runs."""
    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    return add_display_method_column(df)


def load_summary(csv_path):
    """Load aggregated experiment results and add display-method labels."""
    df = pd.read_csv(csv_path)
    return add_display_method_column(df)


def add_display_method_column(df):
    """Add a method label derived only from the provided results table."""
    labeled_df = df.copy()

    if "label_completion_method" not in labeled_df.columns:
        labeled_df["label_completion_method"] = pd.NA

    def _display_method(row):
        method = str(row["method"]).strip().lower()
        completion = row["label_completion_method"]

        if method != "unlabeled":
            return method

        if pd.isna(completion):
            return "unlabeled"

        completion_name = str(completion).strip().lower()
        if completion_name == "logistic":
            return "unlabeled-logreg"
        return f"unlabeled-{completion_name}"

    labeled_df["display_method"] = labeled_df.apply(_display_method, axis=1)
    return labeled_df


def _sort_categories(values, preferred_order):
    """Sort values by a preferred order, then alphabetically."""
    order_map = {value: index for index, value in enumerate(preferred_order)}
    return sorted(values, key=lambda value: (order_map.get(value, len(order_map)), value))


def _validate_metric(metric):
    """Validate the chosen metric name."""
    if metric not in VALID_METRICS:
        raise ValueError(f"Unsupported metric: {metric}. Valid metrics: {', '.join(VALID_METRICS)}")


def _set_dynamic_metric_ylim(ax, values, lower_bound=0.0, upper_bound=1.0, padding_ratio=0.08, min_span=0.05):
    """Set a tighter y-axis range around the observed metric values."""
    series = pd.Series(values).dropna()
    if series.empty:
        ax.set_ylim(lower_bound, upper_bound)
        return

    value_min = float(series.min())
    value_max = float(series.max())
    span = max(value_max - value_min, min_span)
    padding = span * padding_ratio

    ymin = max(lower_bound, value_min - padding)
    ymax = min(upper_bound, value_max + padding)

    if ymax - ymin < min_span:
        center = (ymin + ymax) / 2
        ymin = max(lower_bound, center - min_span / 2)
        ymax = min(upper_bound, center + min_span / 2)

        if ymax - ymin < min_span:
            if ymin == lower_bound:
                ymax = min(upper_bound, ymin + min_span)
            else:
                ymin = max(lower_bound, ymax - min_span)

    ax.set_ylim(ymin, ymax)


def _get_metric_summary_columns(metric):
    """Return mean and std column names for a summary metric."""
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    return mean_column, std_column


def _split_oracle_reference(summary_df):
    """Split a summary table into oracle and non-oracle subsets."""
    oracle_df = summary_df[summary_df["display_method"] == "oracle"].copy()
    non_oracle_df = summary_df[summary_df["display_method"] != "oracle"].copy()
    return non_oracle_df, oracle_df


def _get_oracle_reference_stats(oracle_df, mean_column, std_column):
    """Return a single oracle reference mean/std from repeated summary rows."""
    if oracle_df.empty:
        return None, None

    oracle_mean = float(oracle_df[mean_column].mean())
    oracle_std = float(oracle_df[std_column].fillna(0.0).mean())
    return oracle_mean, oracle_std


def _get_oracle_raw_stats(oracle_df, metric):
    """Return raw oracle values and their mean from repeated result rows."""
    if oracle_df.empty:
        return pd.Series(dtype=float), None

    oracle_values = oracle_df[metric].dropna()
    if oracle_values.empty:
        return oracle_values, None

    return oracle_values, float(oracle_values.mean())


def _build_summary_lookup(summary_df, index_column, mean_column, std_column):
    """Build lookup tables for grouped summary means and standard deviations."""
    mean_lookup = {}
    std_lookup = {}

    for _, row in summary_df.iterrows():
        key = (row[index_column], row["display_method"])
        mean_lookup[key] = row[mean_column]
        std_lookup[key] = row[std_column]

    return mean_lookup, std_lookup


def _plot_difference_lines(ax, x_values, diff_map, title_methods, ylabel):
    """Plot one line per method from a precomputed difference map."""
    plotted_values = []

    for method_index, method in enumerate(title_methods):
        y_values = [diff_map.get((x, method)) for x in x_values]
        if all(value is None or pd.isna(value) for value in y_values):
            continue

        clean_values = [float(value) if value is not None and not pd.isna(value) else float("nan") for value in y_values]
        ax.plot(
            x_values,
            clean_values,
            marker="o",
            linewidth=2,
            color=f"C{method_index}",
            label=method,
        )
        plotted_values.extend([value for value in clean_values if not pd.isna(value)])

    ax.axhline(0.0, linestyle="--", linewidth=1.2, color="black", alpha=0.6)
    ax.set_ylabel(ylabel)
    return plotted_values


def create_methods_boxplot(df, metric, missing_rate, dataset, output_path=None, show=False):
    """Create a grouped boxplot comparing methods across generating schemes.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw experiment results.
    metric : str
        One of accuracy, balanced_accuracy, f1, roc_auc.
    missing_rate : float
        Missing-rate value used to filter rows.
    dataset : str
        Dataset name used to filter rows.
    output_path : Path or None, default=None
        Optional destination for saving the figure.
    show : bool, default=False
        Whether to display the plot interactively.
    """
    _validate_metric(metric)
    filtered_df = df[
        (df["dataset"] == dataset)
        & (df["missing_rate"] == missing_rate)
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No rows found for dataset={dataset!r} and missing_rate={missing_rate}."
        )

    oracle_df = filtered_df[filtered_df["display_method"] == "oracle"].copy()
    non_oracle_df = filtered_df[filtered_df["display_method"] != "oracle"].copy()
    oracle_values, oracle_mean = _get_oracle_raw_stats(oracle_df, metric)

    schemes = _sort_categories(filtered_df["scheme"].unique(), SCHEME_ORDER)
    methods = _sort_categories(non_oracle_df["display_method"].unique(), METHOD_ORDER)

    fig, ax = plt.subplots(figsize=(11, 6))
    base_positions = list(range(len(schemes)))
    box_width = 0.22

    for method_index, method in enumerate(methods):
        method_data = []
        positions = []
        offset = (method_index - (len(methods) - 1) / 2) * box_width

        for scheme_index, scheme in enumerate(schemes):
            values = non_oracle_df[
                (non_oracle_df["scheme"] == scheme)
                & (non_oracle_df["display_method"] == method)
            ][metric].dropna()
            if values.empty:
                continue
            method_data.append(values.to_numpy())
            positions.append(base_positions[scheme_index] + offset)

        if method_data:
            ax.boxplot(
                method_data,
                positions=positions,
                widths=box_width * 0.9,
                patch_artist=True,
                boxprops={"facecolor": f"C{method_index}", "alpha": 0.45},
                medianprops={"color": "black"},
                whiskerprops={"color": f"C{method_index}"},
                capprops={"color": f"C{method_index}"},
                flierprops={"markerfacecolor": f"C{method_index}", "markeredgecolor": f"C{method_index}", "alpha": 0.5},
            )
            ax.plot([], [], color=f"C{method_index}", linewidth=8, alpha=0.45, label=method)

    if oracle_mean is not None:
        oracle_x = -1
        ax.axhline(
            oracle_mean,
            linestyle="--",
            linewidth=1.8,
            color="black",
            alpha=0.8,
            label="oracle mean",
        )
        ax.boxplot(
            [oracle_values.to_numpy()],
            positions=[oracle_x],
            widths=box_width * 1.2,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": "black", "alpha": 0.8},
            medianprops={"color": "black"},
            whiskerprops={"color": "black"},
            capprops={"color": "black"},
            flierprops={"markerfacecolor": "black", "markeredgecolor": "black", "alpha": 0.5},
        )
        ax.plot([], [], color="black", linewidth=1.8, linestyle="--", label="oracle")
        ax.set_xlim(oracle_x - 0.6, max(base_positions) + 0.6)
        ax.set_xticks([oracle_x] + base_positions)
        ax.set_xticklabels(["oracle"] + schemes)
    else:
        ax.set_xticks(base_positions)
        ax.set_xticklabels(schemes)

    ax.set_xlabel("Generating Method")
    ax.set_ylabel(metric.replace("_", " ").title())
    _set_dynamic_metric_ylim(ax, filtered_df[metric])
    ax.set_title(f"{dataset}: {metric.replace('_', ' ').title()} by Scheme and Method (missing_rate={missing_rate})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Method")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def create_missing_rate_comparison_plot(df, metric, dataset, generating_method, output_path=None, show=False):
    """Create boxplots comparing methods across missing-rate values for one scheme.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw experiment results.
    metric : str
        One of accuracy, balanced_accuracy, f1, roc_auc.
    dataset : str
        Dataset name used to filter rows.
    generating_method : str
        Missingness scheme, e.g. mcar, mar1, mar2, mnar.
    output_path : Path or None, default=None
        Optional destination for saving the figure.
    show : bool, default=False
        Whether to display the plot interactively.
    """
    _validate_metric(metric)
    filtered_df = df[
        (df["dataset"] == dataset)
        & (df["scheme"] == generating_method)
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No rows found for dataset={dataset!r} and generating_method={generating_method!r}."
        )

    oracle_df = filtered_df[filtered_df["display_method"] == "oracle"].copy()
    non_oracle_df = filtered_df[filtered_df["display_method"] != "oracle"].copy()
    oracle_values, oracle_mean = _get_oracle_raw_stats(oracle_df, metric)

    missing_rates = sorted(filtered_df["missing_rate"].unique())
    methods = _sort_categories(non_oracle_df["display_method"].unique(), METHOD_ORDER)

    fig, ax = plt.subplots(figsize=(11, 6))
    base_positions = list(range(len(missing_rates)))
    box_width = 0.22

    for method_index, method in enumerate(methods):
        method_data = []
        positions = []
        offset = (method_index - (len(methods) - 1) / 2) * box_width

        for rate_index, missing_rate in enumerate(missing_rates):
            values = non_oracle_df[
                (non_oracle_df["missing_rate"] == missing_rate)
                & (non_oracle_df["display_method"] == method)
            ][metric].dropna()
            if values.empty:
                continue
            method_data.append(values.to_numpy())
            positions.append(base_positions[rate_index] + offset)

        if method_data:
            ax.boxplot(
                method_data,
                positions=positions,
                widths=box_width * 0.9,
                patch_artist=True,
                boxprops={"facecolor": f"C{method_index}", "alpha": 0.45},
                medianprops={"color": "black"},
                whiskerprops={"color": f"C{method_index}"},
                capprops={"color": f"C{method_index}"},
                flierprops={"markerfacecolor": f"C{method_index}", "markeredgecolor": f"C{method_index}", "alpha": 0.5},
            )
            ax.plot([], [], color=f"C{method_index}", linewidth=8, alpha=0.45, label=method)

    if oracle_mean is not None:
        oracle_x = -1
        ax.axhline(
            oracle_mean,
            linestyle="--",
            linewidth=1.8,
            color="black",
            alpha=0.8,
            label="oracle mean",
        )
        ax.boxplot(
            [oracle_values.to_numpy()],
            positions=[oracle_x],
            widths=box_width * 1.2,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": "black", "alpha": 0.8},
            medianprops={"color": "black"},
            whiskerprops={"color": "black"},
            capprops={"color": "black"},
            flierprops={"markerfacecolor": "black", "markeredgecolor": "black", "alpha": 0.5},
        )
        ax.plot([], [], color="black", linewidth=1.8, linestyle="--", label="oracle")
        ax.set_xlim(oracle_x - 0.6, max(base_positions) + 0.6)
        ax.set_xticks([oracle_x] + base_positions)
        ax.set_xticklabels(["oracle"] + [str(rate) for rate in missing_rates])
    else:
        ax.set_xticks(base_positions)
        ax.set_xticklabels([str(rate) for rate in missing_rates])

    ax.set_xlabel("Missing Rate")
    ax.set_ylabel(metric.replace("_", " ").title())
    _set_dynamic_metric_ylim(ax, filtered_df[metric])
    ax.set_title(f"{dataset}: {metric.replace('_', ' ').title()} by Missing Rate ({generating_method})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Method")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def create_scheme_summary_plot(summary_df, metric, dataset, missing_rate=0.3, output_path=None, show=False):
    """Create a report-friendly mean plot with error bars across schemes."""
    _validate_metric(metric)
    mean_column, std_column = _get_metric_summary_columns(metric)
    filtered_df = summary_df[
        (summary_df["dataset"] == dataset)
        & (summary_df["missing_rate"] == missing_rate)
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No summary rows found for dataset={dataset!r} and missing_rate={missing_rate}."
        )

    non_oracle_df, oracle_df = _split_oracle_reference(filtered_df)
    schemes = _sort_categories(filtered_df["scheme"].unique(), SCHEME_ORDER)
    methods = _sort_categories(non_oracle_df["display_method"].unique(), METHOD_ORDER)
    oracle_mean, oracle_std = _get_oracle_reference_stats(oracle_df, mean_column, std_column)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = list(range(len(schemes)))

    for method_index, method in enumerate(methods):
        method_df = non_oracle_df[non_oracle_df["display_method"] == method].copy()
        method_df["scheme"] = pd.Categorical(method_df["scheme"], categories=schemes, ordered=True)
        method_df = method_df.sort_values("scheme")
        ax.errorbar(
            x_positions,
            method_df[mean_column],
            yerr=method_df[std_column],
            marker="o",
            linewidth=2,
            capsize=4,
            label=method,
            color=f"C{method_index}",
        )

    if oracle_mean is not None:
        oracle_x = -1
        ax.axhline(
            oracle_mean,
            linestyle="--",
            linewidth=1.8,
            color="black",
            alpha=0.8,
            label="oracle mean",
        )
        ax.errorbar(
            [oracle_x],
            [oracle_mean],
            yerr=[oracle_std],
            marker="s",
            markersize=7,
            linestyle="None",
            capsize=4,
            color="black",
            label="oracle",
        )
        ax.set_xlim(oracle_x - 0.6, max(x_positions) + 0.6)
        ax.set_xticks([oracle_x] + x_positions)
        ax.set_xticklabels(["oracle"] + schemes)
    else:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(schemes)

    ax.set_xlabel("Missingness Scheme")
    ax.set_ylabel(metric.replace("_", " ").title())
    _set_dynamic_metric_ylim(ax, filtered_df[mean_column])
    ax.set_title(
        f"{dataset}: {metric.replace('_', ' ').title()} mean by scheme (missing_rate={missing_rate})"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Method")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def create_mcar_trend_plot(summary_df, metric, dataset, output_path=None, show=False):
    """Create a trend plot across MCAR missing rates using summary means and std."""
    _validate_metric(metric)
    mean_column, std_column = _get_metric_summary_columns(metric)
    filtered_df = summary_df[
        (summary_df["dataset"] == dataset)
        & (summary_df["scheme"] == "mcar")
    ].copy()

    if filtered_df.empty:
        raise ValueError(f"No summary rows found for dataset={dataset!r} and scheme='mcar'.")

    missing_rates = sorted(filtered_df["missing_rate"].unique())
    non_oracle_df, oracle_df = _split_oracle_reference(filtered_df)
    methods = _sort_categories(non_oracle_df["display_method"].unique(), METHOD_ORDER)
    oracle_mean, oracle_std = _get_oracle_reference_stats(oracle_df, mean_column, std_column)

    fig, ax = plt.subplots(figsize=(10, 6))

    for method_index, method in enumerate(methods):
        method_df = non_oracle_df[non_oracle_df["display_method"] == method].copy()
        method_df = method_df.sort_values("missing_rate")
        x_values = method_df["missing_rate"].to_numpy()
        mean_values = method_df[mean_column].to_numpy()
        std_values = method_df[std_column].fillna(0.0).to_numpy()
        color = f"C{method_index}"

        ax.plot(
            x_values,
            mean_values,
            marker="o",
            linewidth=2,
            label=method,
            color=color,
        )
        ax.fill_between(
            x_values,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.15,
            color=color,
        )

    if oracle_mean is not None:
        oracle_x = min(missing_rates) - 0.12
        ax.axhline(
            oracle_mean,
            linestyle="--",
            linewidth=1.8,
            color="black",
            alpha=0.8,
            label="oracle mean",
        )
        ax.errorbar(
            [oracle_x],
            [oracle_mean],
            yerr=[oracle_std],
            marker="s",
            markersize=7,
            linestyle="None",
            capsize=4,
            color="black",
            label="oracle",
        )
        ax.set_xlim(oracle_x - 0.06, max(missing_rates) + 0.06)
        ax.set_xticks([oracle_x] + missing_rates)
        ax.set_xticklabels(["oracle"] + [str(rate) for rate in missing_rates])
    else:
        ax.set_xticks(missing_rates)
        ax.set_xticklabels([str(rate) for rate in missing_rates])

    ax.set_xlabel("MCAR missing_rate")
    ax.set_ylabel(metric.replace("_", " ").title())
    _set_dynamic_metric_ylim(ax, filtered_df[mean_column])
    ax.set_title(f"{dataset}: {metric.replace('_', ' ').title()} trend across MCAR missing_rate")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Method")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def create_unlabeled_improvement_over_naive_by_scheme_plot(
    summary_df,
    metric,
    dataset,
    missing_rate=0.3,
    output_path=None,
    show=False,
):
    """Plot unlabeled-method improvement over naive across schemes."""
    _validate_metric(metric)
    mean_column, std_column = _get_metric_summary_columns(metric)
    filtered_df = summary_df[
        (summary_df["dataset"] == dataset)
        & (summary_df["missing_rate"] == missing_rate)
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No summary rows found for dataset={dataset!r} and missing_rate={missing_rate}."
        )

    schemes = _sort_categories(filtered_df["scheme"].unique(), SCHEME_ORDER)
    mean_lookup, _ = _build_summary_lookup(filtered_df, "scheme", mean_column, std_column)
    methods = [method for method in METHOD_ORDER if method.startswith("unlabeled-")]
    diff_map = {}

    for scheme in schemes:
        naive_mean = mean_lookup.get((scheme, "naive"))
        if naive_mean is None or pd.isna(naive_mean):
            continue
        for method in methods:
            method_mean = mean_lookup.get((scheme, method))
            if method_mean is None or pd.isna(method_mean):
                continue
            diff_map[(scheme, method)] = float(method_mean - naive_mean)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_values = _plot_difference_lines(
        ax=ax,
        x_values=schemes,
        diff_map=diff_map,
        title_methods=methods,
        ylabel=f"{metric.replace('_', ' ').title()} improvement over naive",
    )
    _set_dynamic_metric_ylim(ax, plotted_values, lower_bound=-1.0, upper_bound=1.0, min_span=0.02)
    ax.set_xlabel("Missingness Scheme")
    ax.set_title(
        f"{dataset}: unlabeled improvement over naive by scheme (missing_rate={missing_rate})"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Method")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def create_unlabeled_improvement_over_naive_by_missing_rate_plot(
    summary_df,
    metric,
    dataset,
    generating_method="mcar",
    output_path=None,
    show=False,
):
    """Plot unlabeled-method improvement over naive across missing rates."""
    _validate_metric(metric)
    mean_column, std_column = _get_metric_summary_columns(metric)
    filtered_df = summary_df[
        (summary_df["dataset"] == dataset)
        & (summary_df["scheme"] == generating_method)
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No summary rows found for dataset={dataset!r} and generating_method={generating_method!r}."
        )

    missing_rates = sorted(filtered_df["missing_rate"].unique())
    mean_lookup, _ = _build_summary_lookup(filtered_df, "missing_rate", mean_column, std_column)
    methods = [method for method in METHOD_ORDER if method.startswith("unlabeled-")]
    diff_map = {}

    for rate in missing_rates:
        naive_mean = mean_lookup.get((rate, "naive"))
        if naive_mean is None or pd.isna(naive_mean):
            continue
        for method in methods:
            method_mean = mean_lookup.get((rate, method))
            if method_mean is None or pd.isna(method_mean):
                continue
            diff_map[(rate, method)] = float(method_mean - naive_mean)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_values = _plot_difference_lines(
        ax=ax,
        x_values=missing_rates,
        diff_map=diff_map,
        title_methods=methods,
        ylabel=f"{metric.replace('_', ' ').title()} improvement over naive",
    )
    _set_dynamic_metric_ylim(ax, plotted_values, lower_bound=-1.0, upper_bound=1.0, min_span=0.02)
    ax.set_xlabel("Missing Rate")
    ax.set_title(
        f"{dataset}: unlabeled improvement over naive by missing rate ({generating_method})"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Method")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def create_delta_to_oracle_by_scheme_plot(
    summary_df,
    metric,
    dataset,
    missing_rate=0.3,
    output_path=None,
    show=False,
):
    """Plot method delta to oracle across schemes."""
    _validate_metric(metric)
    mean_column, std_column = _get_metric_summary_columns(metric)
    filtered_df = summary_df[
        (summary_df["dataset"] == dataset)
        & (summary_df["missing_rate"] == missing_rate)
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No summary rows found for dataset={dataset!r} and missing_rate={missing_rate}."
        )

    schemes = _sort_categories(filtered_df["scheme"].unique(), SCHEME_ORDER)
    mean_lookup, _ = _build_summary_lookup(filtered_df, "scheme", mean_column, std_column)
    methods = [method for method in METHOD_ORDER if method != "oracle"]
    diff_map = {}

    for scheme in schemes:
        oracle_mean = mean_lookup.get((scheme, "oracle"))
        if oracle_mean is None or pd.isna(oracle_mean):
            continue
        for method in methods:
            method_mean = mean_lookup.get((scheme, method))
            if method_mean is None or pd.isna(method_mean):
                continue
            diff_map[(scheme, method)] = float(oracle_mean - method_mean)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_values = _plot_difference_lines(
        ax=ax,
        x_values=schemes,
        diff_map=diff_map,
        title_methods=methods,
        ylabel=f"Delta to oracle in {metric.replace('_', ' ').title()}",
    )
    _set_dynamic_metric_ylim(ax, plotted_values, lower_bound=-1.0, upper_bound=1.0, min_span=0.02)
    ax.set_xlabel("Missingness Scheme")
    ax.set_title(f"{dataset}: delta to oracle by scheme (missing_rate={missing_rate})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Method")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def create_delta_to_oracle_by_missing_rate_plot(
    summary_df,
    metric,
    dataset,
    generating_method="mcar",
    output_path=None,
    show=False,
):
    """Plot method delta to oracle across missing rates."""
    _validate_metric(metric)
    mean_column, std_column = _get_metric_summary_columns(metric)
    filtered_df = summary_df[
        (summary_df["dataset"] == dataset)
        & (summary_df["scheme"] == generating_method)
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No summary rows found for dataset={dataset!r} and generating_method={generating_method!r}."
        )

    missing_rates = sorted(filtered_df["missing_rate"].unique())
    mean_lookup, _ = _build_summary_lookup(filtered_df, "missing_rate", mean_column, std_column)
    methods = [method for method in METHOD_ORDER if method != "oracle"]
    diff_map = {}

    for rate in missing_rates:
        oracle_mean = mean_lookup.get((rate, "oracle"))
        if oracle_mean is None or pd.isna(oracle_mean):
            continue
        for method in methods:
            method_mean = mean_lookup.get((rate, method))
            if method_mean is None or pd.isna(method_mean):
                continue
            diff_map[(rate, method)] = float(oracle_mean - method_mean)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_values = _plot_difference_lines(
        ax=ax,
        x_values=missing_rates,
        diff_map=diff_map,
        title_methods=methods,
        ylabel=f"Delta to oracle in {metric.replace('_', ' ').title()}",
    )
    _set_dynamic_metric_ylim(ax, plotted_values, lower_bound=-1.0, upper_bound=1.0, min_span=0.02)
    ax.set_xlabel("Missing Rate")
    ax.set_title(f"{dataset}: delta to oracle by missing rate ({generating_method})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Method")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def parse_args():
    """Parse CLI arguments for the plotting helper."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV file with raw experiment results.",
    )
    parser.add_argument(
        "--summary-input",
        type=Path,
        default=DEFAULT_SUMMARY_INPUT_PATH,
        help="CSV file with aggregated experiment results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated plots should be saved.",
    )
    parser.add_argument(
        "--plot-type",
        choices=[
            "methods-boxplot",
            "missing-rate-boxplot",
            "scheme-summary",
            "mcar-trend",
            "improvement-over-naive-scheme",
            "improvement-over-naive-missing-rate",
            "delta-to-oracle-scheme",
            "delta-to-oracle-missing-rate",
            "both",
        ],
        default="both",
        help="Which plot type to create.",
    )
    parser.add_argument(
        "--metric",
        required=True,
        choices=VALID_METRICS,
        help="Metric used on the y-axis.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset to plot.",
    )
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=None,
        help="Missing-rate filter for the methods-boxplot.",
    )
    parser.add_argument(
        "--generating-method",
        choices=SCHEME_ORDER,
        default=None,
        help="Missingness scheme for the missing-rate-boxplot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def main():
    """Create the requested plots and print the saved output paths."""
    args = parse_args()
    df = load_results(args.input)
    summary_df = None
    saved_paths = []

    if args.plot_type in {"methods-boxplot", "both"}:
        if args.missing_rate is None:
            raise ValueError("--missing-rate is required for methods-boxplot.")
        methods_plot_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_methods_by_scheme_missing_rate_{args.missing_rate}.png"
        )
        create_methods_boxplot(
            df=df,
            metric=args.metric,
            missing_rate=args.missing_rate,
            dataset=args.dataset,
            output_path=methods_plot_path,
            show=args.show,
        )
        saved_paths.append(methods_plot_path)

    if args.plot_type in {"missing-rate-boxplot", "both"}:
        if args.generating_method is None:
            raise ValueError("--generating-method is required for missing-rate-boxplot.")
        rates_plot_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_missing_rate_comparison_{args.generating_method}.png"
        )
        create_missing_rate_comparison_plot(
            df=df,
            metric=args.metric,
            dataset=args.dataset,
            generating_method=args.generating_method,
            output_path=rates_plot_path,
            show=args.show,
        )
        saved_paths.append(rates_plot_path)

    if args.plot_type in {
        "scheme-summary",
        "mcar-trend",
        "improvement-over-naive-scheme",
        "improvement-over-naive-missing-rate",
        "delta-to-oracle-scheme",
        "delta-to-oracle-missing-rate",
        "both",
    }:
        summary_df = load_summary(args.summary_input)

    if args.plot_type in {"scheme-summary", "both"}:
        if args.missing_rate is None:
            raise ValueError("--missing-rate is required for scheme-summary.")
        scheme_summary_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_scheme_summary_missing_rate_{args.missing_rate}.png"
        )
        create_scheme_summary_plot(
            summary_df=summary_df,
            metric=args.metric,
            dataset=args.dataset,
            missing_rate=args.missing_rate,
            output_path=scheme_summary_path,
            show=args.show,
        )
        saved_paths.append(scheme_summary_path)

    if args.plot_type in {"mcar-trend", "both"}:
        mcar_trend_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_mcar_trend.png"
        )
        create_mcar_trend_plot(
            summary_df=summary_df,
            metric=args.metric,
            dataset=args.dataset,
            output_path=mcar_trend_path,
            show=args.show,
        )
        saved_paths.append(mcar_trend_path)

    if args.plot_type in {"improvement-over-naive-scheme", "both"}:
        if args.missing_rate is None:
            raise ValueError("--missing-rate is required for improvement-over-naive-scheme.")
        improvement_scheme_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_improvement_over_naive_by_scheme_missing_rate_{args.missing_rate}.png"
        )
        create_unlabeled_improvement_over_naive_by_scheme_plot(
            summary_df=summary_df,
            metric=args.metric,
            dataset=args.dataset,
            missing_rate=args.missing_rate,
            output_path=improvement_scheme_path,
            show=args.show,
        )
        saved_paths.append(improvement_scheme_path)

    if args.plot_type in {"improvement-over-naive-missing-rate", "both"}:
        if args.generating_method is None:
            raise ValueError("--generating-method is required for improvement-over-naive-missing-rate.")
        improvement_rate_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_improvement_over_naive_by_missing_rate_{args.generating_method}.png"
        )
        create_unlabeled_improvement_over_naive_by_missing_rate_plot(
            summary_df=summary_df,
            metric=args.metric,
            dataset=args.dataset,
            generating_method=args.generating_method,
            output_path=improvement_rate_path,
            show=args.show,
        )
        saved_paths.append(improvement_rate_path)

    if args.plot_type in {"delta-to-oracle-scheme", "both"}:
        if args.missing_rate is None:
            raise ValueError("--missing-rate is required for delta-to-oracle-scheme.")
        delta_scheme_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_delta_to_oracle_by_scheme_missing_rate_{args.missing_rate}.png"
        )
        create_delta_to_oracle_by_scheme_plot(
            summary_df=summary_df,
            metric=args.metric,
            dataset=args.dataset,
            missing_rate=args.missing_rate,
            output_path=delta_scheme_path,
            show=args.show,
        )
        saved_paths.append(delta_scheme_path)

    if args.plot_type in {"delta-to-oracle-missing-rate", "both"}:
        if args.generating_method is None:
            raise ValueError("--generating-method is required for delta-to-oracle-missing-rate.")
        delta_rate_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_delta_to_oracle_by_missing_rate_{args.generating_method}.png"
        )
        create_delta_to_oracle_by_missing_rate_plot(
            summary_df=summary_df,
            metric=args.metric,
            dataset=args.dataset,
            generating_method=args.generating_method,
            output_path=delta_rate_path,
            show=args.show,
        )
        saved_paths.append(delta_rate_path)

    print("Saved plots:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()

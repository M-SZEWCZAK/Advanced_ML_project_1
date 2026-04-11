"""Create report-style plots from outputs/tables/final_results.csv."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "outputs" / "tables" / "final_results.csv"
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


def _get_oracle_raw_stats(oracle_df, metric):
    """Return raw oracle values and their mean from repeated result rows."""
    if oracle_df.empty:
        return pd.Series(dtype=float), None

    oracle_values = oracle_df[metric].dropna()
    if oracle_values.empty:
        return oracle_values, None

    return oracle_values, float(oracle_values.mean())


def _create_boxplot(ax, data, positions, color):
    """Draw a boxplot with both median and mean markers shown."""
    ax.boxplot(
        data,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showmeans=True,
        boxprops={"facecolor": color, "alpha": 0.45},
        medianprops={"color": "black", "linewidth": 1.4},
        meanprops={
            "marker": "D",
            "markerfacecolor": color,
            "markeredgecolor": "black",
            "markersize": 5,
        },
        whiskerprops={"color": color},
        capprops={"color": color},
        flierprops={"markerfacecolor": color, "markeredgecolor": color, "alpha": 0.5},
    )


def _aggregate_mean_std(df, x_column, metric, methods):
    """Aggregate raw results into mean/std tables for plotting."""
    rows = []
    for x_value in sorted(df[x_column].dropna().unique()):
        for method in methods:
            values = df[(df[x_column] == x_value) & (df["display_method"] == method)][metric].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    x_column: x_value,
                    "display_method": method,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _compute_pairwise_differences(df, baseline_method, comparison_methods, metric, group_columns):
    """Compute seed-level metric differences between methods inside each group."""
    subset = df[group_columns + ["display_method", metric]].copy()
    pivot = subset.pivot_table(
        index=group_columns,
        columns="display_method",
        values=metric,
        aggfunc="first",
    ).reset_index()

    rows = []
    for method in comparison_methods:
        if baseline_method not in pivot.columns or method not in pivot.columns:
            continue
        method_rows = pivot[group_columns].copy()
        method_rows["display_method"] = method
        method_rows["difference"] = pivot[method] - pivot[baseline_method]
        rows.append(method_rows.dropna(subset=["difference"]))

    if not rows:
        return pd.DataFrame(columns=group_columns + ["display_method", "difference"])

    return pd.concat(rows, ignore_index=True)


def _compute_delta_to_oracle(df, metric, group_columns, methods):
    """Compute seed-level oracle-minus-method deltas inside each group."""
    subset = df[group_columns + ["display_method", metric]].copy()
    pivot = subset.pivot_table(
        index=group_columns,
        columns="display_method",
        values=metric,
        aggfunc="first",
    ).reset_index()

    rows = []
    for method in methods:
        if "oracle" not in pivot.columns or method not in pivot.columns:
            continue
        method_rows = pivot[group_columns].copy()
        method_rows["display_method"] = method
        method_rows["difference"] = pivot["oracle"] - pivot[method]
        rows.append(method_rows.dropna(subset=["difference"]))

    if not rows:
        return pd.DataFrame(columns=group_columns + ["display_method", "difference"])

    return pd.concat(rows, ignore_index=True)


def _plot_difference_line_from_raw(ax, diff_df, x_column, methods, ylabel):
    """Plot mean differences with std shading from seed-level raw differences."""
    plotted_values = []

    for method_index, method in enumerate(methods):
        method_df = diff_df[diff_df["display_method"] == method].copy()
        if method_df.empty:
            continue
        summary = method_df.groupby(x_column)["difference"].agg(["mean", "std"]).reset_index()
        x_values = summary[x_column].to_numpy()
        mean_values = summary["mean"].to_numpy()
        std_values = summary["std"].fillna(0.0).to_numpy()
        color = f"C{method_index}"

        ax.plot(
            x_values,
            mean_values,
            marker="o",
            linewidth=2,
            color=color,
            label=method,
        )
        ax.fill_between(
            x_values,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.15,
            color=color,
        )
        plotted_values.extend(mean_values.tolist())
        plotted_values.extend((mean_values - std_values).tolist())
        plotted_values.extend((mean_values + std_values).tolist())

    ax.axhline(0.0, linestyle="--", linewidth=1.2, color="black", alpha=0.6)
    ax.set_ylabel(ylabel)
    return plotted_values


def _plot_difference_boxplot_from_raw(ax, diff_df, x_values, x_column, methods, ylabel):
    """Plot seed-level differences as grouped boxplots."""
    base_positions = list(range(len(x_values)))
    box_width = 0.22
    plotted_values = []

    for method_index, method in enumerate(methods):
        method_data = []
        positions = []
        offset = (method_index - (len(methods) - 1) / 2) * box_width

        for x_index, x_value in enumerate(x_values):
            values = diff_df[
                (diff_df[x_column] == x_value) & (diff_df["display_method"] == method)
            ]["difference"].dropna()
            if values.empty:
                continue
            method_data.append(values.to_numpy())
            positions.append(base_positions[x_index] + offset)
            plotted_values.extend(values.tolist())

        if method_data:
            _create_boxplot(ax, method_data, positions, f"C{method_index}")
            ax.plot([], [], color=f"C{method_index}", linewidth=8, alpha=0.45, label=method)

    ax.axhline(0.0, linestyle="--", linewidth=1.2, color="black", alpha=0.6)
    ax.set_xticks(base_positions)
    ax.set_xticklabels([str(value) for value in x_values])
    ax.set_ylabel(ylabel)
    return plotted_values


def create_methods_boxplot(df, metric, dataset, output_path=None, show=False):
    """Create a grouped boxplot comparing methods across generating schemes.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw experiment results.
    metric : str
        One of accuracy, balanced_accuracy, f1, roc_auc.
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
        & (df["missing_rate"] == 0.3)
    ].copy()

    if filtered_df.empty:
        raise ValueError(f"No rows found for dataset={dataset!r} and missing_rate=0.3.")

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
            _create_boxplot(ax, method_data, positions, f"C{method_index}")
            ax.plot([], [], color=f"C{method_index}", linewidth=8, alpha=0.45, label=method)

    if oracle_mean is not None:
        oracle_x = -1
        ax.axhline(
            oracle_mean,
            linestyle="--",
            linewidth=1.8,
            color="black",
            alpha=0.35,
            label="oracle mean",
        )
        _create_boxplot(ax, [oracle_values.to_numpy()], [oracle_x], "#bdbdbd")
        ax.set_xlim(oracle_x - 0.6, max(base_positions) + 0.6)
        ax.set_xticks([oracle_x] + base_positions)
        ax.set_xticklabels(["oracle"] + schemes)
    else:
        ax.set_xticks(base_positions)
        ax.set_xticklabels(schemes)

    ax.set_xlabel("Missingness Schemes")
    ax.set_ylabel(metric.replace("_", " ").title())
    _set_dynamic_metric_ylim(ax, filtered_df[metric])
    ax.set_title(f"{dataset}: {metric.replace('_', ' ').title()} by missingness scheme")
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


def create_missing_rate_comparison_plot(df, metric, dataset, output_path=None, show=False):
    """Create boxplots comparing methods across missing-rate values for one scheme.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw experiment results.
    metric : str
        One of accuracy, balanced_accuracy, f1, roc_auc.
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
        & (df["scheme"] == "mcar")
    ].copy()

    if filtered_df.empty:
        raise ValueError(f"No rows found for dataset={dataset!r} and generating_method='mcar'.")

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
            _create_boxplot(ax, method_data, positions, f"C{method_index}")
            ax.plot([], [], color=f"C{method_index}", linewidth=8, alpha=0.45, label=method)

    if oracle_mean is not None:
        oracle_x = -1
        ax.axhline(
            oracle_mean,
            linestyle="--",
            linewidth=1.8,
            color="black",
            alpha=0.35,
            label="oracle mean",
        )
        _create_boxplot(ax, [oracle_values.to_numpy()], [oracle_x], "#bdbdbd")
        ax.set_xlim(oracle_x - 0.6, max(base_positions) + 0.6)
        ax.set_xticks([oracle_x] + base_positions)
        ax.set_xticklabels(["oracle"] + [str(rate) for rate in missing_rates])
    else:
        ax.set_xticks(base_positions)
        ax.set_xticklabels([str(rate) for rate in missing_rates])

    ax.set_xlabel("Missing Rate")
    ax.set_ylabel(metric.replace("_", " ").title())
    _set_dynamic_metric_ylim(ax, filtered_df[metric])
    ax.set_title(f"{dataset}: {metric.replace('_', ' ').title()} by MCAR missing rate")
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


def create_mcar_trend_plot(df, metric, dataset, output_path=None, show=False):
    """Create a trend plot across MCAR missing rates using raw results only."""
    _validate_metric(metric)
    filtered_df = df[
        (df["dataset"] == dataset)
        & (df["scheme"] == "mcar")
    ].copy()

    if filtered_df.empty:
        raise ValueError(f"No rows found for dataset={dataset!r} and scheme='mcar'.")

    missing_rates = sorted(filtered_df["missing_rate"].unique())
    oracle_df = filtered_df[filtered_df["display_method"] == "oracle"].copy()
    non_oracle_df = filtered_df[filtered_df["display_method"] != "oracle"].copy()
    methods = _sort_categories(non_oracle_df["display_method"].unique(), METHOD_ORDER)
    oracle_values, oracle_mean = _get_oracle_raw_stats(oracle_df, metric)
    aggregated_df = _aggregate_mean_std(non_oracle_df, "missing_rate", metric, methods)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_values = []

    for method_index, method in enumerate(methods):
        method_df = aggregated_df[aggregated_df["display_method"] == method].copy()
        method_df = method_df.sort_values("missing_rate")
        x_values = method_df["missing_rate"].to_numpy()
        mean_values = method_df["mean"].to_numpy()
        std_values = method_df["std"].fillna(0.0).to_numpy()
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
        plotted_values.extend(mean_values.tolist())
        plotted_values.extend((mean_values - std_values).tolist())
        plotted_values.extend((mean_values + std_values).tolist())

    if oracle_mean is not None:
        oracle_x = min(missing_rates) - 0.12
        ax.axhline(
            oracle_mean,
            linestyle="--",
            linewidth=1.8,
            color="black",
            alpha=0.35,
            label="oracle mean",
        )
        ax.errorbar(
            [oracle_x],
            [oracle_mean],
            yerr=[float(oracle_values.std(ddof=1)) if len(oracle_values) > 1 else 0.0],
            marker="s",
            markersize=7,
            linestyle="None",
            capsize=4,
            color="black",
        )
        ax.set_xlim(oracle_x - 0.06, max(missing_rates) + 0.06)
        ax.set_xticks([oracle_x] + missing_rates)
        ax.set_xticklabels(["oracle"] + [str(rate) for rate in missing_rates])
    else:
        ax.set_xticks(missing_rates)
        ax.set_xticklabels([str(rate) for rate in missing_rates])

    ax.set_xlabel("Missing Rate")
    ax.set_ylabel(metric.replace("_", " ").title())
    if oracle_mean is not None:
        plotted_values.extend(oracle_values.tolist())
    _set_dynamic_metric_ylim(
        ax,
        plotted_values,
        lower_bound=0.0,
        upper_bound=1.0,
        padding_ratio=0.28,
        min_span=0.10,
    )
    ax.set_title(f"{dataset}: {metric.replace('_', ' ').title()} trend across MCAR missing rate")
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
    df,
    metric,
    dataset,
    output_path=None,
    show=False,
):
    """Plot seed-level unlabeled improvement over naive across schemes as boxplots."""
    _validate_metric(metric)
    filtered_df = df[
        (df["dataset"] == dataset)
        & (df["missing_rate"] == 0.3)
    ].copy()

    if filtered_df.empty:
        raise ValueError(f"No rows found for dataset={dataset!r} and missing_rate=0.3.")

    schemes = _sort_categories(filtered_df["scheme"].unique(), SCHEME_ORDER)
    methods = [method for method in METHOD_ORDER if method.startswith("unlabeled-")]
    diff_df = _compute_pairwise_differences(
        df=filtered_df,
        baseline_method="naive",
        comparison_methods=methods,
        metric=metric,
        group_columns=["scheme", "seed"],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_values = _plot_difference_boxplot_from_raw(
        ax=ax,
        diff_df=diff_df,
        x_values=schemes,
        x_column="scheme",
        methods=methods,
        ylabel=f"{metric.replace('_', ' ').title()} improvement over naive",
    )
    _set_dynamic_metric_ylim(ax, plotted_values, lower_bound=-1.0, upper_bound=1.0, min_span=0.02)
    ax.set_xlabel("Missingness Scheme")
    ax.set_title(
        f"{dataset}: Unlabeled improvement over naive by missingness scheme"
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
    df,
    metric,
    dataset,
    output_path=None,
    show=False,
):
    """Plot unlabeled-method improvement over naive across MCAR missing rates."""
    _validate_metric(metric)
    filtered_df = df[
        (df["dataset"] == dataset)
        & (df["scheme"] == "mcar")
    ].copy()

    if filtered_df.empty:
        raise ValueError(f"No rows found for dataset={dataset!r} and scheme='mcar'.")

    missing_rates = sorted(filtered_df["missing_rate"].unique())
    methods = [method for method in METHOD_ORDER if method.startswith("unlabeled-")]
    diff_df = _compute_pairwise_differences(
        df=filtered_df,
        baseline_method="naive",
        comparison_methods=methods,
        metric=metric,
        group_columns=["missing_rate", "seed"],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_values = _plot_difference_line_from_raw(
        ax=ax,
        diff_df=diff_df,
        x_column="missing_rate",
        methods=methods,
        ylabel=f"{metric.replace('_', ' ').title()} improvement over naive",
    )
    _set_dynamic_metric_ylim(ax, plotted_values, lower_bound=-1.0, upper_bound=1.0, min_span=0.02)
    ax.set_xlabel("Missing Rate")
    ax.set_title(
        f"{dataset}: Unlabeled improvement over naive by MCAR missing rate"
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


def create_unlabeled_improvement_over_naive_by_missing_rate_boxplot(
    df,
    metric,
    dataset,
    output_path=None,
    show=False,
):
    """Plot seed-level unlabeled improvement over naive across MCAR missing rates as boxplots."""
    _validate_metric(metric)
    filtered_df = df[
        (df["dataset"] == dataset)
        & (df["scheme"] == "mcar")
    ].copy()

    if filtered_df.empty:
        raise ValueError(f"No rows found for dataset={dataset!r} and scheme='mcar'.")

    missing_rates = sorted(filtered_df["missing_rate"].unique())
    methods = [method for method in METHOD_ORDER if method.startswith("unlabeled-")]
    diff_df = _compute_pairwise_differences(
        df=filtered_df,
        baseline_method="naive",
        comparison_methods=methods,
        metric=metric,
        group_columns=["missing_rate", "seed"],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_values = _plot_difference_boxplot_from_raw(
        ax=ax,
        diff_df=diff_df,
        x_values=missing_rates,
        x_column="missing_rate",
        methods=methods,
        ylabel=f"{metric.replace('_', ' ').title()} improvement over naive",
    )
    _set_dynamic_metric_ylim(ax, plotted_values, lower_bound=-1.0, upper_bound=1.0, min_span=0.02)
    ax.set_xlabel("Missing Rate")
    ax.set_title(f"{dataset}: Unlabeled improvement over naive by MCAR missing rate")
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
    df,
    metric,
    dataset,
    output_path=None,
    show=False,
):
    """Plot seed-level delta to oracle across schemes as boxplots."""
    _validate_metric(metric)
    filtered_df = df[
        (df["dataset"] == dataset)
        & (df["missing_rate"] == 0.3)
    ].copy()

    if filtered_df.empty:
        raise ValueError(f"No rows found for dataset={dataset!r} and missing_rate=0.3.")

    schemes = _sort_categories(filtered_df["scheme"].unique(), SCHEME_ORDER)
    methods = [method for method in METHOD_ORDER if method != "oracle"]
    diff_df = _compute_delta_to_oracle(
        df=filtered_df,
        metric=metric,
        group_columns=["scheme", "seed"],
        methods=methods,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_values = _plot_difference_boxplot_from_raw(
        ax=ax,
        diff_df=diff_df,
        x_values=schemes,
        x_column="scheme",
        methods=methods,
        ylabel=f"Delta to oracle in {metric.replace('_', ' ').title()}",
    )
    _set_dynamic_metric_ylim(ax, plotted_values, lower_bound=-1.0, upper_bound=1.0, min_span=0.02)
    ax.set_xlabel("Missingness Scheme")
    ax.set_title(f"{dataset}: Delta to oracle by missingness scheme")
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
    df,
    metric,
    dataset,
    output_path=None,
    show=False,
):
    """Plot method delta to oracle across MCAR missing rates."""
    _validate_metric(metric)
    filtered_df = df[
        (df["dataset"] == dataset)
        & (df["scheme"] == "mcar")
    ].copy()

    if filtered_df.empty:
        raise ValueError(f"No rows found for dataset={dataset!r} and scheme='mcar'.")

    missing_rates = sorted(filtered_df["missing_rate"].unique())
    methods = [method for method in METHOD_ORDER if method != "oracle"]
    diff_df = _compute_delta_to_oracle(
        df=filtered_df,
        metric=metric,
        group_columns=["missing_rate", "seed"],
        methods=methods,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_values = _plot_difference_line_from_raw(
        ax=ax,
        diff_df=diff_df,
        x_column="missing_rate",
        methods=methods,
        ylabel=f"Delta to oracle in {metric.replace('_', ' ').title()}",
    )
    _set_dynamic_metric_ylim(ax, plotted_values, lower_bound=-1.0, upper_bound=1.0, min_span=0.02)
    ax.set_xlabel("Missing Rate")
    ax.set_title(f"{dataset}: Delta to oracle by MCAR missing rate")
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


def create_delta_to_oracle_by_missing_rate_boxplot(
    df,
    metric,
    dataset,
    output_path=None,
    show=False,
):
    """Plot seed-level delta to oracle across MCAR missing rates as boxplots."""
    _validate_metric(metric)
    filtered_df = df[
        (df["dataset"] == dataset)
        & (df["scheme"] == "mcar")
    ].copy()

    if filtered_df.empty:
        raise ValueError(f"No rows found for dataset={dataset!r} and scheme='mcar'.")

    missing_rates = sorted(filtered_df["missing_rate"].unique())
    methods = [method for method in METHOD_ORDER if method != "oracle"]
    diff_df = _compute_delta_to_oracle(
        df=filtered_df,
        metric=metric,
        group_columns=["missing_rate", "seed"],
        methods=methods,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_values = _plot_difference_boxplot_from_raw(
        ax=ax,
        diff_df=diff_df,
        x_values=missing_rates,
        x_column="missing_rate",
        methods=methods,
        ylabel=f"Delta to oracle in {metric.replace('_', ' ').title()}",
    )
    _set_dynamic_metric_ylim(ax, plotted_values, lower_bound=-1.0, upper_bound=1.0, min_span=0.02)
    ax.set_xlabel("Missing Rate")
    ax.set_title(f"{dataset}: Delta to oracle by MCAR missing rate")
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

def generate_all_boxplots(df, output_dir, show=False):
    datasets = sorted(df["dataset"].dropna().unique())
    metrics = VALID_METRICS

    saved_paths = []

    for dataset in datasets:
        for metric in metrics:
            print(f"Processing dataset={dataset}, metric={metric}")

            base_dir = output_dir / dataset / metric

            try:
                path = base_dir / "methods_boxplot.png"
                create_methods_boxplot(df, metric, dataset, path, show)
                saved_paths.append(path)
            except Exception as e:
                print(f"[SKIP] methods-boxplot: {e}")

            try:
                path = base_dir / "missing_rate_boxplot.png"
                create_missing_rate_comparison_plot(df, metric, dataset, path, show)
                saved_paths.append(path)
            except Exception as e:
                print(f"[SKIP] missing-rate-boxplot: {e}")

            try:
                path = base_dir / "mcar_trend.png"
                create_mcar_trend_plot(df, metric, dataset, path, show)
                saved_paths.append(path)
            except Exception as e:
                print(f"[SKIP] mcar-trend: {e}")

            try:
                path = base_dir / "improvement_over_naive_scheme.png"
                create_unlabeled_improvement_over_naive_by_scheme_plot(df, metric, dataset, path, show)
                saved_paths.append(path)
            except Exception as e:
                print(f"[SKIP] improvement-over-naive-scheme: {e}")

            try:
                path = base_dir / "improvement_over_naive_missing_rate.png"
                create_unlabeled_improvement_over_naive_by_missing_rate_plot(df, metric, dataset, path, show)
                saved_paths.append(path)
            except Exception as e:
                print(f"[SKIP] improvement-over-naive-missing-rate: {e}")

            try:
                path = base_dir / "improvement_over_naive_missing_rate_boxplot.png"
                create_unlabeled_improvement_over_naive_by_missing_rate_boxplot(df, metric, dataset, path, show)
                saved_paths.append(path)
            except Exception as e:
                print(f"[SKIP] improvement-over-naive-missing-rate-boxplot: {e}")

            try:
                path = base_dir / "delta_to_oracle_scheme.png"
                create_delta_to_oracle_by_scheme_plot(df, metric, dataset, path, show)
                saved_paths.append(path)
            except Exception as e:
                print(f"[SKIP] delta-to-oracle-scheme: {e}")

            try:
                path = base_dir / "delta_to_oracle_missing_rate.png"
                create_delta_to_oracle_by_missing_rate_plot(df, metric, dataset, path, show)
                saved_paths.append(path)
            except Exception as e:
                print(f"[SKIP] delta-to-oracle-missing-rate: {e}")

            try:
                path = base_dir / "delta_to_oracle_missing_rate_boxplot.png"
                create_delta_to_oracle_by_missing_rate_boxplot(df, metric, dataset, path, show)
                saved_paths.append(path)
            except Exception as e:
                print(f"[SKIP] delta-to-oracle-missing-rate-boxplot: {e}")

    return saved_paths

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
            "mcar-trend",
            "improvement-over-naive-scheme",
            "improvement-over-naive-missing-rate",
            "improvement-over-naive-missing-rate-boxplot",
            "delta-to-oracle-scheme",
            "delta-to-oracle-missing-rate",
            "delta-to-oracle-missing-rate-boxplot",
            "both",
        ],
        default="both",
        help="Which plot type to create.",
    )
    parser.add_argument(
    "--metric",
    choices=VALID_METRICS,
    help="Metric used on the y-axis.",
    )

    parser.add_argument(
        "--dataset",
        help="Dataset to plot.",
    )

    parser.add_argument(
        "--all-boxplots",
        action="store_true",
        help="Generate all plots for all datasets and metrics.",
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
    saved_paths = []

    if args.all_boxplots:
        saved_paths = generate_all_boxplots(
            df=df,
            output_dir=args.output_dir,
            show=args.show,
        )

        print("Saved plots:")
        for path in saved_paths:
            print(path)

        return
    
    if args.metric is None or args.dataset is None:
        raise ValueError(
            "--metric and --dataset are required unless using --all-boxplots"
        )
    
    if args.plot_type in {"methods-boxplot", "both"}:
        methods_plot_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_methods_by_scheme_missing_rate_0.3.png"
        )
        create_methods_boxplot(
            df=df,
            metric=args.metric,
            dataset=args.dataset,
            output_path=methods_plot_path,
            show=args.show,
        )
        saved_paths.append(methods_plot_path)

    if args.plot_type in {"missing-rate-boxplot", "both"}:
        rates_plot_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_missing_rate_comparison_mcar.png"
        )
        create_missing_rate_comparison_plot(
            df=df,
            metric=args.metric,
            dataset=args.dataset,
            output_path=rates_plot_path,
            show=args.show,
        )
        saved_paths.append(rates_plot_path)

    if args.plot_type in {"mcar-trend", "both"}:
        mcar_trend_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_mcar_trend.png"
        )
        create_mcar_trend_plot(
            df=df,
            metric=args.metric,
            dataset=args.dataset,
            output_path=mcar_trend_path,
            show=args.show,
        )
        saved_paths.append(mcar_trend_path)

    if args.plot_type in {"improvement-over-naive-scheme", "both"}:
        improvement_scheme_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_improvement_over_naive_by_scheme.png"
        )
        create_unlabeled_improvement_over_naive_by_scheme_plot(
            df=df,
            metric=args.metric,
            dataset=args.dataset,
            output_path=improvement_scheme_path,
            show=args.show,
        )
        saved_paths.append(improvement_scheme_path)

    if args.plot_type in {"improvement-over-naive-missing-rate", "both"}:
        improvement_rate_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_improvement_over_naive_by_missing_rate_mcar.png"
        )
        create_unlabeled_improvement_over_naive_by_missing_rate_plot(
            df=df,
            metric=args.metric,
            dataset=args.dataset,
            output_path=improvement_rate_path,
            show=args.show,
        )
        saved_paths.append(improvement_rate_path)

    if args.plot_type in {"improvement-over-naive-missing-rate-boxplot", "both"}:
        improvement_rate_boxplot_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_improvement_over_naive_by_missing_rate_mcar_boxplot.png"
        )
        create_unlabeled_improvement_over_naive_by_missing_rate_boxplot(
            df=df,
            metric=args.metric,
            dataset=args.dataset,
            output_path=improvement_rate_boxplot_path,
            show=args.show,
        )
        saved_paths.append(improvement_rate_boxplot_path)

    if args.plot_type in {"delta-to-oracle-scheme", "both"}:
        delta_scheme_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_delta_to_oracle_by_scheme.png"
        )
        create_delta_to_oracle_by_scheme_plot(
            df=df,
            metric=args.metric,
            dataset=args.dataset,
            output_path=delta_scheme_path,
            show=args.show,
        )
        saved_paths.append(delta_scheme_path)

    if args.plot_type in {"delta-to-oracle-missing-rate", "both"}:
        delta_rate_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_delta_to_oracle_by_missing_rate_mcar.png"
        )
        create_delta_to_oracle_by_missing_rate_plot(
            df=df,
            metric=args.metric,
            dataset=args.dataset,
            output_path=delta_rate_path,
            show=args.show,
        )
        saved_paths.append(delta_rate_path)

    if args.plot_type in {"delta-to-oracle-missing-rate-boxplot", "both"}:
        delta_rate_boxplot_path = args.output_dir / (
            f"{args.dataset}_{args.metric}_delta_to_oracle_by_missing_rate_mcar_boxplot.png"
        )
        create_delta_to_oracle_by_missing_rate_boxplot(
            df=df,
            metric=args.metric,
            dataset=args.dataset,
            output_path=delta_rate_boxplot_path,
            show=args.show,
        )
        saved_paths.append(delta_rate_boxplot_path)

    print("Saved plots:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()

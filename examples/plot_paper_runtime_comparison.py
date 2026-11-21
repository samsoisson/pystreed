"""Compare published STreeD runtimes with the 14-dataset experiment.

python examples/plot_paper_runtime_comparison.py
python examples/plot_paper_runtime_comparison.py --runtime-metric training_runtime_s
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plot_binary_classification_results import benchmark_datasets


CONFIGURATIONS = (("accuracy", 4, 10), ("accuracy", 5, 10), ("f1", 3, 7), ("f1", 4, 7))
SEEDS = {11, 12, 13, 14, 15}


def parse_paper_runtime(raw, timeout=600):
    """Keep exact times distinct from upper/lower bounds and absent reports."""
    if raw is None or pd.isna(raw):
        return "not_reported", np.nan, np.nan, "N/R"
    value = str(raw).strip()
    if value.replace(" ", "") == "<1":
        return "under_1", np.nan, 1., "<1"
    if value == "-":
        return "timeout", np.nan, float(timeout), f">{timeout:g}"
    seconds = float(value)
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"Invalid published runtime: {raw!r}")
    return "exact", seconds, np.nan, f"{seconds:g}"


def comparison_rows(results, paper, datasets, runtime_metric):
    rows = []
    for objective, depth, table in CONFIGURATIONS:
        paper_path = paper / f"streed_paper_table{table}_runtimes.csv"
        published = pd.read_csv(paper_path, dtype=str)
        if published.dataset.duplicated().any():
            raise ValueError(f"Duplicate datasets in {paper_path}")
        published = published.set_index("dataset")
        paper_column = f"streed_depth{depth}_runtime_s"
        if paper_column not in published:
            raise ValueError(f"Missing {paper_column} in {paper_path}")
        trial_path = results / f"depth{depth}" / f"trials_{objective}.csv"
        trials = pd.read_csv(trial_path)
        required = {"dataset", "seed", "status", runtime_metric}
        if required - set(trials):
            raise ValueError(f"Missing columns {sorted(required - set(trials))} in {trial_path}")
        trials = trials[trials.dataset.isin(datasets)].copy()
        if trials.duplicated(["dataset", "seed"]).any():
            raise ValueError(f"Duplicate trial seeds in {trial_path}")
        for dataset in datasets:
            if dataset in published.index:
                record = published.loc[dataset]
                raw = record[paper_column]
                timeout = float(record.get("timeout_s", 600))
            else:
                raw, timeout = None, 600.
            kind, seconds, bound, label = parse_paper_runtime(raw, timeout)
            group = trials[trials.dataset == dataset]
            ok = group[group.status == "ok"]
            complete = len(group) == 5 and len(ok) == 5 and set(ok.seed) == SEEDS
            values = pd.to_numeric(ok[runtime_metric], errors="coerce")
            valid = complete and np.isfinite(values).all() and values.gt(0).all()
            rows.append({
                "objective": objective, "max_depth": depth, "dataset": dataset,
                "paper_table": table, "paper_reported": raw, "paper_status": kind,
                "paper_runtime_s": seconds, "paper_bound_s": bound, "paper_label": label,
                "experiment_status": "complete" if valid else ("not_started" if group.empty else "incomplete"),
                "successful_trials": len(ok), "runtime_metric": runtime_metric,
                "experiment_mean_s": values.mean() if valid else np.nan,
                "experiment_seed_std_s": values.std(ddof=1) if valid else np.nan,
                "paper_source": str(paper_path.resolve()), "experiment_source": str(trial_path.resolve()),
            })
    return pd.DataFrame(rows)


def runtime_error_extents(means, deviations, floor):
    """Clip only the displayed lower whisker; retain the true SD in the CSV."""
    means, deviations = np.asarray(means), np.asarray(deviations)
    return np.minimum(deviations, np.maximum(0, means - floor)), deviations


def plot_comparisons(frame, output, scale="log"):
    paper_color, experiment_color = "#3976AF", "#E88927"
    for (objective, depth), group in frame.groupby(["objective", "max_depth"], sort=False):
        group = group.reset_index(drop=True)
        x = np.arange(len(group))
        width = .36
        display_paper = group.paper_runtime_s.fillna(group.paper_bound_s).to_numpy()
        experiment = group.experiment_mean_s.to_numpy()
        deviations = group.experiment_seed_std_s.to_numpy()
        finite = np.concatenate((display_paper[np.isfinite(display_paper)], experiment[np.isfinite(experiment)]))
        if not len(finite):
            raise ValueError(f"No plottable runtimes for {objective}, depth {depth}")
        lower = 10 ** np.floor(np.log10(finite.min() / 3)) if scale == "log" else 0.
        error_tops = experiment + deviations
        upper = np.concatenate((finite, error_tops[np.isfinite(error_tops)])).max() * (3 if scale == "log" else 1.3)
        fig, ax = plt.subplots(figsize=(18, 7.5), layout="constrained")
        left = ax.bar(x - width / 2, display_paper, width, color=paper_color)
        right = ax.bar(x + width / 2, experiment, width, color=experiment_color)
        if scale == "log":
            ax.set_yscale("log")
        ax.set_ylim(lower, upper)
        valid = np.isfinite(experiment) & np.isfinite(deviations)
        downward, upward = runtime_error_extents(experiment[valid], deviations[valid], lower)
        ax.errorbar(x[valid] + width / 2, experiment[valid], yerr=np.vstack((downward, upward)),
                    fmt="none", ecolor="black", elinewidth=1.1, capsize=3, capthick=1.1)
        clipped = valid & ((experiment - deviations) < lower)
        if clipped.any():
            ax.scatter(x[clipped] + width / 2, np.full(clipped.sum(), lower * 1.05 if scale == "log" else lower),
                       marker="v", color="black", s=18, zorder=4, clip_on=False)
        missing_y = lower * 1.5 if scale == "log" else upper * .025
        for index, row in group.iterrows():
            if row.paper_status == "not_reported":
                ax.text(index - width / 2, missing_y, "N/R", ha="center", va="bottom", fontsize=8, rotation=90)
            else:
                if row.paper_status != "exact":
                    left[index].set_hatch("///" if row.paper_status == "under_1" else "xxx")
                    left[index].set_edgecolor("white")
                ax.annotate(row.paper_label, (index - width / 2, display_paper[index]),
                            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
            if row.experiment_status != "complete":
                ax.text(index + width / 2, missing_y, "Incomplete", ha="center", va="bottom", fontsize=8, rotation=90)
            else:
                ax.annotate(f"{experiment[index]:.2f}", (index + width / 2, error_tops[index]),
                            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        labels = [name.replace("-", " ").title() for name in group.dataset]
        ax.set_xticks(x, labels, rotation=40, ha="right")
        metric = group.runtime_metric.iloc[0]
        scope = "final refit" if metric == "refit_runtime_s" else "tuning plus refit"
        ax.set_title(f"Paper versus experiment: {objective.upper()} optimization, maximum depth {depth}\nExperiment runtime: {scope}")
        ax.set_ylabel(f"Runtime (seconds; {scale} scale)")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, which="major", alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=[Patch(color=paper_color, label="Paper STreeD: reported five-run mean"),
                           Patch(color=experiment_color, label="Experiment: five-seed mean ±1 SD")], loc="upper left")
        fig.supxlabel(
            "Experiment whiskers: ±1 sample SD over five seeds; downward triangles mark lower whiskers clipped at the axis floor.\n"
            "Hatched <1 bars show an upper bound; N/R = not reported; timeout bars show a lower bound. Paper SD is unavailable.\n"
            "Reference comparison: data, constraints, hardware and timing scopes differ.",
            fontsize=9)
        for extension in ("png", "svg"):
            fig.savefig(output / f"{objective}_depth{depth}.{extension}", dpi=200)
        plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=root / "results/binary_classification")
    parser.add_argument("--paper-dir", type=Path, default=root / "results/paper")
    parser.add_argument("--output-dir", type=Path, help="Default: <results-dir>/graphs/paper_comparison")
    parser.add_argument("--runtime-metric", choices=("refit_runtime_s", "training_runtime_s"), default="refit_runtime_s")
    parser.add_argument("--scale", choices=("log", "linear"), default="log")
    args = parser.parse_args()
    datasets = benchmark_datasets()
    try:
        frame = comparison_rows(args.results_dir, args.paper_dir, datasets, args.runtime_metric)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot compare runtimes: {exc}\n")
    output = args.output_dir or args.results_dir / "graphs/paper_comparison"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "paper_runtime_comparison.csv", index=False)
    plot_comparisons(frame, output, args.scale)
    (output / "comparison_config.json").write_text(json.dumps({
        "datasets": list(datasets), "runtime_metric": args.runtime_metric, "scale": args.scale,
        "paper_sources": sorted(frame.paper_source.unique()),
        "experiment_sources": sorted(frame.experiment_source.unique()),
        "aggregation": "experiment mean of five successful seeds for each dataset/configuration",
        "paper_bound_display": "<1 shown at upper bound 1; timeout at threshold; missing values unfilled",
        "error_bars": "experiment plus/minus one sample seed SD (ddof=1); lower whiskers clipped at axis floor and marked; paper SD unavailable",
        "comparison_scope": "reference only; experiment uses train/test splitting and CV-selected complexity",
    }, indent=2), encoding="utf-8")
    print(frame.groupby(["objective", "max_depth", "paper_status", "experiment_status"]).size().to_string())
    print(f"Saved four two-color charts as PNG and SVG under {output.resolve()}")


if __name__ == "__main__":
    main()

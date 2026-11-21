"""Plot saved experiment summaries; no models are trained.

python examples/plot_binary_classification_results.py
python examples/plot_binary_classification_results.py --results-dir results/other_run
"""

import argparse
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = {
    "accuracy": ("Accuracy", "Mean test accuracy"),
    "f1": ("F1 score", "Mean test F1 score"),
    "mcc": ("MCC score", "Mean test MCC"),
    "training_runtime_s": ("Training runtime", "Mean training runtime (seconds)"),
}


def benchmark_datasets():
    """Read the training cohort without importing or executing the trainer."""
    path = Path(__file__).with_name("binary_classification_trials.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DATASETS" for target in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise ValueError(f"No DATASETS cohort found in {path}")


def filter_benchmark(frame, datasets):
    selected = frame[frame.dataset.isin(datasets)].copy()
    if selected.empty:
        raise ValueError("No results for the selected 14-dataset benchmark")
    return selected


def read_summaries(root):
    """Read one summary per export, using metadata for older files."""
    files = sorted(root.rglob("summary*.csv"))
    frames, sources = [], []
    for path in files:
        if path.name != "summary.csv" and not path.name.startswith("summary_"):
            continue
        # The original unsuffixed export is accuracy; prefer its newer replacement.
        if path.name == "summary.csv" and (path.parent / "summary_accuracy.csv").exists():
            print(f"Skipping legacy duplicate: {path}")
            continue
        suffix = path.stem[len("summary"):]
        config_path = path.with_name(f"config{suffix}.json")
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            raise ValueError(f"Empty summary file: {path}") from None
        for column, fallback in (
            ("objective", config.get("objective", "accuracy" if not suffix else suffix[1:])),
            ("max_depth", config.get("max_depth")),
        ):
            if column not in frame:
                if fallback is None:
                    raise ValueError(f"{path}: missing {column} and matching configuration metadata")
                frame[column] = fallback
            elif fallback is not None and not frame[column].dropna().eq(fallback).all():
                raise ValueError(f"{path}: {column} disagrees with configuration metadata/filename")
        required = {"dataset", "objective", "max_depth", "complete", "successful_trials"}
        required.update(f"{metric}_mean" for metric in METRICS)
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{path}: no dataset results")
        if frame[["dataset", "objective", "max_depth"]].isna().any().any():
            raise ValueError(f"{path}: missing dataset/objective/depth labels")
        depth = pd.to_numeric(frame.max_depth, errors="raise")
        if ((depth < 0) | (depth > 20) | (depth % 1 != 0)).any():
            raise ValueError(f"{path}: invalid maximum tree depth")
        frame["max_depth"] = depth.astype(int)
        # Both objectives now write depth subdirectories. Prefer those exports
        # over older base-directory files for the same objective and depth.
        if frame[["objective", "max_depth"]].drop_duplicates().shape[0] == 1:
            objective = frame.objective.iloc[0]
            maximum_depth = frame.max_depth.iloc[0]
            replacement = path.parent / f"depth{maximum_depth}" / f"summary_{objective}.csv"
            if replacement.exists():
                print(f"Skipping superseded base-directory export: {path}")
                continue
        frame["complete"] = frame.complete.astype(str).str.lower().eq("true")
        frame["successful_trials"] = pd.to_numeric(frame.successful_trials, errors="raise")
        valid = frame.complete & frame.successful_trials.eq(5)
        for metric in METRICS:
            column = f"{metric}_mean"
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            values = frame[column]
            valid &= np.isfinite(values)
            if metric == "training_runtime_s":
                valid &= values.ge(0)
            else:
                valid &= values.between(-1 if metric == "mcc" else 0, 1)
        frame["valid"] = valid
        frame["source"] = str(path.resolve())
        frames.append(frame)
        sources.append(str(path.resolve()))
    if not frames:
        raise ValueError(f"No summary CSV files found under {root}")
    combined = pd.concat(frames, ignore_index=True)
    keys = ["objective", "max_depth", "dataset"]
    duplicates = combined.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError("Duplicate configuration/dataset results; select a single run directory:\n" +
                         combined.loc[duplicates, keys + ["source"]].to_string(index=False))
    return combined, sources


def aggregate_results(frame, policy="common"):
    """Mean the five-seed dataset means, giving every dataset equal weight."""
    if policy not in ("common", "available"):
        raise ValueError("dataset policy must be common or available")
    groups = list(frame.groupby(["objective", "max_depth"], sort=True))
    usable = [set(group.loc[group.valid, "dataset"]) for _, group in groups]
    common = set.intersection(*usable)
    if policy == "common" and not common:
        raise ValueError("No complete datasets are shared by all configurations")
    all_datasets = sorted(set(frame.dataset))
    rows, coverage = [], []
    for (objective, depth), group in groups:
        included = set(group.loc[group.valid, "dataset"]) if policy == "available" else common
        if not included:
            raise ValueError(f"No complete results for {objective}, depth {depth}")
        selected = group[group.dataset.isin(included) & group.valid]
        row = {"objective": objective, "max_depth": int(depth), "n_datasets": len(selected),
               "dataset_policy": policy}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(selected[f"{metric}_mean"].mean())
            row[f"{metric}_dataset_std"] = float(selected[f"{metric}_mean"].std(ddof=1))
        rows.append(row)
        indexed = group.set_index("dataset")
        for dataset in all_datasets:
            reason = "included" if dataset in included else (
                "missing configuration result" if dataset not in indexed.index else
                "incomplete/invalid result" if not indexed.loc[dataset, "valid"] else
                "not complete in every configuration"
            )
            coverage.append({"objective": objective, "max_depth": depth, "dataset": dataset,
                             "included": dataset in included, "reason": reason})
    return pd.DataFrame(rows), pd.DataFrame(coverage)


def add_seed_statistics(aggregate, coverage, summaries):
    """Average datasets within each seed, then calculate sample SD over seeds."""
    aggregate = aggregate.copy()
    seed_rows = []
    expected = {11, 12, 13, 14, 15}
    for index, config in aggregate.iterrows():
        reference = summaries[(summaries.objective == config.objective) &
                              (summaries.max_depth == config.max_depth)]
        sources = reference.source.unique()
        if len(sources) != 1:
            raise ValueError(f"Expected one trial source for {config.objective}, depth {config.max_depth}")
        source = Path(sources[0])
        path = source.with_name(source.name.replace("summary", "trials", 1))
        trials = pd.read_csv(path)
        selected = coverage[(coverage.objective == config.objective) &
                            (coverage.max_depth == config.max_depth) & coverage.included]
        names = set(selected.dataset)
        required = {"dataset", "seed", "status", *METRICS}
        if required - set(trials):
            raise ValueError(f"Missing trial columns {sorted(required - set(trials))} in {path}")
        trials = trials[trials.dataset.isin(names)].copy()
        if set(trials.dataset) != names or trials.duplicated(["dataset", "seed"]).any():
            raise ValueError(f"Missing datasets or duplicate seed records in {path}")
        for name, group in trials.groupby("dataset"):
            if len(group) != 5 or set(group.seed) != expected or not group.status.eq("ok").all():
                raise ValueError(f"Need five successful seeds for {name} in {path}")
        reference = reference.set_index("dataset").loc[sorted(names)]
        for metric in METRICS:
            trials[metric] = pd.to_numeric(trials[metric], errors="raise")
            if not np.isfinite(trials[metric]).all():
                raise ValueError(f"Nonfinite {metric} in {path}")
            actual = trials.groupby("dataset")[metric].mean().reindex(reference.index)
            if not np.allclose(actual, reference[f"{metric}_mean"], rtol=1e-9, atol=1e-10):
                raise ValueError(f"Stale summary: {metric} does not match {path}")
        by_seed = trials.groupby("seed")[list(METRICS)].mean().sort_index()
        for metric in METRICS:
            aggregate.loc[index, f"{metric}_mean"] = by_seed[metric].mean()
            aggregate.loc[index, f"{metric}_seed_std"] = by_seed[metric].std(ddof=1)
        for seed, values in by_seed.iterrows():
            seed_rows.append({"objective": config.objective, "max_depth": config.max_depth,
                              "seed": int(seed), "n_datasets": len(names), **values.to_dict()})
    return aggregate, pd.DataFrame(seed_rows)


def plot_results(aggregate, output):
    objectives = sorted(aggregate.objective.unique())
    palette = plt.get_cmap("tab10")
    colors = {objective: palette(i % 10) for i, objective in enumerate(objectives)}
    labels = [f"{row.objective.upper()}\nDepth {row.max_depth}"
              for row in aggregate.itertuples()]
    for metric, (title, ylabel) in METRICS.items():
        fig, ax = plt.subplots(figsize=(max(6, 2 * len(aggregate)), 5), layout="constrained")
        values = aggregate[f"{metric}_mean"].to_numpy()
        errors = aggregate[f"{metric}_seed_std"].to_numpy()
        bars = ax.bar(np.arange(len(aggregate)), values,
                      color=[colors[x] for x in aggregate.objective], width=0.6,
                      yerr=errors, capsize=5, error_kw={"elinewidth": 1.4, "capthick": 1.4})
        ax.set_xticks(np.arange(len(aggregate)), labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} by training objective and maximum depth")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if metric in ("accuracy", "f1"):
            ax.set_ylim(0, 1.1)
        elif metric == "mcc":
            ax.set_ylim(-1, 1.1)
            ax.axhline(0, color="grey", linewidth=0.7)
        else:
            ax.set_ylim(0, max(1, float((values + errors).max()) * 1.2))
        for position, (value, error) in enumerate(zip(values, errors)):
            label = f"{value:.3f}" if metric != "training_runtime_s" else f"{value:.2f}"
            ax.annotate(label, (position, value + error), xytext=(0, 5), textcoords="offset points", ha="center")
        fig.supxlabel("Equal weight per dataset; error bars = ±1 sample SD across five seed-level dataset averages", fontsize=9)
        for extension in ("png", "svg"):
            fig.savefig(output / f"{metric}.{extension}", dpi=200)
        plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=root / "results/binary_classification")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: <results-dir>/graphs")
    parser.add_argument("--dataset-policy", choices=("common", "available"), default="common",
                        help="Use shared complete datasets, or each configuration's available datasets")
    args = parser.parse_args()
    try:
        frame, sources = read_summaries(args.results_dir)
        datasets = benchmark_datasets()
        excluded = sorted(set(frame.dataset) - set(datasets))
        frame = filter_benchmark(frame, datasets)
        frame = frame[~((frame.objective == "accuracy") & (frame.max_depth == 3))].copy()
        if frame.empty:
            raise ValueError("No configurations remain after excluding accuracy depth 3")
        aggregate, coverage = aggregate_results(frame, args.dataset_policy)
        aggregate, seed_averages = add_seed_statistics(aggregate, coverage, frame)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Cannot plot results: {exc}\n")
    output = args.output_dir or args.results_dir / "graphs"
    output.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output / "configuration_averages.csv", index=False)
    coverage.to_csv(output / "dataset_coverage.csv", index=False)
    seed_averages.to_csv(output / "seed_averages.csv", index=False)
    (output / "plot_config.json").write_text(json.dumps({
        "sources": sources, "dataset_policy": args.dataset_policy,
        "dataset_cohort": list(datasets), "excluded_historical_datasets": excluded,
        "excluded_configurations": [{"objective": "accuracy", "max_depth": 3}],
        "aggregation": "unweighted mean of complete five-seed dataset means",
        "runtime": "total tuning and final refit seconds, as recorded by the experiment",
        "error_bars": "plus/minus one sample SD (ddof=1) across five per-seed dataset averages; not a confidence interval",
    }, indent=2), encoding="utf-8")
    plot_results(aggregate, output)
    print(f"Restricted graphs and coverage to the {len(datasets)}-dataset training cohort")
    if excluded:
        print(f"Ignored historical datasets: {', '.join(excluded)}")
    print(aggregate.to_string(index=False))
    print(f"Excluded dataset/configuration pairs: {(~coverage.included).sum()}; see dataset_coverage.csv")
    if len(aggregate) == 1:
        print("Only one configuration is available; add other objective/depth results for comparison.")
    print(f"Saved four graphs as PNG and SVG under {output.resolve()}")


if __name__ == "__main__":
    main()

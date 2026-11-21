import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def comparison(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/plot_paper_runtime_comparison.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("paper_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value, kind, exact, bound, label", [
    ("< 1", "under_1", None, 1, "<1"),
    ("-", "timeout", None, 600, ">600"),
    ("7", "exact", 7, None, "7"),
    (None, "not_reported", None, None, "N/R"),
])
def test_paper_values_preserve_censoring(comparison, value, kind, exact, bound, label):
    actual_kind, actual_exact, actual_bound, actual_label = comparison.parse_paper_runtime(value)
    assert actual_kind == kind and actual_label == label
    assert np.isnan(actual_exact) if exact is None else actual_exact == exact
    assert np.isnan(actual_bound) if bound is None else actual_bound == bound


def exports(comparison, root):
    results, paper = root / "experiment", root / "paper"
    paper.mkdir()
    pd.DataFrame([{"dataset": "anneal", "streed_depth4_runtime_s": "< 1",
                   "streed_depth5_runtime_s": "5", "timeout_s": 600}]).to_csv(
        paper / "streed_paper_table10_runtimes.csv", index=False)
    pd.DataFrame([{"dataset": name, "streed_depth3_runtime_s": "< 1",
                   "streed_depth4_runtime_s": "2", "timeout_s": 600}
                  for name in ("anneal", "primary-tumor")]).to_csv(
        paper / "streed_paper_table7_runtimes.csv", index=False)
    for objective, depth, _ in comparison.CONFIGURATIONS:
        directory = results / f"depth{depth}"
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"dataset": name, "seed": seed, "status": "ok",
                       "refit_runtime_s": seed - 10, "training_runtime_s": 10 * (seed - 10)}
                      for name in ("anneal", "primary-tumor", "german-credit")
                      for seed in range(11, 16)]).to_csv(directory / f"trials_{objective}.csv", index=False)
    return results, paper


def test_comparison_uses_correct_tables_metrics_and_retained_datasets(comparison, tmp_path):
    results, paper = exports(comparison, tmp_path)
    frame = comparison.comparison_rows(results, paper, ("anneal", "primary-tumor"), "refit_runtime_s")
    assert len(frame) == 8
    assert frame.experiment_mean_s.eq(3).all()
    assert np.allclose(frame.experiment_seed_std_s, np.std([1, 2, 3, 4, 5], ddof=1))
    assert frame.loc[frame.objective == "accuracy", "paper_table"].eq(10).all()
    assert frame.loc[frame.objective == "f1", "paper_table"].eq(7).all()
    missing = frame[(frame.objective == "accuracy") & (frame.dataset == "primary-tumor")]
    assert len(missing) == 2 and missing.paper_status.eq("not_reported").all()
    assert missing.paper_runtime_s.isna().all() and missing.paper_bound_s.isna().all()
    total = comparison.comparison_rows(results, paper, ("anneal",), "training_runtime_s")
    assert total.experiment_mean_s.eq(30).all()


def test_incomplete_trials_are_not_averaged(comparison, tmp_path):
    results, paper = exports(comparison, tmp_path)
    path = results / "depth4/trials_f1.csv"
    frame = pd.read_csv(path)
    frame.loc[(frame.dataset == "anneal") & (frame.seed == 13), "status"] = "failed"
    frame.to_csv(path, index=False)
    compared = comparison.comparison_rows(results, paper, ("anneal",), "refit_runtime_s")
    failed = compared[(compared.objective == "f1") & (compared.max_depth == 4)].iloc[0]
    assert failed.experiment_status == "incomplete" and failed.successful_trials == 4
    assert np.isnan(failed.experiment_mean_s)


def test_four_two_color_charts_exported_with_missing_paper_values(comparison, tmp_path):
    results, paper = exports(comparison, tmp_path)
    frame = comparison.comparison_rows(results, paper, ("anneal", "primary-tumor"), "refit_runtime_s")
    output = tmp_path / "graphs"
    output.mkdir()
    comparison.plot_comparisons(frame, output)
    for objective, depth, _ in comparison.CONFIGURATIONS:
        for extension in ("png", "svg"):
            assert (output / f"{objective}_depth{depth}.{extension}").stat().st_size > 100


def test_runtime_whiskers_clip_only_the_displayed_lower_end(comparison):
    downward, upward = comparison.runtime_error_extents(np.array([1., 5.]), np.array([2., 1.]), .01)
    np.testing.assert_allclose(downward, [.99, 1.])
    np.testing.assert_allclose(upward, [2., 1.])

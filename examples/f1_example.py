"""Reproduce one F1-optimized tree and verify scores against scikit-learn.

Run from any directory with the locally built pystreed environment.
Use --test-size 0 --dataset yeast --depth 2 to reproduce the C++ F1 fixture.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

from pystreed import STreeDClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="soybean")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--time-limit", type=float, default=120)
    args = parser.parse_args()
    if not 0 <= args.test_size < 1:
        parser.error("--test-size must be in [0, 1)")

    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "classification" / f"{args.dataset}.csv"
    data = np.loadtxt(path, dtype=np.int32)
    X, y = data[:, 1:], data[:, 0]
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("This example requires labels 0 and 1.")
    if args.test_size:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=args.seed
        )
    else:
        X_train, y_train = X, y

    model = STreeDClassifier(
        optimization_task="f1-score",
        max_depth=args.depth,
        max_num_nodes=2**args.depth - 1,
        time_limit=args.time_limit,
        random_seed=args.seed,
        hyper_tune=False,
    )
    start = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - start
    if not model.is_fitted():
        raise RuntimeError("The solver did not return a feasible tree.")

    def evaluate(features, labels):
        predictions = model.predict(features)
        expected = f1_score(labels, predictions, zero_division=0)
        actual = model.score(features, labels)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        return {
            "rows": len(labels),
            "f1": float(expected),
            "mcc": float(matthews_corrcoef(labels, predictions)),
            "accuracy": float(accuracy_score(labels, predictions)),
            "confusion": {key: int(value) for key, value in zip(
                ("tn", "fp", "fn", "tp"), (tn, fp, fn, tp)
            )},
        }

    train = evaluate(X_train, y_train)
    np.testing.assert_allclose(model.fit_result.score(), train["f1"], rtol=0, atol=1e-12)
    result = {
        "dataset": args.dataset,
        "seed": args.seed,
        "features": X.shape[1],
        "max_depth": args.depth,
        "max_num_nodes": model.max_num_nodes,
        "time_limit_seconds": args.time_limit,
        "fit_seconds": fit_seconds,
        "is_optimal": model.fit_result.is_optimal(),
        "tree_depth": model.get_depth(),
        "branching_nodes": model.fit_result.tree_nodes(),
        "train": train,
        "score_checks": "passed",
    }
    if args.test_size:
        result["test"] = evaluate(X_test, y_test)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Run with: build/venv/bin/python -m unittest discover -s test -p 'test_mcc.py'."""

from functools import lru_cache
from itertools import product
import unittest

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

from pystreed import STreeDClassifier


def all_predictions(X, depth, nodes, min_leaf=1):
    """Enumerate every feasible prediction vector, without pruning or DP costs."""
    @lru_cache(None)
    def enumerate_subtree(rows, remaining_depth, remaining_nodes):
        # Bit masks represent predictions on the original row indices.
        candidates = {0, sum(1 << row for row in rows)}
        if remaining_depth == 0 or remaining_nodes == 0:
            return candidates
        for feature in range(X.shape[1]):
            left = tuple(row for row in rows if X[row, feature] == 0)
            right = tuple(row for row in rows if X[row, feature] == 1)
            if min(len(left), len(right)) < min_leaf:
                continue
            for left_budget in range(remaining_nodes):
                right_budget = remaining_nodes - 1 - left_budget
                for lpred in enumerate_subtree(left, remaining_depth - 1, left_budget):
                    for rpred in enumerate_subtree(right, remaining_depth - 1, right_budget):
                        candidates.add(lpred | rpred)
        return candidates

    return np.array([
        [(mask >> row) & 1 for row in range(len(X))]
        for mask in sorted(enumerate_subtree(tuple(range(len(X))), depth, nodes))
    ])


def best_correlation(y, predictions):
    """Independent Pearson-correlation oracle for binary MCC, including constants."""
    centered_y = y - y.mean()
    centered_predictions = predictions - predictions.mean(axis=1, keepdims=True)
    denominators = np.linalg.norm(centered_predictions, axis=1) * np.linalg.norm(centered_y)
    scores = np.divide(
        centered_predictions @ centered_y,
        denominators,
        out=np.zeros(len(predictions)),
        where=denominators != 0,
    )
    return scores.max()


class MCCTest(unittest.TestCase):
    def fit_and_check(self, X, y, depth=2, nodes=3, **options):
        model = STreeDClassifier(
            optimization_task="mcc", max_depth=depth, max_num_nodes=nodes,
            time_limit=30, random_seed=11, **options
        ).fit(X, y)
        self.assertTrue(model.fit_result.is_optimal())
        score = matthews_corrcoef(y, model.predict(X))
        self.assertAlmostEqual(model.fit_result.score(), score, places=12)
        self.assertAlmostEqual(model.score(X, y), score, places=12)
        self.assertLessEqual(model.get_depth(), depth)
        self.assertLessEqual(model.fit_result.tree_nodes(), nodes)
        return model

    def test_all_binary_labelings_against_exhaustive_search(self):
        X = np.array(list(product([0, 1], repeat=3)))
        predictions = all_predictions(X, 2, 3)
        for mask in range(1, 255):
            y = np.array([(mask >> row) & 1 for row in range(8)])
            with self.subTest(mask=mask):
                model = self.fit_and_check(X, y)
                self.assertAlmostEqual(model.fit_result.score(), best_correlation(y, predictions), places=12)

    def test_pruning_caching_and_terminal_solver(self):
        rng = np.random.default_rng(42)
        for case in range(5):
            X = rng.integers(0, 2, size=(12, 4))
            y = np.array([0] * 8 + [1] * 4)
            rng.shuffle(y)
            predictions = all_predictions(X, 3, 4, min_leaf=2)
            expected = best_correlation(y, predictions)
            for terminal, bounds, branch_cache in product([False, True], repeat=3):
                with self.subTest(case=case, terminal=terminal, bounds=bounds, branch_cache=branch_cache):
                    model = self.fit_and_check(
                        X, y, depth=3, nodes=4, min_leaf_node_size=2,
                        use_terminal_solver=terminal, use_upper_bound=bounds,
                        use_lower_bound=bounds, use_similarity_lower_bound=bounds,
                        use_branch_caching=branch_cache, use_dataset_caching=not branch_cache,
                    )
                    self.assertAlmostEqual(model.fit_result.score(), expected, places=12)

    def test_test_totals_negative_scores_and_degenerate_cases(self):
        X = np.array([[0], [0], [1], [1]])
        y = X[:, 0]
        model = self.fit_and_check(X, y, depth=1, nodes=1)
        self.assertEqual(model.score(X, 1 - y), -1)
        # A different class distribution catches accidental use of training totals.
        test_X = np.array([[0]] * 7 + [[1]] * 3)
        test_y = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1])
        self.assertAlmostEqual(model.score(test_X, test_y), matthews_corrcoef(test_y, model.predict(test_X)), places=12)
        for label in [0, 1]:
            labels = np.full(len(test_X), label)
            self.assertEqual(model.score(test_X, labels), 0)
        constant = self.fit_and_check(X, y, depth=0, nodes=0)
        self.assertEqual(constant.fit_result.score(), 0)

    def test_mcc_and_f1_select_different_trees(self):
        # Find a tiny fixture with disjoint F1-optimal and MCC-optimal predictions.
        X = np.array(list(product([0, 1], repeat=3)))
        candidates = all_predictions(X, 1, 1)
        found = False
        for mask in range(1, 255):
            y = np.array([(mask >> row) & 1 for row in range(8)])
            f1 = np.array([f1_score(y, pred) for pred in candidates])
            mcc = np.array([matthews_corrcoef(y, pred) for pred in candidates])
            if np.any(np.isclose(f1, f1.max()) & np.isclose(mcc, mcc.max())):
                continue
            found = True
            model = self.fit_and_check(X, y, depth=1, nodes=1)
            baseline = STreeDClassifier("f1-score", max_depth=1, max_num_nodes=1).fit(X, y)
            self.assertTrue(baseline.fit_result.is_optimal())
            self.assertGreater(model.fit_result.score(), matthews_corrcoef(y, baseline.predict(X)))
            self.assertAlmostEqual(model.fit_result.score(), mcc.max(), places=12)
            break
        self.assertTrue(found, "Expected a fixture separating the objectives")

    def test_invalid_labels_are_rejected_before_cpp(self):
        X = np.array([[0], [0], [1], [1]])
        for y in [[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 2, 0], [-1, 1, -1, 1], [0, 0.5, 1, 1]]:
            with self.subTest(y=y), self.assertRaisesRegex(ValueError, "MCC training"):
                STreeDClassifier("mcc").fit(X, y)
        model = self.fit_and_check(X, np.array([0, 0, 1, 1]))
        for y in [[-1, 0, 1, 1], [0, 1, 2, 1], [0, 0.5, 1, 1]]:
            with self.assertRaisesRegex(ValueError, "MCC scoring"):
                model.score(X, y)

    def test_upper_bound_warning_and_repeated_fit(self):
        with self.assertWarnsRegex(UserWarning, "ignored for mcc"):
            STreeDClassifier("mcc", upper_bound=0.5)
        X = np.array(list(product([0, 1], repeat=3)))
        model = self.fit_and_check(X, X[:, 0])
        model.fit(X, X[:, 1])
        self.assertEqual(model.score(X, X[:, 1]), 1)


if __name__ == "__main__":
    unittest.main()

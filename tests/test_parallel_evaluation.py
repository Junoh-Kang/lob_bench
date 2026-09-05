import contextlib
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import impact
import in_memory_bench
import metrics
import run_bench


def _sequence(offset):
    rows = 256
    event_types = np.resize(np.array([4, 4, 1, 1, 2, 3]), rows)
    messages = pd.DataFrame(
        {
            "time": pd.date_range("2025-03-10", periods=rows, freq="10us"),
            "event_type": event_types,
            "order_id": np.arange(rows),
            "size": np.full(rows, 10),
            "price": 1_000 + offset + np.arange(rows),
            "direction": np.resize(np.array([-1, 1]), rows),
        }
    )
    book = np.zeros((rows, 40), dtype=np.int64)
    book[:, 0] = 1_000 + offset + np.arange(rows)
    book[:, 1] = 100
    book[:, 2] = 990 + offset + np.arange(rows)
    book[:, 3] = 100
    return messages, pd.DataFrame(book)


def _loader():
    sequences = []
    for offset in (0, 100):
        real_messages, real_book = _sequence(offset)
        generated_messages, generated_book = _sequence(offset + 5)
        sequences.append(
            in_memory_bench.InMemorySequence(
                date="2025-03-10",
                real_id=offset,
                m_real=real_messages,
                b_real=real_book,
                m_gen=(generated_messages,),
                b_gen=(generated_book,),
                m_cond=real_messages,
                b_cond=real_book,
            )
        )
    return in_memory_bench.InMemoryLoader(sequences)


def _price_score(messages, book):
    del book
    return messages.price.to_numpy()


def _score_points(scores):
    points = {}
    for score_name, metric_values in scores.items():
        if isinstance(metric_values, list):
            points[score_name] = [value[0] for value in metric_values]
        else:
            points[score_name] = {
                metric_name: value[0]
                for metric_name, value in metric_values.items()
            }
    return points


@contextlib.contextmanager
def _deterministic_bootstrap():
    original = impact.st.bootstrap

    def bootstrap(data, statistic, **_kwargs):
        mean = float(statistic(data[0]))
        return type(
            "BootstrapResult",
            (),
            {"confidence_interval": impact.ConfidenceInterval(mean - 0.25, mean + 0.25)},
        )()

    impact.st.bootstrap = bootstrap
    try:
        yield
    finally:
        impact.st.bootstrap = original


class ParallelEvaluationTest(unittest.TestCase):
    def test_score_only_artifacts_preserve_scores_and_omit_score_frames(self):
        scoring_config = {"price": {"fn": _price_score}}
        conditional_config = {
            "price | price": {
                "eval": {"fn": _price_score},
                "cond": {"fn": _price_score},
            }
        }
        metric_config = {"l1": metrics.l1_by_group}
        with tempfile.TemporaryDirectory() as full_dir, tempfile.TemporaryDirectory() as compact_dir:
            with contextlib.redirect_stdout(None):
                run_bench.run_benchmark_for_loader(
                    _loader(),
                    save_dir=full_dir,
                    stock="TEST",
                    model_name="full",
                    scoring_config=scoring_config,
                    scoring_config_cond=conditional_config,
                    metric_config=metric_config,
                )
                run_bench.run_benchmark_for_loader(
                    _loader(),
                    save_dir=compact_dir,
                    stock="TEST",
                    model_name="compact",
                    scoring_config=scoring_config,
                    scoring_config_cond=conditional_config,
                    metric_config=metric_config,
                    include_score_dfs=False,
                )

            for mode in ("uncond", "cond", "div"):
                full_path = next((Path(full_dir) / "scores").glob(f"scores_{mode}_*.pkl"))
                compact_path = next((Path(compact_dir) / "scores").glob(f"scores_{mode}_*.pkl"))
                full_scores, full_dfs = run_bench.load_results(full_path)
                compact_scores, compact_dfs = run_bench.load_results(compact_path)
                self.assertEqual(_score_points(compact_scores), _score_points(full_scores))
                self.assertIsNotNone(full_dfs)
                self.assertIsNone(compact_dfs)

    def test_parallel_score_modes_and_impact_preserve_point_estimates(self):
        scoring_config = {"price": {"fn": _price_score}}
        conditional_config = {
            "price | price": {
                "eval": {"fn": _price_score},
                "cond": {"fn": _price_score},
            }
        }
        metric_config = {"l1": metrics.l1_by_group}

        with tempfile.TemporaryDirectory() as sequential_dir, tempfile.TemporaryDirectory() as parallel_dir:
            with _deterministic_bootstrap(), contextlib.redirect_stdout(None):
                run_bench.run_benchmark_for_loader(
                    _loader(),
                    save_dir=sequential_dir,
                    stock="TEST",
                    model_name="sequential",
                    divergence_horizon=100,
                    scoring_config=scoring_config,
                    scoring_config_cond=conditional_config,
                    metric_config=metric_config,
                    workers=1,
                    bootstrap_workers=1,
                )
                run_bench.run_benchmark_for_loader(
                    _loader(),
                    save_dir=parallel_dir,
                    stock="TEST",
                    model_name="parallel",
                    divergence_horizon=100,
                    scoring_config=scoring_config,
                    scoring_config_cond=conditional_config,
                    metric_config=metric_config,
                    workers=2,
                    bootstrap_workers=2,
                )

                sequential_impact = impact.impact_compare(
                    _loader(),
                    ticker="TEST",
                    model="sequential",
                    save_dir=sequential_dir,
                    workers=1,
                )["Saved_Tuple"]
                parallel_impact = impact.impact_compare(
                    _loader(),
                    ticker="TEST",
                    model="parallel",
                    save_dir=parallel_dir,
                    workers=2,
                )["Saved_Tuple"]

            for mode in ("uncond", "cond", "div"):
                sequential_path = next(
                    (Path(sequential_dir) / "scores").glob(f"scores_{mode}_*.pkl")
                )
                parallel_path = next(
                    (Path(parallel_dir) / "scores").glob(f"scores_{mode}_*.pkl")
                )
                sequential_scores, _ = run_bench.load_results(sequential_path)
                parallel_scores, _ = run_bench.load_results(parallel_path)
                self.assertEqual(_score_points(parallel_scores), _score_points(sequential_scores))

        for parallel_curves, sequential_curves in zip(
            parallel_impact[1:3], sequential_impact[1:3]
        ):
            for parallel_curve, sequential_curve in zip(
                parallel_curves, sequential_curves
            ):
                np.testing.assert_array_equal(parallel_curve, sequential_curve)
        np.testing.assert_array_equal(parallel_impact[5], sequential_impact[5])


if __name__ == "__main__":
    unittest.main()

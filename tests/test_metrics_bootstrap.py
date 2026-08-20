import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import metrics


def _score_frame():
    return pd.DataFrame(
        {
            "type": ["real"] * 6 + ["generated"] * 5,
            "score": [1.0, 1.5, 2.0, 3.5, 5.0, 8.0, 1.2, 1.8, 2.5, 4.0, 7.0],
            "group": [0, 0, 1, 1, 2, 3, 0, 1, 1, 2, 4],
        }
    )


def _reference_bootstrap(score_df, loss_fn, n_bootstrap, ci_alpha, whole_data_loss, rng):
    losses = [whole_data_loss]
    real_indices = score_df.loc[score_df["type"] == "real"].index
    generated_indices = score_df.loc[score_df["type"] == "generated"].index
    n_real = len(real_indices)
    n_generated = len(generated_indices)
    real = score_df.loc[real_indices]
    generated = score_df.loc[generated_indices]
    real_indices = rng.integers(0, n_real, size=(n_bootstrap, n_real))
    generated_indices = rng.integers(
        0, n_generated, size=(n_bootstrap, n_generated)
    )
    for real_index, generated_index in zip(real_indices, generated_indices):
        sample = pd.concat(
            [real.iloc[real_index], generated.iloc[generated_index]], axis=0
        )
        losses.append(loss_fn(sample))
    losses = np.asarray(losses)
    ci = np.percentile(losses, q=[ci_alpha / 2 * 100, 100 - ci_alpha / 2 * 100])
    return ci, losses


def _reference_wasserstein(score_df, n_bootstrap, rng):
    frame = score_df.copy()
    frame.score = (frame.score - frame.score.mean()) / frame.score.std()

    def loss(sample):
        return stats.wasserstein_distance(
            sample.loc[sample["type"] == "real", "score"],
            sample.loc[sample["type"] == "generated", "score"],
        )

    whole = loss(frame)
    ci, losses = _reference_bootstrap(frame, loss, n_bootstrap, 0.01, whole, rng)
    return whole, ci, losses


def _reference_l1_by_group(score_df, n_bootstrap, rng):
    def loss(sample):
        counts = sample.groupby(["type", "group"]).count()
        counts = pd.merge(
            counts.loc["real"], counts.loc["generated"], on="group", how="outer"
        ).fillna(0)
        counts = counts[["score_x", "score_y"]]
        counts /= counts.sum(axis=0)
        return (counts.score_x - counts.score_y).abs().sum() / 2.0

    whole = loss(score_df)
    ci, losses = _reference_bootstrap(
        score_df, loss, n_bootstrap, 0.01, whole, rng
    )
    return whole, ci, losses


class MetricsBootstrapTest(unittest.TestCase):
    def test_wasserstein_bootstrap_matches_dataframe_reference_exactly(self):
        score_df = _score_frame()
        expected = _reference_wasserstein(
            score_df, n_bootstrap=11, rng=np.random.default_rng(20260820)
        )
        actual = metrics.wasserstein(
            score_df, n_bootstrap=11, rng_np=np.random.default_rng(20260820)
        )

        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])

    def test_l1_bootstrap_matches_dataframe_reference_exactly(self):
        score_df = _score_frame()
        expected = _reference_l1_by_group(
            score_df, n_bootstrap=11, rng=np.random.default_rng(20260820)
        )
        actual = metrics.l1_by_group(
            score_df, n_bootstrap=11, rng_np=np.random.default_rng(20260820)
        )

        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])

    def test_l1_bootstrap_omits_bins_absent_from_a_resample(self):
        score_df = pd.DataFrame(
            {
                "type": ["real"] * 17 + ["generated"] * 17,
                "score": [1.0] * 34,
                "group": list(range(17)) * 2,
            }
        )
        expected = _reference_l1_by_group(
            score_df, n_bootstrap=1, rng=np.random.default_rng(0)
        )
        actual = metrics.l1_by_group(
            score_df, n_bootstrap=1, rng_np=np.random.default_rng(0)
        )

        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])

    def test_l1_bootstrap_preserves_duplicate_index_resampling(self):
        score_df = pd.DataFrame(
            {
                "type": ["real"] * 3 + ["generated"] * 3,
                "score": [1.0] * 6,
                "group": [0, 1, 2, 0, 2, 3],
            },
            index=[0, 0, 1, 2, 2, 3],
        )
        expected = _reference_l1_by_group(
            score_df, n_bootstrap=3, rng=np.random.default_rng(0)
        )
        actual = metrics.l1_by_group(
            score_df, n_bootstrap=3, rng_np=np.random.default_rng(0)
        )

        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])

    def test_wasserstein_bootstrap_preserves_duplicate_index_resampling(self):
        score_df = pd.DataFrame(
            {
                "type": ["real"] * 3 + ["generated"] * 3,
                "score": [1.0, 2.0, 4.0, 1.5, 3.0, 5.0],
                "group": [0, 1, 2, 0, 2, 3],
            },
            index=[0, 0, 1, 2, 2, 3],
        )
        expected = _reference_wasserstein(
            score_df, n_bootstrap=3, rng=np.random.default_rng(0)
        )
        actual = metrics.wasserstein(
            score_df, n_bootstrap=3, rng_np=np.random.default_rng(0)
        )

        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])

    def test_l1_bootstrap_uses_score_count_not_row_count(self):
        score_df = pd.DataFrame(
            {
                "type": ["real", "real", "generated", "generated"],
                "score": [1.0, np.nan, 1.0, 1.0],
                "group": [0, 1, 0, 1],
            }
        )
        expected = _reference_l1_by_group(
            score_df, n_bootstrap=3, rng=np.random.default_rng(4)
        )
        actual = metrics.l1_by_group(
            score_df, n_bootstrap=3, rng_np=np.random.default_rng(4)
        )

        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])


if __name__ == "__main__":
    unittest.main()

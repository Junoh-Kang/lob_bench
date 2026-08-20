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
    real = score_df.loc[score_df["type"] == "real"]
    generated = score_df.loc[score_df["type"] == "generated"]
    real_indices = rng.integers(0, len(real), size=(n_bootstrap, len(real)))
    generated_indices = rng.integers(
        0, len(generated), size=(n_bootstrap, len(generated))
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


if __name__ == "__main__":
    unittest.main()

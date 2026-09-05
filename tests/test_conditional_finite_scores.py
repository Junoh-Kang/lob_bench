import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scoring


class ConditionalFiniteScoresTest(unittest.TestCase):
    def test_nonfinite_condition_and_score_rows_are_excluded_before_binning(self):
        condition_score = object()
        evaluation_score = object()
        scores = {
            condition_score: (
                (np.array([1.0, np.nan, 2.0]),),
                (np.array([1.0, np.inf, 3.0]),),
            ),
            evaluation_score: (
                (np.array([10.0, 20.0, np.nan]),),
                (np.array([10.0, 30.0, 40.0]),),
            ),
        }

        with patch.object(
            scoring.partitioning,
            "score_real_gen",
            side_effect=lambda _loader, score_fn: scores[score_fn],
        ):
            score_df = scoring.score_data_cond(
                loader=object(),
                scoring_fn=evaluation_score,
                scoring_fn_cond=condition_score,
                score_kwargs={"discrete": True},
                score_cond_kwargs={"discrete": True},
            )

        self.assertTrue(np.isfinite(score_df.score).all())
        self.assertTrue(np.isfinite(score_df.score_cond).all())
        self.assertEqual(score_df.attrs["nonfinite"]["real"], {
            "total": 3,
            "condition_nonfinite": 1,
            "score_nonfinite": 1,
            "kept": 1,
            "dropped_rate": 2 / 3,
        })
        self.assertEqual(score_df.attrs["nonfinite"]["generated"], {
            "total": 3,
            "condition_nonfinite": 1,
            "score_nonfinite": 0,
            "kept": 2,
            "dropped_rate": 1 / 3,
        })


if __name__ == "__main__":
    unittest.main()

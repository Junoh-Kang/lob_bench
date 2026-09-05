import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import eval
import scoring


class SpreadEmptyBookTest(unittest.TestCase):
    def test_empty_touch_quote_has_undefined_spread(self):
        messages = pd.DataFrame({"time": pd.date_range(
            "2025-03-10", periods=4, freq="s"
        )})
        book = pd.DataFrame({
            0: [10_100, -1, 10_200, -1],
            1: [1, 1, 1, 1],
            2: [10_000, 9_990, -1, -1],
        })

        actual = eval.spread(messages, book)

        self.assertEqual(actual.iloc[0], 100)
        self.assertTrue(actual.iloc[1:].isna().all())
        self.assertEqual(actual.name, "spread")

    def test_unconditional_scoring_excludes_undefined_spread(self):
        scores = (
            (np.array([100.0, np.nan]),),
            (np.array([110.0, np.nan]),),
        )

        with patch.object(
            scoring.partitioning,
            "score_real_gen",
            return_value=scores,
        ):
            score_df = scoring.score_data(
                loader=object(), scoring_fn=object(), discrete=True
            )

        self.assertTrue(np.isfinite(score_df.score).all())
        self.assertEqual(score_df.attrs["nonfinite"]["real"]["dropped_rate"], 0.5)
        self.assertEqual(
            score_df.attrs["nonfinite"]["generated"]["dropped_rate"], 0.5
        )


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import data_loading
import partitioning
import scoring


def _sequence():
    rows = 4
    messages = pd.DataFrame(
        {
            "time": pd.date_range("2025-03-10", periods=rows, freq="1s"),
            "event_type": [1, 2, 3, 4],
            "order_id": np.arange(rows),
            "size": np.full(rows, 10),
            "price": np.arange(1_000, 1_000 + rows),
            "direction": [1, -1, 1, -1],
        }
    )
    book = pd.DataFrame(np.arange(rows * 40).reshape(rows, 40))
    generated_messages = messages.assign(price=lambda frame: frame.price + 10)
    generated_book = book + 10
    conditioning_messages = messages.assign(price=lambda frame: frame.price - 10)
    conditioning_book = book - 10
    return data_loading.Lobster_Sequence(
        date="2025-03-10",
        real_id=1,
        m_real=messages,
        b_real=book,
        num_gen_series=(1,),
        m_gen=(generated_messages,),
        b_gen=(generated_book,),
        m_cond=conditioning_messages,
        b_cond=conditioning_book,
    )


def _price_score(messages, book):
    del book
    return messages.price.values


class ScoreCacheTest(unittest.TestCase):
    def test_cache_preserves_scores_and_partitions_each_horizon_once(self):
        loader = (_sequence(),)
        expected_real, expected_generated = partitioning.score_real_gen(
            loader, _price_score
        )
        expected_horizon = partitioning.score_gen(
            tuple(
                partitioning.get_subseqs(sequence, subseq_len=2)
                for sequence in loader
            ),
            _price_score,
        )
        cache = scoring.ScoreCache(loader)
        real_calls = 0
        generated_calls = 0
        partition_calls = 0
        original_real = partitioning.score_real
        original_generated = partitioning.score_gen
        original_partition = partitioning.get_subseqs

        def score_real(*args, **kwargs):
            nonlocal real_calls
            real_calls += 1
            return original_real(*args, **kwargs)

        def score_generated(*args, **kwargs):
            nonlocal generated_calls
            generated_calls += 1
            return original_generated(*args, **kwargs)

        def get_subseqs(*args, **kwargs):
            nonlocal partition_calls
            partition_calls += 1
            return original_partition(*args, **kwargs)

        with patch.object(partitioning, "score_real", score_real), patch.object(
            partitioning, "score_gen", score_generated
        ), patch.object(partitioning, "get_subseqs", get_subseqs):
            actual_real, actual_generated = cache.score_real_gen(_price_score)
            repeat_real, repeat_generated = cache.score_real_gen(_price_score)
            actual_horizon = cache.generated_horizon_scores(_price_score, 2)
            repeat_horizon = cache.generated_horizon_scores(_price_score, 2)

        np.testing.assert_array_equal(actual_real[0], expected_real[0])
        np.testing.assert_array_equal(actual_generated[0], expected_generated[0])
        np.testing.assert_array_equal(repeat_real[0], expected_real[0])
        np.testing.assert_array_equal(repeat_generated[0], expected_generated[0])
        np.testing.assert_array_equal(
            actual_horizon[0][0][0], expected_horizon[0][0][0]
        )
        np.testing.assert_array_equal(
            repeat_horizon[0][0][1], expected_horizon[0][0][1]
        )
        self.assertEqual(real_calls, 1)
        self.assertEqual(generated_calls, 2)
        self.assertEqual(partition_calls, 1)


if __name__ == "__main__":
    unittest.main()

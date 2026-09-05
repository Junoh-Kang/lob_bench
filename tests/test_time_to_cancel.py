import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import eval
import in_memory_bench
import run_bench


def _raw_message(order_id, event_type, time_s, reference_time_s):
    message = np.zeros(14, dtype=np.int64)
    message[0] = order_id
    message[1] = event_type
    message[2] = 1
    message[3] = 1_000
    message[5] = 10
    message[8] = time_s
    message[12] = reference_time_s
    return message


class TimeToCancelTest(unittest.TestCase):
    def test_all_cancellations_use_the_original_add_timestamp(self):
        raw_messages = np.stack(
            (
                _raw_message(10, 2, 12, 3),
                _raw_message(10, 3, 13, 3),
                _raw_message(10, 4, 14, 3),
                _raw_message(20, 2, 8, 5),
            )
        )
        messages = in_memory_bench.message_frame_from_raw(
            raw_messages, date="2025-03-10"
        )

        actual = eval.time_to_cancel_all(messages).dt.total_seconds().to_numpy()

        np.testing.assert_array_equal(actual, np.array([9.0, 10.0, 3.0]))
        self.assertIn("log_time_to_cancel", run_bench.DEFAULT_SCORING_CONFIG)
        self.assertIn("log_time_to_cancel_all", run_bench.RAW_REFERENCE_SCORING_CONFIG)


if __name__ == "__main__":
    unittest.main()

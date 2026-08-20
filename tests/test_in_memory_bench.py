import csv
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import in_memory_bench
import data_loading


def _messages(offset=0):
    return np.array(
        [
            [100 + offset, 1, 1, 1_000 + offset, 0, 4, 0, 0, 34_200, 123],
            [101 + offset, 4, 0, 1_001 + offset, 0, 5, 0, 0, 34_201, 456],
        ],
        dtype=np.int64,
    )


def _book(offset=0):
    return np.arange(offset, offset + 80, dtype=np.int64).reshape(2, 40)


def _entry(real_id, gen_id, offset=0):
    return {
        "date": "2025-03-10",
        "real_id": real_id,
        "gen_id": gen_id,
        "real_messages": _messages(0),
        "real_orderbook": _book(0),
        "generated_messages": _messages(offset),
        "generated_orderbook": _book(offset),
        "conditioning_messages": _messages(20),
        "conditioning_orderbook": _book(20),
    }


def _payload(*entries):
    return {
        "schema_version": 1,
        "stock": "GOOG",
        "model_name": "s5",
        "time_period": "2025-03-10",
        "sequences": entries,
    }


class InMemoryBenchTest(unittest.TestCase):
    def test_message_frame_matches_the_former_lobster_csv_loader(self):
        frame = in_memory_bench.message_frame_from_raw(
            _messages(), date="2025-03-10"
        )

        with tempfile.TemporaryDirectory() as directory:
            message_path = Path(directory) / "messages.csv"
            with message_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                for message in _messages():
                    writer.writerow(
                        [
                            f"{int(message[8])}.{int(message[9]):09d}",
                            int(message[1]),
                            int(message[0]),
                            int(message[5]),
                            int(message[3]),
                            (int(message[2]) * 2) - 1,
                        ]
                    )
            expected = data_loading.add_date_to_time(
                data_loading.load_message_df(str(message_path)), "2025-03-10"
            )
        pd.testing.assert_frame_equal(frame, expected)

    def test_book_frame_matches_the_former_lobster_csv_loader(self):
        frame = in_memory_bench.book_frame_from_raw(_book())

        with tempfile.TemporaryDirectory() as directory:
            book_path = Path(directory) / "book.csv"
            pd.DataFrame(_book()).to_csv(book_path, header=False, index=False)
            expected = data_loading.load_book_df(str(book_path))
        pd.testing.assert_frame_equal(frame, expected)

    def test_loader_groups_generated_samples_and_keeps_file_loader_order(self):
        loader = in_memory_bench.loader_from_payload(
            _payload(
                _entry(2, 2, 2),
                _entry(10, 0, 10),
                _entry(2, 10, 10),
            )
        )

        self.assertEqual([sequence.real_id for sequence in loader], [10, 2])
        self.assertEqual(len(loader[1].m_gen), 2)
        self.assertEqual(loader[1].m_gen[0].order_id.iloc[0], 110)
        self.assertEqual(loader[1].m_gen[1].order_id.iloc[0], 102)
        self.assertEqual(loader[0].b_real.shape, (2, 40))
        self.assertEqual(loader[0].m_cond.shape, (2, 6))

    def test_run_payload_reuses_one_loader_for_both_metric_families(self):
        observed = {}
        original_run = in_memory_bench.run_bench.run_benchmark_for_loader
        original_impact = in_memory_bench.impact.impact_compare

        def run_benchmark_for_loader(loader, **kwargs):
            observed["distributional"] = (loader, kwargs)

        def impact_compare(loader, **kwargs):
            observed["impact"] = (loader, kwargs)

        in_memory_bench.run_bench.run_benchmark_for_loader = run_benchmark_for_loader
        in_memory_bench.impact.impact_compare = impact_compare
        try:
            loader = in_memory_bench.run_payload(
                _payload(_entry(1, 0)),
                save_dir="/tmp/lob-bench-results",
                stock="GOOG",
                model_name="s5",
                time_period="2025-03-10",
                divergence_horizon=100,
            )
        finally:
            in_memory_bench.run_bench.run_benchmark_for_loader = original_run
            in_memory_bench.impact.impact_compare = original_impact

        self.assertIs(observed["distributional"][0], loader)
        self.assertIs(observed["impact"][0], loader)
        self.assertEqual(observed["distributional"][1]["divergence_horizon"], 100)
        self.assertEqual(
            observed["impact"][1]["save_dir"],
            "/tmp/lob-bench-results/impact/GOOG/s5",
        )

    def test_shared_loader_runner_preserves_all_three_score_modes(self):
        class Sequence:
            def __init__(self):
                self.materialized = 0

            def materialize(self):
                self.materialized += 1

        sequence = Sequence()
        loader = [sequence]
        calls = []
        original = in_memory_bench.run_bench.scoring.run_benchmark

        def score(*args, **kwargs):
            calls.append((args, kwargs))
            return {}, {}, None

        in_memory_bench.run_bench.scoring.run_benchmark = score
        try:
            with tempfile.TemporaryDirectory() as directory:
                in_memory_bench.run_bench.run_benchmark_for_loader(
                    loader,
                    save_dir=directory,
                    stock="GOOG",
                    model_name="s5",
                    divergence_horizon=100,
                )
                score_files = tuple((Path(directory) / "scores").glob("*.pkl"))
        finally:
            in_memory_bench.run_bench.scoring.run_benchmark = original

        self.assertEqual(sequence.materialized, 1)
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[2][1]["divergence"])
        self.assertEqual(calls[2][1]["divergence_horizon"], 100)
        self.assertEqual(len(score_files), 3)


if __name__ == "__main__":
    unittest.main()

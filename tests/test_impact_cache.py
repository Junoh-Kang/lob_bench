import contextlib
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import impact


class _Sequence:
    def __init__(self, real_messages, real_book, generated_messages, generated_book):
        self.date = "2025-03-10"
        self.real_id = 1
        self.m_real = real_messages
        self.b_real = real_book
        self.m_gen = (generated_messages,)
        self.b_gen = (generated_book,)


class _Loader:
    def __init__(self, sequences):
        self._sequences = tuple(sequences)

    def __len__(self):
        return len(self._sequences)

    def __getitem__(self, index):
        return self._sequences[index]


def _sequence(offset):
    rows = 512
    event_types = np.resize(np.array([4, 4, 1, 1, 2, 3]), rows)
    ask_price = 1_000 + offset
    bid_price = 990 + offset
    ask_volume = 100
    bid_volume = 100
    books = []
    for index, event_type in enumerate(event_types):
        if index % 2 == 0:
            ask_price += 1
            bid_price += 1
        else:
            ask_volume += 1
            bid_volume += 1
        books.append((ask_price, ask_volume, bid_price, bid_volume))
    messages = pd.DataFrame(
        {
            "time": pd.date_range("2025-03-10", periods=rows, freq="10us"),
            "event_type": event_types,
            "order_id": np.arange(rows),
            "size": np.full(rows, 10),
            "price": [book[0] for book in books],
            "direction": np.resize(np.array([-1, 1]), rows),
        }
    )
    return messages, pd.DataFrame(books)


def _two_window_loader():
    sequences = []
    for offset in (0, 100):
        real_messages, real_book = _sequence(offset)
        generated_messages, generated_book = _sequence(offset + 5)
        sequences.append(
            _Sequence(real_messages, real_book, generated_messages, generated_book)
        )
    return _Loader(sequences)


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


@contextlib.contextmanager
def _quiet_progress():
    original = impact.tqdm.tqdm
    impact.tqdm.tqdm = lambda iterable, **_kwargs: iterable
    try:
        yield
    finally:
        impact.tqdm.tqdm = original


def _reference_impact_compare(loader):
    lags = (10 ** np.arange(0.3, 2.4, step=0.1)).astype(int)
    events = ("MO_0", "MO_1", "LO_0", "LO_1", "CA_0", "CA_1")
    real_curves = []
    generated_curves = []
    real_intervals = []
    generated_intervals = []
    differences = []
    for event in events:
        real, generated = zip(*[
            impact._apply_to_Sequences(
                impact._response_func,
                0.99,
                8192,
                (event, lag, 100),
                loader,
            )
            for lag in lags
        ])
        real_means, real_ci = zip(*real)
        generated_means, generated_ci = zip(*generated)
        real_curves.append(np.asarray(real_means))
        generated_curves.append(np.asarray(generated_means))
        real_intervals.append(real_ci)
        generated_intervals.append(generated_ci)
        differences.append(
            np.sum(np.abs(np.asarray(real_means) - np.asarray(generated_means)))
        )
    return (
        lags,
        real_curves,
        generated_curves,
        real_intervals,
        generated_intervals,
        np.asarray(differences),
    )


def _assert_saved_tuples_equal(test_case, expected, actual):
    np.testing.assert_array_equal(actual[0], expected[0])
    for actual_curves, expected_curves in zip(actual[1:3], expected[1:3]):
        for actual_curve, expected_curve in zip(actual_curves, expected_curves):
            np.testing.assert_allclose(actual_curve, expected_curve, equal_nan=True)
    for actual_intervals, expected_intervals in zip(actual[3:5], expected[3:5]):
        for actual_event, expected_event in zip(actual_intervals, expected_intervals):
            for actual_ci, expected_ci in zip(actual_event, expected_event):
                test_case.assertEqual(actual_ci, expected_ci)
    np.testing.assert_allclose(actual[5], expected[5], equal_nan=True)


class ImpactCacheTest(unittest.TestCase):
    def test_parallel_impact_preserves_point_estimates(self):
        with _deterministic_bootstrap(), _quiet_progress(), contextlib.redirect_stdout(None):
            with tempfile.TemporaryDirectory() as sequential_output, tempfile.TemporaryDirectory() as parallel_output:
                sequential = impact.impact_compare(
                    _two_window_loader(),
                    ticker="TEST",
                    model="sequential",
                    save_dir=sequential_output,
                    workers=1,
                )["Saved_Tuple"]
                parallel = impact.impact_compare(
                    _two_window_loader(),
                    ticker="TEST",
                    model="parallel",
                    save_dir=parallel_output,
                    workers=2,
                )["Saved_Tuple"]

        np.testing.assert_array_equal(parallel[0], sequential[0])
        for parallel_curves, sequential_curves in zip(
            parallel[1:3], sequential[1:3]
        ):
            for parallel_curve, sequential_curve in zip(
                parallel_curves, sequential_curves
            ):
                np.testing.assert_array_equal(parallel_curve, sequential_curve)
        np.testing.assert_array_equal(parallel[5], sequential[5])

    def test_cached_impact_matches_reference_and_classifies_each_sequence_once(self):
        loader = _two_window_loader()
        with _deterministic_bootstrap(), _quiet_progress():
            with contextlib.redirect_stdout(None):
                expected = _reference_impact_compare(loader)
            calls = 0
            original = impact._classify_event_types

            def count_calls(frame):
                nonlocal calls
                calls += 1
                return original(frame)

            impact._classify_event_types = count_calls
            try:
                with tempfile.TemporaryDirectory() as output, contextlib.redirect_stdout(None):
                    result = impact.impact_compare(
                        loader,
                        ticker="TEST",
                        model="cached",
                        save_dir=output,
                    )
            finally:
                impact._classify_event_types = original

        _assert_saved_tuples_equal(self, expected, result["Saved_Tuple"])
        self.assertEqual(calls, 4)

    def test_cached_impact_is_ten_times_faster_on_two_windows(self):
        with _deterministic_bootstrap(), _quiet_progress(), contextlib.redirect_stdout(None):
            started = time.perf_counter()
            _reference_impact_compare(_two_window_loader())
            reference_seconds = time.perf_counter() - started

            started = time.perf_counter()
            with tempfile.TemporaryDirectory() as output:
                impact.impact_compare(
                    _two_window_loader(),
                    ticker="TEST",
                    model="cached",
                    save_dir=output,
                )
            cached_seconds = time.perf_counter() - started

        self.assertLess(
            cached_seconds,
            reference_seconds / 10,
            msg=(
                f"expected at least 10x speedup, got "
                f"{reference_seconds / cached_seconds:.1f}x "
                f"({reference_seconds:.3f}s -> {cached_seconds:.3f}s)"
            ),
        )


if __name__ == "__main__":
    unittest.main()

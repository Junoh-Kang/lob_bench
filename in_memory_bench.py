"""Run LOB-Bench from rollout arrays received on standard input."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import pickle
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

import impact
import run_bench


PAYLOAD_SCHEMA_VERSION = 1
MESSAGE_COLUMNS = ("time", "event_type", "order_id", "size", "price", "direction")


@dataclass
class InMemorySequence:
    """The subset of ``Lobster_Sequence`` consumed by scoring and impact code."""

    date: str
    real_id: int
    m_real: pd.DataFrame
    b_real: pd.DataFrame
    m_gen: tuple[pd.DataFrame, ...]
    b_gen: tuple[pd.DataFrame, ...]
    m_cond: pd.DataFrame
    b_cond: pd.DataFrame

    def materialize(self) -> None:
        """Keep the file-loader protocol while frames are already in memory."""


class InMemoryLoader:
    def __init__(self, sequences: Iterable[InMemorySequence]) -> None:
        self._sequences = tuple(sequences)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, index: int) -> InMemorySequence:
        return self._sequences[index]

    def __iter__(self):
        return iter(self._sequences)


def _array(value: object, *, label: str, min_columns: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] < min_columns:
        raise ValueError(
            f"{label} must have shape (N, >={min_columns}), got {array.shape}"
        )
    if len(array) == 0:
        raise ValueError(f"{label} must not be empty")
    return array


def message_frame_from_raw(raw_messages: object, *, date: str) -> pd.DataFrame:
    """Recreate the LOBSTER message frame formerly read back from a CSV file."""
    messages = _array(raw_messages, label="raw_messages", min_columns=10)
    seconds = messages[:, 8].astype(np.int64)
    nanoseconds = messages[:, 9].astype(np.int64)
    timestamps = (
        pd.Timestamp(date)
        + pd.to_timedelta(seconds, unit="s")
        + pd.to_timedelta(nanoseconds, unit="ns")
    )
    return pd.DataFrame(
        {
            "time": timestamps,
            "event_type": messages[:, 1].astype(np.int32),
            "order_id": messages[:, 0].astype(np.int32),
            "size": messages[:, 5].astype(np.int32),
            "price": messages[:, 3].astype(np.int32),
            "direction": (messages[:, 2].astype(np.int32) * 2) - 1,
        },
        columns=MESSAGE_COLUMNS,
    )


def book_frame_from_raw(raw_book: object) -> pd.DataFrame:
    """Recreate the top-ten L2 book frame formerly read back from a CSV file."""
    book = _array(raw_book, label="raw_orderbook", min_columns=40)
    return pd.DataFrame(book[:, :40].astype(np.int64, copy=False))


def loader_from_payload(payload: object) -> InMemoryLoader:
    """Convert the portable parent-process payload into score-ready sequences."""
    if not isinstance(payload, dict):
        raise ValueError("in-memory LOB-Bench payload must be a dictionary")
    if payload.get("schema_version") != PAYLOAD_SCHEMA_VERSION:
        raise ValueError("unsupported in-memory LOB-Bench payload schema")
    entries = payload.get("sequences")
    if not isinstance(entries, tuple) or not entries:
        raise ValueError("in-memory LOB-Bench payload needs at least one sequence")

    grouped: dict[int, list[dict[str, Any]]] = {}
    coordinates: set[tuple[int, int]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each in-memory LOB-Bench sequence must be a dictionary")
        date = entry.get("date")
        real_id = entry.get("real_id")
        gen_id = entry.get("gen_id")
        if not isinstance(date, str):
            raise ValueError("each in-memory LOB-Bench sequence needs a date")
        if not isinstance(real_id, int) or isinstance(real_id, bool) or real_id < 0:
            raise ValueError("each in-memory LOB-Bench sequence needs a real_id")
        if not isinstance(gen_id, int) or isinstance(gen_id, bool) or gen_id < 0:
            raise ValueError("each in-memory LOB-Bench sequence needs a gen_id")
        coordinate = (real_id, gen_id)
        if coordinate in coordinates:
            raise ValueError("each in-memory LOB-Bench coordinate must be unique")
        coordinates.add(coordinate)
        grouped.setdefault(real_id, []).append(entry)

    sequences: list[InMemorySequence] = []
    for real_id, generated_entries in sorted(
        grouped.items(), key=lambda item: (str(item[1][0]["date"]), str(item[0]))
    ):
        first = generated_entries[0]
        date = first["date"]
        real_messages = _array(
            first.get("real_messages"), label="real_messages", min_columns=10
        )
        real_book = _array(
            first.get("real_orderbook"), label="real_orderbook", min_columns=40
        )
        conditioning_messages = _array(
            first.get("conditioning_messages"),
            label="conditioning_messages",
            min_columns=10,
        )
        conditioning_book = _array(
            first.get("conditioning_orderbook"),
            label="conditioning_orderbook",
            min_columns=40,
        )
        if len(real_messages) != len(real_book):
            raise ValueError("real message/book lengths differ")
        if len(conditioning_messages) != len(conditioning_book):
            raise ValueError("conditioning message/book lengths differ")

        generated_messages: list[pd.DataFrame] = []
        generated_books: list[pd.DataFrame] = []
        for entry in sorted(generated_entries, key=lambda value: str(value["gen_id"])):
            if entry["date"] != date:
                raise ValueError("one real_id cannot span multiple dates")
            for key, expected in (
                ("real_messages", real_messages),
                ("real_orderbook", real_book),
                ("conditioning_messages", conditioning_messages),
                ("conditioning_orderbook", conditioning_book),
            ):
                if not np.array_equal(_array(entry.get(key), label=key, min_columns=10 if "messages" in key else 40), expected):
                    raise ValueError(
                        "one real_id must share real and conditioning data"
                    )
            generated_message = _array(
                entry.get("generated_messages"),
                label="generated_messages",
                min_columns=10,
            )
            generated_book = _array(
                entry.get("generated_orderbook"),
                label="generated_orderbook",
                min_columns=40,
            )
            if len(generated_message) != len(generated_book):
                raise ValueError("generated message/book lengths differ")
            generated_messages.append(
                message_frame_from_raw(generated_message, date=date)
            )
            generated_books.append(book_frame_from_raw(generated_book))

        sequences.append(
            InMemorySequence(
                date=date,
                real_id=real_id,
                m_real=message_frame_from_raw(real_messages, date=date),
                b_real=book_frame_from_raw(real_book),
                m_gen=tuple(generated_messages),
                b_gen=tuple(generated_books),
                m_cond=message_frame_from_raw(conditioning_messages, date=date),
                b_cond=book_frame_from_raw(conditioning_book),
            )
        )
    return InMemoryLoader(sequences)


def run_payload(
    payload: object,
    *,
    save_dir: str,
    stock: str,
    model_name: str,
    time_period: str,
    divergence_horizon: int,
) -> InMemoryLoader:
    """Run distributional and impact metrics over one shared in-memory loader."""
    if not isinstance(payload, dict):
        raise ValueError("in-memory LOB-Bench payload must be a dictionary")
    for key, value in (("stock", stock), ("model_name", model_name), ("time_period", time_period)):
        if payload.get(key) != value:
            raise ValueError(f"payload {key} does not match command")
    loader = loader_from_payload(payload)
    run_bench.run_benchmark_for_loader(
        loader,
        save_dir=save_dir,
        stock=stock,
        model_name=model_name,
        divergence_horizon=divergence_horizon,
    )
    impact.impact_compare(
        loader,
        ticker=stock,
        model=model_name,
        save_dir=f"{save_dir}/impact/{stock}/{model_name}",
    )
    return loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", required=True)
    parser.add_argument("--time_period", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--divergence_horizon", type=int, default=100)
    args = parser.parse_args()
    if args.divergence_horizon <= 0:
        raise ValueError("divergence_horizon must be positive")

    payload = pickle.load(sys.stdin.buffer)
    loader = run_payload(
        payload,
        save_dir=args.save_dir,
        stock=args.stock,
        model_name=args.model_name,
        time_period=args.time_period,
        divergence_horizon=args.divergence_horizon,
    )
    print(json.dumps({"status": "complete", "sequence_count": len(loader)}))


if __name__ == "__main__":
    main()

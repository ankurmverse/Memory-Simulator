"""Seeded synthetic workload generation and trace-file serialization."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

TraceWorkload = Literal[
    "sequential", "looping", "zipf", "hot_set_noise", "phase_change"
]
TRACE_FORMAT_VERSION = 1
_WORKLOADS = {"sequential", "looping", "zipf", "hot_set_noise", "phase_change"}


@dataclass(frozen=True)
class SyntheticTrace:
    """A reference sequence and the parameters needed to reproduce it."""

    pages: tuple[int, ...]
    workload: TraceWorkload
    seed: int
    parameters: Mapping[str, int | float]


def generate_trace(
    workload: TraceWorkload,
    length: int,
    seed: int = 0,
    working_set_size: int = 16,
    *,
    hot_set_size: int | None = None,
    noise_probability: float = 0.1,
) -> SyntheticTrace:
    """Generate a deterministic reference trace for a named workload."""
    if not isinstance(workload, str) or workload not in _WORKLOADS:
        raise ValueError(f"unknown workload {workload!r}; choose from {sorted(_WORKLOADS)}")
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise ValueError("length must be a non-negative integer")
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise ValueError("seed must be an integer")
    if (
        not isinstance(working_set_size, int)
        or isinstance(working_set_size, bool)
        or working_set_size <= 0
    ):
        raise ValueError("working_set_size must be a positive integer")
    if hot_set_size is None:
        hot_set_size = max(1, working_set_size // 4)
    if (
        not isinstance(hot_set_size, int)
        or isinstance(hot_set_size, bool)
        or not 1 <= hot_set_size <= working_set_size
    ):
        raise ValueError("hot_set_size must be between 1 and working_set_size")
    if (
        not isinstance(noise_probability, (int, float))
        or isinstance(noise_probability, bool)
        or not 0 <= noise_probability <= 1
    ):
        raise ValueError("noise_probability must be a finite number from 0 to 1")

    rng = random.Random(seed)
    pages: list[int]
    parameters: dict[str, int | float] = {"working_set_size": working_set_size}

    if workload == "sequential":
        pages = list(range(length))
    elif workload == "looping":
        pages = [index % working_set_size for index in range(length)]
    elif workload == "zipf":
        weights = [1 / rank**1.2 for rank in range(1, working_set_size + 1)]
        pages = rng.choices(range(working_set_size), weights=weights, k=length)
        parameters["exponent"] = 1.2
    elif workload == "hot_set_noise":
        parameters["hot_set_size"] = hot_set_size
        parameters["noise_probability"] = float(noise_probability)
        pages = [
            rng.randrange(hot_set_size)
            if rng.random() >= noise_probability
            else rng.randrange(working_set_size)
            for _ in range(length)
        ]
    else:
        parameters["phase_boundary"] = length // 2
        pages = [
            rng.randrange(
                0 if index < length // 2 else working_set_size,
                working_set_size if index < length // 2 else 2 * working_set_size,
            )
            for index in range(length)
        ]

    return SyntheticTrace(tuple(pages), workload, seed, parameters)


def write_trace(trace: SyntheticTrace, csv_path: str | Path) -> tuple[Path, Path]:
    """Write a trace CSV and JSON metadata sidecar; return both paths."""
    path = Path(csv_path)
    metadata_path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("reference_index", "page"))
        writer.writerows(enumerate(trace.pages))

    metadata = {
        "format_version": TRACE_FORMAT_VERSION,
        "workload": trace.workload,
        "seed": trace.seed,
        "length": len(trace.pages),
        "parameters": dict(trace.parameters),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path, metadata_path


def read_trace(csv_path: str | Path) -> SyntheticTrace:
    """Load a trace CSV and its adjacent JSON metadata, validating both."""
    path = Path(csv_path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("trace metadata must be a JSON object")
    version = metadata.get("format_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != TRACE_FORMAT_VERSION
    ):
        raise ValueError(
            f"unsupported trace format version: {version!r}"
        )

    workload = metadata.get("workload")
    if not isinstance(workload, str) or workload not in _WORKLOADS:
        raise ValueError(f"unknown workload in metadata: {workload!r}")
    seed = metadata.get("seed")
    length = metadata.get("length")
    parameters = metadata.get("parameters")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("trace metadata seed must be an integer")
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise ValueError("trace metadata length must be a non-negative integer")
    if not isinstance(parameters, dict):
        raise ValueError("trace metadata parameters must be an object")
    for key, value in parameters.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise ValueError("trace metadata parameters must contain numeric values")

    pages: list[int] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["reference_index", "page"]:
            raise ValueError("trace CSV must have reference_index,page columns")
        for expected_index, row in enumerate(reader):
            if None in row:
                raise ValueError(f"unexpected extra value in trace row {expected_index + 2}")
            try:
                index = int(row["reference_index"])
                page = int(row["page"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid integer value in trace row {expected_index + 2}") from error
            if index != expected_index:
                raise ValueError(
                    f"trace reference indexes must be contiguous from 0; "
                    f"expected {expected_index}, got {index}"
                )
            pages.append(page)

    if len(pages) != length:
        raise ValueError(
            f"trace metadata declares {length} references, but CSV contains {len(pages)}"
        )
    return SyntheticTrace(tuple(pages), workload, seed, parameters)

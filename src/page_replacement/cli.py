"""Command-line tools for generating traces and comparing replacement policies."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

from page_replacement.policies import CLOCKPolicy, FIFOPolicy, LRUPolicy
from page_replacement.simulator import SimulationResult, simulate
from page_replacement.traces import (
    SyntheticTrace,
    generate_trace,
    read_trace,
    write_trace,
)

_POLICIES = {
    "fifo": FIFOPolicy,
    "lru": LRUPolicy,
    "clock": CLOCKPolicy,
}
_METRIC_FIELDS = (
    "workload",
    "seed",
    "policy",
    "frames",
    "references",
    "hits",
    "faults",
    "evictions",
    "fault_rate",
    "hit_rate",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="page-replacement",
        description="Generate reference traces and compare page replacement policies.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate and save a synthetic trace")
    generate.add_argument(
        "--workload",
        choices=("sequential", "looping", "zipf", "hot_set_noise", "phase_change"),
        required=True,
    )
    generate.add_argument("--length", type=int, default=1_000)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--working-set-size", type=int, default=16)
    generate.add_argument("--hot-set-size", type=int)
    generate.add_argument("--noise-probability", type=float, default=0.1)
    generate.add_argument("--output", type=Path, required=True, help="output CSV path")

    compare = commands.add_parser("compare", help="compare policies on a trace")
    source = compare.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path, help="trace CSV with adjacent JSON metadata")
    source.add_argument(
        "--workload",
        choices=("sequential", "looping", "zipf", "hot_set_noise", "phase_change"),
        help="generate a trace for this workload",
    )
    compare.add_argument("--length", type=int, default=1_000)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--working-set-size", type=int, default=16)
    compare.add_argument("--hot-set-size", type=int)
    compare.add_argument("--noise-probability", type=float, default=0.1)
    compare.add_argument("--frames", type=int, default=4)
    compare.add_argument(
        "--policies", nargs="+", choices=(*_POLICIES, "all"), default=["all"]
    )
    compare.add_argument("--csv", type=Path, help="write metrics to this CSV path")
    compare.add_argument("--plot", type=Path, help="save a fault-rate comparison plot")

    train = commands.add_parser("train", help="train and save the learned policy model")
    train.add_argument(
        "--workloads",
        nargs="+",
        choices=("sequential", "looping", "zipf", "hot_set_noise", "phase_change"),
        default=("looping", "zipf", "hot_set_noise", "phase_change"),
    )
    train.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2, 3))
    train.add_argument("--length", type=int, default=2_000)
    train.add_argument("--working-set-size", type=int, default=16)
    train.add_argument("--frames", type=int, default=4)
    train.add_argument("--trees", type=int, default=100)
    train.add_argument("--model", type=Path, default=Path("models/reuse-distance.joblib"))

    evaluate = commands.add_parser(
        "evaluate", help="evaluate the learned policy on a separate trace"
    )
    eval_source = evaluate.add_mutually_exclusive_group(required=True)
    eval_source.add_argument(
        "--trace", type=Path, help="held-out trace CSV with adjacent JSON metadata"
    )
    eval_source.add_argument(
        "--workload",
        choices=("sequential", "looping", "zipf", "hot_set_noise", "phase_change"),
        help="generate a held-out workload trace",
    )
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--length", type=int, default=2_000)
    evaluate.add_argument("--seed", type=int, default=100)
    evaluate.add_argument("--working-set-size", type=int, default=16)
    evaluate.add_argument("--hot-set-size", type=int)
    evaluate.add_argument("--noise-probability", type=float, default=0.1)
    evaluate.add_argument("--frames", type=int, default=4)
    evaluate.add_argument("--csv", type=Path, help="write metrics to this CSV path")
    evaluate.add_argument("--plot", type=Path, help="save a fault-rate comparison plot")
    return parser


def _generate(args: argparse.Namespace) -> SyntheticTrace:
    return generate_trace(
        args.workload,
        length=args.length,
        seed=args.seed,
        working_set_size=args.working_set_size,
        hot_set_size=args.hot_set_size,
        noise_probability=args.noise_probability,
    )


def _load_compare_trace(args: argparse.Namespace) -> SyntheticTrace:
    if args.trace is not None:
        return read_trace(args.trace)
    return _generate(args)


def _metric_row(
    trace: SyntheticTrace, policy_name: str, frames: int, result: SimulationResult
) -> dict[str, str | int | float]:
    hit_rate = result.hits / result.references if result.references else 0.0
    return {
        "workload": trace.workload,
        "seed": trace.seed,
        "policy": policy_name,
        "frames": frames,
        "references": result.references,
        "hits": result.hits,
        "faults": result.faults,
        "evictions": result.evictions,
        "fault_rate": result.fault_rate,
        "hit_rate": hit_rate,
    }


def _write_metrics(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ValueError(
            "plotting requires Matplotlib; install it with "
            "`python -m pip install -e '.[plots]'`"
        ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    policies = [str(row["policy"]).upper() for row in rows]
    fault_rates = [float(row["fault_rate"]) for row in rows]
    figure, axis = plt.subplots()
    axis.bar(policies, fault_rates, color="#4472C4")
    axis.set(title="Page fault rate by replacement policy", ylabel="Fault rate")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _print_metrics(rows: list[dict[str, str | int | float]]) -> None:
    print(
        f"{'Policy':<8} {'Frames':>6} {'Refs':>8} {'Hits':>8} {'Faults':>8} "
        f"{'Evictions':>10} {'Fault rate':>12} {'Hit rate':>10}"
    )
    for row in rows:
        print(
            f"{str(row['policy']).upper():<8} {row['frames']:>6} "
            f"{row['references']:>8} {row['hits']:>8} {row['faults']:>8} "
            f"{row['evictions']:>10} {float(row['fault_rate']):>11.2%} "
            f"{float(row['hit_rate']):>9.2%}"
        )


def _compare(args: argparse.Namespace) -> int:
    trace = _load_compare_trace(args)
    policy_names = tuple(_POLICIES) if "all" in args.policies else tuple(args.policies)
    rows = [
        _metric_row(
            trace,
            name,
            args.frames,
            simulate(trace.pages, args.frames, _POLICIES[name](), record_events=False),
        )
        for name in policy_names
    ]
    _print_metrics(rows)
    if args.csv is not None:
        _write_metrics(args.csv, rows)
        print(f"Metrics CSV: {args.csv}")
    if args.plot is not None:
        _write_plot(args.plot, rows)
        print(f"Fault-rate plot: {args.plot}")
    return 0


def _train(args: argparse.Namespace) -> int:
    from page_replacement.model import train_model

    traces = [
        generate_trace(
            workload,
            length=args.length,
            seed=seed,
            working_set_size=args.working_set_size,
        )
        for workload in args.workloads
        for seed in args.seeds
    ]
    summary = train_model(
        traces,
        args.frames,
        args.model,
        n_estimators=args.trees,
    )
    print(f"Training samples: {summary.sample_count}")
    print(f"Eviction decisions: {summary.eviction_decisions}")
    print(f"Never-used labels: {summary.never_reused_fraction:.1%}")
    print(f"Model: {args.model}")
    print("Feature importance:")
    for name, importance in summary.feature_importances:
        print(f"  {name}: {importance:.4f}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from page_replacement.model import evaluate_learned_policy

    trace = (
        read_trace(args.trace)
        if args.trace is not None
        else _generate(args)
    )
    results = evaluate_learned_policy(trace.pages, args.frames, args.model)
    rows = [
        _metric_row(trace, name, args.frames, result)
        for name, result in results.items()
    ]
    _print_metrics(rows)
    if args.csv is not None:
        _write_metrics(args.csv, rows)
        print(f"Metrics CSV: {args.csv}")
    if args.plot is not None:
        _write_plot(args.plot, rows)
        print(f"Fault-rate plot: {args.plot}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            trace = _generate(args)
            csv_path, metadata_path = write_trace(trace, args.output)
            print(f"Generated {len(trace.pages)} references: {csv_path}")
            print(f"Metadata: {metadata_path}")
            return 0
        if args.command == "compare":
            return _compare(args)
        if args.command == "train":
            return _train(args)
        return _evaluate(args)
    except (ImportError, OSError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    sys.exit(main())

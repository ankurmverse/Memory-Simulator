import csv

import pytest

from page_replacement.cli import main
from page_replacement.traces import generate_trace, write_trace


def test_generate_command_writes_csv_and_metadata(tmp_path, capsys):
    output = tmp_path / "nested" / "loop.csv"

    assert main(
        [
            "generate",
            "--workload",
            "looping",
            "--length",
            "8",
            "--working-set-size",
            "3",
            "--seed",
            "9",
            "--output",
            str(output),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "Generated 8 references" in captured.out
    assert output.exists()
    assert output.with_suffix(".json").exists()


def test_compare_command_reports_correct_metrics_and_writes_csv(tmp_path, capsys):
    output = tmp_path / "reports" / "metrics.csv"

    assert main(
        [
            "compare",
            "--workload",
            "looping",
            "--length",
            "6",
            "--working-set-size",
            "3",
            "--frames",
            "2",
            "--policies",
            "fifo",
            "lru",
            "--csv",
            str(output),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "FIFO" in captured.out
    assert "LRU" in captured.out
    assert output.exists()
    with output.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["policy"] for row in rows] == ["fifo", "lru"]
    assert [int(row["faults"]) for row in rows] == [6, 6]
    assert [float(row["fault_rate"]) for row in rows] == [1.0, 1.0]


def test_compare_can_load_existing_trace_csv(tmp_path):
    trace = generate_trace("looping", 9, seed=3, working_set_size=3)
    trace_path, _ = write_trace(trace, tmp_path / "trace.csv")
    metrics_path = tmp_path / "out.csv"

    assert main(
        [
            "compare",
            "--trace",
            str(trace_path),
            "--frames",
            "3",
            "--policies",
            "fifo",
            "--csv",
            str(metrics_path),
        ]
    ) == 0
    with metrics_path.open(newline="") as file:
        row = next(csv.DictReader(file))
    assert row["workload"] == "looping"
    assert row["seed"] == "3"
    assert row["faults"] == "3"
    assert row["hits"] == "6"


def test_compare_all_policies_disables_event_recording(capsys):
    assert main(
        [
            "compare",
            "--workload",
            "sequential",
            "--length",
            "4",
            "--frames",
            "2",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert all(name in output for name in ("FIFO", "LRU", "CLOCK"))
    assert "4" in output


def test_invalid_arguments_are_reported_by_argparse(capsys):
    with pytest.raises(SystemExit) as error:
        main(["compare", "--workload", "looping", "--frames", "0"])

    assert error.value.code == 2
    assert "positive integer" in capsys.readouterr().err


def test_train_and_held_out_evaluate_commands(tmp_path, capsys):
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    model_path = tmp_path / "reuse.joblib"
    metrics_path = tmp_path / "held-out.csv"

    assert main(
        [
            "train",
            "--workloads",
            "looping",
            "zipf",
            "--seeds",
            "0",
            "1",
            "--length",
            "120",
            "--working-set-size",
            "7",
            "--frames",
            "3",
            "--trees",
            "8",
            "--model",
            str(model_path),
        ]
    ) == 0
    train_output = capsys.readouterr().out
    assert "Training samples:" in train_output
    assert "Feature importance:" in train_output
    assert model_path.exists()

    assert main(
        [
            "evaluate",
            "--model",
            str(model_path),
            "--workload",
            "hot_set_noise",
            "--seed",
            "100",
            "--length",
            "120",
            "--working-set-size",
            "7",
            "--frames",
            "3",
            "--csv",
            str(metrics_path),
        ]
    ) == 0
    evaluate_output = capsys.readouterr().out
    assert all(name in evaluate_output for name in ("FIFO", "LRU", "CLOCK", "ML"))
    with metrics_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["policy"] for row in rows] == ["fifo", "lru", "clock", "ml"]
    assert {row["seed"] for row in rows} == {"100"}

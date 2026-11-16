import json

import pytest

from page_replacement.traces import (
    TRACE_FORMAT_VERSION,
    generate_trace,
    read_trace,
    write_trace,
)


@pytest.mark.parametrize(
    "workload",
    ("sequential", "looping", "zipf", "hot_set_noise", "phase_change"),
)
def test_workloads_are_reproducible(workload):
    first = generate_trace(workload, length=100, seed=23, working_set_size=8)
    second = generate_trace(workload, length=100, seed=23, working_set_size=8)

    assert first == second
    assert len(first.pages) == 100


def test_sequential_and_looping_have_expected_reference_patterns():
    sequential = generate_trace("sequential", length=5, seed=1)
    looping = generate_trace("looping", length=8, seed=1, working_set_size=3)

    assert sequential.pages == (0, 1, 2, 3, 4)
    assert looping.pages == (0, 1, 2, 0, 1, 2, 0, 1)


def test_hot_set_noise_respects_hot_set_and_working_set():
    trace = generate_trace(
        "hot_set_noise",
        length=500,
        seed=7,
        working_set_size=20,
        hot_set_size=4,
        noise_probability=0,
    )

    assert set(trace.pages) <= {0, 1, 2, 3}
    assert trace.parameters["noise_probability"] == 0.0


def test_phase_change_switches_to_disjoint_page_region():
    trace = generate_trace("phase_change", length=10, seed=11, working_set_size=5)

    assert set(trace.pages[:5]) <= set(range(5))
    assert set(trace.pages[5:]) <= set(range(5, 10))
    assert set(trace.pages[:5]).isdisjoint(trace.pages[5:])


def test_empty_traces_are_supported():
    trace = generate_trace("zipf", length=0, seed=2)

    assert trace.pages == ()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"workload": "other"}, "unknown workload"),
        ({"length": -1}, "length"),
        ({"length": True}, "length"),
        ({"seed": False}, "seed"),
        ({"working_set_size": 0}, "working_set_size"),
        ({"hot_set_size": 0}, "hot_set_size"),
        ({"noise_probability": 1.1}, "noise_probability"),
    ),
)
def test_invalid_generation_arguments_are_rejected(kwargs, message):
    arguments = {
        "workload": "looping",
        "length": 8,
        "seed": 0,
        "working_set_size": 4,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        generate_trace(**arguments)


def test_trace_csv_and_metadata_round_trip(tmp_path):
    expected = generate_trace(
        "hot_set_noise",
        length=32,
        seed=19,
        working_set_size=10,
        hot_set_size=3,
        noise_probability=0.2,
    )
    csv_path, metadata_path = write_trace(expected, tmp_path / "nested" / "sample.csv")

    assert csv_path.exists()
    assert metadata_path.exists()
    assert read_trace(csv_path) == expected
    assert json.loads(metadata_path.read_text()) == {
        "format_version": TRACE_FORMAT_VERSION,
        "length": 32,
        "parameters": {
            "hot_set_size": 3,
            "noise_probability": 0.2,
            "working_set_size": 10,
        },
        "seed": 19,
        "workload": "hot_set_noise",
    }


def test_trace_reader_rejects_noncontiguous_indexes(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("reference_index,page\n0,1\n2,2\n")
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "format_version": TRACE_FORMAT_VERSION,
                "workload": "looping",
                "seed": 0,
                "length": 2,
                "parameters": {"working_set_size": 2},
            }
        )
    )

    with pytest.raises(ValueError, match="contiguous"):
        read_trace(path)


def test_trace_reader_rejects_length_mismatch(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("reference_index,page\n0,1\n")
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "format_version": TRACE_FORMAT_VERSION,
                "workload": "looping",
                "seed": 0,
                "length": 2,
                "parameters": {"working_set_size": 2},
            }
        )
    )

    with pytest.raises(ValueError, match="declares 2 references"):
        read_trace(path)


def test_trace_reader_rejects_boolean_format_version(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("reference_index,page\n")
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "format_version": True,
                "workload": "looping",
                "seed": 0,
                "length": 0,
                "parameters": {"working_set_size": 2},
            }
        )
    )

    with pytest.raises(ValueError, match="format version"):
        read_trace(path)

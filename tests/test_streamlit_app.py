import pytest

streamlit = pytest.importorskip("streamlit")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

from pathlib import Path


APP_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "page_replacement"
    / "streamlit_app.py"
)


def test_dashboard_renders_comparison_and_walkthrough(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()

    assert not app.exception
    assert app.title[0].value == "Page Replacement Lab"
    assert any("Compare the policies" in item.value for item in app.subheader)
    assert any("Memory-capacity what-if" in item.value for item in app.subheader)
    assert any(
        "Algorithm and system trade-offs" in item.value for item in app.subheader
    )
    assert any("Walk through memory" in item.value for item in app.subheader)
    assert [item.label for item in app.metric[:3]] == [
        "FIFO page faults",
        "LRU page faults",
        "CLOCK page faults",
    ]


def test_walkthrough_can_step_forward_and_select_a_policy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    app.selectbox(key="workload").select("looping").run()
    app.selectbox(key="walkthrough_policy").select("LRU").run()
    app.button[2].click().run()

    assert not app.exception
    assert any("Looping workload" in item.value for item in app.caption)
    assert app.slider(key="walkthrough_step").value == 1
    assert any("Reference 2 of" in item.value for item in app.markdown)


def test_custom_trace_drives_frame_walkthrough(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    app.text_input(key="custom_trace_text").set_value("1, 2, 3, 1, 4, 2").run()

    assert not app.exception
    assert any("Custom trace" in item.value for item in app.caption)
    assert app.slider(key="walkthrough_step").max == 5
    assert [item.label for item in app.metric[:3]] == [
        "FIFO page faults",
        "LRU page faults",
        "CLOCK page faults",
    ]


def test_dashboard_shows_learned_eviction_explanation(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from page_replacement.model import train_model
    from page_replacement.traces import generate_trace

    monkeypatch.chdir(tmp_path)
    model_path = tmp_path / "reuse.joblib"
    train_model(
        [
            generate_trace("looping", 100, seed=2, working_set_size=8),
            generate_trace("zipf", 100, seed=3, working_set_size=8),
        ],
        frame_capacity=4,
        model_path=model_path,
        n_estimators=8,
    )
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    app.text_input[1].set_value(str(model_path)).run()

    assert not app.exception
    assert any("Why did the learned policy" in item.value for item in app.subheader)
    assert [item.label for item in app.metric[:4]] == [
        "FIFO page faults",
        "LRU page faults",
        "CLOCK page faults",
        "ML page faults",
    ]
    assert any("Actual next use (future)" in str(frame.value) for frame in app.dataframe)


def test_dashboard_can_train_a_model_from_the_expander(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    monkeypatch.chdir(tmp_path)
    model_path = tmp_path / "trained.joblib"
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.text_input[0].set_value("0")
    app.text_input[1].set_value(str(model_path))
    app.multiselect(key="training_workloads").set_value(["looping"])
    app.number_input[0].set_value(50)
    app.number_input[1].set_value(10)
    next(button for button in app.button if button.label == "Train model").click().run()

    assert not app.exception
    assert model_path.is_file()
    assert any("Model trained on" in item.value for item in app.success)
    assert any(
        "Training feature importance" in item.value for item in app.subheader
    )

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


def test_dashboard_renders_comparison_and_walkthrough():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()

    assert not app.exception
    assert app.title[0].value == "Page Replacement Lab"
    assert any("Policy comparison" in item.value for item in app.subheader)
    assert any("Step-by-step memory state" in item.value for item in app.subheader)
    assert len(app.metric) == 3


def test_dashboard_can_change_workload_and_reference():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    app.selectbox(key="workload").select("looping").run()
    app.slider(key="reference_select").set_value(5).run()

    assert not app.exception
    assert "Looping workload" in app.caption[0].value
    assert app.slider(key="reference_select").value == 5


def test_dashboard_shows_learned_eviction_explanation(tmp_path):
    pytest.importorskip("sklearn")
    from page_replacement.model import train_model
    from page_replacement.traces import generate_trace

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
    assert len(app.metric) == 4
    assert any("Actual next use (future)" in str(frame.value) for frame in app.dataframe)


def test_dashboard_can_train_a_model_from_the_expander(tmp_path):
    pytest.importorskip("sklearn")
    model_path = tmp_path / "trained.joblib"
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.text_input[0].set_value("0")
    app.text_input[1].set_value(str(model_path))
    app.multiselect(key="training_workloads").set_value(["looping"])
    app.number_input[0].set_value(50)
    app.number_input[1].set_value(10)
    app.button[0].click().run()

    assert not app.exception
    assert model_path.is_file()
    assert any("Model trained on" in item.value for item in app.success)
    assert any(
        "Training feature importance" in item.value for item in app.subheader
    )

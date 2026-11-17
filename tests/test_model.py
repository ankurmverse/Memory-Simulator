import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from page_replacement.features import build_candidate_samples
from page_replacement.model import (
    FEATURE_NAMES,
    LearnedReplacementPolicy,
    evaluate_learned_policy,
    load_model,
    train_model,
)
from page_replacement.policies import FIFOPolicy
from page_replacement.simulator import simulate
from page_replacement.traces import generate_trace


class AccessCountRegressor:
    feature_importances_ = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def predict(self, values):
        return [row[0] for row in values]


def _training_traces():
    return [
        generate_trace("looping", 160, seed=3, working_set_size=7),
        generate_trace("zipf", 160, seed=5, working_set_size=7),
        generate_trace("hot_set_noise", 160, seed=8, working_set_size=7),
    ]


def test_learned_policy_uses_max_predicted_next_use_candidate():
    result = simulate(
        [1, 2, 1, 3], 2, LearnedReplacementPolicy(AccessCountRegressor())
    )

    assert result.events[3].evicted_page == 1


def test_train_save_reload_and_evaluate_on_held_out_trace(tmp_path):
    model_path = tmp_path / "nested" / "reuse.joblib"
    summary = train_model(
        _training_traces(), frame_capacity=3, model_path=model_path, n_estimators=12
    )

    assert model_path.exists()
    assert summary.sample_count > 0
    assert summary.eviction_decisions > 0
    assert 0 <= summary.never_reused_fraction <= 1
    assert {name for name, _ in summary.feature_importances} == set(FEATURE_NAMES)

    policy = load_model(model_path)
    held_out = generate_trace("hot_set_noise", 160, seed=101, working_set_size=7)
    prediction = simulate(held_out.pages, 3, policy, record_events=False)
    results = evaluate_learned_policy(held_out.pages, 3, model_path)

    assert prediction.references == 160
    assert tuple(results) == ("fifo", "lru", "clock", "ml")
    assert all(result.references == 160 for result in results.values())


def test_training_rejects_empty_data_and_invalid_settings(tmp_path):
    trace = generate_trace("looping", 2, seed=1, working_set_size=2)
    with pytest.raises(ValueError, match="no eviction candidates"):
        train_model([trace], frame_capacity=4, model_path=tmp_path / "empty.joblib")
    with pytest.raises(ValueError, match="positive integer"):
        train_model(_training_traces(), frame_capacity=0, model_path=tmp_path / "bad.joblib")
    with pytest.raises(ValueError, match="positive integer"):
        train_model(
            _training_traces(),
            frame_capacity=3,
            model_path=tmp_path / "bad.joblib",
            n_estimators=0,
        )


def test_loading_incompatible_model_artifact_is_reported(tmp_path):
    import joblib

    path = tmp_path / "wrong.joblib"
    joblib.dump({"format_version": 99}, path)

    with pytest.raises(ValueError, match="format version"):
        load_model(path)


def test_fifo_training_sample_label_matches_target_distance():
    samples = build_candidate_samples([1, 2, 1, 3, 2], 2, FIFOPolicy())
    sample = next(
        row
        for row in samples
        if row.reference_index == 3 and row.candidate_page == 2
    )

    assert sample.next_use_distance == 1

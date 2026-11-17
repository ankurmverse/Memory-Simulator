"""Interactive Streamlit demonstration for page replacement policies."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import streamlit as st

from page_replacement.model import (
    PolicyDecision,
    TrainingSummary,
    load_model,
    train_model,
)
from page_replacement.policies import CLOCKPolicy, FIFOPolicy, LRUPolicy
from page_replacement.simulator import SimulationResult, simulate
from page_replacement.traces import SyntheticTrace, generate_trace

_WORKLOADS = (
    "sequential",
    "looping",
    "zipf",
    "hot_set_noise",
    "phase_change",
)
_BASELINE_POLICIES = {
    "FIFO": FIFOPolicy,
    "LRU": LRUPolicy,
    "CLOCK": CLOCKPolicy,
}


def _show_training_summary(summary: TrainingSummary) -> None:
    st.success(
        f"Model trained on {summary.sample_count:,} candidate examples from "
        f"{summary.eviction_decisions:,} eviction decisions."
    )
    st.caption(
        f"{summary.never_reused_fraction:.1%} of training candidates were not "
        "referenced again in their trace."
    )
    st.subheader("Training feature importance")
    st.bar_chart(
        {
            "Importance": {
                name: importance for name, importance in summary.feature_importances
            }
        }
    )
    st.caption(
        "Importance measures how much the fitted trees used a feature; it does not "
        "show causation or guarantee better page-fault performance."
    )


def _render_candidate_decision(
    trace: SyntheticTrace,
    decisions: list[PolicyDecision],
    result: SimulationResult,
) -> None:
    if not decisions:
        st.info("This trace did not produce any learned-policy eviction decisions.")
        return

    decision_index = st.selectbox(
        "Eviction decision reference",
        options=list(range(len(decisions))),
        format_func=lambda index: (
            f"Reference {decisions[index].reference_index}: "
            f"request page {decisions[index].incoming_page}"
        ),
        key="ml_decision_select",
    )
    decision = decisions[decision_index]
    future_positions: dict[int, list[int]] = {}
    for index, page in enumerate(trace.pages):
        future_positions.setdefault(page, []).append(index)

    rows = []
    for candidate in decision.candidates:
        occurrences = future_positions.get(candidate.page, [])
        next_position = bisect_right(occurrences, decision.reference_index)
        actual_distance = (
            occurrences[next_position] - decision.reference_index
            if next_position < len(occurrences)
            else None
        )
        rows.append(
            {
                "Page": candidate.page,
                "Accesses so far": candidate.features.access_count,
                "Refs since last access": (
                    candidate.features.references_since_last_access
                ),
                "Mean reuse distance": round(
                    candidate.features.mean_reuse_distance, 2
                ),
                "Time in memory": candidate.features.residency_age,
                "Predicted next use": round(
                    candidate.predicted_next_use_distance, 2
                ),
                "Actual next use (future)": (
                    actual_distance if actual_distance is not None else "Never"
                ),
                "Selected victim": candidate.page == decision.victim_page,
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "The model sees only the historical feature columns and chooses the page "
        "with the largest predicted distance. The actual future column is shown "
        "for explanation only and is never passed to the online policy."
    )
    if decision.reference_index < len(result.events):
        event = result.events[decision.reference_index]
        st.write(
            f"At reference {event.index}, page **{event.page}** faults; "
            f"the learned policy evicts **{decision.victim_page}**."
        )


def main() -> None:
    st.set_page_config(page_title="Page Replacement Lab", page_icon="📚", layout="wide")
    st.title("Page Replacement Lab")
    st.markdown(
        "Explore how FIFO, LRU, CLOCK, and a learned policy respond to the same "
        "page-reference workload."
    )

    with st.sidebar:
        st.header("Workload")
        workload = st.selectbox("Pattern", _WORKLOADS, key="workload")
        length = st.slider("References", min_value=20, max_value=2_000, value=300, step=20)
        seed = st.number_input("Workload seed", min_value=0, value=42, step=1)
        working_set_size = st.slider(
            "Working-set size", min_value=2, max_value=64, value=12
        )
        frame_capacity = st.slider(
            "Memory frames", min_value=1, max_value=32, value=4
        )
        hot_set_size = min(
            st.slider(
                "Hot-set size",
                min_value=1,
                max_value=working_set_size,
                value=min(4, working_set_size),
                disabled=workload != "hot_set_noise",
            ),
            working_set_size,
        )
        noise_probability = st.slider(
            "Hot-set noise probability",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.05,
            disabled=workload != "hot_set_noise",
        )
        model_path = Path(
            st.text_input("Model file", value="models/reuse-distance-v1.joblib")
        )

    trace = generate_trace(
        workload,
        length=length,
        seed=int(seed),
        working_set_size=working_set_size,
        hot_set_size=hot_set_size,
        noise_probability=noise_probability,
    )
    st.subheader("Reference string")
    st.code(" → ".join(map(str, trace.pages[:60])) + (" → …" if length > 60 else ""))
    st.caption(
        f"{trace.workload.replace('_', ' ').title()} workload · seed {trace.seed} · "
        f"{len(trace.pages):,} references · {frame_capacity} frames"
    )

    results: dict[str, SimulationResult] = {
        name: simulate(
            trace.pages,
            frame_capacity,
            policy_factory(),
            record_events=True,
        )
        for name, policy_factory in _BASELINE_POLICIES.items()
    }

    if model_path.is_file():
        try:
            learned_policy = load_model(model_path)
            ml_result = simulate(
                trace.pages, frame_capacity, learned_policy, record_events=True
            )
        except (ImportError, OSError, ValueError, RuntimeError) as error:
            st.error(f"Could not load or run the learned model: {error}")
        else:
            results["ML"] = ml_result
            st.session_state["ml_decisions"] = learned_policy.decisions
    else:
        st.info(
            "Train a model below to add the learned policy to this comparison. "
            "Training runs only when you press the button."
        )

    st.subheader("Policy comparison")
    metric_columns = st.columns(len(results))
    for column, (name, result) in zip(metric_columns, results.items()):
        column.metric(f"{name} page faults", f"{result.faults:,}")
        column.caption(
            f"{result.fault_rate:.1%} fault rate · "
            f"{result.hits:,} hits · {result.evictions:,} evictions"
        )
    st.bar_chart(
        {
            "Policy": list(results),
            "Page faults": [result.faults for result in results.values()],
        },
        x="Policy",
        y="Page faults",
    )

    st.subheader("Step-by-step memory state")
    selected_reference = st.slider(
        "Reference to inspect",
        min_value=0,
        max_value=len(trace.pages) - 1,
        value=min(10, len(trace.pages) - 1),
        key="reference_select",
    )
    event_rows = []
    for name, result in results.items():
        event = result.events[selected_reference]
        event_rows.append(
            {
                "Policy": name,
                "Page requested": event.page,
                "Outcome": "Hit" if event.hit else "Page fault",
                "Page evicted": (
                    event.evicted_page if event.evicted_page is not None else "—"
                ),
                "Frames after access": ", ".join(map(str, event.resident_pages)),
            }
        )
    st.dataframe(event_rows, hide_index=True, width="stretch")

    if "ML" in results:
        st.subheader("Why did the learned policy evict that page?")
        decisions = st.session_state.get("ml_decisions", [])
        _render_candidate_decision(trace, decisions, results["ML"])

    with st.expander("Train or retrain the learned policy"):
        st.write(
            "Training uses separately seeded synthetic traces. Keep the workload "
            "seed above out of the training seed list when demonstrating held-out "
            "performance."
        )
        training_workloads = st.multiselect(
            "Training workloads",
            _WORKLOADS,
            default=("looping", "zipf", "hot_set_noise", "phase_change"),
            key="training_workloads",
        )
        training_seeds = st.text_input(
            "Training seeds (comma-separated)",
            value="0,1,2,3",
            key="training_seeds",
        )
        training_length = st.number_input(
            "References per training trace",
            min_value=50,
            max_value=20_000,
            value=1_000,
            step=50,
        )
        tree_count = st.number_input(
            "Random-forest trees", min_value=10, max_value=500, value=100, step=10
        )
        train_clicked = st.button("Train model", type="primary")
        if train_clicked:
            if not training_workloads:
                st.error("Select at least one training workload.")
            else:
                try:
                    seeds = [int(value.strip()) for value in training_seeds.split(",")]
                    if not seeds:
                        raise ValueError("enter at least one training seed")
                    training_traces = [
                        generate_trace(
                            training_workload,
                            length=int(training_length),
                            seed=training_seed,
                            working_set_size=working_set_size,
                            hot_set_size=hot_set_size,
                            noise_probability=noise_probability,
                        )
                        for training_workload in training_workloads
                        for training_seed in seeds
                    ]
                    with st.spinner("Training the random-forest model…"):
                        summary = train_model(
                            training_traces,
                            frame_capacity,
                            model_path,
                            n_estimators=int(tree_count),
                        )
                except (ImportError, OSError, ValueError, RuntimeError) as error:
                    st.error(f"Model training failed: {error}")
                else:
                    st.session_state["training_summary"] = summary
                    st.success(f"Saved trained model to {model_path}")
                    st.rerun()

        if "training_summary" in st.session_state:
            _show_training_summary(st.session_state["training_summary"])
        if model_path.is_file():
            st.caption(
                f"Model artifact found at `{model_path}`. The current trace seed is "
                "used for evaluation, not training unless you explicitly add it above."
            )

    st.caption(
        "Educational simulation only. The learned policy is a heuristic and may "
        "perform better or worse than classic policies depending on the workload."
    )


if __name__ == "__main__":
    main()

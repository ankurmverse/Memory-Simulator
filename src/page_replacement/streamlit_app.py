"""Interactive Streamlit demonstration for page replacement policies."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
import re

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
    trace_pages: tuple[int, ...],
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
    for index, page in enumerate(trace_pages):
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


def _step_back() -> None:
    st.session_state["walkthrough_step"] = max(
        0, st.session_state["walkthrough_step"] - 1
    )


def _step_forward(total_references: int) -> None:
    st.session_state["walkthrough_step"] = min(
        total_references - 1, st.session_state["walkthrough_step"] + 1
    )


def _toggle_playback() -> None:
    st.session_state["walkthrough_playing"] = not st.session_state.get(
        "walkthrough_playing", False
    )


@st.fragment(run_every="1s")
def _render_walkthrough(
    trace_pages: tuple[int, ...],
    trace_label: str,
    results: dict[str, SimulationResult],
    trace_key: tuple[int, ...],
) -> None:
    if st.session_state.get("walkthrough_trace_key") != trace_key:
        st.session_state["walkthrough_trace_key"] = trace_key
        st.session_state["walkthrough_step"] = 0
        st.session_state["walkthrough_playing"] = False

    st.subheader("Walk through memory, one request at a time")
    st.caption(
        "1. Choose a trace · 2. Set frame capacity · 3. Select a policy · "
        "4. Step through its decisions."
    )
    names = tuple(results)
    selected_policy = st.selectbox(
        "Watch policy",
        names,
        key="walkthrough_policy",
        help="Choose one replacement rule and inspect its frames after each request.",
    )
    result = results[selected_policy]
    total_references = len(trace_pages)
    step = st.session_state["walkthrough_step"]
    if st.session_state["walkthrough_playing"]:
        if step >= total_references - 1:
            st.session_state["walkthrough_playing"] = False
        else:
            step += 1
            st.session_state["walkthrough_step"] = step

    previous_column, play_column, next_column = st.columns(3)
    previous_column.button(
        "← Previous",
        disabled=step == 0,
        on_click=_step_back,
        width="stretch",
    )
    play_column.button(
        "❚❚ Pause" if st.session_state["walkthrough_playing"] else "▶ Play",
        on_click=_toggle_playback,
        width="stretch",
    )
    next_column.button(
        "Next →",
        disabled=step >= total_references - 1,
        on_click=_step_forward,
        args=(total_references,),
        width="stretch",
    )

    selected_step = st.slider(
        "Walkthrough step",
        min_value=0,
        max_value=total_references - 1,
        key="walkthrough_step",
        format="%d",
    )
    event = result.events[selected_step]
    status = "HIT" if event.hit else "PAGE FAULT"
    st.markdown(f"#### Reference {selected_step + 1} of {total_references}")
    st.write(
        f"**{status}** · Page **{event.page}** requested"
        + (
            f" · evicted page **{event.evicted_page}**"
            if event.evicted_page is not None
            else ""
        )
    )

    frame_columns = st.columns(len(event.resident_pages) or 1)
    for frame_index in range(max(len(event.resident_pages), 1)):
        page = (
            event.resident_pages[frame_index]
            if frame_index < len(event.resident_pages)
            else "Empty"
        )
        frame_columns[frame_index].metric(f"Frame {frame_index + 1}", page)

    context_start = max(0, selected_step - 9)
    context_end = min(total_references, context_start + 20)
    timeline = []
    for index in range(context_start, context_end):
        timeline.append(
            {
                "Reference": index + 1,
                "Page": trace_pages[index],
                "Result": "Hit" if result.events[index].hit else "Fault",
                "Current": index == selected_step,
            }
        )
    st.dataframe(timeline, hide_index=True, width="stretch")
    st.caption(
        f"{trace_label} · {total_references:,} references · "
        f"{result.hits:,} hits · {result.faults:,} faults · "
        f"{result.fault_rate:.1%} fault rate"
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
        custom_trace_text = st.text_input(
            "Custom page-reference trace",
            placeholder="1, 2, 3, 1, 4, 2",
            key="custom_trace_text",
            help="Optional: enter comma- or space-separated page IDs to inspect your own trace.",
        )

    trace = generate_trace(
        workload,
        length=length,
        seed=int(seed),
        working_set_size=working_set_size,
        hot_set_size=hot_set_size,
        noise_probability=noise_probability,
    )
    trace_pages = trace.pages
    trace_label = (
        f"{trace.workload.replace('_', ' ').title()} workload · seed {trace.seed}"
    )
    if custom_trace_text.strip():
        try:
            trace_pages = tuple(
                int(token)
                for token in re.split(r"[\s,]+", custom_trace_text.strip())
                if token
            )
            if not trace_pages:
                raise ValueError("enter at least one page ID")
            if len(trace_pages) > 500:
                raise ValueError("custom traces are limited to 500 page IDs")
        except ValueError as error:
            st.error(f"Invalid custom trace: {error}")
            trace_pages = trace.pages
        else:
            trace_label = "Custom trace"

    results: dict[str, SimulationResult] = {
        name: simulate(
            trace_pages,
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
                trace_pages, frame_capacity, learned_policy, record_events=True
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

    st.subheader("Compare the policies")
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

    _render_walkthrough(trace_pages, trace_label, results, trace_pages)

    if "ML" in results:
        st.subheader("Why did the learned policy evict that page?")
        decisions = st.session_state.get("ml_decisions", [])
        _render_candidate_decision(trace_pages, decisions, results["ML"])

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

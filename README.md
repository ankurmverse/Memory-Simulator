# Page Replacement Simulator

A small, trace-driven simulator that demonstrates how page-replacement decisions affect hits, page faults, and resident memory frames. This first milestone establishes the simulation contract and FIFO baseline. Later milestones add more policies, workloads, benchmarking, machine learning, and a Streamlit demonstration.

> This is an educational simulator. It does not modify or benchmark an operating-system memory manager.

## Phase 1: simulator and FIFO

The simulator accepts a sequence of integer page IDs, a positive frame capacity, and a replacement policy. It returns aggregate reference/hit/fault/eviction counts and, by default, an event record for each request. FIFO evicts the resident page loaded earliest; a hit does not change FIFO order.

### Install

Requires Python 3.11 or newer:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Run the tests

```bash
python -m pytest
```

### Try the simulator

```bash
python - <<'PY'
from page_replacement.policies import FIFOPolicy
from page_replacement.simulator import simulate

result = simulate([1, 2, 3, 1, 4, 2], frame_capacity=3, policy=FIFOPolicy())
print(f"hits={result.hits}, faults={result.faults}, evictions={result.evictions}")
for event in result.events:
    print(event)
PY
```

Expected aggregate counts: **2 hits, 4 faults, 1 eviction**. On the fifth reference, page 4 faults and FIFO evicts page 1, the earliest-loaded resident.

## Eight-phase GitHub plan

Each phase is intended to be tested and pushed as a separate milestone. Future phases are not included in this Phase 1 code.

| Phase | Scope | Status |
|---|---|---|
| 1 | Project setup, simulator contract, FIFO, core tests, starter documentation | Current |
| 2 | LRU and CLOCK policies, victim-order and edge-case tests | Planned |
| 3 | Seeded synthetic workloads and trace CSV/metadata format | Planned |
| 4 | CLI for simulation/comparison, metrics, CSV output, and benchmark plots | Planned |
| 5 | Causal ML feature definitions and next-use training labels | Planned |
| 6 | Random-forest training, saved model, learned policy, held-out evaluation | Planned |
| 7 | Streamlit walkthrough, policy comparisons, per-eviction ML feature display | Planned |
| 8 | Final documentation, CI, packaging/release polish, measured demo/resume results | Planned |

After pushing a phase, preserve that commit as the checkpoint before beginning the next one. Do not claim ML improvements until held-out results support the claim.


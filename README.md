# Page Replacement Simulator

A small, trace-driven simulator that demonstrates how page-replacement decisions affect hits, page faults, and resident memory frames. This milestone includes FIFO, LRU, and CLOCK. Later milestones add workloads, benchmarking, machine learning, and a Streamlit demonstration.

> This is an educational simulator. It does not modify or benchmark an operating-system memory manager.

## Phase 2: FIFO, LRU, and CLOCK

The simulator accepts a sequence of integer page IDs, a positive frame capacity, and a replacement policy. It returns aggregate reference/hit/fault/eviction counts and, by default, an event record for each request.

- **FIFO** evicts the resident page loaded earliest. A hit does not change FIFO order.
- **LRU** evicts the resident page whose most recent reference was earliest. Hits update recency.
- **CLOCK** approximates LRU with a circular hand and reference bits. A page with its bit set gets a second chance: CLOCK clears the bit and advances; the first page later encountered with a clear bit is evicted. A loaded or hit page has its reference bit set.

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
from page_replacement.policies import CLOCKPolicy, FIFOPolicy, LRUPolicy
from page_replacement.simulator import simulate

trace = [1, 2, 3, 1, 4, 2]
for name, policy in (
    ("FIFO", FIFOPolicy()),
    ("LRU", LRUPolicy()),
    ("CLOCK", CLOCKPolicy()),
):
    result = simulate(trace, frame_capacity=3, policy=policy)
    print(f"{name}: hits={result.hits}, faults={result.faults}, evictions={result.evictions}")
PY
```

For this trace, FIFO has **2 hits, 4 faults, and 1 eviction**; LRU has **1 hit and 5 faults** because it evicts page 2 after page 1 is reused. CLOCK's victim choice depends on its reference bits and hand position.

## Eight-phase GitHub plan

Each phase is intended to be tested and pushed as a separate milestone. Future phases are not included in this Phase 1 code.

| Phase | Scope | Status |
|---|---|---|
| 1 | Project setup, simulator contract, FIFO, core tests, starter documentation | Complete |
| 2 | LRU and CLOCK policies, victim-order and edge-case tests | Current |
| 3 | Seeded synthetic workloads and trace CSV/metadata format | Planned |
| 4 | CLI for simulation/comparison, metrics, CSV output, and benchmark plots | Planned |
| 5 | Causal ML feature definitions and next-use training labels | Planned |
| 6 | Random-forest training, saved model, learned policy, held-out evaluation | Planned |
| 7 | Streamlit walkthrough, policy comparisons, per-eviction ML feature display | Planned |
| 8 | Final documentation, CI, packaging/release polish, measured demo/resume results | Planned |

After pushing a phase, preserve that commit as the checkpoint before beginning the next one. Do not claim ML improvements until held-out results support the claim.

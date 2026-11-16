# Page Replacement Simulator

A small, trace-driven simulator that demonstrates how page-replacement decisions affect hits, page faults, and resident memory frames. It includes FIFO, LRU, and CLOCK policies, seeded synthetic workloads, a CLI for reproducible comparisons, and causal ML training samples. Later milestones add model training and a Streamlit demonstration.

> This is an educational simulator. It does not modify or benchmark an operating-system memory manager.

## Phase 5: causal features and next-use labels

The simulator accepts a sequence of integer page IDs, a positive frame capacity, and a replacement policy. It returns aggregate reference/hit/fault/eviction counts and, by default, an event record for each request.

- **FIFO** evicts the resident page loaded earliest. A hit does not change FIFO order.
- **LRU** evicts the resident page whose most recent reference was earliest. Hits update recency.
- **CLOCK** approximates LRU with a circular hand and reference bits. A page with its bit set gets a second chance: CLOCK clears the bit and advances; the first page later encountered with a clear bit is evicted. A loaded or hit page has its reference bit set.

The trace generator provides five workloads: sequential scans, repeating loops, Zipf-skewed requests, hot-set traffic with configurable noise, and two disjoint phases. Pass the same seed and parameters to reproduce a generated trace.

```python
from page_replacement.traces import generate_trace, read_trace, write_trace

trace = generate_trace(
    "hot_set_noise",
    length=1_000,
    seed=42,
    working_set_size=32,
    hot_set_size=6,
    noise_probability=0.15,
)
csv_path, metadata_path = write_trace(trace, "data/traces/hot-set.csv")
loaded_trace = read_trace(csv_path)
assert loaded_trace == trace
```

Each trace CSV has `reference_index,page` columns, with zero-based contiguous reference indexes. Its sibling JSON file (`same-name.json`) records the format version, workload, seed, length, and generation parameters. The reader checks the schema and rejects inconsistent indexes or reference counts. See the small [looping trace](examples/traces/looping.csv) and its [metadata](examples/traces/looping.json).

For every fault that requires eviction, feature engineering creates one sample per currently resident candidate. The numeric features describe only the prefix before that decision:

- `access_count`: candidate references observed so far.
- `references_since_last_access`: distance from its most recent prior reference.
- `last_reuse_distance`, `mean_reuse_distance`, `max_reuse_distance`: statistics over observed gaps between consecutive references.
- `residency_age`: references since the candidate entered memory.

The offline training label `next_use_distance` is the number of references from the fault decision to the candidate's next occurrence. `None` means it is never referenced again in the available trace. This future-derived label is for training/evaluation only; it is not an input feature available to an online policy.

```python
from page_replacement.features import build_candidate_samples
from page_replacement.policies import LRUPolicy
from page_replacement.traces import read_trace

trace = read_trace("examples/traces/looping.csv")
samples = build_candidate_samples(trace.pages, frame_capacity=2, policy=LRUPolicy())
for sample in samples[:3]:
    print(
        sample.reference_index,
        sample.candidate_page,
        sample.features,
        sample.next_use_distance,
    )
```

Generate and persist a trace:

```bash
page-replacement generate --workload hot_set_noise --length 1000 --seed 42 \
  --working-set-size 32 --hot-set-size 6 --noise-probability 0.15 \
  --output data/traces/hot-set.csv
```

Compare all three policies on a generated trace (or pass `--trace path/to/file.csv` to use an existing trace). Results include faults, hits, evictions, fault rate, and hit rate. Metrics CSV and plot outputs are optional:

```bash
page-replacement compare --workload looping --length 1000 --seed 42 \
  --working-set-size 8 --frames 4 --csv results/looping.csv
python -m page_replacement compare --trace data/traces/hot-set.csv --frames 8 \
  --policies fifo lru clock --plot results/hot-set.png
```

Plots use the optional Matplotlib dependency. Install it with `python -m pip install -e ".[plots]"`; simulations and CSV export need no plotting dependency. The CLI can also be run from a source checkout with `python -m page_replacement`.

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

Each phase is intended to be tested and pushed as a separate milestone. Future phases are not included in this Phase 5 code.

| Phase | Scope | Status |
|---|---|---|
| 1 | Project setup, simulator contract, FIFO, core tests, starter documentation | Complete |
| 2 | LRU and CLOCK policies, victim-order and edge-case tests | Complete |
| 3 | Seeded synthetic workloads and trace CSV/metadata format | Complete |
| 4 | CLI for simulation/comparison, metrics, CSV output, and benchmark plots | Complete |
| 5 | Causal feature engineering and next-use training labels | Current |
| 6 | Random-forest training, saved model, learned policy, held-out evaluation | Planned |
| 7 | Streamlit walkthrough, policy comparisons, per-eviction ML feature display | Planned |
| 8 | Final documentation, CI, packaging/release polish, measured demo/resume results | Planned |

After pushing a phase, preserve that commit as the checkpoint before beginning the next one. Do not claim ML improvements until held-out results support the claim.

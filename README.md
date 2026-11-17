# Page Replacement Simulator

A trace-driven simulator for exploring how memory page replacement affects hits, page faults, and evictions. Compare classic policies with a machine-learning policy that estimates when a resident page will be needed again.

> This is an educational simulator. It does not modify or benchmark an operating-system memory manager.

## Features

- Simulate FIFO, LRU, and CLOCK page replacement.
- Generate reproducible sequential, looping, Zipf-skewed, hot-set/noise, and phase-change workloads.
- Save and reload traces as CSV with JSON metadata.
- Compare policies with hit/fault statistics, CSV reports, and optional plots.
- Train a random-forest policy from causal access-history features and evaluate it on held-out traces.
- Inspect per-reference simulation events and the learned model's feature importances.

## Install

Requires Python 3.11 or newer:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ml,plots]"
```

The `ml` and `plots` extras are optional. For simulation and CSV output only:

```bash
python -m pip install -e ".[dev]"
```

## Generate a trace and compare policies

```bash
page-replacement generate --workload hot_set_noise --length 1000 --seed 42 \
  --working-set-size 32 --hot-set-size 6 --noise-probability 0.15 \
  --output data/traces/hot-set.csv

page-replacement compare --trace data/traces/hot-set.csv --frames 8 \
  --policies fifo lru clock --csv results/hot-set.csv \
  --plot results/hot-set.png
```

The generated CSV contains `reference_index,page` columns. Its same-name JSON file stores the format version, workload, seed, length, and generation parameters. The plot option requires Matplotlib; CSV reports do not.

## Train and evaluate the learned policy

Training and evaluation require the `ml` extra. Training defaults to several workload families and seeds. Evaluation uses a different seed by default; keep it separate from the training seeds for a meaningful held-out result.

```bash
page-replacement train --workloads looping zipf hot_set_noise phase_change \
  --seeds 0 1 2 3 --length 2000 --working-set-size 16 --frames 4 \
  --model models/reuse-distance.joblib

page-replacement evaluate --model models/reuse-distance.joblib \
  --workload hot_set_noise --seed 100 --length 2000 \
  --working-set-size 16 --frames 4 --csv results/held-out.csv \
  --plot results/held-out.png
```

You can evaluate an existing trace instead of generating one with `--trace path/to/trace.csv`. Compare the learned policy's held-out fault count/rate with FIFO, LRU, and CLOCK; performance depends on workload and should be reported from measured results rather than assumed.

### What the model learns

At each eviction decision, training creates one candidate row for every resident page. The features are computed from the reference history available before that decision: number of prior accesses, references since the last access, observed reuse-distance statistics, and time in memory. The training target is the distance to the candidate page's next reference; pages never referenced again in the trace receive a target beyond that trace's length.

That future-derived next-use distance is a training label, not an online feature. At runtime, the model predicts a next-use distance for each resident candidate and evicts the one predicted to be needed farthest in the future. This is a learned heuristic, not a guarantee of outperforming classic policies.

Training traces are generated separately from the held-out evaluation seed. For credible comparisons, evaluate multiple workloads and seeds, and do not tune the model on the reported held-out traces.

## Python API

```python
from page_replacement.policies import CLOCKPolicy, FIFOPolicy, LRUPolicy
from page_replacement.simulator import simulate
from page_replacement.traces import generate_trace

trace = generate_trace("looping", length=100, seed=7, working_set_size=8)
for name, policy in (
    ("FIFO", FIFOPolicy()),
    ("LRU", LRUPolicy()),
    ("CLOCK", CLOCKPolicy()),
):
    result = simulate(trace.pages, frame_capacity=4, policy=policy)
    print(name, result.hits, result.faults, result.fault_rate)
```

## Tests

```bash
python -m pytest
```

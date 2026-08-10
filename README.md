# Reading Redistricting Plans by Shape

A graph-neural-network complement to vote-based redistricting outlier tests. The
model reads a districting plan's geometry against a neutral ReCom ensemble and
returns two things: a plan-level outlier verdict and a per-precinct localization
of where the plan departs from the ensemble.

## Layout

- `src/` — core library
  - `config.py` — global constants and seeds
  - `synthetic_state.py` — synthetic state generator for controlled experiments
  - `data_adapter.py` — loads real MGGG-states VTD maps and enacted plans
  - `features.py` — node feature assembly and standardization
  - `gnn_model.py` — GIN teacher/student and sparse adjacency
  - `ensemble.py` — ReCom recombination Markov chain
  - `planted.py` — packing/cracking strip planting, correlated and decorrelated
  - `experiments.py` — model fitting and plan scoring
  - `baselines.py` — boundary, cut-edge, degree, density baselines
  - `stats_tests.py` — Chikina-Frieze-Pegden outlier test and helpers
  - `sim_enacted.py` — enacted-plan simulation
- root scripts — experiment drivers (`run_all.py`, `run_why.py`, `run_robust.py`,
  `run_elections.py`, localization sweeps, and diagnostics)
- `paper/` — manuscript (`urtc_paper.md`), figure and table sources, references

## Running

```
python run_all.py --quick            # fast correctness check
python run_all.py --real NC          # real North Carolina map and enacted plan
python run_all.py --real PA
python run_all.py --real MD
```

Real runs read cached state pickles from `cache/`, rebuilt on first use from the
MGGG-states data.

## Dependencies

See `requirements.txt`.

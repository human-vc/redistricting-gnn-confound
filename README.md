# Reading Redistricting Plans by Shape

A graph-neural-network complement to vote-based redistricting outlier tests. The
model reads a districting plan's geometry against a neutral ReCom ensemble and
returns two things: a plan-level outlier verdict and a per-precinct localization
of where the plan departs from the ensemble.

## Layout

- `src/` — core library
  - `config.py` — global constants and seeds
  - `synthetic_state.py` — synthetic state generator for controlled experiments
  - `state_data.py` — MGGG-states shapefile schemas and topology repair for NC, PA, MD
  - `data_adapter.py` — loads real MGGG-states VTD maps and enacted plans
  - `features.py` — node feature assembly and standardization
  - `gnn_model.py` — GIN teacher/student and sparse adjacency
  - `ensemble.py` — ReCom recombination Markov chain
  - `planted.py` — packing/cracking strip planting, correlated and decorrelated
  - `experiments.py` — model fitting and plan scoring
  - `baselines.py` — boundary, cut-edge, degree, density baselines
  - `stats_tests.py` — two-sided empirical tail position against the ensemble, and helpers
  - `sim_enacted.py` — enacted-plan simulation
- `scripts/download_data.sh` — fetches the MGGG-states shapefiles into `data/raw/`
- root scripts — experiment drivers (`run_all.py`, `run_why.py`, `run_robust.py`,
  `run_elections.py`, `make_table1.py`, `gnn_win_full.py`, and diagnostics)
- `paper/` — manuscript (`urtc_paper.md`), figure and table sources, references

## Running

```
bash scripts/download_data.sh
python run_all.py --quick
python run_all.py --real NC
python run_all.py --real PA
python run_all.py --real MD
python make_table1.py
NSEEDS=3 python gnn_win_full.py
```

Real runs build each state from the shapefiles on first use and cache it in `cache/`.

## Dependencies

See `requirements.txt`.

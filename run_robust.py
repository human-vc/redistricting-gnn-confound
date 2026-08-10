import sys
sys.path.insert(0, "src")
import numpy as np
import config as cfg
import baselines as bl
import stats_tests as st
from data_adapter import load_real_state
from ensemble import run_chain

SB = cfg.GLOBAL_SEED + 2

def extreme(ens, test):
    p = st.percentile_rank(ens, test)
    return round(max(p, 100.0 - p), 1)

CONFIGS = [
    ("main", cfg.CHAIN_STEPS, cfg.REAL_POP_EPSILON),
    ("half", 3000, cfg.REAL_POP_EPSILON),
    ("double", 12000, cfg.REAL_POP_EPSILON),
    ("eps=0.01", cfg.CHAIN_STEPS, 0.01),
    ("eps=0.05", cfg.CHAIN_STEPS, 0.05),
]

for postal in ["NC", "PA", "MD"]:
    state, enacted = load_real_state(postal)
    for tag, steps, eps in CONFIGS:
        rng = np.random.default_rng(SB)
        chain = run_chain(state.graph, state.population, state.n_districts, eps, rng, steps, cfg.CHAIN_BURN_IN, cfg.CHAIN_THIN)
        sh = [bl.district_dem_shares(state, a) for a in chain]
        st_ = bl.district_dem_shares(state, enacted)
        mm = extreme([bl.mean_median(s) for s in sh], bl.mean_median(st_))
        eg = extreme([bl.efficiency_gap(state, a) for a in chain], bl.efficiency_gap(state, enacted))
        ce = extreme([bl.cut_edges(state, a) for a in chain], bl.cut_edges(state, enacted))
        print(f"ROBUST {postal} {tag:9s} n={len(chain):4d}  cut_edges={ce:5.1f}  mean_median={mm:5.1f}  eff_gap={eg:5.1f}", flush=True)

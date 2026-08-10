import sys
sys.path.insert(0, "src")
import numpy as np
import config as cfg
import baselines as bl
import stats_tests as st
from data_adapter import load_real_state
from ensemble import run_chain

SEED_BASE = cfg.GLOBAL_SEED

def extreme(ens, test):
    p = st.percentile_rank(ens, test)
    return round(max(p, 100.0 - p), 1)

for postal in ["NC", "PA", "MD"]:
    state, enacted = load_real_state(postal)
    rng_b = np.random.default_rng(SEED_BASE + 2)
    chain = run_chain(state.graph, state.population, state.n_districts,
                      cfg.REAL_POP_EPSILON, rng_b, cfg.CHAIN_STEPS, cfg.CHAIN_BURN_IN, cfg.CHAIN_THIN)
    shares = [bl.district_dem_shares(state, a) for a in chain]
    s_test = bl.district_dem_shares(state, enacted)
    mm = extreme([bl.mean_median(s) for s in shares], bl.mean_median(s_test))
    eg = extreme([bl.efficiency_gap(state, a) for a in chain], bl.efficiency_gap(state, enacted))
    dec_ens = [bl.declination(s) for s in shares]
    dec = extreme([d for d in dec_ens if not np.isnan(d)], bl.declination(s_test))
    ce = extreme([bl.cut_edges(state, a) for a in chain], bl.cut_edges(state, enacted))
    pp = extreme([bl.mean_polsby_popper(state, a) for a in chain], bl.mean_polsby_popper(state, enacted))
    print(f"TABLE1 {postal} n={len(chain)}  mm={mm}  eg={eg}  decl={dec}  cut_edges={ce}  polsby_popper={pp}", flush=True)

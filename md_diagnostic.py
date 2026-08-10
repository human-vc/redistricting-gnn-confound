import os, sys
sys.path.insert(0, "src")
os.environ.setdefault("THREADS", "5")
import numpy as np
import config as cfg
import features as feat
from ensemble import run_chain
from planted import generate_planted_set
from data_adapter import load_real_state

sb = cfg.GLOBAL_SEED

def moved_boundary_stats(state, chain_b, n):
    eu, ev, _, _ = feat._edge_arrays(state)
    rng_eval = np.random.default_rng(sb + 20)
    strip_sizes, moved_bf, interior_frac = [], [], []
    for condition in ("decorrelated", "correlated"):
        plants = []
        for et in ("pack", "crack"):
            plants += generate_planted_set(state, chain_b, condition, et, cfg.PLANTS_PER_CONDITION, rng_eval,
                                           epsilon=cfg.PLANT_EPSILON, strip_size_range=(cfg.STRIP_SIZE_MIN, cfg.STRIP_SIZE_MAX))
        for pl in plants:
            moved = list(pl.moved_nodes)
            if not (0 < len(moved) < n):
                continue
            bf = feat.compute_topology_features(state, pl.new_assignment)[:, 1]
            med = np.median(bf)
            strip_sizes.append(len(moved))
            moved_bf.append(bf[moved].mean())
            interior_frac.append(float((bf[moved] <= med).mean()))
    return np.mean(strip_sizes), np.mean(moved_bf), np.mean(interior_frac)

for postal in ["MD"]:
    state, enacted = load_real_state(postal)
    n = state.n_precincts
    eps, steps, burn, thin = cfg.REAL_POP_EPSILON, cfg.CHAIN_STEPS, cfg.CHAIN_BURN_IN, cfg.CHAIN_THIN
    chain_b = run_chain(state.graph, state.population, state.n_districts, eps, np.random.default_rng(sb + 2), steps, burn, thin)
    ss, mbf, itf = moved_boundary_stats(state, chain_b, n)
    ndist = state.n_districts
    print(f"DIAG {postal}: n={n} districts={ndist} precincts_per_district={n/ndist:.0f} "
          f"strip_size={ss:.1f} moved_boundary_exposure={mbf:.3f} interior_frac_of_moved={itf:.3f}", flush=True)
print("DIAGDONE", flush=True)

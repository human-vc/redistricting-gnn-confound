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

def ball3(adj, v):
    seen = {v}; frontier = {v}
    for _ in range(3):
        nxt = set()
        for u in frontier:
            nxt |= adj[u]
        nxt -= seen; seen |= nxt; frontier = nxt
    return seen

for postal in ["MD"]:
    state, _ = load_real_state(postal)
    n = state.n_precincts
    adj = [set(int(x) for x in state.graph.neighbors(i)) for i in range(n)]
    chain_b = run_chain(state.graph, state.population, state.n_districts, cfg.REAL_POP_EPSILON,
                        np.random.default_rng(sb + 2), cfg.CHAIN_STEPS, cfg.CHAIN_BURN_IN, cfg.CHAIN_THIN)
    rng = np.random.default_rng(sb + 20)
    interior, perim_area, lowbound, hop3 = [], [], [], []
    for cond in ["decorrelated", "correlated"]:
        plants = []
        for et in ["pack", "crack"]:
            plants += generate_planted_set(state, chain_b, cond, et, cfg.PLANTS_PER_CONDITION, rng,
                                           epsilon=cfg.PLANT_EPSILON, strip_size_range=(cfg.STRIP_SIZE_MIN, cfg.STRIP_SIZE_MAX))
        for pl in plants:
            M = set(int(x) for x in pl.moved_nodes)
            if not (0 < len(M) < n):
                continue
            bf = feat.compute_topology_features(state, pl.new_assignment)[:, 1]
            med = float(np.median(bf))
            interior.append(sum(1 for v in M if adj[v] <= M) / len(M))
            bedges = sum(1 for v in M for u in adj[v] if u not in M)
            perim_area.append(bedges / len(M))
            lowbound.append(float((bf[list(M)] <= med).mean()))
            hop3.append(float(np.mean([(lambda b: len(M & b) / len(b))(ball3(adj, v)) for v in M])))
    print(f"MORPH {postal}: interior_frac={np.mean(interior):.3f} perim_per_area={np.mean(perim_area):.3f} "
          f"lowboundary_moved={np.mean(lowbound):.3f} mean_3hop_moved_frac={np.mean(hop3):.3f}", flush=True)
print("MORPHDONE", flush=True)

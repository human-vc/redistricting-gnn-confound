import os, sys
sys.path.insert(0, "src")
os.environ.setdefault("THREADS", "5")
import numpy as np
import config as cfg
import baselines as bl
from ensemble import run_chain
from data_adapter import load_real_state

sb = cfg.GLOBAL_SEED

for postal in ["NC", "MD"]:
    state, _ = load_real_state(postal)
    n = state.n_precincts
    eps, steps, burn, thin = cfg.REAL_POP_EPSILON, cfg.CHAIN_STEPS, cfg.CHAIN_BURN_IN, cfg.CHAIN_THIN
    chain_b = run_chain(state.graph, state.population, state.n_districts, eps, np.random.default_rng(sb + 2), steps, burn, thin)
    A = np.stack([np.asarray(a) for a in chain_b])
    P = A.shape[0]

    ent = np.zeros(n); distinct = np.zeros(n)
    for i in range(n):
        _, counts = np.unique(A[:, i], return_counts=True)
        p = counts / counts.sum()
        ent[i] = float(-(p * np.log(p)).sum())
        distinct[i] = len(counts)
    ce = np.array([bl.cut_edges(state, a) for a in chain_b])
    ce_cv = float(ce.std() / ce.mean())
    print(f"DIV {postal}: n={n} districts={state.n_districts} plans={P} "
          f"mean_node_entropy={ent.mean():.3f} frac_nodes_ever_move={(distinct > 1).mean():.3f} "
          f"mean_distinct_districts={distinct.mean():.2f} cutedge_cv={ce_cv:.4f}", flush=True)
print("DIVDONE", flush=True)

import os, sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import config as cfg
import baselines as bl
import features as feat
import experiments as exp
from ensemble import run_chain
from planted import generate_planted_set
from data_adapter import load_real_state

N_SEEDS = int(os.environ.get("NSEEDS", cfg.N_MODEL_SEEDS))

def incident_cut_count(state, assignment):
    eu, ev, L, deg = feat._edge_arrays(state)
    a = np.asarray(assignment)
    cut = a[eu] != a[ev]
    n = state.n_precincts
    return np.bincount(eu[cut], minlength=n).astype(float) + np.bincount(ev[cut], minlength=n).astype(float)

def boundary_exposure(state, assignment):
    return feat.compute_topology_features(state, assignment)[:, 1]

rows = []
for postal in ["NC", "PA", "MD"]:
    sb = cfg.GLOBAL_SEED
    state, enacted = load_real_state(postal)
    eps, steps, burn, thin = cfg.REAL_POP_EPSILON, cfg.CHAIN_STEPS, cfg.CHAIN_BURN_IN, cfg.CHAIN_THIN
    print(f"[{postal}] chains", flush=True)
    chain_a = run_chain(state.graph, state.population, state.n_districts, eps, np.random.default_rng(sb + 1), steps, burn, thin)
    chain_b = run_chain(state.graph, state.population, state.n_districts, eps, np.random.default_rng(sb + 2), steps, burn, thin)
    print(f"[{postal}] training {N_SEEDS} topo + {N_SEEDS} demo seeds", flush=True)
    topo = [exp.fit_variant(state, chain_a, include_demo=False, seed=sb + 100 + s) for s in range(N_SEEDS)]
    demo = [exp.fit_variant(state, chain_a, include_demo=True, seed=sb + 200 + s) for s in range(N_SEEDS)]
    deg_null = bl.degree_null_scores(state)
    den_null = bl.density_null_scores(state)
    rng_eval = np.random.default_rng(sb + 20)
    for condition in ["decorrelated", "correlated"]:
        plants = []
        for edit_type in ["pack", "crack"]:
            plants += generate_planted_set(state, chain_b, condition, edit_type, cfg.PLANTS_PER_CONDITION,
                                           rng_eval, epsilon=cfg.PLANT_EPSILON,
                                           strip_size_range=(cfg.STRIP_SIZE_MIN, cfg.STRIP_SIZE_MAX))
        acc = {k: [] for k in ["gnn_topo", "gnn_demo", "boundary", "cutedge", "degree", "density"]}
        for pl in plants:
            y = np.zeros(state.n_precincts)
            for node in pl.moved_nodes:
                y[node] = 1.0
            if not (0 < y.sum() < len(y)):
                continue
            asg = pl.new_assignment
            acc["gnn_topo"].append(np.mean([roc_auc_score(y, exp.score_single(b, state, asg)[1]) for b in topo]))
            acc["gnn_demo"].append(np.mean([roc_auc_score(y, exp.score_single(b, state, asg)[1]) for b in demo]))
            acc["boundary"].append(roc_auc_score(y, boundary_exposure(state, asg)))
            acc["cutedge"].append(roc_auc_score(y, incident_cut_count(state, asg)))
            acc["degree"].append(roc_auc_score(y, deg_null))
            acc["density"].append(roc_auc_score(y, den_null))
        row = {"state": postal, "condition": condition, "n": len(acc["gnn_topo"])}
        for k, v in acc.items():
            row[k] = float(np.mean(v))
            row[k + "_std"] = float(np.std(v))
        rows.append(row)
        pd.DataFrame(rows).to_csv("paper/baseline_localization.csv", index=False)
        print("BASELINE {} {:12s} n={} gnn_topo={:.3f} gnn_demo={:.3f} boundary={:.3f} cutedge={:.3f} degree={:.3f} density={:.3f}".format(
            postal, condition, row["n"], row["gnn_topo"], row["gnn_demo"], row["boundary"], row["cutedge"], row["degree"], row["density"]), flush=True)

print("DONE -> paper/baseline_localization.csv", flush=True)

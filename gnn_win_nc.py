import os, sys
sys.path.insert(0, "src")
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import config as cfg
import features as feat
import experiments as exp
from ensemble import run_chain
from planted import generate_planted_set
from data_adapter import load_real_state

EPS = 1e-6
sb = cfg.GLOBAL_SEED

def incident_cut(state, a):
    eu, ev, L, deg = feat._edge_arrays(state)
    a = np.asarray(a); cut = a[eu] != a[ev]; n = state.n_precincts
    return np.bincount(eu[cut], minlength=n).astype(float) + np.bincount(ev[cut], minlength=n).astype(float)

def embeds(bundle, state, assignments):
    feats = np.stack([feat.assemble_features(state, a, bundle.include_demo) for a in assignments])
    X = torch.tensor(bundle.standardizer.transform(feats), dtype=torch.float32)
    bundle.model.eval()
    with torch.no_grad():
        Ht = bundle.model.teacher(X, bundle.A)
        Hs = bundle.model.student(X, bundle.A)
        disc = ((Ht - Hs) ** 2).mean(dim=-1)
    return Hs.numpy(), disc.numpy()

state, enacted = load_real_state("NC")
n = state.n_precincts
eps, steps, burn, thin = cfg.REAL_POP_EPSILON, cfg.CHAIN_STEPS, cfg.CHAIN_BURN_IN, cfg.CHAIN_THIN
print("chains", flush=True)
chain_a = run_chain(state.graph, state.population, state.n_districts, eps, np.random.default_rng(sb + 1), steps, burn, thin)
chain_b = run_chain(state.graph, state.population, state.n_districts, eps, np.random.default_rng(sb + 2), steps, burn, thin)
print("train topo", flush=True)
bundle = exp.fit_variant(state, chain_a, include_demo=False, seed=sb + 100)

hid = None
sHs = sHs2 = None
sD = np.zeros(n); sD2 = np.zeros(n)
sBF = np.zeros(n); sBF2 = np.zeros(n)
sCE = np.zeros(n); sCE2 = np.zeros(n)
Nn = 0
CH = 32
for i in range(0, len(chain_b), CH):
    batch = chain_b[i:i + CH]
    Hs, disc = embeds(bundle, state, batch)
    if hid is None:
        hid = Hs.shape[-1]; sHs = np.zeros((n, hid)); sHs2 = np.zeros((n, hid))
    sHs += Hs.sum(0); sHs2 += (Hs ** 2).sum(0)
    sD += disc.sum(0); sD2 += (disc ** 2).sum(0)
    for a in batch:
        ft = feat.compute_topology_features(state, a)
        bf = ft[:, 1]; ce = incident_cut(state, a)
        sBF += bf; sBF2 += bf ** 2; sCE += ce; sCE2 += ce ** 2
    Nn += len(batch)
mu_Hs = sHs / Nn; sd_Hs = np.sqrt(np.maximum(sHs2 / Nn - mu_Hs ** 2, 0)) + EPS
mu_D = sD / Nn; sd_D = np.sqrt(np.maximum(sD2 / Nn - mu_D ** 2, 0)) + EPS
mu_BF = sBF / Nn; sd_BF = np.sqrt(np.maximum(sBF2 / Nn - mu_BF ** 2, 0)) + EPS
mu_CE = sCE / Nn; sd_CE = np.sqrt(np.maximum(sCE2 / Nn - mu_CE ** 2, 0)) + EPS
print(f"neutral stats over {Nn} plans, hid={hid}", flush=True)

rng_eval = np.random.default_rng(sb + 20)
METHODS = ["disc_raw", "disc_cond", "embZ", "embZ+disc", "boundary_raw", "boundary_cond", "cutedge_raw", "cutedge_cond"]
for condition in ["decorrelated", "correlated"]:
    plants = []
    for et in ["pack", "crack"]:
        plants += generate_planted_set(state, chain_b, condition, et, cfg.PLANTS_PER_CONDITION, rng_eval,
                                       epsilon=cfg.PLANT_EPSILON, strip_size_range=(cfg.STRIP_SIZE_MIN, cfg.STRIP_SIZE_MAX))
    aucs = {m: [] for m in METHODS}
    for pl in plants:
        y = np.zeros(n)
        for node in pl.moved_nodes:
            y[node] = 1.0
        if not (0 < y.sum() < n):
            continue
        a = pl.new_assignment
        Hs, disc = embeds(bundle, state, [a]); Hs = Hs[0]; disc = disc[0]
        ft = feat.compute_topology_features(state, a); bf = ft[:, 1]; ce = incident_cut(state, a)
        disc_cond = (disc - mu_D) / sd_D
        embZ = np.sqrt((((Hs - mu_Hs) / sd_Hs) ** 2).mean(axis=1))
        def z(x): return (x - x.mean()) / (x.std() + EPS)
        sc = {
            "disc_raw": disc,
            "disc_cond": disc_cond,
            "embZ": embZ,
            "embZ+disc": z(embZ) + z(disc_cond),
            "boundary_raw": bf,
            "boundary_cond": (bf - mu_BF) / sd_BF,
            "cutedge_raw": ce,
            "cutedge_cond": (ce - mu_CE) / sd_CE,
        }
        for m in METHODS:
            aucs[m].append(roc_auc_score(y, sc[m]))
    print(f"\n=== NC {condition} (n={len(aucs['disc_raw'])}) ===", flush=True)
    for m in METHODS:
        print(f"  {m:14s} AUC = {np.mean(aucs[m]):.3f}", flush=True)
print("\nDONE", flush=True)

import os, sys
sys.path.insert(0, "src")
os.environ.setdefault("THREADS", "5")
import numpy as np
import torch
torch.set_num_threads(int(os.environ["THREADS"]))
import config as cfg
import features as feat
import experiments as exp
from ensemble import run_chain
from planted import generate_planted_set
from data_adapter import load_real_state

EPS = 1e-6
sb = cfg.GLOBAL_SEED
POSTAL = "NC"

def fused(model, state, assignments, A, std):
    feats = np.stack([feat.assemble_features(state, a, include_demo=False) for a in assignments])
    X = torch.tensor(std.transform(feats), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        Ht = model.teacher(X, A)
        Hs = model.student(X, A)
        disc = ((Ht - Hs) ** 2).mean(dim=-1)
    return Hs.numpy(), disc.numpy(), feats

state, enacted = load_real_state(POSTAL)
n = state.n_precincts
eu, ev, _, _ = feat._edge_arrays(state)

def incident_cut(a):
    a = np.asarray(a); cut = a[eu] != a[ev]
    return np.bincount(eu[cut], minlength=n).astype(float) + np.bincount(ev[cut], minlength=n).astype(float)

eps, steps, burn, thin = cfg.REAL_POP_EPSILON, cfg.CHAIN_STEPS, cfg.CHAIN_BURN_IN, cfg.CHAIN_THIN
chain_a = run_chain(state.graph, state.population, state.n_districts, eps, np.random.default_rng(sb + 1), steps, burn, thin)
chain_b = run_chain(state.graph, state.population, state.n_districts, eps, np.random.default_rng(sb + 2), steps, burn, thin)
print(f"[{POSTAL}] chains done (n={n}); training", flush=True)
bundle = exp.fit_variant(state, chain_a, include_demo=False, seed=sb + 100)
model, A, stdz = bundle.model, bundle.A, bundle.standardizer

sH = sH2 = sHHt = None
sD = np.zeros(n); sD2 = np.zeros(n)
sBF = np.zeros(n); sBF2 = np.zeros(n)
sCE = np.zeros(n); sCE2 = np.zeros(n)
Nn = 0
for i in range(0, len(chain_b), 32):
    batch = chain_b[i:i + 32]
    Hs, D, feats = fused(model, state, batch, A, stdz)
    if sH is None:
        hd = Hs.shape[-1]
        sH = np.zeros((n, hd)); sH2 = np.zeros((n, hd)); sHHt = np.zeros((n, hd, hd))
    sH += Hs.sum(0); sH2 += (Hs ** 2).sum(0)
    sHHt += np.einsum("bnd,bne->nde", Hs, Hs)
    sD += D.sum(0); sD2 += (D ** 2).sum(0)
    BF = feats[:, :, 1]; CE = np.stack([incident_cut(a) for a in batch])
    sBF += BF.sum(0); sBF2 += (BF ** 2).sum(0)
    sCE += CE.sum(0); sCE2 += (CE ** 2).sum(0)
    Nn += len(batch)

mu_Hs = sH / Nn
sd_Hs = np.sqrt(np.maximum(sH2 / Nn - mu_Hs ** 2, 0)) + EPS
cov_Hs = sHHt / Nn - np.einsum("nd,ne->nde", mu_Hs, mu_Hs)

def ms(s1, s2):
    mu = s1 / Nn; return mu, np.sqrt(np.maximum(s2 / Nn - mu ** 2, 0)) + EPS
mu_D, sd_D = ms(sD, sD2); mu_BF, sd_BF = ms(sBF, sBF2); mu_CE, sd_CE = ms(sCE, sCE2)
print(f"[{POSTAL}] neutral stats over {Nn} plans, hid={hd}", flush=True)

rng_eval = np.random.default_rng(sb + 20)
pH, pD, pBF, pCE, pY, pcond = [], [], [], [], [], []
for ci, condition in enumerate(("decorrelated", "correlated")):
    plants = []
    for et in ("pack", "crack"):
        plants += generate_planted_set(state, chain_b, condition, et, cfg.PLANTS_PER_CONDITION, rng_eval,
                                       epsilon=cfg.PLANT_EPSILON, strip_size_range=(cfg.STRIP_SIZE_MIN, cfg.STRIP_SIZE_MAX))
    for i in range(0, len(plants), 32):
        batch = plants[i:i + 32]
        Hs, D, feats = fused(model, state, [p.new_assignment for p in batch], A, stdz)
        for b, pl in enumerate(batch):
            y = np.zeros(n); y[list(pl.moved_nodes)] = 1.0
            if not (0 < y.sum() < n):
                continue
            pH.append(Hs[b].astype(np.float32)); pD.append(D[b].astype(np.float32))
            pBF.append(feats[b, :, 1].astype(np.float32)); pCE.append(incident_cut(pl.new_assignment).astype(np.float32))
            pY.append(y.astype(np.int8)); pcond.append(ci)
    print(f"[{POSTAL}] {condition}: {pcond.count(ci)} plants cached", flush=True)

np.savez_compressed(
    f"cache_{POSTAL}.npz",
    mu_Hs=mu_Hs.astype(np.float32), sd_Hs=sd_Hs.astype(np.float32), cov_Hs=cov_Hs.astype(np.float32),
    mu_D=mu_D, sd_D=sd_D, mu_BF=mu_BF, sd_BF=sd_BF, mu_CE=mu_CE, sd_CE=sd_CE,
    plant_Hs=np.stack(pH), plant_D=np.stack(pD), plant_BF=np.stack(pBF),
    plant_CE=np.stack(pCE), plant_y=np.stack(pY), plant_cond=np.array(pcond, dtype=np.int8),
)
print(f"[{POSTAL}] wrote cache_{POSTAL}.npz  ({len(pH)} plants, hid={hd})\nCACHEDONE", flush=True)

import os, sys
sys.path.insert(0, "src")
os.environ.setdefault("THREADS", "5")
import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(int(os.environ["THREADS"]))
from sklearn.metrics import roc_auc_score
import config as cfg
import features as feat
import experiments as exp
from gnn_model import GIN, build_sparse_adjacency
from ensemble import run_chain
from planted import generate_planted_set
from data_adapter import load_real_state

EPS = 1e-6
sb = cfg.GLOBAL_SEED
torch.manual_seed(sb)

class GAE(nn.Module):

    def __init__(self, in_dim, hidden=32, code=4, layers=3):
        super().__init__()
        self.enc = GIN(in_dim, hidden, layers)
        self.bottleneck = nn.Sequential(nn.Linear(hidden, code), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(code, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))

    def recon_err(self, X, A):
        Z = self.enc(X, A)
        R = self.dec(self.bottleneck(Z))
        return ((R - X) ** 2).mean(dim=-1)

def feats_of(state, assignments, stdz):
    F = np.stack([feat.assemble_features(state, a, include_demo=False) for a in assignments])
    return torch.tensor(stdz.transform(F), dtype=torch.float32), F

def train_gae(gae, Xtr, A, epochs=120, lr=1e-3):
    opt = torch.optim.Adam(gae.parameters(), lr=lr, weight_decay=1e-5)
    gae.train()
    for ep in range(epochs):
        opt.zero_grad()
        Z = gae.enc(Xtr, A)
        R = gae.dec(gae.bottleneck(Z))
        loss = ((R - Xtr) ** 2).mean()
        loss.backward(); opt.step()
    return float(loss.item())

state, enacted = load_real_state("NC")
n = state.n_precincts
eu, ev, _, _ = feat._edge_arrays(state)

def incident_cut(a):
    a = np.asarray(a); cut = a[eu] != a[ev]
    return np.bincount(eu[cut], minlength=n).astype(float) + np.bincount(ev[cut], minlength=n).astype(float)

eps, steps, burn, thin = cfg.REAL_POP_EPSILON, cfg.CHAIN_STEPS, cfg.CHAIN_BURN_IN, cfg.CHAIN_THIN
chain_a = run_chain(state.graph, state.population, state.n_districts, eps, np.random.default_rng(sb + 1), steps, burn, thin)
chain_b = run_chain(state.graph, state.population, state.n_districts, eps, np.random.default_rng(sb + 2), steps, burn, thin)
print(f"[NC] chains done (n={n})", flush=True)

from features import Standardizer
Ftr = np.stack([feat.assemble_features(state, a, include_demo=False) for a in chain_a[:128]])
in_dim = Ftr.shape[-1]
stdz = Standardizer().fit(Ftr)
A = build_sparse_adjacency(state.graph, n)
Xtr = torch.tensor(stdz.transform(Ftr), dtype=torch.float32)

trained = GAE(in_dim); untrained = GAE(in_dim)
loss = train_gae(trained, Xtr, A, epochs=120)
print(f"[NC] GAE trained, final recon loss {loss:.4f}", flush=True)

def neutral_stats(gae):
    s1 = np.zeros(n); s2 = np.zeros(n); N = 0
    gae.eval()
    with torch.no_grad():
        for i in range(0, len(chain_b), 32):
            Xb, _ = feats_of(state, chain_b[i:i + 32], stdz)
            E = gae.recon_err(Xb, A).numpy()
            s1 += E.sum(0); s2 += (E ** 2).sum(0); N += E.shape[0]
    mu = s1 / N; sd = np.sqrt(np.maximum(s2 / N - mu ** 2, 0)) + EPS
    return mu, sd
mu_t, sd_t = neutral_stats(trained)
mu_u, sd_u = neutral_stats(untrained)

sBF = np.zeros(n); sBF2 = np.zeros(n); NB = 0
for i in range(0, len(chain_b), 32):
    for a in chain_b[i:i + 32]:
        bf = feat.compute_topology_features(state, a)[:, 1]; sBF += bf; sBF2 += bf ** 2; NB += 1
mu_BF = sBF / NB; sd_BF = np.sqrt(np.maximum(sBF2 / NB - mu_BF ** 2, 0)) + EPS

rng_eval = np.random.default_rng(sb + 20)
for condition in ("decorrelated", "correlated"):
    plants = []
    for et in ("pack", "crack"):
        plants += generate_planted_set(state, chain_b, condition, et, cfg.PLANTS_PER_CONDITION, rng_eval,
                                       epsilon=cfg.PLANT_EPSILON, strip_size_range=(cfg.STRIP_SIZE_MIN, cfg.STRIP_SIZE_MAX))
    A_ = {k: [] for k in ["trained_raw", "trained_cond", "untrained_raw", "untrained_cond", "boundary_cond"]}
    for i in range(0, len(plants), 32):
        batch = plants[i:i + 32]
        Xb, _ = feats_of(state, [p.new_assignment for p in batch], stdz)
        trained.eval(); untrained.eval()
        with torch.no_grad():
            Et = trained.recon_err(Xb, A).numpy(); Eu = untrained.recon_err(Xb, A).numpy()
        for b, pl in enumerate(batch):
            y = np.zeros(n); y[list(pl.moved_nodes)] = 1.0
            if not (0 < y.sum() < n):
                continue
            bf = feat.compute_topology_features(state, pl.new_assignment)[:, 1]
            A_["trained_raw"].append(roc_auc_score(y, Et[b]))
            A_["trained_cond"].append(roc_auc_score(y, (Et[b] - mu_t) / sd_t))
            A_["untrained_raw"].append(roc_auc_score(y, Eu[b]))
            A_["untrained_cond"].append(roc_auc_score(y, (Eu[b] - mu_u) / sd_u))
            A_["boundary_cond"].append(roc_auc_score(y, (bf - mu_BF) / sd_BF))
    print(f"\n=== NC {condition} (n={len(A_['trained_raw'])}) ===", flush=True)
    for k in A_:
        print(f"  {k:16s} AUC = {np.mean(A_[k]):.3f}", flush=True)
print("\nGAEDONE", flush=True)

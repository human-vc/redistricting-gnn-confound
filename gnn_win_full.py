import os, sys
sys.path.insert(0, "src")
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import config as cfg
import features as feat
import experiments as exp
from ensemble import run_chain
from planted import generate_planted_set
from data_adapter import load_real_state

torch.set_num_threads(int(os.environ.get("THREADS", max(1, (os.cpu_count() or 4)))))
EPS = 1e-6
sb = cfg.GLOBAL_SEED
NSEEDS = int(os.environ.get("NSEEDS", "1"))
STATES = os.environ.get("STATES", "NC,PA,MD").split(",")
SUPERVISED = os.environ.get("SUPERVISED", "1") == "1"

def fused_forward(model, state, assignments, A, std):

    feats = np.stack([feat.assemble_features(state, a, include_demo=False) for a in assignments])
    X = torch.tensor(std.transform(feats), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        Ht = model.teacher(X, A)
        Hs = model.student(X, A)
        disc = ((Ht - Hs) ** 2).mean(dim=-1)
    return Hs.numpy(), Ht.numpy(), disc.numpy(), feats

def zc(x):
    return (x - x.mean()) / (x.std() + EPS)

for postal in STATES:
    state, enacted = load_real_state(postal)
    n = state.n_precincts
    eu, ev, _, _ = feat._edge_arrays(state)

    def incident_cut(a):
        a = np.asarray(a); cut = a[eu] != a[ev]
        return np.bincount(eu[cut], minlength=n).astype(float) + np.bincount(ev[cut], minlength=n).astype(float)

    eps, steps, burn, thin = cfg.REAL_POP_EPSILON, cfg.CHAIN_STEPS, cfg.CHAIN_BURN_IN, cfg.CHAIN_THIN
    chain_a = run_chain(state.graph, state.population, state.n_districts, eps, np.random.default_rng(sb + 1), steps, burn, thin)
    chain_b = run_chain(state.graph, state.population, state.n_districts, eps, np.random.default_rng(sb + 2), steps, burn, thin)
    print(f"[{postal}] chains done (n={n}), training {NSEEDS} seed(s)", flush=True)

    per_seed = {"decorrelated": [], "correlated": []}
    for s in range(NSEEDS):
        bundle = exp.fit_variant(state, chain_a, include_demo=False, seed=sb + 100 + s)
        model, A, stdz = bundle.model, bundle.A, bundle.standardizer

        sums = None
        Nn = 0
        for i in range(0, len(chain_b), 32):
            batch = chain_b[i:i + 32]
            Hs, Ht, D, feats = fused_forward(model, state, batch, A, stdz)
            BF = feats[:, :, 1]
            CE = np.stack([incident_cut(a) for a in batch])
            if sums is None:
                hd = Hs.shape[-1]
                sums = {k: np.zeros((n, hd)) for k in ("Hs1", "Hs2", "Ht1", "Ht2")}
                sums.update({k: np.zeros(n) for k in ("D1", "D2", "BF1", "BF2", "CE1", "CE2")})
            sums["Hs1"] += Hs.sum(0); sums["Hs2"] += (Hs ** 2).sum(0)
            sums["Ht1"] += Ht.sum(0); sums["Ht2"] += (Ht ** 2).sum(0)
            sums["D1"] += D.sum(0); sums["D2"] += (D ** 2).sum(0)
            sums["BF1"] += BF.sum(0); sums["BF2"] += (BF ** 2).sum(0)
            sums["CE1"] += CE.sum(0); sums["CE2"] += (CE ** 2).sum(0)
            Nn += len(batch)

        def mstd(k1, k2):
            mu = sums[k1] / Nn
            return mu, np.sqrt(np.maximum(sums[k2] / Nn - mu ** 2, 0)) + EPS
        muHs, sdHs = mstd("Hs1", "Hs2"); muHt, sdHt = mstd("Ht1", "Ht2")
        muD, sdD = mstd("D1", "D2"); muBF, sdBF = mstd("BF1", "BF2"); muCE, sdCE = mstd("CE1", "CE2")

        rng_eval = np.random.default_rng(sb + 20)
        methods = ["disc_raw", "disc_cond", "embZ_student", "embZ_teacher", "embZ+disc",
                   "boundary_raw", "boundary_cond", "cutedge_raw", "cutedge_cond"]
        for condition in ("decorrelated", "correlated"):
            plants = []
            for et in ("pack", "crack"):
                plants += generate_planted_set(state, chain_b, condition, et, cfg.PLANTS_PER_CONDITION, rng_eval,
                                               epsilon=cfg.PLANT_EPSILON, strip_size_range=(cfg.STRIP_SIZE_MIN, cfg.STRIP_SIZE_MAX))
            aucs = {m: [] for m in methods}
            sup_X, sup_y, sup_grp = [], [], []
            for i in range(0, len(plants), 32):
                batch = plants[i:i + 32]
                Hs, Ht, D, feats = fused_forward(model, state, [p.new_assignment for p in batch], A, stdz)
                for b, pl in enumerate(batch):
                    y = np.zeros(n); y[list(pl.moved_nodes)] = 1.0
                    if not (0 < y.sum() < n):
                        continue
                    bf = feats[b, :, 1]; ce = incident_cut(pl.new_assignment)
                    embZs = np.sqrt((((Hs[b] - muHs) / sdHs) ** 2).mean(1))
                    embZt = np.sqrt((((Ht[b] - muHt) / sdHt) ** 2).mean(1))
                    dcond = (D[b] - muD) / sdD
                    sc = {"disc_raw": D[b], "disc_cond": dcond, "embZ_student": embZs, "embZ_teacher": embZt,
                          "embZ+disc": zc(embZs) + zc(dcond),
                          "boundary_raw": bf, "boundary_cond": (bf - muBF) / sdBF,
                          "cutedge_raw": ce, "cutedge_cond": (ce - muCE) / sdCE}
                    for m in methods:
                        aucs[m].append(roc_auc_score(y, sc[m]))
                    if SUPERVISED:
                        sup_X.append(np.column_stack([Hs[b], (Hs[b] - muHs) / sdHs, bf, (bf - muBF) / sdBF]))
                        sup_y.append(y); sup_grp.append(np.full(n, i + b))

            row = {m: float(np.mean(aucs[m])) for m in methods}
            row["supervised_cv"] = float("nan")
            if SUPERVISED:
                Xs = np.vstack(sup_X); ys = np.concatenate(sup_y); gs = np.concatenate(sup_grp)
                sup_auc = []
                for tr, te in GroupKFold(n_splits=5).split(Xs, ys, gs):
                    clf = LogisticRegression(max_iter=200).fit(Xs[tr], ys[tr])
                    p = clf.predict_proba(Xs[te])[:, 1]
                    for g in np.unique(gs[te]):
                        m = gs[te] == g
                        if 0 < ys[te][m].sum() < m.sum():
                            sup_auc.append(roc_auc_score(ys[te][m], p[m]))
                row["supervised_cv"] = float(np.mean(sup_auc))
            sd = {m: float(np.std(aucs[m])) for m in methods}
            per_seed[condition].append(row)
            print(f"[{postal}] seed{s} {condition}: " + "  ".join(f"{k}={row[k]:.3f}(sd{sd.get(k, 0.0):.2f})" for k in methods + ["supervised_cv"]), flush=True)

    for condition in ("decorrelated", "correlated"):
        rows = per_seed[condition]
        print(f"\n### {postal} {condition} (mean over {len(rows)} seed(s)) ###", flush=True)
        for k in rows[0]:
            vals = [r[k] for r in rows]
            tail = f" (sd {np.std(vals):.3f})" if len(vals) > 1 else ""
            print(f"    {k:14s} {np.mean(vals):.3f}{tail}", flush=True)
print("\nALLDONE", flush=True)

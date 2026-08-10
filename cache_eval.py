import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-6

def load_cache(postal="NC"):
    z = np.load(f"cache_{postal}.npz")
    return {k: z[k] for k in z.files}

def evaluate(score_fn, cache, conditions=("decorrelated", "correlated")):

    out = {}
    y = cache["plant_y"]; cond = cache["plant_cond"]
    for ci, name in enumerate(conditions):
        idx = np.where(cond == ci)[0]
        aucs = []
        for i in idx:
            yi = y[i]
            if 0 < yi.sum() < len(yi):
                aucs.append(roc_auc_score(yi, score_fn(cache, i)))
        out[name] = float(np.mean(aucs)) if aucs else float("nan")
    return out

def embZ_student(cache, i):
    Hs = cache["plant_Hs"][i]
    return np.sqrt((((Hs - cache["mu_Hs"]) / cache["sd_Hs"]) ** 2).mean(axis=1))

def boundary_raw(cache, i):
    return cache["plant_BF"][i]

def boundary_cond(cache, i):
    return (cache["plant_BF"][i] - cache["mu_BF"]) / cache["sd_BF"]

def cutedge_cond(cache, i):
    return (cache["plant_CE"][i] - cache["mu_CE"]) / cache["sd_CE"]

def disc_raw(cache, i):
    return cache["plant_D"][i]

def disc_cond(cache, i):
    return (cache["plant_D"][i] - cache["mu_D"]) / cache["sd_D"]

REFERENCE = {
    "embZ_student": embZ_student, "boundary_raw": boundary_raw, "boundary_cond": boundary_cond,
    "cutedge_cond": cutedge_cond, "disc_raw": disc_raw, "disc_cond": disc_cond,
}

def maha_fullcov_precompute(cache, shrink=0.1):

    cov = cache["cov_Hs"].astype(np.float64)
    n, h, _ = cov.shape
    tr = np.trace(cov, axis1=1, axis2=2) / h
    reg = cov * (1 - shrink) + shrink * tr[:, None, None] * np.eye(h)[None]
    return np.linalg.inv(reg)

def report(cache, extra=None):

    methods = dict(REFERENCE)
    if extra:
        methods.update(extra)
    rows = {name: evaluate(fn, cache) for name, fn in methods.items()}
    bar = rows["boundary_cond"]
    print(f"{'method':18s} {'decorr':>8s} {'corr':>8s}   beats boundary_cond?")
    for name, r in rows.items():
        d, c = r["decorrelated"], r["correlated"]
        flag = "WIN" if (d > bar["decorrelated"] and c > bar["correlated"]) else ""
        print(f"{name:18s} {d:8.3f} {c:8.3f}   {flag}")
    return rows

if __name__ == "__main__":
    cache = load_cache("NC")
    report(cache)

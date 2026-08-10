from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, "src")

import numpy as np

import config as cfg
import stats_tests as st
from data_adapter import load_real_state
from ensemble import run_chain

def cut_edges(state, assignment) -> int:
    a = np.asarray(assignment)
    return int(sum(1 for u, v in state.graph.edges() if a[u] != a[v]))

def mean_polsby_popper(state, assignment) -> float:
    a = np.asarray(assignment)
    k = state.n_districts
    area = np.bincount(a, weights=state.area, minlength=k)
    perim = np.array([state.perimeter[a == d].sum() for d in range(k)], dtype=float)

    for u, v in state.graph.edges():
        if a[u] == a[v]:
            L = state.shared_boundary.get((u, v) if u < v else (v, u), 0.0)
            perim[a[u]] -= 2.0 * L
    perim = np.maximum(perim, 1e-9)
    pp = 4.0 * math.pi * area / (perim ** 2)
    return float(np.mean(pp))

def _extreme(ensemble, test) -> float:
    v = [x for x in ensemble if not np.isnan(x)]
    if not v or np.isnan(test):
        return float("nan")
    p = st.percentile_rank(v, test)
    return max(p, 100.0 - p)

def diagnose(postal: str, steps: int, burn: int, thin: int, seed: int, out_dir: str) -> dict:
    import baselines as bl
    import pandas as pd

    state, enacted = load_real_state(postal)
    rng = np.random.default_rng(seed)
    print(f"[{postal}] building neutral ensemble (n={state.n_precincts}, k={state.n_districts})", flush=True)
    chain = run_chain(state.graph, state.population, state.n_districts, cfg.REAL_POP_EPSILON, rng, steps, burn, thin)

    ce_ens = np.array([cut_edges(state, a) for a in chain], dtype=float)
    pp_ens = np.array([mean_polsby_popper(state, a) for a in chain], dtype=float)
    shares_ens = [bl.district_dem_shares(state, a) for a in chain]
    mm_ens = np.array([bl.mean_median(s) for s in shares_ens])
    eg_ens = np.array([bl.efficiency_gap(state, a) for a in chain])
    dec_ens = np.array([bl.declination(s) for s in shares_ens])

    shares_test = bl.district_dem_shares(state, enacted)
    ce_test = float(cut_edges(state, enacted))
    pp_test = mean_polsby_popper(state, enacted)
    mm_test = bl.mean_median(shares_test)
    eg_test = bl.efficiency_gap(state, enacted)
    dec_test = bl.declination(shares_test)

    pd.DataFrame({
        "cut_edges": ce_ens, "polsby_popper": pp_ens,
        "mean_median": mm_ens, "efficiency_gap": eg_ens, "declination": dec_ens,
    }).to_csv(os.path.join(out_dir, f"ensemble_{postal}.csv"), index=False)

    ce_pct = st.percentile_rank(ce_ens, ce_test)
    print(f"[{postal}] cut_edges: enacted={ce_test:.0f} ensemble={ce_ens.mean():.0f}+-{ce_ens.std():.0f} "
          f"pctile={ce_pct:.1f} (extreme {max(ce_pct, 100-ce_pct):.1f})", flush=True)
    return {
        "state": postal, "n_ensemble": len(chain),
        "cut_edges_enacted": ce_test, "cut_edges_ens_mean": round(float(ce_ens.mean()), 1),
        "cut_edges_ens_std": round(float(ce_ens.std()), 1),
        "cut_edges_extreme_pct": round(max(ce_pct, 100 - ce_pct), 1),
        "polsby_popper_enacted": round(pp_test, 4),
        "polsby_popper_extreme_pct": round(_extreme(pp_ens, pp_test), 1),
        "mean_median_enacted": round(mm_test, 4), "mean_median_extreme_pct": round(_extreme(mm_ens, mm_test), 1),
        "efficiency_gap_enacted": round(eg_test, 4), "efficiency_gap_extreme_pct": round(_extreme(eg_ens, eg_test), 1),
        "declination_enacted": round(dec_test, 4), "declination_extreme_pct": round(_extreme(dec_ens, dec_test), 1),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", default=["NC", "PA", "MD"])
    ap.add_argument("--steps", type=int, default=cfg.CHAIN_STEPS)
    ap.add_argument("--burn", type=int, default=cfg.CHAIN_BURN_IN)
    ap.add_argument("--thin", type=int, default=cfg.CHAIN_THIN)
    ap.add_argument("--out", default="outputs_why")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    import pandas as pd
    rows = [diagnose(s, args.steps, args.burn, args.thin, cfg.GLOBAL_SEED + i, args.out) for i, s in enumerate(args.states)]
    table = pd.DataFrame(rows)
    path = os.path.join(args.out, "enacted_stats.csv")
    table.to_csv(path, index=False)
    print("\n" + "=" * 78)
    print("ENACTED PLAN STATISTICS vs NEUTRAL ENSEMBLE (default election)")
    print("=" * 78)
    print(table.to_string(index=False))
    print(f"\ntable written to {path}")

if __name__ == "__main__":
    main()

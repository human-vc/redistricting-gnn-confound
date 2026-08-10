from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

import baselines
import config as cfg
import stats_tests as st
from data_adapter import load_election_panel, load_real_state
from ensemble import run_chain

def _extreme_pct(ensemble_vals, test_val):
    vals = [v for v in ensemble_vals if not np.isnan(v)]
    if not vals or np.isnan(test_val):
        return float("nan")
    p = st.percentile_rank(vals, test_val)
    return max(p, 100.0 - p)

def sweep_state(postal: str, steps: int, burn: int, thin: int, seed: int) -> pd.DataFrame:
    state, enacted = load_real_state(postal)
    panel = load_election_panel(postal)
    thresh = cfg.THRESH_CFP_PERCENTILE * 100

    print(f"[{postal}] building neutral ensemble (n={state.n_precincts}, k={state.n_districts})", flush=True)
    rng = np.random.default_rng(seed)
    chain = run_chain(state.graph, state.population, state.n_districts, cfg.REAL_POP_EPSILON, rng, steps, burn, thin)
    print(f"[{postal}]   ensemble size {len(chain)}", flush=True)

    rows = []
    for label, (dem, rep) in panel.items():
        total = dem + rep
        dem_share = np.where(total > 0, dem / np.maximum(total, 1e-9), 0.5)
        state_e = dataclasses.replace(state, dem_share=np.clip(dem_share, 0.0, 1.0), total_votes=total)

        shares_ens = [baselines.district_dem_shares(state_e, a) for a in chain]
        shares_test = baselines.district_dem_shares(state_e, enacted)
        mm = _extreme_pct([baselines.mean_median(s) for s in shares_ens], baselines.mean_median(shares_test))
        eg = _extreme_pct([baselines.efficiency_gap(state_e, a) for a in chain], baselines.efficiency_gap(state_e, enacted))
        dec = _extreme_pct([baselines.declination(s) for s in shares_ens], baselines.declination(shares_test))

        rows.append({
            "state": postal, "election": label,
            "statewide_dem_share": round(float(np.average(dem_share, weights=np.maximum(total, 1e-9))), 4),
            "mean_median_pct": round(mm, 2), "mm_flags": bool(mm >= thresh),
            "efficiency_gap_pct": round(eg, 2), "eg_flags": bool(eg >= thresh),
            "declination_pct": round(dec, 2), "dec_flags": bool(dec >= thresh),
        })
        print(f"[{postal}] {label:>20}: MM {mm:5.1f} {'FLAG' if mm>=thresh else '  . '} | "
              f"EG {eg:5.1f} {'FLAG' if eg>=thresh else '  . '} | "
              f"DEC {dec:5.1f} {'FLAG' if dec>=thresh else '  . '}", flush=True)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", default=["NC", "PA"])
    ap.add_argument("--steps", type=int, default=cfg.CHAIN_STEPS)
    ap.add_argument("--burn", type=int, default=cfg.CHAIN_BURN_IN)
    ap.add_argument("--thin", type=int, default=cfg.CHAIN_THIN)
    ap.add_argument("--out", default="outputs_elections")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    frames = [sweep_state(s, args.steps, args.burn, args.thin, cfg.GLOBAL_SEED + i)
              for i, s in enumerate(args.states)]
    table = pd.concat(frames, ignore_index=True)
    path = os.path.join(args.out, "multi_election_robustness.csv")
    table.to_csv(path, index=False)

    print("\n" + "=" * 78)
    print("MULTI-ELECTION ROBUSTNESS (enacted plan vs neutral ensemble, extreme percentile)")
    print("=" * 78)
    for s in args.states:
        sub = table[table.state == s]
        n_races = len(sub)
        print(f"\n{s}: mean-median flags {int(sub.mm_flags.sum())}/{n_races} elections | "
              f"efficiency-gap {int(sub.eg_flags.sum())}/{n_races} | declination {int(sub.dec_flags.sum())}/{n_races}")
    print(f"\ntable written to {path}")
    print(f"total runtime {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()

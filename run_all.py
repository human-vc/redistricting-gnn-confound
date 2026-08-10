from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import torch

import baselines
import config as cfg
import experiments as exp
from ensemble import run_chain
from planted import generate_planted_set, measure_strip_confound
from sim_enacted import build_sim_enacted
from synthetic_state import generate_synthetic_state

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def aggregate_seeds(results: list[dict], value_keys: list[str], pass_key: str | None = None) -> dict:

    out = {"n_seeds": len(results)}
    for key in value_keys:
        vals = np.array([r[key] for r in results], dtype=float)
        out[key] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
        out[f"{key}_seeds"] = vals
    if pass_key is not None:
        votes = [bool(r[pass_key]) for r in results]
        out["pass"] = bool(np.mean(votes) > 0.5)
        out["pass_votes"] = votes
    return out

def run_state_full(state_name: str, n_precincts: int, n_districts: int, seed_base: int, quick: bool,
                   real: tuple | None = None) -> dict:
    steps = 800 if quick else cfg.CHAIN_STEPS
    burn = 100 if quick else cfg.CHAIN_BURN_IN
    thin = 6 if quick else cfg.CHAIN_THIN
    epochs = 20 if quick else cfg.GNN_EPOCHS
    plants_n = 10 if quick else cfg.PLANTS_PER_CONDITION
    n_shuffles = 40 if quick else cfg.N_VOTE_SHUFFLES
    n_boot = 200 if quick else cfg.N_BOOTSTRAP
    n_perm = 200 if quick else cfg.N_PERMUTATION
    n_seeds = 2 if quick else cfg.N_MODEL_SEEDS

    cfg.seed_everything(seed_base)
    chain_eps = cfg.POP_EPSILON
    if real is not None:
        state, real_enacted = real
        n_districts = state.n_districts
        n_precincts = state.n_precincts
        chain_eps = cfg.REAL_POP_EPSILON
        log(f"[{state_name}] REAL state loaded (n={n_precincts}, k={n_districts}, "
            f"chain epsilon={chain_eps})")
    else:
        rng_state = np.random.default_rng(seed_base)
        log(f"[{state_name}] generating synthetic precinct universe (n={n_precincts}, k={n_districts})")
        state = generate_synthetic_state(state_name, n_precincts, n_districts, rng_state)

    log(f"[{state_name}] running training chain A")
    rng_a = np.random.default_rng(seed_base + 1)
    chain_a = run_chain(state.graph, state.population, n_districts, chain_eps, rng_a, steps, burn, thin)
    log(f"[{state_name}]   chain A size = {len(chain_a)}")

    log(f"[{state_name}] running evaluation chain B (independent seed)")
    rng_b = np.random.default_rng(seed_base + 2)
    chain_b = run_chain(state.graph, state.population, n_districts, chain_eps, rng_b, steps, burn, thin)
    log(f"[{state_name}]   chain B size = {len(chain_b)}")

    topo_bundles, demo_bundles = [], []
    for s in range(n_seeds):
        log(f"[{state_name}] training topology-only variant, seed {s + 1}/{n_seeds}")
        tb = exp.fit_variant(state, chain_a, include_demo=False, seed=seed_base + 100 + s)
        log(f"[{state_name}]   best val loss = {tb.history['best_val_loss']:.5f}")
        topo_bundles.append(tb)

        log(f"[{state_name}] training +demographic variant, seed {s + 1}/{n_seeds}")
        db = exp.fit_variant(state, chain_a, include_demo=True, seed=seed_base + 200 + s)
        log(f"[{state_name}]   best val loss = {db.history['best_val_loss']:.5f}")
        demo_bundles.append(db)
    topo_bundle, demo_bundle = topo_bundles[0], demo_bundles[0]

    rng_eval = np.random.default_rng(seed_base + 20)

    if real is not None:
        log(f"[{state_name}] using REAL adjudicated enacted plan as positive control")
        sim_assignment = real_enacted
        sim_res = {
            "assignment": sim_assignment,
            "efficiency_gap": baselines.efficiency_gap(state, sim_assignment),
            "mean_median": baselines.mean_median(baselines.district_dem_shares(state, sim_assignment)),
        }
        log(f"[{state_name}]   ENACTED EG={sim_res['efficiency_gap']:.4f}  MM={sim_res['mean_median']:.4f}")
    else:
        log(f"[{state_name}] building SIM-ENACTED plan (directed hill-climb on chain B)")
        sim_res = build_sim_enacted(
            state, chain_b, cfg.POP_EPSILON, rng_eval,
            n_iters=100 if quick else 400, candidates_per_iter=6,
        )
        sim_assignment = sim_res["assignment"]
        log(f"[{state_name}]   SIM-ENACTED EG={sim_res['efficiency_gap']:.4f}  MM={sim_res['mean_median']:.4f}")

    log(f"[{state_name}] Table 1: classical baseline reproduction")
    classical = exp.classical_baseline_reproduction(state, chain_b, sim_assignment, rng_eval)

    log(f"[{state_name}] Table 2: CFP epsilon-outlier test (topo + demo), seed-averaged")
    cfp_topo_seeds = [exp.cfp_outlier_test(state, chain_b, tb, sim_assignment) for tb in topo_bundles]
    cfp_demo_seeds = [exp.cfp_outlier_test(state, chain_b, db, sim_assignment) for db in demo_bundles]
    cfp_topo = aggregate_seeds(cfp_topo_seeds, ["p_value", "percentile", "extreme_percentile", "eps"], pass_key="pass")
    cfp_demo = aggregate_seeds(cfp_demo_seeds, ["p_value", "percentile", "extreme_percentile", "eps"], pass_key="pass")
    cfp_topo["ensemble_scores"] = cfp_topo_seeds[0]["ensemble_scores"]
    cfp_topo["test_score"] = cfp_topo_seeds[0]["test_score"]
    cfp_topo["threshold_percentile"] = cfp_topo_seeds[0]["threshold_percentile"]
    cfp_demo["ensemble_scores"] = cfp_demo_seeds[0]["ensemble_scores"]
    cfp_demo["test_score"] = cfp_demo_seeds[0]["test_score"]
    cfp_demo["threshold_percentile"] = cfp_demo_seeds[0]["threshold_percentile"]

    log(f"[{state_name}] Table 3: confound-isolation horse race, seed-averaged")
    horse_seeds = [
        exp.confound_isolation_horse_race(state, chain_b, tb, db, rng_eval, n_boot)
        for tb, db in zip(topo_bundles, demo_bundles)
    ]
    horse = aggregate_seeds(horse_seeds, ["spearman_rho", "spearman_p", "kendall_tau", "kendall_p", "spearman_ci_lo", "spearman_ci_hi"], pass_key="pass")
    horse["topo_scores"] = horse_seeds[0]["topo_scores"]
    horse["demo_scores"] = horse_seeds[0]["demo_scores"]
    horse["threshold"] = cfg.THRESH_SPEARMAN_HIGH

    log(f"[{state_name}] Table 4: vote-share shuffle invariance, seed-averaged")
    shuffle_topo_seeds = [
        exp.vote_share_shuffle_invariance(state, sim_assignment, tb, cfp["ensemble_scores"], n_shuffles, rng_eval)
        for tb, cfp in zip(topo_bundles, cfp_topo_seeds)
    ]
    shuffle_demo_seeds = [
        exp.vote_share_shuffle_invariance(state, sim_assignment, db, cfp["ensemble_scores"], n_shuffles, rng_eval)
        for db, cfp in zip(demo_bundles, cfp_demo_seeds)
    ]
    shuffle_topo = aggregate_seeds(shuffle_topo_seeds, ["base_score", "mean_abs_delta", "ensemble_spread", "ratio"], pass_key="pass_invariant")
    shuffle_demo = aggregate_seeds(shuffle_demo_seeds, ["base_score", "mean_abs_delta", "ensemble_spread", "ratio"], pass_key="pass_invariant")
    shuffle_topo["deltas"] = np.concatenate([s["deltas"] for s in shuffle_topo_seeds])
    shuffle_demo["deltas"] = np.concatenate([s["deltas"] for s in shuffle_demo_seeds])
    shuffle_topo["threshold"] = cfg.THRESH_SHUFFLE_RATIO_SMALL
    shuffle_demo["threshold"] = cfg.THRESH_SHUFFLE_RATIO_SMALL

    log(f"[{state_name}] generating planted gerrymander sets (decorrelated + correlated, pack + crack)")
    plants = {}
    for condition in ["decorrelated", "correlated"]:
        combined = []
        for edit_type in ["pack", "crack"]:
            combined += generate_planted_set(
                state, chain_b, condition, edit_type, plants_n, rng_eval,
                epsilon=cfg.PLANT_EPSILON, strip_size_range=(cfg.STRIP_SIZE_MIN, cfg.STRIP_SIZE_MAX),
            )
        plants[condition] = combined
        log(f"[{state_name}]   {condition}: {len(combined)} planted plans")

    strip_confound = {c: measure_strip_confound(state, plants[c]) for c in ["decorrelated", "correlated"]}
    for c in ["decorrelated", "correlated"]:
        sc = strip_confound[c]
        log(f"[{state_name}]   {c} strip density z={sc['mean_density_z']:.3f}, "
            f"density-partisan corr={sc['density_partisan_corr']:.3f}")

    log(f"[{state_name}] Table 5: node-level localization (seed-averaged across {n_seeds} seeds)")
    null_scores = {"degree": baselines.degree_null_scores(state), "density": baselines.density_null_scores(state)}
    localization = {}
    for variant_name, bundles in [("topology_only", topo_bundles), ("plus_demographic", demo_bundles)]:
        localization[variant_name] = {}
        for condition in ["decorrelated", "correlated"]:
            loc = exp.evaluate_localization_set(state, plants[condition], bundles, null_scores, rng_eval)
            comparisons = {
                null_name: exp.compare_gnn_vs_null(loc, null_name, n_perm, rng_eval)
                for null_name in list(null_scores.keys()) + ["random"]
            }
            localization[variant_name][condition] = {"loc": loc, "comparisons": comparisons}

    log(f"[{state_name}] Table 6: residual-threat measurement (topology-feature vs demographic correlation)")
    topo_feats = np.stack([exp.assemble_features(state, a, include_demo=False) for a in chain_b])
    demo_feats = np.stack([exp.assemble_features(state, a, include_demo=True) for a in chain_b])
    from features import DEMOGRAPHIC_FEATURE_NAMES, TOPOLOGY_FEATURE_NAMES
    mean_topo = topo_feats.mean(axis=0)
    demo_only = demo_feats[:, :, len(TOPOLOGY_FEATURE_NAMES):].mean(axis=0)
    residual_corr = {}
    from scipy.stats import spearmanr
    for i, tname in enumerate(TOPOLOGY_FEATURE_NAMES):
        for j, dname in enumerate(DEMOGRAPHIC_FEATURE_NAMES):
            rho, p = spearmanr(mean_topo[:, i], demo_only[:, j])
            residual_corr[f"{tname}__{dname}"] = {"rho": float(rho), "p": float(p)}

    return {
        "state": state, "chain_a": chain_a, "chain_b": chain_b,
        "topo_bundle": topo_bundle, "demo_bundle": demo_bundle,
        "sim_assignment": sim_assignment, "sim_res": sim_res,
        "classical": classical, "cfp_topo": cfp_topo, "cfp_demo": cfp_demo,
        "horse": horse, "shuffle_topo": shuffle_topo, "shuffle_demo": shuffle_demo,
        "plants": plants, "localization": localization, "residual_corr": residual_corr,
        "strip_confound": strip_confound,
    }

def run_state_classical_only(state_name: str, n_precincts: int, n_districts: int, seed_base: int, quick: bool) -> dict:

    steps = 800 if quick else cfg.CHAIN_STEPS
    burn = 100 if quick else cfg.CHAIN_BURN_IN
    thin = 6 if quick else cfg.CHAIN_THIN

    rng_state = cfg.seed_everything(seed_base)
    log(f"[{state_name}] generating synthetic precinct universe (n={n_precincts}, k={n_districts})")
    state = generate_synthetic_state(state_name, n_precincts, n_districts, rng_state)

    rng_a = np.random.default_rng(seed_base + 1)
    log(f"[{state_name}] running training chain A")
    chain_a = run_chain(state.graph, state.population, n_districts, cfg.POP_EPSILON, rng_a, steps, burn, thin)

    rng_b = np.random.default_rng(seed_base + 2)
    log(f"[{state_name}] running evaluation chain B (independent seed)")
    chain_b = run_chain(state.graph, state.population, n_districts, cfg.POP_EPSILON, rng_b, steps, burn, thin)
    log(f"[{state_name}]   chain A size = {len(chain_a)}  chain B size = {len(chain_b)}")

    log(f"[{state_name}] training topology-only variant (for CFP reproduction)")
    topo_bundle = exp.fit_variant(state, chain_a, include_demo=False, seed=seed_base + 10)

    rng_eval = np.random.default_rng(seed_base + 20)
    log(f"[{state_name}] building SIM-ENACTED plan")
    sim_res = build_sim_enacted(state, chain_b, cfg.POP_EPSILON, rng_eval, n_iters=100 if quick else 400, candidates_per_iter=6)
    sim_assignment = sim_res["assignment"]

    classical = exp.classical_baseline_reproduction(state, chain_b, sim_assignment, rng_eval)
    cfp_topo = exp.cfp_outlier_test(state, chain_b, topo_bundle, sim_assignment)

    return {"state": state, "chain_b": chain_b, "sim_res": sim_res, "sim_assignment": sim_assignment,
            "classical": classical, "cfp_topo": cfp_topo}

def build_tables(alpha: dict, beta: dict, out_dir: str) -> dict:
    state_rows = [(alpha["state"].name, alpha)]
    if beta is not None:
        state_rows.append((beta["state"].name, beta))
    rows1 = []
    for name, r in state_rows:
        c = r["classical"]
        rows1.append({
            "state": name, "statistic": "mean_median", "sim_enacted_value": c["mean_median_test"],
            "extreme_percentile": c["mean_median_extreme_percentile"], "threshold_pct": c["threshold_percentile"],
            "pass": c["pass_mean_median"],
        })
        rows1.append({
            "state": name, "statistic": "efficiency_gap", "sim_enacted_value": c["efficiency_gap_test"],
            "extreme_percentile": c["efficiency_gap_extreme_percentile"], "threshold_pct": c["threshold_percentile"],
            "pass": c["pass_efficiency_gap"],
        })
        rows1.append({
            "state": name, "statistic": "declination", "sim_enacted_value": c["declination_test"],
            "extreme_percentile": c["declination_extreme_percentile"], "threshold_pct": c["threshold_percentile"],
            "pass": c["pass_declination"],
        })
        rows1.append({
            "state": name, "statistic": "partition_distance", "sim_enacted_value": c["partition_distance_test"],
            "extreme_percentile": c["partition_distance_percentile"], "threshold_pct": c["threshold_percentile"],
            "pass": c["pass_partition_distance"],
        })
    table1 = pd.DataFrame(rows1)
    table1.to_csv(os.path.join(out_dir, "table1_classical_reproduction.csv"), index=False)

    rows2 = []
    for name, r in state_rows:
        for variant, cfp in [("topology_only", r["cfp_topo"])] + ([("plus_demographic", r["cfp_demo"])] if "cfp_demo" in r else []):
            rows2.append({
                "state": name, "variant": variant, "eps": cfp["eps"], "p_value": cfp["p_value"],
                "extreme_percentile": cfp["extreme_percentile"], "threshold_pct": cfp["threshold_percentile"],
                "pass": cfp["pass"],
            })
    table2 = pd.DataFrame(rows2)
    table2.to_csv(os.path.join(out_dir, "table2_cfp_outlier_test.csv"), index=False)

    horse = alpha["horse"]
    table3 = pd.DataFrame([{
        "state": "STATE_ALPHA", "spearman_rho": horse["spearman_rho"], "spearman_p": horse["spearman_p"],
        "spearman_ci_lo": horse["spearman_ci_lo"], "spearman_ci_hi": horse["spearman_ci_hi"],
        "kendall_tau": horse["kendall_tau"], "kendall_p": horse["kendall_p"],
        "threshold": horse["threshold"], "pass_geometry_sufficient": horse["pass"],
    }])
    table3.to_csv(os.path.join(out_dir, "table3_confound_isolation_rank_correlation.csv"), index=False)

    rows4 = []
    for variant, sh in [("topology_only", alpha["shuffle_topo"]), ("plus_demographic", alpha["shuffle_demo"])]:
        rows4.append({
            "state": "STATE_ALPHA", "variant": variant, "base_score": sh["base_score"],
            "mean_abs_delta": sh["mean_abs_delta"], "ensemble_spread": sh["ensemble_spread"],
            "ratio": sh["ratio"], "ratio_std": sh["ratio_std"], "threshold": sh["threshold"], "pass_invariant": sh["pass"],
        })
    table4 = pd.DataFrame(rows4)
    table4.to_csv(os.path.join(out_dir, "table4_vote_share_shuffle_invariance.csv"), index=False)

    rows5 = []
    for variant_name, per_variant in alpha["localization"].items():
        for condition, block in per_variant.items():
            summary = block["loc"]["summary"]
            comparisons = block["comparisons"]
            for null_name in ["degree", "density", "random"]:
                comp = comparisons[null_name]
                rows5.append({
                    "state": "STATE_ALPHA", "variant": variant_name, "condition": condition, "null": null_name,
                    "n_plants": block["loc"]["n_plants"],
                    "gnn_mean_auc": summary["gnn"]["mean_auc"], "gnn_std_auc": summary["gnn"]["std_auc"],
                    "null_mean_auc": summary[null_name]["mean_auc"],
                    "gnn_prec_at_true_k": summary["gnn"]["mean_prec_true_k"],
                    "null_prec_at_true_k": summary[null_name]["mean_prec_true_k"],
                    "permutation_p": comp["permutation_p"], "dm_stat": comp["dm_stat"], "dm_p": comp["dm_p"],
                    "pass_localization": comp["pass"],
                })
    table5 = pd.DataFrame(rows5)
    table5.to_csv(os.path.join(out_dir, "table5_node_localization.csv"), index=False)

    rows6 = []
    for key, v in alpha["residual_corr"].items():
        tname, dname = key.split("__")
        rows6.append({"state": "STATE_ALPHA", "topology_feature": tname, "demographic_covariate": dname, "spearman_rho": v["rho"], "p_value": v["p"]})
    table6 = pd.DataFrame(rows6).sort_values("spearman_rho", key=lambda s: -s.abs())
    table6.to_csv(os.path.join(out_dir, "table6_residual_density_confound.csv"), index=False)

    return {"table1": table1, "table2": table2, "table3": table3, "table4": table4, "table5": table5, "table6": table6}

def build_figures(alpha: dict, beta: dict, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    eg_ens = alpha["classical"]["efficiency_gap_ensemble"]
    axes[0].hist(eg_ens, bins=25, color="#4C72B0", alpha=0.8)
    axes[0].axvline(alpha["classical"]["efficiency_gap_test"], color="#C44E52", lw=2, label="SIM-ENACTED")
    axes[0].set_title("Efficiency gap: ensemble vs SIM-ENACTED")
    axes[0].set_xlabel("Efficiency gap")
    axes[0].legend(fontsize=7)

    gnn_ens = alpha["cfp_topo"]["ensemble_scores"]
    axes[1].hist(gnn_ens, bins=25, color="#55A868", alpha=0.8)
    axes[1].axvline(alpha["cfp_topo"]["test_score"], color="#C44E52", lw=2, label="SIM-ENACTED")
    axes[1].set_title("GNN score (topology-only): ensemble vs SIM-ENACTED")
    axes[1].set_xlabel("GNN anomaly score")
    axes[1].legend(fontsize=7)
    fig.suptitle("STATE_ALPHA (synthetic) -- ensemble outlier reproduction", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_ensemble_outlier_reproduction.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.scatter(alpha["horse"]["topo_scores"], alpha["horse"]["demo_scores"], s=14, alpha=0.6, color="#4C72B0")
    ax.set_xlabel("Topology-only score")
    ax.set_ylabel("+Demographic score")
    ax.set_title(f"Rank agreement (Spearman rho={alpha['horse']['spearman_rho']:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_topo_vs_demo_scatter.png"), bbox_inches="tight")
    plt.close(fig)

    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    block = alpha["localization"]["plus_demographic"]["decorrelated"]
    plants = alpha["plants"]["decorrelated"]
    y_all, gnn_all, deg_all, dens_all, rand_all = [], [], [], [], []
    for i, plant in enumerate(plants):
        y = np.zeros(alpha["state"].n_precincts)
        for node in plant.moved_nodes:
            y[node] = 1.0
        y_all.append(y)
        gnn_all.append(block["loc"]["per_plan"]["gnn"]["auc"][i])
    for method, color in [("gnn", "#C44E52"), ("degree", "#8172B2"), ("density", "#CCB974"), ("random", "#999999")]:
        aucs = np.nan_to_num(block["loc"]["summary"][method]["auc_series"], nan=0.5)
        ax.plot([0, 1], [0, 1], color, alpha=0)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    bar_x = np.arange(4)
    methods = ["gnn", "degree", "density", "random"]
    means = [block["loc"]["summary"][m]["mean_auc"] for m in methods]
    ax.clear()
    ax.bar(bar_x, means, color=["#C44E52", "#8172B2", "#CCB974", "#999999"])
    ax.axhline(0.5, color="k", linestyle="--", lw=1)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(["GNN", "degree null", "density null", "random null"], rotation=20, ha="right")
    ax.set_ylabel("Mean localization AUC")
    ax.set_title("Node localization, decorrelated set\n(+demographic variant)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_localization_auc_decorrelated.png"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, condition in zip(axes, ["decorrelated", "correlated"]):
        width = 0.35
        x = np.arange(4)
        for offset, (variant, color) in zip([-width / 2, width / 2], [("topology_only", "#4C72B0"), ("plus_demographic", "#C44E52")]):
            block = alpha["localization"][variant][condition]
            vals = [block["loc"]["summary"][m]["mean_prec_true_k"] for m in ["gnn", "degree", "density", "random"]]
            ax.bar(x + offset, vals, width=width, label=variant, color=color, alpha=0.5 if variant == "topology_only" else 0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(["GNN", "degree", "density", "random"], rotation=20, ha="right")
        ax.set_title(condition)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Precision@|true edit|")
    axes[0].legend(fontsize=7)
    fig.suptitle("Node localization precision@k", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_precision_at_k.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.hist(alpha["shuffle_topo"]["deltas"], bins=25, alpha=0.6, label="topology-only", color="#4C72B0", density=True)
    ax.hist(alpha["shuffle_demo"]["deltas"], bins=25, alpha=0.6, label="+demographic", color="#C44E52", density=True)
    ax.set_xlabel("|score delta| under partisan-label shuffle")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_title("Vote-share shuffle invariance, SIM-ENACTED")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_shuffle_invariance.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    rc = alpha["residual_corr"]
    labels = list(rc.keys())
    rhos = [rc[k]["rho"] for k in labels]
    order = np.argsort(np.abs(rhos))[::-1]
    labels = [labels[i].replace("__", " vs ") for i in order]
    rhos = [rhos[i] for i in order]
    colors = ["#C44E52" if abs(r) >= 0.3 else "#4C72B0" for r in rhos]
    ax.barh(range(len(labels)), rhos, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Spearman rho")
    ax.set_title("Residual threat: topology features vs demographic covariates")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_residual_density_confound.png"), bbox_inches="tight")
    plt.close(fig)

def print_summary(alpha: dict, beta: dict, tables: dict) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY (all numbers from SYNTHETIC STATE_ALPHA / STATE_BETA testbeds)")
    print("=" * 78)
    print("\n--- Table 1: classical baseline reproduction ---")
    print(tables["table1"].to_string(index=False))
    print("\n--- Table 2: CFP epsilon-outlier test ---")
    print(tables["table2"].to_string(index=False))
    print("\n--- Table 3: confound-isolation rank correlation ---")
    print(tables["table3"].to_string(index=False))
    print("\n--- Table 4: vote-share shuffle invariance ---")
    print(tables["table4"].to_string(index=False))
    print("\n--- Table 5: node-level localization (STATE_ALPHA) ---")
    print(tables["table5"].to_string(index=False))
    print("\n--- Table 6: residual density-confound correlations (top 6) ---")
    print(tables["table6"].head(6).to_string(index=False))

    print("\n--- PASS/FAIL against design thresholds ---")
    a_name = alpha["state"].name
    c = alpha["classical"]
    print(f"Classical reproduction ({a_name}):  MM {'PASS' if c['pass_mean_median'] else 'FAIL'} | EG {'PASS' if c['pass_efficiency_gap'] else 'FAIL'} | partition-dist {'PASS' if c['pass_partition_distance'] else 'FAIL'}")
    if beta is not None:
        cb = beta["classical"]
        print(f"Classical reproduction ({beta['state'].name}):   MM {'PASS' if cb['pass_mean_median'] else 'FAIL'} | EG {'PASS' if cb['pass_efficiency_gap'] else 'FAIL'}")
    print(f"CFP test topology-only ({a_name}):  {'PASS' if alpha['cfp_topo']['pass'] else 'FAIL'} (p={alpha['cfp_topo']['p_value']:.4f})")
    print(f"CFP test +demographic (ALPHA):   {'PASS' if alpha['cfp_demo']['pass'] else 'FAIL'} (p={alpha['cfp_demo']['p_value']:.4f})")
    print(f"Confound isolation (geometry sufficient): {'YES' if alpha['horse']['pass'] else 'NO'} (rho={alpha['horse']['spearman_rho']:.3f}, threshold={alpha['horse']['threshold']})")
    print(f"Shuffle invariance topology-only: {'PASS (invariant)' if alpha['shuffle_topo']['pass'] else 'FAIL (sensitive)'} (ratio={alpha['shuffle_topo']['ratio']:.3f} +/- {alpha['shuffle_topo']['ratio_std']:.3f})")
    print(f"Shuffle invariance +demographic:  {'PASS (invariant)' if alpha['shuffle_demo']['pass'] else 'FAIL (sensitive)'} (ratio={alpha['shuffle_demo']['ratio']:.3f} +/- {alpha['shuffle_demo']['ratio_std']:.3f})")
    for variant_name, per_variant in alpha["localization"].items():
        for condition, block in per_variant.items():
            comp = block["comparisons"]["degree"]
            print(f"Localization[{variant_name}][{condition}] vs degree-null: {'PASS' if comp['pass'] else 'FAIL'} (AUC {comp['gnn_mean_auc']:.3f} vs {comp['null_mean_auc']:.3f}, perm p={comp['permutation_p']:.4f})")

def main():

    torch.set_num_threads(min(10, os.cpu_count() or 4))

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fast, small-scale correctness check")
    parser.add_argument("--real", choices=["PA", "NC", "MD"], help="run on a real MGGG-states VTD map with its adjudicated enacted plan")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out or (f"outputs_real_{args.real}" if args.real else "outputs")
    tables_dir = os.path.join(out, "tables")
    figures_dir = os.path.join(out, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    t0 = time.time()

    if args.real:
        from data_adapter import load_real_state
        log(f"loading real state {args.real} (shapefile -> repaired dual graph); first load is slow")
        real_state, enacted = load_real_state(args.real)
        alpha = run_state_full(f"{args.real}_real", real_state.n_precincts, real_state.n_districts,
                               seed_base=cfg.GLOBAL_SEED, quick=args.quick, real=(real_state, enacted))
        beta = None
    else:
        n_a = 120 if args.quick else cfg.N_PRECINCTS_ALPHA
        k_a = 6 if args.quick else cfg.N_DISTRICTS_ALPHA
        n_b = 100 if args.quick else cfg.N_PRECINCTS_BETA
        k_b = 5 if args.quick else cfg.N_DISTRICTS_BETA
        alpha = run_state_full("STATE_ALPHA", n_a, k_a, seed_base=cfg.GLOBAL_SEED, quick=args.quick)
        beta = run_state_classical_only("STATE_BETA", n_b, k_b, seed_base=cfg.GLOBAL_SEED + 1000, quick=args.quick)

    tables = build_tables(alpha, beta, tables_dir)
    build_figures(alpha, beta, figures_dir)
    print_summary(alpha, beta, tables)

    log(f"total runtime {time.time() - t0:.1f}s")
    log(f"tables written to {tables_dir}")
    log(f"figures written to {figures_dir}")

if __name__ == "__main__":
    main()

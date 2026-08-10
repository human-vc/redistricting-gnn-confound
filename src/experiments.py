from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import baselines
import config as cfg
import stats_tests as st
from features import Standardizer, assemble_features
from gnn_model import GLocalKDScorer, MomentCalibrator, build_sparse_adjacency, train_scorer
from planted import PlantedEdit
from synthetic_state import SyntheticState

@dataclass
class ModelBundle:
    model: GLocalKDScorer
    standardizer: Standardizer
    calibrator: MomentCalibrator
    A: torch.Tensor
    include_demo: bool
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    history: dict

def _feature_stack(state: SyntheticState, assignments: list[np.ndarray], include_demo: bool) -> np.ndarray:
    return np.stack([assemble_features(state, a, include_demo) for a in assignments])

def fit_variant(
    state: SyntheticState,
    ensemble: list[np.ndarray],
    include_demo: bool,
    seed: int,
    verbose: bool = False,
) -> ModelBundle:
    rng = np.random.default_rng(seed)
    n = len(ensemble)
    perm = rng.permutation(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    train_idx, val_idx, test_idx = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]

    feats = _feature_stack(state, ensemble, include_demo)
    fit_idx = np.concatenate([train_idx, val_idx])
    standardizer = Standardizer().fit(feats[fit_idx])
    feats_std = standardizer.transform(feats)
    X = torch.tensor(feats_std, dtype=torch.float32)

    A = build_sparse_adjacency(state.graph, state.n_precincts)
    torch.manual_seed(seed)
    model = GLocalKDScorer(in_dim=feats.shape[-1], hidden=cfg.GNN_HIDDEN, n_layers=cfg.GNN_LAYERS)
    history = train_scorer(
        model, A, X[train_idx], X[val_idx],
        epochs=cfg.GNN_EPOCHS, lr=cfg.GNN_LR, weight_decay=cfg.GNN_WEIGHT_DECAY,
        patience=cfg.GNN_PATIENCE, verbose=verbose,
    )

    model.eval()
    with torch.no_grad():
        disc_calib = model.node_discrepancy(X[np.concatenate([train_idx, val_idx])], A)
    calibrator = MomentCalibrator().fit(disc_calib)

    return ModelBundle(model=model, standardizer=standardizer, calibrator=calibrator, A=A,
                        include_demo=include_demo, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                        history=history)

def score_assignments(bundle: ModelBundle, state: SyntheticState, assignments: list[np.ndarray], dem_share_override: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:

    feats = np.stack([
        assemble_features(state, a, bundle.include_demo, dem_share_override=dem_share_override) for a in assignments
    ])
    feats_std = bundle.standardizer.transform(feats)
    X = torch.tensor(feats_std, dtype=torch.float32)
    bundle.model.eval()
    with torch.no_grad():
        disc = bundle.model.node_discrepancy(X, bundle.A)
    scores = bundle.calibrator.score(disc)
    return scores, disc.numpy()

def score_single(bundle: ModelBundle, state: SyntheticState, assignment: np.ndarray, dem_share_override: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    scores, disc = score_assignments(bundle, state, [assignment], dem_share_override=dem_share_override)
    return float(scores[0]), disc[0]

def confound_isolation_horse_race(
    state: SyntheticState, chain_b: list[np.ndarray], topo_bundle: ModelBundle, demo_bundle: ModelBundle,
    rng: np.random.Generator, n_boot: int,
) -> dict:
    topo_scores, _ = score_assignments(topo_bundle, state, chain_b)
    demo_scores, _ = score_assignments(demo_bundle, state, chain_b)

    corr = st.rank_correlations(topo_scores, demo_scores)

    def spearman_stat(x, y):
        rho, _ = spearmanr(x, y)
        return rho

    boot = st.bootstrap_ci((topo_scores, demo_scores), spearman_stat, n_boot, rng)

    passed = corr["spearman_rho"] >= cfg.THRESH_SPEARMAN_HIGH
    return {
        "topo_scores": topo_scores, "demo_scores": demo_scores, **corr,
        "spearman_ci_lo": boot["ci_lo"], "spearman_ci_hi": boot["ci_hi"],
        "threshold": cfg.THRESH_SPEARMAN_HIGH, "pass": bool(passed),
    }

def vote_share_shuffle_invariance(
    state: SyntheticState, sim_enacted_assignment: np.ndarray, bundle: ModelBundle,
    ensemble_scores: np.ndarray, n_shuffles: int, rng: np.random.Generator,
) -> dict:
    base_score, _ = score_single(bundle, state, sim_enacted_assignment)
    deltas = np.empty(n_shuffles)
    for i in range(n_shuffles):
        shuffled = rng.permutation(state.dem_share)
        s, _ = score_single(bundle, state, sim_enacted_assignment, dem_share_override=shuffled if bundle.include_demo else None)
        deltas[i] = abs(s - base_score)

    spread = float(np.std(ensemble_scores))
    mean_abs_delta = float(np.mean(deltas))
    ratio = mean_abs_delta / spread if spread > 0 else float("inf")
    return {
        "base_score": base_score, "deltas": deltas, "mean_abs_delta": mean_abs_delta,
        "ensemble_spread": spread, "ratio": ratio, "threshold": cfg.THRESH_SHUFFLE_RATIO_SMALL,
        "pass_invariant": bool(ratio <= cfg.THRESH_SHUFFLE_RATIO_SMALL),
    }

def _precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)
    top_k = order[:k]
    return float(y_true[top_k].sum() / k)

def evaluate_localization_set(
    state: SyntheticState, plants: list[PlantedEdit], bundles: list[ModelBundle] | ModelBundle,
    null_scores: dict[str, np.ndarray], rng: np.random.Generator, fixed_k: int = 10,
) -> dict:

    if isinstance(bundles, ModelBundle):
        bundles = [bundles]
    methods = ["gnn"] + list(null_scores.keys()) + ["random"]
    per_plan = {m: {"auc": [], "prec_at_true_k": [], "prec_at_fixed_k": []} for m in methods}

    for plant in plants:
        y = np.zeros(state.n_precincts)
        for node in plant.moved_nodes:
            y[node] = 1.0
        k_true = max(1, len(plant.moved_nodes))
        k_fixed = min(fixed_k, state.n_precincts)
        valid = 0 < y.sum() < len(y)
        rand_scores = baselines.random_null_scores(state.n_precincts, rng)

        seed_vals = {m: {"auc": [], "ptrue": [], "pfixed": []} for m in methods}
        for bundle in bundles:
            _, disc = score_single(bundle, state, plant.new_assignment)
            scores_by_method = dict(null_scores)
            scores_by_method["random"] = rand_scores
            scores_by_method["gnn"] = disc
            for m, sc in scores_by_method.items():
                seed_vals[m]["auc"].append(roc_auc_score(y, sc) if valid else np.nan)
                seed_vals[m]["ptrue"].append(_precision_at_k(y, sc, k_true))
                seed_vals[m]["pfixed"].append(_precision_at_k(y, sc, k_fixed))

        for m in methods:
            per_plan[m]["auc"].append(float(np.nanmean(seed_vals[m]["auc"])) if valid else np.nan)
            per_plan[m]["prec_at_true_k"].append(float(np.mean(seed_vals[m]["ptrue"])))
            per_plan[m]["prec_at_fixed_k"].append(float(np.mean(seed_vals[m]["pfixed"])))

    summary = {}
    for m in methods:
        aucs = np.array(per_plan[m]["auc"])
        aucs_valid = aucs[~np.isnan(aucs)]
        prec_true = per_plan[m]["prec_at_true_k"]
        prec_fixed = per_plan[m]["prec_at_fixed_k"]
        summary[m] = {
            "mean_auc": float(np.mean(aucs_valid)) if len(aucs_valid) else float("nan"),
            "std_auc": float(np.std(aucs_valid)) if len(aucs_valid) else float("nan"),
            "mean_prec_true_k": float(np.mean(prec_true)) if prec_true else float("nan"),
            "mean_prec_fixed_k": float(np.mean(prec_fixed)) if prec_fixed else float("nan"),
            "auc_series": aucs,
        }
    return {"per_plan": per_plan, "summary": summary, "n_plants": len(plants), "n_seeds": len(bundles), "n_obs": len(plants)}

def compare_gnn_vs_null(loc_result: dict, null_name: str, n_perm: int, rng: np.random.Generator) -> dict:
    gnn_auc = np.nan_to_num(loc_result["summary"]["gnn"]["auc_series"], nan=0.5)
    null_auc = np.nan_to_num(loc_result["summary"][null_name]["auc_series"], nan=0.5)
    if len(gnn_auc) < 2:
        return {
            "gnn_mean_auc": float("nan"), "null_mean_auc": float("nan"),
            "permutation_p": float("nan"), "observed_diff": float("nan"),
            "dm_stat": float("nan"), "dm_p": float("nan"), "pass": False,
            "note": "fewer than 2 planted plans available; comparison skipped",
        }
    perm = st.paired_permutation_test(gnn_auc, null_auc, n_perm, rng, alternative="greater")
    dm = st.diebold_mariano(1.0 - gnn_auc, 1.0 - null_auc, h=1)
    return {
        "gnn_mean_auc": float(np.mean(gnn_auc)), "null_mean_auc": float(np.mean(null_auc)),
        "permutation_p": perm["p_value"], "observed_diff": perm["observed_diff"],
        "dm_stat": dm["dm_stat_adj"], "dm_p": dm["p_value"],
        "pass": bool(perm["p_value"] < cfg.THRESH_LOCALIZATION_ALPHA and perm["observed_diff"] > 0),
    }

def classical_baseline_reproduction(
    state: SyntheticState,
    chain_b: list[np.ndarray],
    sim_enacted_assignment: np.ndarray,
    rng: np.random.Generator | None = None,
) -> dict:
    shares_ensemble = [baselines.district_dem_shares(state, a) for a in chain_b]
    shares_test = baselines.district_dem_shares(state, sim_enacted_assignment)
    mm_ensemble = [baselines.mean_median(s) for s in shares_ensemble]
    eg_ensemble = [baselines.efficiency_gap(state, a) for a in chain_b]
    dec_ensemble = [baselines.declination(s) for s in shares_ensemble]
    mm_test = baselines.mean_median(shares_test)
    eg_test = baselines.efficiency_gap(state, sim_enacted_assignment)
    dec_test = baselines.declination(shares_test)

    mm_pct = st.percentile_rank(mm_ensemble, mm_test)
    eg_pct = st.percentile_rank(eg_ensemble, eg_test)
    dec_valid = [d for d in dec_ensemble if not np.isnan(d)]
    dec_pct = st.percentile_rank(dec_valid, dec_test) if dec_valid and not np.isnan(dec_test) else float("nan")
    thresh_pct = cfg.THRESH_CFP_PERCENTILE * 100
    mm_extreme_pct = max(mm_pct, 100 - mm_pct)
    eg_extreme_pct = max(eg_pct, 100 - eg_pct)
    dec_extreme_pct = max(dec_pct, 100 - dec_pct) if not np.isnan(dec_pct) else float("nan")

    ce_ensemble = [baselines.cut_edges(state, a) for a in chain_b]
    pp_ensemble = [baselines.mean_polsby_popper(state, a) for a in chain_b]
    ce_test = baselines.cut_edges(state, sim_enacted_assignment)
    pp_test = baselines.mean_polsby_popper(state, sim_enacted_assignment)
    ce_pct = st.percentile_rank(ce_ensemble, ce_test)
    pp_pct = st.percentile_rank(pp_ensemble, pp_test)
    ce_extreme_pct = max(ce_pct, 100.0 - ce_pct)
    pp_extreme_pct = max(pp_pct, 100.0 - pp_pct)

    if rng is None:
        rng = np.random.default_rng(cfg.GLOBAL_SEED)
    k = state.n_districts
    n_pairs = min(len(chain_b) * 4, 800)
    pdist_ensemble = baselines.ensemble_internal_distance_distribution(chain_b, state.population, k, n_pairs, rng)
    pdist_test = baselines.mean_partition_distance_to_ensemble(
        sim_enacted_assignment, chain_b, state.population, k, sample_size=min(len(chain_b), 200), rng=rng
    )
    pdist_pct = st.percentile_rank(pdist_ensemble, pdist_test)

    return {
        "mean_median_ensemble": mm_ensemble, "efficiency_gap_ensemble": eg_ensemble,
        "mean_median_test": mm_test, "efficiency_gap_test": eg_test,
        "mean_median_percentile": mm_pct, "efficiency_gap_percentile": eg_pct,
        "mean_median_extreme_percentile": mm_extreme_pct, "efficiency_gap_extreme_percentile": eg_extreme_pct,
        "declination_ensemble": dec_ensemble, "declination_test": dec_test,
        "declination_percentile": dec_pct, "declination_extreme_percentile": dec_extreme_pct,
        "partition_distance_ensemble": pdist_ensemble, "partition_distance_test": pdist_test,
        "partition_distance_percentile": pdist_pct,
        "threshold_percentile": thresh_pct,
        "pass_mean_median": bool(mm_extreme_pct >= thresh_pct),
        "pass_efficiency_gap": bool(eg_extreme_pct >= thresh_pct),
        "pass_declination": bool(dec_extreme_pct >= thresh_pct) if not np.isnan(dec_extreme_pct) else False,
        "pass_partition_distance": bool(pdist_pct >= thresh_pct),
    }

def cfp_outlier_test(state: SyntheticState, chain_b: list[np.ndarray], bundle: ModelBundle, sim_enacted_assignment: np.ndarray) -> dict:
    ensemble_scores, _ = score_assignments(bundle, state, chain_b)
    test_score, _ = score_single(bundle, state, sim_enacted_assignment)
    result = st.cfp_epsilon_test(ensemble_scores, test_score)
    thresh_pct = cfg.THRESH_CFP_PERCENTILE * 100
    eps_threshold = 1.0 - cfg.THRESH_CFP_PERCENTILE
    extreme_pct = 100.0 * (1.0 - result.eps)
    return {
        "ensemble_scores": ensemble_scores, "test_score": test_score,
        "eps": result.eps, "p_value": result.p_value, "percentile": result.percentile,
        "extreme_percentile": extreme_pct, "threshold_percentile": thresh_pct,
        "pass": bool(result.eps <= eps_threshold),
    }

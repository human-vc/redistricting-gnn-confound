from __future__ import annotations

import numpy as np

from baselines import district_dem_shares, efficiency_gap, mean_median
from ensemble import recom_step
from synthetic_state import SyntheticState

def _joint_partisan_objective(state: SyntheticState, assignment: np.ndarray, eg_scale: float, mm_scale: float) -> float:

    eg = efficiency_gap(state, assignment)
    mm = mean_median(district_dem_shares(state, assignment))
    return eg / eg_scale + mm / mm_scale

def build_sim_enacted(
    state: SyntheticState,
    ensemble: list[np.ndarray],
    epsilon: float,
    rng: np.random.Generator,
    n_iters: int = 400,
    candidates_per_iter: int = 8,
    target_direction: int = 1,
    sideways_accept_prob: float = 0.05,
    verbose: bool = False,
) -> dict:
    eg_ensemble = np.array([efficiency_gap(state, a) for a in ensemble])
    mm_ensemble = np.array([mean_median(district_dem_shares(state, a)) for a in ensemble])
    eg_scale = max(float(np.std(eg_ensemble)), 1e-6)
    mm_scale = max(float(np.std(mm_ensemble)), 1e-6)

    def obj(assignment):
        return target_direction * _joint_partisan_objective(state, assignment, eg_scale, mm_scale)

    current = ensemble[int(rng.integers(len(ensemble)))].copy()
    current_score = obj(current)
    best_assignment = current.copy()
    best_score = current_score

    for it in range(n_iters):
        candidates = []
        for _ in range(candidates_per_iter):
            new_assign, ok = recom_step(state.graph, current, state.population, state.n_districts, epsilon, rng)
            if ok:
                candidates.append(new_assign)
        if not candidates:
            continue
        scored = [(obj(c), c) for c in candidates]
        scored.sort(key=lambda x: -x[0])
        top_score, top_assign = scored[0]

        if top_score >= current_score or rng.random() < sideways_accept_prob:
            current = top_assign
            current_score = top_score
        if current_score > best_score:
            best_score = current_score
            best_assignment = current.copy()

        if verbose and (it % 50 == 0 or it == n_iters - 1):
            eg_now = efficiency_gap(state, best_assignment)
            mm_now = mean_median(district_dem_shares(state, best_assignment))
            print(f"    sim-enacted search it {it:3d}  best_joint={best_score:.4f}  EG={eg_now:.4f}  MM={mm_now:.4f}")

    return {
        "assignment": best_assignment,
        "efficiency_gap": efficiency_gap(state, best_assignment),
        "mean_median": mean_median(district_dem_shares(state, best_assignment)),
        "target_direction": target_direction,
        "n_iters": n_iters,
    }

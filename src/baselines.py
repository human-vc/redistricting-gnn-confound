from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from synthetic_state import SyntheticState

def district_dem_shares(state: SyntheticState, assignment: np.ndarray) -> np.ndarray:
    dem_votes = state.dem_share * state.total_votes
    shares = np.zeros(state.n_districts)
    for d in range(state.n_districts):
        mask = assignment == d
        total = state.total_votes[mask].sum()
        shares[d] = dem_votes[mask].sum() / total if total > 0 else 0.5
    return shares

def cut_edges(state: SyntheticState, assignment: np.ndarray) -> int:

    a = np.asarray(assignment)
    return int(sum(1 for u, v in state.graph.edges() if a[u] != a[v]))

def mean_polsby_popper(state: SyntheticState, assignment: np.ndarray) -> float:

    a = np.asarray(assignment)
    k = state.n_districts
    area = np.bincount(a, weights=state.area, minlength=k)
    perim = np.array([state.perimeter[a == d].sum() for d in range(k)], dtype=float)
    for u, v in state.graph.edges():
        if a[u] == a[v]:
            perim[a[u]] -= 2.0 * state.shared_boundary.get((u, v) if u < v else (v, u), 0.0)
    perim = np.maximum(perim, 1e-9)
    return float(np.mean(4.0 * np.pi * area / (perim ** 2)))

def mean_median(district_shares: np.ndarray) -> float:
    return float(np.median(district_shares) - np.mean(district_shares))

def declination(district_shares: np.ndarray) -> float:

    shares = np.sort(np.asarray(district_shares, dtype=float))
    n = len(shares)
    lose = shares[shares < 0.5]
    win = shares[shares >= 0.5]
    if n == 0 or len(lose) == 0 or len(win) == 0:
        return float("nan")
    theta = np.arctan((0.5 - lose.mean()) / (len(lose) / (2.0 * n)))
    gamma = np.arctan((win.mean() - 0.5) / (len(win) / (2.0 * n)))
    return float(2.0 * (gamma - theta) / np.pi)

def efficiency_gap(state: SyntheticState, assignment: np.ndarray) -> float:
    total_votes_cast = 0.0
    wasted_dem_total = 0.0
    wasted_rep_total = 0.0
    for d in range(state.n_districts):
        mask = assignment == d
        T = state.total_votes[mask].sum()
        if T <= 0:
            continue
        D = (state.dem_share[mask] * state.total_votes[mask]).sum()
        R = T - D
        if D > R:
            wasted_dem = D - T / 2.0
            wasted_rep = R
        else:
            wasted_rep = R - T / 2.0
            wasted_dem = D
        wasted_dem_total += wasted_dem
        wasted_rep_total += wasted_rep
        total_votes_cast += T
    return float((wasted_rep_total - wasted_dem_total) / total_votes_cast)

def overlap_matrix(assignment_a: np.ndarray, assignment_b: np.ndarray, population: np.ndarray, k: int) -> np.ndarray:
    M = np.zeros((k, k))
    for i in range(k):
        mask_i = assignment_a == i
        if not mask_i.any():
            continue
        for j in range(k):
            mask_ij = mask_i & (assignment_b == j)
            if mask_ij.any():
                M[i, j] = population[mask_ij].sum()
    return M

def partition_distance(assignment_a: np.ndarray, assignment_b: np.ndarray, population: np.ndarray, k: int) -> float:

    M = overlap_matrix(assignment_a, assignment_b, population, k)
    row_idx, col_idx = linear_sum_assignment(-M)
    matched = M[row_idx, col_idx].sum()
    total = population.sum()
    return float(1.0 - matched / total)

def mean_partition_distance_to_ensemble(
    test_assignment: np.ndarray, ensemble: list[np.ndarray], population: np.ndarray, k: int, sample_size: int | None = None, rng=None
) -> float:
    idxs = range(len(ensemble))
    if sample_size is not None and sample_size < len(ensemble):
        idxs = rng.choice(len(ensemble), size=sample_size, replace=False)
    dists = [partition_distance(test_assignment, ensemble[i], population, k) for i in idxs]
    return float(np.mean(dists))

def ensemble_internal_distance_distribution(
    ensemble: list[np.ndarray], population: np.ndarray, k: int, n_pairs: int, rng
) -> np.ndarray:

    n = len(ensemble)
    dists = np.empty(n_pairs)
    for t in range(n_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        dists[t] = partition_distance(ensemble[i], ensemble[j], population, k)
    return dists

def degree_null_scores(state: SyntheticState) -> np.ndarray:
    return np.array([state.graph.degree(i) for i in range(state.n_precincts)], dtype=float)

def density_null_scores(state: SyntheticState) -> np.ndarray:
    return state.density.copy()

def random_null_scores(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random(n)

from __future__ import annotations

import numpy as np

from synthetic_state import SyntheticState

TOPOLOGY_FEATURE_NAMES = [
    "cut_edge_indicator",
    "boundary_fraction",
    "boundary_compactness",
    "district_pop_share",
    "area",
    "perimeter",
    "degree",
]

DEMOGRAPHIC_FEATURE_NAMES = [
    "vap_minority_share",
    "dem_share",
    "population_density",
]

def _edge_arrays(state: SyntheticState):

    g = state.graph
    cache = g.graph.get("_feat_edge_cache")
    if cache is not None and cache[0] == state.n_precincts:
        return cache[1]
    n = state.n_precincts
    edges = list(g.edges())
    eu = np.fromiter((u for u, _ in edges), dtype=np.int64, count=len(edges))
    ev = np.fromiter((v for _, v in edges), dtype=np.int64, count=len(edges))
    L = np.array([state.shared_boundary.get((min(u, v), max(u, v)), 0.0) for u, v in edges], dtype=float)
    degree = np.array([g.degree(i) for i in range(n)], dtype=float)
    arrays = (eu, ev, L, degree)
    g.graph["_feat_edge_cache"] = (n, arrays)
    return arrays

def compute_topology_features(state: SyntheticState, assignment: np.ndarray) -> np.ndarray:
    n = state.n_precincts
    eu, ev, L, degree = _edge_arrays(state)
    assignment = np.asarray(assignment)

    cut = assignment[eu] != assignment[ev]
    cut_indicator = np.zeros(n)
    cut_indicator[eu[cut]] = 1.0
    cut_indicator[ev[cut]] = 1.0

    boundary_len_other = (
        np.bincount(eu[cut], weights=L[cut], minlength=n)
        + np.bincount(ev[cut], weights=L[cut], minlength=n)
    )
    same = ~cut
    same_district_neighbors = (
        np.bincount(eu[same], minlength=n).astype(float)
        + np.bincount(ev[same], minlength=n).astype(float)
    )
    neighbor_count = np.maximum(degree, 1)
    boundary_frac = 1.0 - (same_district_neighbors / neighbor_count)

    boundary_compactness = boundary_len_other / np.maximum(state.area, 1e-9)

    district_totals = np.bincount(assignment, weights=state.population, minlength=state.n_districts)
    district_pop_share = state.population / np.maximum(district_totals[assignment], 1e-9)

    feats = np.column_stack([
        cut_indicator,
        boundary_frac,
        boundary_compactness,
        district_pop_share,
        state.area,
        state.perimeter,
        degree,
    ])
    return feats

def compute_demographic_features(state: SyntheticState) -> np.ndarray:
    density = state.population / np.maximum(state.area, 1e-9)
    return np.column_stack([state.vap_minority_share, state.dem_share, density])

def compute_demographic_features_with_dem_share(state: SyntheticState, dem_share_override: np.ndarray) -> np.ndarray:

    density = state.population / np.maximum(state.area, 1e-9)
    return np.column_stack([state.vap_minority_share, dem_share_override, density])

def assemble_features(state: SyntheticState, assignment: np.ndarray, include_demo: bool, dem_share_override: np.ndarray | None = None) -> np.ndarray:
    topo = compute_topology_features(state, assignment)
    if not include_demo:
        return topo
    if dem_share_override is not None:
        demo = compute_demographic_features_with_dem_share(state, dem_share_override)
    else:
        demo = compute_demographic_features(state)
    return np.hstack([topo, demo])

class Standardizer:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, feature_stack: np.ndarray) -> "Standardizer":
        self.mean_ = feature_stack.mean(axis=(0, 1))
        self.std_ = feature_stack.std(axis=(0, 1))
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform_stack(self, feature_stack: np.ndarray) -> np.ndarray:
        self.fit(feature_stack)
        return self.transform(feature_stack)

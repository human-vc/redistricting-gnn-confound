from __future__ import annotations

import os
import pickle
import sys

import networkx as nx
import numpy as np

from synthetic_state import SyntheticState

_SIBLING_SRC = os.path.expanduser("~/redistricting-confound-gnn/src")
_CACHE_DIR = os.path.expanduser("~/redistricting-gnn-confound/cache")

ELECTION_PANELS = {
    "NC": [
        ("2008 Governor", "EL08G_GV_D", "EL08G_GV_R"),
        ("2012 Governor", "EL12G_GV_D", "EL12G_GV_R"),
        ("2012 President", "EL12G_PR_D", "EL12G_PR_R"),
        ("2016 Governor", "EL16G_GV_D", "EL16G_GV_R"),
        ("2016 President", "EL16G_PR_D", "EL16G_PR_R"),
    ],
    "PA": [
        ("2010 Governor", "GOV10D", "GOV10R"),
        ("2010 US Senate", "SEN10D", "SEN10R"),
        ("2012 President", "PRES12D", "PRES12R"),
        ("2012 US Senate", "USS12D", "USS12R"),
        ("2012 Atty General", "ATG12D", "ATG12R"),
        ("2014 Governor", "GOV14D", "GOV14R"),
        ("2016 President", "T16PRESD", "T16PRESR"),
        ("2016 US Senate", "T16SEND", "T16SENR"),
        ("2016 Atty General", "T16ATGD", "T16ATGR"),
    ],
    "MD": [
        ("2012 President", "PRES12D", "PRES12R"),
        ("2012 US Senate", "SEN12D", "SEN12R"),
        ("2014 Governor", "GOV14D", "GOV14R"),
        ("2016 President", "PRES16D", "PRES16R"),
        ("2016 US Senate", "SEN16D", "SEN16R"),
        ("2018 Governor", "GOV18D", "GOV18R"),
    ],
}

def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0

def _connect_components(graph) -> int:

    from scipy.spatial import cKDTree

    cent = {
        i: (graph.nodes[i]["geometry"].centroid.x, graph.nodes[i]["geometry"].centroid.y)
        for i in graph.nodes()
    }
    added = 0
    while True:
        comps = sorted(nx.connected_components(graph), key=len, reverse=True)
        if len(comps) <= 1:
            return added
        main = list(comps[0])
        rest = [n for c in comps[1:] for n in c]
        tree = cKDTree(np.array([cent[i] for i in main]))
        d, idx = tree.query(np.array([cent[i] for i in rest]))
        j = int(np.argmin(d))
        graph.add_edge(rest[j], main[int(idx[j])])
        added += 1

def _load_real_graph(postal: str):
    if _SIBLING_SRC not in sys.path:
        sys.path.insert(0, _SIBLING_SRC)
    from gerrychain import Graph
    from rgnn_data.config import STATE_CONFIGS
    from rgnn_data.graph_io import load_repaired_geodataframe

    config = STATE_CONFIGS[postal]

    sibling_root = os.path.dirname(_SIBLING_SRC)
    prev = os.getcwd()
    try:
        os.chdir(sibling_root)
        gdf = load_repaired_geodataframe(config)
        graph = Graph.from_geodataframe(
            gdf, adjacency="rook", cols_to_add=config.cols_to_add, reproject=False, ignore_errors=False
        )
    finally:
        os.chdir(prev)

    for u, v in config.enacted_plan_fix_edges:
        if graph.has_node(u) and graph.has_node(v) and not graph.has_edge(u, v):
            graph.add_edge(u, v)
    _connect_components(graph)
    return config, graph

def load_real_state(postal: str, use_cache: bool = True) -> tuple[SyntheticState, np.ndarray]:

    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_CACHE_DIR, f"real_state_{postal}.pkl")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    config, graph = _load_real_graph(postal)
    nodes = sorted(graph.nodes())
    if nodes != list(range(len(nodes))):
        raise ValueError(
            f"{postal}: expected contiguous integer node labels 0..n-1, got a "
            "different labeling; the documented contiguity fix-edge indices assume "
            "the shapefile's default row order."
        )
    n = len(nodes)

    population = np.zeros(n)
    dem = np.zeros(n)
    rep = np.zeros(n)
    vap_total = np.zeros(n)
    vap_white = np.zeros(n)
    area = np.zeros(n)
    perimeter = np.zeros(n)
    coords = np.zeros((n, 2))
    enacted_raw = np.empty(n, dtype=object)

    for i in range(n):
        d = graph.nodes[i]
        population[i] = _num(d[config.pop_col])
        dem[i] = _num(d[config.election_dem_col])
        rep[i] = _num(d[config.election_rep_col])
        vap_total[i] = _num(d[config.vap_total_col])
        vap_white[i] = _num(d[config.vap_white_col])
        geom = d["geometry"]
        area[i] = geom.area
        perimeter[i] = geom.length
        c = geom.centroid
        coords[i] = (c.x, c.y)
        enacted_raw[i] = d[config.enacted_plan_col]

    total_votes = dem + rep
    dem_share = np.where(total_votes > 0, dem / np.maximum(total_votes, 1e-9), 0.5)
    dem_share = np.clip(dem_share, 0.0, 1.0)
    vap_minority_share = np.where(
        vap_total > 0, (vap_total - vap_white) / np.maximum(vap_total, 1e-9), 0.0
    )

    shared_boundary: dict = {}
    for u, v, edata in graph.edges(data=True):
        key = (u, v) if u < v else (v, u)
        shared_boundary[key] = float(edata.get("shared_perim", 0.0))

    clean = nx.Graph()
    clean.add_nodes_from(range(n))
    clean.add_edges_from((u, v) for u, v in graph.edges())

    density = population / np.maximum(area, 1e-9)
    urbanicity = density.argsort().argsort() / max(n - 1, 1)

    enacted_assignment = np.unique(enacted_raw, return_inverse=True)[1].astype(int)

    state = SyntheticState(
        name=f"{postal}_real",
        n_precincts=n,
        n_districts=config.n_districts,
        coords=coords,
        graph=clean,
        area=area,
        perimeter=perimeter,
        shared_boundary=shared_boundary,
        population=population,
        urbanicity=urbanicity,
        vap_minority_share=vap_minority_share,
        dem_share=dem_share,
        total_votes=total_votes,
    )

    result = (state, enacted_assignment)
    if use_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
    return result

def load_election_panel(postal: str, use_cache: bool = True) -> dict:

    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_CACHE_DIR, f"election_panel_{postal}.pkl")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    config, graph = _load_real_graph(postal)
    n = graph.number_of_nodes()
    panel = {}
    for label, dcol, rcol in ELECTION_PANELS[postal]:
        dem = np.array([_num(graph.nodes[i].get(dcol, 0.0)) for i in range(n)])
        rep = np.array([_num(graph.nodes[i].get(rcol, 0.0)) for i in range(n)])
        panel[label] = (dem, rep)

    if use_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(panel, f)
    return panel

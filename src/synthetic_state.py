from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay, Voronoi
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

@dataclass
class SyntheticState:
    name: str
    n_precincts: int
    n_districts: int
    coords: np.ndarray
    graph: nx.Graph
    area: np.ndarray
    perimeter: np.ndarray
    shared_boundary: dict
    population: np.ndarray
    urbanicity: np.ndarray
    vap_minority_share: np.ndarray
    dem_share: np.ndarray
    total_votes: np.ndarray

    @property
    def density(self) -> np.ndarray:
        return self.population / np.maximum(self.area, 1e-9)

def _clipped_voronoi_polygons(coords: np.ndarray, bbox: tuple[float, float, float, float]) -> list[Polygon]:

    xmin, ymin, xmax, ymax = bbox
    span_x, span_y = xmax - xmin, ymax - ymin
    pad = 10 * max(span_x, span_y)
    guard = np.array([
        [xmin - pad, ymin - pad], [xmin - pad, ymax + pad],
        [xmax + pad, ymin - pad], [xmax + pad, ymax + pad],
        [xmin - pad, (ymin + ymax) / 2], [xmax + pad, (ymin + ymax) / 2],
        [(xmin + xmax) / 2, ymin - pad], [(xmin + xmax) / 2, ymax + pad],
    ])
    aug = np.vstack([coords, guard])
    vor = Voronoi(aug)
    frame = box(xmin, ymin, xmax, ymax)

    polys = []
    for i in range(len(coords)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if not region or -1 in region:
            polys.append(None)
            continue
        pts = vor.vertices[region]
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        clipped = poly.intersection(frame)
        polys.append(clipped if not clipped.is_empty else None)
    return polys

def _shared_boundaries(polys: list, edges: list[tuple[int, int]]) -> dict:
    out = {}
    for i, j in edges:
        pi, pj = polys[i], polys[j]
        if pi is None or pj is None:
            out[(i, j)] = 0.0
            continue
        inter = pi.boundary.intersection(pj.boundary)
        length = inter.length if hasattr(inter, "length") else 0.0
        out[(i, j)] = float(length)
    return out

def _min_distance_points(n: int, sampler, rng: np.random.Generator, min_dist: float, preexisting: np.ndarray | None = None, max_attempts_per_point: int = 200) -> np.ndarray:

    pre = preexisting if preexisting is not None else np.empty((0, 2))
    pts = np.empty((n, 2))
    count = 0
    while count < n:
        placed = np.vstack([pre, pts[:count]]) if count > 0 or len(pre) else pre
        best_pt, best_d = None, -1.0
        for _ in range(max_attempts_per_point):
            cand = sampler(rng)
            if len(placed) == 0:
                best_pt = cand
                break
            d = np.min(np.linalg.norm(placed - cand, axis=1))
            if d >= min_dist:
                best_pt = cand
                break
            if d > best_d:
                best_d, best_pt = d, cand
        pts[count] = best_pt
        count += 1
    return pts

def generate_synthetic_state(
    name: str,
    n_precincts: int,
    n_districts: int,
    rng: np.random.Generator,
    urban_frac: float = 0.35,
    density_partisan_corr: float = 0.72,
    density_demo_corr: float = 0.68,
) -> SyntheticState:
    n_urban = int(round(urban_frac * n_precincts))
    n_rural = n_precincts - n_urban
    urban_center = rng.uniform(0.3, 0.7, size=2)
    min_dist = 0.35 / np.sqrt(n_precincts)

    def urban_sampler(r):
        return np.clip(r.normal(loc=urban_center, scale=0.07, size=2), 0.001, 0.999)

    def rural_sampler(r):
        return r.uniform(0.0, 1.0, size=2)

    urban_pts = _min_distance_points(n_urban, urban_sampler, rng, min_dist)
    rural_pts = _min_distance_points(n_rural, rural_sampler, rng, min_dist, preexisting=urban_pts)
    coords = np.clip(np.vstack([urban_pts, rural_pts]), 0.001, 0.999)

    tri = Delaunay(coords)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_precincts))
    for simplex in tri.simplices:
        for a in range(3):
            for b in range(a + 1, 3):
                graph.add_edge(int(simplex[a]), int(simplex[b]))

    lengths = {e: float(np.linalg.norm(coords[e[0]] - coords[e[1]])) for e in graph.edges()}
    cutoff = np.quantile(list(lengths.values()), 0.98)
    for e, L in lengths.items():
        if L > cutoff:
            graph.remove_edge(*e)
    if not nx.is_connected(graph):
        comps = list(nx.connected_components(graph))
        comps.sort(key=len, reverse=True)
        for extra in comps[1:]:
            u = next(iter(extra))
            v = min(comps[0], key=lambda w: np.linalg.norm(coords[u] - coords[w]))
            graph.add_edge(u, v)
        comps[0] |= extra

    polys = _clipped_voronoi_polygons(coords, (0.0, 0.0, 1.0, 1.0))
    area = np.array([p.area if p is not None else 1e-6 for p in polys])
    perimeter = np.array([p.length if p is not None else 0.0 for p in polys])
    shared_boundary = _shared_boundaries(polys, list(graph.edges()))

    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    k = min(15, n_precincts - 1)
    dist_k, _ = tree.query(coords, k=k + 1)
    local_density_raw = 1.0 / (dist_k[:, 1:].mean(axis=1) + 1e-6)
    urbanicity = local_density_raw.argsort().argsort() / (n_precincts - 1)

    population = np.maximum(
        rng.lognormal(mean=np.log(3000) + 0.6 * urbanicity, sigma=0.28, size=n_precincts), 50
    )

    def _latent_to_share(u, corr, noise_scale=1.0):
        z = corr * (u - 0.5) * 4.0 + np.sqrt(max(1e-6, 1 - corr ** 2)) * rng.normal(0, noise_scale, size=n_precincts)
        return 1.0 / (1.0 + np.exp(-z))

    vap_minority_share = _latent_to_share(urbanicity, density_demo_corr)
    dem_share = _latent_to_share(urbanicity, density_partisan_corr)
    dem_share = np.clip(dem_share, 0.02, 0.98)

    total_votes = np.maximum((population * rng.uniform(0.35, 0.55, size=n_precincts)).astype(int), 20)

    return SyntheticState(
        name=name,
        n_precincts=n_precincts,
        n_districts=n_districts,
        coords=coords,
        graph=graph,
        area=area,
        perimeter=perimeter,
        shared_boundary=shared_boundary,
        population=population,
        urbanicity=urbanicity,
        vap_minority_share=vap_minority_share,
        dem_share=dem_share,
        total_votes=total_votes,
    )

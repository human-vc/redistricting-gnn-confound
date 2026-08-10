from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

import config as cfg
from ensemble import population_deviation
from synthetic_state import SyntheticState

@dataclass
class PlantedEdit:
    new_assignment: np.ndarray
    moved_nodes: set
    target_district: int
    source_district: int
    condition: str
    edit_type: str
    base_index: int

def _district_neighbors(graph: nx.Graph, assignment: np.ndarray, district: int) -> set[int]:
    neighbors = set()
    for u, v in graph.edges():
        du, dv = assignment[u], assignment[v]
        if du == district and dv != district:
            neighbors.add(dv)
        elif dv == district and du != district:
            neighbors.add(du)
    return neighbors

def _grow_strip(
    graph: nx.Graph,
    pool: set[int],
    seed: int,
    pop_budget: float,
    population: np.ndarray,
    priority: np.ndarray | None,
    rng: np.random.Generator,
    max_precincts: int,
) -> set[int]:

    if seed not in pool:
        return set()
    strip = {seed}
    strip_pop = float(population[seed])
    remaining = pool - strip
    if remaining and not nx.is_connected(graph.subgraph(remaining)):
        return set()

    while strip_pop < pop_budget and len(strip) < max_precincts:
        frontier = list((set().union(*[set(graph.neighbors(s)) for s in strip]) & pool) - strip)
        if not frontier:
            break
        if priority is not None:
            frontier.sort(key=lambda v: priority[v], reverse=True)
        else:
            rng.shuffle(frontier)

        added = False
        for cand in frontier:
            trial_remaining = remaining - {cand}
            if not trial_remaining or nx.is_connected(graph.subgraph(trial_remaining)):
                strip.add(cand)
                strip_pop += float(population[cand])
                remaining = trial_remaining
                added = True
                break
        if not added:
            break
    return strip

def generate_planted_edit(
    state: SyntheticState,
    base_assignment: np.ndarray,
    condition: str,
    edit_type: str,
    rng: np.random.Generator,
    epsilon: float,
    strip_size_range: tuple[int, int],
    base_index: int,
    max_attempts: int = 40,
) -> PlantedEdit | None:
    assert condition in ("correlated", "decorrelated")
    assert edit_type in ("pack", "crack")

    graph = state.graph
    n_districts = state.n_districts
    ideal_pop = state.population.sum() / n_districts
    dens_z = (state.density - state.density.mean()) / (state.density.std() + 1e-12)

    district_lean = np.array([
        state.dem_share[base_assignment == d].mean() if (base_assignment == d).any() else 0.5
        for d in range(n_districts)
    ])

    best_edit = None
    best_objective = -np.inf

    for _attempt in range(max_attempts):
        if condition == "correlated":
            target = int(np.argmax(np.abs(district_lean - 0.5)))
        else:
            target = int(rng.integers(n_districts))

        neighbor_districts = _district_neighbors(graph, base_assignment, target)
        if not neighbor_districts:
            continue

        pop_budget = rng.uniform(0.5, 0.95) * epsilon * ideal_pop
        max_precincts = 2 * strip_size_range[1]

        if edit_type == "pack":
            source = int(rng.choice(list(neighbor_districts))) if condition == "decorrelated" else min(
                neighbor_districts, key=lambda d: abs(district_lean[d] - district_lean[target])
            )
            pool = set(np.where(base_assignment == source)[0])
            boundary_seed_candidates = list(dict.fromkeys(
                u for u, v in graph.edges()
                if (base_assignment[u] == source and base_assignment[v] == target)
                or (base_assignment[v] == source and base_assignment[u] == target)
                for u in ([u] if base_assignment[u] == source else [v])
            ))
        else:
            source = target
            pool = set(np.where(base_assignment == target)[0])
            boundary_seed_candidates = list(dict.fromkeys(
                u for u, v in graph.edges()
                if (base_assignment[u] == target and base_assignment[v] in neighbor_districts)
                or (base_assignment[v] == target and base_assignment[u] in neighbor_districts)
                for u in ([u] if base_assignment[u] == target else [v])
            ))

        if not boundary_seed_candidates:
            continue

        if condition == "correlated":
            priority = dens_z
            seed_order = sorted(boundary_seed_candidates, key=lambda u: dens_z[u], reverse=True)
        else:
            priority = None
            seed_order = list(boundary_seed_candidates)
            rng.shuffle(seed_order)

        pop_floor = 0.4 * pop_budget
        strip = set()
        for seed in seed_order:
            candidate_strip = _grow_strip(
                graph, pool, seed, pop_budget, state.population, priority, rng, max_precincts
            )
            if len(candidate_strip) >= 2 and state.population[list(candidate_strip)].sum() >= pop_floor:
                strip = candidate_strip
                break
        if not strip:
            continue

        if edit_type == "pack":
            new_target_nodes = set(np.where(base_assignment == target)[0]) | strip
            new_source_nodes = pool - strip
            final_target = target
        else:
            new_source_nodes = set(np.where(base_assignment == source)[0]) - strip
            recipient = int(rng.choice(list(neighbor_districts))) if condition == "decorrelated" else min(
                neighbor_districts, key=lambda d: abs(district_lean[d] - district_lean[target])
            )
            new_target_nodes = set(np.where(base_assignment == recipient)[0]) | strip
            final_target = recipient

        if not new_source_nodes or not nx.is_connected(graph.subgraph(new_source_nodes)):
            continue
        if not nx.is_connected(graph.subgraph(new_target_nodes)):
            continue

        new_assignment = base_assignment.copy()
        for node in strip:
            new_assignment[node] = final_target
        if population_deviation(new_assignment, state.population, n_districts) > epsilon:
            continue

        strip_z = float(dens_z[list(strip)].mean())
        if condition == "correlated":
            objective, passes = strip_z, strip_z >= cfg.DENSITY_CONTRAST_MIN_Z
        else:
            objective, passes = -abs(strip_z), abs(strip_z) <= cfg.DENSITY_NEUTRAL_MAX_Z

        edit = PlantedEdit(
            new_assignment=new_assignment,
            moved_nodes=set(strip),
            target_district=final_target,
            source_district=source,
            condition=condition,
            edit_type=edit_type,
            base_index=base_index,
        )
        if passes:
            return edit
        if objective > best_objective:
            best_objective, best_edit = objective, edit

    return best_edit

def generate_planted_set(
    state: SyntheticState,
    ensemble: list[np.ndarray],
    condition: str,
    edit_type: str,
    n_plants: int,
    rng: np.random.Generator,
    epsilon: float,
    strip_size_range: tuple[int, int],
    max_total_attempts: int = 4000,
) -> list[PlantedEdit]:
    plants = []
    total_attempts = 0
    while len(plants) < n_plants and total_attempts < max_total_attempts:
        base_index = int(rng.integers(len(ensemble)))
        base_assignment = ensemble[base_index]
        edit = generate_planted_edit(
            state, base_assignment, condition, edit_type, rng, epsilon,
            strip_size_range, base_index, max_attempts=10,
        )
        total_attempts += 1
        if edit is not None:
            plants.append(edit)
    return plants

def measure_strip_confound(state: SyntheticState, plants: list[PlantedEdit]) -> dict:

    if not plants:
        return {"n": 0, "mean_density_z": float("nan"), "density_partisan_corr": float("nan")}
    dens = (state.density - state.density.mean()) / (state.density.std() + 1e-12)
    part = np.abs(state.dem_share - 0.5)
    strip_density = np.array([dens[list(p.moved_nodes)].mean() for p in plants])
    strip_partisan = np.array([part[list(p.moved_nodes)].mean() for p in plants])
    if len(plants) > 2 and strip_density.std() > 1e-9 and strip_partisan.std() > 1e-9:
        corr = float(np.corrcoef(strip_density, strip_partisan)[0, 1])
    else:
        corr = float("nan")
    return {
        "n": len(plants),
        "mean_density_z": float(strip_density.mean()),
        "density_partisan_corr": corr,
    }

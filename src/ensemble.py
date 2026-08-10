from __future__ import annotations

import networkx as nx
import numpy as np

def _random_spanning_tree(sub: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    for u, v in sub.edges():
        sub[u][v]["_w"] = rng.random()
    return nx.maximum_spanning_tree(sub, weight="_w", algorithm="kruskal")

def _tree_subtree_populations(tree: nx.Graph, population: np.ndarray) -> tuple[dict, dict]:
    root = next(iter(tree.nodes()))
    parent = {root: None}
    order = []
    stack = [root]
    visited = {root}
    while stack:
        u = stack.pop()
        order.append(u)
        for w in tree.neighbors(u):
            if w not in visited:
                visited.add(w)
                parent[w] = u
                stack.append(w)
    subtree_pop = {v: population[v] for v in tree.nodes()}
    for u in reversed(order):
        p = parent[u]
        if p is not None:
            subtree_pop[p] += subtree_pop[u]
    return parent, subtree_pop

def _recursive_assign(
    sub: nx.Graph,
    population: np.ndarray,
    district_ids: list[int],
    assignment: np.ndarray,
    rng: np.random.Generator,
    attempts: int,
    ideal: float,
    epsilon: float,
) -> None:
    if len(district_ids) == 1:
        for node in sub.nodes():
            assignment[node] = district_ids[0]
        return

    k1 = len(district_ids) // 2
    target_frac = k1 / len(district_ids)
    total_pop = sum(population[n] for n in sub.nodes())
    lo_a, hi_a = k1 * ideal * (1 - epsilon), k1 * ideal * (1 + epsilon)
    k2 = len(district_ids) - k1
    lo_b, hi_b = k2 * ideal * (1 - epsilon), k2 * ideal * (1 + epsilon)

    best = None
    best_valid = None
    for _ in range(attempts):
        tree = _random_spanning_tree(sub, rng)
        parent, subtree_pop = _tree_subtree_populations(tree, population)
        for v in tree.nodes():
            p = parent[v]
            if p is None:
                continue
            side_a = subtree_pop[v]
            side_b = total_pop - side_a
            err = abs(side_a / total_pop - target_frac)
            if best is None or err < best[0]:
                best = (err, tree, v, p)
            if lo_a <= side_a <= hi_a and lo_b <= side_b <= hi_b:
                if best_valid is None or err < best_valid[0]:
                    best_valid = (err, tree, v, p)
        if best_valid is not None:
            break

    _, tree, v, p = best_valid if best_valid is not None else best
    tree_work = tree.copy()
    tree_work.remove_edge(v, p)
    side_a = nx.node_connected_component(tree_work, v)
    side_b = set(tree.nodes()) - side_a

    sub_a = sub.subgraph(side_a).copy()
    sub_b = sub.subgraph(side_b).copy()
    _recursive_assign(sub_a, population, district_ids[:k1], assignment, rng, attempts, ideal, epsilon)
    _recursive_assign(sub_b, population, district_ids[k1:], assignment, rng, attempts, ideal, epsilon)

def initial_partition(
    graph: nx.Graph,
    population: np.ndarray,
    n_districts: int,
    epsilon: float,
    rng: np.random.Generator,
    attempts: int = 150,
    max_seed_tries: int = 25,
) -> np.ndarray:

    ideal = population.sum() / n_districts
    n = graph.number_of_nodes()
    best_assignment, best_dev = None, np.inf
    for _ in range(max_seed_tries):
        assignment = -np.ones(n, dtype=int)
        _recursive_assign(
            graph.copy(), population, list(range(n_districts)), assignment, rng, attempts, ideal, epsilon
        )
        dev = population_deviation(assignment, population, n_districts)
        if dev < best_dev:
            best_assignment, best_dev = assignment, dev
        if dev <= epsilon:
            return assignment
    return best_assignment

def _adjacent_district_pairs(graph: nx.Graph, assignment: np.ndarray) -> list[tuple[int, int]]:
    pairs = set()
    for u, v in graph.edges():
        du, dv = assignment[u], assignment[v]
        if du != dv:
            pairs.add((min(du, dv), max(du, dv)))
    return list(pairs)

def recom_step(
    graph: nx.Graph,
    assignment: np.ndarray,
    population: np.ndarray,
    n_districts: int,
    epsilon: float,
    rng: np.random.Generator,
    max_tree_attempts: int = 60,
) -> tuple[np.ndarray, bool]:
    ideal = population.sum() / n_districts
    lo, hi = ideal * (1 - epsilon), ideal * (1 + epsilon)

    pairs = _adjacent_district_pairs(graph, assignment)
    if not pairs:
        return assignment, False
    d1, d2 = pairs[rng.integers(len(pairs))]

    merged_nodes = np.where((assignment == d1) | (assignment == d2))[0]
    sub = graph.subgraph(merged_nodes).copy()
    if not nx.is_connected(sub):
        return assignment, False

    for _ in range(max_tree_attempts):
        tree = _random_spanning_tree(sub, rng)

        root = next(iter(tree.nodes()))
        parent = {root: None}
        order = []
        stack = [root]
        visited = {root}
        while stack:
            u = stack.pop()
            order.append(u)
            for w in tree.neighbors(u):
                if w not in visited:
                    visited.add(w)
                    parent[w] = u
                    stack.append(w)

        subtree_pop = {v: population[v] for v in tree.nodes()}
        for u in reversed(order):
            p = parent[u]
            if p is not None:
                subtree_pop[p] += subtree_pop[u]

        total = subtree_pop[root]
        valid_edges = []
        for v in tree.nodes():
            p = parent[v]
            if p is None:
                continue
            side_a = subtree_pop[v]
            side_b = total - side_a
            if lo <= side_a <= hi and lo <= side_b <= hi:
                valid_edges.append((v, p, side_a))

        if valid_edges:
            v, p, side_a = valid_edges[rng.integers(len(valid_edges))]
            tree_work = tree.copy()
            tree_work.remove_edge(v, p)
            comp_v = nx.node_connected_component(tree_work, v)
            new_assignment = assignment.copy()
            for node in comp_v:
                new_assignment[node] = d1
            for node in (set(tree.nodes()) - comp_v):
                new_assignment[node] = d2
            return new_assignment, True

    return assignment, False

def run_chain(
    graph: nx.Graph,
    population: np.ndarray,
    n_districts: int,
    epsilon: float,
    rng: np.random.Generator,
    steps: int,
    burn_in: int,
    thin: int,
    verbose_every: int | None = None,
) -> list[np.ndarray]:
    assignment = initial_partition(graph, population, n_districts, epsilon, rng)
    seed_dev = population_deviation(assignment, population, n_districts)
    if seed_dev > epsilon and verbose_every:
        print(f"  warning: seed population deviation {seed_dev:.3f} exceeds epsilon {epsilon:.3f}")
    ensemble = []
    accepted = 0
    for t in range(steps):
        new_assignment, ok = recom_step(graph, assignment, population, n_districts, epsilon, rng)
        if ok:
            assignment = new_assignment
            accepted += 1
        if t >= burn_in and (t - burn_in) % thin == 0:
            ensemble.append(assignment.copy())
        if verbose_every and (t + 1) % verbose_every == 0:
            print(f"  chain step {t + 1}/{steps}, accept rate so far {accepted / (t + 1):.2f}, ensemble size {len(ensemble)}")
    worst_dev = max((population_deviation(a, population, n_districts) for a in ensemble), default=0.0)
    bound = max(seed_dev, epsilon) + 1e-9
    if worst_dev > bound:
        raise AssertionError(
            f"ReCom raised worst population deviation to {worst_dev:.4f}, above the "
            f"non-increasing bound max(seed_dev, epsilon)={bound:.4f}; the balanced-cut "
            f"acceptance in recom_step is not being enforced"
        )
    return ensemble

def population_deviation(assignment: np.ndarray, population: np.ndarray, n_districts: int) -> float:
    ideal = population.sum() / n_districts
    dpop = np.array([population[assignment == d].sum() for d in range(n_districts)])
    return float(np.max(np.abs(dpop - ideal)) / ideal)

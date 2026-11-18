"""Graph-intervention semantics used by both planners and the verifier.

Attack-path enumeration is a bounded *search heuristic*.  Certificate validity
does not depend on that bound: final attack isolation is checked with exact
directed reachability.  Service requirements are endpoint pairs, so a service
remains available when any alternate route survives.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import math

import networkx as nx

Edge = Tuple[str, str]
Path = Tuple[str, ...]
ServicePair = Tuple[str, str]


@dataclass(frozen=True)
class PathSearchScope:
    """Declared limits for heuristic path enumeration.

    ``verification_semantics`` is deliberately unbounded: the verifier uses
    graph reachability rather than interpreting an empty bounded list as proof.
    """

    cutoff_edges: int = 8
    max_paths_per_pair: int = 80
    max_expansions_per_pair: int = 600
    verification_semantics: str = "unbounded_directed_reachability"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def path_edges(path: Sequence[str]) -> Set[Edge]:
    return set(zip(path[:-1], path[1:]))


def edge_set_from_paths(paths: Iterable[Sequence[str]]) -> Set[Edge]:
    out: Set[Edge] = set()
    for path in paths:
        out |= path_edges(path)
    return out


def normalize_service_pairs(requirements: Iterable[Sequence[str]]) -> List[ServicePair]:
    """Return unique endpoint pairs from pairs or legacy path declarations."""

    pairs: List[ServicePair] = []
    seen: Set[ServicePair] = set()
    for requirement in requirements:
        values = tuple(requirement)
        if len(values) < 2:
            raise ValueError("a service requirement needs distinct source and destination endpoints")
        pair = (str(values[0]), str(values[-1]))
        if pair[0] == pair[1]:
            raise ValueError("service endpoints must be distinct")
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def active_graph(graph: nx.DiGraph) -> nx.DiGraph:
    if not any(data.get("blocked", False) for _, _, data in graph.edges(data=True)):
        return graph
    active = nx.DiGraph()
    active.add_nodes_from(graph.nodes(data=True))
    for source, destination, data in graph.edges(data=True):
        if not data.get("blocked", False):
            active.add_edge(source, destination, **data)
    return active


def enumerate_attack_paths(
    graph: nx.DiGraph,
    sources: Iterable[str],
    targets: Iterable[str],
    cutoff: int = 8,
    max_paths_per_pair: int = 80,
    max_expansions_per_pair: int = 600,
) -> List[Path]:
    """Enumerate a deterministic bounded family for scoring/planning only."""

    active = active_graph(graph)
    paths: List[Path] = []
    for source in sorted(set(sources)):
        if source not in active:
            continue
        for target in sorted(set(targets)):
            if target not in active or source == target:
                continue
            count = 0
            expansions = 0
            stack = [(source, [source])]
            while stack and count < max_paths_per_pair and expansions < max_expansions_per_pair:
                node, path = stack.pop()
                if len(path) - 1 >= cutoff:
                    continue
                expansions += 1
                successors = sorted(
                    active.successors(node),
                    key=lambda n: (-float(active[node][n].get("risk_prob", 0.0)), str(n)),
                )
                # Reverse before stack insertion so the highest-ranked item is visited first.
                for neighbor in reversed(successors):
                    if neighbor in path:
                        continue
                    new_path = path + [neighbor]
                    if neighbor == target:
                        paths.append(tuple(new_path))
                        count += 1
                        if count >= max_paths_per_pair:
                            break
                    else:
                        stack.append((neighbor, new_path))
    return list(dict.fromkeys(paths))


def reachable_pairs(
    graph: nx.DiGraph,
    sources: Iterable[str],
    targets: Iterable[str],
) -> List[Tuple[str, str]]:
    """Return all reachable source-target pairs using unbounded reachability."""

    active = active_graph(graph)
    residual: List[Tuple[str, str]] = []
    for source in sorted(set(sources)):
        if source not in active:
            continue
        reachable = nx.descendants(active, source)
        for target in sorted(set(targets)):
            if target != source and target in reachable:
                residual.append((source, target))
    return residual


def deterministic_shortest_path(graph: nx.DiGraph, source: str, target: str) -> Path:
    """Return the lexicographically smallest shortest path independent of insertion order."""

    if source == target:
        return (source,)
    reverse_distances = nx.single_source_shortest_path_length(graph.reverse(copy=False), target)
    if source not in reverse_distances:
        raise nx.NetworkXNoPath(f"no path from {source!r} to {target!r}")
    path = [source]
    current = source
    while current != target:
        next_distance = reverse_distances[current] - 1
        candidates = sorted(
            neighbor
            for neighbor in graph.successors(current)
            if reverse_distances.get(neighbor) == next_distance
        )
        if not candidates:
            raise nx.NetworkXNoPath(f"no path from {source!r} to {target!r}")
        current = candidates[0]
        path.append(current)
    return tuple(path)


def residual_attack_witnesses(
    graph: nx.DiGraph,
    sources: Iterable[str],
    targets: Iterable[str],
    limit: int = 20,
) -> List[Path]:
    """Return deterministic shortest witnesses for residual attack reachability."""

    active = active_graph(graph)
    witnesses: List[Path] = []
    for source, target in reachable_pairs(active, sources, targets):
        witnesses.append(deterministic_shortest_path(active, source, target))
        if len(witnesses) >= limit:
            break
    return witnesses


def all_attack_routes_blocked(
    graph: nx.DiGraph,
    sources: Iterable[str],
    targets: Iterable[str],
) -> bool:
    return not reachable_pairs(graph, sources, targets)


def risk_mass(graph: nx.DiGraph, paths: Iterable[Path]) -> float:
    total = 0.0
    for path in paths:
        log_probability = 0.0
        valid = True
        for source, destination in zip(path[:-1], path[1:]):
            if not graph.has_edge(source, destination) or graph[source][destination].get("blocked", False):
                valid = False
                break
            probability = float(graph[source][destination].get("risk_prob", 0.05))
            log_probability += math.log(max(1e-6, min(0.999, probability)))
        if valid:
            total += math.exp(log_probability)
    return float(total)


def apply_interventions(graph: nx.DiGraph, interventions: Iterable[Edge]) -> nx.DiGraph:
    intervened = graph.copy()
    for source, destination in interventions:
        if intervened.has_edge(source, destination):
            intervened.remove_edge(source, destination)
    return intervened


def pair_reachable(graph: nx.DiGraph, pair: ServicePair) -> bool:
    source, destination = pair
    active = active_graph(graph)
    return source in active and destination in active and nx.has_path(active, source, destination)


def service_preservation(graph: nx.DiGraph, required_pairs: Iterable[Sequence[str]]) -> float:
    pairs = normalize_service_pairs(required_pairs)
    if not pairs:
        return 1.0
    active = active_graph(graph)
    reachability: Dict[str, Set[str]] = {}
    preserved = 0
    for source, destination in pairs:
        if source not in active or destination not in active:
            continue
        if source not in reachability:
            reachability[source] = nx.descendants(active, source)
        if destination in reachability[source]:
            preserved += 1
    return preserved / len(pairs)


def service_disrupted_pairs(
    graph: nx.DiGraph,
    required_pairs: Iterable[Sequence[str]],
) -> List[ServicePair]:
    pairs = normalize_service_pairs(required_pairs)
    active = active_graph(graph)
    reachability: Dict[str, Set[str]] = {}
    disrupted: List[ServicePair] = []
    for source, destination in pairs:
        if source not in active or destination not in active:
            disrupted.append((source, destination))
            continue
        if source not in reachability:
            reachability[source] = nx.descendants(active, source)
        if destination not in reachability[source]:
            disrupted.append((source, destination))
    return disrupted


def service_witnesses(
    graph: nx.DiGraph,
    required_pairs: Iterable[Sequence[str]],
) -> Dict[str, object]:
    """Return injective, canonical path records for reachable service pairs.

    A previous representation used ``"source->destination"`` as a mapping key.
    That encoding was ambiguous when a legal node identifier itself contained
    ``"->"``.  Structured records keep the two identifiers separate and sort
    normalized endpoint pairs so the witness object is independent of input
    pair order.
    """

    active = active_graph(graph)
    pairs = sorted(normalize_service_pairs(required_pairs))
    entries: List[Dict[str, object]] = []
    for source, destination in pairs:
        if source in active and destination in active and nx.has_path(active, source, destination):
            entries.append(
                {
                    "source": source,
                    "destination": destination,
                    "path": list(deterministic_shortest_path(active, source, destination)),
                }
            )
    return {
        "schema": "cin.service-path-witnesses.v1",
        "required_pair_count": len(pairs),
        "entries": entries,
    }


def hit_count(edge: Edge, paths: Iterable[Path]) -> int:
    return sum(1 for path in paths if edge in path_edges(path))


def uncovered_paths(paths: Iterable[Path], interventions: Iterable[Edge]) -> List[Path]:
    intervention_set = set(interventions)
    return [path for path in paths if not (path_edges(path) & intervention_set)]


def policy_conflicts(
    graph: nx.DiGraph,
    interventions: Iterable[Edge],
    required_pairs: Iterable[Sequence[str]],
) -> Set[ServicePair]:
    """Return service requirements made unreachable by the full policy."""

    return set(service_disrupted_pairs(apply_interventions(graph, interventions), required_pairs))

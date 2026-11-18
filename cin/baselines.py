"""Action-count-bounded comparison policies using a common capsule wrapper."""
from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple
import random

import networkx as nx

from .capsule import (
    DefenseCapsule,
    EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
    EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
    build_capsule,
    experiment_issuance_time,
)
from .repair import SubsetMinimalRepair
from .verifier import canonical_action_set
from .simulator import (
    PathSearchScope,
    all_attack_routes_blocked,
    apply_interventions,
    enumerate_attack_paths,
    path_edges,
    reachable_pairs,
    residual_attack_witnesses,
    service_preservation,
)

Edge = Tuple[str, str]


def _wrap(
    method: str,
    graph: nx.DiGraph,
    interventions: Sequence[Edge],
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    required_service_pairs: Sequence[Sequence[str]],
    budget: int,
    cutoff: int,
    candidate_actions: Sequence[Edge],
) -> DefenseCapsule:
    transformation = SubsetMinimalRepair.apply(
        graph,
        interventions,
        attack_sources,
        protected_assets,
        required_service_pairs,
        tau=1.0,
    )
    return build_capsule(
        method=method,
        graph=graph,
        proposal_transformation=transformation,
        candidate_actions=candidate_actions,
        attack_sources=attack_sources,
        protected_assets=protected_assets,
        required_service_pairs=required_service_pairs,
        action_budget=budget,
        tau=1.0,
        path_scope=PathSearchScope(cutoff_edges=cutoff),
        issued_at=experiment_issuance_time(graph),
        max_snapshot_age_at_issue_seconds=EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
        max_capsule_age_seconds=EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
    )


def static_microsegmentation(
    graph: nx.DiGraph,
    attack_sources,
    protected_assets,
    required_service_pairs,
    budget: int,
    cutoff: int = 8,
    *,
    candidate_actions: Sequence[Edge],
) -> DefenseCapsule:
    allowed = set(canonical_action_set(candidate_actions, field="candidate_actions"))
    candidates: List[Tuple[float, Edge]] = []
    for source, destination, _data in graph.edges(data=True):
        if (source, destination) not in allowed:
            continue
        cross_department = graph.nodes[source].get("dept") != graph.nodes[destination].get("dept")
        tier_increase = graph.nodes[destination].get("tier", 0) > graph.nodes[source].get("tier", 0)
        if cross_department or tier_increase:
            score = float(cross_department) + 0.5 * max(
                0,
                graph.nodes[destination].get("tier", 0) - graph.nodes[source].get("tier", 0),
            )
            candidates.append((score, (source, destination)))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    interventions = [edge for _, edge in candidates[:budget]]
    return _wrap(
        "StaticMicrosegmentation",
        graph,
        interventions,
        attack_sources,
        protected_assets,
        required_service_pairs,
        budget,
        cutoff,
        candidate_actions,
    )


def anomaly_block(
    graph: nx.DiGraph,
    attack_sources,
    protected_assets,
    required_service_pairs,
    budget: int,
    cutoff: int = 8,
    *,
    candidate_actions: Sequence[Edge],
) -> DefenseCapsule:
    allowed = set(canonical_action_set(candidate_actions, field="candidate_actions"))
    ranked = sorted(
        (edge for edge in graph.edges if edge in allowed),
        key=lambda edge: (-float(graph[edge[0]][edge[1]].get("anomaly_score", 0.0)), edge),
    )
    return _wrap(
        "AnomalyBlock",
        graph,
        ranked[:budget],
        attack_sources,
        protected_assets,
        required_service_pairs,
        budget,
        cutoff,
        candidate_actions,
    )


def greedy_risk(
    graph: nx.DiGraph,
    attack_sources,
    protected_assets,
    required_service_pairs,
    budget: int,
    cutoff: int = 8,
    *,
    candidate_actions: Sequence[Edge],
) -> DefenseCapsule:
    allowed = set(canonical_action_set(candidate_actions, field="candidate_actions"))
    ranked = sorted(
        (edge for edge in graph.edges if edge in allowed),
        key=lambda edge: (-float(graph[edge[0]][edge[1]].get("risk_prob", 0.0)), edge),
    )
    return _wrap(
        "GreedyRisk",
        graph,
        ranked[:budget],
        attack_sources,
        protected_assets,
        required_service_pairs,
        budget,
        cutoff,
        candidate_actions,
    )


def centrality_anomaly_block(
    graph: nx.DiGraph,
    attack_sources,
    protected_assets,
    required_service_pairs,
    budget: int,
    cutoff: int = 8,
    *,
    candidate_actions: Sequence[Edge],
) -> DefenseCapsule:
    allowed = set(canonical_action_set(candidate_actions, field="candidate_actions"))
    try:
        centrality = nx.edge_betweenness_centrality(graph, k=min(40, graph.number_of_nodes()), seed=17)
    except TypeError:
        centrality = nx.edge_betweenness_centrality(graph)
    ranked = sorted(
        (edge for edge in graph.edges if edge in allowed),
        key=lambda edge: (
            -(0.55 * centrality.get(edge, 0.0) + 0.45 * float(graph[edge[0]][edge[1]].get("anomaly_score", 0.0))),
            edge,
        ),
    )
    return _wrap(
        "GraphAnomalyCentralityBlock",
        graph,
        ranked[:budget],
        attack_sources,
        protected_assets,
        required_service_pairs,
        budget,
        cutoff,
        candidate_actions,
    )


def best_of_8_weighted_random_rollouts(
    graph: nx.DiGraph,
    attack_sources,
    protected_assets,
    required_service_pairs,
    budget: int,
    seed: int = 0,
    rollouts: int = 8,
    cutoff: int = 8,
    *,
    candidate_actions: Sequence[Edge],
) -> DefenseCapsule:
    rng = random.Random(seed)
    candidates = list(canonical_action_set(candidate_actions, field="candidate_actions"))
    if any(not graph.has_edge(*edge) for edge in candidates):
        raise ValueError("candidate_actions contains an edge absent from the graph")
    weights = [
        max(
            0.01,
            float(graph[source][destination].get("risk_prob", 0.05))
            + float(graph[source][destination].get("anomaly_score", 0.0)),
        )
        for source, destination in candidates
    ]
    best_reward = float("-inf")
    best_interventions: List[Edge] = []
    for _ in range(rollouts):
        remaining = candidates[:]
        remaining_weights = weights[:]
        selected: List[Edge] = []
        for _step in range(budget):
            if not remaining:
                break
            edge = rng.choices(remaining, weights=remaining_weights, k=1)[0]
            index = remaining.index(edge)
            remaining.pop(index)
            remaining_weights.pop(index)
            selected.append(edge)
        intervened = apply_interventions(graph, selected)
        residual_pairs = len(reachable_pairs(intervened, attack_sources, protected_assets))
        preserved = service_preservation(intervened, required_service_pairs)
        reward = -float(residual_pairs) + 2.0 * preserved - 0.01 * len(selected)
        if reward > best_reward:
            best_reward = reward
            best_interventions = selected
    return _wrap(
        "BestOf8WeightedRandomRollouts",
        graph,
        best_interventions,
        attack_sources,
        protected_assets,
        required_service_pairs,
        budget,
        cutoff,
        candidate_actions,
    )


def service_aware_greedy_cut(
    graph: nx.DiGraph,
    attack_sources,
    protected_assets,
    required_service_pairs,
    budget: int,
    cutoff: int = 8,
    *,
    candidate_actions: Sequence[Edge],
) -> DefenseCapsule:
    """Strong non-learned baseline: greedy path coverage under service safety."""

    scope = PathSearchScope(cutoff_edges=cutoff)
    allowed = set(canonical_action_set(candidate_actions, field="candidate_actions"))
    if any(not graph.has_edge(*edge) for edge in allowed):
        raise ValueError("candidate_actions contains an edge absent from the graph")
    selected: List[Edge] = []
    while len(selected) < budget:
        current = apply_interventions(graph, selected)
        if all_attack_routes_blocked(current, attack_sources, protected_assets):
            break
        planning_paths = enumerate_attack_paths(
            current,
            attack_sources,
            protected_assets,
            cutoff=scope.cutoff_edges,
            max_paths_per_pair=scope.max_paths_per_pair,
            max_expansions_per_pair=scope.max_expansions_per_pair,
        )
        planning_paths = list(
            dict.fromkeys(
                planning_paths
                + residual_attack_witnesses(
                    current,
                    attack_sources,
                    protected_assets,
                    limit=max(20, len(attack_sources) * len(protected_assets)),
                )
            )
        )
        candidates = sorted(
            {
                edge
                for path in planning_paths
                for edge in path_edges(path)
                if edge in allowed and edge not in selected
            }
        )
        current_pairs = len(reachable_pairs(current, attack_sources, protected_assets))
        best_edge: Edge | None = None
        best_value = float("-inf")
        for edge in candidates:
            trial_graph = current.copy()
            if trial_graph.has_edge(*edge):
                trial_graph.remove_edge(*edge)
            if service_preservation(trial_graph, required_service_pairs) < 1.0:
                continue
            pair_gain = current_pairs - len(reachable_pairs(trial_graph, attack_sources, protected_assets))
            hit = sum(1 for path in planning_paths if edge in path_edges(path))
            cost = float(graph[edge[0]][edge[1]].get("action_cost", 1.0))
            value = 2.0 * pair_gain + hit / max(1, len(planning_paths)) - 0.03 * cost
            if value > best_value + 1e-12 or (abs(value - best_value) <= 1e-12 and (best_edge is None or edge < best_edge)):
                best_edge = edge
                best_value = value
        if best_edge is None:
            break
        selected.append(best_edge)
    return _wrap(
        "ServiceAwareGreedyCut",
        graph,
        selected,
        attack_sources,
        protected_assets,
        required_service_pairs,
        budget,
        cutoff,
        candidate_actions,
    )


def exact_small_cut_oracle(
    graph: nx.DiGraph,
    attack_sources,
    protected_assets,
    required_service_pairs,
    budget: int,
    cutoff: int = 8,
    max_candidates: int = 18,
    candidate_pool: Sequence[Edge] | None = None,
    *,
    candidate_actions: Sequence[Edge],
) -> DefenseCapsule:
    """Exact lexicographic oracle on an explicit, fully disclosed universe.

    With ``candidate_pool=None``, the universe is every edge on every simple
    attack path and the function raises rather than truncate it.  Experiments
    may pass a bounded universe, in which case output is labelled
    ``ExactBoundedCandidateOracle`` and is exact only within that universe.
    """

    capability = set(canonical_action_set(candidate_actions, field="candidate_actions"))
    if any(not graph.has_edge(*edge) for edge in capability):
        raise ValueError("candidate_actions contains an edge absent from the graph")
    bounded = candidate_pool is not None
    candidate_set = set(candidate_pool or [])
    if not candidate_set.issubset(capability):
        raise ValueError("candidate_pool must be a subset of candidate_actions")
    if not bounded:
        active = apply_interventions(graph, [])
        for source in sorted(set(attack_sources)):
            for asset in sorted(set(protected_assets)):
                if source not in active or asset not in active:
                    continue
                for path in nx.all_simple_paths(active, source, asset, cutoff=max(1, graph.number_of_nodes() - 1)):
                    candidate_set.update(edge for edge in path_edges(path) if edge in capability)
                    if len(candidate_set) > max_candidates:
                        raise ValueError(
                            f"exact oracle candidate universe has more than {max_candidates} edges; use only on small instances"
                        )
    candidate_set = {edge for edge in candidate_set if graph.has_edge(*edge)}
    if len(candidate_set) > max_candidates:
        raise ValueError(f"explicit oracle candidate universe has {len(candidate_set)} edges; maximum is {max_candidates}")
    candidates = sorted(candidate_set)
    best: List[Edge] = []
    best_objective = (len(reachable_pairs(graph, attack_sources, protected_assets)), 0, 0.0)
    for size in range(0, min(budget, len(candidates)) + 1):
        for combo in combinations(candidates, size):
            intervened = apply_interventions(graph, combo)
            if service_preservation(intervened, required_service_pairs) < 1.0:
                continue
            objective = (
                len(reachable_pairs(intervened, attack_sources, protected_assets)),
                size,
                sum(float(graph[source][destination].get("action_cost", 1.0)) for source, destination in combo),
            )
            if objective < best_objective:
                best_objective = objective
                best = list(combo)
    return _wrap(
        "ExactBoundedCandidateOracle" if bounded else "ExactSmallCutOracle",
        graph,
        best,
        attack_sources,
        protected_assets,
        required_service_pairs,
        budget,
        cutoff,
        sorted(candidate_set),
    )


BASELINE_FUNCS = {
    "ServiceAwareGreedyCut": service_aware_greedy_cut,
    "StaticMicrosegmentation": static_microsegmentation,
    "AnomalyBlock": anomaly_block,
    "GreedyRisk": greedy_risk,
    "BestOf8WeightedRandomRollouts": best_of_8_weighted_random_rollouts,
    "GraphAnomalyCentralityBlock": centrality_anomaly_block,
}

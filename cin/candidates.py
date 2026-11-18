"""Deterministic candidate-action capability models for controlled studies."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Dict, Mapping, Sequence, Tuple

import networkx as nx

from .verifier import stable_hash

Edge = Tuple[str, str]
EXCLUSION_REASONS = (
    "unavailable",
    "protected_edge",
    "controller_inexpressible",
    "unauthorized",
    "high_cost",
)
SELECTION_POLICY = "causal-stratified-hash-round-robin-v2"


@dataclass(frozen=True)
class CandidateActionUniverse:
    schema: str
    requested_availability: float
    candidate_actions: Tuple[Edge, ...]
    excluded_actions: Tuple[Edge, ...]
    exclusion_reasons: Mapping[Edge, str]
    candidate_actions_sha256: str
    excluded_actions_sha256: str
    exclusion_reasons_sha256: str
    selection_policy: str = SELECTION_POLICY

    @staticmethod
    def _actions_payload(actions: Sequence[Edge]) -> list[list[str]]:
        return [[source, destination] for source, destination in actions]

    @property
    def candidate_actions_json(self) -> str:
        return json.dumps(
            self._actions_payload(self.candidate_actions),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @property
    def excluded_actions_json(self) -> str:
        return json.dumps(
            self._actions_payload(self.excluded_actions),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @property
    def exclusion_records(self) -> list[dict[str, object]]:
        return [
            {
                "action": [source, destination],
                "reason": self.exclusion_reasons[(source, destination)],
            }
            for source, destination in self.excluded_actions
        ]

    @property
    def exclusion_reasons_json(self) -> str:
        return json.dumps(
            self.exclusion_records,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "requested_availability": self.requested_availability,
            "candidate_actions": [list(edge) for edge in self.candidate_actions],
            "excluded_actions": [list(edge) for edge in self.excluded_actions],
            "exclusion_reasons": self.exclusion_records,
            "candidate_actions_sha256": self.candidate_actions_sha256,
            "excluded_actions_sha256": self.excluded_actions_sha256,
            "exclusion_reasons_sha256": self.exclusion_reasons_sha256,
            "selection_policy": self.selection_policy,
        }


def _rank(snapshot_id: str, edge: Edge, seed: int) -> str:
    return hashlib.sha256(
        f"cin-candidate-study-v2|{snapshot_id}|{seed}|{edge[0]}|{edge[1]}".encode("utf-8")
    ).hexdigest()


def _reason(graph: nx.DiGraph, edge: Edge) -> str:
    """Assign one causally interpretable capability-policy stratum."""

    source, destination = edge
    data = graph[source][destination]
    # Synthetic benchmark generators call protected nodes ``asset``.  Keep the
    # older spelling for externally supplied graphs without silently missing
    # the generator's actual node type.
    if graph.nodes[destination].get("type") in {
        "asset",
        "monitor",
        "protected_asset",
    }:
        return "protected_edge"
    if data.get("kind") in {"shadow", "iot_bridge", "privilege"}:
        return "controller_inexpressible"
    if graph.nodes[source].get("dept") != graph.nodes[destination].get("dept"):
        return "unauthorized"
    if float(data.get("action_cost", 1.0)) > 1.5:
        return "high_cost"
    # Remaining otherwise-expressible edges form the controlled generic
    # availability-loss stratum; no pseudo-causal operational label is added.
    return "unavailable"


def _causal_stratified_order(
    graph: nx.DiGraph,
    edges: Sequence[Edge],
    snapshot_id: str,
    seed: int,
) -> tuple[list[Edge], dict[Edge, str]]:
    """Interleave deterministic hash rankings from genuine reason strata."""

    reasons = {edge: _reason(graph, edge) for edge in edges}
    buckets = {
        reason: sorted(
            (edge for edge in edges if reasons[edge] == reason),
            key=lambda edge: (_rank(snapshot_id, edge, seed), edge),
        )
        for reason in EXCLUSION_REASONS
    }
    positions = {reason: 0 for reason in EXCLUSION_REASONS}
    ordered: list[Edge] = []
    while len(ordered) < len(edges):
        progressed = False
        for reason in EXCLUSION_REASONS:
            position = positions[reason]
            bucket = buckets[reason]
            if position < len(bucket):
                ordered.append(bucket[position])
                positions[reason] += 1
                progressed = True
        if not progressed:  # Defensive guard against an inconsistent bucket map.
            raise RuntimeError("candidate exclusion stratification did not cover all edges")
    return ordered, reasons


def derive_candidate_action_universe(
    graph: nx.DiGraph,
    availability: float,
    *,
    seed: int = 0,
) -> CandidateActionUniverse:
    """Select an exact-size, nested capability subset by causal strata.

    Every edge first enters exactly one capability-policy stratum based on a
    graph fact (protected destination, unsupported relation, authorization
    boundary, declared high cost) or the explicitly controlled generic
    unavailability stratum.  A deterministic round-robin over hash-ranked
    strata selects the excluded prefix.  Thus labels determine the controlled
    subset instead of being decorative post-hoc explanations.  Prefix nesting
    also guarantees that the 50% candidate set is a subset of the 75% set.
    """

    if (
        not isinstance(availability, (int, float))
        or isinstance(availability, bool)
        or not math.isfinite(float(availability))
        or float(availability) not in {1.0, 0.75, 0.5}
    ):
        raise ValueError("candidate availability must be one of 1.0, 0.75, or 0.5")
    availability = float(availability)
    snapshot_id = str(graph.graph.get("snapshot_id", "missing-snapshot-id"))
    edges = sorted((str(source), str(destination)) for source, destination in graph.edges)
    exclusion_order, all_reasons = _causal_stratified_order(
        graph, edges, snapshot_id, seed
    )
    keep_count = round(len(edges) * availability)
    exclude_count = len(edges) - keep_count
    excluded_set = set(exclusion_order[:exclude_count])
    candidate_actions = tuple(edge for edge in edges if edge not in excluded_set)
    excluded_actions = tuple(edge for edge in edges if edge in excluded_set)
    reasons = {edge: all_reasons[edge] for edge in excluded_actions}
    candidate_payload = [list(edge) for edge in candidate_actions]
    excluded_payload = [list(edge) for edge in excluded_actions]
    reason_payload = [
        {"action": list(edge), "reason": reasons[edge]} for edge in excluded_actions
    ]
    return CandidateActionUniverse(
        schema="cin.candidate-action-universe.v2",
        requested_availability=availability,
        candidate_actions=candidate_actions,
        excluded_actions=excluded_actions,
        exclusion_reasons=reasons,
        candidate_actions_sha256=stable_hash(candidate_payload),
        excluded_actions_sha256=stable_hash(excluded_payload),
        exclusion_reasons_sha256=stable_hash(reason_payload),
    )

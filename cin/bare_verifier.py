"""Minimal authenticated direct-recomputation verifier baseline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx

from .simulator import all_attack_routes_blocked, apply_interventions, service_preservation
from .verifier import (
    _parse_utc_timestamp,
    canonical_action_set,
    canonical_service_pairs,
    canonical_json_value,
    graph_from_dict,
    graph_to_dict,
    stable_hash,
    validate_freshness,
)

Edge = Tuple[str, str]


@dataclass(frozen=True)
class BareVerificationResult:
    schema: str
    passed: bool
    bundle_hash_matches: bool
    schema_valid: bool
    threat_scope_valid: bool
    service_constraints_valid: bool
    candidate_edges_valid: bool
    intervention_edges_valid: bool
    freshness_valid: bool
    candidate_membership_valid: bool
    budget_valid: bool
    attack_reachability_blocked: bool
    service_preservation_valid: bool
    subset_minimal: bool
    fault_code: str
    messages: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["messages"] = list(self.messages)
        return payload


def make_bare_verification_bundle(
    graph: nx.DiGraph,
    actions: Sequence[Edge],
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    required_service_pairs: Sequence[Sequence[str]],
    *,
    candidate_actions: Sequence[Edge],
    action_budget: int,
    freshness: Mapping[str, object],
    tau: float = 1.0,
) -> Dict[str, object]:
    """Serialize only the inputs needed for authenticated recomputation."""

    exact_actions = canonical_action_set(actions, field="actions")
    candidates = canonical_action_set(candidate_actions, field="candidate_actions")
    if any(not graph.has_edge(*edge) for edge in exact_actions):
        raise ValueError("actions contains an edge absent from the graph")
    if any(not graph.has_edge(*edge) for edge in candidates):
        raise ValueError("candidate_actions contains an edge absent from the graph")
    if not set(exact_actions).issubset(candidates):
        raise ValueError("actions contains an edge outside candidate_actions")
    if not isinstance(attack_sources, (list, tuple)) or not isinstance(
        protected_assets, (list, tuple)
    ):
        raise ValueError("threat endpoints must be arrays")
    sources = tuple(attack_sources)
    assets = tuple(protected_assets)
    if (
        not sources
        or not assets
        or any(type(node) is not str or not node for node in sources + assets)
        or len(sources) != len(set(sources))
        or len(assets) != len(set(assets))
        or not set(sources).isdisjoint(assets)
        or any(node not in graph for node in sources + assets)
    ):
        raise ValueError(
            "threat scope must contain nonempty, unique, existing, disjoint "
            "source and asset node lists"
        )
    service_pairs = canonical_service_pairs(required_service_pairs)
    if any(
        source not in graph or destination not in graph
        for source, destination in service_pairs
    ):
        raise ValueError(
            "required_service_pairs contains an endpoint absent from the graph"
        )
    if type(action_budget) is not int or action_budget < 0:
        raise ValueError("action_budget must be a non-negative integer")
    if type(tau) not in {int, float} or float(tau) != 1.0:
        raise ValueError("service threshold tau must equal 1.0")
    return {
        "schema": "cin.bare-authenticated-verification-bundle.v1",
        "graph": graph_to_dict(graph),
        "threat_model": {
            "attack_sources": sorted(sources),
            "protected_assets": sorted(assets),
        },
        "constraints": {
            "required_service_pairs": [
                list(pair)
                for pair in service_pairs
            ],
            "candidate_actions": [list(edge) for edge in candidates],
            "service_threshold": float(tau),
            "action_budget": action_budget,
            "freshness": canonical_json_value(dict(freshness)),
        },
        "actions": [list(edge) for edge in exact_actions],
    }


def bare_verification_bundle_hash(bundle: Mapping[str, object]) -> str:
    return stable_hash(bundle)


def _failure(
    *,
    bundle_hash_matches: bool,
    code: str,
    message: str,
) -> BareVerificationResult:
    return BareVerificationResult(
        schema="cin.bare-verification-result.v1",
        passed=False,
        bundle_hash_matches=bundle_hash_matches,
        schema_valid=False,
        threat_scope_valid=False,
        service_constraints_valid=False,
        candidate_edges_valid=False,
        intervention_edges_valid=False,
        freshness_valid=False,
        candidate_membership_valid=False,
        budget_valid=False,
        attack_reachability_blocked=False,
        service_preservation_valid=False,
        subset_minimal=False,
        fault_code=code,
        messages=(message,),
    )


def verify_bare_verification_bundle(
    bundle: Mapping[str, object],
    *,
    expected_bundle_sha256: str,
    verification_time: datetime | str,
) -> BareVerificationResult:
    """Authenticate and directly recompute the minimal semantic contract."""

    if not isinstance(bundle, Mapping):
        return _failure(
            bundle_hash_matches=False,
            code="malformed_bundle",
            message="bundle must be an object",
        )
    try:
        actual_root = stable_hash(bundle)
    except (TypeError, ValueError) as error:
        return _failure(
            bundle_hash_matches=False,
            code="noncanonical_bundle",
            message=str(error),
        )
    hash_matches = actual_root == expected_bundle_sha256
    required_top = {"schema", "graph", "threat_model", "constraints", "actions"}
    if set(bundle) != required_top or bundle.get("schema") != "cin.bare-authenticated-verification-bundle.v1":
        return _failure(
            bundle_hash_matches=hash_matches,
            code="schema_mismatch",
            message="bare bundle schema or top-level field set is invalid",
        )
    try:
        graph_payload = bundle["graph"]
        threat = bundle["threat_model"]
        constraints = bundle["constraints"]
        if not isinstance(graph_payload, Mapping):
            raise ValueError("graph must be an object")
        if not isinstance(threat, Mapping) or set(threat) != {
            "attack_sources",
            "protected_assets",
        }:
            raise ValueError("threat_model field set is invalid")
        if not isinstance(constraints, Mapping) or set(constraints) != {
            "required_service_pairs",
            "candidate_actions",
            "service_threshold",
            "action_budget",
            "freshness",
        }:
            raise ValueError("constraints field set is invalid")
        graph = graph_from_dict(graph_payload)
        actions = canonical_action_set(bundle["actions"], field="actions")
        candidates = canonical_action_set(
            constraints["candidate_actions"], field="candidate_actions"
        )
        if not isinstance(threat["attack_sources"], (list, tuple)) or not isinstance(
            threat["protected_assets"], (list, tuple)
        ):
            raise ValueError("threat endpoints must be arrays")
        attack_sources = tuple(threat["attack_sources"])
        protected_assets = tuple(threat["protected_assets"])
        if any(
            type(node) is not str or not node
            for node in attack_sources + protected_assets
        ):
            raise ValueError("threat endpoints must be nonempty strings")
        required_pairs = list(
            canonical_service_pairs(constraints["required_service_pairs"])
        )
        budget = constraints["action_budget"]
        if type(budget) is not int or budget < 0:
            raise ValueError("action_budget must be a non-negative integer")
        tau = constraints["service_threshold"]
        if type(tau) not in {int, float} or float(tau) != 1.0:
            raise ValueError("service threshold must equal 1.0")
        verification = (
            verification_time
            if isinstance(verification_time, datetime)
            else _parse_utc_timestamp(verification_time)
        )
        if verification is None:
            raise ValueError("verification_time must be a zoned RFC 3339 timestamp")
    except (KeyError, TypeError, ValueError) as error:
        return _failure(
            bundle_hash_matches=hash_matches,
            code="malformed_bound_input",
            message=str(error),
        )

    freshness_valid, freshness_messages = validate_freshness(
        constraints["freshness"], graph, verification
    )
    threat_scope_valid = (
        bool(attack_sources)
        and bool(protected_assets)
        and len(attack_sources) == len(set(attack_sources))
        and len(protected_assets) == len(set(protected_assets))
        and set(attack_sources).isdisjoint(protected_assets)
        and all(node in graph for node in attack_sources + protected_assets)
    )
    service_constraints_valid = all(
        source in graph and destination in graph
        for source, destination in required_pairs
    )
    candidate_edges_valid = all(graph.has_edge(*edge) for edge in candidates)
    intervention_edges_valid = all(graph.has_edge(*edge) for edge in actions)
    candidate_membership = (
        candidate_edges_valid
        and intervention_edges_valid
        and set(actions).issubset(candidates)
    )
    budget_valid = len(actions) <= budget
    after = apply_interventions(graph, actions)
    attack_blocked = (
        threat_scope_valid
        and intervention_edges_valid
        and all_attack_routes_blocked(after, attack_sources, protected_assets)
    )
    service_valid = (
        service_constraints_valid
        and intervention_edges_valid
        and service_preservation(after, required_pairs) >= float(tau) - 1e-12
    )
    subset_minimal = (
        candidate_membership
        and attack_blocked
        and service_valid
        and all(
            not (
                all_attack_routes_blocked(
                    apply_interventions(
                        graph, actions[:index] + actions[index + 1 :]
                    ),
                    attack_sources,
                    protected_assets,
                )
                and service_preservation(
                    apply_interventions(
                        graph, actions[:index] + actions[index + 1 :]
                    ),
                    required_pairs,
                )
                >= float(tau) - 1e-12
            )
            for index in range(len(actions))
        )
    )
    checks = (
        (hash_matches, "authenticated_root_mismatch"),
        (threat_scope_valid, "invalid_threat_scope"),
        (service_constraints_valid, "invalid_service_constraints"),
        (candidate_edges_valid, "candidate_edge_absent"),
        (intervention_edges_valid, "intervention_edge_absent"),
        (freshness_valid, "freshness_invalid"),
        (candidate_membership, "out_of_candidate"),
        (budget_valid, "over_budget"),
        (attack_blocked, "threat_reachable"),
        (service_valid, "service_violation"),
        (subset_minimal, "not_subset_minimal"),
    )
    fault_code = next((code for passed, code in checks if not passed), "accepted")
    messages: List[str] = list(freshness_messages)
    if not hash_matches:
        messages.insert(0, "authenticated bundle root differs from expected root")
    for passed, code in checks[1:]:
        if not passed:
            messages.append(code)
    passed = all(value for value, _code in checks)
    return BareVerificationResult(
        schema="cin.bare-verification-result.v1",
        passed=passed,
        bundle_hash_matches=hash_matches,
        schema_valid=True,
        threat_scope_valid=threat_scope_valid,
        service_constraints_valid=service_constraints_valid,
        candidate_edges_valid=candidate_edges_valid,
        intervention_edges_valid=intervention_edges_valid,
        freshness_valid=freshness_valid,
        candidate_membership_valid=candidate_membership,
        budget_valid=budget_valid,
        attack_reachability_blocked=attack_blocked,
        service_preservation_valid=service_valid,
        subset_minimal=subset_minimal,
        fault_code=fault_code,
        messages=tuple(messages),
    )

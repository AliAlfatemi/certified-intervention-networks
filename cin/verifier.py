"""Independent, machine-checkable verifier for defense-capsule claims.

The planner may use bounded path enumeration, but this checker proves its
model-relative attack claim with exact directed reachability.  Certificates
bind the graph, threat model, service constraints, budget, interventions, and
all policy artifacts through SHA-256 digests.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields as dataclass_fields
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple
import hashlib
import hmac
import json
import math
import re

import networkx as nx
import yaml

from .canonical_json import canonical_json_bytes, canonical_json_value

from .simulator import (
    PathSearchScope,
    active_graph,
    all_attack_routes_blocked,
    apply_interventions,
    enumerate_attack_paths,
    normalize_service_pairs,
    path_edges,
    policy_conflicts,
    reachable_pairs,
    residual_attack_witnesses,
    service_preservation,
    service_witnesses,
)

Edge = Tuple[str, str]
ServicePair = Tuple[str, str]

LEGACY_CERTIFICATE_SCHEMA = "cin.certificate.v1"
LEGACY_CHECKER_VERSION = "cin-verifier-1.4"
CERTIFICATE_SCHEMA = "cin.certificate.v2"
CHECKER_VERSION = "cin-verifier-1.7"
ATTACK_INVARIANT_SCHEMA = "cin.attack-reachability-invariant.v1"
ATTACK_INVARIANT_DESCRIPTION = (
    "Typed source-to-protected-asset reachability patterns and the proposed "
    "cut-edge patterns."
)
ATTACK_INVARIANT_CLAIM_SCOPE = "model-relative directed graph reachability"
ABSTRACT_POLICY_SCHEMA = "cin.policy.v1"
ABSTRACT_POLICY_SEMANTICS = (
    "deny exactly the listed directed graph edges; otherwise preserve prior policy"
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RFC3339_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)


def _jsonable(value: Any) -> Any:
    """Compatibility name for the strict JSON-domain validator/normalizer."""

    return canonical_json_value(value)


def _type_sensitive_json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's ``bool == int`` aliasing."""

    return canonical_json_bytes(left) == canonical_json_bytes(right)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def verification_bundle_hash(bundle: Mapping[str, object]) -> str:
    """Return the canonical SHA-256 root for a complete verification bundle."""

    if not isinstance(bundle, Mapping):
        raise TypeError("verification bundle must be a mapping")
    return stable_hash(bundle)


def canonical_action_set(
    actions: object,
    *,
    field: str = "actions",
) -> Tuple[Edge, ...]:
    """Validate and lexicographically canonicalize an exact directed-edge set.

    Ordering is representational only.  No value is coerced and no duplicate is
    removed: malformed and duplicate inputs are rejected before sorting.
    """

    if not isinstance(actions, (list, tuple)):
        raise ValueError(f"{field} must be an array of two-string directed edges")
    parsed: List[Edge] = []
    seen: Set[Edge] = set()
    for index, action in enumerate(actions):
        if not isinstance(action, (list, tuple)) or len(action) != 2:
            raise ValueError(
                f"{field}[{index}] must be a two-string directed edge"
            )
        source, destination = action
        if (
            type(source) is not str
            or type(destination) is not str
            or not source
            or not destination
        ):
            raise ValueError(
                f"{field}[{index}] must be a two-nonempty-string directed edge"
            )
        edge = (source, destination)
        if edge in seen:
            raise ValueError(f"{field} contains duplicate directed edge {edge!r}")
        seen.add(edge)
        parsed.append(edge)
    return tuple(sorted(parsed))


def canonical_service_pairs(
    requirements: object,
    *,
    field: str = "required_service_pairs",
) -> Tuple[ServicePair, ...]:
    """Validate exact v2 endpoint pairs without identifier coercion or deduplication."""

    if not isinstance(requirements, (list, tuple)):
        raise ValueError(f"{field} must be an array of two-string endpoint pairs")
    parsed: List[ServicePair] = []
    seen: Set[ServicePair] = set()
    for index, requirement in enumerate(requirements):
        if not isinstance(requirement, (list, tuple)) or len(requirement) != 2:
            raise ValueError(f"{field}[{index}] must be a two-string endpoint pair")
        source, destination = requirement
        if (
            type(source) is not str
            or type(destination) is not str
            or not source
            or not destination
        ):
            raise ValueError(
                f"{field}[{index}] must be a two-nonempty-string endpoint pair"
            )
        pair = (source, destination)
        if source == destination:
            raise ValueError(f"{field}[{index}] endpoints must be distinct")
        if pair in seen:
            raise ValueError(f"{field} contains duplicate endpoint pair {pair!r}")
        seen.add(pair)
        parsed.append(pair)
    return tuple(sorted(parsed))


def graph_to_dict(graph: nx.DiGraph) -> Dict[str, object]:
    invalid_nodes = [
        node for node in graph.nodes if type(node) is not str or not node
    ]
    if invalid_nodes:
        raise ValueError("verification graph node identifiers must be strings and non-empty")
    return {
        "directed": True,
        "graph_attributes": _jsonable(graph.graph),
        "nodes": [
            {"id": node, "attributes": _jsonable(data)}
            for node, data in sorted(graph.nodes(data=True), key=lambda item: item[0])
        ],
        "edges": [
            {"source": source, "destination": destination, "attributes": _jsonable(data)}
            for source, destination, data in sorted(graph.edges(data=True), key=lambda item: (item[0], item[1]))
        ],
    }


def graph_from_dict(payload: Mapping[str, object]) -> nx.DiGraph:
    if payload.get("directed") is not True:
        raise ValueError("verification graph must be directed")
    graph = nx.DiGraph()
    graph.graph.update(dict(payload.get("graph_attributes", {})))
    seen_nodes: Set[str] = set()
    for item in payload.get("nodes", []):
        if not isinstance(item, Mapping) or "id" not in item:
            raise ValueError("malformed node in verification graph")
        node_id = item["id"]
        if type(node_id) is not str or not node_id:
            raise ValueError("verification graph node identifiers must be strings and non-empty")
        if node_id in seen_nodes:
            raise ValueError(f"duplicate node id in verification graph: {node_id}")
        seen_nodes.add(node_id)
        graph.add_node(node_id, **dict(item.get("attributes", {})))
    seen_edges: Set[Edge] = set()
    for item in payload.get("edges", []):
        if not isinstance(item, Mapping) or "source" not in item or "destination" not in item:
            raise ValueError("malformed edge in verification graph")
        if type(item["source"]) is not str or type(item["destination"]) is not str:
            raise ValueError("verification graph edge endpoints must be strings")
        edge = (item["source"], item["destination"])
        if edge in seen_edges:
            raise ValueError(f"duplicate directed edge in verification graph: {edge}")
        if edge[0] not in seen_nodes or edge[1] not in seen_nodes:
            raise ValueError(f"verification edge references an undeclared node: {edge}")
        seen_edges.add(edge)
        graph.add_edge(*edge, **dict(item.get("attributes", {})))
    return graph


def graph_hash(graph: nx.DiGraph) -> str:
    return stable_hash(graph_to_dict(graph))


@dataclass
class VerificationResult:
    schema: str
    checker_version: str
    verification_bundle_sha256: str | None
    expected_bundle_sha256: str | None
    bundle_hash_matches: bool | None
    embedded_issuance_certificate_valid: bool | None
    passed: bool
    attack_reachability_blocked: bool
    service_pairs_preserved: bool
    policy_conflict_free: bool
    rollback_schema_valid: bool
    rollback_defined: bool
    typed_sketch_schema_valid: bool
    identifier_omission_valid: bool
    policy_equivalent: bool
    freshness_valid: bool
    capsule_freshness_matches: bool
    capsule_budget_matches: bool
    submitted_witnesses_valid: bool
    proof_witnesses_valid: bool
    threat_scope_valid: bool
    service_constraints_valid: bool
    scope_declared: bool
    budget_respected: bool
    interventions_valid: bool
    subset_minimal: bool
    minimality_status: str
    minimality_ratio_bound: float | None
    residual_attack_pairs: int
    service_preservation: float
    conflicts: int
    action_budget: int
    action_count: int
    path_search_scope: Dict[str, object]
    freshness: Dict[str, object]
    verified_at: str
    bindings: Dict[str, object]
    witnesses: Dict[str, object]
    messages: List[str]
    candidate_actions_valid: bool | None = None
    candidate_action_count: int | None = None

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        if self.schema == LEGACY_CERTIFICATE_SCHEMA:
            payload.pop("candidate_actions_valid", None)
            payload.pop("candidate_action_count", None)
        return payload


def _require_exact_fields(
    value: object,
    expected: Set[str],
    *,
    field: str,
) -> Mapping[str, object]:
    """Return an object only when its declared v2 field set is exact."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{field} field set is invalid (missing={missing}, unexpected={unexpected})"
        )
    return value


def _v2_edge_pattern(graph: nx.DiGraph, edge: Edge) -> Dict[str, object]:
    """Reconstruct the exact typed edge record emitted by the v2 builder."""

    source, destination = edge
    attributes = graph[source][destination]
    return {
        "src_type": graph.nodes[source].get("type"),
        "dst_type": graph.nodes[destination].get("type"),
        "src_tier": graph.nodes[source].get("tier"),
        "dst_tier": graph.nodes[destination].get("tier"),
        "relation": attributes.get("kind"),
        "cross_dept": not _type_sensitive_json_equal(
            graph.nodes[source].get("dept"),
            graph.nodes[destination].get("dept"),
        ),
    }


def _expected_v2_attack_invariant(
    graph: nx.DiGraph,
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    interventions: Sequence[Edge],
    scope: PathSearchScope,
) -> Dict[str, object]:
    """Independently reconstruct the deterministic descriptive invariant."""

    attack_paths = enumerate_attack_paths(
        graph,
        attack_sources,
        protected_assets,
        cutoff=scope.cutoff_edges,
        max_paths_per_pair=scope.max_paths_per_pair,
        max_expansions_per_pair=scope.max_expansions_per_pair,
    )
    typed_paths = [
        {
            "node_types": [
                graph.nodes[node].get("type") for node in path if node in graph
            ],
            "edge_relations": [
                graph[source][destination].get("kind")
                for source, destination in zip(path[:-1], path[1:])
                if graph.has_edge(source, destination)
            ],
            "tier_sequence": [
                graph.nodes[node].get("tier") for node in path if node in graph
            ],
        }
        for path in attack_paths[:20]
    ]
    return {
        "schema": ATTACK_INVARIANT_SCHEMA,
        "description": ATTACK_INVARIANT_DESCRIPTION,
        "claim_scope": ATTACK_INVARIANT_CLAIM_SCOPE,
        "typed_path_patterns": typed_paths,
        "cut_edge_patterns": [
            _v2_edge_pattern(graph, edge) for edge in interventions
        ],
    }


def _v2_capsule_contract_errors(
    capsule: Mapping[str, object],
    graph: nx.DiGraph,
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    interventions: Sequence[Edge],
    scope: PathSearchScope,
    recomputed_bindings: Mapping[str, object],
) -> Tuple[List[str], bool]:
    """Check v2 capsule metadata that is outside the issuance certificate body."""

    messages: List[str] = []
    method = capsule.get("method")
    method_valid = (
        type(method) is str
        and bool(method)
        and method == method.strip()
    )
    if not method_valid:
        messages.append(
            "capsule method must be a non-empty, whitespace-trimmed string"
        )

    abstract = capsule.get("abstract_policy_json")
    abstract_method = abstract.get("method") if isinstance(abstract, Mapping) else None
    policy_method_matches = method_valid and abstract_method == method
    if not policy_method_matches:
        messages.append("capsule method differs from the exact abstract policy method")

    capsule_id = capsule.get("capsule_id")
    expected_capsule_id = (
        f"cin-{stable_hash({'method': method, 'bindings': recomputed_bindings})[:20]}"
        if method_valid
        else None
    )
    if type(capsule_id) is not str or capsule_id != expected_capsule_id:
        messages.append(
            "capsule_id differs from the deterministic method-and-binding identifier"
        )

    invariant = capsule.get("attack_reachability_invariant")
    expected_invariant = _expected_v2_attack_invariant(
        graph,
        attack_sources,
        protected_assets,
        interventions,
        scope,
    )
    if not isinstance(invariant, Mapping) or not _type_sensitive_json_equal(
        dict(invariant), expected_invariant
    ):
        messages.append(
            "attack_reachability_invariant differs from deterministic reconstruction"
        )
    return messages, policy_method_matches


def _apply_v2_capsule_contract_checks(
    result: VerificationResult,
    messages: Sequence[str],
    *,
    policy_method_matches: bool,
) -> None:
    if not policy_method_matches:
        result.policy_equivalent = False
    if messages:
        result.passed = False
        result.messages.extend(
            message for message in messages if message not in result.messages
        )


def _validate_v2_bundle_field_sets(bundle: Mapping[str, object]) -> None:
    """Enforce the closed object shapes of the v2 verification contract.

    Graph, node, and edge attribute maps intentionally remain open JSON maps:
    their feature names are application data, not protocol fields. Every
    protocol object surrounding those maps has an exact declared field set.
    """

    _require_exact_fields(
        bundle,
        {"schema", "graph", "threat_model", "constraints", "capsule"},
        field="verification bundle",
    )
    graph = _require_exact_fields(
        bundle.get("graph"),
        {"directed", "graph_attributes", "nodes", "edges"},
        field="verification graph",
    )
    if not isinstance(graph.get("graph_attributes"), Mapping):
        raise ValueError("verification graph graph_attributes must be an object")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("verification graph nodes and edges must be arrays")
    for index, node in enumerate(nodes):
        item = _require_exact_fields(
            node,
            {"id", "attributes"},
            field=f"verification graph node[{index}]",
        )
        if not isinstance(item.get("attributes"), Mapping):
            raise ValueError(
                f"verification graph node[{index}].attributes must be an object"
            )
    for index, edge in enumerate(edges):
        item = _require_exact_fields(
            edge,
            {"source", "destination", "attributes"},
            field=f"verification graph edge[{index}]",
        )
        if not isinstance(item.get("attributes"), Mapping):
            raise ValueError(
                f"verification graph edge[{index}].attributes must be an object"
            )

    threat = _require_exact_fields(
        bundle.get("threat_model"),
        {"attack_sources", "protected_assets"},
        field="threat_model",
    )
    for field in ("attack_sources", "protected_assets"):
        if not isinstance(threat.get(field), list):
            raise ValueError(f"threat_model.{field} must be an array")
    constraints = _require_exact_fields(
        bundle.get("constraints"),
        {
            "required_service_pairs",
            "candidate_actions",
            "service_threshold",
            "action_budget",
            "path_search_scope",
            "freshness",
        },
        field="constraints",
    )
    for field in ("required_service_pairs", "candidate_actions"):
        if not isinstance(constraints.get(field), list):
            raise ValueError(f"constraints.{field} must be an array")
    _require_exact_fields(
        constraints.get("path_search_scope"),
        {
            "cutoff_edges",
            "max_paths_per_pair",
            "max_expansions_per_pair",
            "verification_semantics",
        },
        field="constraints.path_search_scope",
    )
    freshness_fields = {
        "schema",
        "snapshot_id",
        "topology_version",
        "snapshot_time",
        "issued_at",
        "expires_at",
        "max_snapshot_age_at_issue_seconds",
        "max_capsule_age_seconds",
    }
    _require_exact_fields(
        constraints.get("freshness"),
        freshness_fields,
        field="constraints.freshness",
    )

    capsule = _require_exact_fields(
        bundle.get("capsule"),
        {
            "capsule_id",
            "schema",
            "method",
            "raw_proposal",
            "repaired_proposal",
            "proposal_transformation",
            "attack_reachability_invariant",
            "interventions",
            "candidate_actions",
            "candidate_action_capability",
            "action_budget",
            "freshness",
            "abstract_policy_json",
            "kubernetes_network_policy_yaml",
            "ebpf_xdp_pseudo_policy",
            "sdn_flow_rule_pseudo_policy",
            "service_safety_certificate",
            "rollback_condition",
            "identifier_omitting_typed_sketch",
        },
        field="capsule",
    )
    for field in (
        "raw_proposal",
        "repaired_proposal",
        "interventions",
        "candidate_actions",
    ):
        if not isinstance(capsule.get(field), list):
            raise ValueError(f"capsule.{field} must be an array")
    _require_exact_fields(
        capsule.get("freshness"),
        freshness_fields,
        field="capsule.freshness",
    )
    capability = _require_exact_fields(
        capsule.get("candidate_action_capability"),
        {
            "schema",
            "enforcement_domain",
            "candidate_actions_sha256",
            "candidate_action_count",
        },
        field="capsule.candidate_action_capability",
    )
    capability_count = capability.get("candidate_action_count")
    if type(capability_count) is not int or capability_count < 0:
        raise ValueError(
            "capsule.candidate_action_capability.candidate_action_count "
            "must be a non-negative integer"
        )
    transformation = _require_exact_fields(
        capsule.get("proposal_transformation"),
        {
            "schema",
            "algorithm",
            "transformation_version",
            "status",
            "raw_actions",
            "repaired_actions",
            "raw_actions_sha256",
            "repaired_actions_sha256",
            "steps",
            "repair_log_sha256",
        },
        field="capsule.proposal_transformation",
    )
    transformation_steps = transformation.get("steps")
    for field in ("raw_actions", "repaired_actions"):
        if not isinstance(transformation.get(field), list):
            raise ValueError(
                f"capsule.proposal_transformation.{field} must be an array"
            )
    if not isinstance(transformation_steps, list):
        raise ValueError("capsule.proposal_transformation.steps must be an array")
    transformation_step_fields = {
        "sequence",
        "pass",
        "candidate",
        "before_actions_sha256",
        "trial_actions_sha256",
        "trial_feasible",
        "decision",
        "after_actions_sha256",
    }
    for index, step in enumerate(transformation_steps):
        step = _require_exact_fields(
            step,
            transformation_step_fields,
            field=f"capsule.proposal_transformation.steps[{index}]",
        )
        if not isinstance(step.get("candidate"), list):
            raise ValueError(
                "capsule.proposal_transformation."
                f"steps[{index}].candidate must be an array"
            )
    invariant = _require_exact_fields(
        capsule.get("attack_reachability_invariant"),
        {
            "schema",
            "description",
            "claim_scope",
            "typed_path_patterns",
            "cut_edge_patterns",
        },
        field="capsule.attack_reachability_invariant",
    )
    typed_paths = invariant.get("typed_path_patterns")
    cut_patterns = invariant.get("cut_edge_patterns")
    if not isinstance(typed_paths, list) or not isinstance(cut_patterns, list):
        raise ValueError("attack-reachability invariant pattern fields must be arrays")
    for index, pattern in enumerate(typed_paths):
        pattern = _require_exact_fields(
            pattern,
            {"node_types", "edge_relations", "tier_sequence"},
            field=f"capsule.attack_reachability_invariant.typed_path_patterns[{index}]",
        )
        for field in ("node_types", "edge_relations", "tier_sequence"):
            if not isinstance(pattern.get(field), list):
                raise ValueError(
                    "capsule.attack_reachability_invariant."
                    f"typed_path_patterns[{index}].{field} must be an array"
                )
    edge_pattern_fields = {
        "src_type",
        "dst_type",
        "src_tier",
        "dst_tier",
        "relation",
        "cross_dept",
    }
    for index, pattern in enumerate(cut_patterns):
        _require_exact_fields(
            pattern,
            edge_pattern_fields,
            field=f"capsule.attack_reachability_invariant.cut_edge_patterns[{index}]",
        )

    abstract = _require_exact_fields(
        capsule.get("abstract_policy_json"),
        {"schema", "method", "semantics", "manifest", "rules"},
        field="capsule.abstract_policy_json",
    )
    _require_exact_fields(
        abstract.get("manifest"),
        {"intervention_hash", "rule_count"},
        field="capsule.abstract_policy_json.manifest",
    )
    rules = abstract.get("rules")
    if not isinstance(rules, list):
        raise ValueError("capsule.abstract_policy_json.rules must be an array")
    for index, rule in enumerate(rules):
        _require_exact_fields(
            rule,
            {"id", "action", "source", "destination", "relation"},
            field=f"capsule.abstract_policy_json.rules[{index}]",
        )
    _require_exact_fields(
        capsule.get("rollback_condition"),
        {
            "schema",
            "metric",
            "operator",
            "threshold",
            "window_seconds",
            "consecutive_windows",
            "action",
            "restore_policy_hash",
        },
        field="capsule.rollback_condition",
    )
    typed_sketch = _require_exact_fields(
        capsule.get("identifier_omitting_typed_sketch"),
        {
            "schema",
            "type",
            "version",
            "pattern_hashes",
            "typed_patterns",
            "raw_identifier_count",
            "identifier_namespace",
        },
        field="capsule.identifier_omitting_typed_sketch",
    )
    sketch_patterns = typed_sketch.get("typed_patterns")
    if not isinstance(typed_sketch.get("pattern_hashes"), list) or not isinstance(
        sketch_patterns, list
    ):
        raise ValueError(
            "capsule.identifier_omitting_typed_sketch pattern fields must be arrays"
        )
    for index, pattern in enumerate(sketch_patterns):
        _require_exact_fields(
            pattern,
            edge_pattern_fields,
            field=(
                "capsule.identifier_omitting_typed_sketch."
                f"typed_patterns[{index}]"
            ),
        )

    certificate = _require_exact_fields(
        capsule.get("service_safety_certificate"),
        {field.name for field in dataclass_fields(VerificationResult)},
        field="capsule.service_safety_certificate",
    )
    if not isinstance(certificate.get("messages"), list):
        raise ValueError("capsule.service_safety_certificate.messages must be an array")
    certificate_verified_at = _parse_utc_timestamp(certificate.get("verified_at"))
    bound_issued_at = _parse_utc_timestamp(capsule.get("freshness", {}).get("issued_at"))
    if (
        certificate_verified_at is None
        or bound_issued_at is None
        or certificate_verified_at != bound_issued_at
    ):
        raise ValueError(
            "capsule.service_safety_certificate.verified_at must be a zoned "
            "RFC 3339 timestamp equal to capsule freshness issued_at"
        )
    _require_exact_fields(
        certificate.get("path_search_scope"),
        {
            "cutoff_edges",
            "max_paths_per_pair",
            "max_expansions_per_pair",
            "verification_semantics",
        },
        field="capsule.service_safety_certificate.path_search_scope",
    )
    _require_exact_fields(
        certificate.get("freshness"),
        freshness_fields,
        field="capsule.service_safety_certificate.freshness",
    )
    bindings = _require_exact_fields(
        certificate.get("bindings"),
        {
            "checker_version_sha256",
            "graph_sha256",
            "threat_model_sha256",
            "constraints_sha256",
            "interventions_sha256",
            "policy_artifact_sha256",
            "rollback_condition_sha256",
            "typed_sketch_sha256",
            "freshness_sha256",
            "candidate_actions_sha256",
            "candidate_action_count",
            "raw_interventions_sha256",
            "repaired_interventions_sha256",
            "repair_log_sha256",
            "transformation_version",
        },
        field="capsule.service_safety_certificate.bindings",
    )
    binding_candidate_count = bindings.get("candidate_action_count")
    if type(binding_candidate_count) is not int or binding_candidate_count < 0:
        raise ValueError(
            "capsule.service_safety_certificate.bindings.candidate_action_count "
            "must be a non-negative integer"
        )
    _require_exact_fields(
        bindings.get("policy_artifact_sha256"),
        {"abstract", "kubernetes_intent", "ebpf_pseudo", "sdn_pseudo"},
        field="capsule.service_safety_certificate.bindings.policy_artifact_sha256",
    )
    witnesses = _require_exact_fields(
        certificate.get("witnesses"),
        {
            "residual_attack_shortest_paths",
            "threat_nonreachability",
            "necessary_action_paths",
            "surviving_service_paths",
            "removable_interventions",
        },
        field="capsule.service_safety_certificate.witnesses",
    )
    for field in ("residual_attack_shortest_paths", "removable_interventions"):
        if not isinstance(witnesses.get(field), list):
            raise ValueError(
                f"capsule.service_safety_certificate.witnesses.{field} must be an array"
            )
    threat_nonreachability = _require_exact_fields(
        witnesses.get("threat_nonreachability"),
        {
            "semantics",
            "source_reachable_closure",
            "protected_assets_in_closure",
            "active_edges_leaving_closure",
            "closed_under_active_edges",
            "proves_source_asset_nonreachability",
        },
        field="capsule.service_safety_certificate.witnesses.threat_nonreachability",
    )
    for field in (
        "source_reachable_closure",
        "protected_assets_in_closure",
        "active_edges_leaving_closure",
    ):
        if not isinstance(threat_nonreachability.get(field), list):
            raise ValueError(
                "capsule.service_safety_certificate.witnesses."
                f"threat_nonreachability.{field} must be an array"
            )
    necessary = witnesses.get("necessary_action_paths")
    if not isinstance(necessary, list):
        raise ValueError("necessary_action_paths must be an array")
    for index, witness in enumerate(necessary):
        witness = _require_exact_fields(
            witness,
            {
                "action",
                "restored_attack_path",
                "source",
                "protected_asset",
                "action_on_restored_path",
            },
            field=f"necessary_action_paths[{index}]",
        )
        for field in ("action", "restored_attack_path"):
            if not isinstance(witness.get(field), list):
                raise ValueError(
                    f"necessary_action_paths[{index}].{field} must be an array"
                )
    service_witnesses = _require_exact_fields(
        witnesses.get("surviving_service_paths"),
        {"schema", "required_pair_count", "entries"},
        field="surviving_service_paths",
    )
    entries = service_witnesses.get("entries")
    if not isinstance(entries, list):
        raise ValueError("surviving_service_paths.entries must be an array")
    for index, entry in enumerate(entries):
        entry = _require_exact_fields(
            entry,
            {"source", "destination", "path"},
            field=f"surviving_service_paths.entries[{index}]",
        )
        if not isinstance(entry.get("path"), list):
            raise ValueError(
                f"surviving_service_paths.entries[{index}].path must be an array"
            )


def _graph_identifiers_in(value: Any, node_ids: Set[str]) -> Set[str]:
    exposed: Set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            exposed |= _graph_identifiers_in(key, node_ids)
            exposed |= _graph_identifiers_in(item, node_ids)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            exposed |= _graph_identifiers_in(item, node_ids)
    elif isinstance(value, str) and value in node_ids:
        exposed.add(value)
    return exposed


def validate_identifier_omitting_typed_sketch(
    typed_sketch: object,
    graph: nx.DiGraph,
    *,
    strict_integer_types: bool = False,
) -> Tuple[bool, bool, List[str]]:
    """Validate typed-pattern structure and literal graph-identifier omission.

    Identifier omission is a narrow syntactic property, not a confidentiality
    or information-leakage guarantee.
    """

    if not isinstance(typed_sketch, Mapping):
        return False, False, ["identifier-omitting typed sketch is missing or is not an object"]
    schema_messages: List[str] = []
    omission_messages: List[str] = []
    required = {
        "schema",
        "type",
        "version",
        "pattern_hashes",
        "typed_patterns",
        "raw_identifier_count",
        "identifier_namespace",
    }
    if set(typed_sketch) != required:
        missing = sorted(required - set(typed_sketch))
        unexpected = sorted(set(typed_sketch) - required)
        schema_messages.append(
            f"typed-sketch field set is invalid (missing={missing}, unexpected={unexpected})"
        )
    version = typed_sketch.get("version")
    version_valid = (
        type(version) is int and version == 1
        if strict_integer_types
        else version == 1
    )
    if typed_sketch.get("schema") != "cin.typed-sketch.v1" or not version_valid:
        schema_messages.append("typed-sketch schema/version is invalid")
    if typed_sketch.get("type") != "typed-edge-patterns":
        schema_messages.append("typed-sketch type is invalid")
    if typed_sketch.get("identifier_namespace") != "omitted":
        schema_messages.append("typed-sketch identifier namespace is invalid")
        omission_messages.append("typed sketch does not declare an omitted identifier namespace")
    raw_identifier_count = typed_sketch.get("raw_identifier_count")
    raw_identifier_count_valid = (
        type(raw_identifier_count) is int and raw_identifier_count == 0
        if strict_integer_types
        else raw_identifier_count == 0
    )
    if not raw_identifier_count_valid:
        schema_messages.append("typed-sketch raw_identifier_count is invalid")
        omission_messages.append("typed sketch declares one or more raw identifiers")
    patterns = typed_sketch.get("typed_patterns")
    hashes = typed_sketch.get("pattern_hashes")
    allowed_keys = {"src_type", "dst_type", "src_tier", "dst_tier", "relation", "cross_dept"}
    if not isinstance(patterns, list) or not isinstance(hashes, list) or len(patterns) != len(hashes):
        schema_messages.append("typed-sketch pattern/hash lists are malformed")
    else:
        if strict_integer_types:
            node_types = [
                canonical_json_bytes(data.get("type"))
                for _, data in graph.nodes(data=True)
            ]
            node_tiers = [
                canonical_json_bytes(data.get("tier"))
                for _, data in graph.nodes(data=True)
            ]
            relations = [
                canonical_json_bytes(data.get("kind"))
                for _, _, data in graph.edges(data=True)
            ]
        else:
            node_types = {data.get("type") for _, data in graph.nodes(data=True)}
            node_tiers = {data.get("tier") for _, data in graph.nodes(data=True)}
            relations = {data.get("kind") for _, _, data in graph.edges(data=True)}
        for pattern, digest in zip(patterns, hashes):
            if not isinstance(pattern, Mapping) or set(pattern) != allowed_keys:
                schema_messages.append("typed edge pattern has an invalid field set")
                break
            pattern_src_type = pattern.get("src_type")
            pattern_dst_type = pattern.get("dst_type")
            pattern_src_tier = pattern.get("src_tier")
            pattern_dst_tier = pattern.get("dst_tier")
            pattern_relation = pattern.get("relation")
            if strict_integer_types:
                pattern_src_type = canonical_json_bytes(pattern_src_type)
                pattern_dst_type = canonical_json_bytes(pattern_dst_type)
                pattern_src_tier = canonical_json_bytes(pattern_src_tier)
                pattern_dst_tier = canonical_json_bytes(pattern_dst_tier)
                pattern_relation = canonical_json_bytes(pattern_relation)
            if (
                pattern_src_type not in node_types
                or pattern_dst_type not in node_types
                or pattern_src_tier not in node_tiers
                or pattern_dst_tier not in node_tiers
                or pattern_relation not in relations
                or not isinstance(pattern.get("cross_dept"), bool)
            ):
                schema_messages.append("typed edge pattern contains values outside the graph ontology")
                break
            if not isinstance(digest, str) or not HEX64.fullmatch(digest) or stable_hash(pattern) != digest:
                schema_messages.append("typed edge pattern hash mismatch")
                break
    # Typed ontology values can legitimately equal a graph identifier by
    # coincidence (for example a node named ``asset`` with type ``asset``).
    # Treat values in declared typed fields as ontology tokens and scan only
    # undeclared payload locations, where raw identifiers have no schema role.
    node_ids = {str(node) for node in graph.nodes}
    exposed: Set[str] = set()
    for key, value in typed_sketch.items():
        if key not in required:
            exposed |= _graph_identifiers_in(key, node_ids)
            exposed |= _graph_identifiers_in(value, node_ids)
    if isinstance(patterns, list):
        for pattern in patterns:
            if not isinstance(pattern, Mapping):
                exposed |= _graph_identifiers_in(pattern, node_ids)
                continue
            for key, value in pattern.items():
                if key not in allowed_keys:
                    exposed |= _graph_identifiers_in(key, node_ids)
                    exposed |= _graph_identifiers_in(value, node_ids)
    if exposed:
        omission_messages.append(
            f"typed sketch contains {len(exposed)} literal graph identifier(s)"
        )
    messages = list(dict.fromkeys(schema_messages + omission_messages))
    return not schema_messages, not omission_messages, messages


def validate_rollback(condition: object, tau: float) -> Tuple[bool, List[str]]:
    messages: List[str] = []
    if not isinstance(condition, Mapping):
        return False, ["rollback condition is missing or is not an object"]
    required = {
        "schema",
        "metric",
        "operator",
        "threshold",
        "window_seconds",
        "consecutive_windows",
        "action",
        "restore_policy_hash",
    }
    if not required.issubset(condition):
        messages.append(f"rollback condition missing fields: {sorted(required - set(condition))}")
    if condition.get("schema") != "cin.rollback.v1":
        messages.append("rollback schema is invalid")
    if condition.get("metric") != "service_endpoint_reachability_fraction" or condition.get("operator") != "<":
        messages.append("rollback metric/operator is invalid")
    threshold = condition.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or abs(float(threshold) - tau) > 1e-12:
        messages.append("rollback threshold does not equal the declared service threshold")
    for field in ("window_seconds", "consecutive_windows"):
        value = condition.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            messages.append(f"rollback {field} must be a positive integer")
    if condition.get("action") != "restore_previous_policy":
        messages.append("rollback action is invalid")
    restore_hash = condition.get("restore_policy_hash")
    if not isinstance(restore_hash, str) or not HEX64.fullmatch(restore_hash):
        messages.append("rollback restore_policy_hash is invalid")
    return not messages, messages


def _policy_bundle_hashes(policy_bundle: Mapping[str, object] | None) -> Dict[str, str]:
    if not isinstance(policy_bundle, Mapping):
        return {}
    return {str(name): stable_hash(artifact) for name, artifact in sorted(policy_bundle.items())}


def validate_policy_equivalence(
    policy_bundle: object,
    interventions: Sequence[Edge],
    *,
    graph: nx.DiGraph | None = None,
    strict: bool = False,
) -> Tuple[bool, List[str]]:
    messages: List[str] = []
    if not isinstance(policy_bundle, Mapping):
        return False, ["policy bundle is missing"]
    abstract = policy_bundle.get("abstract")
    if not isinstance(abstract, Mapping) or abstract.get("schema") != ABSTRACT_POLICY_SCHEMA:
        return False, ["abstract policy schema is invalid"]
    rules = abstract.get("rules")
    if not isinstance(rules, list):
        return False, ["abstract policy rules are missing"]
    rule_edges: List[Edge] = []
    for rule in rules:
        if not isinstance(rule, Mapping) or rule.get("action") != "deny-edge":
            messages.append("abstract policy contains a malformed or unsupported rule")
            continue
        if not isinstance(rule.get("source"), str) or not isinstance(rule.get("destination"), str):
            messages.append("abstract policy rule endpoints are invalid")
            continue
        rule_edges.append((str(rule["source"]), str(rule["destination"])))
    expected = sorted(dict.fromkeys(tuple(edge) for edge in interventions))
    if rule_edges != expected:
        messages.append("abstract policy rules are not exactly equivalent to interventions")
    intervention_digest = stable_hash([list(edge) for edge in expected])
    manifest = abstract.get("manifest")
    if not isinstance(manifest, Mapping) or manifest.get("intervention_hash") != intervention_digest:
        messages.append("abstract policy intervention manifest mismatch")

    if strict:
        expected_abstract_fields = {
            "schema",
            "method",
            "semantics",
            "manifest",
            "rules",
        }
        if set(abstract) != expected_abstract_fields:
            messages.append("abstract policy field set is invalid")
        method = abstract.get("method")
        if (
            type(method) is not str
            or not method
            or method != method.strip()
        ):
            messages.append(
                "abstract policy method must be a non-empty, whitespace-trimmed string"
            )
        if abstract.get("semantics") != ABSTRACT_POLICY_SEMANTICS:
            messages.append("abstract policy semantics declaration is invalid")
        if not isinstance(manifest, Mapping) or set(manifest) != {
            "intervention_hash",
            "rule_count",
        }:
            messages.append("abstract policy manifest field set is invalid")
        elif (
            type(manifest.get("rule_count")) is not int
            or manifest.get("rule_count") != len(expected)
        ):
            messages.append("abstract policy manifest rule_count mismatch")
        if graph is None:
            messages.append("strict abstract policy validation requires the bound graph")
        else:
            expected_rules = [
                {
                    "id": f"deny-{index:03d}",
                    "action": "deny-edge",
                    "source": source,
                    "destination": destination,
                    "relation": graph[source][destination].get("kind", "comm"),
                }
                for index, (source, destination) in enumerate(expected)
                if graph.has_edge(source, destination)
            ]
            if not _type_sensitive_json_equal(rules, expected_rules):
                messages.append(
                    "abstract policy rule ids, endpoints, actions, order, or relations "
                    "differ from deterministic reconstruction"
                )

    # Pseudo-backends are not claimed as deployable compilers.  Their explicit
    # semantic manifest must nevertheless bind them to the exact abstract rule set.
    marker = f"cin-intervention-hash: {intervention_digest}"
    for backend in ("kubernetes_intent", "ebpf_pseudo", "sdn_pseudo"):
        artifact = policy_bundle.get(backend)
        if not isinstance(artifact, str) or marker not in artifact:
            messages.append(f"{backend} is missing the intervention semantic hash")
    if isinstance(policy_bundle.get("kubernetes_intent"), str):
        try:
            parsed = yaml.safe_load(str(policy_bundle["kubernetes_intent"]))
            annotation = parsed["metadata"]["annotations"]["cin-intervention-hash"]
            if annotation != intervention_digest:
                messages.append("Kubernetes intent annotation does not match interventions")
        except (KeyError, TypeError, yaml.YAMLError):
            messages.append("Kubernetes intent is not valid bound YAML")
    return not messages, messages


def _scope_valid(scope: PathSearchScope) -> bool:
    return (
        type(scope.cutoff_edges) is int
        and scope.cutoff_edges > 0
        and type(scope.max_paths_per_pair) is int
        and scope.max_paths_per_pair > 0
        and type(scope.max_expansions_per_pair) is int
        and scope.max_expansions_per_pair > 0
        and scope.verification_semantics == "unbounded_directed_reachability"
    )


def _parse_utc_timestamp(value: object) -> datetime | None:
    """Parse an explicitly zoned RFC 3339 timestamp and normalize it to UTC."""

    if not isinstance(value, str) or RFC3339_TIMESTAMP.fullmatch(value) is None:
        return None
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except (OverflowError, ValueError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    try:
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def validate_graph_provenance(graph: nx.DiGraph) -> Tuple[bool, List[str]]:
    """Validate the mandatory identity, version, and time of a graph snapshot."""

    messages: List[str] = []
    required = ("snapshot_id", "topology_version", "snapshot_time")
    for field in required:
        if field not in graph.graph:
            messages.append(f"graph provenance is missing required field: {field}")

    snapshot_id = graph.graph.get("snapshot_id")
    if "snapshot_id" in graph.graph and (
        not isinstance(snapshot_id, str)
        or not snapshot_id.strip()
        or snapshot_id != snapshot_id.strip()
    ):
        messages.append("graph snapshot_id must be a non-empty, whitespace-trimmed string")

    topology_version = graph.graph.get("topology_version")
    if "topology_version" in graph.graph and (
        not isinstance(topology_version, int)
        or isinstance(topology_version, bool)
        or topology_version < 1
    ):
        messages.append("graph topology_version must be a positive integer")

    if "snapshot_time" in graph.graph and _parse_utc_timestamp(
        graph.graph.get("snapshot_time")
    ) is None:
        messages.append(
            "graph snapshot_time must be an RFC 3339 timestamp with an explicit timezone"
        )
    return not messages, messages


def require_graph_provenance(graph: nx.DiGraph) -> datetime:
    """Return the normalized snapshot time or raise on invalid provenance."""

    valid, messages = validate_graph_provenance(graph)
    if not valid:
        raise ValueError("invalid graph provenance: " + "; ".join(messages))
    snapshot_time = _parse_utc_timestamp(graph.graph["snapshot_time"])
    assert snapshot_time is not None  # established by validate_graph_provenance
    return snapshot_time


def _validate_freshness_v1(
    freshness: object,
    graph: nx.DiGraph,
    verification_time: datetime,
) -> Tuple[bool, List[str]]:
    """Validate a deterministic logical verification time against a bound TTL."""

    messages: List[str] = []
    graph_provenance_ok, graph_messages = validate_graph_provenance(graph)
    messages.extend(graph_messages)
    if (
        not isinstance(verification_time, datetime)
        or verification_time.tzinfo is None
        or verification_time.utcoffset() is None
    ):
        messages.append("verification_time must be a timezone-aware datetime")
        return False, messages
    verification_time = verification_time.astimezone(timezone.utc)
    if not isinstance(freshness, Mapping):
        messages.append("freshness declaration is missing or is not an object")
        return False, messages
    required = {
        "schema",
        "snapshot_id",
        "topology_version",
        "issued_at",
        "expires_at",
        "max_age_seconds",
    }
    if set(freshness) != required:
        messages.append("freshness declaration has a missing or unsupported field")
    if freshness.get("schema") != "cin.freshness.v1":
        messages.append("freshness schema is invalid")
    freshness_snapshot_id = freshness.get("snapshot_id")
    if (
        not isinstance(freshness_snapshot_id, str)
        or not freshness_snapshot_id.strip()
        or freshness_snapshot_id != freshness_snapshot_id.strip()
    ):
        messages.append("freshness snapshot_id must be a non-empty, whitespace-trimmed string")
    if freshness_snapshot_id != graph.graph.get("snapshot_id"):
        messages.append("freshness snapshot_id does not match the graph snapshot")
    version = freshness.get("topology_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        messages.append("freshness topology_version must be a positive integer")
    if version != graph.graph.get("topology_version"):
        messages.append("freshness topology_version does not match the graph")
    issued = _parse_utc_timestamp(freshness.get("issued_at"))
    expires = _parse_utc_timestamp(freshness.get("expires_at"))
    max_age = freshness.get("max_age_seconds")
    snapshot_time = _parse_utc_timestamp(graph.graph.get("snapshot_time"))
    if issued is None or expires is None:
        messages.append(
            "freshness issued_at and expires_at must be RFC 3339 timestamps with explicit timezones"
        )
    if graph_provenance_ok and issued is not None and issued != snapshot_time:
        messages.append("freshness issued_at does not match the bound graph snapshot_time")
    max_age_valid = (
        isinstance(max_age, int)
        and not isinstance(max_age, bool)
        and max_age > 0
    )
    if not max_age_valid:
        messages.append("freshness max_age_seconds must be a positive integer")
    if issued is not None and expires is not None:
        if expires <= issued:
            messages.append("freshness expires_at must be later than issued_at")
        elif max_age_valid:
            try:
                expected_expiry = issued + timedelta(seconds=max_age)
            except OverflowError:
                messages.append(
                    "freshness max_age_seconds exceeds the supported timestamp range"
                )
            else:
                if expires != expected_expiry:
                    messages.append(
                        "freshness expiry does not match issued_at plus max_age_seconds"
                    )
    if issued is not None and verification_time < issued:
        messages.append(
            "graph snapshot_time and certificate issued_at are in the future relative to verification_time"
        )
    if expires is not None and verification_time >= expires:
        messages.append("certificate is stale at the external verification time")
    return not messages, messages


def _validate_freshness_v2(
    freshness: object,
    graph: nx.DiGraph,
    verification_time: datetime,
) -> Tuple[bool, List[str]]:
    """Validate distinct snapshot, issuance, and capsule-lifetime clocks."""

    messages: List[str] = []
    graph_provenance_ok, graph_messages = validate_graph_provenance(graph)
    messages.extend(graph_messages)
    if (
        not isinstance(verification_time, datetime)
        or verification_time.tzinfo is None
        or verification_time.utcoffset() is None
    ):
        messages.append("verification_time must be a timezone-aware datetime")
        return False, messages
    verification_time = verification_time.astimezone(timezone.utc)
    if not isinstance(freshness, Mapping):
        return False, messages + ["freshness declaration is missing or is not an object"]
    required = {
        "schema",
        "snapshot_id",
        "topology_version",
        "snapshot_time",
        "issued_at",
        "expires_at",
        "max_snapshot_age_at_issue_seconds",
        "max_capsule_age_seconds",
    }
    if set(freshness) != required:
        messages.append("freshness declaration has a missing or unsupported field")
    if freshness.get("schema") != "cin.freshness.v2":
        messages.append("freshness schema is invalid")
    snapshot_id = freshness.get("snapshot_id")
    if (
        type(snapshot_id) is not str
        or not snapshot_id.strip()
        or snapshot_id != snapshot_id.strip()
    ):
        messages.append("freshness snapshot_id must be a non-empty, whitespace-trimmed string")
    if snapshot_id != graph.graph.get("snapshot_id"):
        messages.append("freshness snapshot_id does not match the graph snapshot")
    version = freshness.get("topology_version")
    if type(version) is not int or version < 1:
        messages.append("freshness topology_version must be a positive integer")
    if version != graph.graph.get("topology_version"):
        messages.append("freshness topology_version does not match the graph")

    graph_snapshot = _parse_utc_timestamp(graph.graph.get("snapshot_time"))
    snapshot = _parse_utc_timestamp(freshness.get("snapshot_time"))
    issued = _parse_utc_timestamp(freshness.get("issued_at"))
    expires = _parse_utc_timestamp(freshness.get("expires_at"))
    if snapshot is None or issued is None or expires is None:
        messages.append(
            "freshness snapshot_time, issued_at, and expires_at must be RFC 3339 timestamps with explicit timezones"
        )
    if graph_provenance_ok and snapshot is not None and snapshot != graph_snapshot:
        messages.append("freshness snapshot_time does not match the bound graph snapshot_time")

    max_snapshot_age = freshness.get("max_snapshot_age_at_issue_seconds")
    max_capsule_age = freshness.get("max_capsule_age_seconds")
    max_snapshot_age_valid = type(max_snapshot_age) is int and max_snapshot_age >= 0
    max_capsule_age_valid = type(max_capsule_age) is int and max_capsule_age > 0
    if not max_snapshot_age_valid:
        messages.append(
            "freshness max_snapshot_age_at_issue_seconds must be a non-negative integer"
        )
    if not max_capsule_age_valid:
        messages.append("freshness max_capsule_age_seconds must be a positive integer")

    if snapshot is not None and issued is not None:
        if issued < snapshot:
            messages.append("freshness issued_at must not precede snapshot_time")
        elif max_snapshot_age_valid:
            try:
                allowed_snapshot_age = timedelta(seconds=max_snapshot_age)
            except OverflowError:
                messages.append(
                    "freshness max_snapshot_age_at_issue_seconds exceeds the supported duration range"
                )
            else:
                if issued - snapshot > allowed_snapshot_age:
                    messages.append("graph snapshot is too old at capsule issuance")
    if issued is not None and expires is not None:
        if expires <= issued:
            messages.append("freshness expires_at must be later than issued_at")
        elif max_capsule_age_valid:
            try:
                expected_expiry = issued + timedelta(seconds=max_capsule_age)
            except OverflowError:
                messages.append(
                    "freshness max_capsule_age_seconds exceeds the supported timestamp range"
                )
            else:
                if expires != expected_expiry:
                    messages.append(
                        "freshness expiry does not match issued_at plus max_capsule_age_seconds"
                    )
    if issued is not None and verification_time < issued:
        messages.append("capsule issued_at is in the future relative to verification_time")
    if expires is not None and verification_time >= expires:
        messages.append("certificate is stale at the external verification time")
    return not messages, messages


def validate_freshness(
    freshness: object,
    graph: nx.DiGraph,
    verification_time: datetime,
) -> Tuple[bool, List[str]]:
    """Dispatch freshness validation by an explicit supported schema."""

    schema = freshness.get("schema") if isinstance(freshness, Mapping) else None
    if schema == "cin.freshness.v1":
        return _validate_freshness_v1(freshness, graph, verification_time)
    if schema == "cin.freshness.v2":
        return _validate_freshness_v2(freshness, graph, verification_time)
    return False, ["freshness declaration has an unsupported schema"]


def _feasible(
    graph: nx.DiGraph,
    interventions: Sequence[Edge],
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    required_pairs: Sequence[ServicePair],
    tau: float,
) -> bool:
    intervened = apply_interventions(graph, interventions)
    return all_attack_routes_blocked(intervened, attack_sources, protected_assets) and service_preservation(intervened, required_pairs) >= tau


def _service_witnesses_complete(
    payload: object,
    graph: nx.DiGraph,
    required_pairs: Sequence[ServicePair],
) -> bool:
    """Check that a structured witness contains exactly one valid path per pair."""

    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "required_pair_count",
        "entries",
    }:
        return False
    pairs = sorted(required_pairs)
    if (
        payload.get("schema") != "cin.service-path-witnesses.v1"
        or payload.get("required_pair_count") != len(pairs)
    ):
        return False
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != len(pairs):
        return False
    active = active_graph(graph)
    for entry, (source, destination) in zip(entries, pairs):
        if not isinstance(entry, Mapping) or set(entry) != {"source", "destination", "path"}:
            return False
        if entry.get("source") != source or entry.get("destination") != destination:
            return False
        path = entry.get("path")
        if (
            not isinstance(path, list)
            or len(path) < 2
            or path[0] != source
            or path[-1] != destination
            or any(not isinstance(node, str) for node in path)
            or any(not active.has_edge(u, v) for u, v in zip(path[:-1], path[1:]))
        ):
            return False
    return True


def _verify_intervention_set_core(
    graph: nx.DiGraph,
    interventions: object,
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    required_service_pairs: Sequence[Sequence[str]],
    rollback_condition: Dict[str, object] | None,
    identifier_omitting_typed_sketch: object | None,
    *,
    tau: float = 1.0,
    action_budget: int,
    path_scope: PathSearchScope | None = None,
    policy_bundle: Mapping[str, object] | None = None,
    freshness: Mapping[str, object] | None = None,
    verification_time: datetime | str | None = None,
    expected_bindings: Mapping[str, object] | None = None,
    candidate_actions: object | None,
    proposal_bindings: Mapping[str, object] | None,
    strict_actions: bool,
    explicit_verification_time: bool,
    freshness_schema: str,
    certificate_schema: str,
    checker_version: str,
) -> VerificationResult:
    """Verify a policy independently of how its interventions were proposed."""

    messages: List[str] = []
    if not isinstance(tau, (int, float)) or isinstance(tau, bool) or not math.isfinite(float(tau)):
        raise ValueError("service threshold tau must be the finite number 1.0")
    tau = float(tau)
    if tau != 1.0:
        raise ValueError("service threshold tau must equal 1.0 in this certificate schema")
    scope = path_scope or PathSearchScope()
    pairs = (
        list(
            canonical_service_pairs(
                required_service_pairs, field="required_service_pairs"
            )
        )
        if strict_actions
        else normalize_service_pairs(required_service_pairs)
    )
    if strict_actions and (
        not isinstance(attack_sources, (list, tuple))
        or not isinstance(protected_assets, (list, tuple))
    ):
        raise ValueError(
            "threat scope attack_sources and protected_assets must be arrays"
        )
    source_list = list(attack_sources)
    asset_list = list(protected_assets)
    if not all(isinstance(node, str) for node in source_list + asset_list):
        raise ValueError("threat scope endpoints must be strings")
    threat_ok = (
        bool(source_list)
        and bool(asset_list)
        and len(source_list) == len(set(source_list))
        and len(asset_list) == len(set(asset_list))
        and all(node in graph for node in source_list + asset_list)
        and set(source_list).isdisjoint(asset_list)
    )
    if not threat_ok:
        messages.append("threat scope must contain nonempty, unique, existing, disjoint source and asset node lists")
    service_constraints_ok = all(source in graph and destination in graph for source, destination in pairs)
    if not service_constraints_ok:
        messages.append("one or more service requirement endpoints are absent from the certified graph")
    if strict_actions:
        intervention_list = list(
            canonical_action_set(interventions, field="interventions")
        )
        candidate_list = list(
            canonical_action_set(candidate_actions, field="candidate_actions")
        )
        valid_edges = all(graph.has_edge(*edge) for edge in intervention_list)
        candidate_edges_valid = all(graph.has_edge(*edge) for edge in candidate_list)
        candidate_membership = set(intervention_list).issubset(candidate_list)
        candidate_actions_ok = candidate_edges_valid and candidate_membership
        interventions_ok = valid_edges and candidate_actions_ok
        if not valid_edges:
            messages.append("one or more interventions do not exist in the certified graph")
        if not candidate_edges_valid:
            messages.append("one or more candidate actions do not exist in the certified graph")
        if not candidate_membership:
            messages.append("one or more interventions are outside the bound candidate action set")
    else:
        raw_interventions = [tuple(edge) for edge in interventions]
        intervention_list = list(dict.fromkeys(raw_interventions))
        well_formed_edges = all(
            len(edge) == 2 and all(isinstance(endpoint, str) for endpoint in edge)
            for edge in intervention_list
        )
        if well_formed_edges:
            intervention_list = sorted(intervention_list)
        valid_edges = well_formed_edges and all(
            graph.has_edge(*edge) for edge in intervention_list
        )
        duplicate_free = len(raw_interventions) == len(intervention_list)
        candidate_list = sorted(graph.edges)
        candidate_actions_ok = True
        interventions_ok = valid_edges and duplicate_free
        if not valid_edges:
            messages.append("one or more interventions do not exist in the certified graph")
        if not duplicate_free:
            messages.append("intervention list contains duplicates")

    budget_ok = isinstance(action_budget, int) and not isinstance(action_budget, bool) and action_budget >= 0 and len(intervention_list) <= action_budget
    if not budget_ok:
        messages.append(f"action count {len(intervention_list)} exceeds or invalidates budget {action_budget}")
    scope_ok = _scope_valid(scope)
    if not scope_ok:
        messages.append("path-search scope is missing or invalid")

    intervened = apply_interventions(graph, intervention_list)
    residual_pairs = reachable_pairs(intervened, attack_sources, protected_assets)
    attack_ok = not residual_pairs
    preservation = service_preservation(intervened, pairs)
    service_ok = preservation + 1e-12 >= tau
    conflicts = policy_conflicts(graph, intervention_list, pairs)
    conflict_ok = not conflicts
    if not attack_ok:
        messages.append(f"residual unbounded attack-reachable source/asset pairs: {len(residual_pairs)}")
    if not service_ok:
        messages.append(f"service endpoint preservation {preservation:.3f} below tau={tau:.3f}")
    if not conflict_ok:
        messages.append(f"policy disconnects {len(conflicts)} required service endpoint pairs")

    rollback_ok, rollback_messages = validate_rollback(rollback_condition, tau)
    typed_sketch_schema_ok, identifier_omission_ok, typed_sketch_messages = (
        validate_identifier_omitting_typed_sketch(
            identifier_omitting_typed_sketch,
            graph,
            strict_integer_types=strict_actions,
        )
    )
    policy_ok, policy_messages = validate_policy_equivalence(
        policy_bundle,
        intervention_list,
        graph=graph if strict_actions else None,
        strict=strict_actions,
    )
    if verification_time is None and explicit_verification_time:
        raise ValueError("verification_time is required for exact proposal verification")
    if verification_time is None:
        checked_at = datetime.now(timezone.utc)
    elif isinstance(verification_time, str):
        checked_at = _parse_utc_timestamp(verification_time)
        if checked_at is None:
            raise ValueError("verification_time must be a timezone-aware ISO-8601 value")
    elif isinstance(verification_time, datetime):
        if verification_time.tzinfo is None or verification_time.utcoffset() is None:
            raise ValueError("verification_time must be timezone-aware")
        checked_at = verification_time.astimezone(timezone.utc)
    else:
        raise TypeError("verification_time must be datetime, ISO-8601 string, or None")
    if freshness_schema == "cin.freshness.v1":
        freshness_ok, freshness_messages = _validate_freshness_v1(
            freshness, graph, checked_at
        )
    elif freshness_schema == "cin.freshness.v2":
        freshness_ok, freshness_messages = _validate_freshness_v2(
            freshness, graph, checked_at
        )
    else:
        raise ValueError("unsupported verifier freshness schema")
    messages.extend(rollback_messages)
    messages.extend(typed_sketch_messages)
    messages.extend(policy_messages)
    messages.extend(freshness_messages)

    feasible = attack_ok and service_ok and interventions_ok
    removable: List[Edge] = []
    if feasible:
        for index, edge in enumerate(intervention_list):
            trial = intervention_list[:index] + intervention_list[index + 1 :]
            if _feasible(graph, trial, attack_sources, protected_assets, pairs, tau):
                removable.append(edge)
    subset_minimal = feasible and not removable
    if not feasible:
        minimality_status = "not_applicable_infeasible"
    elif removable:
        minimality_status = "not_subset_minimal"
        messages.append(f"{len(removable)} interventions can be individually removed while retaining feasibility")
    else:
        minimality_status = "subset_minimal"

    scope_payload = scope.to_dict()
    constraints_payload: Dict[str, object] = {
        "required_service_pairs": sorted([list(pair) for pair in pairs]),
        "service_threshold": float(tau),
        "action_budget": action_budget,
        "path_search_scope": scope_payload,
    }
    if strict_actions:
        constraints_payload["candidate_actions"] = [list(edge) for edge in candidate_list]
    bindings: Dict[str, object] = {
        "checker_version_sha256": stable_hash(checker_version),
        "graph_sha256": graph_hash(graph),
        "threat_model_sha256": stable_hash({"attack_sources": sorted(set(attack_sources)), "protected_assets": sorted(set(protected_assets))}),
        "constraints_sha256": stable_hash(constraints_payload),
        "interventions_sha256": stable_hash([list(edge) for edge in intervention_list]),
        "policy_artifact_sha256": _policy_bundle_hashes(policy_bundle),
        "rollback_condition_sha256": stable_hash(rollback_condition),
        "typed_sketch_sha256": stable_hash(identifier_omitting_typed_sketch),
        "freshness_sha256": stable_hash(dict(freshness or {})),
    }
    if strict_actions:
        bindings["candidate_actions_sha256"] = stable_hash(
            [list(edge) for edge in candidate_list]
        )
        bindings["candidate_action_count"] = len(candidate_list)
        if not isinstance(proposal_bindings, Mapping):
            raise ValueError("proposal_bindings is required for exact proposal verification")
        required_proposal_bindings = {
            "raw_interventions_sha256",
            "repaired_interventions_sha256",
            "repair_log_sha256",
            "transformation_version",
        }
        if set(proposal_bindings) != required_proposal_bindings:
            raise ValueError("proposal_bindings has a missing or unsupported field")
        for field in (
            "raw_interventions_sha256",
            "repaired_interventions_sha256",
            "repair_log_sha256",
        ):
            value = proposal_bindings.get(field)
            if type(value) is not str or HEX64.fullmatch(value) is None:
                raise ValueError(f"proposal_bindings {field} must be a lowercase SHA-256 digest")
        transformation_version = proposal_bindings.get("transformation_version")
        if type(transformation_version) is not str or not transformation_version:
            raise ValueError("proposal_bindings transformation_version must be a non-empty string")
        if proposal_bindings.get("repaired_interventions_sha256") != bindings[
            "interventions_sha256"
        ]:
            raise ValueError(
                "proposal_bindings repaired_interventions_sha256 does not bind the exact interventions"
            )
        bindings.update(canonical_json_value(dict(proposal_bindings)))
    if expected_bindings is None:
        bindings_ok = True
    elif strict_actions:
        expected_candidate_count = expected_bindings.get("candidate_action_count")
        bindings_ok = (
            type(expected_candidate_count) is int
            and expected_candidate_count >= 0
            and expected_candidate_count == len(candidate_list)
            and _type_sensitive_json_equal(expected_bindings, bindings)
        )
    else:
        bindings_ok = canonical_json_value(expected_bindings) == canonical_json_value(
            bindings
        )
    if not bindings_ok:
        messages.append("supplied certificate bindings do not match recomputed inputs")

    # Exact closure witness for the nonreachability claim.  R contains every
    # node reachable from any declared source in the active post-policy graph.
    # By construction, no active edge may leave R; an accepted certificate also
    # requires that R contain no protected asset.
    active_after = active_graph(intervened)
    source_reachable_closure: Set[str] = set()
    for source in sorted(set(attack_sources)):
        if source in active_after:
            source_reachable_closure.add(source)
            source_reachable_closure.update(nx.descendants(active_after, source))
    closure_boundary_edges = sorted(
        (source, destination)
        for source, destination in active_after.edges
        if source in source_reachable_closure and destination not in source_reachable_closure
    )
    protected_assets_in_closure = sorted(set(protected_assets) & source_reachable_closure)
    nonreachability_witness = {
        "semantics": "exact_union_of_source_reachable_nodes_in_active_graph",
        "source_reachable_closure": sorted(source_reachable_closure),
        "protected_assets_in_closure": protected_assets_in_closure,
        "active_edges_leaving_closure": [list(edge) for edge in closure_boundary_edges],
        "closed_under_active_edges": not closure_boundary_edges,
        "proves_source_asset_nonreachability": not closure_boundary_edges and not protected_assets_in_closure,
    }

    # Necessity witnesses for subset minimality.  Restoring any selected action
    # means evaluating G \ (I \ {e}); for a feasible subset-minimal policy this
    # must restore an exact source-to-asset path, and that path must contain e.
    necessary_action_paths: List[Dict[str, object]] = []
    if feasible:
        for index, edge in enumerate(intervention_list):
            trial = intervention_list[:index] + intervention_list[index + 1 :]
            restored_graph = apply_interventions(graph, trial)
            restored_paths = residual_attack_witnesses(
                restored_graph,
                attack_sources,
                protected_assets,
                limit=1,
            )
            restored_path = restored_paths[0] if restored_paths else tuple()
            necessary_action_paths.append(
                {
                    "action": list(edge),
                    "restored_attack_path": list(restored_path),
                    "source": restored_path[0] if restored_path else None,
                    "protected_asset": restored_path[-1] if restored_path else None,
                    "action_on_restored_path": bool(restored_path and edge in path_edges(restored_path)),
                }
            )
    surviving_service_paths = service_witnesses(intervened, pairs)
    service_witnesses_complete = _service_witnesses_complete(
        surviving_service_paths,
        intervened,
        pairs,
    )
    if feasible and not service_witnesses_complete:
        messages.append("canonical service-path witness set is incomplete or malformed")
    witnesses: Dict[str, object] = {
        "residual_attack_shortest_paths": [list(path) for path in residual_attack_witnesses(intervened, attack_sources, protected_assets)],
        "threat_nonreachability": nonreachability_witness,
        "necessary_action_paths": necessary_action_paths,
        "surviving_service_paths": surviving_service_paths,
        "removable_interventions": [list(edge) for edge in removable],
    }
    proof_witnesses_ok = (
        feasible
        and nonreachability_witness["proves_source_asset_nonreachability"] is True
        and service_witnesses_complete
        and len(necessary_action_paths) == len(intervention_list)
        and all(
            witness["restored_attack_path"] and witness["action_on_restored_path"]
            for witness in necessary_action_paths
        )
    )

    passed = all(
        (
            attack_ok,
            threat_ok,
            service_constraints_ok,
            service_ok,
            conflict_ok,
            rollback_ok,
            typed_sketch_schema_ok,
            identifier_omission_ok,
            policy_ok,
            freshness_ok,
            scope_ok,
            budget_ok,
            interventions_ok,
            candidate_actions_ok,
            subset_minimal,
            proof_witnesses_ok,
            bindings_ok,
        )
    )
    return VerificationResult(
        schema=certificate_schema,
        checker_version=checker_version,
        verification_bundle_sha256=None,
        expected_bundle_sha256=None,
        bundle_hash_matches=None,
        embedded_issuance_certificate_valid=None,
        passed=passed,
        attack_reachability_blocked=attack_ok,
        service_pairs_preserved=service_ok,
        policy_conflict_free=conflict_ok,
        rollback_schema_valid=rollback_ok,
        rollback_defined=rollback_ok,
        typed_sketch_schema_valid=typed_sketch_schema_ok,
        identifier_omission_valid=identifier_omission_ok,
        policy_equivalent=policy_ok,
        freshness_valid=freshness_ok,
        capsule_freshness_matches=True,
        capsule_budget_matches=True,
        submitted_witnesses_valid=True,
        proof_witnesses_valid=proof_witnesses_ok,
        threat_scope_valid=threat_ok,
        service_constraints_valid=service_constraints_ok,
        scope_declared=scope_ok,
        budget_respected=budget_ok,
        interventions_valid=interventions_ok,
        subset_minimal=subset_minimal,
        minimality_status=minimality_status,
        # Subset minimality is not an approximation-ratio or global-optimality
        # statement. Preserve the legacy field for schema compatibility while
        # declining to populate a mathematically misleading numeric bound.
        minimality_ratio_bound=None,
        residual_attack_pairs=len(residual_pairs),
        service_preservation=float(preservation),
        conflicts=len(conflicts),
        action_budget=action_budget,
        action_count=len(intervention_list),
        path_search_scope=scope_payload,
        freshness=dict(freshness or {}),
        verified_at=checked_at.isoformat().replace("+00:00", "Z"),
        bindings=bindings,
        witnesses=witnesses,
        messages=messages,
        candidate_actions_valid=candidate_actions_ok if strict_actions else None,
        candidate_action_count=len(candidate_list) if strict_actions else None,
    )


def verify_intervention_set(
    graph: nx.DiGraph,
    interventions: Sequence[Edge],
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    required_service_pairs: Sequence[Sequence[str]],
    rollback_condition: Dict[str, object] | None,
    identifier_omitting_typed_sketch: object | None,
    *,
    tau: float = 1.0,
    action_budget: int,
    path_scope: PathSearchScope | None = None,
    policy_bundle: Mapping[str, object] | None = None,
    freshness: Mapping[str, object] | None = None,
    verification_time: datetime | str | None = None,
    expected_bindings: Mapping[str, object] | None = None,
) -> VerificationResult:
    """Legacy v1 verifier retained only for v1 issuance/replay compatibility."""

    return _verify_intervention_set_core(
        graph,
        interventions,
        attack_sources,
        protected_assets,
        required_service_pairs,
        rollback_condition,
        identifier_omitting_typed_sketch,
        tau=tau,
        action_budget=action_budget,
        path_scope=path_scope,
        policy_bundle=policy_bundle,
        freshness=freshness,
        verification_time=verification_time,
        expected_bindings=expected_bindings,
        candidate_actions=None,
        proposal_bindings=None,
        strict_actions=False,
        explicit_verification_time=False,
        freshness_schema="cin.freshness.v1",
        certificate_schema=LEGACY_CERTIFICATE_SCHEMA,
        checker_version=LEGACY_CHECKER_VERSION,
    )


def verify_exact_proposal(
    graph: nx.DiGraph,
    interventions: object,
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    required_service_pairs: Sequence[Sequence[str]],
    rollback_condition: Dict[str, object] | None,
    identifier_omitting_typed_sketch: object | None,
    *,
    candidate_actions: object,
    proposal_bindings: Mapping[str, object],
    verification_time: datetime | str,
    tau: float = 1.0,
    action_budget: int,
    path_scope: PathSearchScope | None = None,
    policy_bundle: Mapping[str, object] | None = None,
    freshness: Mapping[str, object] | None = None,
    expected_bindings: Mapping[str, object] | None = None,
) -> VerificationResult:
    """Purely verify the exact v2 action set supplied at the admission boundary."""

    return _verify_intervention_set_core(
        graph,
        interventions,
        attack_sources,
        protected_assets,
        required_service_pairs,
        rollback_condition,
        identifier_omitting_typed_sketch,
        tau=tau,
        action_budget=action_budget,
        path_scope=path_scope,
        policy_bundle=policy_bundle,
        freshness=freshness,
        verification_time=verification_time,
        expected_bindings=expected_bindings,
        candidate_actions=candidate_actions,
        proposal_bindings=proposal_bindings,
        strict_actions=True,
        explicit_verification_time=True,
        freshness_schema="cin.freshness.v2",
        certificate_schema=CERTIFICATE_SCHEMA,
        checker_version=CHECKER_VERSION,
    )


def make_verification_bundle(
    graph: nx.DiGraph,
    capsule: Mapping[str, object],
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    required_service_pairs: Sequence[Sequence[str]],
    *,
    candidate_actions: object,
    tau: float,
    action_budget: int,
    path_scope: PathSearchScope,
    freshness: Mapping[str, object],
) -> Dict[str, object]:
    if (
        not isinstance(tau, (int, float))
        or isinstance(tau, bool)
        or not math.isfinite(float(tau))
        or float(tau) != 1.0
    ):
        raise ValueError("service threshold tau must equal 1.0 in this certificate schema")
    if not isinstance(attack_sources, (list, tuple)) or not isinstance(
        protected_assets, (list, tuple)
    ):
        raise ValueError(
            "threat scope attack_sources and protected_assets must be arrays"
        )
    require_graph_provenance(graph)
    candidates = list(
        canonical_action_set(candidate_actions, field="candidate_actions")
    )
    if not all(graph.has_edge(*edge) for edge in candidates):
        raise ValueError("candidate_actions contains an edge absent from the graph")
    issued_at = _parse_utc_timestamp(freshness.get("issued_at")) if isinstance(freshness, Mapping) else None
    if issued_at is None:
        raise ValueError("v2 freshness issued_at must be an explicit RFC 3339 timestamp")
    freshness_valid, freshness_messages = _validate_freshness_v2(
        freshness,
        graph,
        issued_at,
    )
    if not freshness_valid:
        raise ValueError(
            "invalid freshness declaration: " + "; ".join(freshness_messages)
        )
    return {
        "schema": "cin.verification-bundle.v2",
        "graph": graph_to_dict(graph),
        "threat_model": {
            "attack_sources": list(attack_sources),
            "protected_assets": list(protected_assets),
        },
        "constraints": {
            "required_service_pairs": [
                list(pair)
                for pair in canonical_service_pairs(
                    required_service_pairs, field="required_service_pairs"
                )
            ],
            "candidate_actions": [list(edge) for edge in candidates],
            "service_threshold": float(tau),
            "action_budget": action_budget,
            "path_search_scope": path_scope.to_dict(),
            "freshness": dict(freshness),
        },
        "capsule": canonical_json_value(capsule),
    }


def _certificate_matches_issuance_recomputation(
    embedded: Mapping[str, object],
    recomputed: Mapping[str, object],
    *,
    compare_verified_at: bool = False,
    type_sensitive: bool = False,
) -> bool:
    """Compare the complete issuance certificate except execution time's value."""

    if set(embedded) != set(recomputed):
        return False
    execution_dependent_fields = set() if compare_verified_at else {"verified_at"}
    for key in recomputed:
        if key in execution_dependent_fields:
            continue
        if type_sensitive:
            if not _type_sensitive_json_equal(embedded[key], recomputed[key]):
                return False
        elif _jsonable(embedded[key]) != _jsonable(recomputed[key]):
            return False
    return True


def _apply_capsule_copy_checks(
    result: VerificationResult,
    capsule: Mapping[str, object],
    constraints: Mapping[str, object],
    embedded_certificate: Mapping[str, object],
    *,
    type_sensitive: bool = False,
) -> None:
    if type_sensitive:
        capsule_freshness_matches = _type_sensitive_json_equal(
            capsule.get("freshness"), constraints.get("freshness")
        )
        capsule_budget_matches = _type_sensitive_json_equal(
            capsule.get("action_budget"), constraints.get("action_budget")
        )
        submitted_witnesses_valid = _type_sensitive_json_equal(
            embedded_certificate.get("witnesses"), result.witnesses
        )
    else:
        capsule_freshness_matches = _jsonable(
            capsule.get("freshness")
        ) == _jsonable(constraints.get("freshness"))
        capsule_budget_matches = capsule.get("action_budget") == constraints.get(
            "action_budget"
        )
        submitted_witnesses_valid = _jsonable(
            embedded_certificate.get("witnesses")
        ) == _jsonable(result.witnesses)
    result.capsule_freshness_matches = capsule_freshness_matches
    result.capsule_budget_matches = capsule_budget_matches
    result.submitted_witnesses_valid = submitted_witnesses_valid
    if not capsule_freshness_matches:
        result.passed = False
        result.messages.append(
            "capsule freshness declaration differs from the bound verification constraint"
        )
    if not capsule_budget_matches:
        result.passed = False
        result.messages.append(
            "capsule action_budget differs from the bound verification constraint"
        )
    if not submitted_witnesses_valid:
        result.passed = False
        result.messages.append(
            "submitted certificate witnesses differ from deterministic verifier recomputation"
        )


def _derive_v2_proposal_bindings(
    capsule: Mapping[str, object],
    graph: nx.DiGraph,
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    required_service_pairs: Sequence[Sequence[str]],
    *,
    candidate_actions: Sequence[Edge],
    tau: float,
) -> Tuple[Dict[str, object], List[Edge], List[Edge]]:
    raw = list(
        canonical_action_set(capsule.get("raw_proposal"), field="capsule.raw_proposal")
    )
    repaired = list(
        canonical_action_set(
            capsule.get("repaired_proposal"), field="capsule.repaired_proposal"
        )
    )
    transformation = capsule.get("proposal_transformation")
    if not isinstance(transformation, Mapping):
        raise ValueError("capsule proposal_transformation must be an object")
    transformation = canonical_json_value(dict(transformation))
    # Local import avoids the module-level cycle: repair uses the verifier's
    # strict action canonicalizer and stable hash, while bundle replay needs the
    # independently replayable transformation validator only after both modules
    # have finished loading.
    from .repair import validate_transformation

    transformation_ok, transformation_messages, replayed = validate_transformation(
        transformation,
        graph,
        attack_sources,
        protected_assets,
        required_service_pairs,
        tau=tau,
    )
    if not transformation_ok or replayed is None:
        raise ValueError(
            "invalid proposal transformation: "
            + "; ".join(transformation_messages)
        )
    if raw != list(replayed.raw_actions):
        raise ValueError(
            "capsule raw_proposal differs from deterministic transformation input"
        )
    if repaired != list(replayed.repaired_actions):
        raise ValueError(
            "capsule repaired_proposal differs from deterministic transformation output"
        )
    candidate_set = set(candidate_actions)
    if any(not graph.has_edge(*edge) for edge in raw + repaired):
        raise ValueError("raw or repaired proposal contains an edge absent from the graph")
    if any(edge not in candidate_set for edge in raw + repaired):
        raise ValueError("raw or repaired proposal contains an out-of-candidate action")

    transformation_version = replayed.transformation_version
    steps = [dict(step) for step in replayed.steps]
    return (
        {
            "raw_interventions_sha256": stable_hash([list(edge) for edge in raw]),
            "repaired_interventions_sha256": stable_hash(
                [list(edge) for edge in repaired]
            ),
            "repair_log_sha256": stable_hash(steps),
            "transformation_version": transformation_version,
        },
        raw,
        repaired,
    )


def _apply_v2_action_copy_checks(
    result: VerificationResult,
    capsule: Mapping[str, object],
    candidate_actions: Sequence[Edge],
    repaired_actions: Sequence[Edge],
) -> None:
    exact_interventions = list(
        canonical_action_set(
            capsule.get("interventions"), field="capsule.interventions"
        )
    )
    repaired_copy_matches = exact_interventions == list(repaired_actions)
    capsule_candidates = list(
        canonical_action_set(
            capsule.get("candidate_actions"), field="capsule.candidate_actions"
        )
    )
    candidate_copy_matches = capsule_candidates == list(candidate_actions)
    expected_capability = {
        "schema": "cin.candidate-action-capability.v1",
        "enforcement_domain": "directed-edge-denial-on-bound-observed-graph",
        "candidate_actions_sha256": stable_hash(
            [list(edge) for edge in candidate_actions]
        ),
        "candidate_action_count": len(candidate_actions),
    }
    capability = capsule.get("candidate_action_capability")
    capability_count = (
        capability.get("candidate_action_count")
        if isinstance(capability, Mapping)
        else None
    )
    capability_matches = (
        isinstance(capability, Mapping)
        and type(capability_count) is int
        and capability_count >= 0
        and capability_count == len(candidate_actions)
        and _type_sensitive_json_equal(dict(capability), expected_capability)
    )
    if not repaired_copy_matches:
        result.passed = False
        result.interventions_valid = False
        result.messages.append(
            "capsule interventions differ from the exact repaired proposal"
        )
    if not candidate_copy_matches:
        result.passed = False
        result.candidate_actions_valid = False
        result.messages.append(
            "capsule candidate_actions differ from bound constraint candidate_actions"
        )
    if not capability_matches:
        result.passed = False
        result.candidate_actions_valid = False
        result.messages.append(
            "capsule candidate_action_capability differs from bound candidate_actions"
        )


def _verify_capsule_bundle_v1(
    bundle: Mapping[str, object],
    *,
    expected_bundle_sha256: str,
    verification_time: datetime | str | None = None,
) -> VerificationResult:
    if (
        not isinstance(expected_bundle_sha256, str)
        or HEX64.fullmatch(expected_bundle_sha256) is None
    ):
        raise ValueError(
            "expected_bundle_sha256 must be an externally trusted lowercase SHA-256 digest"
        )
    actual_bundle_sha256 = verification_bundle_hash(bundle)
    bundle_hash_matches = hmac.compare_digest(
        actual_bundle_sha256,
        expected_bundle_sha256,
    )
    if bundle.get("schema") != "cin.verification-bundle.v1":
        raise ValueError("unsupported verification bundle schema")
    graph = graph_from_dict(dict(bundle["graph"]))
    threat = dict(bundle["threat_model"])
    constraints = dict(bundle["constraints"])
    capsule = dict(bundle["capsule"])
    if capsule.get("schema") != "cin.defense-capsule.v1":
        raise ValueError("unsupported defense capsule schema")
    scope_data = dict(constraints["path_search_scope"])
    scope = PathSearchScope(**scope_data)
    policy_bundle = {
        "abstract": capsule.get("abstract_policy_json"),
        "kubernetes_intent": capsule.get("kubernetes_network_policy_yaml"),
        "ebpf_pseudo": capsule.get("ebpf_xdp_pseudo_policy"),
        "sdn_pseudo": capsule.get("sdn_flow_rule_pseudo_policy"),
    }
    embedded_certificate = capsule.get("service_safety_certificate")
    if not isinstance(embedded_certificate, Mapping):
        raise ValueError("embedded certificate is missing")
    if embedded_certificate.get("schema") != LEGACY_CERTIFICATE_SCHEMA:
        raise ValueError("unsupported embedded certificate schema")
    if embedded_certificate.get("checker_version") != LEGACY_CHECKER_VERSION:
        raise ValueError("unsupported embedded checker version")
    freshness_payload = constraints.get("freshness")
    if not isinstance(freshness_payload, Mapping):
        raise ValueError(
            "verification bundle freshness declaration is missing or is not an object"
        )
    freshness = dict(freshness_payload)
    tau_value = constraints.get("service_threshold")
    if not isinstance(tau_value, (int, float)) or isinstance(tau_value, bool) or not math.isfinite(float(tau_value)) or float(tau_value) != 1.0:
        raise ValueError("service threshold must be the finite number 1.0")
    budget_value = constraints.get("action_budget")
    if not isinstance(budget_value, int) or isinstance(budget_value, bool) or budget_value < 0:
        raise ValueError("action budget must be a non-negative integer")
    expected_bindings = embedded_certificate.get("bindings") if isinstance(embedded_certificate, Mapping) else {}
    result = verify_intervention_set(
        graph,
        capsule.get("interventions", []),
        threat.get("attack_sources", []),
        threat.get("protected_assets", []),
        constraints.get("required_service_pairs", []),
        capsule.get("rollback_condition"),
        capsule.get("identifier_omitting_typed_sketch"),
        tau=float(tau_value),
        action_budget=budget_value,
        path_scope=scope,
        policy_bundle=policy_bundle,
        freshness=freshness,
        verification_time=verification_time,
        expected_bindings=expected_bindings,
    )
    _apply_capsule_copy_checks(result, capsule, constraints, embedded_certificate)

    issuance_certificate_valid = False
    issuance_time = freshness.get("issued_at")
    try:
        issuance_result = verify_intervention_set(
            graph,
            capsule.get("interventions", []),
            threat.get("attack_sources", []),
            threat.get("protected_assets", []),
            constraints.get("required_service_pairs", []),
            capsule.get("rollback_condition"),
            capsule.get("identifier_omitting_typed_sketch"),
            tau=float(tau_value),
            action_budget=budget_value,
            path_scope=scope,
            policy_bundle=policy_bundle,
            freshness=freshness,
            verification_time=issuance_time,
        )
    except (TypeError, ValueError):
        issuance_result = None
    if issuance_result is not None:
        _apply_capsule_copy_checks(
            issuance_result,
            capsule,
            constraints,
            embedded_certificate,
        )
        issuance_certificate_valid = _certificate_matches_issuance_recomputation(
            embedded_certificate,
            issuance_result.to_dict(),
        )

    result.verification_bundle_sha256 = actual_bundle_sha256
    result.expected_bundle_sha256 = expected_bundle_sha256
    result.bundle_hash_matches = bundle_hash_matches
    result.embedded_issuance_certificate_valid = issuance_certificate_valid
    if not bundle_hash_matches:
        result.passed = False
        result.messages.append(
            "verification bundle SHA-256 does not match the externally trusted expected root"
        )
    if not issuance_certificate_valid:
        result.passed = False
        result.messages.append(
            "embedded issuance certificate differs from deterministic issuance-time recomputation"
        )
    return result


def _verify_capsule_bundle_v2(
    bundle: Mapping[str, object],
    *,
    expected_bundle_sha256: str,
    verification_time: datetime | str | None,
) -> VerificationResult:
    if verification_time is None:
        raise ValueError("verification_time is required for v2 bundle replay")
    if (
        type(expected_bundle_sha256) is not str
        or HEX64.fullmatch(expected_bundle_sha256) is None
    ):
        raise ValueError(
            "expected_bundle_sha256 must be an externally trusted lowercase SHA-256 digest"
        )
    actual_bundle_sha256 = verification_bundle_hash(bundle)
    bundle_hash_matches = hmac.compare_digest(
        actual_bundle_sha256, expected_bundle_sha256
    )
    graph_payload = bundle.get("graph")
    threat_payload = bundle.get("threat_model")
    constraints_payload = bundle.get("constraints")
    capsule_payload = bundle.get("capsule")
    if not all(
        isinstance(value, Mapping)
        for value in (graph_payload, threat_payload, constraints_payload, capsule_payload)
    ):
        raise ValueError("v2 verification bundle contains a malformed object")
    if not isinstance(threat_payload.get("attack_sources"), list) or not isinstance(
        threat_payload.get("protected_assets"), list
    ):
        raise ValueError(
            "v2 threat_model attack_sources and protected_assets must be arrays"
        )
    graph = graph_from_dict(graph_payload)
    threat = dict(threat_payload)
    constraints = dict(constraints_payload)
    capsule = dict(capsule_payload)
    if capsule.get("schema") != "cin.defense-capsule.v2":
        raise ValueError("unsupported defense capsule schema")

    candidate_actions = list(
        canonical_action_set(
            constraints.get("candidate_actions"),
            field="constraints.candidate_actions",
        )
    )
    scope_data = constraints.get("path_search_scope")
    if not isinstance(scope_data, Mapping):
        raise ValueError("path_search_scope is missing or malformed")
    scope = PathSearchScope(**dict(scope_data))
    policy_bundle = {
        "abstract": capsule.get("abstract_policy_json"),
        "kubernetes_intent": capsule.get("kubernetes_network_policy_yaml"),
        "ebpf_pseudo": capsule.get("ebpf_xdp_pseudo_policy"),
        "sdn_pseudo": capsule.get("sdn_flow_rule_pseudo_policy"),
    }
    embedded_certificate = capsule.get("service_safety_certificate")
    if not isinstance(embedded_certificate, Mapping):
        raise ValueError("embedded certificate is missing")
    if embedded_certificate.get("schema") != CERTIFICATE_SCHEMA:
        raise ValueError("unsupported embedded certificate schema")
    if embedded_certificate.get("checker_version") != CHECKER_VERSION:
        raise ValueError("unsupported embedded checker version")
    freshness_payload = constraints.get("freshness")
    if not isinstance(freshness_payload, Mapping):
        raise ValueError(
            "verification bundle freshness declaration is missing or is not an object"
        )
    freshness = dict(freshness_payload)
    if freshness.get("schema") != "cin.freshness.v2":
        raise ValueError("v2 bundle requires cin.freshness.v2")
    embedded_verified_at = _parse_utc_timestamp(embedded_certificate.get("verified_at"))
    bound_issued_at = _parse_utc_timestamp(freshness.get("issued_at"))
    embedded_verified_at_valid = (
        embedded_verified_at is not None
        and bound_issued_at is not None
        and embedded_verified_at == bound_issued_at
    )
    tau_value = constraints.get("service_threshold")
    if (
        not isinstance(tau_value, (int, float))
        or isinstance(tau_value, bool)
        or not math.isfinite(float(tau_value))
        or float(tau_value) != 1.0
    ):
        raise ValueError("service threshold must be the finite number 1.0")
    budget_value = constraints.get("action_budget")
    if type(budget_value) is not int or budget_value < 0:
        raise ValueError("action budget must be a non-negative integer")

    proposal_bindings, _raw_actions, repaired_actions = (
        _derive_v2_proposal_bindings(
            capsule,
            graph,
            threat.get("attack_sources", []),
            threat.get("protected_assets", []),
            constraints.get("required_service_pairs", []),
            candidate_actions=candidate_actions,
            tau=float(tau_value),
        )
    )
    expected_bindings = embedded_certificate.get("bindings")
    if not isinstance(expected_bindings, Mapping):
        raise ValueError("embedded certificate bindings are missing")
    result = verify_exact_proposal(
        graph,
        capsule.get("interventions"),
        threat.get("attack_sources", []),
        threat.get("protected_assets", []),
        constraints.get("required_service_pairs", []),
        capsule.get("rollback_condition"),
        capsule.get("identifier_omitting_typed_sketch"),
        candidate_actions=candidate_actions,
        proposal_bindings=proposal_bindings,
        verification_time=verification_time,
        tau=float(tau_value),
        action_budget=budget_value,
        path_scope=scope,
        policy_bundle=policy_bundle,
        freshness=freshness,
        expected_bindings=expected_bindings,
    )
    _apply_capsule_copy_checks(
        result,
        capsule,
        constraints,
        embedded_certificate,
        type_sensitive=True,
    )
    _apply_v2_action_copy_checks(
        result, capsule, candidate_actions, repaired_actions
    )
    capsule_contract_messages, policy_method_matches = (
        _v2_capsule_contract_errors(
            capsule,
            graph,
            threat.get("attack_sources", []),
            threat.get("protected_assets", []),
            repaired_actions,
            scope,
            result.bindings,
        )
    )
    _apply_v2_capsule_contract_checks(
        result,
        capsule_contract_messages,
        policy_method_matches=policy_method_matches,
    )

    issuance_certificate_valid = False
    try:
        issuance_result = verify_exact_proposal(
            graph,
            capsule.get("interventions"),
            threat.get("attack_sources", []),
            threat.get("protected_assets", []),
            constraints.get("required_service_pairs", []),
            capsule.get("rollback_condition"),
            capsule.get("identifier_omitting_typed_sketch"),
            candidate_actions=candidate_actions,
            proposal_bindings=proposal_bindings,
            verification_time=freshness.get("issued_at"),
            tau=float(tau_value),
            action_budget=budget_value,
            path_scope=scope,
            policy_bundle=policy_bundle,
            freshness=freshness,
        )
    except (TypeError, ValueError):
        issuance_result = None
    if issuance_result is not None:
        _apply_capsule_copy_checks(
            issuance_result,
            capsule,
            constraints,
            embedded_certificate,
            type_sensitive=True,
        )
        _apply_v2_action_copy_checks(
            issuance_result, capsule, candidate_actions, repaired_actions
        )
        _apply_v2_capsule_contract_checks(
            issuance_result,
            capsule_contract_messages,
            policy_method_matches=policy_method_matches,
        )
        issuance_certificate_valid = (
            embedded_verified_at_valid
            and _certificate_matches_issuance_recomputation(
                embedded_certificate,
                issuance_result.to_dict(),
                compare_verified_at=True,
                type_sensitive=True,
            )
        )

    result.verification_bundle_sha256 = actual_bundle_sha256
    result.expected_bundle_sha256 = expected_bundle_sha256
    result.bundle_hash_matches = bundle_hash_matches
    result.embedded_issuance_certificate_valid = issuance_certificate_valid
    if not bundle_hash_matches:
        result.passed = False
        result.messages.append(
            "verification bundle SHA-256 does not match the externally trusted expected root"
        )
    if not issuance_certificate_valid:
        result.passed = False
        result.messages.append(
            "embedded issuance certificate differs from deterministic issuance-time recomputation"
        )
    if not embedded_verified_at_valid:
        result.passed = False
        result.messages.append(
            "embedded certificate verified_at must be a zoned RFC 3339 timestamp "
            "equal to bound issued_at"
        )
    try:
        _validate_v2_bundle_field_sets(bundle)
    except ValueError as error:
        # Preserve the verifier API for parseable-but-invalid bundles: schema
        # violations are negative verification results, not parser failures.
        # Inputs that cannot be replayed at all still fail earlier with a
        # ValueError (for example, a fabricated transformation log).
        result.passed = False
        result.messages.append(str(error))
    return result


def verify_capsule_bundle(
    bundle: Mapping[str, object],
    *,
    expected_bundle_sha256: str,
    verification_time: datetime | str | None = None,
) -> VerificationResult:
    """Replay a strict v2 bundle or a byte-compatible legacy v1 bundle."""

    if not isinstance(bundle, Mapping):
        raise TypeError("verification bundle must be a mapping")
    canonical_json_value(bundle)
    schema = bundle.get("schema")
    if schema == "cin.verification-bundle.v1":
        return _verify_capsule_bundle_v1(
            bundle,
            expected_bundle_sha256=expected_bundle_sha256,
            verification_time=verification_time,
        )
    if schema == "cin.verification-bundle.v2":
        return _verify_capsule_bundle_v2(
            bundle,
            expected_bundle_sha256=expected_bundle_sha256,
            verification_time=verification_time,
        )
    raise ValueError("unsupported verification bundle schema")

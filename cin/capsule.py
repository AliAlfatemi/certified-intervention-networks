"""Budgeted intervention planning and defense-capsule construction."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Mapping, Sequence, Tuple
import hashlib
import math

import networkx as nx
import yaml

from .canonical_json import canonical_json_bytes
from .risk_model import ObservedRiskScorer
from .simulator import (
    PathSearchScope,
    all_attack_routes_blocked,
    apply_interventions,
    enumerate_attack_paths,
    normalize_service_pairs,
    path_edges,
    reachable_pairs,
    residual_attack_witnesses,
    service_preservation,
)
from .repair import (
    RepairResult,
    SubsetMinimalRepair,
    identity_transformation,
    validate_transformation,
)
from .verifier import (
    _parse_utc_timestamp,
    canonical_action_set,
    canonical_service_pairs,
    require_graph_provenance,
    stable_hash,
    verify_exact_proposal,
)

Edge = Tuple[str, str]
Path = Tuple[str, ...]

# A fixed, finite replay horizon is part of the offline experiment protocol so
# historical runs remain byte-reproducible.  It is not a recommended
# operational cache lifetime: callers integrating live discovery should pass a
# tighter domain-specific values through the two explicit v2 age policies.
EXPERIMENT_DECISION_DELAY_SECONDS = 60
EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS = 5 * 60
EXPERIMENT_MAX_CAPSULE_AGE_SECONDS = 30 * 24 * 60 * 60
# Compatibility name for archived v1 callers only. New v2 issuance uses the
# two independent policies above.
EXPERIMENT_FRESHNESS_MAX_AGE_SECONDS = EXPERIMENT_MAX_CAPSULE_AGE_SECONDS


@dataclass
class DefenseCapsule:
    capsule_id: str
    schema: str
    method: str
    raw_proposal: List[Edge]
    repaired_proposal: List[Edge]
    proposal_transformation: Dict[str, object]
    attack_reachability_invariant: Dict[str, object]
    interventions: List[Edge]
    candidate_actions: List[Edge]
    candidate_action_capability: Dict[str, object]
    action_budget: int
    freshness: Dict[str, object]
    abstract_policy_json: Dict[str, object]
    kubernetes_network_policy_yaml: str
    ebpf_xdp_pseudo_policy: str
    sdn_flow_rule_pseudo_policy: str
    service_safety_certificate: Dict[str, object]
    rollback_condition: Dict[str, object]
    identifier_omitting_typed_sketch: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["raw_proposal"] = [list(edge) for edge in self.raw_proposal]
        payload["repaired_proposal"] = [list(edge) for edge in self.repaired_proposal]
        payload["interventions"] = [list(edge) for edge in self.interventions]
        payload["candidate_actions"] = [list(edge) for edge in self.candidate_actions]
        return payload


def experiment_issuance_time(graph: nx.DiGraph) -> datetime:
    """Return the explicitly declared historical decision time for experiments."""

    snapshot_time = require_graph_provenance(graph)
    return snapshot_time + timedelta(seconds=EXPERIMENT_DECISION_DELAY_SECONDS)


def _edge_pattern(graph: nx.DiGraph, edge: Edge) -> Dict[str, object]:
    source, destination = edge
    data = graph[source][destination]
    return {
        "src_type": graph.nodes[source].get("type"),
        "dst_type": graph.nodes[destination].get("type"),
        "src_tier": graph.nodes[source].get("tier"),
        "dst_tier": graph.nodes[destination].get("tier"),
        "relation": data.get("kind"),
        "cross_dept": canonical_json_bytes(
            graph.nodes[source].get("dept")
        )
        != canonical_json_bytes(graph.nodes[destination].get("dept")),
    }


def build_invariant(
    graph: nx.DiGraph,
    attack_paths: Sequence[Path],
    interventions: Sequence[Edge],
) -> Dict[str, object]:
    typed_paths = []
    for path in attack_paths[:20]:
        typed_paths.append(
            {
                "node_types": [graph.nodes[node].get("type") for node in path if node in graph],
                "edge_relations": [
                    graph[source][destination].get("kind")
                    for source, destination in zip(path[:-1], path[1:])
                    if graph.has_edge(source, destination)
                ],
                "tier_sequence": [graph.nodes[node].get("tier") for node in path if node in graph],
            }
        )
    return {
        "schema": "cin.attack-reachability-invariant.v1",
        "description": "Typed source-to-protected-asset reachability patterns and the proposed cut-edge patterns.",
        "claim_scope": "model-relative directed graph reachability",
        "typed_path_patterns": typed_paths,
        "cut_edge_patterns": [_edge_pattern(graph, edge) for edge in interventions if graph.has_edge(*edge)],
    }


def compile_policies(
    graph: nx.DiGraph,
    interventions: Sequence[Edge],
    method: str,
) -> Tuple[Dict[str, object], str, str, str]:
    """Emit bound policy intents.

    The non-JSON artifacts are deliberately labelled intents/pseudocode.  The
    verifier checks their semantic hash but does not claim platform deployment.
    """

    edges = sorted(dict.fromkeys(tuple(edge) for edge in interventions))
    intervention_hash = stable_hash([list(edge) for edge in edges])
    rules = [
        {
            "id": f"deny-{index:03d}",
            "action": "deny-edge",
            "source": source,
            "destination": destination,
            "relation": graph[source][destination].get("kind", "comm") if graph.has_edge(source, destination) else "unknown",
        }
        for index, (source, destination) in enumerate(edges)
    ]
    abstract = {
        "schema": "cin.policy.v1",
        "method": method,
        "semantics": "deny exactly the listed directed graph edges; otherwise preserve prior policy",
        "manifest": {"intervention_hash": intervention_hash, "rule_count": len(rules)},
        "rules": rules,
    }
    kubernetes_intent = {
        "apiVersion": "research.cin.ai/v1alpha1",
        "kind": "NetworkPolicyIntent",
        "metadata": {
            "name": f"cin-{stable_hash({'method': method, 'edges': edges})[:12]}",
            "annotations": {
                "cin-intervention-hash": intervention_hash,
                "cin-status": "research-intent-not-deployable-networkpolicy",
            },
        },
        "spec": {"semantics": "deny-directed-edges", "rules": rules},
    }
    kubernetes_yaml = (
        f"# cin-intervention-hash: {intervention_hash}\n"
        + yaml.safe_dump(kubernetes_intent, sort_keys=False)
    )
    ebpf_lines = [
        f"// cin-intervention-hash: {intervention_hash}",
        "// Research pseudocode; deployment requires an independently validated compiler.",
        "int cin_policy(flow_key key) {",
    ]
    for source, destination in edges:
        ebpf_lines.append(f'  if (match_edge(key, "{source}", "{destination}")) return DROP;')
    ebpf_lines.extend(["  return PRESERVE_PRIOR_POLICY;", "}"])
    sdn_lines = [
        f"# cin-intervention-hash: {intervention_hash}",
        "# Research pseudocode; not an installable controller configuration.",
    ]
    sdn_lines.extend(
        f"deny-directed-edge source={source} destination={destination}" for source, destination in edges
    )
    return abstract, kubernetes_yaml, "\n".join(ebpf_lines), "\n".join(sdn_lines)


def build_identifier_omitting_typed_sketch(
    graph: nx.DiGraph,
    interventions: Sequence[Edge],
    *,
    unsafe_raw: bool = False,
) -> Dict[str, object]:
    if unsafe_raw:
        return {
            "schema": "cin.typed-sketch.invalid-raw.v0",
            "type": "raw-edge-rules",
            "version": 0,
            "edges": [list(edge) for edge in interventions],
            "raw_identifier_count": len(set(node for edge in interventions for node in edge)),
        }
    patterns = [_edge_pattern(graph, edge) for edge in interventions if graph.has_edge(*edge)]
    return {
        "schema": "cin.typed-sketch.v1",
        "type": "typed-edge-patterns",
        "version": 1,
        "pattern_hashes": [stable_hash(pattern) for pattern in patterns],
        "typed_patterns": patterns,
        "raw_identifier_count": 0,
        "identifier_namespace": "omitted",
    }


def _safe_feasible(
    graph: nx.DiGraph,
    interventions: Sequence[Edge],
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    required_service_pairs: Sequence[Sequence[str]],
    tau: float,
) -> bool:
    intervened = apply_interventions(graph, interventions)
    return all_attack_routes_blocked(intervened, attack_sources, protected_assets) and service_preservation(
        intervened, required_service_pairs
    ) >= tau


def prune_redundant_interventions(
    graph: nx.DiGraph,
    interventions: Sequence[Edge],
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    required_service_pairs: Sequence[Sequence[str]],
    tau: float,
) -> List[Edge]:
    """Compatibility view of the explicit proposer-side repair result."""

    return list(
        SubsetMinimalRepair.apply(
            graph,
            interventions,
            attack_sources,
            protected_assets,
            required_service_pairs,
            tau=tau,
        ).repaired_actions
    )


def build_capsule(
    *,
    method: str,
    graph: nx.DiGraph,
    proposal_transformation: RepairResult,
    candidate_actions: Sequence[Edge],
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    required_service_pairs: Sequence[Sequence[str]],
    action_budget: int,
    tau: float = 1.0,
    path_scope: PathSearchScope | None = None,
    issued_at: datetime | str,
    max_snapshot_age_at_issue_seconds: int,
    max_capsule_age_seconds: int,
) -> DefenseCapsule:
    """Compile and verify an already explicit proposal transformation.

    ``graph.graph`` must explicitly supply ``snapshot_id``, a positive integer
    ``topology_version``, and a timezone-qualified ``snapshot_time``.  The
    issuance time and both age policies are mandatory caller inputs. This
    function validates but never changes either the raw or repaired proposal.
    """

    if type(method) is not str or not method or method != method.strip():
        raise ValueError("method must be a non-empty, whitespace-trimmed string")

    if (
        not isinstance(tau, (int, float))
        or isinstance(tau, bool)
        or not math.isfinite(float(tau))
        or float(tau) != 1.0
    ):
        raise ValueError("service threshold tau must equal 1.0 in this certificate schema")
    tau = float(tau)
    if type(action_budget) is not int or action_budget < 0:
        raise ValueError("action_budget must be a non-negative integer")
    snapshot_time = require_graph_provenance(graph)
    if isinstance(issued_at, datetime):
        if issued_at.tzinfo is None or issued_at.utcoffset() is None:
            raise ValueError("issued_at must be timezone-aware")
        issued = issued_at.astimezone(timezone.utc)
    else:
        issued = _parse_utc_timestamp(issued_at)
        if issued is None:
            raise ValueError("issued_at must be a zoned RFC 3339 timestamp")
    if (
        type(max_snapshot_age_at_issue_seconds) is not int
        or max_snapshot_age_at_issue_seconds < 0
    ):
        raise ValueError(
            "max_snapshot_age_at_issue_seconds must be a non-negative integer"
        )
    if type(max_capsule_age_seconds) is not int or max_capsule_age_seconds <= 0:
        raise ValueError("max_capsule_age_seconds must be a positive integer")
    scope = path_scope or PathSearchScope()
    strict_required_pairs = list(
        canonical_service_pairs(
            required_service_pairs, field="required_service_pairs"
        )
    )
    candidates = canonical_action_set(candidate_actions, field="candidate_actions")
    if any(not graph.has_edge(*edge) for edge in candidates):
        raise ValueError("candidate_actions contains an edge absent from the bound graph")
    transformation_ok, transformation_messages, replayed = validate_transformation(
        proposal_transformation.to_dict(),
        graph,
        attack_sources,
        protected_assets,
        strict_required_pairs,
        tau=tau,
    )
    if not transformation_ok or replayed is None:
        raise ValueError(
            "invalid proposal transformation: " + "; ".join(transformation_messages)
        )
    raw_interventions = list(proposal_transformation.raw_actions)
    selected = list(proposal_transformation.repaired_actions)
    if selected != list(replayed.repaired_actions):
        raise ValueError("proposal transformation object differs from deterministic replay")
    candidate_set = set(candidates)
    if any(edge not in candidate_set for edge in raw_interventions + selected):
        raise ValueError("proposal contains an action outside candidate_actions")
    if len(selected) > action_budget:
        raise ValueError(
            f"proposal contains {len(selected)} actions but the exogenous budget is {action_budget}"
        )
    attack_paths = enumerate_attack_paths(
        graph,
        attack_sources,
        protected_assets,
        cutoff=scope.cutoff_edges,
        max_paths_per_pair=scope.max_paths_per_pair,
        max_expansions_per_pair=scope.max_expansions_per_pair,
    )
    abstract, kubernetes, ebpf, sdn = compile_policies(graph, selected, method)
    policy_bundle = {
        "abstract": abstract,
        "kubernetes_intent": kubernetes,
        "ebpf_pseudo": ebpf,
        "sdn_pseudo": sdn,
    }
    typed_sketch = build_identifier_omitting_typed_sketch(graph, selected)
    rollback = {
        "schema": "cin.rollback.v1",
        "metric": "service_endpoint_reachability_fraction",
        "operator": "<",
        "threshold": float(tau),
        "window_seconds": 300,
        "consecutive_windows": 1,
        "action": "restore_previous_policy",
        "restore_policy_hash": stable_hash({"schema": "cin.policy.v1", "state": "pre-capsule"}),
    }
    try:
        expires = issued + timedelta(seconds=max_capsule_age_seconds)
    except OverflowError as error:
        raise ValueError(
            "max_capsule_age_seconds exceeds the supported timestamp range"
        ) from error
    freshness = {
        "schema": "cin.freshness.v2",
        "snapshot_id": graph.graph["snapshot_id"],
        "topology_version": graph.graph["topology_version"],
        "snapshot_time": snapshot_time.isoformat().replace("+00:00", "Z"),
        "issued_at": issued.isoformat().replace("+00:00", "Z"),
        "expires_at": expires.isoformat().replace("+00:00", "Z"),
        "max_snapshot_age_at_issue_seconds": max_snapshot_age_at_issue_seconds,
        "max_capsule_age_seconds": max_capsule_age_seconds,
    }
    transformation_payload = proposal_transformation.to_dict()
    proposal_bindings = {
        "raw_interventions_sha256": proposal_transformation.raw_actions_sha256,
        "repaired_interventions_sha256": proposal_transformation.repaired_actions_sha256,
        "repair_log_sha256": proposal_transformation.repair_log_sha256,
        "transformation_version": proposal_transformation.transformation_version,
    }
    certificate = verify_exact_proposal(
        graph,
        selected,
        attack_sources,
        protected_assets,
        strict_required_pairs,
        rollback,
        typed_sketch,
        tau=tau,
        action_budget=action_budget,
        candidate_actions=candidates,
        path_scope=scope,
        policy_bundle=policy_bundle,
        freshness=freshness,
        # Issuance-time self-check. Serialized bundles never carry or control
        # the independent verifier's clock.
        verification_time=issued,
        proposal_bindings=proposal_bindings,
    ).to_dict()
    invariant = build_invariant(graph, attack_paths, selected)
    capsule_id = f"cin-{stable_hash({'method': method, 'bindings': certificate['bindings']})[:20]}"
    return DefenseCapsule(
        capsule_id=capsule_id,
        schema="cin.defense-capsule.v2",
        method=method,
        raw_proposal=raw_interventions,
        repaired_proposal=selected,
        proposal_transformation=transformation_payload,
        attack_reachability_invariant=invariant,
        interventions=selected,
        candidate_actions=list(candidates),
        candidate_action_capability={
            "schema": "cin.candidate-action-capability.v1",
            "enforcement_domain": "directed-edge-denial-on-bound-observed-graph",
            "candidate_actions_sha256": stable_hash(
                [list(edge) for edge in candidates]
            ),
            "candidate_action_count": len(candidates),
        },
        action_budget=action_budget,
        freshness=freshness,
        abstract_policy_json=abstract,
        kubernetes_network_policy_yaml=kubernetes,
        ebpf_xdp_pseudo_policy=ebpf,
        sdn_flow_rule_pseudo_policy=sdn,
        service_safety_certificate=certificate,
        rollback_condition=rollback,
        identifier_omitting_typed_sketch=typed_sketch,
    )


class CertifiedInterventionPlanner:
    """Risk-guided, service-aware greedy cut planner with an exogenous budget."""

    def __init__(
        self,
        tau: float = 1.0,
        cutoff: int = 8,
        max_paths_per_pair: int = 80,
        max_expansions_per_pair: int = 600,
        lambda_complexity: float = 0.03,
    ):
        if (
            not isinstance(tau, (int, float))
            or isinstance(tau, bool)
            or not math.isfinite(float(tau))
            or float(tau) != 1.0
        ):
            raise ValueError("service threshold tau must equal 1.0 in this certificate schema")
        self.tau = float(tau)
        self.scope = PathSearchScope(cutoff, max_paths_per_pair, max_expansions_per_pair)
        self.lambda_complexity = lambda_complexity

    def synthesize(
        self,
        graph: nx.DiGraph,
        attack_sources: Sequence[str],
        protected_assets: Sequence[str],
        required_service_pairs: Sequence[Sequence[str]],
        model: ObservedRiskScorer,
        *,
        action_budget: int,
        candidate_actions: Sequence[Edge],
        variant: str = "full",
    ) -> DefenseCapsule:
        if type(action_budget) is not int or action_budget < 0:
            raise ValueError("action_budget must be a non-negative integer")
        supported_variants = {
            "full",
            "uniform_scores",
            "no_service_guard",
            "bounded_only",
            "random_tiebreak",
            "no_reverse_delete",
        }
        if variant not in supported_variants:
            raise ValueError(f"unsupported planner variant {variant!r}")
        required_pairs = list(
            canonical_service_pairs(
                required_service_pairs, field="required_service_pairs"
            )
        )
        candidates_allowed = canonical_action_set(
            candidate_actions, field="candidate_actions"
        )
        candidate_set = set(candidates_allowed)
        if any(not graph.has_edge(*edge) for edge in candidates_allowed):
            raise ValueError("candidate_actions contains an edge absent from the graph")
        scores = dict(model.scores())
        if variant == "uniform_scores":
            scores = {edge: 0.5 for edge in graph.edges}
        elif variant == "random_tiebreak":
            scores = {
                edge: int(hashlib.sha256(repr(edge).encode("utf-8")).hexdigest(), 16) % 1_000_000 / 1_000_000
                for edge in graph.edges
            }

        selected: List[Edge] = []
        while len(selected) < action_budget:
            current = apply_interventions(graph, selected)
            if all_attack_routes_blocked(current, attack_sources, protected_assets):
                break
            planning_paths = enumerate_attack_paths(
                current,
                attack_sources,
                protected_assets,
                cutoff=self.scope.cutoff_edges,
                max_paths_per_pair=self.scope.max_paths_per_pair,
                max_expansions_per_pair=self.scope.max_expansions_per_pair,
            )
            if variant != "bounded_only":
                # Ensure routes longer than the heuristic cutoff cannot silently
                # disappear from planning: add one exact shortest witness per
                # residual source/asset pair.
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
            if not planning_paths:
                break
            candidates = sorted(
                {
                    edge
                    for path in planning_paths
                    for edge in path_edges(path)
                    if edge in candidate_set and edge not in selected
                }
            )
            current_residual_pairs = len(reachable_pairs(current, attack_sources, protected_assets))
            best_edge: Edge | None = None
            best_key: Tuple[float, float, float, float] | None = None
            for edge in candidates:
                trial_graph = current.copy()
                if trial_graph.has_edge(*edge):
                    trial_graph.remove_edge(*edge)
                if variant != "no_service_guard" and service_preservation(trial_graph, required_pairs) + 1e-12 < self.tau:
                    continue
                hit_count = sum(1 for path in planning_paths if edge in path_edges(path))
                pair_gain = current_residual_pairs - len(reachable_pairs(trial_graph, attack_sources, protected_assets))
                cost = float(graph[edge[0]][edge[1]].get("action_cost", 1.0))
                # Security coverage is lexicographically primary. Observed risk
                # only breaks structural ties, preventing a noisy score from
                # sacrificing an exactly measurable reachability gain.
                key = (
                    float(pair_gain),
                    float(hit_count),
                    scores.get(edge, 0.0),
                    -self.lambda_complexity * cost,
                )
                if best_key is None or key > best_key or (key == best_key and (best_edge is None or edge < best_edge)):
                    best_edge = edge
                    best_key = key
            if best_edge is None:
                break
            selected.append(best_edge)

        transformation = (
            identity_transformation(selected)
            if variant == "no_reverse_delete"
            else SubsetMinimalRepair.apply(
                graph,
                selected,
                attack_sources,
                protected_assets,
                required_pairs,
                tau=self.tau,
            )
        )
        return build_capsule(
            method="CIN-CertifiedGreedy" if variant == "full" else f"CIN-Ablation-{variant}",
            graph=graph,
            proposal_transformation=transformation,
            candidate_actions=candidates_allowed,
            attack_sources=attack_sources,
            protected_assets=protected_assets,
            required_service_pairs=required_pairs,
            action_budget=action_budget,
            tau=self.tau,
            path_scope=self.scope,
            issued_at=experiment_issuance_time(graph),
            max_snapshot_age_at_issue_seconds=EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
            max_capsule_age_seconds=EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
        )

"""Evaluation metrics for action-count-bounded graph interventions."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import json
import time

import networkx as nx

from .canonical_json import canonical_json_bytes
from .graph_generator import NetworkInstance
from .capsule import DefenseCapsule
from .simulator import (
    all_attack_routes_blocked,
    apply_interventions,
    enumerate_attack_paths,
    path_edges,
    reachable_pairs,
    risk_mass,
    service_disrupted_pairs,
    service_preservation,
    PathSearchScope,
)
from .verifier import (
    canonical_action_set,
    make_verification_bundle,
    stable_hash,
    verification_bundle_hash,
    verify_capsule_bundle,
)

Edge = Tuple[str, str]


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return default if abs(denominator) < 1e-12 else numerator / denominator


def _family_reduction(paths, interventions: Sequence[Edge]) -> float:
    paths = list(paths)
    if not paths:
        return float("nan")
    blocked = sum(1 for path in paths if path_edges(path) & set(interventions))
    return blocked / len(paths)


def _proposal_outcomes(
    instance: NetworkInstance,
    actions: Sequence[Edge],
    *,
    candidate_actions: Sequence[Edge],
    action_budget: int,
) -> Dict[str, object]:
    """Evaluate one supplied action set without repairing or normalizing it."""

    canonical = canonical_action_set(actions, field="proposal")
    candidate_set = set(candidate_actions)
    rejection_codes: List[str] = []
    if any(not instance.observed_graph.has_edge(*edge) for edge in canonical):
        rejection_codes.append("action_absent_from_graph")
    if any(edge not in candidate_set for edge in canonical):
        rejection_codes.append("out_of_candidate")
    if len(canonical) > action_budget:
        rejection_codes.append("over_budget")
    observed_after = apply_interventions(instance.observed_graph, canonical)
    attack_blocked = all_attack_routes_blocked(
        observed_after, instance.attack_sources, instance.protected_assets
    )
    service_safe = (
        service_preservation(
            observed_after, instance.required_service_pairs_observed
        )
        >= 1.0 - 1e-12
    )
    if not attack_blocked:
        rejection_codes.append("threat_reachable")
    if not service_safe:
        rejection_codes.append("service_violation")
    subset_minimal = False
    if attack_blocked and service_safe:
        subset_minimal = all(
            not (
                all_attack_routes_blocked(
                    apply_interventions(
                        instance.observed_graph,
                        canonical[:index] + canonical[index + 1 :],
                    ),
                    instance.attack_sources,
                    instance.protected_assets,
                )
                and service_preservation(
                    apply_interventions(
                        instance.observed_graph,
                        canonical[:index] + canonical[index + 1 :],
                    ),
                    instance.required_service_pairs_observed,
                )
                >= 1.0 - 1e-12
            )
            for index in range(len(canonical))
        )
        if not subset_minimal:
            rejection_codes.append("not_subset_minimal")
    true_after = apply_interventions(instance.true_graph, canonical)
    hidden_feasible = all_attack_routes_blocked(
        true_after, instance.attack_sources, instance.protected_assets
    ) and service_preservation(
        true_after, instance.required_service_pairs_true
    ) >= 1.0 - 1e-12
    cost = sum(
        float(instance.observed_graph[source][destination].get("action_cost", 1.0))
        for source, destination in canonical
        if instance.observed_graph.has_edge(source, destination)
    )
    return {
        "actions": canonical,
        "json": json.dumps([list(edge) for edge in canonical], separators=(",", ":")),
        "sha256": stable_hash([list(edge) for edge in canonical]),
        "action_count": len(canonical),
        "action_cost": cost,
        "observed_admission": float(not rejection_codes),
        "rejection_codes": "accepted" if not rejection_codes else ";".join(rejection_codes),
        "hidden_feasible": float(hidden_feasible),
        "attack_blocked_observed": float(attack_blocked),
        "service_safe_observed": float(service_safe),
        "subset_minimal_observed": float(subset_minimal),
    }


def evaluate_capsule(
    instance: NetworkInstance,
    capsule: DefenseCapsule,
    runtime_s: float,
    cutoff: int = 8,
) -> Dict[str, object]:
    config = instance.config
    interventions = list(canonical_action_set(capsule.interventions, field="capsule.interventions"))
    candidate_actions = list(
        canonical_action_set(capsule.candidate_actions, field="capsule.candidate_actions")
    )
    raw_outcomes = _proposal_outcomes(
        instance,
        capsule.raw_proposal,
        candidate_actions=candidate_actions,
        action_budget=capsule.action_budget,
    )
    repaired_outcomes = _proposal_outcomes(
        instance,
        capsule.repaired_proposal,
        candidate_actions=candidate_actions,
        action_budget=capsule.action_budget,
    )

    base_observed_paths = enumerate_attack_paths(
        instance.observed_graph,
        instance.attack_sources,
        instance.protected_assets,
        cutoff=cutoff,
    )
    base_true_paths = enumerate_attack_paths(
        instance.true_graph,
        instance.attack_sources,
        instance.protected_assets,
        cutoff=cutoff,
    )
    observed_after = apply_interventions(instance.observed_graph, interventions)
    true_after = apply_interventions(instance.true_graph, interventions)

    base_observed_pairs = reachable_pairs(instance.observed_graph, instance.attack_sources, instance.protected_assets)
    base_true_pairs = reachable_pairs(instance.true_graph, instance.attack_sources, instance.protected_assets)
    residual_observed_pairs = reachable_pairs(observed_after, instance.attack_sources, instance.protected_assets)
    residual_true_pairs = reachable_pairs(true_after, instance.attack_sources, instance.protected_assets)

    observed_path_reduction = _family_reduction(base_observed_paths, interventions)
    true_path_reduction = _family_reduction(base_true_paths, interventions)
    observed_pair_isolation = _safe_div(
        len(base_observed_pairs) - len(residual_observed_pairs),
        len(base_observed_pairs),
        1.0,
    )
    true_pair_isolation = _safe_div(
        len(base_true_pairs) - len(residual_true_pairs),
        len(base_true_pairs),
        1.0,
    )

    base_observed_risk = risk_mass(instance.observed_graph, base_observed_paths)
    base_true_risk = risk_mass(instance.true_graph, base_true_paths)
    residual_observed_risk = risk_mass(observed_after, base_observed_paths)
    residual_true_risk = risk_mass(true_after, base_true_paths)

    observed_preservation = service_preservation(
        observed_after,
        instance.required_service_pairs_observed,
    )
    true_preservation = service_preservation(
        true_after,
        instance.required_service_pairs_true,
    )
    observed_conflicts = len(service_disrupted_pairs(observed_after, instance.required_service_pairs_observed))
    true_conflicts = len(service_disrupted_pairs(true_after, instance.required_service_pairs_true))
    certificate = capsule.service_safety_certificate or {}
    certificate_pass = bool(certificate.get("passed", False))
    true_attack_blocked = all_attack_routes_blocked(
        true_after,
        instance.attack_sources,
        instance.protected_assets,
    )
    threat_present_observed = bool(base_observed_pairs)
    threat_present_true = bool(base_true_pairs)
    true_feasible = (
        float(true_attack_blocked and true_preservation >= 1.0 - 1e-12)
        if threat_present_true
        else float("nan")
    )
    action_cost = sum(
        float(instance.observed_graph[source][destination].get("action_cost", 1.0))
        for source, destination in interventions
        if instance.observed_graph.has_edge(source, destination)
    )
    source_set = set(instance.attack_sources)
    action_origin_counts = {"attack_source": 0, "gateway": 0, "broker": 0, "other": 0}
    for source, _destination in interventions:
        if source in source_set:
            action_origin_counts["attack_source"] += 1
        else:
            origin_type = instance.observed_graph.nodes[source].get("type")
            action_origin_counts[origin_type if origin_type in {"gateway", "broker"} else "other"] += 1

    # Negative control for the benchmark's former shortcut: deny every
    # observed edge leaving an attack source, then report threat blocking,
    # endpoint-service preservation, budget fit, and their conjunction as
    # separate quantities.
    source_boundary = sorted(
        (source, destination)
        for source, destination in instance.observed_graph.edges
        if source in source_set
    )
    source_boundary_after = apply_interventions(instance.observed_graph, source_boundary)
    source_boundary_attack_blocked = all_attack_routes_blocked(
        source_boundary_after,
        instance.attack_sources,
        instance.protected_assets,
    )
    source_boundary_service = service_preservation(
        source_boundary_after,
        instance.required_service_pairs_observed,
    )
    source_boundary_within_budget = len(source_boundary) <= capsule.action_budget
    source_boundary_threat_service_feasible = (
        source_boundary_attack_blocked and source_boundary_service >= 1.0 - 1e-12
    )
    source_boundary_admissible_under_budget = (
        source_boundary_threat_service_feasible and source_boundary_within_budget
    )
    source_departments = {
        instance.observed_graph.nodes[source].get("dept")
        for source in source_set
        if source in instance.observed_graph
    }
    attack_brokers = {
        node
        for node, attributes in instance.observed_graph.nodes(data=True)
        if attributes.get("type") == "broker" and attributes.get("dept") in source_departments
    }
    broker_boundary = sorted(
        (source, destination)
        for source, destination in instance.observed_graph.edges
        if source in attack_brokers
        and instance.observed_graph.nodes[destination].get("type") != "monitor"
    )
    broker_boundary_after = apply_interventions(instance.observed_graph, broker_boundary)
    broker_boundary_attack_blocked = all_attack_routes_blocked(
        broker_boundary_after,
        instance.attack_sources,
        instance.protected_assets,
    )
    broker_boundary_service = service_preservation(
        broker_boundary_after,
        instance.required_service_pairs_observed,
    )
    broker_boundary_within_budget = len(broker_boundary) <= capsule.action_budget
    broker_boundary_threat_service_feasible = (
        broker_boundary_attack_blocked and broker_boundary_service >= 1.0 - 1e-12
    )
    broker_boundary_admissible_under_budget = (
        broker_boundary_threat_service_feasible and broker_boundary_within_budget
    )

    # Artifact-centric measurements replay the serialized bundle through the
    # same verifier, independently of the proposer. This timing is kept
    # separate from proposal time.
    capsule_payload = capsule.to_dict()
    certificate_scope = dict(certificate.get("path_search_scope", {}))
    path_scope = PathSearchScope(**certificate_scope) if certificate_scope else PathSearchScope(cutoff_edges=cutoff)
    bundle = make_verification_bundle(
        instance.observed_graph,
        capsule_payload,
        instance.attack_sources,
        instance.protected_assets,
        instance.required_service_pairs_observed,
        tau=1.0,
        action_budget=capsule.action_budget,
        path_scope=path_scope,
        freshness=capsule.freshness,
        candidate_actions=candidate_actions,
    )
    capsule_bytes = json.dumps(capsule_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    witness_bytes = json.dumps(
        certificate.get("witnesses", {}), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    bundle_bytes = json.dumps(bundle, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    replay_start = time.perf_counter()
    replay = verify_capsule_bundle(
        json.loads(bundle_bytes),
        expected_bundle_sha256=verification_bundle_hash(bundle),
        verification_time=capsule.freshness["issued_at"],
    )
    serialized_replay_s = time.perf_counter() - replay_start

    return {
        "method": capsule.method,
        "seed": config.seed,
        "size": config.size,
        "noise": config.noise,
        "hidden": config.hidden,
        "density": config.density,
        "topology_regime": config.topology_regime,
        "action_budget": capsule.action_budget,
        "candidate_availability": _safe_div(
            len(candidate_actions), instance.observed_graph.number_of_edges(), 0.0
        ),
        "candidate_set_sha256": stable_hash([list(edge) for edge in candidate_actions]),
        "candidate_action_count": len(candidate_actions),
        "raw_proposal_json": raw_outcomes["json"],
        "raw_proposal_sha256": raw_outcomes["sha256"],
        "raw_proposal_action_count": raw_outcomes["action_count"],
        "raw_proposal_action_cost": raw_outcomes["action_cost"],
        "raw_proposal_observed_admission": raw_outcomes["observed_admission"],
        "raw_proposal_rejection_codes": raw_outcomes["rejection_codes"],
        "raw_proposal_hidden_feasible": raw_outcomes["hidden_feasible"],
        "raw_proposal_attack_blocked_observed": raw_outcomes["attack_blocked_observed"],
        "raw_proposal_service_safe_observed": raw_outcomes["service_safe_observed"],
        "raw_proposal_subset_minimal_observed": raw_outcomes["subset_minimal_observed"],
        "repaired_proposal_json": repaired_outcomes["json"],
        "repaired_proposal_sha256": repaired_outcomes["sha256"],
        "repaired_proposal_action_count": repaired_outcomes["action_count"],
        "repaired_proposal_action_cost": repaired_outcomes["action_cost"],
        "repaired_proposal_observed_admission": repaired_outcomes["observed_admission"],
        "repaired_proposal_rejection_codes": repaired_outcomes["rejection_codes"],
        "repaired_proposal_hidden_feasible": repaired_outcomes["hidden_feasible"],
        "repaired_proposal_attack_blocked_observed": repaired_outcomes["attack_blocked_observed"],
        "repaired_proposal_service_safe_observed": repaired_outcomes["service_safe_observed"],
        "repaired_proposal_subset_minimal_observed": repaired_outcomes["subset_minimal_observed"],
        "repair_status": capsule.proposal_transformation.get("status", "unknown"),
        "repair_version": capsule.proposal_transformation.get("transformation_version", "unknown"),
        "repair_log_json": canonical_json_bytes(
            capsule.proposal_transformation.get("steps", [])
        ).decode("ascii"),
        "repair_log_sha256": capsule.proposal_transformation.get("repair_log_sha256", ""),
        "repair_removed_action_count": int(raw_outcomes["action_count"]) - int(repaired_outcomes["action_count"]),
        "nodes": instance.true_graph.number_of_nodes(),
        "edges_true": instance.true_graph.number_of_edges(),
        "edges_observed": instance.observed_graph.number_of_edges(),
        "eligible_hidden_edges": instance.metadata.get("eligible_hidden_edge_count", 0),
        "hidden_edge_count": instance.metadata.get("hidden_edge_count", len(instance.hidden_edges)),
        "realized_hidden_fraction_eligible": instance.metadata.get(
            "realized_hidden_fraction_eligible", 0.0
        ),
        "missing_true_edges": instance.metadata.get("missing_true_edge_count", 0),
        "spurious_observed_edges": instance.metadata.get(
            "spurious_observed_edge_count", len(instance.spurious_edges)
        ),
        "edge_observation_recall": instance.metadata.get("edge_observation_recall", 1.0),
        "edge_observation_precision": instance.metadata.get("edge_observation_precision", 1.0),
        "required_edge_missing_count": instance.metadata.get("required_edge_missing_count", 0),
        "attack_edge_observation_recall": instance.metadata.get(
            "attack_edge_observation_recall", 1.0
        ),
        "service_specification_coverage": instance.metadata.get(
            "service_specification_coverage", 1.0
        ),
        "required_service_pairs_true": len(instance.required_service_pairs_true),
        "required_service_pairs_observed": len(instance.required_service_pairs_observed),
        "base_attack_paths_observed": len(base_observed_paths),
        "base_attack_paths_true": len(base_true_paths),
        "base_attack_pairs_observed": len(base_observed_pairs),
        "base_attack_pairs_true": len(base_true_pairs),
        "threat_present_observed": float(threat_present_observed),
        "threat_present_true": float(threat_present_true),
        "residual_attack_pairs_observed": len(residual_observed_pairs),
        "residual_attack_pairs_true": len(residual_true_pairs),
        "attack_path_reduction_observed": observed_path_reduction,
        "attack_path_reduction_true": true_path_reduction,
        "attack_pair_isolation_observed": observed_pair_isolation if threat_present_observed else float("nan"),
        "attack_pair_isolation_true": true_pair_isolation if threat_present_true else float("nan"),
        "compromise_risk_reduction_observed": _safe_div(
            base_observed_risk - residual_observed_risk,
            base_observed_risk,
            float("nan"),
        ),
        "compromise_risk_reduction_true": _safe_div(
            base_true_risk - residual_true_risk,
            base_true_risk,
            float("nan"),
        ),
        "residual_risk_observed": residual_observed_risk,
        "residual_risk_true": residual_true_risk,
        "service_pair_preservation_observed": observed_preservation,
        "service_pair_preservation_true": true_preservation,
        # Compatibility aliases are numerically identical and explicitly
        # endpoint-based in the canonical README/schema.
        "service_path_preservation_observed": observed_preservation,
        "service_path_preservation_true": true_preservation,
        "false_block_rate_observed": 1.0 - observed_preservation,
        "false_block_rate_true": 1.0 - true_preservation,
        "service_disruption_score_observed": 1.0 - observed_preservation,
        "service_disruption_score_true": 1.0 - true_preservation,
        "policy_conflicts_observed": observed_conflicts,
        "policy_conflicts_true": true_conflicts,
        "runtime_s": runtime_s,
        "policy_size": len(interventions),
        "num_interventions": len(interventions),
        "action_cost": action_cost,
        "budget_utilization": _safe_div(len(interventions), capsule.action_budget, 0.0),
        "actions_from_attack_source": action_origin_counts["attack_source"],
        "actions_from_gateway": action_origin_counts["gateway"],
        "actions_from_broker": action_origin_counts["broker"],
        "actions_from_other": action_origin_counts["other"],
        "action_fraction_from_attack_source": _safe_div(action_origin_counts["attack_source"], len(interventions), 0.0),
        "action_fraction_from_gateway": _safe_div(action_origin_counts["gateway"], len(interventions), 0.0),
        "action_fraction_from_broker": _safe_div(action_origin_counts["broker"], len(interventions), 0.0),
        "source_boundary_action_count": len(source_boundary),
        "source_boundary_within_budget": float(source_boundary_within_budget),
        "source_boundary_attack_blocked_observed": float(source_boundary_attack_blocked),
        "source_boundary_service_preservation_observed": source_boundary_service,
        "source_boundary_threat_service_feasible_observed": float(source_boundary_threat_service_feasible),
        "source_boundary_admissible_under_budget_observed": float(source_boundary_admissible_under_budget),
        "broker_boundary_action_count": len(broker_boundary),
        "broker_boundary_within_budget": float(broker_boundary_within_budget),
        "broker_boundary_attack_blocked_observed": float(broker_boundary_attack_blocked),
        "broker_boundary_service_preservation_observed": broker_boundary_service,
        "broker_boundary_threat_service_feasible_observed": float(broker_boundary_threat_service_feasible),
        "broker_boundary_admissible_under_budget_observed": float(broker_boundary_admissible_under_budget),
        "capsule_json_bytes": len(capsule_bytes),
        "essential_witness_json_bytes": len(witness_bytes),
        "verification_bundle_json_bytes": len(bundle_bytes),
        "serialized_replay_verifier_s": serialized_replay_s,
        "serialized_replay_pass": float(replay.passed),
        "certificate_pass": float(certificate_pass),
        "verifier_rejection": float(not certificate_pass),
        "true_attack_reachability_blocked": float(true_attack_blocked),
        "true_feasible": true_feasible,
        "accepted_true_feasible": float(certificate_pass and bool(true_feasible)) if threat_present_true else float("nan"),
        "subset_minimal": float(bool(certificate.get("subset_minimal", False))),
        "cert_minimality_status": certificate.get("minimality_status", "none"),
    }


def summarize(dataframe):
    excluded = {
        "method",
        "seed",
        "size",
        "noise",
        "hidden",
        "density",
        "topology_regime",
        "cert_minimality_status",
    }
    metric_columns = [
        column
        for column in dataframe.select_dtypes(include=["number", "bool"]).columns
        if column not in excluded
    ]
    grouped = dataframe.groupby(["method", "size"], dropna=False)[metric_columns].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(str(value) for value in column if value) for column in grouped.columns]
    return grouped

#!/usr/bin/env python3
"""Run the controlled explicit candidate-action capability study."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from cin.baselines import service_aware_greedy_cut
from cin.candidates import EXCLUSION_REASONS, derive_candidate_action_universe
from cin.capsule import CertifiedInterventionPlanner
from cin.exact_optimizer import solve_full_universe_milp
from cin.graph_generator import BenchmarkConfig, generate_network
from cin.metrics import evaluate_capsule
from cin.risk_model import ObservedRiskScorer
from cin.simulator import (
    all_attack_routes_blocked,
    apply_interventions,
    service_preservation,
)

AVAILABILITY_LEVELS = (1.0, 0.75, 0.5)
ACTION_BUDGETS = {"small": 6, "medium": 12}
FEASIBLE_WITNESS = "feasible_witness"
CONCLUSIVELY_OPTIMAL_INFEASIBLE = "conclusively_optimal_infeasible"
UNKNOWN_FEASIBILITY = "unknown_solver_nonoptimal"
UNKNOWN_MODEL_INCONSISTENT = "unknown_solver_semantic_inconsistency"


def classify_exact_feasibility(diagnostics: object) -> Dict[str, object]:
    """Classify existence feasibility without treating uncertainty as failure.

    A selected action set that independently satisfies the complete observed
    specification is a constructive witness even when the solver timed out.
    Conversely, absence of such a witness proves infeasibility only when the
    primary model itself was proven infeasible, or when its optimal isolation
    count agrees with independent action-domain, service, and reachability
    recomputation and leaves a positive residual. Optional lexicographic
    tie-break completion has no bearing on that existence conclusion.
    """

    status = str(getattr(diagnostics, "status"))
    witness_feasible = bool(
        getattr(diagnostics, "independently_recomputed_feasible")
    )
    declared_model_consistent = bool(
        getattr(diagnostics, "independent_model_consistent", False)
    )
    independent_residual_count = getattr(
        diagnostics, "independently_recomputed_residual_pair_count", None
    )
    primary_model_isolated_pair_count = getattr(
        diagnostics, "primary_model_isolated_pair_count", None
    )
    primary_model_attack_pair_count = getattr(
        diagnostics, "primary_model_attack_pair_count", None
    )
    model_consistent = (
        declared_model_consistent
        and type(primary_model_attack_pair_count) is int
        and type(primary_model_isolated_pair_count) is int
        and type(independent_residual_count) is int
        and primary_model_isolated_pair_count + independent_residual_count
        == primary_model_attack_pair_count
    )
    consistent_positive_residual = (
        model_consistent
        and type(independent_residual_count) is int
        and independent_residual_count > 0
    )
    primary_optimal = status == "optimal" or status.startswith("tie_break_")
    primary_infeasible = status == "infeasible"
    if status.startswith("tie_break_"):
        tie_break_status = status.removeprefix("tie_break_")
        tie_break_complete = False
        primary_status = "optimal"
    elif primary_optimal:
        tie_break_status = "complete"
        tie_break_complete = bool(getattr(diagnostics, "optimal", False))
        primary_status = "optimal"
    else:
        tie_break_status = "not_started_primary_nonoptimal"
        tie_break_complete = False
        primary_status = status

    if witness_feasible:
        feasibility_status = FEASIBLE_WITNESS
        feasible: bool | None = True
        unknown_reason = ""
    elif primary_infeasible:
        feasibility_status = CONCLUSIVELY_OPTIMAL_INFEASIBLE
        feasible = False
        unknown_reason = ""
    elif primary_optimal and consistent_positive_residual:
        feasibility_status = CONCLUSIVELY_OPTIMAL_INFEASIBLE
        feasible = False
        unknown_reason = ""
    elif primary_optimal:
        feasibility_status = UNKNOWN_MODEL_INCONSISTENT
        feasible = None
        unknown_reason = "primary_solution_semantic_inconsistency"
    else:
        feasibility_status = UNKNOWN_FEASIBILITY
        feasible = None
        unknown_reason = "primary_solver_nonoptimal"
    return {
        "oracle_assessed": feasible is not None,
        "observed_feasibility_status": feasibility_status,
        "observed_instance_feasible": feasible,
        "exact_primary_solver_status": primary_status,
        "exact_primary_optimality_proven": primary_optimal,
        "exact_primary_infeasibility_proven": primary_infeasible,
        "exact_tie_break_status": tie_break_status,
        "exact_tie_break_complete": tie_break_complete,
        "exact_primary_model_isolated_pair_count": primary_model_isolated_pair_count,
        "exact_primary_model_attack_pair_count": primary_model_attack_pair_count,
        "exact_independent_residual_pair_count": independent_residual_count,
        "exact_independent_action_domain_valid": bool(
            getattr(diagnostics, "independent_action_domain_valid", False)
        ),
        "exact_independent_service_valid": bool(
            getattr(diagnostics, "independent_service_valid", False)
        ),
        "exact_independent_model_consistent": model_consistent,
        "exact_feasibility_consistency_error": (
            feasibility_status == UNKNOWN_MODEL_INCONSISTENT
        ),
        "observed_feasibility_unknown_reason": unknown_reason,
    }


def _run_one(task: Tuple[int, str, float]) -> List[Dict[str, object]]:
    seed, size, availability = task
    action_budget = ACTION_BUDGETS[size]
    density = ("low", "medium", "high")[seed % 3]
    instance = generate_network(
        BenchmarkConfig(
            size=size,
            noise=0.1,
            hidden=0.1,
            density=density,
            seed=60000 + seed,
            topology_regime="shared_broker",
        )
    )
    universe = derive_candidate_action_universe(
        instance.observed_graph, availability, seed=70000 + seed
    )
    candidates = list(universe.candidate_actions)
    model = ObservedRiskScorer(cutoff=8)
    model.fit(
        instance.observed_graph, instance.attack_sources, instance.protected_assets
    )
    planner = CertifiedInterventionPlanner(tau=1.0, cutoff=8)

    exact_started = time.perf_counter()
    exact = solve_full_universe_milp(
        instance.observed_graph,
        instance.attack_sources,
        instance.protected_assets,
        instance.required_service_pairs_observed,
        action_budget=action_budget,
        candidate_actions=candidates,
        time_limit_s=120.0,
        mip_rel_gap=0.0,
        method="ExactDeclaredCandidateMILP",
    )
    exact_runtime = time.perf_counter() - exact_started
    feasibility = classify_exact_feasibility(exact.diagnostics)
    cin_started = time.perf_counter()
    cin_capsule = planner.synthesize(
        instance.observed_graph,
        instance.attack_sources,
        instance.protected_assets,
        instance.required_service_pairs_observed,
        model,
        action_budget=action_budget,
        candidate_actions=candidates,
    )
    cin_runtime = time.perf_counter() - cin_started
    service_started = time.perf_counter()
    service_capsule = service_aware_greedy_cut(
        instance.observed_graph,
        instance.attack_sources,
        instance.protected_assets,
        instance.required_service_pairs_observed,
        action_budget,
        cutoff=8,
        candidate_actions=candidates,
    )
    service_runtime = time.perf_counter() - service_started

    instance_feasible = feasibility["observed_instance_feasible"]
    source_departments = {
        instance.observed_graph.nodes[source].get("dept")
        for source in instance.attack_sources
    }
    brokers = {
        node
        for node, attributes in instance.observed_graph.nodes(data=True)
        if attributes.get("type") == "broker"
        and attributes.get("dept") in source_departments
    }
    broker_cut = sorted(
        (source, destination)
        for source, destination in instance.observed_graph.edges
        if source in brokers
        and instance.observed_graph.nodes[destination].get("type") != "monitor"
    )
    broker_cut_available = all(edge in set(candidates) for edge in broker_cut)
    broker_after = apply_interventions(instance.observed_graph, broker_cut)
    broker_threat_service_feasible = all_attack_routes_blocked(
        broker_after, instance.attack_sources, instance.protected_assets
    ) and service_preservation(
        broker_after, instance.required_service_pairs_observed
    ) >= 1.0 - 1e-12

    reason_counts = {
        reason: sum(
            1 for value in universe.exclusion_reasons.values() if value == reason
        )
        for reason in EXCLUSION_REASONS
    }
    common: Dict[str, object] = {
        "study_seed": seed,
        "graph_seed": 60000 + seed,
        "size": size,
        "density": density,
        "noise": 0.1,
        "hidden": 0.1,
        "action_budget": action_budget,
        "service_threshold": 1.0,
        "candidate_availability_requested": availability,
        "candidate_availability_realized": len(candidates)
        / max(1, instance.observed_graph.number_of_edges()),
        "candidate_action_count": len(candidates),
        "candidate_set_sha256": universe.candidate_actions_sha256,
        "candidate_actions_json": universe.candidate_actions_json,
        "excluded_action_count": len(universe.excluded_actions),
        "excluded_actions_sha256": universe.excluded_actions_sha256,
        "excluded_actions_json": universe.excluded_actions_json,
        "candidate_exclusion_reasons_json": universe.exclusion_reasons_json,
        "candidate_exclusion_reasons_sha256": universe.exclusion_reasons_sha256,
        "candidate_selection_policy": universe.selection_policy,
        "candidate_universe_schema": universe.schema,
        **feasibility,
        "observed_instance_feasible": (
            "" if instance_feasible is None else float(instance_feasible)
        ),
        "oracle_attempted": True,
        "broker_cut_action_count": len(broker_cut),
        "broker_cut_within_budget": float(len(broker_cut) <= action_budget),
        "broker_cut_fully_available": float(broker_cut_available),
        "broker_cut_threat_service_feasible": float(
            broker_threat_service_feasible
        ),
        "exact_solver_status": exact.diagnostics.status,
        "exact_solver_primal_objective": exact.diagnostics.primal_objective,
        "exact_solver_dual_bound": exact.diagnostics.dual_bound,
        "exact_solver_mip_gap": exact.diagnostics.mip_gap,
        "exact_solver_runtime_s": exact.diagnostics.solver_runtime_s,
        "exact_solver_time_limit_s": 120.0,
        "exact_independently_recomputed_feasible": exact.diagnostics.independently_recomputed_feasible,
        **{f"excluded_{reason}_count": count for reason, count in reason_counts.items()},
    }
    rows: List[Dict[str, object]] = []
    for capsule, runtime in (
        (exact.capsule, exact_runtime),
        (cin_capsule, cin_runtime),
        (service_capsule, service_runtime),
    ):
        row = evaluate_capsule(instance, capsule, runtime)
        row.update(common)
        row["proposer_success_conditional_on_feasible"] = (
            row["repaired_proposal_observed_admission"]
            if instance_feasible is True
            else float("nan")
        )
        row["raw_proposal_success_conditional_on_feasible"] = (
            row["raw_proposal_observed_admission"]
            if instance_feasible is True
            else float("nan")
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_trust_boundary_v2/candidate_capability_results.csv"),
    )
    parser.add_argument("--jobs", type=int, default=0)
    args = parser.parse_args()
    if args.seeds < 10:
        raise ValueError("candidate-capability study requires at least ten seed clusters")
    tasks = [
        (seed, size, availability)
        for seed in range(args.seeds)
        for size in ACTION_BUDGETS
        for availability in AVAILABILITY_LEVELS
    ]
    workers = max(1, min(8, args.jobs or 8))
    rows: List[Dict[str, object]] = []
    if workers == 1:
        for task in tasks:
            rows.extend(_run_one(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for part in executor.map(_run_one, tasks, chunksize=1):
                rows.extend(part)
    dataframe = pd.DataFrame(rows).sort_values(
        ["study_seed", "candidate_availability_requested", "method"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output, index=False)
    print(f"wrote {len(dataframe)} rows to {args.output}")


if __name__ == "__main__":
    main()

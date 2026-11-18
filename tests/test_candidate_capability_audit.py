"""Audit regressions for candidate capability and exact-feasibility tri-state."""
from __future__ import annotations

import json
from types import SimpleNamespace

import networkx as nx

from cin.candidates import EXCLUSION_REASONS, derive_candidate_action_universe
from cin.verifier import stable_hash
from scripts.run_candidate_capability_study import (
    CONCLUSIVELY_OPTIMAL_INFEASIBLE,
    FEASIBLE_WITNESS,
    UNKNOWN_FEASIBILITY,
    UNKNOWN_MODEL_INCONSISTENT,
    classify_exact_feasibility,
)


def _capability_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.graph["snapshot_id"] = "candidate-audit-snapshot"
    graph.add_node("source", type="router", dept="a")
    destinations = {
        "asset": {"type": "asset", "dept": "a"},
        "monitor": {"type": "monitor", "dept": "a"},
        "controller_gap": {"type": "router", "dept": "a"},
        "foreign": {"type": "router", "dept": "b"},
        "costly": {"type": "router", "dept": "a"},
        **{
            f"ordinary_{index}": {"type": "router", "dept": "a"}
            for index in range(5)
        },
    }
    for node, attributes in destinations.items():
        graph.add_node(node, **attributes)
        kind = "shadow" if node == "controller_gap" else "comm"
        cost = 2.0 if node == "costly" else 1.0
        graph.add_edge("source", node, kind=kind, action_cost=cost)
    return graph


def test_candidate_subsets_are_nested_exact_and_causally_stratified() -> None:
    graph = _capability_graph()
    full = derive_candidate_action_universe(graph, 1.0, seed=11)
    three_quarters = derive_candidate_action_universe(graph, 0.75, seed=11)
    half = derive_candidate_action_universe(graph, 0.5, seed=11)

    assert len(full.candidate_actions) == 10
    assert len(three_quarters.candidate_actions) == round(10 * 0.75)
    assert len(half.candidate_actions) == round(10 * 0.5)
    assert set(half.candidate_actions) <= set(three_quarters.candidate_actions)
    assert set(three_quarters.candidate_actions) <= set(full.candidate_actions)
    assert set(three_quarters.excluded_actions) <= set(half.excluded_actions)
    assert set(half.exclusion_reasons.values()) == set(EXCLUSION_REASONS)

    for edge, reason in half.exclusion_reasons.items():
        source, destination = edge
        data = graph[source][destination]
        if reason == "protected_edge":
            assert graph.nodes[destination]["type"] in {
                "asset",
                "monitor",
                "protected_asset",
            }
        elif reason == "controller_inexpressible":
            assert data["kind"] in {"shadow", "iot_bridge", "privilege"}
        elif reason == "unauthorized":
            assert graph.nodes[source]["dept"] != graph.nodes[destination]["dept"]
        elif reason == "high_cost":
            assert data["action_cost"] > 1.5
        else:
            assert reason == "unavailable"

    assert half.schema == "cin.candidate-action-universe.v2"
    assert half.selection_policy == "causal-stratified-hash-round-robin-v2"
    assert half.candidate_actions_sha256 == stable_hash(
        json.loads(half.candidate_actions_json)
    )
    assert half.excluded_actions_sha256 == stable_hash(
        json.loads(half.excluded_actions_json)
    )
    assert half.exclusion_reasons_sha256 == stable_hash(
        json.loads(half.exclusion_reasons_json)
    )
    assert [record["action"] for record in json.loads(half.exclusion_reasons_json)] == json.loads(
        half.excluded_actions_json
    )


def test_exact_feasibility_classifier_preserves_solver_unknowns() -> None:
    unknown = classify_exact_feasibility(
        SimpleNamespace(
            status="limit_reached",
            independently_recomputed_feasible=False,
            optimal=False,
        )
    )
    assert unknown["observed_feasibility_status"] == UNKNOWN_FEASIBILITY
    assert unknown["observed_instance_feasible"] is None
    assert unknown["oracle_assessed"] is False

    timeout_witness = classify_exact_feasibility(
        SimpleNamespace(
            status="limit_reached",
            independently_recomputed_feasible=True,
            optimal=False,
        )
    )
    assert timeout_witness["observed_feasibility_status"] == FEASIBLE_WITNESS
    assert timeout_witness["observed_instance_feasible"] is True
    assert timeout_witness["oracle_assessed"] is True

    incomplete_tie_break = classify_exact_feasibility(
        SimpleNamespace(
            status="tie_break_limit_reached",
            independently_recomputed_feasible=False,
            optimal=False,
            independent_model_consistent=True,
            independently_recomputed_residual_pair_count=1,
            primary_model_isolated_pair_count=2,
            primary_model_attack_pair_count=3,
        )
    )
    assert (
        incomplete_tie_break["observed_feasibility_status"]
        == CONCLUSIVELY_OPTIMAL_INFEASIBLE
    )
    assert incomplete_tie_break["exact_primary_optimality_proven"] is True
    assert incomplete_tie_break["exact_tie_break_complete"] is False
    assert incomplete_tie_break["exact_tie_break_status"] == "limit_reached"

    complete_primary = classify_exact_feasibility(
        SimpleNamespace(
            status="optimal",
            independently_recomputed_feasible=False,
            optimal=True,
            independent_model_consistent=True,
            independently_recomputed_residual_pair_count=1,
            primary_model_isolated_pair_count=2,
            primary_model_attack_pair_count=3,
        )
    )
    assert complete_primary["observed_instance_feasible"] is False
    assert complete_primary["exact_primary_optimality_proven"] is True
    assert complete_primary["exact_tie_break_complete"] is True

    inconsistent_primary = classify_exact_feasibility(
        SimpleNamespace(
            status="optimal",
            independently_recomputed_feasible=False,
            optimal=True,
            independent_model_consistent=False,
            independently_recomputed_residual_pair_count=0,
            primary_model_isolated_pair_count=1,
            primary_model_attack_pair_count=1,
        )
    )
    assert (
        inconsistent_primary["observed_feasibility_status"]
        == UNKNOWN_MODEL_INCONSISTENT
    )
    assert inconsistent_primary["observed_instance_feasible"] is None
    assert inconsistent_primary["oracle_assessed"] is False
    assert inconsistent_primary["exact_feasibility_consistency_error"] is True
    assert inconsistent_primary["observed_feasibility_unknown_reason"] == (
        "primary_solution_semantic_inconsistency"
    )

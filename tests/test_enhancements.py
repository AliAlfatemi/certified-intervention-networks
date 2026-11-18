from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from cin.capsule import (
    EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
    EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
    build_capsule,
    experiment_issuance_time,
)
from cin.controller import (
    compile_openflow13_policy,
    exercise_bad_policy_rollback,
    staged_install_with_monitoring,
    verify_compiler_equivalence,
)
from cin.exact_optimizer import solve_full_universe_milp
from cin.learned_proposer import (
    LearnedServiceAwarePlanner,
    evaluate_edge_model,
    fit_calibrated_edge_model,
)
from cin.provenance import (
    AuthorizationPolicy,
    authorize_interventions,
    deterministic_experiment_keypair,
    sign_snapshot_attestation,
    verify_snapshot_attestation,
)
from cin.repair import identity_transformation
from cin.public_topologies import generate_abilene_instance, load_abilene_underlay
from cin.simulator import PathSearchScope, apply_interventions, reachable_pairs, service_preservation
from cin.uncertainty import verify_declared_completion_set
from cin.verifier import _jsonable, make_verification_bundle


def _edge_attrs(required: bool = False) -> dict[str, object]:
    return {
        "kind": "comm",
        "risk_prob": 0.5,
        "anomaly_score": 0.5,
        "latency_ms": 1.0,
        "action_cost": 1.0,
        "uncertainty": 0.0,
        "required": required,
        "observed": True,
    }


def _manual_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    for node, node_type, tier in [
        ("s", "endpoint", 0),
        ("x", "router", 1),
        ("y", "router", 1),
        ("t", "asset", 3),
        ("c", "client", 0),
        ("q", "service", 2),
    ]:
        graph.add_node(node, type=node_type, dept="d", tier=tier, criticality=float(node == "t"))
    graph.add_edge("s", "x", **_edge_attrs())
    graph.add_edge("x", "t", **_edge_attrs())
    graph.add_edge("s", "y", **_edge_attrs())
    graph.add_edge("y", "t", **_edge_attrs())
    graph.add_edge("c", "q", **_edge_attrs(required=True))
    graph.graph.update(
        snapshot_id="manual-snapshot",
        topology_version=1,
        snapshot_time="2026-07-27T00:00:00Z",
    )
    return graph


def test_full_universe_milp_proves_known_optimum() -> None:
    graph = _manual_graph()
    result = solve_full_universe_milp(
        graph,
        ["s"],
        ["t"],
        [["c", "q"]],
        action_budget=2,
        candidate_actions=sorted(graph.edges),
        time_limit_s=30.0,
    )
    assert result.diagnostics.optimal
    assert result.diagnostics.residual_pair_count == 0
    assert result.diagnostics.selected_action_count == 2
    after = apply_interventions(graph, result.capsule.interventions)
    assert not reachable_pairs(after, ["s"], ["t"])
    assert service_preservation(after, [["c", "q"]]) == 1.0
    assert result.capsule.service_safety_certificate["passed"] is True
    assert result.capsule.interventions == [("s", "x"), ("s", "y")]
    assert result.diagnostics.tie_break_solve_count == 1
    assert result.diagnostics.primary_model_isolated_pair_count == 1
    assert result.diagnostics.independently_recomputed_residual_pair_count == 0
    assert result.diagnostics.independent_action_domain_valid is True
    assert result.diagnostics.independent_service_valid is True
    assert result.diagnostics.independent_model_consistent is True

    # Graph insertion order and solver-internal tied-optimum choices cannot
    # change the declared canonical intervention tuple.
    reordered = nx.DiGraph()
    for node, attrs in reversed(list(graph.nodes(data=True))):
        reordered.add_node(node, **dict(attrs))
    for source, destination, attrs in reversed(list(graph.edges(data=True))):
        reordered.add_edge(source, destination, **dict(attrs))
    reordered.graph.update(graph.graph)
    repeated = solve_full_universe_milp(
        reordered,
        ["s"],
        ["t"],
        [["c", "q"]],
        action_budget=2,
        candidate_actions=sorted(reordered.edges),
        time_limit_s=30.0,
    )
    assert repeated.diagnostics.optimal
    assert repeated.capsule.interventions == result.capsule.interventions


@pytest.mark.parametrize(
    ("attribute", "value", "match"),
    [
        ("action_cost", -1.0, "finite non-negative"),
        ("action_cost", float("inf"), "finite non-negative"),
        ("blocked", True, "no pre-blocked edges"),
    ],
)
def test_full_universe_milp_rejects_semantic_domain_mismatches(
    attribute: str, value: object, match: str
) -> None:
    graph = _manual_graph()
    graph["s"]["x"][attribute] = value
    with pytest.raises(ValueError, match=match):
        solve_full_universe_milp(
            graph,
            ["s"],
            ["t"],
            [["c", "q"]],
            action_budget=2,
            candidate_actions=sorted(graph.edges),
            time_limit_s=30.0,
        )


def test_full_universe_model_consistency_includes_initially_unreachable_pairs() -> None:
    graph = _manual_graph()
    graph.add_node(
        "orphan-source",
        type="endpoint",
        dept="d",
        tier=0,
        criticality=0.0,
    )
    result = solve_full_universe_milp(
        graph,
        ["s", "orphan-source"],
        ["t"],
        [["c", "q"]],
        action_budget=0,
        candidate_actions=sorted(graph.edges),
        time_limit_s=30.0,
    )

    # The Cartesian model contains two source/asset pairs. One was already
    # unreachable and the other remains reachable with a zero-action budget.
    assert result.diagnostics.attack_pair_count == 1
    assert result.diagnostics.primary_model_attack_pair_count == 2
    assert result.diagnostics.primary_model_isolated_pair_count == 1
    assert result.diagnostics.independently_recomputed_residual_pair_count == 1
    assert result.diagnostics.independent_model_consistent is True


def test_declared_completion_set_is_exact_for_additive_set() -> None:
    graph = nx.DiGraph()
    for node in ["s", "a", "b", "g", "t", "c", "q"]:
        graph.add_node(node, type="router", dept="d", tier=1, criticality=0.0)
    for edge in [("s", "a"), ("a", "g"), ("g", "t"), ("c", "q")]:
        graph.add_edge(*edge, **_edge_attrs(required=edge == ("c", "q")))
    optional = [("s", "b", _edge_attrs()), ("b", "g", _edge_attrs())]

    robust = verify_declared_completion_set(
        graph,
        optional,
        [("g", "t")],
        ["s"],
        ["t"],
        [["c", "q"]],
    )
    nonrobust = verify_declared_completion_set(
        graph,
        optional,
        [("s", "a")],
        ["s"],
        ["t"],
        [["c", "q"]],
    )
    assert robust.robust_pass is True
    assert robust.maximal_completion_threat_blocked is True
    assert robust.minimal_completion_service_preservation == 1.0
    assert nonrobust.robust_pass is False
    assert nonrobust.residual_pairs_maximal == [("s", "t")]


def test_public_abilene_underlay_and_overlay_integrity() -> None:
    underlay, source = load_abilene_underlay()
    assert underlay.number_of_nodes() == 11
    assert underlay.number_of_edges() == 14
    assert nx.is_connected(underlay)
    assert "public" in source["source_kind"].lower()

    instance = generate_abilene_instance(7)
    assert instance.metadata["public_underlay_node_count"] == 11
    assert instance.metadata["public_underlay_undirected_edge_count"] == 14
    assert instance.metadata["overlay_is_synthetic"] is True
    assert instance.observed_graph.graph["public_topology"] == "Abilene"
    assert service_preservation(instance.observed_graph, instance.required_service_pairs_observed) == 1.0


def test_learned_proposer_trains_on_disjoint_exact_labels() -> None:
    # Hand-constructed examples keep the unit test fast while exercising the
    # complete feature, calibration, serialization, and proposal path.
    examples = []
    for index in range(8):
        graph = _manual_graph()
        graph.graph["snapshot_id"] = f"train-{index}"
        # Alternate between the two minimum cuts so both selected and
        # unselected edges occur across structurally similar examples.
        selected = [("x", "t"), ("y", "t")] if index % 2 == 0 else [("s", "x"), ("s", "y")]
        examples.append((graph, ["s"], ["t"], selected))
    model = fit_calibrated_edge_model(examples[:5], examples[5:])
    metrics = evaluate_edge_model(model, examples[5:])
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["ece_10"] <= 1.0
    assert np.isfinite(model.predict_proba(np.zeros((1, len(model.feature_names))))).all()

    graph = _manual_graph()
    capsule = LearnedServiceAwarePlanner(model).synthesize(
        graph,
        ["s"],
        ["t"],
        [["c", "q"]],
        action_budget=2,
        candidate_actions=sorted(graph.edges),
    )
    assert capsule.method == "Learned-CIN"
    assert len(capsule.interventions) <= 2
    # Admission authority is still the deterministic checker embedded by
    # build_capsule, regardless of learned proposal quality.
    assert capsule.service_safety_certificate["checker_version"] == "cin-verifier-1.7"


def test_independent_node_checker_accepts_and_rejects_tampering(tmp_path: Path) -> None:
    graph = _manual_graph()
    capsule = build_capsule(
        method="UnitTestMinimalCut",
        graph=graph,
        proposal_transformation=identity_transformation([("x", "t"), ("y", "t")]),
        candidate_actions=sorted(graph.edges),
        attack_sources=["s"],
        protected_assets=["t"],
        required_service_pairs=[["c", "q"]],
        action_budget=2,
        tau=1.0,
        path_scope=PathSearchScope(),
        issued_at=experiment_issuance_time(graph),
        max_snapshot_age_at_issue_seconds=EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
        max_capsule_age_seconds=EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
    )
    assert capsule.service_safety_certificate["passed"] is True
    bundle = make_verification_bundle(
        graph,
        capsule.to_dict(),
        ["s"],
        ["t"],
        [["c", "q"]],
        candidate_actions=capsule.candidate_actions,
        tau=1.0,
        action_budget=2,
        path_scope=PathSearchScope(),
        freshness=capsule.freshness,
    )
    raw = json.dumps(_jsonable(bundle), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    root = hashlib.sha256(raw).hexdigest()
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_bytes(raw)
    checker = Path(__file__).resolve().parents[1] / "independent_checker" / "verify_bundle.mjs"
    result = subprocess.run(
        [
            "node",
            str(checker),
            "--bundle",
            str(bundle_path),
            "--expected-root",
            root,
            "--verification-time",
            capsule.freshness["issued_at"],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads(result.stdout)
    assert result.returncode == 0, report
    assert report["passed"] is True

    tampered = json.loads(raw)
    tampered["capsule"]["method"] = "tampered"
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    rejected = subprocess.run(
        [
            "node",
            str(checker),
            "--bundle",
            str(tampered_path),
            "--expected-root",
            root,
            "--verification-time",
            capsule.freshness["issued_at"],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert json.loads(rejected.stdout)["passed"] is False



def test_documented_bundle_generator_interoperates_with_node(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    bundle_path = tmp_path / "generated_bundle.json"
    generated = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "verify_certificate.py"),
            "--output",
            str(bundle_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert generated.returncode == 0, generated.stderr or generated.stdout
    generation_report = json.loads(generated.stdout)
    root = generation_report["verification_bundle_sha256"]
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    issued_at = bundle["constraints"]["freshness"]["issued_at"]

    checked = subprocess.run(
        [
            "node",
            str(project_root / "independent_checker" / "verify_bundle.mjs"),
            "--bundle",
            str(bundle_path),
            "--expected-root",
            root,
            "--verification-time",
            issued_at,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads(checked.stdout)
    assert checked.returncode == 0, report
    assert report["passed"] is True
    assert report["actual_bundle_sha256"] == root


def test_controller_semantics_provenance_authorization_and_rollback() -> None:
    graph = _manual_graph()
    interventions = [("x", "t"), ("y", "t")]
    policy = compile_openflow13_policy(graph, interventions)
    equivalence = verify_compiler_equivalence(graph, interventions, policy)
    assert equivalence["passed"] is True

    staged = staged_install_with_monitoring(
        graph,
        policy,
        ["s"],
        ["t"],
        [["c", "q"]],
        batch_size=1,
    )
    assert staged.committed is True
    assert staged.service_preservation_after_install == 1.0
    assert staged.residual_attack_pair_count_after_install == 0

    rollback = exercise_bad_policy_rollback(graph, [["c", "q"]])
    assert rollback["exercised"] is True
    assert rollback["restored"] is True

    private_key, public_key = deterministic_experiment_keypair()
    attestation = sign_snapshot_attestation(
        graph,
        private_key,
        public_key,
        signer_id="unit-test-discovery",
        discovery_source="unit-test",
        discovery_evidence={"fixture": "manual"},
        issued_at=graph.graph["snapshot_time"],
    )
    trusted = {"unit-test-discovery": public_key}
    assert verify_snapshot_attestation(graph, attestation, trusted_signers=trusted)["passed"] is True
    tampered_graph = graph.copy()
    tampered_graph.remove_edge("x", "t")
    assert verify_snapshot_attestation(tampered_graph, attestation, trusted_signers=trusted)["passed"] is False

    types = tuple(sorted({str(data["type"]) for _, data in graph.nodes(data=True)}))
    authorization = authorize_interventions(
        graph,
        interventions,
        AuthorizationPolicy(
            schema="cin.authorization-policy.v1",
            policy_id="unit-test-policy",
            max_actions=2,
            allowed_source_types=types,
            allowed_destination_types=types,
        ),
    )
    assert authorization["passed"] is True

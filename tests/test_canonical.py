"""Regression and adversarial tests for the canonical artifact."""
from __future__ import annotations

from copy import deepcopy
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import networkx as nx
import pandas as pd

from cin.baselines import exact_small_cut_oracle, greedy_risk, service_aware_greedy_cut
from cin.capsule import (
    CertifiedInterventionPlanner,
    EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
    EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
    build_capsule,
    experiment_issuance_time,
)
from cin.repair import SubsetMinimalRepair, identity_transformation
from cin.risk_model import ObservedRiskScorer
from cin.experiments import (
    _paired_cluster_data,
    write_latex_fragments,
    write_paper_tables,
    write_reviewed_latex_fragments,
    write_statistical_report,
)
from cin.graph_generator import BenchmarkConfig, generate_network
from cin.metrics import evaluate_capsule
from cin.simulator import (
    PathSearchScope,
    all_attack_routes_blocked,
    apply_interventions,
    enumerate_attack_paths,
    service_preservation,
)
from cin.verifier import (
    make_verification_bundle,
    stable_hash,
    verification_bundle_hash,
    verify_capsule_bundle,
)


def _edge(graph: nx.DiGraph, source: str, destination: str, **overrides) -> None:
    values = {
        "kind": "comm",
        "risk_prob": 0.5,
        "anomaly_score": 0.5,
        "action_cost": 1.0,
    }
    values.update(overrides)
    graph.add_edge(source, destination, **values)


def _set_test_provenance(graph: nx.DiGraph, snapshot_id: str) -> None:
    graph.graph.update(
        snapshot_id=snapshot_id,
        topology_version=1,
        snapshot_time="2026-07-26T00:00:00Z",
    )


def _toy_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    _set_test_provenance(graph, "toy-snapshot")
    for node, node_type, tier in [
        ("source", "iot", 0),
        ("left", "service", 1),
        ("right", "service", 1),
        ("asset", "asset", 3),
        ("client", "device", 0),
        ("database", "database", 2),
    ]:
        graph.add_node(node, type=node_type, tier=tier, dept="d0")
    _edge(graph, "source", "left")
    _edge(graph, "left", "asset")
    _edge(graph, "source", "right")
    _edge(graph, "right", "asset")
    _edge(graph, "client", "left", required=True)
    _edge(graph, "left", "database", required=True)
    _edge(graph, "client", "right", required=True)
    _edge(graph, "right", "database", required=True)
    return graph


def _passing_capsule_and_bundle(*, risk_prob: float = 0.5):
    graph = nx.DiGraph()
    _set_test_provenance(graph, "passing-capsule-snapshot")
    for node in ("source", "middle", "asset", "client", "database"):
        graph.add_node(node, type="service", tier=1, dept="d0")
    _edge(graph, "source", "middle", risk_prob=risk_prob)
    _edge(graph, "middle", "asset")
    _edge(graph, "client", "database", required=True)
    required_pairs = [("client", "database")]
    capsule = build_capsule(
        method="TestPlanner",
        graph=graph,
        proposal_transformation=identity_transformation([("source", "middle")]),
        candidate_actions=sorted(graph.edges),
        attack_sources=["source"],
        protected_assets=["asset"],
        required_service_pairs=required_pairs,
        action_budget=1,
        path_scope=PathSearchScope(cutoff_edges=2),
        issued_at=experiment_issuance_time(graph),
        max_snapshot_age_at_issue_seconds=EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
        max_capsule_age_seconds=EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
    )
    bundle = make_verification_bundle(
        graph,
        capsule.to_dict(),
        ["source"],
        ["asset"],
        required_pairs,
        candidate_actions=capsule.candidate_actions,
        tau=1.0,
        action_budget=1,
        path_scope=PathSearchScope(cutoff_edges=2),
        freshness=capsule.freshness,
    )
    return capsule, bundle


def _historical_verify(bundle, *, expected_bundle_sha256=None):
    """Deterministic replay used by tests; production default is wall-clock UTC."""
    issued_at = bundle["constraints"]["freshness"]["issued_at"]
    return verify_capsule_bundle(
        bundle,
        expected_bundle_sha256=(
            expected_bundle_sha256 or verification_bundle_hash(bundle)
        ),
        verification_time=issued_at,
    )


class SemanticsTests(unittest.TestCase):
    def test_service_requirement_uses_alternate_endpoint_route(self):
        graph = _toy_graph()
        intervened = apply_interventions(graph, [("left", "database")])
        self.assertEqual(service_preservation(intervened, [("client", "database")]), 1.0)

    def test_long_attack_path_cannot_bypass_cutoff(self):
        graph = nx.DiGraph()
        _set_test_provenance(graph, "long-path-snapshot")
        nodes = ["source", "n1", "n2", "n3", "n4", "n5", "n6", "asset"]
        for node in nodes:
            graph.add_node(node, type="service", tier=1, dept="d0")
        for source, destination in zip(nodes[:-1], nodes[1:]):
            _edge(graph, source, destination)
        self.assertEqual(enumerate_attack_paths(graph, ["source"], ["asset"], cutoff=6), [])
        capsule = build_capsule(
            method="BoundedPlanner",
            graph=graph,
            proposal_transformation=identity_transformation([]),
            candidate_actions=sorted(graph.edges),
            attack_sources=["source"],
            protected_assets=["asset"],
            required_service_pairs=[],
            action_budget=0,
            path_scope=PathSearchScope(cutoff_edges=6),
            issued_at=experiment_issuance_time(graph),
            max_snapshot_age_at_issue_seconds=EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
            max_capsule_age_seconds=EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
        )
        certificate = capsule.service_safety_certificate
        self.assertFalse(certificate["passed"])
        self.assertFalse(certificate["attack_reachability_blocked"])
        self.assertEqual(certificate["residual_attack_pairs"], 1)
        self.assertEqual(len(certificate["witnesses"]["residual_attack_shortest_paths"][0]) - 1, 7)

    def test_common_pruner_produces_subset_minimal_policy(self):
        capsule, _bundle = _passing_capsule_and_bundle()
        graph = nx.DiGraph()
        _set_test_provenance(graph, "redundant-proposal-snapshot")
        for node in ("source", "middle", "asset"):
            graph.add_node(node, type="service", tier=1, dept="d0")
        _edge(graph, "source", "middle")
        _edge(graph, "middle", "asset")
        redundant = build_capsule(
            method="RedundantProposal",
            graph=graph,
            proposal_transformation=SubsetMinimalRepair.apply(
                graph,
                [("source", "middle"), ("middle", "asset")],
                ["source"],
                ["asset"],
                [],
                tau=1.0,
            ),
            candidate_actions=sorted(graph.edges),
            attack_sources=["source"],
            protected_assets=["asset"],
            required_service_pairs=[],
            action_budget=2,
            issued_at=experiment_issuance_time(graph),
            max_snapshot_age_at_issue_seconds=EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
            max_capsule_age_seconds=EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
        )
        self.assertEqual(len(redundant.interventions), 1)
        self.assertTrue(redundant.service_safety_certificate["subset_minimal"])
        self.assertTrue(redundant.service_safety_certificate["passed"])
        self.assertTrue(capsule.service_safety_certificate["passed"])

        certificate = capsule.service_safety_certificate
        witnesses = certificate["witnesses"]
        self.assertEqual(
            set(witnesses),
            {
                "residual_attack_shortest_paths",
                "threat_nonreachability",
                "necessary_action_paths",
                "surviving_service_paths",
                "removable_interventions",
            },
        )
        self.assertEqual(witnesses["residual_attack_shortest_paths"], [])
        self.assertEqual(witnesses["removable_interventions"], [])
        self.assertEqual(
            witnesses["surviving_service_paths"],
            {
                "schema": "cin.service-path-witnesses.v1",
                "required_pair_count": 1,
                "entries": [
                    {
                        "source": "client",
                        "destination": "database",
                        "path": ["client", "database"],
                    }
                ],
            },
        )
        nonreachability = certificate["witnesses"]["threat_nonreachability"]
        self.assertEqual(nonreachability["source_reachable_closure"], ["source"])
        self.assertEqual(nonreachability["protected_assets_in_closure"], [])
        self.assertEqual(nonreachability["active_edges_leaving_closure"], [])
        self.assertTrue(nonreachability["proves_source_asset_nonreachability"])
        necessity = certificate["witnesses"]["necessary_action_paths"]
        self.assertEqual(len(necessity), len(capsule.interventions))
        self.assertEqual(necessity[0]["restored_attack_path"], ["source", "middle", "asset"])
        self.assertTrue(necessity[0]["action_on_restored_path"])
        self.assertTrue(certificate["proof_witnesses_valid"])

        capsule_payload = capsule.to_dict()
        typed_sketch = capsule_payload["identifier_omitting_typed_sketch"]
        self.assertEqual(typed_sketch["schema"], "cin.typed-sketch.v1")
        self.assertEqual(typed_sketch["identifier_namespace"], "omitted")
        self.assertEqual(typed_sketch["raw_identifier_count"], 0)
        self.assertTrue(certificate["typed_sketch_schema_valid"])
        self.assertTrue(certificate["identifier_omission_valid"])
        self.assertIn("typed_sketch_sha256", certificate["bindings"])

    def test_service_witness_records_are_injective_for_delimiter_identifiers(self):
        graph = nx.DiGraph()
        _set_test_provenance(graph, "delimiter-witness-snapshot")
        for node in ("source", "middle", "asset", "a->b", "c", "a", "b->c"):
            graph.add_node(node, type="service", tier=1, dept="d0")
        _edge(graph, "source", "middle")
        _edge(graph, "middle", "asset")
        _edge(graph, "a->b", "c", required=True)
        _edge(graph, "a", "b->c", required=True)
        required_pairs = [("a->b", "c"), ("a", "b->c")]
        capsule = build_capsule(
            method="DelimiterSafeWitness",
            graph=graph,
            proposal_transformation=identity_transformation([("middle", "asset")]),
            candidate_actions=sorted(graph.edges),
            attack_sources=["source"],
            protected_assets=["asset"],
            required_service_pairs=required_pairs,
            action_budget=1,
            issued_at=experiment_issuance_time(graph),
            max_snapshot_age_at_issue_seconds=EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
            max_capsule_age_seconds=EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
        )
        certificate = capsule.service_safety_certificate
        self.assertTrue(certificate["passed"])
        self.assertTrue(certificate["proof_witnesses_valid"])
        service_witnesses = certificate["witnesses"]["surviving_service_paths"]
        self.assertEqual(service_witnesses["schema"], "cin.service-path-witnesses.v1")
        self.assertEqual(service_witnesses["required_pair_count"], 2)
        self.assertEqual(
            [(entry["source"], entry["destination"]) for entry in service_witnesses["entries"]],
            [("a", "b->c"), ("a->b", "c")],
        )

        bundle = make_verification_bundle(
            graph,
            capsule.to_dict(),
            ["source"],
            ["asset"],
            required_pairs,
            candidate_actions=capsule.candidate_actions,
            tau=1.0,
            action_budget=1,
            path_scope=PathSearchScope(),
            freshness=capsule.freshness,
        )
        self.assertTrue(_historical_verify(bundle).passed)

        incomplete = deepcopy(bundle)
        incomplete["capsule"]["service_safety_certificate"]["witnesses"][
            "surviving_service_paths"
        ]["entries"].pop()
        result = _historical_verify(incomplete)
        self.assertFalse(result.passed)
        self.assertFalse(result.submitted_witnesses_valid)

    def test_action_set_serialization_is_canonical_across_all_permutations(self):
        graph = nx.DiGraph()
        _set_test_provenance(graph, "permutation-snapshot")
        for node in (
            "source",
            "left",
            "middle",
            "right",
            "asset_left",
            "asset_middle",
            "asset_right",
            "client",
            "database",
        ):
            graph.add_node(node, type="service", tier=1, dept="d0")
        actions = [
            ("left", "asset_left"),
            ("middle", "asset_middle"),
            ("right", "asset_right"),
        ]
        for branch, asset in (
            ("left", "asset_left"),
            ("middle", "asset_middle"),
            ("right", "asset_right"),
        ):
            _edge(graph, "source", branch)
            _edge(graph, branch, asset)
        _edge(graph, "client", "database", required=True)

        capsule_payloads = []
        bundle_roots = set()
        for action_order in permutations(actions):
            capsule = build_capsule(
                method="PermutationCanonical",
                graph=graph,
                proposal_transformation=identity_transformation(list(action_order)),
                candidate_actions=sorted(graph.edges),
                attack_sources=["source"],
                protected_assets=["asset_left", "asset_middle", "asset_right"],
                required_service_pairs=[("client", "database")],
                action_budget=3,
                issued_at=experiment_issuance_time(graph),
                max_snapshot_age_at_issue_seconds=EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
                max_capsule_age_seconds=EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
            )
            self.assertEqual(capsule.interventions, sorted(actions))
            payload = capsule.to_dict()
            capsule_payloads.append(payload)
            bundle = make_verification_bundle(
                graph,
                payload,
                ["source"],
                ["asset_left", "asset_middle", "asset_right"],
                [("client", "database")],
                candidate_actions=capsule.candidate_actions,
                tau=1.0,
                action_budget=3,
                path_scope=PathSearchScope(),
                freshness=capsule.freshness,
            )
            self.assertTrue(_historical_verify(bundle).passed)
            bundle_roots.add(verification_bundle_hash(bundle))
        self.assertTrue(all(payload == capsule_payloads[0] for payload in capsule_payloads))
        self.assertEqual(len(bundle_roots), 1)

        with self.assertRaisesRegex(ValueError, "duplicate directed edge"):
            build_capsule(
                method="DuplicateProposal",
                graph=graph,
                proposal_transformation=identity_transformation(actions + [actions[0]]),
                candidate_actions=sorted(graph.edges),
                attack_sources=["source"],
                protected_assets=["asset_left", "asset_middle", "asset_right"],
                required_service_pairs=[("client", "database")],
                action_budget=4,
                issued_at=experiment_issuance_time(graph),
                max_snapshot_age_at_issue_seconds=EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
                max_capsule_age_seconds=EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
            )

        duplicate_bundle = make_verification_bundle(
            graph,
            capsule_payloads[0],
            ["source"],
            ["asset_left", "asset_middle", "asset_right"],
            [("client", "database")],
            candidate_actions=capsule_payloads[0]["candidate_actions"],
            tau=1.0,
            action_budget=3,
            path_scope=PathSearchScope(),
            freshness=capsule_payloads[0]["freshness"],
        )
        duplicate_bundle["capsule"]["interventions"].append(list(actions[0]))
        with self.assertRaisesRegex(ValueError, "duplicate directed edge"):
            _historical_verify(duplicate_bundle)


class AdversarialVerifierTests(unittest.TestCase):
    def test_untampered_serialized_bundle_passes(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        root = verification_bundle_hash(bundle)
        result = _historical_verify(bundle, expected_bundle_sha256=root)
        self.assertTrue(result.passed)
        self.assertEqual(result.verification_bundle_sha256, root)
        self.assertEqual(result.expected_bundle_sha256, root)
        self.assertTrue(result.bundle_hash_matches)
        self.assertTrue(result.embedded_issuance_certificate_valid)

    def test_fully_recomputed_bundle_substitution_fails_original_external_root(self):
        _original_capsule, original = _passing_capsule_and_bundle(risk_prob=0.5)
        _substitute_capsule, substitute = _passing_capsule_and_bundle(risk_prob=0.7)
        original_root = verification_bundle_hash(original)
        substitute_root = verification_bundle_hash(substitute)
        self.assertNotEqual(original_root, substitute_root)
        self.assertTrue(
            _historical_verify(
                substitute,
                expected_bundle_sha256=substitute_root,
            ).passed
        )

        result = _historical_verify(
            substitute,
            expected_bundle_sha256=original_root,
        )
        self.assertFalse(result.passed)
        self.assertFalse(result.bundle_hash_matches)
        self.assertEqual(result.verification_bundle_sha256, substitute_root)
        self.assertEqual(result.expected_bundle_sha256, original_root)
        self.assertTrue(result.embedded_issuance_certificate_valid)
        self.assertTrue(
            any("externally trusted expected root" in message for message in result.messages)
        )

    def test_cli_returns_failure_status_for_rejected_bundle(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        issued_at = bundle["constraints"]["freshness"]["issued_at"]
        code_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "bundle.json"
            output_path = Path(directory) / "verification.json"
            input_path.write_text(
                json.dumps(bundle, allow_nan=False),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(code_root / "scripts" / "verify_certificate.py"),
                    "--input",
                    str(input_path),
                    "--expected-bundle-sha256",
                    "0" * 64,
                    "--verification-time",
                    issued_at,
                    "--output",
                    str(output_path),
                ],
                cwd=code_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 1, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertFalse(summary["certificate_passed"])
            self.assertFalse(summary["bundle_hash_matches"])
            self.assertTrue(output_path.exists())

    def test_cli_rejects_nonfinite_json_constants(self):
        code_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "nonfinite.json"
            input_path.write_text('{"schema": NaN}', encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(code_root / "scripts" / "verify_certificate.py"),
                    "--input",
                    str(input_path),
                    "--expected-bundle-sha256",
                    "0" * 64,
                ],
                cwd=code_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("non-finite JSON constant is not permitted", completed.stderr)

    def test_cli_reports_malformed_replay_inputs_as_usage_errors(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        code_root = Path(__file__).resolve().parents[1]
        malformed_bundle = {"schema": "cin.verification-bundle.v1"}
        issued_at = bundle["constraints"]["freshness"]["issued_at"]
        cases = [
            ("digest", bundle, "not-a-digest", issued_at, "expected_bundle_sha256"),
            (
                "time",
                bundle,
                verification_bundle_hash(bundle),
                "not-a-time",
                "verification_time must be",
            ),
            (
                "structure",
                malformed_bundle,
                verification_bundle_hash(malformed_bundle),
                None,
                "invalid verification bundle",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for label, payload, expected_root, verification_time, message in cases:
                with self.subTest(label=label):
                    input_path = Path(directory) / f"{label}.json"
                    output_path = Path(directory) / f"{label}-result.json"
                    input_path.write_text(
                        json.dumps(payload, allow_nan=False),
                        encoding="utf-8",
                    )
                    command = [
                        sys.executable,
                        str(code_root / "scripts" / "verify_certificate.py"),
                        "--input",
                        str(input_path),
                        "--expected-bundle-sha256",
                        expected_root,
                        "--output",
                        str(output_path),
                    ]
                    if verification_time is not None:
                        command.extend(["--verification-time", verification_time])
                    completed = subprocess.run(
                        command,
                        cwd=code_root,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(completed.returncode, 2)
                    self.assertIn(message, completed.stderr)
                    self.assertNotIn("Traceback", completed.stderr)
                    self.assertFalse(output_path.exists())

    def test_embedded_decision_fields_must_match_issuance_recomputation(self):
        for field in ("passed", "attack_reachability_blocked", "service_pairs_preserved"):
            with self.subTest(field=field):
                _capsule, bundle = _passing_capsule_and_bundle()
                tampered = deepcopy(bundle)
                certificate = tampered["capsule"]["service_safety_certificate"]
                certificate[field] = not certificate[field]
                # Trust the modified transport root to isolate the independent
                # embedded-certificate recomputation check.
                result = _historical_verify(
                    tampered,
                    expected_bundle_sha256=verification_bundle_hash(tampered),
                )
                self.assertFalse(result.passed)
                self.assertTrue(result.bundle_hash_matches)
                self.assertFalse(result.embedded_issuance_certificate_valid)
                self.assertTrue(
                    any(
                        "issuance-time recomputation" in message
                        for message in result.messages
                    )
                )

    def test_graph_tampering_breaks_hash_binding(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        tampered = deepcopy(bundle)
        tampered["graph"]["edges"][0]["attributes"]["risk_prob"] = 0.123456
        result = _historical_verify(tampered)
        self.assertFalse(result.passed)
        self.assertTrue(any("bindings" in message for message in result.messages))

    def test_policy_tampering_fails_equivalence_and_binding(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        tampered = deepcopy(bundle)
        tampered["capsule"]["abstract_policy_json"]["rules"][0]["destination"] = "asset"
        result = _historical_verify(tampered)
        self.assertFalse(result.passed)
        self.assertFalse(result.policy_equivalent)

    def test_unknown_capsule_and_checker_versions_are_rejected(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        unknown_capsule = deepcopy(bundle)
        unknown_capsule["capsule"]["schema"] = "cin.defense-capsule.v999"
        with self.assertRaisesRegex(ValueError, "unsupported defense capsule schema"):
            _historical_verify(unknown_capsule)
        unknown_checker = deepcopy(bundle)
        unknown_checker["capsule"]["service_safety_certificate"]["checker_version"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported embedded checker version"):
            _historical_verify(unknown_checker)

    def test_invalid_rollback_schema_is_rejected(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        tampered = deepcopy(bundle)
        tampered["capsule"]["rollback_condition"]["operator"] = ">"
        result = _historical_verify(tampered)
        self.assertFalse(result.passed)
        self.assertFalse(result.rollback_schema_valid)

    def test_submitted_witness_tampering_is_rejected(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        tampered = deepcopy(bundle)
        tampered["capsule"]["service_safety_certificate"]["witnesses"]["surviving_service_paths"] = {
            "fabricated->route": ["fabricated", "route"]
        }
        result = _historical_verify(tampered)
        self.assertFalse(result.passed)
        self.assertFalse(result.submitted_witnesses_valid)

    def test_nonreachability_and_action_necessity_witness_tampering_is_rejected(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        closure_tamper = deepcopy(bundle)
        closure_tamper["capsule"]["service_safety_certificate"]["witnesses"]["threat_nonreachability"][
            "source_reachable_closure"
        ].append("asset")
        closure_result = _historical_verify(closure_tamper)
        self.assertFalse(closure_result.passed)
        self.assertFalse(closure_result.submitted_witnesses_valid)

        necessity_tamper = deepcopy(bundle)
        necessity_tamper["capsule"]["service_safety_certificate"]["witnesses"]["necessary_action_paths"][0][
            "restored_attack_path"
        ] = ["source", "asset"]
        necessity_result = _historical_verify(necessity_tamper)
        self.assertFalse(necessity_result.passed)
        self.assertFalse(necessity_result.submitted_witnesses_valid)

    def test_literal_graph_identifier_in_typed_sketch_is_rejected(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        tampered = deepcopy(bundle)
        tampered["capsule"]["identifier_omitting_typed_sketch"]["leaked_identifier"] = "source"
        result = _historical_verify(tampered)
        self.assertFalse(result.passed)
        self.assertFalse(result.typed_sketch_schema_valid)
        self.assertFalse(result.identifier_omission_valid)
        self.assertTrue(any("literal graph identifier" in message for message in result.messages))

    def test_capsule_and_constraint_freshness_must_match(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        tampered = deepcopy(bundle)
        tampered["capsule"]["freshness"]["expires_at"] = "2099-01-01T00:00:00Z"
        result = _historical_verify(tampered)
        self.assertFalse(result.passed)
        self.assertFalse(result.capsule_freshness_matches)

    def test_stale_certificate_is_rejected(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        stale = deepcopy(bundle)
        # A malicious bundle-supplied time is unsupported and cannot override
        # the independent checker's external clock.
        stale["constraints"]["freshness"]["verification_time"] = "2026-07-26T00:00:00Z"
        result = verify_capsule_bundle(
            stale,
            expected_bundle_sha256=verification_bundle_hash(stale),
            verification_time="2026-09-01T00:00:00Z",
        )
        self.assertFalse(result.passed)
        self.assertFalse(result.freshness_valid)
        self.assertTrue(any("stale" in message for message in result.messages))

    def test_duplicate_nodes_and_edges_are_rejected_during_deserialization(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        duplicate_node = deepcopy(bundle)
        duplicate_node["graph"]["nodes"].append(deepcopy(duplicate_node["graph"]["nodes"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate node"):
            _historical_verify(duplicate_node)
        duplicate_edge = deepcopy(bundle)
        duplicate_edge["graph"]["edges"].append(deepcopy(duplicate_edge["graph"]["edges"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate directed edge"):
            _historical_verify(duplicate_edge)

    def test_budget_tampering_is_rejected(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        tampered = deepcopy(bundle)
        tampered["constraints"]["action_budget"] = 0
        result = _historical_verify(tampered)
        self.assertFalse(result.passed)
        self.assertFalse(result.budget_respected)

    def test_capsule_budget_copy_must_match_bound_constraint(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        tampered = deepcopy(bundle)
        tampered["capsule"]["action_budget"] = 99
        result = _historical_verify(tampered)
        self.assertFalse(result.passed)
        self.assertFalse(result.capsule_budget_matches)

    def test_invalid_threat_scopes_are_rejected(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        cases = []
        empty = deepcopy(bundle)
        empty["threat_model"]["attack_sources"] = []
        cases.append(empty)
        absent = deepcopy(bundle)
        absent["threat_model"]["protected_assets"] = ["absent-node"]
        cases.append(absent)
        overlap = deepcopy(bundle)
        overlap["threat_model"]["protected_assets"] = ["source"]
        cases.append(overlap)
        duplicate = deepcopy(bundle)
        duplicate["threat_model"]["attack_sources"] = ["source", "source"]
        cases.append(duplicate)
        for case in cases:
            with self.subTest(case=case["threat_model"]):
                result = _historical_verify(case)
                self.assertFalse(result.passed)
                self.assertFalse(result.threat_scope_valid)

    def test_malformed_service_threshold_is_rejected_without_coercion(self):
        _capsule, bundle = _passing_capsule_and_bundle()
        for invalid in ("1.0", True, -0.1, 0.0, 0.5, 0.999, 1.1):
            tampered = deepcopy(bundle)
            tampered["constraints"]["service_threshold"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "service threshold"):
                _historical_verify(tampered)

    def test_capsule_apis_explicitly_reject_fractional_service_thresholds(self):
        graph = nx.DiGraph()
        _set_test_provenance(graph, "fractional-threshold-snapshot")
        for node in ("source", "asset"):
            graph.add_node(node, type="service", tier=1, dept="d0")
        _edge(graph, "source", "asset")
        with self.assertRaisesRegex(ValueError, "must equal 1.0"):
            build_capsule(
                method="FractionalThreshold",
                graph=graph,
                proposal_transformation=identity_transformation([("source", "asset")]),
                candidate_actions=sorted(graph.edges),
                attack_sources=["source"],
                protected_assets=["asset"],
                required_service_pairs=[],
                action_budget=1,
                tau=0.5,
                issued_at=experiment_issuance_time(graph),
                max_snapshot_age_at_issue_seconds=EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
                max_capsule_age_seconds=EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
            )
        with self.assertRaisesRegex(ValueError, "must equal 1.0"):
            CertifiedInterventionPlanner(tau=0.5)

        capsule, bundle = _passing_capsule_and_bundle()
        with self.assertRaisesRegex(ValueError, "must equal 1.0"):
            make_verification_bundle(
                graph,
                capsule.to_dict(),
                ["source"],
                ["asset"],
                [],
                candidate_actions=sorted(graph.edges),
                tau=0.5,
                action_budget=1,
                path_scope=PathSearchScope(),
                freshness=capsule.freshness,
            )

    def test_capsule_builder_never_silently_truncates_over_budget_proposal(self):
        graph = nx.DiGraph()
        _set_test_provenance(graph, "over-budget-snapshot")
        for node in ("source", "middle", "asset"):
            graph.add_node(node, type="service", tier=1, dept="d0")
        _edge(graph, "source", "middle")
        _edge(graph, "middle", "asset")
        with self.assertRaisesRegex(ValueError, "exogenous budget"):
            build_capsule(
                method="OverBudget",
                graph=graph,
                proposal_transformation=identity_transformation(
                    [("source", "middle"), ("middle", "asset")]
                ),
                candidate_actions=sorted(graph.edges),
                attack_sources=["source"],
                protected_assets=["asset"],
                required_service_pairs=[],
                action_budget=1,
                issued_at=experiment_issuance_time(graph),
                max_snapshot_age_at_issue_seconds=EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
                max_capsule_age_seconds=EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
            )


class PipelineTests(unittest.TestCase):
    def test_random_tiebreak_is_explicit_and_removed_source_ablation_is_rejected(self):
        graph = nx.DiGraph()
        _set_test_provenance(graph, "random-tiebreak-snapshot")
        for node, node_type, tier in [
            ("source", "iot", 0),
            ("left", "service", 1),
            ("right", "service", 1),
            ("asset_left", "asset", 3),
            ("asset_right", "asset", 3),
        ]:
            graph.add_node(node, type=node_type, tier=tier, dept="d0")
        _edge(graph, "source", "left", risk_prob=0.01, anomaly_score=0.01, kind="comm")
        _edge(graph, "left", "asset_left", risk_prob=0.95, anomaly_score=1.0, kind="privilege")
        _edge(graph, "source", "right", risk_prob=0.01, anomaly_score=0.01, kind="comm")
        _edge(graph, "right", "asset_right", risk_prob=0.95, anomaly_score=1.0, kind="privilege")
        model = ObservedRiskScorer(cutoff=8)
        model.fit(graph, ["source"], ["asset_left", "asset_right"])
        planner = CertifiedInterventionPlanner(tau=1.0, cutoff=8)
        random_tiebreak = planner.synthesize(
            graph,
            ["source"],
            ["asset_left", "asset_right"],
            [],
            model,
            action_budget=2,
            candidate_actions=sorted(graph.edges),
            variant="random_tiebreak",
        )
        self.assertEqual(random_tiebreak.method, "CIN-Ablation-random_tiebreak")
        with self.assertRaisesRegex(ValueError, "unsupported planner variant"):
            planner.synthesize(
                graph,
                ["source"],
                ["asset_left", "asset_right"],
                [],
                model,
                action_budget=2,
                candidate_actions=sorted(graph.edges),
                variant="no_source_boundary",
            )

    def test_generated_ingress_prefix_is_shared_required_and_unbypassable(self):
        instance = generate_network(
            BenchmarkConfig(size="small", noise=0.3, hidden=0.2, density="high", seed=7)
        )
        true_pairs = set(instance.required_service_pairs_true)
        observed_pairs = set(instance.required_service_pairs_observed)
        for graph in (instance.true_graph, instance.observed_graph):
            for source in instance.attack_sources:
                dept_index = int(graph.nodes[source]["dept"].removeprefix("dept"))
                gateway = f"gateway_{dept_index}"
                broker = f"broker_{dept_index}"
                monitor = f"monitor_{dept_index}"
                self.assertEqual(set(graph.successors(source)), {gateway})
                self.assertEqual(set(graph.successors(gateway)), {broker})
                self.assertTrue(graph.has_edge(broker, monitor))
                self.assertIn((source, monitor), true_pairs)
                self.assertIn((source, monitor), observed_pairs)
                paths = [
                    path
                    for path in enumerate_attack_paths(
                        graph,
                        [source],
                        instance.protected_assets,
                        cutoff=8,
                    )
                    if path[0] == source
                ]
                self.assertTrue(paths)
                self.assertTrue(all(path[:3] == (source, gateway, broker) for path in paths))
        self.assertTrue(
            all(
                instance.observed_graph.nodes[source].get("type") not in {"iot", "gateway", "monitor"}
                for source, _destination in instance.spurious_edges
            )
        )

    def test_source_boundary_breaks_service_but_interior_cut_is_feasible(self):
        instance = generate_network(
            BenchmarkConfig(size="small", noise=0.2, hidden=0.2, density="high", seed=7)
        )
        source_set = set(instance.attack_sources)
        boundary = sorted(
            (source, destination)
            for source, destination in instance.observed_graph.edges
            if source in source_set
        )
        self.assertEqual(len(boundary), len(source_set))
        for edge in boundary:
            self.assertLess(
                service_preservation(
                    apply_interventions(instance.observed_graph, [edge]),
                    instance.required_service_pairs_observed,
                ),
                1.0,
            )
        boundary_graph = apply_interventions(instance.observed_graph, boundary)
        self.assertTrue(
            all_attack_routes_blocked(
                boundary_graph,
                instance.attack_sources,
                instance.protected_assets,
            )
        )
        self.assertLess(
            service_preservation(boundary_graph, instance.required_service_pairs_observed),
            1.0,
        )

        brokers = {
            f"broker_{int(instance.observed_graph.nodes[source]['dept'].removeprefix('dept'))}"
            for source in source_set
        }
        interior = sorted(
            (source, destination)
            for source, destination in instance.observed_graph.edges
            if source in brokers and instance.observed_graph.nodes[destination].get("type") != "monitor"
        )
        self.assertTrue(interior)
        interior_graph = apply_interventions(instance.observed_graph, interior)
        self.assertTrue(
            all_attack_routes_blocked(
                interior_graph,
                instance.attack_sources,
                instance.protected_assets,
            )
        )
        self.assertEqual(
            service_preservation(interior_graph, instance.required_service_pairs_observed),
            1.0,
        )

    def test_boundary_joint_admissibility_requires_feasibility_and_budget(self):
        instance = generate_network(
            BenchmarkConfig(size="small", noise=0.2, hidden=0.2, density="low", seed=1)
        )
        capsule = greedy_risk(
            instance.observed_graph,
            instance.attack_sources,
            instance.protected_assets,
            instance.required_service_pairs_observed,
            budget=6,
            candidate_actions=sorted(instance.observed_graph.edges),
        )
        row = evaluate_capsule(instance, capsule, runtime_s=0.0)

        self.assertEqual(row["source_boundary_threat_service_feasible_observed"], 0.0)
        self.assertEqual(row["source_boundary_admissible_under_budget_observed"], 0.0)
        self.assertEqual(row["broker_boundary_threat_service_feasible_observed"], 1.0)
        brokers = {
            f"broker_{int(instance.observed_graph.nodes[source]['dept'].removeprefix('dept'))}"
            for source in set(instance.attack_sources)
        }
        broker_boundary = [
            (source, destination)
            for source, destination in instance.observed_graph.edges
            if source in brokers
            and instance.observed_graph.nodes[destination].get("type") != "monitor"
        ]
        expected_within_budget = float(len(broker_boundary) <= capsule.action_budget)
        self.assertEqual(row["broker_boundary_within_budget"], expected_within_budget)
        self.assertEqual(
            row["broker_boundary_admissible_under_budget_observed"],
            row["broker_boundary_threat_service_feasible_observed"] * expected_within_budget,
        )
        for prefix in ("source_boundary", "broker_boundary"):
            self.assertEqual(
                row[f"{prefix}_admissible_under_budget_observed"],
                row[f"{prefix}_threat_service_feasible_observed"]
                * row[f"{prefix}_within_budget"],
            )
            self.assertNotIn(f"{prefix}_spec_feasible_observed", row)

    def test_generated_canonical_planners_do_not_cut_sources_or_gateways(self):
        configs = [
            BenchmarkConfig(size="small", noise=0.0, hidden=0.0, density="low", seed=18),
            BenchmarkConfig(size="small", noise=0.2, hidden=0.2, density="medium", seed=7),
            BenchmarkConfig(size="medium", noise=0.1, hidden=0.1, density="low", seed=3),
        ]
        budgets = {"small": 6, "medium": 12, "large": 18}
        passed = 0
        for config in configs:
            with self.subTest(config=config):
                instance = generate_network(config)
                model = ObservedRiskScorer(cutoff=8)
                model.fit(instance.observed_graph, instance.attack_sources, instance.protected_assets)
                capsule = CertifiedInterventionPlanner(tau=1.0, cutoff=8).synthesize(
                    instance.observed_graph,
                    instance.attack_sources,
                    instance.protected_assets,
                    instance.required_service_pairs_observed,
                    model,
                    action_budget=budgets[config.size],
                    candidate_actions=sorted(instance.observed_graph.edges),
                )
                passed += int(capsule.service_safety_certificate["passed"])
                source_set = set(instance.attack_sources)
                self.assertFalse(any(source in source_set for source, _destination in capsule.interventions))
                self.assertFalse(
                    any(
                        instance.observed_graph.nodes[source].get("type") == "gateway"
                        for source, _destination in capsule.interventions
                    )
                )
        self.assertGreaterEqual(passed, 1)

    def test_observed_graph_excludes_generator_only_labels(self):
        instance = generate_network(BenchmarkConfig(size="small", noise=0.2, hidden=0.1, seed=7))
        forbidden = {
            "attack_path_benchmark_label",
            "attack_role",
            "hidden_candidate",
            "required",
            "temporal_stability",
            "spurious",
        }
        for _source, _destination, data in instance.observed_graph.edges(data=True):
            self.assertTrue(forbidden.isdisjoint(data))
        self.assertRegex(instance.observed_graph.graph["snapshot_id"], r"^synthetic-observed-[0-9a-f]{20}$")

    def test_uncertainty_has_no_hidden_rate_offset(self):
        visible = generate_network(BenchmarkConfig(size="small", noise=0.0, hidden=0.0, seed=11))
        hidden = generate_network(BenchmarkConfig(size="small", noise=0.0, hidden=0.2, seed=11))
        shared = set(visible.observed_graph.edges) & set(hidden.observed_graph.edges)
        same_appearance = [
            edge
            for edge in shared
            if visible.observed_graph[edge[0]][edge[1]]["appearance_rate"]
            == hidden.observed_graph[edge[0]][edge[1]]["appearance_rate"]
        ]
        self.assertTrue(same_appearance)
        for edge in same_appearance:
            self.assertAlmostEqual(
                visible.observed_graph[edge[0]][edge[1]]["uncertainty"],
                hidden.observed_graph[edge[0]][edge[1]]["uncertainty"],
            )

    def test_generated_instance_and_planner_are_deterministic(self):
        config = BenchmarkConfig(size="small", noise=0.0, hidden=0.0, density="low", seed=18)
        outputs = []
        for _ in range(2):
            instance = generate_network(config)
            model = ObservedRiskScorer(cutoff=8)
            model.fit(instance.observed_graph, instance.attack_sources, instance.protected_assets)
            start = time.perf_counter()
            capsule = CertifiedInterventionPlanner(tau=1.0, cutoff=8).synthesize(
                instance.observed_graph,
                instance.attack_sources,
                instance.protected_assets,
                instance.required_service_pairs_observed,
                model,
                action_budget=6,
                candidate_actions=sorted(instance.observed_graph.edges),
            )
            row = evaluate_capsule(instance, capsule, time.perf_counter() - start)
            self.assertTrue(capsule.service_safety_certificate["passed"])
            self.assertEqual(row["service_pair_preservation_true"], 1.0)
            self.assertEqual(row["actions_from_attack_source"], 0)
            self.assertEqual(row["actions_from_gateway"], 0)
            self.assertEqual(row["source_boundary_threat_service_feasible_observed"], 0.0)
            self.assertEqual(row["source_boundary_admissible_under_budget_observed"], 0.0)
            self.assertEqual(
                row["broker_boundary_admissible_under_budget_observed"],
                row["broker_boundary_threat_service_feasible_observed"]
                * row["broker_boundary_within_budget"],
            )
            self.assertNotIn("source_boundary_spec_feasible_observed", row)
            self.assertNotIn("broker_boundary_spec_feasible_observed", row)
            self.assertEqual(row["serialized_replay_pass"], 1.0)
            self.assertEqual(
                json.loads(row["repair_log_json"]),
                capsule.proposal_transformation["steps"],
            )
            self.assertEqual(
                stable_hash(json.loads(row["repair_log_json"])),
                row["repair_log_sha256"],
            )
            self.assertGreater(row["capsule_json_bytes"], row["essential_witness_json_bytes"])
            outputs.append((capsule.interventions, capsule.service_safety_certificate["bindings"]))
        self.assertEqual(outputs[0], outputs[1])

    def test_baseline_receives_same_valid_metadata_wrapper(self):
        instance = generate_network(BenchmarkConfig(seed=18))
        capsule = greedy_risk(
            instance.observed_graph,
            instance.attack_sources,
            instance.protected_assets,
            instance.required_service_pairs_observed,
            budget=12,
            candidate_actions=sorted(instance.observed_graph.edges),
        )
        certificate = capsule.service_safety_certificate
        self.assertTrue(certificate["rollback_schema_valid"])
        self.assertTrue(certificate["typed_sketch_schema_valid"])
        self.assertTrue(certificate["identifier_omission_valid"])
        self.assertTrue(certificate["policy_equivalent"])
        self.assertTrue(certificate["freshness_valid"])

    def test_exact_oracle_is_exact_on_declared_toy_pool(self):
        graph = _toy_graph()
        candidate_pool = [("source", "left"), ("left", "asset"), ("source", "right"), ("right", "asset")]
        oracle = exact_small_cut_oracle(
            graph,
            ["source"],
            ["asset"],
            [("client", "database")],
            budget=2,
            candidate_pool=candidate_pool,
            candidate_actions=candidate_pool,
        )
        baseline = service_aware_greedy_cut(
            graph,
            ["source"],
            ["asset"],
            [("client", "database")],
            budget=2,
            candidate_actions=sorted(graph.edges),
        )
        self.assertEqual(oracle.method, "ExactBoundedCandidateOracle")
        self.assertTrue(oracle.service_safety_certificate["passed"])
        self.assertLessEqual(len(oracle.interventions), len(baseline.interventions))

    def test_statistics_pair_by_explicit_keys_not_row_position(self):
        rows = []
        for method, order in (("CIN-CertifiedGreedy", [1, 0]), ("Baseline", [0, 1])):
            for seed in order:
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "size": "small",
                        "noise": 0.0,
                        "hidden": 0.0,
                        "density": "low",
                        "action_budget": 12,
                        "score": 0.8 + 0.1 * seed if method == "CIN-CertifiedGreedy" else 0.5 + 0.1 * seed,
                    }
                )
        merged = _paired_cluster_data(pd.DataFrame(rows), "Baseline", "score")
        self.assertTrue((merged.difference.round(8) == 0.3).all())

    def test_generated_latex_reports_joint_controls_and_strict_missing_values(self):
        common = {
            "size": "small",
            "noise": 0.3,
            "action_budget": 6,
            "attack_path_reduction_true": 0.7,
            "attack_pair_isolation_true": 0.6,
            "service_pair_preservation_true": 1.0,
            "threat_present_true": 1.0,
            "action_cost": 3.0,
            "policy_size": 3,
            "runtime_s": 0.1,
            "nodes": 64,
            "capsule_json_bytes": 2048,
            "essential_witness_json_bytes": 1024,
            "verification_bundle_json_bytes": 4096,
            "serialized_replay_verifier_s": 0.002,
            "serialized_replay_pass": 1.0,
            "action_fraction_from_attack_source": 0.0,
            "action_fraction_from_broker": 0.5,
            "source_boundary_threat_service_feasible_observed": 0.0,
            "source_boundary_admissible_under_budget_observed": 0.0,
            "broker_boundary_threat_service_feasible_observed": 1.0,
        }
        rows = []
        for method in ("CIN-CertifiedGreedy", "ServiceAwareGreedyCut"):
            for seed, hidden, density, within, certificate, true_feasible in (
                (0, 0.2, "low", 1.0, 1.0 if method == "CIN-CertifiedGreedy" else 0.0, 1.0),
                (1, 0.1, "high", 0.0, 0.0, 0.0),
            ):
                row = {
                    **common,
                    "method": method,
                    "seed": seed,
                    "hidden": hidden,
                    "density": density,
                    "certificate_pass": certificate,
                    "true_feasible": true_feasible,
                    "broker_boundary_within_budget": within,
                    "broker_boundary_admissible_under_budget_observed": within,
                }
                rows.append(row)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            raw = pd.DataFrame(rows)
            raw.to_csv(path / "raw_results.csv", index=False)
            table = write_paper_tables(str(path))
            canonical = table[table.method == "CIN-CertifiedGreedy"].iloc[0]
            self.assertEqual(canonical.broker_boundary_within_budget_rate, 0.5)
            self.assertEqual(canonical.broker_boundary_threat_service_feasible_rate, 1.0)
            self.assertEqual(canonical.broker_boundary_admissible_under_budget_rate, 0.5)

            comparison_rows = []
            for metric in (
                "attack_pair_isolation_true",
                "attack_path_reduction_true",
                "service_pair_preservation_true",
            ):
                comparison_rows.append(
                    {
                        "baseline": "ServiceAwareGreedyCut",
                        "metric": metric,
                        "mean_difference": 0.1,
                        "ci_low": 0.0,
                        "ci_high": 0.2,
                        "p_value_holm": 0.5,
                    }
                )
            pd.DataFrame(comparison_rows).to_csv(path / "statistical_tests.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "ablation": "full",
                        "attack_pair_isolation_true": 0.6,
                        "service_pair_preservation_true": 1.0,
                        "certificate_pass": 1.0,
                        "true_feasible": 1.0,
                    }
                ]
            ).to_csv(path / "ablation_results.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "method": method,
                        "residual_attack_pairs_observed": residual,
                        "optimality_gap_residual_attack_pairs_observed": gap,
                        "oracle_candidate_count": 10,
                    }
                    for method, residual, gap in (
                        ("CIN-CertifiedGreedy", 1.0, 0.5),
                        ("ExactBoundedCandidateOracle", 0.5, 0.0),
                        ("ServiceAwareGreedyCut", 0.8, 0.3),
                    )
                ]
            ).to_csv(path / "oracle_diagnostics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "size": "small",
                        "budget_regime": "primary",
                        "method": method,
                        "action_budget": 6,
                        "attack_pair_isolation_true": 0.6,
                        "service_pair_preservation_true": 1.0,
                        "true_feasible": 0.5,
                        "certificate_pass": 0.5,
                    }
                    for method in ("CIN-CertifiedGreedy", "ServiceAwareGreedyCut")
                ]
            ).to_csv(path / "budget_sensitivity.csv", index=False)

            latex_dir = write_latex_fragments(str(path))
            expected = {
                "result_macros.tex",
                "robustness_results.tex",
                "boundary_controls.tex",
                "selectivity_by_hidden.tex",
                "main_selectivity_compact.tex",
                "main_scalability_compact.tex",
                "main_oracle_compact.tex",
                "budget_sensitivity.tex",
            }
            self.assertTrue(expected.issubset({item.name for item in latex_dir.iterdir()}))
            rendered = "\n".join(
                item.read_text(encoding="utf-8") for item in latex_dir.glob("*.tex")
            )
            self.assertNotIn("nan", rendered.lower())
            self.assertNotIn("Cert. cov.", rendered)
            self.assertNotIn("Feasible prec.", rendered)
            self.assertIn("jointly admissible", rendered)
            self.assertIn("Broker threat+service", rendered)
            self.assertIn("Joint admissibility", rendered)
            self.assertIn("Admitted hidden feas.", rendered)
            self.assertIn("--", (latex_dir / "selectivity_by_hidden.tex").read_text(encoding="utf-8"))


    def test_one_seed_statistics_serialize_undefined_effect_as_json_null(self):
        metrics = [
            "attack_path_reduction_true",
            "attack_pair_isolation_true",
            "service_pair_preservation_true",
            "true_feasible",
            "certificate_pass",
            "action_cost",
            "runtime_s",
        ]
        rows = []
        for method, value in (("CIN-CertifiedGreedy", 0.8), ("Baseline", 0.6)):
            row = {
                "method": method,
                "seed": 0,
                "size": "small",
                "noise": 0.0,
                "hidden": 0.0,
                "density": "low",
                "action_budget": 12,
            }
            row.update({metric: value for metric in metrics})
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame(rows).to_csv(path / "raw_results.csv", index=False)
            write_statistical_report(str(path))
            payload = json.loads((path / "statistical_report.json").read_text(encoding="utf-8"))
            self.assertIsNone(payload["tests"][0]["paired_effect"])


if __name__ == "__main__":
    unittest.main()

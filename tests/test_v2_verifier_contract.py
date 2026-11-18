from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import networkx as nx
import pytest

from cin.canonical_json import (
    CanonicalJSONError,
    canonical_json_bytes,
    strict_json_loads,
)
from cin.capsule import (
    build_capsule,
    build_identifier_omitting_typed_sketch,
    build_invariant,
    compile_policies,
)
from cin.provenance import AuthorizationPolicy, authorize_interventions
from cin.repair import SubsetMinimalRepair, identity_transformation
from cin.simulator import PathSearchScope, enumerate_attack_paths
from cin.verifier import (
    _parse_utc_timestamp,
    canonical_action_set,
    graph_from_dict,
    graph_to_dict,
    make_verification_bundle,
    stable_hash,
    validate_identifier_omitting_typed_sketch,
    verification_bundle_hash,
    verify_capsule_bundle,
    verify_exact_proposal,
)


SNAPSHOT = "2026-01-01T00:00:00Z"
ISSUED = "2026-01-01T00:01:00Z"
EXPIRES = "2026-01-01T00:11:00Z"


def _graph() -> nx.DiGraph:
    graph = nx.DiGraph(
        snapshot_id="v2-contract-snapshot",
        topology_version=3,
        snapshot_time=SNAPSHOT,
    )
    for node, kind in (
        ("source", "source"),
        ("middle", "service"),
        ("asset", "asset"),
        ("client", "client"),
        ("database", "database"),
    ):
        graph.add_node(node, type=kind, tier=1, dept="d0")
    graph.add_edge("source", "middle", kind="comm", action_cost=1.0)
    graph.add_edge("middle", "asset", kind="comm", action_cost=1.0)
    graph.add_edge("client", "database", kind="required", action_cost=1.0)
    return graph


def _freshness(**updates) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "cin.freshness.v2",
        "snapshot_id": "v2-contract-snapshot",
        "topology_version": 3,
        "snapshot_time": SNAPSHOT,
        "issued_at": ISSUED,
        "expires_at": EXPIRES,
        "max_snapshot_age_at_issue_seconds": 60,
        "max_capsule_age_seconds": 600,
    }
    value.update(updates)
    return value


def _artifacts(graph: nx.DiGraph, actions):
    abstract, kubernetes, ebpf, sdn = compile_policies(graph, actions, "V2Test")
    policy_bundle = {
        "abstract": abstract,
        "kubernetes_intent": kubernetes,
        "ebpf_pseudo": ebpf,
        "sdn_pseudo": sdn,
    }
    rollback = {
        "schema": "cin.rollback.v1",
        "metric": "service_endpoint_reachability_fraction",
        "operator": "<",
        "threshold": 1.0,
        "window_seconds": 60,
        "consecutive_windows": 1,
        "action": "restore_previous_policy",
        "restore_policy_hash": stable_hash(
            {"schema": "cin.policy.v1", "state": "pre-capsule"}
        ),
    }
    sketch = build_identifier_omitting_typed_sketch(graph, actions)
    return policy_bundle, rollback, sketch


def _proposal_record(graph: nx.DiGraph | None = None):
    graph = graph or _graph()
    repair = SubsetMinimalRepair.apply(
        graph,
        [["middle", "asset"], ["source", "middle"]],
        ["source"],
        ["asset"],
        [["client", "database"]],
        tau=1.0,
    )
    transformation = repair.to_dict()
    raw = transformation["raw_actions"]
    repaired = transformation["repaired_actions"]
    bindings = {
        "raw_interventions_sha256": repair.raw_actions_sha256,
        "repaired_interventions_sha256": repair.repaired_actions_sha256,
        "repair_log_sha256": repair.repair_log_sha256,
        "transformation_version": repair.transformation_version,
    }
    return raw, repaired, transformation, bindings


def _direct_result(
    *, freshness=None, candidates=None, verification_time=ISSUED, path_scope=None
):
    graph = _graph()
    _raw, repaired, _transformation, bindings = _proposal_record(graph)
    candidate_actions = candidates or [
        ["middle", "asset"],
        ["source", "middle"],
    ]
    policy_bundle, rollback, sketch = _artifacts(graph, repaired)
    return verify_exact_proposal(
        graph,
        repaired,
        ["source"],
        ["asset"],
        [["client", "database"]],
        rollback,
        sketch,
        candidate_actions=candidate_actions,
        proposal_bindings=bindings,
        verification_time=verification_time,
        action_budget=2,
        path_scope=path_scope if path_scope is not None else PathSearchScope(),
        policy_bundle=policy_bundle,
        freshness=freshness or _freshness(),
    )


def _bundle():
    graph = _graph()
    raw, repaired, transformation, bindings = _proposal_record(graph)
    candidates = [["middle", "asset"], ["source", "middle"]]
    policy_bundle, rollback, sketch = _artifacts(graph, repaired)
    certificate = verify_exact_proposal(
        graph,
        repaired,
        ["source"],
        ["asset"],
        [["client", "database"]],
        rollback,
        sketch,
        candidate_actions=candidates,
        proposal_bindings=bindings,
        verification_time=ISSUED,
        action_budget=2,
        path_scope=PathSearchScope(),
        policy_bundle=policy_bundle,
        freshness=_freshness(),
    ).to_dict()
    canonical_candidates = [list(edge) for edge in canonical_action_set(candidates)]
    capsule = {
        "schema": "cin.defense-capsule.v2",
        "capsule_id": (
            "cin-"
            + stable_hash(
                {"method": "V2Test", "bindings": certificate["bindings"]}
            )[:20]
        ),
        "method": "V2Test",
        "attack_reachability_invariant": build_invariant(
            graph,
            enumerate_attack_paths(graph, ["source"], ["asset"]),
            [tuple(edge) for edge in repaired],
        ),
        "interventions": repaired,
        "raw_proposal": raw,
        "repaired_proposal": repaired,
        "proposal_transformation": transformation,
        "candidate_actions": canonical_candidates,
        "candidate_action_capability": {
            "schema": "cin.candidate-action-capability.v1",
            "enforcement_domain": "directed-edge-denial-on-bound-observed-graph",
            "candidate_actions_sha256": stable_hash(canonical_candidates),
            "candidate_action_count": len(canonical_candidates),
        },
        "action_budget": 2,
        "freshness": _freshness(),
        "abstract_policy_json": policy_bundle["abstract"],
        "kubernetes_network_policy_yaml": policy_bundle["kubernetes_intent"],
        "ebpf_xdp_pseudo_policy": policy_bundle["ebpf_pseudo"],
        "sdn_flow_rule_pseudo_policy": policy_bundle["sdn_pseudo"],
        "service_safety_certificate": certificate,
        "rollback_condition": rollback,
        "identifier_omitting_typed_sketch": sketch,
    }
    return make_verification_bundle(
        graph,
        capsule,
        ["source"],
        ["asset"],
        [["client", "database"]],
        candidate_actions=candidates,
        tau=1.0,
        action_budget=2,
        path_scope=PathSearchScope(),
        freshness=_freshness(),
    )


def _single_character_endpoint_bundle(*, candidate_actions=None):
    graph = nx.DiGraph(
        snapshot_id="single-character-endpoints",
        topology_version=1,
        snapshot_time=SNAPSHOT,
    )
    for node, kind in (
        ("s", "source"),
        ("m", "service"),
        ("a", "asset"),
        ("c", "client"),
        ("d", "database"),
    ):
        graph.add_node(node, type=kind, tier=1, dept="d0")
    graph.add_edge("s", "m", kind="comm", action_cost=1.0)
    graph.add_edge("m", "a", kind="comm", action_cost=1.0)
    graph.add_edge("c", "d", kind="required", action_cost=1.0)
    candidates = candidate_actions or [["m", "a"], ["s", "m"]]
    capsule = build_capsule(
        method="SingleCharacterEndpointTest",
        graph=graph,
        proposal_transformation=identity_transformation([["m", "a"]]),
        candidate_actions=candidates,
        attack_sources=["s"],
        protected_assets=["a"],
        required_service_pairs=[["c", "d"]],
        action_budget=1,
        issued_at=ISSUED,
        max_snapshot_age_at_issue_seconds=60,
        max_capsule_age_seconds=600,
    )
    return make_verification_bundle(
        graph,
        capsule.to_dict(),
        ["s"],
        ["a"],
        [["c", "d"]],
        candidate_actions=capsule.candidate_actions,
        tau=1.0,
        action_budget=1,
        path_scope=PathSearchScope(),
        freshness=capsule.freshness,
    )


def _built_bundle(
    graph: nx.DiGraph,
    *,
    actions,
    candidate_actions,
    attack_sources,
    protected_assets,
    required_service_pairs,
    method: str,
):
    action_budget = len(actions)
    capsule = build_capsule(
        method=method,
        graph=graph,
        proposal_transformation=identity_transformation(actions),
        candidate_actions=candidate_actions,
        attack_sources=attack_sources,
        protected_assets=protected_assets,
        required_service_pairs=required_service_pairs,
        action_budget=action_budget,
        issued_at=ISSUED,
        max_snapshot_age_at_issue_seconds=60,
        max_capsule_age_seconds=600,
    )
    return make_verification_bundle(
        graph,
        capsule.to_dict(),
        attack_sources,
        protected_assets,
        required_service_pairs,
        candidate_actions=capsule.candidate_actions,
        tau=1.0,
        action_budget=action_budget,
        path_scope=PathSearchScope(),
        freshness=capsule.freshness,
    )


def test_strict_canonical_json_rejects_all_known_collision_classes() -> None:
    class StringCollision:
        def __str__(self):
            return "token"

    for invalid in (
        {1: "integer", "1": "string"},
        float("nan"),
        float("inf"),
        2**53,
        {"a", "b"},
        StringCollision(),
    ):
        with pytest.raises(CanonicalJSONError):
            canonical_json_bytes(invalid)
    with pytest.raises(CanonicalJSONError, match="duplicate JSON object key"):
        strict_json_loads('{"a":1,"a":2}')
    with pytest.raises(CanonicalJSONError, match="non-finite"):
        strict_json_loads('{"value":NaN}')


def test_stable_hash_remains_byte_compatible_for_valid_legacy_json() -> None:
    value = {"b": [1, 2.5, True, None], "a": "text"}
    legacy = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    assert canonical_json_bytes(value) == legacy
    assert stable_hash(value) == hashlib.sha256(legacy).hexdigest()


@pytest.mark.parametrize(
    "invalid",
    [
        "ab",
        {"a": 0, "b": 1},
        [["a"]],
        [["a", "b", "c"]],
        [[1, "b"]],
        [["a", "b"], ["a", "b"]],
    ],
)
def test_canonical_action_set_rejects_coercion_duplicates_and_malformed_edges(
    invalid,
) -> None:
    with pytest.raises(ValueError):
        canonical_action_set(invalid)


def test_action_permutations_have_one_exact_canonical_form() -> None:
    left = canonical_action_set([["b", "c"], ["a", "b"]])
    right = canonical_action_set([["a", "b"], ["b", "c"]])
    assert left == right == (("a", "b"), ("b", "c"))


def test_authorization_does_not_coerce_or_silently_deduplicate_actions() -> None:
    graph = _graph()
    policy = AuthorizationPolicy(
        "cin.authorization-policy.v1",
        "strict-actions",
        2,
        ("source", "service"),
        ("service", "asset"),
    )
    for invalid in (
        ["ab"],
        [{"source": 0, "middle": 1}],
        [["source", "middle"], ["source", "middle"]],
    ):
        with pytest.raises(ValueError):
            authorize_interventions(graph, invalid, policy)


def test_graph_serialization_rejects_non_string_node_ids_without_collision() -> None:
    graph = nx.DiGraph()
    graph.add_node(1)
    with pytest.raises(ValueError, match="identifiers must be strings"):
        graph_to_dict(graph)
    with pytest.raises(ValueError, match="identifiers must be strings"):
        graph_from_dict(
            {"directed": True, "graph_attributes": {}, "nodes": [{"id": 1}], "edges": []}
        )


def test_exact_verifier_requires_explicit_time_and_candidate_set() -> None:
    graph = _graph()
    _raw, repaired, _transformation, bindings = _proposal_record()
    policy_bundle, rollback, sketch = _artifacts(graph, repaired)
    with pytest.raises(TypeError):
        verify_exact_proposal(
            graph,
            repaired,
            ["source"],
            ["asset"],
            [["client", "database"]],
            rollback,
            sketch,
            candidate_actions=[["middle", "asset"]],
            proposal_bindings=bindings,
            action_budget=2,
            policy_bundle=policy_bundle,
            freshness=_freshness(),
        )


def test_exact_verifier_rejects_malformed_proposal_binding_hashes() -> None:
    graph = _graph()
    _raw, repaired, _transformation, bindings = _proposal_record()
    policy_bundle, rollback, sketch = _artifacts(graph, repaired)
    bindings["repair_log_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="repair_log_sha256"):
        verify_exact_proposal(
            graph,
            repaired,
            ["source"],
            ["asset"],
            [["client", "database"]],
            rollback,
            sketch,
            candidate_actions=[["middle", "asset"]],
            proposal_bindings=bindings,
            verification_time=ISSUED,
            action_budget=2,
            policy_bundle=policy_bundle,
            freshness=_freshness(),
        )


def test_exact_verifier_never_repairs_redundant_or_over_budget_input() -> None:
    graph = _graph()
    raw = [["middle", "asset"], ["source", "middle"]]
    policy_bundle, rollback, sketch = _artifacts(graph, raw)
    identity = identity_transformation(raw)
    bindings = {
        "raw_interventions_sha256": identity.raw_actions_sha256,
        "repaired_interventions_sha256": identity.repaired_actions_sha256,
        "repair_log_sha256": identity.repair_log_sha256,
        "transformation_version": identity.transformation_version,
    }
    redundant = verify_exact_proposal(
        graph,
        raw,
        ["source"],
        ["asset"],
        [["client", "database"]],
        rollback,
        sketch,
        candidate_actions=[["middle", "asset"], ["source", "middle"]],
        proposal_bindings=bindings,
        verification_time=ISSUED,
        action_budget=2,
        path_scope=PathSearchScope(),
        policy_bundle=policy_bundle,
        freshness=_freshness(),
    )
    assert redundant.passed is False
    assert redundant.action_count == 2
    assert redundant.minimality_status == "not_subset_minimal"
    assert redundant.bindings["interventions_sha256"] == stable_hash(raw)

    necessary = [["middle", "asset"]]
    policy_bundle, rollback, sketch = _artifacts(graph, necessary)
    identity = identity_transformation(necessary)
    over_budget = verify_exact_proposal(
        graph,
        necessary,
        ["source"],
        ["asset"],
        [["client", "database"]],
        rollback,
        sketch,
        candidate_actions=[["middle", "asset"]],
        proposal_bindings={
            "raw_interventions_sha256": identity.raw_actions_sha256,
            "repaired_interventions_sha256": identity.repaired_actions_sha256,
            "repair_log_sha256": identity.repair_log_sha256,
            "transformation_version": identity.transformation_version,
        },
        verification_time=ISSUED,
        action_budget=0,
        path_scope=PathSearchScope(),
        policy_bundle=policy_bundle,
        freshness=_freshness(),
    )
    assert over_budget.passed is False
    assert over_budget.action_count == 1
    assert over_budget.budget_respected is False


def test_exact_verifier_rejects_action_outside_bound_candidates() -> None:
    result = _direct_result(candidates=[["source", "middle"]])
    assert result.passed is False
    assert result.candidate_actions_valid is False
    assert any("outside the bound candidate" in item for item in result.messages)


def test_exact_verifier_rejects_duplicate_or_absent_candidates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _direct_result(
            candidates=[
                ["middle", "asset"],
                ["middle", "asset"],
            ]
        )
    absent = _direct_result(
        candidates=[["middle", "asset"], ["asset", "source"]]
    )
    assert absent.passed is False
    assert absent.candidate_actions_valid is False


def test_v2_time_boundaries_and_snapshot_age_are_exact() -> None:
    assert _direct_result(verification_time=ISSUED).passed is True
    assert _direct_result(verification_time="2026-01-01T00:10:59.999999Z").passed is True
    at_expiry = _direct_result(verification_time=EXPIRES)
    assert at_expiry.passed is False
    assert at_expiry.freshness_valid is False
    before_issue = _direct_result(verification_time=SNAPSHOT)
    assert before_issue.passed is False
    assert before_issue.freshness_valid is False
    too_old = _direct_result(
        freshness=_freshness(max_snapshot_age_at_issue_seconds=59)
    )
    assert too_old.passed is False
    assert any("too old" in item for item in too_old.messages)


def test_v2_expiry_must_equal_issue_plus_declared_capsule_age() -> None:
    malformed = _direct_result(
        freshness=_freshness(expires_at="2026-01-01T00:12:00Z")
    )
    assert malformed.passed is False
    assert any("does not match issued_at plus" in item for item in malformed.messages)


@pytest.mark.parametrize(
    "timestamp",
    [
        "0001-01-01T00:00:00+23:59",
        "9999-12-31T23:59:59-23:59",
    ],
)
def test_timestamp_normalization_overflow_is_rejected(timestamp: str) -> None:
    assert _parse_utc_timestamp(timestamp) is None


@pytest.mark.parametrize(
    "path_scope",
    [
        PathSearchScope(cutoff_edges=True),
        PathSearchScope(max_paths_per_pair=True),
        PathSearchScope(max_expansions_per_pair=True),
    ],
)
def test_v2_path_search_scope_rejects_boolean_integer_bounds(
    path_scope: PathSearchScope,
) -> None:
    result = _direct_result(path_scope=path_scope)

    assert result.passed is False
    assert result.scope_declared is False
    assert "path-search scope is missing or invalid" in result.messages


def test_valid_v2_bundle_replays_and_binds_candidates_and_proposals() -> None:
    bundle = _bundle()
    root = verification_bundle_hash(bundle)
    result = verify_capsule_bundle(
        bundle, expected_bundle_sha256=root, verification_time=ISSUED
    )
    assert result.passed is True
    assert result.schema == "cin.certificate.v2"
    assert result.checker_version == "cin-verifier-1.7"
    assert result.candidate_actions_valid is True
    assert result.bindings["candidate_action_count"] == 2
    assert result.embedded_issuance_certificate_valid is True


@pytest.mark.parametrize(
    ("field", "scalar"),
    [("attack_sources", "s"), ("protected_assets", "a")],
)
def test_v2_replay_rejects_scalar_threat_endpoints_that_mimic_one_item_arrays(
    field: str, scalar: str
) -> None:
    bundle = _single_character_endpoint_bundle()
    bundle["threat_model"][field] = scalar
    with pytest.raises(ValueError, match="must be arrays"):
        verify_capsule_bundle(
            bundle,
            expected_bundle_sha256=verification_bundle_hash(bundle),
            verification_time=ISSUED,
        )


@pytest.mark.parametrize(
    "verified_at",
    ["not-a-time", "2099-01-01T00:00:00Z", 7],
)
def test_v2_replay_rejects_invalid_or_nonissuance_embedded_verified_at(
    verified_at: object,
) -> None:
    bundle = _bundle()
    bundle["capsule"]["service_safety_certificate"]["verified_at"] = verified_at
    result = verify_capsule_bundle(
        bundle,
        expected_bundle_sha256=verification_bundle_hash(bundle),
        verification_time="2026-01-01T00:02:00Z",
    )
    assert result.passed is False
    assert result.embedded_issuance_certificate_valid is False
    assert any("verified_at" in message for message in result.messages)


def test_v2_external_replay_time_may_differ_from_bound_issuance_time() -> None:
    bundle = _bundle()
    result = verify_capsule_bundle(
        bundle,
        expected_bundle_sha256=verification_bundle_hash(bundle),
        verification_time="2026-01-01T00:02:00Z",
    )
    assert result.passed is True
    assert result.verified_at == "2026-01-01T00:02:00Z"
    assert (
        bundle["capsule"]["service_safety_certificate"]["verified_at"]
        == ISSUED
    )


def test_v2_replay_rejects_candidate_capability_and_proposal_copy_tampering() -> None:
    bundle = _bundle()
    capability_tamper = deepcopy(bundle)
    capability_tamper["capsule"]["candidate_action_capability"][
        "candidate_action_count"
    ] = 99
    result = verify_capsule_bundle(
        capability_tamper,
        expected_bundle_sha256=verification_bundle_hash(capability_tamper),
        verification_time=ISSUED,
    )
    assert result.passed is False
    assert any("candidate_action_capability" in item for item in result.messages)

    repaired_tamper = deepcopy(bundle)
    repaired_tamper["capsule"]["repaired_proposal"] = [["source", "middle"]]
    with pytest.raises(ValueError, match="repaired_proposal differs"):
        verify_capsule_bundle(
            repaired_tamper,
            expected_bundle_sha256=verification_bundle_hash(repaired_tamper),
            verification_time=ISSUED,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda bundle: bundle["capsule"]["candidate_action_capability"].__setitem__(
            "candidate_action_count", True
        ),
        lambda bundle: bundle["capsule"]["service_safety_certificate"][
            "bindings"
        ].__setitem__("candidate_action_count", True),
    ],
)
def test_v2_replay_rejects_boolean_candidate_action_counts(mutation) -> None:
    bundle = _single_character_endpoint_bundle(
        candidate_actions=[["m", "a"]]
    )
    mutation(bundle)

    result = verify_capsule_bundle(
        bundle,
        expected_bundle_sha256=verification_bundle_hash(bundle),
        verification_time=ISSUED,
    )

    assert result.passed is False
    assert any("candidate_action_count" in message for message in result.messages)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda bundle: bundle["capsule"].__setitem__("action_budget", True),
        lambda bundle: bundle["capsule"]["service_safety_certificate"].__setitem__(
            "candidate_action_count", True
        ),
        lambda bundle: bundle["capsule"]["service_safety_certificate"][
            "witnesses"
        ]["surviving_service_paths"].__setitem__("required_pair_count", True),
    ],
)
def test_v2_exact_copies_reject_boolean_integer_aliases(mutation) -> None:
    bundle = _single_character_endpoint_bundle(
        candidate_actions=[["m", "a"]]
    )
    mutation(bundle)

    result = verify_capsule_bundle(
        bundle,
        expected_bundle_sha256=verification_bundle_hash(bundle),
        verification_time=ISSUED,
    )

    assert result.passed is False
    assert (
        result.embedded_issuance_certificate_valid is False
        or result.capsule_budget_matches is False
        or result.submitted_witnesses_valid is False
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("version", True), ("raw_identifier_count", False)],
)
def test_v2_typed_sketch_rejects_boolean_integer_metadata(
    field: str, value: bool
) -> None:
    graph = _graph()
    sketch = build_identifier_omitting_typed_sketch(
        graph, [["middle", "asset"]]
    )
    sketch[field] = value

    schema_valid, _, messages = validate_identifier_omitting_typed_sketch(
        sketch, graph, strict_integer_types=True
    )

    assert schema_valid is False
    assert any("typed-sketch" in message for message in messages)


def test_v2_typed_sketch_ontology_membership_is_type_sensitive() -> None:
    graph = _graph()
    sketch = build_identifier_omitting_typed_sketch(
        graph, [["middle", "asset"]]
    )
    pattern = sketch["typed_patterns"][0]
    pattern["src_tier"] = True
    sketch["pattern_hashes"][0] = stable_hash(pattern)

    schema_valid, _, messages = validate_identifier_omitting_typed_sketch(
        sketch, graph, strict_integer_types=True
    )

    assert schema_valid is False
    assert any("outside the graph ontology" in message for message in messages)


def test_v2_typed_sketch_accepts_structured_json_ontology_tokens() -> None:
    graph = _graph()
    graph.nodes["middle"]["type"] = {"role": "service"}
    sketch = build_identifier_omitting_typed_sketch(
        graph, [["middle", "asset"]]
    )

    schema_valid, omission_valid, messages = (
        validate_identifier_omitting_typed_sketch(
            sketch, graph, strict_integer_types=True
        )
    )

    assert schema_valid is True, messages
    assert omission_valid is True, messages


def test_v2_cross_department_tokens_use_json_type_identity() -> None:
    graph = _graph()
    graph.nodes["middle"]["dept"] = 1
    graph.nodes["asset"]["dept"] = True
    bundle = _built_bundle(
        graph,
        actions=[["middle", "asset"]],
        candidate_actions=[["middle", "asset"]],
        attack_sources=["source"],
        protected_assets=["asset"],
        required_service_pairs=[["client", "database"]],
        method="TypedDepartmentTokens",
    )

    assert bundle["capsule"]["attack_reachability_invariant"][
        "cut_edge_patterns"
    ][0]["cross_dept"] is True
    result = verify_capsule_bundle(
        bundle,
        expected_bundle_sha256=verification_bundle_hash(bundle),
        verification_time=ISSUED,
    )
    assert result.passed is True, result.messages


@pytest.mark.parametrize("invalid_method", ["", " V2Test", "V2Test ", 7])
def test_v2_builder_rejects_invalid_method_metadata(invalid_method: object) -> None:
    graph = _graph()
    with pytest.raises(ValueError, match="method must be"):
        build_capsule(
            method=invalid_method,
            graph=graph,
            proposal_transformation=identity_transformation(
                [["middle", "asset"]]
            ),
            candidate_actions=[["middle", "asset"], ["source", "middle"]],
            attack_sources=["source"],
            protected_assets=["asset"],
            required_service_pairs=[["client", "database"]],
            action_budget=1,
            issued_at=ISSUED,
            max_snapshot_age_at_issue_seconds=60,
            max_capsule_age_seconds=600,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda bundle: bundle["capsule"].__setitem__("capsule_id", 7),
            "capsule_id differs",
        ),
        (
            lambda bundle: bundle["capsule"].__setitem__("capsule_id", "cin-fabricated"),
            "capsule_id differs",
        ),
        (
            lambda bundle: bundle["capsule"].__setitem__("method", 7),
            "capsule method must be",
        ),
        (
            lambda bundle: bundle["capsule"].__setitem__("method", "OtherMethod"),
            "capsule method differs",
        ),
        (
            lambda bundle: bundle["capsule"][
                "attack_reachability_invariant"
            ].__setitem__("schema", "fabricated.schema"),
            "attack_reachability_invariant differs",
        ),
        (
            lambda bundle: bundle["capsule"][
                "attack_reachability_invariant"
            ].__setitem__("claim_scope", "global_real_network"),
            "attack_reachability_invariant differs",
        ),
        (
            lambda bundle: bundle["capsule"]["attack_reachability_invariant"][
                "typed_path_patterns"
            ][0]["node_types"].__setitem__(0, "fabricated-type"),
            "attack_reachability_invariant differs",
        ),
        (
            lambda bundle: bundle["capsule"]["attack_reachability_invariant"][
                "typed_path_patterns"
            ][0]["tier_sequence"].__setitem__(0, True),
            "attack_reachability_invariant differs",
        ),
        (
            lambda bundle: bundle["capsule"]["attack_reachability_invariant"][
                "cut_edge_patterns"
            ][0].__setitem__("relation", "fabricated-relation"),
            "attack_reachability_invariant differs",
        ),
    ],
)
def test_v2_replay_rejects_fabricated_capsule_metadata_and_invariant(
    mutation, message: str
) -> None:
    bundle = _bundle()
    mutation(bundle)
    result = verify_capsule_bundle(
        bundle,
        expected_bundle_sha256=verification_bundle_hash(bundle),
        verification_time="2026-01-01T00:02:00Z",
    )
    assert result.passed is False
    assert any(message in item for item in result.messages)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda policy: policy.__setitem__("schema", "fabricated.policy"),
            "abstract policy schema",
        ),
        (
            lambda policy: policy.__setitem__("method", 7),
            "abstract policy method",
        ),
        (
            lambda policy: policy.__setitem__("semantics", 7),
            "abstract policy semantics",
        ),
        (
            lambda policy: policy["manifest"].__setitem__("rule_count", 999),
            "manifest rule_count",
        ),
        (
            lambda policy: policy["rules"][0].__setitem__("id", 7),
            "abstract policy rule ids",
        ),
        (
            lambda policy: policy["rules"][0].__setitem__("relation", 7),
            "abstract policy rule ids",
        ),
    ],
)
def test_v2_replay_rejects_fabricated_abstract_policy_fields(
    mutation, message: str
) -> None:
    bundle = _bundle()
    mutation(bundle["capsule"]["abstract_policy_json"])
    result = verify_capsule_bundle(
        bundle,
        expected_bundle_sha256=verification_bundle_hash(bundle),
        verification_time="2026-01-01T00:02:00Z",
    )
    assert result.passed is False
    assert result.policy_equivalent is False
    assert any(message in item for item in result.messages)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.__setitem__("schema", "unknown"), "schema/version"),
        (lambda record: record.__setitem__("algorithm", "unknown"), "unsupported"),
        (lambda record: record.__setitem__("status", "fabricated"), "log or binding"),
        (
            lambda record: record.__setitem__("raw_actions_sha256", "0" * 64),
            "log or binding",
        ),
        (
            lambda record: record.__setitem__("repair_log_sha256", "0" * 64),
            "log or binding",
        ),
        (
            lambda record: record["steps"][0].__setitem__("sequence", False),
            "log or binding",
        ),
        (
            lambda record: record["steps"][0].__setitem__("pass", False),
            "log or binding",
        ),
        (
            lambda record: record.__setitem__(
                "raw_actions", [["client", "database"]]
            ),
            "deterministic transformation",
        ),
        (lambda record: record.__setitem__("unexpected", True), "field set"),
        (
            lambda record: record["steps"][0].__setitem__("unexpected", True),
            "log or binding",
        ),
    ],
)
def test_v2_replay_rejects_malformed_or_fabricated_transformations(
    mutation, message: str
) -> None:
    bundle = _bundle()
    mutation(bundle["capsule"]["proposal_transformation"])
    with pytest.raises(ValueError, match=message):
        verify_capsule_bundle(
            bundle,
            expected_bundle_sha256=verification_bundle_hash(bundle),
            verification_time=ISSUED,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda bundle: bundle.__setitem__("unexpected", True),
        lambda bundle: bundle["threat_model"].__setitem__("unexpected", True),
        lambda bundle: bundle["constraints"].__setitem__("unexpected", True),
        lambda bundle: bundle["capsule"].__setitem__("unexpected", True),
        lambda bundle: bundle["graph"].__setitem__("unexpected", True),
        lambda bundle: bundle["graph"]["nodes"][0].__setitem__(
            "unexpected", True
        ),
        lambda bundle: bundle["constraints"].__setitem__("path_search_scope", {}),
        lambda bundle: bundle["capsule"]["attack_reachability_invariant"][
            "typed_path_patterns"
        ][0].__setitem__("unexpected", True),
        lambda bundle: bundle["capsule"]["abstract_policy_json"]["rules"][
            0
        ].__setitem__("unexpected", True),
        lambda bundle: bundle["capsule"]["identifier_omitting_typed_sketch"][
            "typed_patterns"
        ][0].__setitem__("unexpected", True),
        lambda bundle: bundle["capsule"]["service_safety_certificate"][
            "bindings"
        ]["policy_artifact_sha256"].__setitem__("unexpected", "0" * 64),
    ],
)
def test_v2_replay_rejects_unknown_or_missing_schema_fields(mutation) -> None:
    bundle = _bundle()
    mutation(bundle)
    result = verify_capsule_bundle(
        bundle,
        expected_bundle_sha256=verification_bundle_hash(bundle),
        verification_time=ISSUED,
    )
    assert result.passed is False
    assert any("field set is invalid" in item for item in result.messages)




def _run_partial_node_checker(
    bundle: dict[str, object], tmp_path: Path, *, name: str
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    raw = canonical_json_bytes(bundle)
    path = tmp_path / name
    path.write_bytes(raw)
    root = hashlib.sha256(raw).hexdigest()
    checker = Path(__file__).resolve().parents[1] / "independent_checker" / "verify_bundle.mjs"
    completed = subprocess.run(
        [
            "node",
            str(checker),
            "--bundle",
            str(path),
            "--expected-root",
            root,
            "--verification-time",
            ISSUED,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads(completed.stdout)
    return completed, report


def test_partial_node_checker_accepts_valid_v2_bundle(tmp_path: Path) -> None:
    completed, report = _run_partial_node_checker(
        _bundle(), tmp_path, name="valid-bundle.json"
    )
    assert completed.returncode == 0, report
    assert report["passed"] is True


def test_partial_node_checker_matches_python_code_point_action_order(
    tmp_path: Path,
) -> None:
    graph = nx.DiGraph(
        snapshot_id="unicode-action-order",
        topology_version=1,
        snapshot_time=SNAPSHOT,
    )
    for node, kind in (
        ("source", "source"),
        ("\ue000", "service"),
        ("😀", "service"),
        ("asset", "asset"),
        ("client", "client"),
        ("database", "database"),
    ):
        graph.add_node(node, type=kind, tier=1, dept="d0")
    graph.add_edge("source", "\ue000", kind="comm", action_cost=1.0)
    graph.add_edge("source", "😀", kind="comm", action_cost=1.0)
    graph.add_edge("\ue000", "asset", kind="comm", action_cost=1.0)
    graph.add_edge("😀", "asset", kind="comm", action_cost=1.0)
    graph.add_edge("client", "database", kind="required", action_cost=1.0)
    actions = [["source", "\ue000"], ["source", "😀"]]
    bundle = _built_bundle(
        graph,
        actions=actions,
        candidate_actions=actions,
        attack_sources=["source"],
        protected_assets=["asset"],
        required_service_pairs=[["client", "database"]],
        method="UnicodeActionOrder",
    )

    completed, report = _run_partial_node_checker(
        bundle, tmp_path, name="unicode-action-order.json"
    )

    assert completed.returncode == 0, report
    assert report["passed"] is True


def test_partial_node_checker_uses_injective_edge_keys_for_nul_ids(
    tmp_path: Path,
) -> None:
    graph = _graph()
    for node in ("left", "right\0tail", "left\0right", "tail"):
        graph.add_node(node, type="inert", tier=1, dept="d0")
    graph.add_edge("left", "right\0tail", kind="inert", action_cost=1.0)
    graph.add_edge("left\0right", "tail", kind="inert", action_cost=1.0)
    bundle = _built_bundle(
        graph,
        actions=[["middle", "asset"]],
        candidate_actions=[["middle", "asset"]],
        attack_sources=["source"],
        protected_assets=["asset"],
        required_service_pairs=[["client", "database"]],
        method="NulEdgeKeys",
    )

    completed, report = _run_partial_node_checker(
        bundle, tmp_path, name="nul-edge-keys.json"
    )

    assert completed.returncode == 0, report
    assert report["passed"] is True


def test_partial_node_checker_honors_preblocked_graph_edges(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    service_edge = next(
        edge
        for edge in bundle["graph"]["edges"]
        if edge["source"] == "client" and edge["destination"] == "database"
    )
    service_edge["attributes"]["blocked"] = True

    completed, report = _run_partial_node_checker(
        bundle, tmp_path, name="preblocked-service-edge.json"
    )

    assert completed.returncode == 1, report
    assert report["passed"] is False
    assert "service preservation below threshold" in report["messages"]


@pytest.mark.parametrize(
    "invalid_snapshot_time",
    [
        "not-a-time",
        7,
        "0000-01-01T00:00:00Z",
        "0001-01-01T00:00:00+23:59",
        "9999-12-31T23:59:59-23:59",
    ],
)
def test_partial_node_checker_rejects_invalid_v2_declared_snapshot_time(
    tmp_path: Path, invalid_snapshot_time: object
) -> None:
    bundle = _bundle()
    bundle["constraints"]["freshness"]["snapshot_time"] = invalid_snapshot_time
    bundle["capsule"]["freshness"]["snapshot_time"] = invalid_snapshot_time

    completed, report = _run_partial_node_checker(
        bundle, tmp_path, name=f"invalid-declared-snapshot-{type(invalid_snapshot_time).__name__}.json"
    )

    assert completed.returncode == 1, report
    assert report["passed"] is False
    assert any(
        "invalid declared freshness snapshot_time" in message
        for message in report["messages"]
    )


@pytest.mark.parametrize(
    ("case_name", "mutation"),
    [
        (
            "embedded-verified-at",
            lambda certificate: certificate.__setitem__("verified_at", 7),
        ),
        (
            "embedded-freshness",
            lambda certificate: certificate["freshness"].__setitem__(
                "snapshot_time", "not-a-time"
            ),
        ),
        (
            "embedded-path-scope",
            lambda certificate: certificate["path_search_scope"].__setitem__(
                "cutoff_edges", False
            ),
        ),
        (
            "unchecked-decision-copy",
            lambda certificate: certificate.__setitem__("freshness_valid", False),
        ),
        (
            "untyped-checked-decision-copy",
            lambda certificate: certificate.__setitem__(
                "attack_reachability_blocked", "not-a-boolean"
            ),
        ),
    ],
)
def test_partial_node_checker_does_not_claim_unsupported_certificate_fields(
    tmp_path: Path, case_name: str, mutation
) -> None:
    bundle = _bundle()
    mutation(bundle["capsule"]["service_safety_certificate"])

    completed, report = _run_partial_node_checker(
        bundle, tmp_path, name=f"unsupported-{case_name}.json"
    )

    assert completed.returncode == 0, report
    assert report["passed"] is True


def test_partial_node_checker_rejects_false_checked_decision_copy(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    bundle["capsule"]["service_safety_certificate"][
        "attack_reachability_blocked"
    ] = False

    completed, report = _run_partial_node_checker(
        bundle, tmp_path, name="false-checked-decision-copy.json"
    )

    assert completed.returncode == 1, report
    assert report["passed"] is False
    assert "embedded attack decision mismatch" in report["messages"]


def test_verification_cli_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    bundle = _bundle()
    canonical = canonical_json_bytes(bundle)
    duplicate = (
        b'{"schema":"cin.verification-bundle.v2",' + canonical[1:]
    )
    path = tmp_path / "duplicate.json"
    path.write_bytes(duplicate)
    output = tmp_path / "unused.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_certificate.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(path),
            "--expected-bundle-sha256",
            verification_bundle_hash(bundle),
            "--verification-time",
            ISSUED,
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "duplicate JSON object key" in completed.stderr
    assert not output.exists()

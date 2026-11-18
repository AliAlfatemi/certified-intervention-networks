"""Adversarial regressions for mandatory snapshot provenance and freshness."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, tzinfo
import hashlib
import json
from pathlib import Path
import subprocess

import networkx as nx
import pytest

from cin.capsule import (
    EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
    EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
    build_capsule,
)
from cin.provenance import deterministic_experiment_keypair, sign_snapshot_attestation
from cin.repair import identity_transformation
from cin.simulator import PathSearchScope
from cin.verifier import (
    make_verification_bundle,
    verification_bundle_hash,
    verify_capsule_bundle,
)


def _graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in ("source", "middle", "asset", "client", "database"):
        graph.add_node(node, type="service", tier=1, dept="d0")
    attributes = {
        "kind": "comm",
        "risk_prob": 0.5,
        "anomaly_score": 0.5,
        "action_cost": 1.0,
    }
    graph.add_edge("source", "middle", **attributes)
    graph.add_edge("middle", "asset", **attributes)
    graph.add_edge("client", "database", required=True, **attributes)
    graph.graph.update(
        snapshot_id="provenance-test-snapshot",
        topology_version=7,
        snapshot_time="2026-07-26T00:00:00Z",
    )
    return graph


TEST_ISSUED_AT = "2026-07-26T00:01:00Z"


def _capsule_and_bundle(
    *,
    max_capsule_age_seconds: int | None = None,
    max_snapshot_age_at_issue_seconds: int = EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
):
    graph = _graph()
    capsule = build_capsule(
        method="ProvenanceTest",
        graph=graph,
        proposal_transformation=identity_transformation([("source", "middle")]),
        candidate_actions=sorted(graph.edges),
        attack_sources=["source"],
        protected_assets=["asset"],
        required_service_pairs=[["client", "database"]],
        action_budget=1,
        path_scope=PathSearchScope(cutoff_edges=2),
        issued_at=TEST_ISSUED_AT,
        max_snapshot_age_at_issue_seconds=max_snapshot_age_at_issue_seconds,
        max_capsule_age_seconds=(
            EXPERIMENT_MAX_CAPSULE_AGE_SECONDS
            if max_capsule_age_seconds is None
            else max_capsule_age_seconds
        ),
    )
    bundle = make_verification_bundle(
        graph,
        capsule.to_dict(),
        ["source"],
        ["asset"],
        [["client", "database"]],
        candidate_actions=capsule.candidate_actions,
        tau=1.0,
        action_budget=1,
        path_scope=PathSearchScope(cutoff_edges=2),
        freshness=capsule.freshness,
    )
    return graph, capsule, bundle


def _verify(bundle, verification_time: str = TEST_ISSUED_AT):
    return verify_capsule_bundle(
        bundle,
        expected_bundle_sha256=verification_bundle_hash(bundle),
        verification_time=verification_time,
    )


@pytest.mark.parametrize("missing", ["snapshot_id", "topology_version", "snapshot_time"])
def test_builder_hard_fails_each_missing_graph_provenance_field_without_mutation(
    missing: str,
) -> None:
    graph = _graph()
    del graph.graph[missing]
    attributes_before = dict(graph.graph)
    with pytest.raises(ValueError, match=rf"missing required field: {missing}"):
        build_capsule(
            method="MissingProvenance",
            graph=graph,
            proposal_transformation=identity_transformation([("source", "middle")]),
            candidate_actions=sorted(graph.edges),
            attack_sources=["source"],
            protected_assets=["asset"],
            required_service_pairs=[["client", "database"]],
            action_budget=1,
            issued_at=TEST_ISSUED_AT,
            max_snapshot_age_at_issue_seconds=EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
            max_capsule_age_seconds=EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
        )
    assert graph.graph == attributes_before
    assert missing not in graph.graph


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("snapshot_id", "", "snapshot_id"),
        ("snapshot_id", " leading-space", "snapshot_id"),
        ("snapshot_id", 7, "snapshot_id"),
        ("topology_version", 0, "positive integer"),
        ("topology_version", -1, "positive integer"),
        ("topology_version", True, "positive integer"),
        ("topology_version", 1.0, "positive integer"),
        ("topology_version", "7", "positive integer"),
        ("snapshot_time", "2026-07-26T00:00:00", "explicit timezone"),
        ("snapshot_time", "2026-07-26 00:00:00Z", "explicit timezone"),
        ("snapshot_time", "2026-07-26T00:00:00z", "explicit timezone"),
        ("snapshot_time", "2026-02-30T00:00:00Z", "explicit timezone"),
        ("snapshot_time", 0, "explicit timezone"),
    ],
)
def test_builder_hard_fails_malformed_graph_provenance(
    field: str,
    invalid: object,
    message: str,
) -> None:
    graph = _graph()
    graph.graph[field] = invalid
    with pytest.raises(ValueError, match=message):
        build_capsule(
            method="MalformedProvenance",
            graph=graph,
            proposal_transformation=identity_transformation([("source", "middle")]),
            candidate_actions=sorted(graph.edges),
            attack_sources=["source"],
            protected_assets=["asset"],
            required_service_pairs=[["client", "database"]],
            action_budget=1,
            issued_at=TEST_ISSUED_AT,
            max_snapshot_age_at_issue_seconds=EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
            max_capsule_age_seconds=EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
        )


def test_explicit_freshness_policy_and_documented_experiment_constant() -> None:
    assert EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS == 300
    assert EXPERIMENT_MAX_CAPSULE_AGE_SECONDS == 2_592_000
    _graph_value, default_capsule, _default_bundle = _capsule_and_bundle()
    assert (
        default_capsule.freshness["max_snapshot_age_at_issue_seconds"]
        == EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS
    )
    assert (
        default_capsule.freshness["max_capsule_age_seconds"]
        == EXPERIMENT_MAX_CAPSULE_AGE_SECONDS
    )
    _graph_value, capsule, _bundle = _capsule_and_bundle(
        max_capsule_age_seconds=900,
        max_snapshot_age_at_issue_seconds=120,
    )
    assert capsule.freshness == {
        "schema": "cin.freshness.v2",
        "snapshot_id": "provenance-test-snapshot",
        "topology_version": 7,
        "snapshot_time": "2026-07-26T00:00:00Z",
        "issued_at": TEST_ISSUED_AT,
        "expires_at": "2026-07-26T00:16:00Z",
        "max_snapshot_age_at_issue_seconds": 120,
        "max_capsule_age_seconds": 900,
    }


def test_builder_requires_explicit_freshness_policy() -> None:
    graph = _graph()
    with pytest.raises(
        TypeError,
        match="max_snapshot_age_at_issue_seconds.*max_capsule_age_seconds",
    ):
        build_capsule(
            method="MissingFreshnessPolicy",
            graph=graph,
            proposal_transformation=identity_transformation([("source", "middle")]),
            candidate_actions=sorted(graph.edges),
            attack_sources=["source"],
            protected_assets=["asset"],
            required_service_pairs=[["client", "database"]],
            action_budget=1,
            issued_at=TEST_ISSUED_AT,
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("max_snapshot_age_at_issue_seconds", value)
        for value in (-1, True, 1.5, "900", 10**1000)
    ]
    + [
        ("max_capsule_age_seconds", value)
        for value in (0, -1, True, 1.5, "900", 10**1000)
    ],
)
def test_builder_rejects_invalid_freshness_policy(field: str, invalid: object) -> None:
    graph = _graph()
    policies: dict[str, object] = {
        "max_snapshot_age_at_issue_seconds": EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
        "max_capsule_age_seconds": EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
    }
    policies[field] = invalid
    with pytest.raises(ValueError, match=field):
        build_capsule(
            method="InvalidFreshnessPolicy",
            graph=graph,
            proposal_transformation=identity_transformation([("source", "middle")]),
            candidate_actions=sorted(graph.edges),
            attack_sources=["source"],
            protected_assets=["asset"],
            required_service_pairs=[["client", "database"]],
            action_budget=1,
            issued_at=TEST_ISSUED_AT,
            **policies,  # type: ignore[arg-type]
        )


def test_timezone_offset_is_explicit_and_normalized_canonically() -> None:
    graph = _graph()
    graph.graph["snapshot_time"] = "2026-07-26T01:30:00+01:30"
    capsule = build_capsule(
        method="OffsetTimestamp",
        graph=graph,
        proposal_transformation=identity_transformation([("source", "middle")]),
        candidate_actions=sorted(graph.edges),
        attack_sources=["source"],
        protected_assets=["asset"],
        required_service_pairs=[["client", "database"]],
        action_budget=1,
        issued_at=TEST_ISSUED_AT,
        max_snapshot_age_at_issue_seconds=EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
        max_capsule_age_seconds=EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
    )
    assert capsule.freshness["snapshot_time"] == "2026-07-26T00:00:00Z"
    assert capsule.freshness["issued_at"] == TEST_ISSUED_AT


@pytest.mark.parametrize(
    ("mutation", "verification_time", "message"),
    [
        (
            lambda bundle: bundle["graph"]["graph_attributes"].pop("snapshot_id"),
            TEST_ISSUED_AT,
            "missing required field: snapshot_id",
        ),
        (
            lambda bundle: bundle["graph"]["graph_attributes"].update(
                snapshot_time="2026-07-26T00:00:00"
            ),
            TEST_ISSUED_AT,
            "snapshot_time must be an RFC 3339 timestamp",
        ),
        (
            lambda bundle: bundle["constraints"]["freshness"].update(
                issued_at="2026-07-26T00:00:00"
            ),
            TEST_ISSUED_AT,
            "must be RFC 3339 timestamps",
        ),
        (
            lambda bundle: bundle["constraints"]["freshness"].update(
                expires_at="2026-07-25T23:59:59Z"
            ),
            TEST_ISSUED_AT,
            "expires_at must be later",
        ),
        (
            lambda bundle: bundle["constraints"]["freshness"].update(
                expires_at=TEST_ISSUED_AT
            ),
            TEST_ISSUED_AT,
            "expires_at must be later",
        ),
        (
            lambda bundle: bundle["constraints"]["freshness"].update(
                expires_at="2026-07-27T00:01:01Z"
            ),
            TEST_ISSUED_AT,
            "expiry does not match",
        ),
        (
            lambda bundle: bundle["constraints"]["freshness"].update(
                topology_version=8
            ),
            TEST_ISSUED_AT,
            "topology_version does not match",
        ),
        (
            lambda bundle: bundle["constraints"]["freshness"].update(
                snapshot_id="different-snapshot"
            ),
            TEST_ISSUED_AT,
            "snapshot_id does not match",
        ),
        (
            lambda bundle: bundle["constraints"]["freshness"].update(
                issued_at="2026-07-27T00:00:00Z",
                expires_at="2026-08-26T00:00:00Z",
            ),
            "2026-07-27T00:00:00Z",
            "snapshot is too old",
        ),
        (
            lambda bundle: bundle["constraints"]["freshness"].update(
                max_capsule_age_seconds=True
            ),
            TEST_ISSUED_AT,
            "max_capsule_age_seconds must be a positive integer",
        ),
    ],
)
def test_replay_rejects_malformed_or_inconsistent_provenance(
    mutation,
    verification_time: str,
    message: str,
) -> None:
    _graph_value, _capsule, bundle = _capsule_and_bundle()
    mutation(bundle)
    result = _verify(bundle, verification_time)
    assert result.passed is False
    assert result.freshness_valid is False
    assert any(message in item for item in result.messages), result.messages


@pytest.mark.parametrize(
    "missing",
    [
        "schema",
        "snapshot_id",
        "topology_version",
        "snapshot_time",
        "issued_at",
        "expires_at",
        "max_snapshot_age_at_issue_seconds",
        "max_capsule_age_seconds",
    ],
)
def test_replay_rejects_each_missing_freshness_field(missing: str) -> None:
    _graph_value, _capsule, bundle = _capsule_and_bundle()
    del bundle["constraints"]["freshness"][missing]
    if missing == "schema":
        with pytest.raises(ValueError, match="v2 bundle requires cin.freshness.v2"):
            _verify(bundle)
        return
    result = _verify(bundle)
    assert result.passed is False
    assert result.freshness_valid is False
    assert any("missing or unsupported field" in item for item in result.messages)


@pytest.mark.parametrize("missing", ["snapshot_id", "topology_version", "snapshot_time"])
def test_replay_rejects_each_missing_graph_provenance_field(missing: str) -> None:
    _graph_value, _capsule, bundle = _capsule_and_bundle()
    del bundle["graph"]["graph_attributes"][missing]
    result = _verify(bundle)
    assert result.passed is False
    assert result.freshness_valid is False
    assert any(f"missing required field: {missing}" in item for item in result.messages)


def test_replay_rejects_future_snapshot_and_issuance() -> None:
    _graph_value, _capsule, bundle = _capsule_and_bundle()
    result = _verify(bundle, "2026-07-25T23:59:59Z")
    assert result.passed is False
    assert result.freshness_valid is False
    assert any("in the future" in item for item in result.messages)


def test_replay_treats_exact_expiry_as_stale() -> None:
    _graph_value, capsule, bundle = _capsule_and_bundle(max_capsule_age_seconds=900)
    result = _verify(bundle, str(capsule.freshness["expires_at"]))
    assert result.passed is False
    assert result.freshness_valid is False
    assert any("stale" in item for item in result.messages)


@pytest.mark.parametrize(
    "verification_time",
    ["2026-07-26T00:00:00", "2026-07-26 00:00:00Z", "not-a-time"],
)
def test_replay_hard_fails_malformed_external_verification_time(
    verification_time: str,
) -> None:
    _graph_value, _capsule, bundle = _capsule_and_bundle()
    with pytest.raises(ValueError, match="verification_time must be a timezone-aware"):
        _verify(bundle, verification_time)


class _NoOffsetTimezone(tzinfo):
    def utcoffset(self, _dt):
        return None

    def dst(self, _dt):
        return None


def test_replay_hard_fails_datetime_with_nonoffset_timezone() -> None:
    _graph_value, _capsule, bundle = _capsule_and_bundle()
    malformed_time = datetime(2026, 7, 26, tzinfo=_NoOffsetTimezone())
    with pytest.raises(ValueError, match="verification_time must be timezone-aware"):
        verify_capsule_bundle(
            bundle,
            expected_bundle_sha256=verification_bundle_hash(bundle),
            verification_time=malformed_time,
        )


def _run_node_checker(bundle: dict, path: Path, verification_time: str):
    raw = json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    bundle_path = path / "bundle.json"
    bundle_path.write_bytes(raw)
    checker = Path(__file__).resolve().parents[1] / "independent_checker" / "verify_bundle.mjs"
    return subprocess.run(
        [
            "node",
            str(checker),
            "--bundle",
            str(bundle_path),
            "--expected-root",
            hashlib.sha256(raw).hexdigest(),
            "--verification-time",
            verification_time,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "verification_time",
    [
        "2026-07-26T00:00:00",
        "2026-07-26 00:00:00Z",
        "2026-07-26T00:00:00z",
        "2026-07-26T00:00:00+0000",
        "2026-02-30T00:00:00Z",
    ],
)
def test_node_checker_rejects_non_rfc3339_external_time(
    tmp_path: Path,
    verification_time: str,
) -> None:
    _graph_value, _capsule, bundle = _capsule_and_bundle()
    result = _run_node_checker(bundle, tmp_path, verification_time)
    assert result.returncode == 1
    assert "invalid RFC 3339 timestamp" in json.loads(result.stdout)["messages"]


def test_node_checker_treats_exact_expiry_as_stale(tmp_path: Path) -> None:
    _graph_value, capsule, bundle = _capsule_and_bundle(max_capsule_age_seconds=900)
    result = _run_node_checker(bundle, tmp_path, str(capsule.freshness["expires_at"]))
    assert result.returncode == 1
    assert "verification time outside half-open interval" in json.loads(result.stdout)["messages"]


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("snapshot_id", "", "invalid graph snapshot id"),
        ("topology_version", 0, "invalid graph topology version"),
    ],
)
def test_node_checker_rejects_invalid_graph_provenance_even_when_copies_agree(
    tmp_path: Path,
    field: str,
    invalid: object,
    message: str,
) -> None:
    _graph_value, _capsule, bundle = _capsule_and_bundle()
    bundle["graph"]["graph_attributes"][field] = invalid
    bundle["constraints"]["freshness"][field] = invalid
    bundle["capsule"]["freshness"][field] = invalid
    result = _run_node_checker(bundle, tmp_path, TEST_ISSUED_AT)
    assert result.returncode == 1
    assert message in json.loads(result.stdout)["messages"]


def test_bundle_constructor_hard_fails_invalid_freshness() -> None:
    graph, capsule, _bundle = _capsule_and_bundle()
    malformed = dict(capsule.freshness)
    malformed.pop("topology_version")
    with pytest.raises(ValueError, match="invalid freshness declaration"):
        make_verification_bundle(
            graph,
            capsule.to_dict(),
            ["source"],
            ["asset"],
            [["client", "database"]],
            candidate_actions=capsule.candidate_actions,
            tau=1.0,
            action_budget=1,
            path_scope=PathSearchScope(cutoff_edges=2),
            freshness=malformed,
        )


def test_replay_hard_fails_nonobject_freshness_with_clear_message() -> None:
    _graph_value, _capsule, bundle = _capsule_and_bundle()
    bundle["constraints"]["freshness"] = None
    with pytest.raises(ValueError, match="freshness declaration is missing or is not an object"):
        _verify(bundle)


def test_attestation_signer_refuses_missing_or_temporally_invalid_provenance() -> None:
    private_key, public_key = deterministic_experiment_keypair()
    graph = _graph()
    del graph.graph["snapshot_id"]
    with pytest.raises(ValueError, match="missing required field: snapshot_id"):
        sign_snapshot_attestation(
            graph,
            private_key,
            public_key,
            signer_id="test",
            discovery_source="fixture",
            discovery_evidence={},
            issued_at="2026-07-26T00:00:00Z",
        )

    graph = _graph()
    for issued_at, message in (
        ("2026-07-26T00:00:00", "explicit timezone"),
        ("2026-07-25T23:59:59Z", "must not precede"),
    ):
        with pytest.raises(ValueError, match=message):
            sign_snapshot_attestation(
                graph,
                private_key,
                public_key,
                signer_id="test",
                discovery_source="fixture",
                discovery_evidence={},
                issued_at=issued_at,
            )


def test_valid_replay_remains_accepted() -> None:
    _graph_value, _capsule, bundle = _capsule_and_bundle()
    result = _verify(bundle)
    assert result.passed is True
    assert result.freshness_valid is True

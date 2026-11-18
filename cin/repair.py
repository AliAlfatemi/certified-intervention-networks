"""Explicit proposer-side proposal transformations.

A proposer may invoke a transformation, bind its complete deterministic log
into a v2 capsule, and then submit the resulting action set to the pure
exact-proposal checker. The checker never transforms that supplied set;
serialized bundle replay calls only ``validate_transformation`` to reproduce
and authenticate the separately declared proposer-side step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx

from .canonical_json import canonical_json_bytes
from .simulator import all_attack_routes_blocked, apply_interventions, service_preservation
from .verifier import canonical_action_set, stable_hash

Edge = Tuple[str, str]

REPAIR_SCHEMA = "cin.proposal-transformation.v1"
REVERSE_DELETE_ALGORITHM = "lexicographic-reverse-delete-fixed-point-v1"
IDENTITY_ALGORITHM = "identity-v1"


def _action_payload(actions: Sequence[Edge]) -> List[List[str]]:
    return [[source, destination] for source, destination in actions]


def action_set_sha256(actions: Sequence[Edge]) -> str:
    """Hash the canonical, duplicate-free semantic action set."""

    canonical = canonical_action_set(actions, field="actions")
    return stable_hash(_action_payload(canonical))


def _feasible(
    graph: nx.DiGraph,
    actions: Sequence[Edge],
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    required_service_pairs: Sequence[Sequence[str]],
    tau: float,
) -> bool:
    active = apply_interventions(graph, actions)
    return all_attack_routes_blocked(
        active, attack_sources, protected_assets
    ) and service_preservation(active, required_service_pairs) + 1e-12 >= tau


@dataclass(frozen=True)
class RepairResult:
    """Canonical result of an explicit proposer-side transformation."""

    schema: str
    algorithm: str
    transformation_version: str
    status: str
    raw_actions: Tuple[Edge, ...]
    repaired_actions: Tuple[Edge, ...]
    raw_actions_sha256: str
    repaired_actions_sha256: str
    steps: Tuple[Mapping[str, object], ...]
    repair_log_sha256: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "algorithm": self.algorithm,
            "transformation_version": self.transformation_version,
            "status": self.status,
            "raw_actions": _action_payload(self.raw_actions),
            "repaired_actions": _action_payload(self.repaired_actions),
            "raw_actions_sha256": self.raw_actions_sha256,
            "repaired_actions_sha256": self.repaired_actions_sha256,
            "steps": [dict(step) for step in self.steps],
            "repair_log_sha256": self.repair_log_sha256,
        }


def _finish(
    *,
    algorithm: str,
    status: str,
    raw_actions: Sequence[Edge],
    repaired_actions: Sequence[Edge],
    steps: Sequence[Mapping[str, object]],
) -> RepairResult:
    raw = canonical_action_set(raw_actions, field="raw_actions")
    repaired = canonical_action_set(repaired_actions, field="repaired_actions")
    core: Dict[str, object] = {
        "schema": REPAIR_SCHEMA,
        "algorithm": algorithm,
        "transformation_version": algorithm,
        "status": status,
        "raw_actions": _action_payload(raw),
        "repaired_actions": _action_payload(repaired),
        "raw_actions_sha256": stable_hash(_action_payload(raw)),
        "repaired_actions_sha256": stable_hash(_action_payload(repaired)),
        "steps": [dict(step) for step in steps],
    }
    return RepairResult(
        schema=REPAIR_SCHEMA,
        algorithm=algorithm,
        transformation_version=algorithm,
        status=status,
        raw_actions=raw,
        repaired_actions=repaired,
        raw_actions_sha256=str(core["raw_actions_sha256"]),
        repaired_actions_sha256=str(core["repaired_actions_sha256"]),
        steps=tuple(dict(step) for step in steps),
        repair_log_sha256=stable_hash(core["steps"]),
    )


def identity_transformation(actions: Sequence[Edge]) -> RepairResult:
    """Bind an explicit no-repair decision."""

    canonical = canonical_action_set(actions, field="raw_actions")
    return _finish(
        algorithm=IDENTITY_ALGORITHM,
        status="not_requested",
        raw_actions=canonical,
        repaired_actions=canonical,
        steps=(),
    )


class SubsetMinimalRepair:
    """The historical reverse-delete stage, now explicit and replayable."""

    @staticmethod
    def apply(
        graph: nx.DiGraph,
        actions: Sequence[Edge],
        attack_sources: Sequence[str],
        protected_assets: Sequence[str],
        required_service_pairs: Sequence[Sequence[str]],
        *,
        tau: float = 1.0,
    ) -> RepairResult:
        raw = canonical_action_set(actions, field="raw_actions")
        selected = list(raw)
        if not _feasible(
            graph,
            selected,
            attack_sources,
            protected_assets,
            required_service_pairs,
            tau,
        ):
            return _finish(
                algorithm=REVERSE_DELETE_ALGORITHM,
                status="not_applied_raw_infeasible",
                raw_actions=raw,
                repaired_actions=raw,
                steps=(),
            )

        steps: List[Mapping[str, object]] = []
        pass_index = 0
        changed = True
        while changed:
            changed = False
            for index in range(len(selected) - 1, -1, -1):
                before = list(selected)
                candidate = before[index]
                trial = before[:index] + before[index + 1 :]
                trial_feasible = _feasible(
                    graph,
                    trial,
                    attack_sources,
                    protected_assets,
                    required_service_pairs,
                    tau,
                )
                after = trial if trial_feasible else before
                steps.append(
                    {
                        "sequence": len(steps),
                        "pass": pass_index,
                        "candidate": list(candidate),
                        "before_actions_sha256": stable_hash(_action_payload(before)),
                        "trial_actions_sha256": stable_hash(_action_payload(trial)),
                        "trial_feasible": trial_feasible,
                        "decision": "remove" if trial_feasible else "keep",
                        "after_actions_sha256": stable_hash(_action_payload(after)),
                    }
                )
                if trial_feasible:
                    selected = trial
                    changed = True
            pass_index += 1
        return _finish(
            algorithm=REVERSE_DELETE_ALGORITHM,
            status="completed_subset_minimal",
            raw_actions=raw,
            repaired_actions=selected,
            steps=steps,
        )


def validate_transformation(
    transformation: Mapping[str, object],
    graph: nx.DiGraph,
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    required_service_pairs: Sequence[Sequence[str]],
    *,
    tau: float,
) -> Tuple[bool, List[str], RepairResult | None]:
    """Replay a submitted transformation and compare every bound field."""

    messages: List[str] = []
    if not isinstance(transformation, Mapping):
        return False, ["proposal transformation is missing or is not an object"], None
    required = {
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
    }
    if set(transformation) != required:
        messages.append("proposal transformation field set is invalid")
    if (
        transformation.get("schema") != REPAIR_SCHEMA
        or not isinstance(transformation.get("transformation_version"), str)
    ):
        messages.append("proposal transformation schema/version is invalid")
    try:
        raw = canonical_action_set(transformation.get("raw_actions"), field="raw_actions")
        repaired = canonical_action_set(
            transformation.get("repaired_actions"), field="repaired_actions"
        )
    except (TypeError, ValueError) as error:
        return False, messages + [str(error)], None
    algorithm = transformation.get("algorithm")
    if algorithm == IDENTITY_ALGORITHM:
        expected = identity_transformation(raw)
    elif algorithm == REVERSE_DELETE_ALGORITHM:
        expected = SubsetMinimalRepair.apply(
            graph,
            raw,
            attack_sources,
            protected_assets,
            required_service_pairs,
            tau=tau,
        )
    else:
        return False, messages + ["unsupported proposal transformation algorithm"], None
    if tuple(repaired) != expected.repaired_actions:
        messages.append("repaired action set differs from deterministic transformation replay")
    if canonical_json_bytes(dict(transformation)) != canonical_json_bytes(
        expected.to_dict()
    ):
        messages.append("proposal transformation log or binding differs from deterministic replay")
    return not messages, messages, expected

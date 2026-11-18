#!/usr/bin/env python3
"""Generate or independently verify a serialized capsule bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cin.graph_generator import BenchmarkConfig, generate_network
from cin.risk_model import ObservedRiskScorer
from cin.capsule import CertifiedInterventionPlanner
from cin.canonical_json import CanonicalJSONError, strict_json_loads
from cin.simulator import PathSearchScope
from cin.verifier import (
    make_verification_bundle,
    verification_bundle_hash,
    verify_capsule_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", help="Existing cin.verification-bundle.v1/v2 JSON to verify"
    )
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--hidden", type=float, default=0.0)
    parser.add_argument("--density", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument(
        "--verification-time",
        help="External checker time (ISO-8601). Default: current UTC. Use only for explicit historical replay.",
    )
    parser.add_argument(
        "--expected-bundle-sha256",
        help="Externally trusted canonical SHA-256 root; required with --input.",
    )
    parser.add_argument("--output", default="capsule_verification_bundle.json")
    args = parser.parse_args()

    if args.input:
        if not args.expected_bundle_sha256:
            parser.error("--expected-bundle-sha256 is required with --input")
        try:
            bundle = strict_json_loads(Path(args.input).read_bytes())
        except (OSError, CanonicalJSONError, ValueError) as exc:
            parser.error(f"cannot load verification bundle: {exc}")
        try:
            result = verify_capsule_bundle(
                bundle,
                expected_bundle_sha256=args.expected_bundle_sha256,
                verification_time=args.verification_time,
            )
        except (KeyError, TypeError, ValueError) as exc:
            parser.error(f"invalid verification bundle or replay argument: {exc}")
        output = {
            "schema": "cin.verification-result.v1",
            "input": args.input,
            "expected_bundle_sha256": args.expected_bundle_sha256,
            "verification": result.to_dict(),
        }
    else:
        instance = generate_network(
            BenchmarkConfig(args.size, args.noise, args.hidden, args.density, args.seed)
        )
        model = ObservedRiskScorer(cutoff=8)
        model.fit(instance.observed_graph, instance.attack_sources, instance.protected_assets)
        planner = CertifiedInterventionPlanner(tau=1.0, cutoff=8)
        capsule = planner.synthesize(
            instance.observed_graph,
            instance.attack_sources,
            instance.protected_assets,
            instance.required_service_pairs_observed,
            model,
            action_budget=args.budget,
            candidate_actions=sorted(instance.observed_graph.edges),
        )
        bundle = make_verification_bundle(
            instance.observed_graph,
            capsule.to_dict(),
            instance.attack_sources,
            instance.protected_assets,
            instance.required_service_pairs_observed,
            candidate_actions=capsule.candidate_actions,
            tau=1.0,
            action_budget=args.budget,
            path_scope=PathSearchScope(cutoff_edges=8),
            freshness=capsule.freshness,
        )
        bundle_sha256 = verification_bundle_hash(bundle)
        # Generation performs an explicit issuance-time self-check so archived
        # examples remain reproducible. --input defaults to the actual UTC clock.
        replay_time = args.verification_time or capsule.freshness["issued_at"]
        result = verify_capsule_bundle(
            bundle,
            expected_bundle_sha256=bundle_sha256,
            verification_time=replay_time,
        )
        output = bundle

    with Path(args.output).open("w", encoding="utf-8") as handle:
        if args.input:
            json.dump(output, handle, indent=2, allow_nan=False)
        else:
            # Write exactly the canonical bytes whose SHA-256 is reported by
            # verification_bundle_hash. This lets the independent Node.js
            # checker verify the same externally trusted root directly.
            handle.write(
                json.dumps(
                    output,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
    print(
        json.dumps(
            {
                "certificate_passed": result.passed,
                "attack_reachability_blocked": result.attack_reachability_blocked,
                "service_pairs_preserved": result.service_pairs_preserved,
                "policy_equivalent": result.policy_equivalent,
                "subset_minimal": result.subset_minimal,
                "verification_bundle_sha256": result.verification_bundle_sha256,
                "expected_bundle_sha256": result.expected_bundle_sha256,
                "bundle_hash_matches": result.bundle_hash_matches,
                "embedded_issuance_certificate_valid": result.embedded_issuance_certificate_valid,
                "output": args.output,
            },
            indent=2,
        )
    )
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

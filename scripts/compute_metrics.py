
#!/usr/bin/env python3
"""Compute attack-path reduction, service preservation, and certificate pass for one CIN run."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, json, time
from cin.graph_generator import BenchmarkConfig, generate_network
from cin.risk_model import ObservedRiskScorer
from cin.capsule import CertifiedInterventionPlanner
from cin.metrics import evaluate_capsule

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=18)
    ap.add_argument('--size', choices=['small','medium','large'], default='small')
    ap.add_argument('--noise', type=float, default=0.0)
    ap.add_argument('--hidden', type=float, default=0.0)
    ap.add_argument('--density', choices=['low','medium','high'], default='low')
    ap.add_argument('--budget', type=int, default=6)
    args = ap.parse_args()
    inst = generate_network(BenchmarkConfig(args.size, args.noise, args.hidden, args.density, args.seed))
    model = ObservedRiskScorer(cutoff=8)
    model.fit(inst.observed_graph, inst.attack_sources, inst.protected_assets)
    t0 = time.perf_counter()
    cap = CertifiedInterventionPlanner(tau=1.0, cutoff=8).synthesize(
        inst.observed_graph, inst.attack_sources, inst.protected_assets,
        inst.required_service_pairs_observed, model, action_budget=args.budget,
        candidate_actions=sorted(inst.observed_graph.edges),
    )
    row = evaluate_capsule(inst, cap, time.perf_counter() - t0)
    print(json.dumps({
        'attack_path_reduction_true': row['attack_path_reduction_true'],
        'service_path_preservation_true': row['service_path_preservation_true'],
        'certificate_pass': row['certificate_pass'],
        'policy_size': row['policy_size'],
        'runtime_s': row['runtime_s'],
    }, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()

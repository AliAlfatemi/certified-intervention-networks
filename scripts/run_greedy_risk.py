
#!/usr/bin/env python3
"""Run the GreedyRisk baseline on one synthetic condition."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, json, time
from cin.graph_generator import BenchmarkConfig, generate_network
from cin.baselines import greedy_risk
from cin.metrics import evaluate_capsule

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=18)
    ap.add_argument('--size', choices=['small','medium','large'], default='small')
    ap.add_argument('--noise', type=float, default=0.0)
    ap.add_argument('--hidden', type=float, default=0.0)
    ap.add_argument('--density', choices=['low','medium','high'], default='low')
    ap.add_argument('--output', default='greedy_risk_metrics.json')
    ap.add_argument('--budget', type=int, default=6)
    args = ap.parse_args()
    inst = generate_network(BenchmarkConfig(args.size, args.noise, args.hidden, args.density, args.seed))
    t0 = time.perf_counter()
    cap = greedy_risk(
        inst.observed_graph, inst.attack_sources, inst.protected_assets,
        inst.required_service_pairs_observed, args.budget,
        candidate_actions=sorted(inst.observed_graph.edges),
    )
    row = evaluate_capsule(inst, cap, time.perf_counter() - t0)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(row, f, indent=2, sort_keys=True)
    print(json.dumps(row, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()

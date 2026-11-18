
#!/usr/bin/env python3
"""Generate a synthetic CIN graph summary at runtime."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, json
from cin.graph_generator import BenchmarkConfig, generate_network

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=18)
    ap.add_argument('--size', choices=['small','medium','large'], default='small')
    ap.add_argument('--noise', type=float, default=0.0)
    ap.add_argument('--hidden', type=float, default=0.0)
    ap.add_argument('--density', choices=['low','medium','high'], default='low')
    ap.add_argument('--output', default='synthetic_graph_summary.json')
    args = ap.parse_args()
    inst = generate_network(BenchmarkConfig(args.size, args.noise, args.hidden, args.density, args.seed))
    summary = {
        'config': inst.config.__dict__,
        'observed_nodes': inst.observed_graph.number_of_nodes(),
        'observed_edges': inst.observed_graph.number_of_edges(),
        'true_edges': inst.true_graph.number_of_edges(),
        'protected_assets': len(inst.protected_assets),
        'attack_sources': len(inst.attack_sources),
        'required_service_pairs_observed': len(inst.required_service_pairs_observed),
        'required_service_pairs_true': len(inst.required_service_pairs_true),
        'hidden_edges': len(inst.hidden_edges),
        'spurious_edges': len(inst.spurious_edges),
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""Fast reviewer-facing smoke test for the source-code-only package."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, time
from cin.graph_generator import BenchmarkConfig, generate_network
from cin.risk_model import ObservedRiskScorer
from cin.capsule import CertifiedInterventionPlanner
from cin.baselines import greedy_risk
from cin.metrics import evaluate_capsule
from cin.simulator import apply_interventions, service_preservation


def main() -> None:
    cfg = BenchmarkConfig(size='small', noise=0.0, hidden=0.0, density='low', seed=18)
    inst = generate_network(cfg)
    assert inst.observed_graph.number_of_nodes() > 0
    assert inst.observed_graph.number_of_edges() > 0
    source_boundary = sorted(
        (source, destination)
        for source, destination in inst.observed_graph.edges
        if source in set(inst.attack_sources)
    )
    boundary_graph = apply_interventions(inst.observed_graph, source_boundary)
    assert service_preservation(boundary_graph, inst.required_service_pairs_observed) < 1.0
    for source in inst.attack_sources:
        dept_index = int(inst.observed_graph.nodes[source]["dept"].removeprefix("dept"))
        assert (source, f"monitor_{dept_index}") in inst.required_service_pairs_observed

    model = ObservedRiskScorer(cutoff=8)
    model.fit(inst.observed_graph, inst.attack_sources, inst.protected_assets)
    assert model.scores()
    t0 = time.perf_counter()
    cin_cap = CertifiedInterventionPlanner(tau=1.0, cutoff=8).synthesize(
        inst.observed_graph, inst.attack_sources, inst.protected_assets,
        inst.required_service_pairs_observed, model, action_budget=6,
        candidate_actions=sorted(inst.observed_graph.edges),
    )
    cin_metrics = evaluate_capsule(inst, cin_cap, time.perf_counter() - t0)
    baseline = greedy_risk(
        inst.observed_graph, inst.attack_sources, inst.protected_assets,
        inst.required_service_pairs_observed, 6,
        candidate_actions=sorted(inst.observed_graph.edges),
    )
    base_metrics = evaluate_capsule(inst, baseline, 0.0)
    assert 0.0 <= cin_metrics['attack_path_reduction_true'] <= 1.0
    assert 0.0 <= cin_metrics['service_path_preservation_true'] <= 1.0
    assert cin_cap.rollback_condition.get('metric')
    assert cin_cap.service_safety_certificate['passed']
    assert cin_metrics['serialized_replay_pass'] == 1.0
    assert cin_metrics['source_boundary_threat_service_feasible_observed'] == 0.0
    assert cin_metrics['source_boundary_admissible_under_budget_observed'] == 0.0
    assert cin_metrics['broker_boundary_admissible_under_budget_observed'] == (
        cin_metrics['broker_boundary_threat_service_feasible_observed']
        * cin_metrics['broker_boundary_within_budget']
    )
    assert not any(source in set(inst.attack_sources) for source, _destination in cin_cap.interventions)
    assert not any(inst.observed_graph.nodes[source].get('type') == 'gateway' for source, _destination in cin_cap.interventions)
    print(json.dumps({
        'status': 'passed',
        'nodes': inst.observed_graph.number_of_nodes(),
        'edges': inst.observed_graph.number_of_edges(),
        'cin_policy_size': cin_metrics['policy_size'],
        'cin_attack_path_reduction_true': cin_metrics['attack_path_reduction_true'],
        'cin_service_path_preservation_true': cin_metrics['service_path_preservation_true'],
        'cin_certificate_pass': cin_metrics['certificate_pass'],
        'cin_actions_from_attack_source': cin_metrics['actions_from_attack_source'],
        'cin_actions_from_gateway': cin_metrics['actions_from_gateway'],
        'cin_actions_from_broker': cin_metrics['actions_from_broker'],
        'source_boundary_threat_service_feasible_observed': cin_metrics['source_boundary_threat_service_feasible_observed'],
        'source_boundary_admissible_under_budget_observed': cin_metrics['source_boundary_admissible_under_budget_observed'],
        'broker_boundary_action_count': cin_metrics['broker_boundary_action_count'],
        'broker_boundary_within_budget': cin_metrics['broker_boundary_within_budget'],
        'broker_boundary_threat_service_feasible_observed': cin_metrics['broker_boundary_threat_service_feasible_observed'],
        'broker_boundary_admissible_under_budget_observed': cin_metrics['broker_boundary_admissible_under_budget_observed'],
        'capsule_json_bytes': cin_metrics['capsule_json_bytes'],
        'essential_witness_json_bytes': cin_metrics['essential_witness_json_bytes'],
        'serialized_replay_verifier_s': cin_metrics['serialized_replay_verifier_s'],
        'greedy_risk_attack_path_reduction_true': base_metrics['attack_path_reduction_true'],
    }, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()

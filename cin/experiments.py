"""Canonical, deterministic experiment and artifact pipeline."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import json
import hashlib
import math
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .graph_generator import BenchmarkConfig, generate_network
from .risk_model import ObservedRiskScorer
from .capsule import CertifiedInterventionPlanner
from .baselines import (
    anomaly_block,
    centrality_anomaly_block,
    exact_small_cut_oracle,
    greedy_risk,
    best_of_8_weighted_random_rollouts,
    service_aware_greedy_cut,
    static_microsegmentation,
)
from .metrics import evaluate_capsule, summarize
from .simulator import enumerate_attack_paths, path_edges

SIZES = ["small", "medium", "large"]
NOISES = [0.0, 0.1, 0.2, 0.3]
HIDDENS = [0.0, 0.1, 0.2]
DENSITIES = ["low", "medium", "high"]
ACTION_BUDGETS = {"small": 6, "medium": 12, "large": 18}
HIGH_BUDGET_SENSITIVITY = {"small": 12, "medium": 24, "large": 36}
PATH_CUTOFF = 8
BOOTSTRAP_REPETITIONS = 10_000
RANDOMIZATION_REPETITIONS = 100_000
PAIRING_KEYS = ["seed", "size", "noise", "hidden", "density", "action_budget"]
CANONICAL_METHOD = "CIN-CertifiedGreedy"
SERVICE_AWARE_BASELINE = "ServiceAwareGreedyCut"
WEIGHTED_RANDOM_ROLLOUTS = "BestOf8WeightedRandomRollouts"
LEGACY_WEIGHTED_RANDOM_ROLLOUTS = "RandomSequentialDefender"
METHOD_DISPLAY_NAMES = {
    WEIGHTED_RANDOM_ROLLOUTS: r"\shortstack[l]{Best-of-8 Weighted\\Random Rollouts}",
    LEGACY_WEIGHTED_RANDOM_ROLLOUTS: r"\shortstack[l]{Best-of-8 Weighted\\Random Rollouts}",
}


def _auto_jobs(jobs: int | None) -> int:
    if jobs is None or jobs == 0:
        return max(1, min(8, os.cpu_count() or 1))
    return max(1, int(jobs))


def _run_one(config: BenchmarkConfig) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    instance = generate_network(config)
    model = ObservedRiskScorer(cutoff=PATH_CUTOFF)
    fit_start = time.perf_counter()
    fit = model.fit(
        instance.observed_graph,
        instance.attack_sources,
        instance.protected_assets,
    )
    model_fit_s = time.perf_counter() - fit_start
    budget = ACTION_BUDGETS[config.size]
    candidate_actions = sorted(instance.observed_graph.edges)
    planner = CertifiedInterventionPlanner(tau=1.0, cutoff=PATH_CUTOFF)
    rows: List[Dict[str, object]] = []

    calls = [
        (
            CANONICAL_METHOD,
            lambda: planner.synthesize(
                instance.observed_graph,
                instance.attack_sources,
                instance.protected_assets,
                instance.required_service_pairs_observed,
                model,
                action_budget=budget,
                candidate_actions=candidate_actions,
                variant="full",
            ),
        ),
        (
            SERVICE_AWARE_BASELINE,
            lambda: service_aware_greedy_cut(
                instance.observed_graph,
                instance.attack_sources,
                instance.protected_assets,
                instance.required_service_pairs_observed,
                budget,
                candidate_actions=candidate_actions,
            ),
        ),
        (
            "StaticMicrosegmentation",
            lambda: static_microsegmentation(
                instance.observed_graph,
                instance.attack_sources,
                instance.protected_assets,
                instance.required_service_pairs_observed,
                budget,
                candidate_actions=candidate_actions,
            ),
        ),
        (
            "AnomalyBlock",
            lambda: anomaly_block(
                instance.observed_graph,
                instance.attack_sources,
                instance.protected_assets,
                instance.required_service_pairs_observed,
                budget,
                candidate_actions=candidate_actions,
            ),
        ),
        (
            "GreedyRisk",
            lambda: greedy_risk(
                instance.observed_graph,
                instance.attack_sources,
                instance.protected_assets,
                instance.required_service_pairs_observed,
                budget,
                candidate_actions=candidate_actions,
            ),
        ),
        (
            WEIGHTED_RANDOM_ROLLOUTS,
            lambda: best_of_8_weighted_random_rollouts(
                instance.observed_graph,
                instance.attack_sources,
                instance.protected_assets,
                instance.required_service_pairs_observed,
                budget,
                seed=config.seed,
                candidate_actions=candidate_actions,
            ),
        ),
        (
            "GraphAnomalyCentralityBlock",
            lambda: centrality_anomaly_block(
                instance.observed_graph,
                instance.attack_sources,
                instance.protected_assets,
                instance.required_service_pairs_observed,
                budget,
                candidate_actions=candidate_actions,
            ),
        ),
    ]
    canonical_capsule = None
    for _name, function in calls:
        start = time.perf_counter()
        capsule = function()
        runtime = time.perf_counter() - start
        end_to_end_runtime = runtime + model_fit_s if capsule.method == CANONICAL_METHOD else runtime
        row = evaluate_capsule(instance, capsule, end_to_end_runtime)
        row["model_fit_s"] = model_fit_s if capsule.method == CANONICAL_METHOD else 0.0
        row["planner_and_verifier_s"] = runtime
        rows.append(row)
        if capsule.method == CANONICAL_METHOD:
            canonical_capsule = capsule

    diagnostics = {
        "config": config.__dict__,
        "action_budget": budget,
        "metadata": instance.metadata,
        "canonical_capsule": canonical_capsule.to_dict() if canonical_capsule else None,
        "risk_scorer": "fixed transparent observed-edge score (not causal discovery)",
    }
    return rows, diagnostics


def _grid_configs(seeds: int, quick: bool = False) -> List[BenchmarkConfig]:
    sizes = SIZES if not quick else ["small"]
    noises = NOISES if not quick else [0.0, 0.2]
    hidden_values = HIDDENS if not quick else [0.0, 0.2]
    densities = DENSITIES if not quick else ["low", "high"]
    return [
        BenchmarkConfig(size=size, noise=noise, hidden=hidden, density=density, seed=seed)
        for seed in range(seeds)
        for size in sizes
        for noise in noises
        for hidden in hidden_values
        for density in densities
    ]


def _collect_parallel(function, items: Sequence[object], jobs: int):
    workers = _auto_jobs(jobs)
    if workers == 1:
        yield from map(function, items)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(function, items, chunksize=1)


def run_grid(seeds: int, output: str, quick: bool = False, jobs: int = 0) -> pd.DataFrame:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    for condition_rows, diagnostic in _collect_parallel(_run_one, _grid_configs(seeds, quick), jobs):
        rows.extend(condition_rows)
        if len(diagnostics) < 8:
            diagnostics.append(diagnostic)
    dataframe = pd.DataFrame(rows).sort_values(PAIRING_KEYS + ["method"]).reset_index(drop=True)
    dataframe.to_csv(output_path / "raw_results.csv", index=False)
    audit_columns = PAIRING_KEYS + [
        "nodes",
        "edges_true",
        "edges_observed",
        "eligible_hidden_edges",
        "hidden_edge_count",
        "realized_hidden_fraction_eligible",
        "missing_true_edges",
        "spurious_observed_edges",
        "edge_observation_recall",
        "edge_observation_precision",
        "required_edge_missing_count",
        "attack_edge_observation_recall",
        "service_specification_coverage",
        "required_service_pairs_true",
        "required_service_pairs_observed",
    ]
    audit = (
        dataframe[audit_columns]
        .drop_duplicates(PAIRING_KEYS)
        .sort_values(PAIRING_KEYS)
        .reset_index(drop=True)
    )
    audit.to_csv(output_path / "benchmark_observation_audit.csv", index=False)
    observation_summary = (
        audit.groupby(["noise", "hidden"], as_index=False)
        .agg(
            graph_instances=("seed", "size"),
            realized_hidden_fraction_eligible=("realized_hidden_fraction_eligible", "mean"),
            edge_observation_recall=("edge_observation_recall", "mean"),
            edge_observation_precision=("edge_observation_precision", "mean"),
            attack_edge_observation_recall=("attack_edge_observation_recall", "mean"),
            required_edge_missing_count=("required_edge_missing_count", "sum"),
            service_specification_coverage=("service_specification_coverage", "mean"),
        )
    )
    observation_summary.to_csv(
        output_path / "benchmark_observation_summary.csv", index=False
    )
    summarize(dataframe).to_csv(output_path / "summary_results.csv", index=False)
    with (output_path / "diagnostic_capsules.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, allow_nan=False)
    return dataframe


def _ablation_configs(seeds: int, quick: bool = False) -> List[BenchmarkConfig]:
    # Use the complete primary factorial grid for full ablations. The original
    # hard-grid-only design produced almost no feasible cases and therefore
    # could not identify the contribution of individual components.
    sizes = SIZES if not quick else ["small"]
    noises = NOISES if not quick else [0.2]
    hidden_values = HIDDENS if not quick else [0.0, 0.2]
    densities = DENSITIES if not quick else ["medium"]
    return [
        BenchmarkConfig(size=size, noise=noise, hidden=hidden, density=density, seed=seed + 4000)
        for seed in range(seeds)
        for size in sizes
        for noise in noises
        for hidden in hidden_values
        for density in densities
    ]


def _run_ablation_one(config: BenchmarkConfig) -> List[Dict[str, object]]:
    variants = [
        "full",
        "uniform_scores",
        "no_service_guard",
        "bounded_only",
        "random_tiebreak",
        "no_reverse_delete",
    ]
    instance = generate_network(config)
    model = ObservedRiskScorer(cutoff=PATH_CUTOFF)
    fit_start = time.perf_counter()
    fit = model.fit(
        instance.observed_graph,
        instance.attack_sources,
        instance.protected_assets,
    )
    model_fit_s = time.perf_counter() - fit_start
    planner = CertifiedInterventionPlanner(tau=1.0, cutoff=PATH_CUTOFF)
    budget = ACTION_BUDGETS[config.size]
    candidate_actions = sorted(instance.observed_graph.edges)
    rows: List[Dict[str, object]] = []
    for variant in variants:
        start = time.perf_counter()
        capsule = planner.synthesize(
            instance.observed_graph,
            instance.attack_sources,
            instance.protected_assets,
            instance.required_service_pairs_observed,
            model,
            action_budget=budget,
            candidate_actions=candidate_actions,
            variant=variant,
        )
        planner_and_verifier_s = time.perf_counter() - start
        row = evaluate_capsule(instance, capsule, model_fit_s + planner_and_verifier_s)
        row["model_fit_s"] = model_fit_s
        row["planner_and_verifier_s"] = planner_and_verifier_s
        row["ablation"] = variant
        rows.append(row)
    return rows


def run_ablations(seeds: int, output: str, quick: bool = False, jobs: int = 0) -> pd.DataFrame:
    output_path = Path(output)
    rows: List[Dict[str, object]] = []
    for part in _collect_parallel(_run_ablation_one, _ablation_configs(seeds, quick), jobs):
        rows.extend(part)
    dataframe = pd.DataFrame(rows).sort_values(PAIRING_KEYS + ["ablation"]).reset_index(drop=True)
    dataframe.to_csv(output_path / "ablation_results.csv", index=False)
    return dataframe


def _run_topology_stress_one(config: BenchmarkConfig) -> List[Dict[str, object]]:
    """Evaluate the two strongest planners under a multi-egress stress regime."""

    instance = generate_network(config)
    model = ObservedRiskScorer(cutoff=PATH_CUTOFF)
    fit_start = time.perf_counter()
    model.fit(instance.observed_graph, instance.attack_sources, instance.protected_assets)
    model_fit_s = time.perf_counter() - fit_start
    budget = ACTION_BUDGETS[config.size]
    candidate_actions = sorted(instance.observed_graph.edges)
    planner = CertifiedInterventionPlanner(tau=1.0, cutoff=PATH_CUTOFF)
    calls = (
        (
            CANONICAL_METHOD,
            lambda: planner.synthesize(
                instance.observed_graph,
                instance.attack_sources,
                instance.protected_assets,
                instance.required_service_pairs_observed,
                model,
                action_budget=budget,
                candidate_actions=candidate_actions,
            ),
        ),
        (
            SERVICE_AWARE_BASELINE,
            lambda: service_aware_greedy_cut(
                instance.observed_graph,
                instance.attack_sources,
                instance.protected_assets,
                instance.required_service_pairs_observed,
                budget,
                cutoff=PATH_CUTOFF,
                candidate_actions=candidate_actions,
            ),
        ),
    )
    rows: List[Dict[str, object]] = []
    for method, function in calls:
        start = time.perf_counter()
        capsule = function()
        planner_s = time.perf_counter() - start
        runtime = planner_s + model_fit_s if method == CANONICAL_METHOD else planner_s
        row = evaluate_capsule(instance, capsule, runtime, cutoff=PATH_CUTOFF)
        row.update(
            {
                "diagnostic_only": True,
                "topology_stress_seed_offset": 20000,
                "model_fit_s": model_fit_s if method == CANONICAL_METHOD else 0.0,
                "planner_and_verifier_s": planner_s,
            }
        )
        rows.append(row)
    return rows


def run_topology_stress(
    seeds: int,
    output: str,
    quick: bool = False,
    jobs: int = 0,
) -> pd.DataFrame:
    """Test dependence on the primary generator's shared broker boundary."""

    output_path = Path(output)
    count = max(1, min(seeds, 10))
    sizes = SIZES if not quick else ["small"]
    noises = [0.0, 0.2] if not quick else [0.2]
    hidden_values = [0.0, 0.2] if not quick else [0.2]
    densities = ["medium", "high"] if not quick else ["medium"]
    regimes = ["shared_broker", "alternate_egress"]
    configs = [
        BenchmarkConfig(
            size=size,
            noise=noise,
            hidden=hidden,
            density=density,
            seed=seed + 20000,
            topology_regime=regime,
        )
        for seed in range(count)
        for size in sizes
        for noise in noises
        for hidden in hidden_values
        for density in densities
        for regime in regimes
    ]
    rows: List[Dict[str, object]] = []
    for part in _collect_parallel(_run_topology_stress_one, configs, jobs):
        rows.extend(part)
    dataframe = pd.DataFrame(rows).sort_values(
        PAIRING_KEYS + ["topology_regime", "method"]
    ).reset_index(drop=True)
    dataframe.to_csv(output_path / "topology_stress.csv", index=False)
    summary = (
        dataframe.groupby(["topology_regime", "method"], as_index=False)
        .agg(
            attack_pair_isolation_true=("attack_pair_isolation_true", "mean"),
            attack_path_reduction_true=("attack_path_reduction_true", "mean"),
            service_pair_preservation_true=("service_pair_preservation_true", "mean"),
            certificate_coverage=("certificate_pass", "mean"),
            true_feasible=("true_feasible", "mean"),
            broker_boundary_attack_blocked=("broker_boundary_attack_blocked_observed", "mean"),
            broker_boundary_admissible=("broker_boundary_admissible_under_budget_observed", "mean"),
        )
    )
    summary.to_csv(output_path / "topology_stress_summary.csv", index=False)
    return dataframe


def _oracle_candidates(instance, capsules, max_candidates: int = 18) -> List[Tuple[str, str]]:
    paths = enumerate_attack_paths(
        instance.observed_graph,
        instance.attack_sources,
        instance.protected_assets,
        cutoff=PATH_CUTOFF,
    )
    coverage: Dict[Tuple[str, str], int] = {}
    for path in paths:
        for edge in path_edges(path):
            coverage[edge] = coverage.get(edge, 0) + 1
    selected = list(dict.fromkeys(edge for capsule in capsules for edge in capsule.interventions))
    ranked = sorted(coverage, key=lambda edge: (-coverage[edge], edge))
    for edge in ranked:
        if edge not in selected and len(selected) < max_candidates:
            selected.append(edge)
    return selected[:max_candidates]


def _run_oracle_one(task: Tuple[BenchmarkConfig, int] | BenchmarkConfig) -> List[Dict[str, object]]:
    if isinstance(task, BenchmarkConfig):
        config = task
        max_candidates = 18
    else:
        config, max_candidates = task
    instance = generate_network(config)
    model = ObservedRiskScorer(cutoff=PATH_CUTOFF)
    fit_start = time.perf_counter()
    model.fit(instance.observed_graph, instance.attack_sources, instance.protected_assets)
    model_fit_s = time.perf_counter() - fit_start
    # The oracle diagnostic has its own predeclared tractable budget; all three
    # diagnostic methods receive exactly the same B=4.
    budget = 4
    candidate_actions = sorted(instance.observed_graph.edges)
    planner = CertifiedInterventionPlanner(tau=1.0, cutoff=PATH_CUTOFF)
    cin_start = time.perf_counter()
    cin_capsule = planner.synthesize(
        instance.observed_graph,
        instance.attack_sources,
        instance.protected_assets,
        instance.required_service_pairs_observed,
        model,
        action_budget=budget,
        candidate_actions=candidate_actions,
    )
    cin_runtime = model_fit_s + (time.perf_counter() - cin_start)
    service_start = time.perf_counter()
    service_capsule = service_aware_greedy_cut(
        instance.observed_graph,
        instance.attack_sources,
        instance.protected_assets,
        instance.required_service_pairs_observed,
        budget,
        candidate_actions=candidate_actions,
    )
    service_runtime = time.perf_counter() - service_start
    candidates = _oracle_candidates(
        instance,
        [cin_capsule, service_capsule],
        max_candidates=max_candidates,
    )
    start = time.perf_counter()
    oracle_capsule = exact_small_cut_oracle(
        instance.observed_graph,
        instance.attack_sources,
        instance.protected_assets,
        instance.required_service_pairs_observed,
        budget,
        candidate_pool=candidates,
        candidate_actions=candidates,
        max_candidates=max_candidates,
    )
    oracle_runtime = time.perf_counter() - start
    evaluated = []
    for capsule, runtime in (
        (cin_capsule, cin_runtime),
        (service_capsule, service_runtime),
        (oracle_capsule, oracle_runtime),
    ):
        row = evaluate_capsule(instance, capsule, runtime)
        evaluated.append(row)
    oracle_residual = next(
        row["residual_attack_pairs_observed"] for row in evaluated if row["method"] == "ExactBoundedCandidateOracle"
    )
    for row in evaluated:
        row.update(
            {
                "oracle_candidate_count": len(candidates),
                "oracle_candidate_scope": "union of both planner outputs plus highest-coverage edges from cutoff-8 paths",
                "oracle_candidate_limit": max_candidates,
                "oracle_exact_within_declared_pool": True,
                "oracle_global_optimality_claim": False,
                "optimality_gap_residual_attack_pairs_observed": row["residual_attack_pairs_observed"] - oracle_residual,
            }
        )
    return evaluated


def run_oracle_study(seeds: int, output: str, quick: bool = False, jobs: int = 0) -> pd.DataFrame:
    output_path = Path(output)
    count = max(1, min(seeds, 10 if not quick else 3))
    max_candidates = 12 if quick else 18
    tasks = [
        (
            BenchmarkConfig(size="small", noise=noise, hidden=hidden, density="medium", seed=seed + 12000),
            max_candidates,
        )
        for seed in range(count)
        for noise in ([0.0, 0.2] if not quick else [0.2])
        for hidden in ([0.0, 0.2] if not quick else [0.1])
    ]
    rows: List[Dict[str, object]] = []
    for part in _collect_parallel(_run_oracle_one, tasks, jobs):
        rows.extend(part)
    dataframe = pd.DataFrame(rows).sort_values(["seed", "noise", "hidden", "method"]).reset_index(drop=True)
    dataframe.to_csv(output_path / "oracle_diagnostics.csv", index=False)
    return dataframe


def _run_budget_sensitivity_one(config: BenchmarkConfig) -> List[Dict[str, object]]:
    """Compare the primary and former high budgets on a separate seed stream."""

    instance = generate_network(config)
    model = ObservedRiskScorer(cutoff=PATH_CUTOFF)
    fit_start = time.perf_counter()
    model.fit(instance.observed_graph, instance.attack_sources, instance.protected_assets)
    model_fit_s = time.perf_counter() - fit_start
    planner = CertifiedInterventionPlanner(tau=1.0, cutoff=PATH_CUTOFF)
    candidate_actions = sorted(instance.observed_graph.edges)
    rows: List[Dict[str, object]] = []
    regimes = (
        ("primary", ACTION_BUDGETS[config.size]),
        ("high_budget_sensitivity", HIGH_BUDGET_SENSITIVITY[config.size]),
    )
    for regime, budget in regimes:
        calls = (
            (
                CANONICAL_METHOD,
                lambda budget=budget: planner.synthesize(
                    instance.observed_graph,
                    instance.attack_sources,
                    instance.protected_assets,
                    instance.required_service_pairs_observed,
                    model,
                    action_budget=budget,
                    candidate_actions=candidate_actions,
                ),
            ),
            (
                SERVICE_AWARE_BASELINE,
                lambda budget=budget: service_aware_greedy_cut(
                    instance.observed_graph,
                    instance.attack_sources,
                    instance.protected_assets,
                    instance.required_service_pairs_observed,
                    budget,
                    cutoff=PATH_CUTOFF,
                    candidate_actions=candidate_actions,
                ),
            ),
        )
        for method, function in calls:
            start = time.perf_counter()
            capsule = function()
            planner_s = time.perf_counter() - start
            runtime = planner_s + model_fit_s if method == CANONICAL_METHOD else planner_s
            row = evaluate_capsule(instance, capsule, runtime, cutoff=PATH_CUTOFF)
            row.update(
                {
                    "budget_regime": regime,
                    "diagnostic_only": True,
                    "sensitivity_seed_offset": 16000,
                    "model_fit_s": model_fit_s if method == CANONICAL_METHOD else 0.0,
                    "planner_and_verifier_s": planner_s,
                }
            )
            rows.append(row)
    return rows


def run_budget_sensitivity(seeds: int, output: str, quick: bool = False, jobs: int = 0) -> pd.DataFrame:
    """Write a small, predeclared high-budget sensitivity diagnostic.

    This study is separate from the main grid and cannot replace its primary
    6/12/18-action results. It uses up to ten offset seeds, fixed noise/hidden
    rates, and two connectivity densities.
    """

    output_path = Path(output)
    count = max(1, min(seeds, 10 if not quick else seeds))
    sizes = ["small"] if quick else SIZES
    densities = ["medium"] if quick else ["medium", "high"]
    configs = [
        BenchmarkConfig(size=size, noise=0.2, hidden=0.2, density=density, seed=seed + 16000)
        for seed in range(count)
        for size in sizes
        for density in densities
    ]
    rows: List[Dict[str, object]] = []
    for part in _collect_parallel(_run_budget_sensitivity_one, configs, jobs):
        rows.extend(part)
    dataframe = pd.DataFrame(rows).sort_values(
        ["seed", "size", "density", "budget_regime", "method"]
    ).reset_index(drop=True)
    dataframe.to_csv(output_path / "budget_sensitivity.csv", index=False)
    return dataframe


def _paired_cluster_data(raw: pd.DataFrame, baseline: str, metric: str) -> pd.DataFrame:
    canonical = raw.loc[raw.method == CANONICAL_METHOD, PAIRING_KEYS + [metric]].rename(columns={metric: "canonical"})
    comparison = raw.loc[raw.method == baseline, PAIRING_KEYS + [metric]].rename(columns={metric: "baseline"})
    if canonical.duplicated(PAIRING_KEYS).any() or comparison.duplicated(PAIRING_KEYS).any():
        raise ValueError(f"duplicate pairing keys for {baseline}/{metric}")
    merged = canonical.merge(comparison, on=PAIRING_KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(canonical) or len(merged) != len(comparison):
        raise ValueError(f"incomplete explicit key merge for {baseline}/{metric}")
    merged["difference"] = merged["canonical"] - merged["baseline"]
    return merged.dropna(subset=["canonical", "baseline", "difference"]).reset_index(drop=True)


def _cluster_inference(
    merged: pd.DataFrame,
    seed: int,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
    randomization_repetitions: int = RANDOMIZATION_REPETITIONS,
) -> Dict[str, float]:
    cluster_means = merged.groupby("seed", sort=True)["difference"].mean().to_numpy(dtype=float)
    if len(cluster_means) == 0:
        return {
            key: float("nan")
            for key in (
                "mean_difference",
                "ci_low",
                "ci_high",
                "p_value",
                "paired_effect",
                "paired_standardized_effect",
                "monte_carlo_standard_error",
            )
        }
    observed = float(cluster_means.mean())
    bootstrap_seed, randomization_seed = np.random.SeedSequence(seed).spawn(2)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    randomization_rng = np.random.default_rng(randomization_seed)
    indices = bootstrap_rng.integers(
        0,
        len(cluster_means),
        size=(bootstrap_repetitions, len(cluster_means)),
    )
    bootstrap = cluster_means[indices].mean(axis=1)
    signs = randomization_rng.choice(
        np.array([-1.0, 1.0]),
        size=(randomization_repetitions, len(cluster_means)),
    )
    null_distribution = (signs * cluster_means).mean(axis=1)
    extreme_count = int(np.sum(np.abs(null_distribution) >= abs(observed) - 1e-15))
    p_value = (1.0 + extreme_count) / (randomization_repetitions + 1.0)
    standard_deviation = float(np.std(cluster_means, ddof=1)) if len(cluster_means) > 1 else float("nan")
    effect = observed / standard_deviation if standard_deviation > 0 else float("nan")
    monte_carlo_standard_error = math.sqrt(p_value * (1.0 - p_value) / (randomization_repetitions + 1.0))
    return {
        "mean_difference": observed,
        "ci_low": float(np.percentile(bootstrap, 2.5)),
        "ci_high": float(np.percentile(bootstrap, 97.5)),
        "p_value": p_value,
        "paired_effect": effect,
        "paired_standardized_effect": effect,
        "randomization_extreme_count": extreme_count,
        "monte_carlo_standard_error": monte_carlo_standard_error,
        "clusters": int(len(cluster_means)),
        "paired_conditions": int(len(merged)),
    }


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    adjusted = [float("nan")] * len(p_values)
    finite = [(index, value) for index, value in enumerate(p_values) if math.isfinite(value)]
    ordered = sorted(finite, key=lambda item: item[1])
    running = 0.0
    total = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[index] = running
    return adjusted


def write_statistical_report(output: str) -> pd.DataFrame:
    output_path = Path(output)
    raw = pd.read_csv(output_path / "raw_results.csv")
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
    baselines = sorted(set(raw.method) - {CANONICAL_METHOD})
    for baseline_index, baseline in enumerate(baselines):
        for metric_index, metric in enumerate(metrics):
            merged = _paired_cluster_data(raw, baseline, metric)
            inference = _cluster_inference(merged, seed=42 + baseline_index * 101 + metric_index)
            rows.append({"baseline": baseline, "metric": metric, **inference})
    report = pd.DataFrame(rows)
    report["p_value_holm"] = _holm_adjust(report["p_value"].tolist())
    report.to_csv(output_path / "statistical_tests.csv", index=False)
    payload = {
        "schema": "cin.clustered-statistics.v2",
        "pairing_keys": PAIRING_KEYS,
        "cluster_key": "seed",
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "randomization_repetitions": RANDOMIZATION_REPETITIONS,
        "randomization_test": "two-sided cluster sign-flip Monte Carlo with plus-one correction",
        "effect_size": "mean paired seed-cluster difference divided by the sample SD of paired seed-cluster differences",
        "multiple_testing": "Holm family-wise adjustment over all reported comparisons",
        # pandas' JSON encoder maps non-finite small-sample effect sizes to
        # null; the CSV intentionally retains NaN for numerical workflows.
        "tests": json.loads(report.to_json(orient="records")),
    }
    with (output_path / "statistical_report.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    return report


def write_ablation_statistical_report(output: str) -> pd.DataFrame:
    """Paired, seed-clustered contrasts of every ablation against full CIN."""

    output_path = Path(output)
    raw = pd.read_csv(output_path / "ablation_results.csv")
    metrics = [
        "attack_path_reduction_true",
        "attack_pair_isolation_true",
        "service_pair_preservation_true",
        "true_feasible",
        "certificate_pass",
        "action_cost",
    ]
    variants = sorted(set(raw.ablation) - {"full"})
    rows: List[Dict[str, object]] = []
    for variant_index, variant in enumerate(variants):
        for metric_index, metric in enumerate(metrics):
            full = raw.loc[raw.ablation == "full", PAIRING_KEYS + [metric]].rename(
                columns={metric: "canonical"}
            )
            comparison = raw.loc[raw.ablation == variant, PAIRING_KEYS + [metric]].rename(
                columns={metric: "baseline"}
            )
            if full.duplicated(PAIRING_KEYS).any() or comparison.duplicated(PAIRING_KEYS).any():
                raise ValueError(f"duplicate ablation pairing keys for {variant}/{metric}")
            merged = full.merge(comparison, on=PAIRING_KEYS, validate="one_to_one")
            if len(merged) != len(full) or len(merged) != len(comparison):
                raise ValueError(f"incomplete ablation pairing for {variant}/{metric}")
            merged["difference"] = merged["canonical"] - merged["baseline"]
            merged = merged.dropna(subset=["canonical", "baseline", "difference"])
            inference = _cluster_inference(
                merged,
                seed=7000 + variant_index * 101 + metric_index,
            )
            rows.append({"ablation": variant, "metric": metric, **inference})
    report = pd.DataFrame(rows)
    report["p_value_holm"] = _holm_adjust(report["p_value"].tolist())
    report.to_csv(output_path / "ablation_statistical_tests.csv", index=False)
    payload = {
        "schema": "cin.ablation-clustered-statistics.v1",
        "contrast": "full minus ablation",
        "pairing_keys": PAIRING_KEYS,
        "cluster_key": "seed",
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "randomization_repetitions": RANDOMIZATION_REPETITIONS,
        "multiple_testing": "Holm family-wise adjustment over all ablation contrasts",
        "tests": json.loads(report.to_json(orient="records")),
    }
    with (output_path / "ablation_statistical_report.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    return report


def write_topology_stress_statistical_report(output: str) -> pd.DataFrame:
    """Quantify planner and topology-regime contrasts in the stress diagnostic."""

    output_path = Path(output)
    raw = pd.read_csv(output_path / "topology_stress.csv")
    metrics = [
        "attack_pair_isolation_true",
        "service_pair_preservation_true",
        "true_feasible",
        "certificate_pass",
    ]
    condition_keys = PAIRING_KEYS + ["topology_regime"]
    rows: List[Dict[str, object]] = []
    for regime_index, regime in enumerate(sorted(raw.topology_regime.unique())):
        regime_data = raw[raw.topology_regime == regime]
        for metric_index, metric in enumerate(metrics):
            canonical = regime_data.loc[
                regime_data.method == CANONICAL_METHOD, PAIRING_KEYS + [metric]
            ].rename(columns={metric: "canonical"})
            baseline = regime_data.loc[
                regime_data.method == SERVICE_AWARE_BASELINE, PAIRING_KEYS + [metric]
            ].rename(columns={metric: "baseline"})
            merged = canonical.merge(baseline, on=PAIRING_KEYS, validate="one_to_one")
            merged["difference"] = merged["canonical"] - merged["baseline"]
            merged = merged.dropna(subset=["difference"])
            inference = _cluster_inference(
                merged,
                seed=9000 + regime_index * 101 + metric_index,
            )
            rows.append(
                {
                    "contrast": "CIN minus ServiceAwareGreedyCut",
                    "topology_regime": regime,
                    "method": "between_methods",
                    "metric": metric,
                    **inference,
                }
            )

    regime_pair_keys = PAIRING_KEYS + ["method"]
    for method_index, method in enumerate(sorted(raw.method.unique())):
        method_data = raw[raw.method == method]
        for metric_index, metric in enumerate(metrics):
            alternate = method_data.loc[
                method_data.topology_regime == "alternate_egress",
                regime_pair_keys + [metric],
            ].rename(columns={metric: "canonical"})
            shared = method_data.loc[
                method_data.topology_regime == "shared_broker",
                regime_pair_keys + [metric],
            ].rename(columns={metric: "baseline"})
            merged = alternate.merge(shared, on=regime_pair_keys, validate="one_to_one")
            merged["difference"] = merged["canonical"] - merged["baseline"]
            merged = merged.dropna(subset=["difference"])
            inference = _cluster_inference(
                merged,
                seed=10000 + method_index * 101 + metric_index,
            )
            rows.append(
                {
                    "contrast": "alternate_egress minus shared_broker",
                    "topology_regime": "alternate_vs_shared",
                    "method": method,
                    "metric": metric,
                    **inference,
                }
            )
    report = pd.DataFrame(rows)
    report["p_value_holm"] = _holm_adjust(report["p_value"].tolist())
    report.to_csv(output_path / "topology_stress_statistical_tests.csv", index=False)
    return report


def write_paper_tables(output: str) -> pd.DataFrame:
    output_path = Path(output)
    raw = pd.read_csv(output_path / "raw_results.csv")
    rows = []
    for method, group in raw.groupby("method", sort=True):
        accepted = group[group.certificate_pass > 0.5]
        rejected = group[group.certificate_pass <= 0.5]
        rows.append(
            {
                "method": method,
                "attack_pair_isolation_true_mean": group.attack_pair_isolation_true.mean(),
                "attack_path_reduction_true_mean": group.attack_path_reduction_true.mean(),
                "attack_path_reduction_true_std": group.attack_path_reduction_true.std(),
                "service_pair_preservation_true_mean": group.service_pair_preservation_true.mean(),
                "certificate_coverage": group.certificate_pass.mean(),
                "true_feasible_rate": group.true_feasible.mean(),
                "true_safe_precision_among_accepted": accepted.true_feasible.mean() if len(accepted) else float("nan"),
                "true_feasible_rate_among_rejected": rejected.true_feasible.mean() if len(rejected) else float("nan"),
                "no_true_threat_conditions": int((group.threat_present_true < 0.5).sum()),
                "action_cost_mean": group.action_cost.mean(),
                "policy_size_mean": group.policy_size.mean(),
                "runtime_s_mean": group.runtime_s.mean(),
                "capsule_json_bytes_mean": group.capsule_json_bytes.mean(),
                "essential_witness_json_bytes_mean": group.essential_witness_json_bytes.mean(),
                "verification_bundle_json_bytes_mean": group.verification_bundle_json_bytes.mean(),
                "serialized_replay_verifier_s_mean": group.serialized_replay_verifier_s.mean(),
                "serialized_replay_decision_match_rate": (
                    group.serialized_replay_pass.round() == group.certificate_pass.round()
                ).mean(),
                "action_fraction_from_attack_source_mean": group.action_fraction_from_attack_source.mean(),
                "action_fraction_from_broker_mean": group.action_fraction_from_broker.mean(),
                "source_boundary_threat_service_feasible_rate": (
                    group.source_boundary_threat_service_feasible_observed.mean()
                ),
                "source_boundary_admissible_under_budget_rate": (
                    group.source_boundary_admissible_under_budget_observed.mean()
                ),
                "broker_boundary_within_budget_rate": group.broker_boundary_within_budget.mean(),
                "broker_boundary_threat_service_feasible_rate": (
                    group.broker_boundary_threat_service_feasible_observed.mean()
                ),
                "broker_boundary_admissible_under_budget_rate": (
                    group.broker_boundary_admissible_under_budget_observed.mean()
                ),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_path / "paper_main_table.csv", index=False)
    return table


def _fmt(value: float, digits: int = 3, *, signed: bool = False) -> str:
    """Format a finite scalar for LaTeX without ever emitting NaN/Inf."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(number):
        return "--"
    sign = "+" if signed else ""
    return f"{number:{sign}.{digits}f}"


def _fmt_percent(value: float, digits: int = 1) -> str:
    try:
        percentage = 100.0 * float(value)
    except (TypeError, ValueError):
        return "--"
    formatted = _fmt(percentage, digits)
    return f"{formatted}\\%" if formatted != "--" else "--"


def _latex_method(value: str) -> str:
    return METHOD_DISPLAY_NAMES.get(value, value.replace("_", "\\_"))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _tex_int(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def write_reviewed_latex_fragments(results: str, destination: str) -> Path:
    """Write human-facing derivatives without mutating frozen execution outputs.

    Archived primary rows retain their historical machine identifiers so the
    execution manifest remains verifiable. This function maps those identifiers
    to current display names and derives all displayed values from the frozen
    CSV files.
    """

    results_path = Path(results)
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(results_path / "raw_results.csv")
    table = pd.read_csv(results_path / "paper_main_table.csv")

    required_methods = {CANONICAL_METHOD, SERVICE_AWARE_BASELINE}
    if not required_methods.issubset(set(raw["method"])):
        missing = sorted(required_methods - set(raw["method"]))
        raise ValueError(f"missing required reviewed-paper methods: {missing}")

    canonical = table[table.method == CANONICAL_METHOD]
    service = table[table.method == SERVICE_AWARE_BASELINE]
    if len(canonical) != 1 or len(service) != 1:
        raise ValueError("paper_main_table.csv must contain one CIN and one ServiceAware row")
    canonical = canonical.iloc[0]
    service = service.iloc[0]
    canonical_raw = raw[raw.method == CANONICAL_METHOD]
    condition_rows = canonical_raw[PAIRING_KEYS].drop_duplicates()
    true_graph_rows = condition_rows[["seed", "size", "density"]].drop_duplicates()
    action_budgets = sorted(int(value) for value in condition_rows.action_budget.unique())

    density_rows: list[dict[str, object]] = []
    for density in DENSITIES:
        for method in [CANONICAL_METHOD, SERVICE_AWARE_BASELINE]:
            group = raw[(raw.method == method) & (raw.density == density)]
            if group.empty:
                raise ValueError(f"missing density rows for {method}/{density}")
            accepted = group[group.certificate_pass > 0.5]
            density_rows.append(
                {
                    "density": density,
                    "method": method,
                    "isolation": group.attack_pair_isolation_true.mean(),
                    "coverage": group.certificate_pass.mean(),
                    "precision": accepted.true_feasible.mean() if len(accepted) else float("nan"),
                    "accepted": len(accepted),
                }
            )
    density_frame = pd.DataFrame(density_rows)

    def density_row(density: str, method: str) -> pd.Series:
        subset = density_frame[
            (density_frame.density == density) & (density_frame.method == method)
        ]
        if len(subset) != 1:
            raise ValueError(f"expected one density aggregate for {density}/{method}")
        return subset.iloc[0]

    cin_low = density_row("low", CANONICAL_METHOD)
    cin_medium = density_row("medium", CANONICAL_METHOD)
    cin_high = density_row("high", CANONICAL_METHOD)
    svc_high = density_row("high", SERVICE_AWARE_BASELINE)
    if int(cin_high.accepted) != 0:
        raise ValueError("reviewed density statement assumes zero high-density CIN acceptances")

    macros = "\n".join(
        [
            "% Auto-generated from frozen results_full CSVs; do not edit by hand.",
            (
                "\\providecommand{\\CanonicalGridDesignFinding}{Across "
                f"{_tex_int(condition_rows.seed.nunique())} seed clusters, "
                f"{_tex_int(len(true_graph_rows))} generated true graphs span "
                f"{_tex_int(condition_rows['size'].nunique())} size tiers and "
                f"{_tex_int(condition_rows.density.nunique())} density levels; crossing "
                f"{_tex_int(condition_rows.noise.nunique())} noise levels with "
                f"{_tex_int(condition_rows.hidden.nunique())} hidden-edge levels yields "
                f"{_tex_int(len(condition_rows))} observed-condition snapshots. "
                "The size-tier action budgets are "
                f"{'/'.join(str(value) for value in action_budgets)}.}}"
            ),
            (
                "\\providecommand{\\CanonicalAbstractFinding}{On "
                f"{_tex_int(len(canonical_raw))} synthetic shared-broker snapshots, the stronger "
                "ServiceAwareGreedyCut proposer exceeds the illustrative CIN-CertifiedGreedy proposer "
                f"in hidden pair isolation ({_fmt(service.attack_pair_isolation_true_mean)} versus "
                f"{_fmt(canonical.attack_pair_isolation_true_mean)}) and hidden-state feasibility among admitted actions "
                f"({_fmt(service.true_safe_precision_among_accepted)} versus "
                f"{_fmt(canonical.true_safe_precision_among_accepted)}). CIN observed-model admission rate falls from "
                f"{_fmt(cin_low.coverage)} at low density to {_fmt(cin_medium.coverage)} at medium density "
                f"and zero at high density; medium-density hidden-state feasibility among admitted actions is {_fmt(cin_medium.precision)}.}}"
            ),
            (
                "\\providecommand{\\CanonicalProposalLevelsFinding}{ServiceAwareGreedyCut is plainly the stronger proposer in this constructed benchmark: "
                f"it achieves {_fmt(service.attack_pair_isolation_true_mean)} versus CIN's {_fmt(canonical.attack_pair_isolation_true_mean)} hidden pair isolation, "
                f"{_fmt(service.service_pair_preservation_true_mean)} versus {_fmt(canonical.service_pair_preservation_true_mean)} service preservation, "
                f"{_fmt(service.certificate_coverage)} versus {_fmt(canonical.certificate_coverage)} observed-model admission rate, "
                f"{_fmt(service.true_feasible_rate)} versus {_fmt(canonical.true_feasible_rate)} hidden true-feasible rate, "
                f"{_fmt(service.true_safe_precision_among_accepted)} versus {_fmt(canonical.true_safe_precision_among_accepted)} hidden-state feasibility among admitted actions, "
                f"and lower mean action cost ({_fmt(service.action_cost_mean, 2)} versus {_fmt(canonical.action_cost_mean, 2)}).}}"
            ),
            (
                "\\providecommand{\\CanonicalDensityFinding}{CIN's low-density observed-model admission rate and hidden-state feasibility among admitted actions are "
                f"{_fmt(cin_low.coverage)} and {_fmt(cin_low.precision)}, but its medium-density observed-model admission rate is "
                f"{_fmt(cin_medium.coverage)} and hidden-state feasibility among admitted actions is only {_fmt(cin_medium.precision)}; "
                "at high density CIN admits no actions, so hidden-state feasibility among admitted actions is undefined rather than zero. "
                f"ServiceAware has high-density observed-model admission rate {_fmt(svc_high.coverage)} and hidden-state feasibility among admitted actions {_fmt(svc_high.precision)}, "
                "and otherwise shows the same sharp density-dependent decline.}"
            ),
        ]
    )
    _write_text(destination_path / "reviewed_result_macros.tex", macros)

    main_lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Pair isolation & APR (cutoff 8) & Service & Obs. admission & Admitted hidden feas. & Cost \\\\",
        "\\midrule",
    ]
    for _, row_value in table.sort_values("method").iterrows():
        main_lines.append(
            f"{_latex_method(row_value.method)} & {_fmt(row_value.attack_pair_isolation_true_mean)} & "
            f"{_fmt(row_value.attack_path_reduction_true_mean)} & {_fmt(row_value.service_pair_preservation_true_mean)} & "
            f"{_fmt(row_value.certificate_coverage)} & {_fmt(row_value.true_safe_precision_among_accepted)} & "
            f"{_fmt(row_value.action_cost_mean, 2)} \\\\"
        )
    main_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(destination_path / "main_results_reviewed.tex", "\n".join(main_lines))

    density_lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Density & Proposer & Hidden iso. & Obs. admission & Admitted hidden feas. \\\\",
        "\\midrule",
    ]
    for density in DENSITIES:
        for method, label in [
            (CANONICAL_METHOD, "CIN"),
            (SERVICE_AWARE_BASELINE, "ServiceAware"),
        ]:
            aggregate = density_row(density, method)
            density_lines.append(
                f"{density.title()} & {label} & {_fmt(aggregate.isolation)} & "
                f"{_fmt(aggregate.coverage)} & {_fmt(aggregate.precision)} \\\\"
            )
    density_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(destination_path / "main_density_compact.tex", "\n".join(density_lines))

    robustness_rows = []
    for (method, noise), group in raw.groupby(["method", "noise"], sort=True):
        accepted = group[group.certificate_pass > 0.5]
        robustness_rows.append(
            {
                "method": method,
                "noise": noise,
                "isolation": group.attack_pair_isolation_true.mean(),
                "service": group.service_pair_preservation_true.mean(),
                "true_feasible": group.true_feasible.mean(),
                "coverage": group.certificate_pass.mean(),
                "precision": accepted.true_feasible.mean() if len(accepted) else float("nan"),
            }
        )
    robust_lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Noise & Pair isolation & Service & True feasible & Obs. admission & Admitted hidden feas. \\\\",
        "\\midrule",
    ]
    for row_value in robustness_rows:
        robust_lines.append(
            f"{_latex_method(str(row_value['method']))} & {_fmt(row_value['noise'], 1)} & "
            f"{_fmt(row_value['isolation'])} & {_fmt(row_value['service'])} & "
            f"{_fmt(row_value['true_feasible'])} & {_fmt(row_value['coverage'])} & "
            f"{_fmt(row_value['precision'])} \\\\"
        )
    robust_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(destination_path / "robustness_results_reviewed.tex", "\n".join(robust_lines))

    scalability = raw.groupby(["method", "size"], as_index=False).agg(
        nodes=("nodes", "mean"),
        runtime_s=("runtime_s", "mean"),
        policy_size=("policy_size", "mean"),
        capsule_bytes=("capsule_json_bytes", "mean"),
        witness_bytes=("essential_witness_json_bytes", "mean"),
        replay_s=("serialized_replay_verifier_s", "mean"),
    )
    scalability["_size_order"] = scalability["size"].map(
        {value: index for index, value in enumerate(SIZES)}
    )
    scalability = scalability.sort_values(["method", "_size_order"])
    scalability_lines = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Method & Size & Nodes & Runtime (s) & Actions & Capsule (KiB) & Witness (KiB) & Replay (ms) \\\\",
        "\\midrule",
    ]
    for _, row_value in scalability.iterrows():
        scalability_lines.append(
            f"{_latex_method(row_value.method)} & {row_value['size']} & {_fmt(row_value.nodes, 0)} & "
            f"{_fmt(row_value.runtime_s, 4)} & {_fmt(row_value.policy_size, 2)} & "
            f"{_fmt(row_value.capsule_bytes / 1024, 1)} & {_fmt(row_value.witness_bytes / 1024, 1)} & "
            f"{_fmt(1000 * row_value.replay_s, 2)} \\\\"
        )
    scalability_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(destination_path / "scalability_table_reviewed.tex", "\n".join(scalability_lines))

    generated = sorted(destination_path.glob("*.tex"))
    manifest = {
        "schema": "cin.reviewed-paper-derivatives.v1",
        "inputs": {
            "raw_results.csv": _file_sha256(results_path / "raw_results.csv"),
            "paper_main_table.csv": _file_sha256(results_path / "paper_main_table.csv"),
        },
        "outputs": {path.name: _file_sha256(path) for path in generated},
        "legacy_machine_key": LEGACY_WEIGHTED_RANDOM_ROLLOUTS,
        "display_name": "Best-of-8 Weighted Random Rollouts",
    }
    _write_text(
        destination_path / "reviewed_table_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True),
    )
    return destination_path


def write_latex_fragments(output: str, paper_tables_dir: str | None = None) -> Path:
    """Generate manuscript fragments from canonical CSVs; never hard-code results."""

    output_path = Path(output)
    latex_dir = output_path / "latex"
    raw = pd.read_csv(output_path / "raw_results.csv")
    table = pd.read_csv(output_path / "paper_main_table.csv")
    tests = pd.read_csv(output_path / "statistical_tests.csv")
    ablations = pd.read_csv(output_path / "ablation_results.csv")
    oracle = pd.read_csv(output_path / "oracle_diagnostics.csv")
    budget_sensitivity = pd.read_csv(output_path / "budget_sensitivity.csv")
    observation_summary_path = output_path / "benchmark_observation_summary.csv"
    topology_summary_path = output_path / "topology_stress_summary.csv"
    observation_summary = (
        pd.read_csv(observation_summary_path)
        if observation_summary_path.exists()
        else pd.DataFrame()
    )
    topology_summary = (
        pd.read_csv(topology_summary_path)
        if topology_summary_path.exists()
        else pd.DataFrame()
    )
    conditions = raw[PAIRING_KEYS].drop_duplicates()
    factorial_settings = conditions.drop(columns=["seed", "action_budget"]).drop_duplicates()
    true_graphs = conditions.drop(columns=["noise", "hidden", "action_budget"]).drop_duplicates()
    seeds = raw.seed.nunique()
    canonical = table[table.method == CANONICAL_METHOD].iloc[0]
    baseline = table[table.method == SERVICE_AWARE_BASELINE].iloc[0]
    comparison = tests[
        (tests.baseline == SERVICE_AWARE_BASELINE) & (tests.metric == "attack_pair_isolation_true")
    ].iloc[0]
    apr_comparison = tests[
        (tests.baseline == SERVICE_AWARE_BASELINE) & (tests.metric == "attack_path_reduction_true")
    ].iloc[0]
    service_comparison = tests[
        (tests.baseline == SERVICE_AWARE_BASELINE) & (tests.metric == "service_pair_preservation_true")
    ].iloc[0]
    maximum_noise = raw.noise.max()
    robust = raw[(raw.method == CANONICAL_METHOD) & (raw.noise == maximum_noise)]
    maximum_hidden = raw.hidden.max()
    robust_max_hidden = robust[robust.hidden == maximum_hidden]
    robust_accepted = robust[robust.certificate_pass > 0.5]
    robust_max_hidden_accepted = robust_max_hidden[robust_max_hidden.certificate_pass > 0.5]
    robust_accepted_precision = (
        robust_accepted.true_feasible.mean() if len(robust_accepted) else float("nan")
    )
    robust_max_hidden_accepted_precision = (
        robust_max_hidden_accepted.true_feasible.mean()
        if len(robust_max_hidden_accepted)
        else float("nan")
    )
    canonical_raw = raw[raw.method == CANONICAL_METHOD]
    broker_joint_admissible = canonical_raw[
        canonical_raw.broker_boundary_admissible_under_budget_observed > 0.5
    ]
    broker_joint_inadmissible = canonical_raw[
        canonical_raw.broker_boundary_admissible_under_budget_observed <= 0.5
    ]
    cin_accept_given_broker_joint = (
        broker_joint_admissible.certificate_pass.mean()
        if len(broker_joint_admissible)
        else float("nan")
    )
    cin_accept_given_broker_complement = (
        broker_joint_inadmissible.certificate_pass.mean()
        if len(broker_joint_inadmissible)
        else float("nan")
    )
    runtime_by_size = raw[raw.method == CANONICAL_METHOD].groupby("size").runtime_s.mean()
    oracle_summary = oracle.groupby("method", as_index=False).agg(
        residual_pairs=("residual_attack_pairs_observed", "mean"),
        gap=("optimality_gap_residual_attack_pairs_observed", "mean"),
        candidates=("oracle_candidate_count", "mean"),
    )
    oracle_cin = oracle_summary[oracle_summary.method == CANONICAL_METHOD].iloc[0]
    topology_shared_cin = (
        topology_summary[
            (topology_summary.topology_regime == "shared_broker")
            & (topology_summary.method == CANONICAL_METHOD)
        ].iloc[0]
        if not topology_summary.empty
        else None
    )
    topology_alternate_cin = (
        topology_summary[
            (topology_summary.topology_regime == "alternate_egress")
            & (topology_summary.method == CANONICAL_METHOD)
        ].iloc[0]
        if not topology_summary.empty
        else None
    )
    maximum_observation_row = (
        observation_summary.sort_values(["noise", "hidden"]).iloc[-1]
        if not observation_summary.empty
        else None
    )
    accepted_comparator = (
        "hidden-state feasibility among admitted actions was undefined because no actions were admitted in this run"
        if pd.isna(canonical.true_safe_precision_among_accepted)
        else f"hidden-state feasibility among admitted actions was {_fmt_percent(canonical.true_safe_precision_among_accepted)}"
    )
    rejected_value = canonical.true_feasible_rate_among_rejected
    rejected_comparator = (
        "no capsules were rejected in this run"
        if pd.isna(rejected_value)
        else (
            "no rejected proposal met that criterion"
            if float(rejected_value) == 0.0
            else f"{_fmt_percent(rejected_value)} of rejected proposals met that criterion"
        )
    )

    macros = "\n".join(
        [
            f"\\providecommand{{\\CanonicalGridSummary}}{{{len(conditions):,} matched observed-condition snapshots derived from {len(true_graphs):,} generated true graphs ({len(factorial_settings)} observation-condition settings across {seeds} seed clusters), with fixed exogenous size-specific budgets.}}",
            (
                "\\providecommand{\\CanonicalMainFinding}{Compared with ServiceAwareGreedyCut, CIN-CertifiedGreedy changed mean exact hidden source--asset pair isolation by "
                f"{_fmt(comparison.mean_difference, signed=True)} (seed-clustered 95\\% CI [{_fmt(comparison.ci_low, signed=True)}, {_fmt(comparison.ci_high, signed=True)}], "
                f"Holm-adjusted $p={_fmt(comparison.p_value_holm, 4)}$) and changed true endpoint-service preservation by "
                f"{_fmt(service_comparison.mean_difference, signed=True)} (95\\% CI [{_fmt(service_comparison.ci_low, signed=True)}, {_fmt(service_comparison.ci_high, signed=True)}], "
                f"Holm-adjusted $p={_fmt(service_comparison.p_value_holm, 4)}$).}}"
            ),
            (
                "\\providecommand{\\CanonicalAPRSecondaryFinding}{For the secondary cutoff-8 attack-path-family metric, the paired difference was "
                f"{_fmt(apr_comparison.mean_difference, signed=True)} (seed-clustered 95\\% CI [{_fmt(apr_comparison.ci_low, signed=True)}, {_fmt(apr_comparison.ci_high, signed=True)}], "
                f"Holm-adjusted $p={_fmt(apr_comparison.p_value_holm, 4)}$).}}"
            ),
            (
                "\\providecommand{\\CanonicalAcceptanceFinding}{In this finite benchmark, CIN-CertifiedGreedy's observed-model admission rate was "
                f"{_fmt_percent(canonical.certificate_coverage)}; {accepted_comparator}, and {rejected_comparator}.}}"
            ),
            (
                "\\providecommand{\\CanonicalRobustnessFinding}{At the maximum evaluated telemetry noise "
                f"({_fmt(maximum_noise, 1)}), CIN-CertifiedGreedy achieved mean exact hidden source--asset pair isolation "
                f"{_fmt(robust.attack_pair_isolation_true.mean())}, true endpoint-service preservation "
                f"{_fmt(robust.service_pair_preservation_true.mean())}, hidden-graph true feasibility across all proposals "
                f"{_fmt(robust.true_feasible.mean())}, and hidden-state feasibility among admitted actions "
                f"{_fmt(robust_accepted_precision)}.  At maximum noise and maximum hidden-edge rate "
                f"({_fmt(maximum_hidden, 1)}), hidden-state feasibility among admitted actions was "
                f"{_fmt(robust_max_hidden_accepted_precision)}.}}"
            ),
            (
                "\\providecommand{\\CanonicalRuntimeFinding}{Mean CIN-CertifiedGreedy end-to-end model-fit, synthesis, and verification runtime ranged from "
                f"{_fmt(runtime_by_size.min())} to {_fmt(runtime_by_size.max())} seconds across graph sizes on the recorded CPU environment.}}"
            ),
            (
                "\\providecommand{\\CanonicalArtifactFinding}{The mean serialized CIN capsule was "
                f"{_fmt(canonical.capsule_json_bytes_mean / 1024, 1)} KiB, including "
                f"{_fmt(canonical.essential_witness_json_bytes_mean / 1024, 1)} KiB of essential witnesses; "
                f"separately timed serialized-bundle replay through the same verifier took {_fmt(1000 * canonical.serialized_replay_verifier_s_mean, 2)} ms on average, "
                f"with {_fmt_percent(canonical.serialized_replay_decision_match_rate)} agreement with the issuance decision.}}"
            ),
            (
                "\\providecommand{\\CanonicalBoundaryFinding}{Across all matched instances, the non-monitor broker-boundary control was within the action budget in "
                f"{_fmt_percent(canonical.broker_boundary_within_budget_rate)}, blocked the observed threat while preserving every required service endpoint pair in "
                f"{_fmt_percent(canonical.broker_boundary_threat_service_feasible_rate)}, and was jointly admissible under the budget in "
                f"{_fmt_percent(canonical.broker_boundary_admissible_under_budget_rate)}.  "
                f"$\\Pr(\\text{{CIN accept}}\\mid\\text{{broker jointly admissible}})$ was {_fmt_percent(cin_accept_given_broker_joint)}, versus "
                f"{_fmt_percent(cin_accept_given_broker_complement)} on its complement.  The all-source-egress control was threat--service feasible in "
                f"{_fmt_percent(canonical.source_boundary_threat_service_feasible_rate)} and jointly admissible in "
                f"{_fmt_percent(canonical.source_boundary_admissible_under_budget_rate)}.  CIN's mean per-instance action fraction was "
                f"{_fmt_percent(canonical.action_fraction_from_attack_source_mean)} attack-source-origin and "
                f"{_fmt_percent(canonical.action_fraction_from_broker_mean)} broker-origin.}}"
            ),
            (
                "\\providecommand{\\CanonicalOracleFinding}{In the separate $B=4$ small-instance diagnostic, the exact oracle's declared candidate pool contained "
                f"{_fmt(oracle_cin.candidates, 1)} edges on average and it exhaustively evaluated subsets of size at most four; CIN-CertifiedGreedy's mean observed residual-pair gap was "
                f"{_fmt(oracle_cin.gap)}. This is exact only within the disclosed at-most-18-edge pool, not a global optimality claim.}}"
            ),
            (
                "\\providecommand{\\CanonicalObservationAuditFinding}{At the maximum nominal observation-error setting, the realized optional-edge hidden fraction was "
                f"{_fmt(maximum_observation_row.realized_hidden_fraction_eligible) if maximum_observation_row is not None else '--'}, mean edge recall was "
                f"{_fmt(maximum_observation_row.edge_observation_recall) if maximum_observation_row is not None else '--'}, mean edge precision was "
                f"{_fmt(maximum_observation_row.edge_observation_precision) if maximum_observation_row is not None else '--'}, and service-specification coverage remained "
                f"{_fmt(maximum_observation_row.service_specification_coverage) if maximum_observation_row is not None else '--'}.}}"
            ),
            (
                "\\providecommand{\\CanonicalTopologyStressFinding}{In the separate alternate-egress stress diagnostic, CIN exact hidden pair isolation changed from "
                f"{_fmt(topology_shared_cin.attack_pair_isolation_true) if topology_shared_cin is not None else '--'} under the shared-broker regime to "
                f"{_fmt(topology_alternate_cin.attack_pair_isolation_true) if topology_alternate_cin is not None else '--'} with broker-bypassing egress, while true endpoint-service preservation changed from "
                f"{_fmt(topology_shared_cin.service_pair_preservation_true) if topology_shared_cin is not None else '--'} to "
                f"{_fmt(topology_alternate_cin.service_pair_preservation_true) if topology_alternate_cin is not None else '--'}.}}"
            ),
        ]
    )
    _write_text(latex_dir / "result_macros.tex", macros)

    main_lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Pair isolation & APR (cutoff 8) & Service & Obs. admission & Admitted hidden feas. & Cost \\\\",
        "\\midrule",
    ]
    for _, row in table.sort_values("method").iterrows():
        main_lines.append(
            f"{_latex_method(row.method)} & {_fmt(row.attack_pair_isolation_true_mean)} & {_fmt(row.attack_path_reduction_true_mean)} & "
            f"{_fmt(row.service_pair_preservation_true_mean)} & {_fmt(row.certificate_coverage)} & "
            f"{_fmt(row.true_safe_precision_among_accepted)} & {_fmt(row.action_cost_mean, 2)} \\\\"
        )
    main_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(latex_dir / "main_results.tex", "\n".join(main_lines))

    robustness_rows = []
    for (method, noise), group in raw.groupby(["method", "noise"], sort=True):
        accepted = group[group.certificate_pass > 0.5]
        robustness_rows.append(
            {
                "method": method,
                "noise": noise,
                "attack_pair_isolation_true": group.attack_pair_isolation_true.mean(),
                "service_pair_preservation_true": group.service_pair_preservation_true.mean(),
                "true_feasible": group.true_feasible.mean(),
                "certificate_coverage": group.certificate_pass.mean(),
                "true_feasible_precision": (
                    accepted.true_feasible.mean() if len(accepted) else float("nan")
                ),
            }
        )
    robustness = pd.DataFrame(robustness_rows)
    robust_lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Noise & Pair isolation & Service & True feasible & Obs. admission & Admitted hidden feas. \\\\",
        "\\midrule",
    ]
    for _, row in robustness.iterrows():
        robust_lines.append(
            f"{_latex_method(row.method)} & {_fmt(row.noise, 1)} & {_fmt(row.attack_pair_isolation_true)} & "
            f"{_fmt(row.service_pair_preservation_true)} & {_fmt(row.true_feasible)} & "
            f"{_fmt(row.certificate_coverage)} & {_fmt(row.true_feasible_precision)} \\\\"
        )
    robust_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(latex_dir / "robustness_results.tex", "\n".join(robust_lines))

    boundary_rows = []
    for density in DENSITIES:
        group = canonical_raw[canonical_raw.density == density]
        if group.empty:
            continue
        boundary_rows.append(
            {
                "density": density,
                "threat_service": group.broker_boundary_threat_service_feasible_observed.mean(),
                "within_budget": group.broker_boundary_within_budget.mean(),
                "joint_admissible": group.broker_boundary_admissible_under_budget_observed.mean(),
                "certificate_coverage": group.certificate_pass.mean(),
            }
        )
    boundary_lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Density & Broker threat+service & Within $B$ & Joint admissibility & CIN obs. admission \\\\",
        "\\midrule",
    ]
    for row in boundary_rows:
        boundary_lines.append(
            f"{row['density']} & {_fmt(row['threat_service'])} & {_fmt(row['within_budget'])} & "
            f"{_fmt(row['joint_admissible'])} & {_fmt(row['certificate_coverage'])} \\\\"
        )
    boundary_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(latex_dir / "boundary_controls.tex", "\n".join(boundary_lines))

    selectivity_rows = []
    selected_methods = raw[raw.method.isin([CANONICAL_METHOD, SERVICE_AWARE_BASELINE])]
    for (method, hidden), group in selected_methods.groupby(["method", "hidden"], sort=True):
        accepted = group[group.certificate_pass > 0.5]
        selectivity_rows.append(
            {
                "method": method,
                "hidden": hidden,
                "certificate_coverage": group.certificate_pass.mean(),
                "true_feasible_precision": (
                    accepted.true_feasible.mean() if len(accepted) else float("nan")
                ),
                "true_feasible": group.true_feasible.mean(),
            }
        )
    selectivity_lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Hidden rate & Obs. admission & Admitted hidden feas. & True feasible \\\\",
        "\\midrule",
    ]
    for row in selectivity_rows:
        selectivity_lines.append(
            f"{_latex_method(row['method'])} & {_fmt(row['hidden'], 1)} & "
            f"{_fmt(row['certificate_coverage'])} & {_fmt(row['true_feasible_precision'])} & "
            f"{_fmt(row['true_feasible'])} \\\\"
        )
    selectivity_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(latex_dir / "selectivity_by_hidden.tex", "\n".join(selectivity_lines))

    selectivity_compact_lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Hidden & \\multicolumn{2}{c}{CIN} & \\multicolumn{2}{c}{Service-aware} \\\\",
        "rate & Admission & Hidden feas. & Admission & Hidden feas. \\\\",
        "\\midrule",
    ]
    for hidden in sorted(selected_methods.hidden.unique()):
        cin_group = selected_methods[
            (selected_methods.method == CANONICAL_METHOD) & (selected_methods.hidden == hidden)
        ]
        service_group = selected_methods[
            (selected_methods.method == SERVICE_AWARE_BASELINE)
            & (selected_methods.hidden == hidden)
        ]
        if cin_group.empty or service_group.empty:
            raise ValueError(f"incomplete compact selectivity rows for hidden={hidden}")
        cin_accepted = cin_group[cin_group.certificate_pass > 0.5]
        service_accepted = service_group[service_group.certificate_pass > 0.5]
        selectivity_compact_lines.append(
            f"{_fmt(hidden, 1)} & {_fmt(cin_group.certificate_pass.mean())} & "
            f"{_fmt(cin_accepted.true_feasible.mean() if len(cin_accepted) else float('nan'))} & "
            f"{_fmt(service_group.certificate_pass.mean())} & "
            f"{_fmt(service_accepted.true_feasible.mean() if len(service_accepted) else float('nan'))} \\\\"
        )
    selectivity_compact_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(
        latex_dir / "main_selectivity_compact.tex",
        "\n".join(selectivity_compact_lines),
    )

    ablation_table = ablations.groupby("ablation", as_index=False)[
        ["attack_pair_isolation_true", "service_pair_preservation_true", "certificate_pass", "true_feasible"]
    ].mean()
    ablation_lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Variant & Pair isolation & Service & Obs. admission & True feasible \\\\",
        "\\midrule",
    ]
    for _, row in ablation_table.iterrows():
        ablation_lines.append(
            f"{_latex_method(row.ablation)} & {_fmt(row.attack_pair_isolation_true)} & {_fmt(row.service_pair_preservation_true)} & "
            f"{_fmt(row.certificate_pass)} & {_fmt(row.true_feasible)} \\\\"
        )
    ablation_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(latex_dir / "ablation_results.tex", "\n".join(ablation_lines))

    scalability = raw.groupby(["method", "size"], as_index=False).agg(
        nodes=("nodes", "mean"),
        runtime_s=("runtime_s", "mean"),
        policy_size=("policy_size", "mean"),
        capsule_bytes=("capsule_json_bytes", "mean"),
        witness_bytes=("essential_witness_json_bytes", "mean"),
        replay_s=("serialized_replay_verifier_s", "mean"),
    )
    scalability["_size_order"] = scalability["size"].map(
        {value: index for index, value in enumerate(SIZES)}
    )
    scalability = scalability.sort_values(["method", "_size_order"]).drop(columns="_size_order")
    scalability_lines = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Method & Size & Nodes & Runtime (s) & Actions & Capsule (KiB) & Witness (KiB) & Replay (ms) \\\\",
        "\\midrule",
    ]
    for _, row in scalability.iterrows():
        scalability_lines.append(
            f"{_latex_method(row.method)} & {row['size']} & {_fmt(row.nodes, 0)} & {_fmt(row.runtime_s, 4)} & "
            f"{_fmt(row.policy_size, 2)} & {_fmt(row.capsule_bytes / 1024, 1)} & "
            f"{_fmt(row.witness_bytes / 1024, 1)} & {_fmt(1000 * row.replay_s, 2)} \\\\"
        )
    scalability_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(latex_dir / "scalability_table.tex", "\n".join(scalability_lines))

    scalability_compact_lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Size & Nodes & Run (s) & Capsule & Witness & Replay \\\\",
        " & & & (KiB) & (KiB) & (ms) \\\\",
        "\\midrule",
    ]
    canonical_scalability = scalability[scalability.method == CANONICAL_METHOD].set_index("size")
    for size in SIZES:
        if size not in canonical_scalability.index:
            continue
        row = canonical_scalability.loc[size]
        scalability_compact_lines.append(
            f"{size.title()} & {_fmt(row.nodes, 0)} & {_fmt(row.runtime_s, 3)} & "
            f"{_fmt(row.capsule_bytes / 1024, 1)} & {_fmt(row.witness_bytes / 1024, 1)} & "
            f"{_fmt(1000 * row.replay_s, 2)} \\\\"
        )
    scalability_compact_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(
        latex_dir / "main_scalability_compact.tex",
        "\n".join(scalability_compact_lines),
    )

    oracle_lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Observed residual pairs & Gap to pool oracle & Pool size \\\\",
        "\\midrule",
    ]
    for _, row in oracle_summary.sort_values("method").iterrows():
        oracle_lines.append(
            f"{_latex_method(row.method)} & {_fmt(row.residual_pairs)} & {_fmt(row.gap)} & {_fmt(row.candidates, 1)} \\\\"
        )
    oracle_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(latex_dir / "oracle_results.tex", "\n".join(oracle_lines))

    oracle_labels = {
        CANONICAL_METHOD: "CIN",
        "ExactBoundedCandidateOracle": "Pool oracle",
        SERVICE_AWARE_BASELINE: "Service-aware",
    }
    oracle_compact_lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Residual pairs & Gap & Pool \\\\",
        "\\midrule",
    ]
    for method in (CANONICAL_METHOD, "ExactBoundedCandidateOracle", SERVICE_AWARE_BASELINE):
        subset = oracle_summary[oracle_summary.method == method]
        if len(subset) != 1:
            raise ValueError(f"expected one compact oracle row for {method}, found {len(subset)}")
        row = subset.iloc[0]
        oracle_compact_lines.append(
            f"{oracle_labels[method]} & {_fmt(row.residual_pairs)} & {_fmt(row.gap)} & "
            f"{_fmt(row.candidates, 0)} \\\\"
        )
    oracle_compact_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(latex_dir / "main_oracle_compact.tex", "\n".join(oracle_compact_lines))

    budget_rows = []
    budget_group_keys = ["size", "budget_regime", "method", "action_budget"]
    for keys, group in budget_sensitivity.groupby(budget_group_keys, sort=False):
        size, regime, method, action_budget = keys
        budget_rows.append(
            {
                "size": size,
                "budget_regime": regime,
                "method": method,
                "action_budget": action_budget,
                "attack_pair_isolation_true": group.attack_pair_isolation_true.mean(),
                "service_pair_preservation_true": group.service_pair_preservation_true.mean(),
                "true_feasible": group.true_feasible.mean(),
                "certificate_coverage": group.certificate_pass.mean(),
            }
        )
    size_order = {value: index for index, value in enumerate(SIZES)}
    regime_order = {"primary": 0, "high_budget_sensitivity": 1}
    budget_rows.sort(
        key=lambda row: (
            size_order.get(row["size"], len(size_order)),
            regime_order.get(row["budget_regime"], len(regime_order)),
            row["method"],
        )
    )
    budget_lines = [
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        "Size & Method & Regime & $B$ & Pair isolation & Service & True feasible & Obs. admission \\\\",
        "\\midrule",
    ]
    for row in budget_rows:
        budget_lines.append(
            f"{row['size']} & {_latex_method(row['method'])} & {_latex_method(row['budget_regime'])} & "
            f"{_fmt(row['action_budget'], 0)} & {_fmt(row['attack_pair_isolation_true'])} & "
            f"{_fmt(row['service_pair_preservation_true'])} & {_fmt(row['true_feasible'])} & "
            f"{_fmt(row['certificate_coverage'])} \\\\"
        )
    budget_lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_text(latex_dir / "budget_sensitivity.tex", "\n".join(budget_lines))

    if not observation_summary.empty:
        observation_lines = [
            "\\begin{tabular}{rrrrrr}",
            "\\toprule",
            "Noise & Hidden & Realized hidden & Edge recall & Edge precision & Service-spec. cov. \\\\",
            "\\midrule",
        ]
        for _, row in observation_summary.iterrows():
            observation_lines.append(
                f"{_fmt(row.noise, 1)} & {_fmt(row.hidden, 1)} & "
                f"{_fmt(row.realized_hidden_fraction_eligible)} & {_fmt(row.edge_observation_recall)} & "
                f"{_fmt(row.edge_observation_precision)} & {_fmt(row.service_specification_coverage)} \\\\"
            )
        observation_lines.extend(["\\bottomrule", "\\end{tabular}"])
        _write_text(latex_dir / "observation_audit.tex", "\n".join(observation_lines))

    if not topology_summary.empty:
        topology_lines = [
            "\\begin{tabular}{llrrrrr}",
            "\\toprule",
            "Regime & Method & Pair isolation & Service & Obs. admission & True feasible & Broker blocks \\\\",
            "\\midrule",
        ]
        for _, row in topology_summary.sort_values(["topology_regime", "method"]).iterrows():
            topology_lines.append(
                f"{_latex_method(row.topology_regime)} & {_latex_method(row.method)} & "
                f"{_fmt(row.attack_pair_isolation_true)} & {_fmt(row.service_pair_preservation_true)} & "
                f"{_fmt(row.certificate_coverage)} & {_fmt(row.true_feasible)} & "
                f"{_fmt(row.broker_boundary_attack_blocked)} \\\\"
            )
        topology_lines.extend(["\\bottomrule", "\\end{tabular}"])
        _write_text(latex_dir / "topology_stress.tex", "\n".join(topology_lines))

    if paper_tables_dir:
        destination = Path(paper_tables_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for source in latex_dir.glob("*.tex"):
            shutil.copy2(source, destination / source.name)
    return latex_dir


def run_all(
    seeds: int,
    output: str,
    quick: bool = False,
    jobs: int = 0,
    paper_tables_dir: str | None = None,
    started_at: datetime | None = None,
) -> None:
    started_at = started_at or datetime.now(timezone.utc)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    run_grid(seeds, output, quick=quick, jobs=jobs)
    auxiliary_seeds = max(1, min(seeds, 10)) if quick else max(3, min(seeds, 10))
    run_ablations(auxiliary_seeds, output, quick=quick, jobs=jobs)
    run_topology_stress(auxiliary_seeds, output, quick=quick, jobs=jobs)
    run_oracle_study(auxiliary_seeds, output, quick=quick, jobs=jobs)
    run_budget_sensitivity(seeds, output, quick=quick, jobs=jobs)
    write_statistical_report(output)
    write_ablation_statistical_report(output)
    write_topology_stress_statistical_report(output)
    write_paper_tables(output)
    write_latex_fragments(output, paper_tables_dir=paper_tables_dir)
    write_manifest(output, seeds=seeds, quick=quick, jobs=jobs, started_at=started_at)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_model() -> str:
    """Return a useful CPU label without recording host or user identifiers."""

    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.splitlines():
                if line.strip().startswith("Chip:"):
                    return line.split(":", 1)[1].strip()
        elif platform.system() == "Linux":
            cpuinfo = Path("/proc/cpuinfo")
            if cpuinfo.exists():
                for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.lower().startswith("model name") and ":" in line:
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine() or "unknown"


def write_manifest(
    output: str,
    *,
    seeds: int,
    quick: bool,
    jobs: int = 0,
    started_at: datetime | None = None,
    invocation: str | None = None,
    provenance_role: str = "generation",
    execution_manifest_path: str | None = None,
    derivative_refresh_note: str | None = None,
    persist: bool = True,
) -> Dict[str, object]:
    """Hash source, configuration, dependency lock, and generated artifacts.

    A post-run deterministic derivative refresh may preserve the original
    execution manifest under a second filename and bind it from a deliverable
    manifest.  This keeps exact execution provenance distinct from the final
    package/source integrity record.

    With ``persist=False``, construct and return the exact manifest candidate
    without modifying the result tree.  Release tooling uses this mode so all
    pre-existing integrity gates can run before installing refreshed metadata.
    """

    output_path = Path(output)
    source_root = Path(__file__).resolve().parents[1]
    source_files = sorted(
        path
        for pattern in (
            "cin/*.py",
            "scripts/*.py",
            "tests/*.py",
            "configs/*.yaml",
            "independent_checker/*.mjs",
            "data/**/*.json",
        )
        for path in source_root.glob(pattern)
    )
    source_files.extend(
        source_root / name
        for name in (
            "run_all.py",
            "requirements.txt",
            "requirements-lock.txt",
            "README.md",
            "CHANGELOG.md",
            "REPRODUCIBILITY.md",
            "PACKAGE_CONTENTS.md",
            "DATA_PROVENANCE.md",
            "RESULT_SEMANTICS.md",
            "INDEPENDENT_CHECKER_SCOPE.md",
            "CLAIM_EVIDENCE_MATRIX.md",
            "REVIEW_REVISION_CHANGELOG.md",
            "VALIDATION_REPORT.md",
            "THIRD_PARTY_NOTICES.md",
            "Makefile",
            "LICENSE",
        )
        if (source_root / name).exists()
    )
    source_hashes = {str(path.relative_to(source_root)): _file_sha256(path) for path in sorted(set(source_files))}
    source_tree_sha256 = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_files = sorted(
        path for path in output_path.rglob("*") if path.is_file() and path.name != "manifest.json"
    )
    output_hashes = {str(path.relative_to(output_path)): _file_sha256(path) for path in output_files}
    generated_at = datetime.now(timezone.utc)
    execution_manifest = None
    if execution_manifest_path is not None:
        relative_execution_manifest = Path(execution_manifest_path)
        if relative_execution_manifest.is_absolute() or ".." in relative_execution_manifest.parts:
            raise ValueError("execution manifest path must be relative to the output directory")
        resolved_execution_manifest = output_path / relative_execution_manifest
        if not resolved_execution_manifest.is_file():
            raise FileNotFoundError(f"missing preserved execution manifest: {resolved_execution_manifest}")
        execution_manifest = {
            "path": relative_execution_manifest.as_posix(),
            "sha256": _file_sha256(resolved_execution_manifest),
        }
    manifest = {
        "schema": "cin.canonical-artifact-manifest.v2",
        "provenance_role": provenance_role,
        "source_revision": f"source-tree-sha256:{source_tree_sha256}",
        "git_revision": None,
        "source_file_sha256": source_hashes,
        "dependency_lock": "requirements-lock.txt",
        "dependency_lock_sha256": _file_sha256(source_root / "requirements-lock.txt"),
        "started_at_utc": (started_at or generated_at).isoformat().replace("+00:00", "Z"),
        "generated_at_utc": generated_at.isoformat().replace("+00:00", "Z"),
        "seeds": seeds,
        "jobs": jobs,
        "invocation": invocation or (
            f"python run_all.py --seeds {seeds} --output <OUTPUT> "
            f"{'--quick ' if quick else ''}--jobs {jobs}"
        ),
        "main_seed_values": list(range(seeds)),
        "ablation_seed_offset": 4000,
        "oracle_seed_offset": 12000,
        "budget_sensitivity_seed_offset": 16000,
        "topology_stress_seed_offset": 20000,
        "quick": quick,
        "grid": {
            "sizes": SIZES if not quick else ["small"],
            "noise": NOISES if not quick else [0.0, 0.2],
            "hidden": HIDDENS if not quick else [0.0, 0.2],
            "density": DENSITIES if not quick else ["low", "high"],
            "topology_regime": "shared_broker",
        },
        "observation_model": {
            "hidden_definition": "fraction of eligible optional hidden-candidate edges, up to integer rounding",
            "required_edges_hidden_or_randomly_missed": False,
            "optional_edge_missing_probability": "min(0.25, 0.45 * noise)",
            "service_specification": "same exogenous endpoint-pair list in true and observed views",
        },
        "ablation_grid": {
            "seeds": min(seeds, 10) if not quick else min(seeds, 10),
            "sizes": SIZES if not quick else ["small"],
            "noise": NOISES if not quick else [0.2],
            "hidden": HIDDENS if not quick else [0.0, 0.2],
            "density": DENSITIES if not quick else ["medium"],
        },
        "topology_stress": {
            "seeds": min(seeds, 10),
            "regimes": ["shared_broker", "alternate_egress"],
            "diagnostic_only": True,
        },
        "inference": {
            "cluster_key": "seed",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "multiple_testing": "Holm family-wise adjustment",
        },
        "action_budgets": ACTION_BUDGETS,
        "high_budget_sensitivity": HIGH_BUDGET_SENSITIVITY,
        "path_cutoff_edges": PATH_CUTOFF,
        "oracle_diagnostic_budget": 4,
        "methods": sorted(pd.read_csv(output_path / "raw_results.csv").method.unique().tolist()),
        "output_file_sha256": output_hashes,
        "runtime_environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
        },
    }
    if execution_manifest is not None:
        manifest["preserved_execution_manifest"] = execution_manifest
    if derivative_refresh_note is not None:
        manifest["derivative_refresh_note"] = derivative_refresh_note
    if persist:
        with (output_path / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)
    return manifest

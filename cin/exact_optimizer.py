"""Full-candidate-universe exact intervention optimization with SciPy MILP.

Exact for the supplied observed directed graph, candidate universe, threat
endpoint pairs, service endpoint pairs, cardinality budget, and declared edge
costs. Equal primary optima are resolved to the lexicographically smallest
sorted intervention tuple. Threat isolation is encoded by directed cut-side
variables; required service preservation is encoded by unit flows over
surviving edges.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .capsule import DefenseCapsule, EXPERIMENT_MAX_CAPSULE_AGE_SECONDS, EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS, build_capsule, experiment_issuance_time
from .repair import identity_transformation
from .verifier import canonical_action_set, canonical_service_pairs
from .simulator import PathSearchScope, all_attack_routes_blocked, apply_interventions, normalize_service_pairs, reachable_pairs, service_preservation

Edge = Tuple[str, str]
Pair = Tuple[str, str]
LEXICOGRAPHIC_TIE_BLOCK_SIZE = 20


@dataclass(frozen=True)
class ExactOptimizationDiagnostics:
    status: str
    success: bool
    optimal: bool
    objective_value: float | None
    primal_objective: float | None
    dual_bound: float | None
    solver_runtime_s: float
    mip_gap: float | None
    node_count: int
    edge_count: int
    attack_pair_count: int
    service_pair_count: int
    action_budget: int
    selected_action_count: int
    isolated_pair_count: int
    residual_pair_count: int
    total_action_cost: float
    independently_recomputed_feasible: bool = False
    primary_model_attack_pair_count: int = 0
    primary_model_isolated_pair_count: int | None = None
    independently_recomputed_residual_pair_count: int = 0
    independent_action_domain_valid: bool = False
    independent_service_valid: bool = False
    independent_model_consistent: bool = False
    candidate_action_count: int = 0
    candidate_universe: str = "explicit_declared_directed_edges"
    formulation: str = "binary_directed_cut_plus_service_flow_with_canonical_edge_tie_break"
    tie_breaker: str = "lexicographically_smallest_sorted_intervention_tuple"
    tie_break_solve_count: int = 0

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ExactOptimizationResult:
    capsule: DefenseCapsule
    diagnostics: ExactOptimizationDiagnostics


def _pairs(graph: nx.DiGraph, sources: Iterable[str], targets: Iterable[str]) -> List[Pair]:
    return [
        (s, t)
        for s in sorted(set(map(str, sources)))
        for t in sorted(set(map(str, targets)))
        if s != t and s in graph and t in graph
    ]


def solve_full_universe_milp(
    graph: nx.DiGraph,
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    required_service_pairs: Sequence[Sequence[str]],
    *,
    action_budget: int,
    candidate_actions: Sequence[Edge],
    time_limit_s: float = 120.0,
    mip_rel_gap: float = 0.0,
    method: str = "ExactFullUniverseMILP",
) -> ExactOptimizationResult:
    """Optimize exactly over the explicitly declared candidate actions.

    Lexicographically: (i) maximize isolated source--asset pairs, (ii) minimize
    action count, (iii) minimize declared action cost, and (iv) choose the
    lexicographically smallest sorted intervention tuple. Every required
    service pair must retain at least one directed path.
    """
    if type(action_budget) is not int or action_budget < 0:
        raise ValueError("action_budget must be a non-negative integer")
    if time_limit_s <= 0:
        raise ValueError("time_limit_s must be positive")
    if any(bool(data.get("blocked", False)) for _, _, data in graph.edges(data=True)):
        raise ValueError(
            "exact optimizer requires a graph with no pre-blocked edges"
        )

    nodes = sorted(map(str, graph.nodes))
    edges: List[Edge] = sorted((str(u), str(v)) for u, v in graph.edges)
    candidates = canonical_action_set(candidate_actions, field="candidate_actions")
    if any(not graph.has_edge(*edge) for edge in candidates):
        raise ValueError("candidate_actions contains an edge absent from the graph")
    candidate_set = set(candidates)
    node_index = {node: i for i, node in enumerate(nodes)}
    attack_pairs = _pairs(graph, attack_sources, protected_assets)
    services = list(
        canonical_service_pairs(
            required_service_pairs, field="required_service_pairs"
        )
    )
    for pair in services:
        if pair[0] not in graph or pair[1] not in graph:
            raise ValueError(f"service pair {pair!r} references an absent node")

    m, n, p_count, r_count = len(edges), len(nodes), len(attack_pairs), len(services)
    x0 = 0
    z0 = m
    y0 = z0 + p_count
    f0 = y0 + p_count * n
    var_count = f0 + r_count * m

    integrality = np.zeros(var_count, dtype=np.int8)
    integrality[x0:f0] = 1
    upper_bounds = np.ones(var_count)
    for edge_index, edge in enumerate(edges):
        if edge not in candidate_set:
            upper_bounds[x0 + edge_index] = 0.0
    bounds = Bounds(np.zeros(var_count), upper_bounds)

    edge_costs: List[float] = []
    for source, destination in edges:
        value = graph[source][destination].get("action_cost", 1.0)
        if (
            type(value) not in {int, float}
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                "exact optimizer action costs must be finite non-negative numbers"
            )
        edge_costs.append(float(value))
    max_edge_cost = max(edge_costs, default=1.0)
    max_cost_sum = max(1.0, action_budget * max_edge_cost)
    action_weight = max_cost_sum + 1.0
    isolation_weight = (action_budget + 1.0) * action_weight + max_cost_sum + 1.0
    objective = np.zeros(var_count)
    for i, (u, v) in enumerate(edges):
        objective[x0 + i] = action_weight + edge_costs[i]
    objective[z0:y0] = -isolation_weight

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    lbs: List[float] = []
    ubs: List[float] = []
    row = 0

    def add(coeff: Dict[int, float], lb: float, ub: float) -> None:
        nonlocal row
        for col, value in coeff.items():
            if value:
                rows.append(row); cols.append(col); vals.append(float(value))
        lbs.append(float(lb)); ubs.append(float(ub)); row += 1

    add({x0 + i: 1.0 for i in range(m)}, -np.inf, float(action_budget))

    # z_p=1 requires a directed cut between source and target. A surviving
    # edge cannot cross from y=0 to y=1.
    for p, (source, target) in enumerate(attack_pairs):
        ybase = y0 + p * n
        zvar = z0 + p
        add({ybase + node_index[source]: 1.0}, 0.0, 0.0)
        add({ybase + node_index[target]: 1.0, zvar: -1.0}, 0.0, 0.0)
        for e, (u, v) in enumerate(edges):
            add({ybase + node_index[v]: 1.0, ybase + node_index[u]: -1.0, x0 + e: -1.0}, -np.inf, 0.0)

    outgoing: Dict[str, List[int]] = {node: [] for node in nodes}
    incoming: Dict[str, List[int]] = {node: [] for node in nodes}
    for e, (u, v) in enumerate(edges):
        outgoing[u].append(e); incoming[v].append(e)

    # A feasible unit flow witnesses each required service path.
    for r, (source, target) in enumerate(services):
        fbase = f0 + r * m
        for node in nodes:
            coeff: Dict[int, float] = {}
            for e in outgoing[node]: coeff[fbase + e] = coeff.get(fbase + e, 0.0) + 1.0
            for e in incoming[node]: coeff[fbase + e] = coeff.get(fbase + e, 0.0) - 1.0
            rhs = 1.0 if node == source else (-1.0 if node == target else 0.0)
            add(coeff, rhs, rhs)
        for e in range(m):
            add({fbase + e: 1.0, x0 + e: 1.0}, -np.inf, 1.0)

    def current_constraints() -> LinearConstraint:
        matrix = coo_matrix((vals, (rows, cols)), shape=(row, var_count)).tocsr()
        return LinearConstraint(matrix, np.asarray(lbs), np.asarray(ubs))

    started = perf_counter()
    primary_result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=current_constraints(),
        options={"time_limit": float(time_limit_s), "mip_rel_gap": float(mip_rel_gap), "presolve": True},
    )

    status_map = {0: "optimal", 1: "limit_reached", 2: "infeasible", 3: "unbounded", 4: "solver_error"}
    primary_status = status_map.get(
        int(primary_result.status), f"status_{primary_result.status}"
    )
    selected_result = primary_result
    tie_break_solve_count = 0
    tie_break_status: str | None = None

    # The primary objective intentionally leaves tied edge sets unresolved.
    # HiGHS may choose different tied optima across platforms, which can alter
    # hidden-graph diagnostics and learned labels even though the declared
    # observed-graph objective is unchanged.  Fix the three primary objective
    # levels, then encode exact lexicographic inclusion in modest 20-bit blocks.
    # Earlier sorted edges receive the most-significant bit.  Fixing each
    # block's optimum before moving to the next yields one canonical x vector
    # without numerically unsafe whole-universe powers of two.
    if int(primary_result.status) == 0 and primary_result.x is not None and m:
        primary_selected_indices = [
            i for i in range(m) if float(primary_result.x[x0 + i]) >= 0.5
        ]
        isolated_optimum = sum(
            float(primary_result.x[z0 + i]) >= 0.5 for i in range(p_count)
        )
        action_count_optimum = len(primary_selected_indices)
        action_cost_optimum = sum(edge_costs[i] for i in primary_selected_indices)
        add({z0 + i: 1.0 for i in range(p_count)}, isolated_optimum, isolated_optimum)
        add({x0 + i: 1.0 for i in range(m)}, action_count_optimum, action_count_optimum)
        add(
            {x0 + i: edge_costs[i] for i in range(m)},
            action_cost_optimum,
            action_cost_optimum,
        )

        for block_start in range(0, m, LEXICOGRAPHIC_TIE_BLOCK_SIZE):
            block = list(
                range(
                    block_start,
                    min(block_start + LEXICOGRAPHIC_TIE_BLOCK_SIZE, m),
                )
            )
            weights = [1 << (len(block) - position - 1) for position in range(len(block))]
            tie_objective = np.zeros(var_count)
            for edge_index, weight in zip(block, weights):
                tie_objective[x0 + edge_index] = -float(weight)
            elapsed = perf_counter() - started
            remaining = float(time_limit_s) - elapsed
            if remaining <= 0:
                tie_break_status = "limit_reached"
                break
            tie_result = milp(
                c=tie_objective,
                integrality=integrality,
                bounds=bounds,
                constraints=current_constraints(),
                options={
                    "time_limit": remaining,
                    "mip_rel_gap": float(mip_rel_gap),
                    "presolve": True,
                },
            )
            tie_break_solve_count += 1
            if int(tie_result.status) != 0 or tie_result.x is None:
                tie_break_status = status_map.get(
                    int(tie_result.status), f"status_{tie_result.status}"
                )
                break
            selected_result = tie_result
            block_code = int(
                round(
                    sum(
                        weight * float(tie_result.x[x0 + edge_index])
                        for edge_index, weight in zip(block, weights)
                    )
                )
            )
            add(
                {
                    x0 + edge_index: float(weight)
                    for edge_index, weight in zip(block, weights)
                },
                block_code,
                block_code,
            )

    runtime = perf_counter() - started
    if tie_break_status is None:
        status = primary_status
    else:
        status = f"tie_break_{tie_break_status}"
    selected = (
        []
        if selected_result.x is None
        else [
            edge
            for i, edge in enumerate(edges)
            if float(selected_result.x[x0 + i]) >= 0.5
        ]
    )

    capsule = build_capsule(
        method=method,
        graph=graph,
        proposal_transformation=identity_transformation(selected),
        candidate_actions=candidates,
        attack_sources=attack_sources,
        protected_assets=protected_assets,
        required_service_pairs=services,
        action_budget=action_budget,
        tau=1.0,
        path_scope=PathSearchScope(),
        issued_at=experiment_issuance_time(graph),
        max_snapshot_age_at_issue_seconds=EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
        max_capsule_age_seconds=EXPERIMENT_MAX_CAPSULE_AGE_SECONDS,
    )
    after = apply_interventions(graph, selected)
    before_pairs = reachable_pairs(graph, attack_sources, protected_assets)
    residual = reachable_pairs(after, attack_sources, protected_assets)
    mip_gap_value = getattr(primary_result, "mip_gap", None)
    dual_bound_value = getattr(primary_result, "mip_dual_bound", None)
    independent_action_domain_valid = (
        len(selected) <= action_budget
        and all(edge in candidate_set for edge in selected)
    )
    independent_service_valid = (
        service_preservation(after, services) >= 1.0 - 1e-12
    )
    independent_attack_valid = all_attack_routes_blocked(
        after, attack_sources, protected_assets
    )
    independent_feasible = (
        independent_action_domain_valid
        and independent_attack_valid
        and independent_service_valid
    )
    primary_model_isolated_pair_count = (
        None
        if selected_result.x is None
        else sum(
            float(selected_result.x[z0 + index]) >= 0.5
            for index in range(p_count)
        )
    )
    independent_model_consistent = (
        primary_model_isolated_pair_count is not None
        and independent_action_domain_valid
        and independent_service_valid
        # The MILP has one z variable for every Cartesian source/asset pair,
        # including pairs already unreachable before intervention. Compare its
        # isolation count with that same universe, not only with initially
        # reachable pairs used by the historical improvement diagnostic.
        and primary_model_isolated_pair_count == len(attack_pairs) - len(residual)
    )
    diagnostics = ExactOptimizationDiagnostics(
        status=status,
        success=bool(primary_result.success) and tie_break_status is None,
        optimal=int(primary_result.status) == 0 and tie_break_status is None,
        objective_value=(
            float(primary_result.fun) if primary_result.fun is not None else None
        ),
        primal_objective=(
            float(primary_result.fun) if primary_result.fun is not None else None
        ),
        dual_bound=(
            float(dual_bound_value) if dual_bound_value is not None else None
        ),
        solver_runtime_s=runtime,
        mip_gap=float(mip_gap_value) if mip_gap_value is not None else None,
        node_count=n,
        edge_count=m,
        attack_pair_count=len(before_pairs),
        service_pair_count=r_count,
        action_budget=action_budget,
        selected_action_count=len(selected),
        isolated_pair_count=len(before_pairs) - len(residual),
        residual_pair_count=len(residual),
        total_action_cost=sum(float(graph[u][v].get("action_cost", 1.0)) for u, v in selected),
        independently_recomputed_feasible=independent_feasible,
        primary_model_attack_pair_count=len(attack_pairs),
        primary_model_isolated_pair_count=primary_model_isolated_pair_count,
        independently_recomputed_residual_pair_count=len(residual),
        independent_action_domain_valid=independent_action_domain_valid,
        independent_service_valid=independent_service_valid,
        independent_model_consistent=independent_model_consistent,
        candidate_action_count=len(candidates),
        tie_break_solve_count=tie_break_solve_count,
    )
    return ExactOptimizationResult(capsule=capsule, diagnostics=diagnostics)

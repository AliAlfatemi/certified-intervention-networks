"""Synthetic enterprise graph generator for certified graph intervention.

The generator is defensive-only. It creates abstract attack-risk paths and
legitimate service dependencies; it never generates payloads, exploits, or
credential-theft procedures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import hashlib
import json
import random

import networkx as nx
import numpy as np

Edge = Tuple[str, str]
Path = Tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkConfig:
    size: str = "small"
    noise: float = 0.0
    hidden: float = 0.0
    density: str = "low"
    seed: int = 0
    time_steps: int = 4
    topology_regime: str = "shared_broker"


@dataclass
class NetworkInstance:
    config: BenchmarkConfig
    true_graph: nx.DiGraph
    observed_graph: nx.DiGraph
    protected_assets: List[str]
    attack_sources: List[str]
    required_service_pairs_observed: List[Tuple[str, str]]
    required_service_pairs_true: List[Tuple[str, str]]
    hidden_edges: Set[Edge]
    spurious_edges: Set[Edge]
    metadata: Dict[str, object]


SIZE_TABLE = {
    "small": dict(departments=3, users=9, devices=15, services=12, apis=6, dbs=3, creds=5, assets=2, iot=3),
    "medium": dict(departments=5, users=20, devices=35, services=25, apis=10, dbs=5, creds=8, assets=3, iot=5),
    "large": dict(departments=7, users=40, devices=70, services=45, apis=18, dbs=8, creds=14, assets=4, iot=8),
}

DENSITY_FACTOR = {"low": 0.65, "medium": 1.0, "high": 1.45}
EDGE_BASE_RISK = {
    "comm": 0.08,
    "auth": 0.12,
    "dependency": 0.05,
    "privilege": 0.30,
    "data": 0.14,
    "admin": 0.24,
    "shadow": 0.26,
    "iot_bridge": 0.28,
}

OBSERVABLE_EDGE_KEYS = {
    "kind",
    "risk_prob",
    "anomaly_score",
    "latency_ms",
    "action_cost",
}


def _dept(i: int, n: int) -> str:
    return f"dept{i % n}"


def _add_node(g: nx.DiGraph, node: str, typ: str, dept: str, tier: int, criticality: float = 0.0) -> None:
    g.add_node(node, type=typ, dept=dept, tier=tier, criticality=criticality)


def _add_edge(
    g: nx.DiGraph,
    u: str,
    v: str,
    kind: str,
    rng: random.Random,
    risk_boost: float = 0.0,
    required: bool = False,
    attack_role: str = "none",
    hidden_candidate: bool = False,
    temporal_stability: float = 0.85,
) -> None:
    base = EDGE_BASE_RISK.get(kind, 0.08)
    risk = min(0.98, max(0.01, base + risk_boost + rng.uniform(-0.015, 0.025)))
    anomaly = min(1.0, max(0.0, risk * 1.8 + rng.uniform(-0.10, 0.18)))
    g.add_edge(
        u,
        v,
        kind=kind,
        risk_prob=risk,
        anomaly_score=anomaly,
        required=required,
        attack_role=attack_role,
        hidden_candidate=hidden_candidate,
        temporal_stability=temporal_stability,
        latency_ms=max(0.1, rng.gauss(1.6 + 3.0 * risk, 0.4)),
        action_cost=1.0 + (0.5 if kind in {"privilege", "admin", "data"} else 0.0),
    )


def _partition(items: Sequence[str], departments: int) -> Dict[str, List[str]]:
    out = {f"dept{i}": [] for i in range(departments)}
    for i, item in enumerate(items):
        out[_dept(i, departments)].append(item)
    return out


def _safe_choice(rng: random.Random, values: Sequence[str]) -> str:
    if not values:
        raise ValueError("empty choice sequence")
    return values[rng.randrange(len(values))]


def _path_edges(path: Sequence[str]) -> Set[Edge]:
    return set(zip(path[:-1], path[1:]))


def _dfs_paths_limited(
    g: nx.DiGraph,
    sources: Iterable[str],
    targets: Iterable[str],
    cutoff: int = 8,
    max_paths_per_pair: int = 60,
) -> List[Path]:
    target_set = set(targets)
    paths: List[Path] = []
    for s in sources:
        if s not in g:
            continue
        for t in target_set:
            if t not in g or s == t:
                continue
            count = 0
            stack = [(s, [s])]
            while stack and count < max_paths_per_pair:
                node, path = stack.pop()
                if len(path) - 1 >= cutoff:
                    continue
                nbrs = list(g.successors(node))
                nbrs.sort(key=lambda x: g[node][x].get("risk_prob", 0.0), reverse=True)
                for nbr in nbrs:
                    if nbr in path:
                        continue
                    new_path = path + [nbr]
                    if nbr == t:
                        paths.append(tuple(new_path))
                        count += 1
                        if count >= max_paths_per_pair:
                            break
                    else:
                        stack.append((nbr, new_path))
    # Deduplicate while preserving order.
    seen = set()
    unique: List[Path] = []
    for p in paths:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def generate_network(config: BenchmarkConfig) -> NetworkInstance:
    if config.size not in SIZE_TABLE:
        raise ValueError(f"unknown graph size {config.size!r}")
    if config.density not in DENSITY_FACTOR:
        raise ValueError(f"unknown density {config.density!r}")
    if config.topology_regime not in {"shared_broker", "alternate_egress"}:
        raise ValueError(f"unknown topology regime {config.topology_regime!r}")

    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)
    params = SIZE_TABLE[config.size]
    deps = params["departments"]
    g = nx.DiGraph()

    users = [f"user_{i}" for i in range(params["users"])]
    devices = [f"device_{i}" for i in range(params["devices"])]
    services = [f"svc_{i}" for i in range(params["services"])]
    apis = [f"api_{i}" for i in range(params["apis"])]
    dbs = [f"db_{i}" for i in range(params["dbs"])]
    creds = [f"cred_{i}" for i in range(params["creds"])]
    assets = [f"asset_{i}" for i in range(params["assets"])]
    iots = [f"iot_{i}" for i in range(params["iot"])]
    gateways = [f"gateway_{i}" for i in range(deps)]
    brokers = [f"broker_{i}" for i in range(deps)]
    monitors = [f"monitor_{i}" for i in range(deps)]

    for i, u in enumerate(users):
        _add_node(g, u, "user", _dept(i, deps), 0)
    for i, d in enumerate(devices):
        _add_node(g, d, "device", _dept(i, deps), 0)
    for i, s in enumerate(services):
        _add_node(g, s, "service", _dept(i, deps), 1)
    for i, a in enumerate(apis):
        _add_node(g, a, "api", _dept(i, deps), 1)
    for i, db in enumerate(dbs):
        _add_node(g, db, "database", _dept(i, deps), 2, criticality=0.4)
    for i, c in enumerate(creds):
        _add_node(g, c, "credential", _dept(i, deps), 1)
    for i, a in enumerate(assets):
        _add_node(g, a, "asset", "shared", 3, criticality=1.0)
    for i, it in enumerate(iots):
        _add_node(g, it, "iot", _dept(i, deps), 0)
    for i, gateway in enumerate(gateways):
        _add_node(g, gateway, "gateway", _dept(i, deps), 0)
    for i, broker in enumerate(brokers):
        _add_node(g, broker, "broker", _dept(i, deps), 1)
    for i, monitor in enumerate(monitors):
        _add_node(g, monitor, "monitor", _dept(i, deps), 2, criticality=0.2)

    by_dept = {
        "users": _partition(users, deps),
        "devices": _partition(devices, deps),
        "services": _partition(services, deps),
        "apis": _partition(apis, deps),
        "dbs": _partition(dbs, deps),
        "creds": _partition(creds, deps),
        "iots": _partition(iots, deps),
    }

    required_paths_true: List[Path] = []

    # Generated per-department ingress and monitoring service.  Every IoT in a
    # department shares one gateway, broker, and monitoring endpoint.  The
    # first two hops are deliberately shared by legitimate telemetry and all
    # generated attack-risk routes.  Consequently, blanket source quarantine
    # is not service-feasible and an intervention must be placed downstream of
    # the broker when tau=1.
    gateway_for_dept = {f"dept{i}": gateways[i] for i in range(deps)}
    broker_for_dept = {f"dept{i}": brokers[i] for i in range(deps)}
    monitor_for_dept = {f"dept{i}": monitors[i] for i in range(deps)}
    for dept in range(deps):
        dn = f"dept{dept}"
        gateway = gateway_for_dept[dn]
        broker = broker_for_dept[dn]
        monitor = monitor_for_dept[dn]
        # Add shared edges once per department; adding them inside the IoT loop
        # would overwrite their deterministic attributes for multi-IoT groups.
        _add_edge(g, gateway, broker, "dependency", rng, required=True)
        _add_edge(g, broker, monitor, "dependency", rng, required=True)
        for iot in by_dept["iots"][dn]:
            _add_edge(g, iot, gateway, "iot_bridge", rng, required=True)
            required_paths_true.append((iot, gateway, broker, monitor))

    # Normal business service paths.
    for dept in range(deps):
        dn = f"dept{dept}"
        dept_devices = by_dept["devices"][dn]
        dept_svcs = by_dept["services"][dn]
        dept_apis = by_dept["apis"][dn] or apis
        dept_dbs = by_dept["dbs"][dn] or dbs
        if not dept_devices or not dept_svcs:
            continue
        service_count = max(2, min(len(dept_svcs), int(2 + DENSITY_FACTOR[config.density] * 2)))
        for j in range(service_count):
            d = _safe_choice(rng, dept_devices)
            s = dept_svcs[j % len(dept_svcs)]
            api = dept_apis[j % len(dept_apis)]
            db = dept_dbs[j % len(dept_dbs)]
            _add_edge(g, d, s, "comm", rng, required=True)
            _add_edge(g, s, api, "dependency", rng, required=True, hidden_candidate=True)
            _add_edge(g, api, db, "data", rng, required=True, hidden_candidate=True)
            required_paths_true.append((d, s, api, db))
            # A subset of paths legitimately reaches a protected asset via a database.
            if j == 0 and assets:
                asset = assets[dept % len(assets)]
                _add_edge(g, db, asset, "data", rng, required=True, hidden_candidate=True)
                required_paths_true.append((d, s, api, db, asset))

    # Identity and privilege graph.
    for u in users:
        user_dept = g.nodes[u]["dept"]
        candidates = by_dept["devices"][user_dept]
        if candidates:
            _add_edge(g, u, _safe_choice(rng, candidates), "auth", rng, required=False)
        if creds:
            _add_edge(g, u, _safe_choice(rng, creds), "auth", rng, risk_boost=0.03)
    for c in creds:
        dept = g.nodes[c]["dept"]
        svc_candidates = by_dept["services"].get(dept, services) or services
        if svc_candidates:
            _add_edge(g, c, _safe_choice(rng, svc_candidates), "privilege", rng, risk_boost=0.03)

    # Benign extra east-west communication.
    benign_extra = int(len(services) * DENSITY_FACTOR[config.density] * 1.2)
    for _ in range(benign_extra):
        u = _safe_choice(rng, services + devices)
        v = _safe_choice(rng, services + apis + dbs)
        if u != v and not g.has_edge(u, v):
            k = "dependency" if g.nodes[u]["type"] == "service" else "comm"
            _add_edge(g, u, v, k, rng, risk_boost=0.02, hidden_candidate=(k == "dependency"))

    # Abstract attack-risk paths. These are risk reachability templates, not
    # exploit steps.  Attack sources legitimately need their department's
    # monitor but never require reachability to a protected asset, so the
    # safety specification remains feasible through a downstream cut.
    eligible_attack_sources = list(iots)
    attack_sources = rng.sample(
        eligible_attack_sources,
        k=min(max(2, params["departments"]), len(eligible_attack_sources)),
    )
    risk_paths_requested = max(3, int((params["departments"] + params["assets"]) * DENSITY_FACTOR[config.density] * 1.8))
    scenario_labels = [
        "lateral_movement_risk",
        "credential_misuse_risk",
        "misconfigured_service_account_risk",
        "unnecessary_east_west_path_risk",
        "shadow_api_dependency_risk",
        "iot_to_database_bridge_risk",
    ]
    for i in range(risk_paths_requested):
        src = attack_sources[i % len(attack_sources)]
        asset = assets[i % len(assets)]
        scenario = scenario_labels[i % len(scenario_labels)]
        src_dept = g.nodes[src]["dept"]
        broker = broker_for_dept[src_dept]
        # choose nodes that often cross trust boundaries
        svc = _safe_choice(rng, services)
        api = _safe_choice(rng, apis)
        db = _safe_choice(rng, dbs)
        cred = _safe_choice(rng, creds)
        if scenario == "credential_misuse_risk":
            branch = [broker, cred, svc, db, asset]
            kinds = ["auth", "privilege", "data", "data"]
        elif scenario == "misconfigured_service_account_risk":
            branch = [broker, svc, cred, db, asset]
            kinds = ["comm", "privilege", "data", "data"]
        elif scenario == "shadow_api_dependency_risk":
            branch = [broker, svc, api, asset]
            kinds = ["comm", "shadow", "data"]
        elif scenario == "iot_to_database_bridge_risk":
            branch = [broker, api, db, asset]
            kinds = ["iot_bridge", "data", "data"]
        elif scenario == "unnecessary_east_west_path_risk":
            svc2 = _safe_choice(rng, [s for s in services if g.nodes[s]["dept"] != src_dept] or services)
            branch = [broker, svc, svc2, db, asset]
            kinds = ["comm", "dependency", "dependency", "data"]
        else:
            # Keep the complete generated route within the declared eight-edge
            # proposal/evaluation horizon: IoT->gateway->broker plus this tail.
            branch = [broker, svc, api, db, asset]
            kinds = ["comm", "dependency", "data", "data"]
        # The required IoT->gateway->broker prefix was added once above.  Only
        # add/boost downstream branch edges here so repeated scenarios cannot
        # saturate shared service-edge risk scores.
        for u, v, k in zip(branch[:-1], branch[1:], kinds):
            if u == v:
                continue
            if not g.has_edge(u, v):
                _add_edge(g, u, v, k, rng, risk_boost=0.22, attack_role=scenario, hidden_candidate=(k in {"dependency", "data"}))
            else:
                g[u][v]["risk_prob"] = min(0.98, g[u][v].get("risk_prob", 0.1) + 0.18)
                g[u][v]["anomaly_score"] = min(1.0, g[u][v].get("anomaly_score", 0.0) + 0.25)
                g[u][v]["attack_role"] = scenario

    # Stress topology: add multiple optional routes that leave the required
    # ingress chain at the gateway and therefore bypass the shared broker
    # boundary. This is a diagnostic regime, not part of the primary grid. It
    # tests whether results depend on one structurally privileged cut surface.
    if config.topology_regime == "alternate_egress":
        for source_index, source in enumerate(attack_sources):
            source_dept = g.nodes[source]["dept"]
            gateway = gateway_for_dept[source_dept]
            for route_index in range(2):
                service = services[(2 * source_index + route_index) % len(services)]
                api = apis[(source_index + route_index) % len(apis)]
                database = dbs[(source_index + 2 * route_index) % len(dbs)]
                asset = assets[(source_index + route_index) % len(assets)]
                branch = [gateway, service, api, database, asset]
                kinds = ["shadow", "dependency", "data", "data"]
                for u, v, kind in zip(branch[:-1], branch[1:], kinds):
                    if u == v:
                        continue
                    if not g.has_edge(u, v):
                        _add_edge(
                            g,
                            u,
                            v,
                            kind,
                            rng,
                            risk_boost=0.24,
                            attack_role="alternate_egress_risk",
                            hidden_candidate=True,
                        )
                    else:
                        g[u][v]["risk_prob"] = min(
                            0.98, g[u][v].get("risk_prob", 0.1) + 0.18
                        )
                        g[u][v]["anomaly_score"] = min(
                            1.0, g[u][v].get("anomaly_score", 0.0) + 0.25
                        )
                        g[u][v]["attack_role"] = "alternate_egress_risk"

    protected_assets = assets

    # Benchmark labels: edges occurring on a generated attack-risk path.  These
    # labels evaluate the optional risk scorer; they are not causal-discovery
    # ground truth.
    true_attack_paths = _dfs_paths_limited(g, attack_sources, protected_assets, cutoff=8, max_paths_per_pair=80)
    benchmark_attack_edges: Set[Edge] = set()
    for p in true_attack_paths:
        benchmark_attack_edges |= _path_edges(p)
    for u, v in benchmark_attack_edges:
        if g.has_edge(u, v):
            g[u][v]["attack_path_benchmark_label"] = True
    for u, v in g.edges:
        g[u][v].setdefault("attack_path_benchmark_label", False)

    # Hide exactly the requested fraction (up to integer rounding) of eligible
    # *optional* dependency/data edges. Required service edges are excluded so
    # the topology-observation experiment does not silently alter the service
    # specification itself. The former implementation first retained only a
    # random 75% of candidates, causing the realized omission rate to be about
    # 0.75 * config.hidden and making the nominal x-axis inaccurate.
    hidden_candidates = [
        (u, v)
        for u, v, data in g.edges(data=True)
        if data.get("hidden_candidate") and not data.get("required")
    ]
    hidden_count = int(round(config.hidden * len(hidden_candidates)))
    hidden_edges: Set[Edge] = (
        set(rng.sample(hidden_candidates, k=min(hidden_count, len(hidden_candidates))))
        if hidden_candidates
        else set()
    )

    observed = nx.DiGraph()
    observed.add_nodes_from(g.nodes(data=True))
    for u, v, data in g.edges(data=True):
        if (u, v) in hidden_edges:
            continue
        # Apply the same missing-observation probability to every optional
        # edge. Conditioning the miss rate on the generator's latent attack
        # labels would leak benchmark construction into data quality and make
        # the observation process optimistically preserve attack routes.
        miss_prob = min(0.25, config.noise * 0.45)
        if rng.random() < miss_prob and not data.get("required"):
            continue
        # Do not leak generator/evaluation labels or latent sampling flags into
        # the planner's observed graph.
        data2 = {key: data[key] for key in OBSERVABLE_EDGE_KEYS if key in data}
        data2["observed"] = True
        # observation noise perturbs anomaly scores and risk estimates
        data2["anomaly_score"] = float(np.clip(data2["anomaly_score"] + np_rng.normal(0, config.noise * 0.25), 0, 1))
        data2["risk_prob"] = float(np.clip(data2["risk_prob"] + np_rng.normal(0, config.noise * 0.12), 0.01, 0.98))
        observed.add_edge(u, v, **data2)

    # Spurious telemetry edges.
    spurious_edges: Set[Edge] = set()
    possible_nodes = list(g.nodes)
    # Observation error is applied only beyond the fixed synthetic
    # ingress/service boundary. IoTs and gateways keep their sole required
    # egress, monitors
    # remain sinks, and brokers may receive spurious downstream branches.
    spurious_sources = [
        node
        for node in possible_nodes
        if g.nodes[node].get("type") not in {"iot", "gateway", "monitor"}
    ]
    spurious_count = int(round(config.noise * max(5, g.number_of_edges() * 0.28)))
    for _ in range(spurious_count * 3):
        if len(spurious_edges) >= spurious_count:
            break
        u = _safe_choice(rng, spurious_sources)
        v = _safe_choice(rng, possible_nodes)
        if u == v or observed.has_edge(u, v):
            continue
        if g.nodes[u]["tier"] > g.nodes[v]["tier"] + 1:
            continue
        _add_edge(observed, u, v, "comm", rng, risk_boost=config.noise * 0.1, attack_role="spurious_observation")
        observable = {key: observed[u][v][key] for key in OBSERVABLE_EDGE_KEYS if key in observed[u][v]}
        observed[u][v].clear()
        observed[u][v].update(observable, observed=True)
        spurious_edges.add((u, v))

    # Service requirements are exogenous endpoint pairs, not a list inferred
    # after observing a corrupted topology. Required edges are never hidden or
    # randomly missed, so every declared endpoint requirement must remain
    # reachable in the observed snapshot. Keeping the same specification in
    # both views prevents topology error from being conflated with an
    # incomplete service-policy input.
    required_service_pairs_true = list(dict.fromkeys((p[0], p[-1]) for p in required_paths_true))
    unreachable_requirements = [
        pair
        for pair in required_service_pairs_true
        if pair[0] not in observed
        or pair[1] not in observed
        or not nx.has_path(observed, pair[0], pair[1])
    ]
    if unreachable_requirements:
        raise RuntimeError(
            "generator violated the exogenous service specification: "
            f"{unreachable_requirements[:3]}"
        )
    required_service_pairs_observed = list(required_service_pairs_true)

    # Temporal telemetry summaries: how often edge appeared across short history.
    for u, v in observed.edges:
        base_stability = 0.8
        appearances = sum(1 for _ in range(config.time_steps) if rng.random() < base_stability)
        observed[u][v]["appearance_rate"] = appearances / max(1, config.time_steps)
        observed[u][v]["temporal_delta"] = abs(observed[u][v]["appearance_rate"] - base_stability)
        expected_anomaly = min(1.0, 1.8 * float(observed[u][v].get("risk_prob", 0.05)))
        telemetry_disagreement = abs(float(observed[u][v].get("anomaly_score", 0.0)) - expected_anomaly)
        observed[u][v]["uncertainty"] = float(
            np.clip(0.75 * (1 - observed[u][v]["appearance_rate"]) + 0.25 * telemetry_disagreement, 0.0, 1.0)
        )

    true_edges = set(g.edges)
    observed_edges = set(observed.edges)
    retained_true_edges = true_edges & observed_edges
    missing_true_edges = true_edges - observed_edges
    observed_spurious_edges = observed_edges - true_edges
    required_true_edges = {
        (source, destination)
        for source, destination, data in g.edges(data=True)
        if data.get("required")
    }
    attack_true_edges = set(benchmark_attack_edges)
    metadata = {
        "topology_regime": config.topology_regime,
        "node_count": g.number_of_nodes(),
        "edge_count_true": g.number_of_edges(),
        "edge_count_observed": observed.number_of_edges(),
        "true_attack_path_count": len(true_attack_paths),
        "observed_required_service_pair_count": len(required_service_pairs_observed),
        "true_required_service_pair_count": len(required_service_pairs_true),
        "eligible_hidden_edge_count": len(hidden_candidates),
        "hidden_edge_count": len(hidden_edges),
        "realized_hidden_fraction_eligible": (
            len(hidden_edges) / len(hidden_candidates) if hidden_candidates else 0.0
        ),
        "missing_true_edge_count": len(missing_true_edges),
        "spurious_observed_edge_count": len(observed_spurious_edges),
        "edge_observation_recall": (
            len(retained_true_edges) / len(true_edges) if true_edges else 1.0
        ),
        "edge_observation_precision": (
            len(retained_true_edges) / len(observed_edges) if observed_edges else 1.0
        ),
        "required_edge_missing_count": len(required_true_edges - observed_edges),
        "attack_edge_observation_recall": (
            len(attack_true_edges & observed_edges) / len(attack_true_edges)
            if attack_true_edges
            else 1.0
        ),
        "service_specification_coverage": (
            len(required_service_pairs_observed) / len(required_service_pairs_true)
            if required_service_pairs_true
            else 1.0
        ),
    }
    for snapshot_graph, view in ((g, "true"), (observed, "observed")):
        topology_payload = {
            "nodes": sorted(map(str, snapshot_graph.nodes)),
            "edges": sorted((str(source), str(destination)) for source, destination in snapshot_graph.edges),
        }
        topology_digest = hashlib.sha256(
            json.dumps(topology_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        snapshot_graph.graph.update(
            {
                "snapshot_id": f"synthetic-{view}-{topology_digest}",
                "topology_version": 1,
                "snapshot_time": "2026-07-26T00:00:00Z",
            }
        )
    return NetworkInstance(
        config=config,
        true_graph=g,
        observed_graph=observed,
        protected_assets=protected_assets,
        attack_sources=attack_sources,
        required_service_pairs_observed=required_service_pairs_observed,
        required_service_pairs_true=required_service_pairs_true,
        hidden_edges=hidden_edges,
        spurious_edges=spurious_edges,
        metadata=metadata,
    )


def generate_pair_for_transfer(config: BenchmarkConfig) -> Tuple[NetworkInstance, NetworkInstance]:
    src = generate_network(config)
    tgt = generate_network(
        BenchmarkConfig(
            size=config.size,
            noise=config.noise,
            hidden=config.hidden,
            density=config.density,
            seed=config.seed + 10_000,
            time_steps=config.time_steps,
            topology_regime=config.topology_regime,
        )
    )
    return src, tgt

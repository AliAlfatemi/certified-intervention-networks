"""Public-topology adapters for external-validity diagnostics.

Only the Abilene underlay is public. Threat roles, required services,
telemetry values, action costs, and endpoint attachments are independently
specified synthetic overlays and are never presented as operational Abilene
measurements.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib, json, random
import networkx as nx
from .graph_generator import BenchmarkConfig, NetworkInstance


def _data_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "public_topologies" / "abilene.json"


def load_abilene_underlay(path: str | Path | None = None):
    payload = json.loads(Path(path or _data_path()).read_text(encoding="utf-8"))
    graph = nx.Graph()
    for node in payload["nodes"]:
        graph.add_node(str(node["id"]), label=str(node["label"]))
    graph.add_edges_from((str(u), str(v)) for u, v in payload["undirected_edges"])
    if graph.number_of_nodes() != 11 or graph.number_of_edges() != 14 or not nx.is_connected(graph):
        raise ValueError("vendored Abilene topology failed integrity checks")
    return graph, payload


def _attrs(rng: random.Random, *, required: bool = False) -> Dict[str, object]:
    risk = 0.08 + rng.random() * 0.18
    return {
        "kind": "access" if required else "backbone",
        "risk_prob": risk,
        "anomaly_score": min(1.0, risk * 1.6 + rng.random() * 0.12),
        "latency_ms": 1.0 + rng.random() * 8.0,
        "action_cost": 2.0 if required else 1.0,
        "uncertainty": 0.03 + rng.random() * 0.12,
        "required": required,
        "observed": True,
    }


def generate_abilene_instance(seed: int = 0) -> NetworkInstance:
    underlay, source = load_abilene_underlay()
    rng = random.Random(seed)
    graph = nx.DiGraph()
    routers = [f"r_{node}" for node in sorted(underlay.nodes, key=int)]
    for node in sorted(underlay.nodes, key=int):
        graph.add_node(f"r_{node}", type="router", dept="public_backbone", tier=1, criticality=0.1,
                       public_label=underlay.nodes[node]["label"])
    for u, v in sorted(underlay.edges):
        graph.add_edge(f"r_{u}", f"r_{v}", **_attrs(rng))
        graph.add_edge(f"r_{v}", f"r_{u}", **_attrs(rng))

    shuffled = routers[:]; rng.shuffle(shuffled)
    src_r, asset_r, monitor_r, client_r, service_r = shuffled[:2], shuffled[2:4], shuffled[4:6], shuffled[6:8], shuffled[8:10]
    sources: List[str] = []; assets: List[str] = []; services: List[Tuple[str,str]] = []
    for i, (router, mon_router) in enumerate(zip(src_r, monitor_r)):
        s, m = f"src_{i}", f"monitor_{i}"
        graph.add_node(s, type="endpoint", dept=f"role_{i}", tier=0, criticality=0.0)
        graph.add_node(m, type="monitor", dept=f"role_{i}", tier=2, criticality=0.3)
        graph.add_edge(s, router, **_attrs(rng, required=True)); graph.add_edge(mon_router, m, **_attrs(rng, required=True))
        sources.append(s); services.append((s, m))
    for i, router in enumerate(asset_r):
        a=f"asset_{i}"; graph.add_node(a, type="asset", dept="protected", tier=3, criticality=1.0)
        graph.add_edge(router, a, **_attrs(rng)); assets.append(a)
    for i, (cr, sr) in enumerate(zip(client_r, service_r)):
        c, s = f"client_{i}", f"service_{i}"
        graph.add_node(c, type="client", dept=f"business_{i}", tier=0, criticality=0.0)
        graph.add_node(s, type="service", dept=f"business_{i}", tier=2, criticality=0.5)
        graph.add_edge(c, cr, **_attrs(rng, required=True)); graph.add_edge(sr, s, **_attrs(rng, required=True))
        services.append((c, s))

    payload = {"underlay":"Abilene","seed":seed,"nodes":sorted(map(str,graph.nodes)),"edges":sorted((str(u),str(v)) for u,v in graph.edges)}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:20]
    graph.graph.update(snapshot_id=f"public-abilene-overlay-{digest}", topology_version=1,
                       snapshot_time="2026-07-27T00:00:00Z", public_topology="Abilene",
                       public_topology_collection=source["collection"],
                       overlay_semantics="synthetic roles, services, telemetry, and costs")
    config=BenchmarkConfig(size="small",noise=0.0,hidden=0.0,density="medium",seed=seed,topology_regime="shared_broker")
    metadata={"topology_regime":"public_abilene","node_count":graph.number_of_nodes(),"edge_count_true":graph.number_of_edges(),
              "edge_count_observed":graph.number_of_edges(),"eligible_hidden_edge_count":0,"hidden_edge_count":0,
              "realized_hidden_fraction_eligible":0.0,"missing_true_edge_count":0,"spurious_observed_edge_count":0,
              "edge_observation_recall":1.0,"edge_observation_precision":1.0,"required_edge_missing_count":0,
              "attack_edge_observation_recall":1.0,"service_specification_coverage":1.0,
              "public_underlay_node_count":11,"public_underlay_undirected_edge_count":14,
              "public_topology_source_kind":source["source_kind"],"overlay_is_synthetic":True}
    return NetworkInstance(config=config,true_graph=graph.copy(),observed_graph=graph.copy(),protected_assets=assets,
                           attack_sources=sources,required_service_pairs_observed=list(services),required_service_pairs_true=list(services),
                           hidden_edges=set(),spurious_edges=set(),metadata=metadata)

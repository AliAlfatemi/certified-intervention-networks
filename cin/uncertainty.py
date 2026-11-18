"""Exact restricted guarantees for finite additive topology-completion sets."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import networkx as nx
from .simulator import apply_interventions, normalize_service_pairs, reachable_pairs, service_preservation
from .verifier import stable_hash

Edge = Tuple[str, str]

@dataclass(frozen=True)
class CompletionSetCertificate:
    schema: str
    semantics: str
    completion_set_digest: str
    optional_edge_count: int
    interventions: List[Edge]
    maximal_completion_threat_blocked: bool
    minimal_completion_service_preservation: float
    robust_pass: bool
    residual_pairs_maximal: List[Tuple[str,str]]
    guarantee_scope: str
    def to_dict(self) -> Dict[str,object]:
        return {
            "schema":self.schema,"semantics":self.semantics,"completion_set_digest":self.completion_set_digest,
            "optional_edge_count":self.optional_edge_count,"interventions":[list(e) for e in self.interventions],
            "maximal_completion_threat_blocked":self.maximal_completion_threat_blocked,
            "minimal_completion_service_preservation":self.minimal_completion_service_preservation,
            "robust_pass":self.robust_pass,"residual_pairs_maximal":[list(p) for p in self.residual_pairs_maximal],
            "guarantee_scope":self.guarantee_scope,
        }


def maximal_completion(observed_graph: nx.DiGraph, optional_edges: Iterable[tuple]) -> nx.DiGraph:
    graph=observed_graph.copy()
    for item in optional_edges:
        if len(item)==2: source,destination=item; attrs={}
        elif len(item)==3: source,destination,attrs=item
        else: raise ValueError("optional edges must contain 2 or 3 elements")
        source,destination=str(source),str(destination)
        if source not in graph or destination not in graph:
            raise ValueError(f"optional edge {(source,destination)!r} references an absent node")
        if not graph.has_edge(source,destination): graph.add_edge(source,destination,**dict(attrs))
    return graph


def verify_declared_completion_set(
    observed_graph: nx.DiGraph,
    optional_edges: Iterable[tuple],
    interventions: Sequence[Edge],
    attack_sources: Sequence[str],
    protected_assets: Sequence[str],
    required_service_pairs: Sequence[Sequence[str]],
    *, tau: float=1.0,
) -> CompletionSetCertificate:
    if not 0.0 <= tau <= 1.0: raise ValueError("tau must lie in [0,1]")
    interventions=list(dict.fromkeys((str(u),str(v)) for u,v in interventions))
    optional=list(optional_edges)
    maximal=maximal_completion(observed_graph, optional)
    minimal_after=apply_interventions(observed_graph,interventions)
    maximal_after=apply_interventions(maximal,interventions)
    services=normalize_service_pairs(required_service_pairs)
    residual=reachable_pairs(maximal_after,attack_sources,protected_assets)
    threat_blocked=not residual
    service_fraction=service_preservation(minimal_after,services)
    normalized=[]
    for item in optional:
        if len(item)==2: u,v=item; attrs={}
        else: u,v,attrs=item
        normalized.append({"source":str(u),"destination":str(v),"attributes":dict(attrs)})
    normalized.sort(key=lambda x:(x["source"],x["destination"]))
    digest=stable_hash({"observed_nodes":sorted(map(str,observed_graph.nodes)),
                        "observed_edges":sorted((str(u),str(v)) for u,v in observed_graph.edges),
                        "optional_edges":normalized,"services":[list(p) for p in services],
                        "threat":{"sources":sorted(map(str,attack_sources)),"assets":sorted(map(str,protected_assets))}})
    return CompletionSetCertificate(
        schema="cin.completion-set-certificate.v1",
        semantics="all subsets of the declared additive optional-edge set",
        completion_set_digest=digest,optional_edge_count=len(normalized),interventions=interventions,
        maximal_completion_threat_blocked=threat_blocked,
        minimal_completion_service_preservation=float(service_fraction),
        robust_pass=threat_blocked and service_fraction+1e-12>=tau,
        residual_pairs_maximal=residual,
        guarantee_scope="exact only for the declared finite additive completion set",
    )

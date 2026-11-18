"""Reference OpenFlow-like compiler and offline staged-enforcement simulator.

This is a semantics study, not hardware/controller validation. Each abstract
``deny directed edge`` action compiles to one deterministic IPv4 source/dest
DROP rule; the offline simulator decodes those rules back to abstract edges,
monitors threat/service reachability after each batch, and exercises rollback.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import hashlib, ipaddress
import networkx as nx
from .simulator import apply_interventions, reachable_pairs, service_preservation
from .verifier import stable_hash

Edge=Tuple[str,str]

def _initial_ip(node:str)->ipaddress.IPv4Address:
    host=1+int.from_bytes(hashlib.sha256(node.encode()).digest()[:2],"big")%253
    return ipaddress.IPv4Address(f"192.0.2.{host}")

def deterministic_endpoint_map(nodes:Iterable[str])->Dict[str,str]:
    used:Set[str]=set(); mapping={}
    for node in sorted(map(str,nodes)):
        candidate=_initial_ip(node)
        for _ in range(253):
            text=str(candidate)
            if text not in used: mapping[node]=text; used.add(text); break
            host=int(text.split(".")[-1]); candidate=ipaddress.IPv4Address(f"192.0.2.{1+host%253}")
        else: raise ValueError("too many nodes for TEST-NET endpoint map")
    return mapping

@dataclass(frozen=True)
class OpenFlowDenyRule:
    rule_id:str; table:int; priority:int; eth_type:str; ipv4_src:str; ipv4_dst:str; action:str; abstract_source:str; abstract_destination:str
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class OpenFlowPolicy:
    schema:str; semantics:str; endpoint_map:Dict[str,str]; rules:List[OpenFlowDenyRule]; intervention_hash:str; policy_hash:str
    def to_dict(self):
        return {"schema":self.schema,"semantics":self.semantics,"endpoint_map":dict(sorted(self.endpoint_map.items())),
                "rules":[r.to_dict() for r in self.rules],"intervention_hash":self.intervention_hash,"policy_hash":self.policy_hash}


def compile_openflow13_policy(graph:nx.DiGraph,interventions:Sequence[Edge])->OpenFlowPolicy:
    edges=list(dict.fromkeys((str(u),str(v)) for u,v in interventions))
    missing=[e for e in edges if not graph.has_edge(*e)]
    if missing: raise ValueError(f"interventions contain absent graph edges: {missing[:3]}")
    mapping=deterministic_endpoint_map(graph.nodes)
    rules=[OpenFlowDenyRule(f"cin-drop-{i:04d}",0,40000,"0x0800",f"{mapping[u]}/32",f"{mapping[v]}/32","DROP",u,v) for i,(u,v) in enumerate(edges)]
    ih=stable_hash([list(e) for e in edges]); core={"schema":"cin.openflow13-edge-deny.v1","semantics":"drop packets matching each directed abstract endpoint pair","endpoint_map":mapping,"rules":[r.to_dict() for r in rules],"intervention_hash":ih}
    return OpenFlowPolicy(core["schema"],core["semantics"],mapping,rules,ih,stable_hash(core))

def policy_edges(policy:OpenFlowPolicy)->List[Edge]: return [(r.abstract_source,r.abstract_destination) for r in policy.rules]

def verify_compiler_equivalence(graph:nx.DiGraph,interventions:Sequence[Edge],policy:OpenFlowPolicy)->Dict[str,object]:
    expected=list(dict.fromkeys((str(u),str(v)) for u,v in interventions)); decoded=policy_edges(policy)
    missing=sorted(set(expected)-set(decoded)); extra=sorted(set(decoded)-set(expected)); duplicates=len(decoded)-len(set(decoded)); bad=[]
    for r in policy.rules:
        if r.action!="DROP" or r.eth_type!="0x0800" or r.priority<=0: bad.append(r.rule_id)
        if r.ipv4_src!=f"{policy.endpoint_map.get(r.abstract_source)}/32" or r.ipv4_dst!=f"{policy.endpoint_map.get(r.abstract_destination)}/32": bad.append(r.rule_id)
    hash_ok=policy.intervention_hash==stable_hash([list(e) for e in expected])
    return {"schema":"cin.compiler-equivalence.v1","passed":not missing and not extra and duplicates==0 and not bad and hash_ok,
            "missing_abstract_edges":[list(e) for e in missing],"extra_compiled_edges":[list(e) for e in extra],
            "duplicate_rule_count":duplicates,"invalid_rule_ids":sorted(set(bad)),"intervention_hash_ok":hash_ok,
            "scope":"reference OpenFlow-like compiler and offline graph-policy simulator"}

class OfflineControllerSimulator:
    def __init__(self,graph:nx.DiGraph): self.base_graph=graph.copy(); self.installed_policy:OpenFlowPolicy|None=None; self._prior_policy:OpenFlowPolicy|None=None
    @property
    def blocked_edges(self): return [] if self.installed_policy is None else policy_edges(self.installed_policy)
    def effective_graph(self): return apply_interventions(self.base_graph,self.blocked_edges)
    def install(self,policy:OpenFlowPolicy):
        eq=verify_compiler_equivalence(self.base_graph,policy_edges(policy),policy)
        if not eq["passed"]: raise ValueError("compiled policy failed equivalence validation")
        self._prior_policy=self.installed_policy; self.installed_policy=policy
    def rollback(self): self.installed_policy=self._prior_policy; self._prior_policy=None

@dataclass(frozen=True)
class StagedEnforcementReport:
    schema:str; dry_run_equivalent:bool; batch_count:int; installed_action_count:int; service_preservation_after_install:float
    residual_attack_pair_count_after_install:int; committed:bool; rollback_exercised:bool; rollback_restored_prior_semantics:bool
    monitoring_trace:List[Dict[str,object]]; scope:str
    def to_dict(self): return asdict(self)


def staged_install_with_monitoring(graph:nx.DiGraph,policy:OpenFlowPolicy,attack_sources:Sequence[str],protected_assets:Sequence[str],
                                   required_service_pairs:Sequence[Sequence[str]],*,batch_size:int=4,require_final_threat_block:bool=True)->StagedEnforcementReport:
    if batch_size<=0: raise ValueError("batch_size must be positive")
    full=policy_edges(policy); dry=verify_compiler_equivalence(graph,full,policy); controller=OfflineControllerSimulator(graph); trace=[]
    committed=bool(dry["passed"]); rollback_exercised=False; rollback_restored=False; prior_edges=list(controller.blocked_edges)
    if committed:
        # Empty policies are valid and require no batches.
        ends=list(range(batch_size,len(full)+batch_size,batch_size)) or [0]
        for end in ends:
            staged=full[:min(end,len(full))]
            staged_policy=compile_openflow13_policy(graph,staged); controller.install(staged_policy)
            effective=controller.effective_graph(); sp=service_preservation(effective,required_service_pairs); residual=len(reachable_pairs(effective,attack_sources,protected_assets))
            trace.append({"installed_action_count":len(staged),"service_preservation":float(sp),"residual_attack_pair_count":residual})
            if sp<1.0-1e-12:
                controller.rollback(); rollback_exercised=True; rollback_restored=(controller.blocked_edges==prior_edges); committed=False; break
        if committed and require_final_threat_block and reachable_pairs(controller.effective_graph(),attack_sources,protected_assets):
            controller.rollback(); rollback_exercised=True; rollback_restored=(controller.blocked_edges==prior_edges); committed=False
    effective=controller.effective_graph(); final_sp=service_preservation(effective,required_service_pairs); final_residual=len(reachable_pairs(effective,attack_sources,protected_assets))
    return StagedEnforcementReport("cin.staged-enforcement-report.v1",bool(dry["passed"]),len(trace),len(controller.blocked_edges),float(final_sp),final_residual,committed,rollback_exercised,rollback_restored,trace,"reference compiler and offline simulator; no live controller or hardware")


def exercise_bad_policy_rollback(graph:nx.DiGraph,required_service_pairs:Sequence[Sequence[str]])->Dict[str,object]:
    """Install a deliberately service-breaking cut and verify rollback."""
    if not required_service_pairs:
        return {"exercised":False,"restored":False,"reason":"no service pair"}
    source,destination=map(str,required_service_pairs[0])
    try:
        cut=sorted((str(u),str(v)) for u,v in nx.minimum_edge_cut(graph,source,destination))
    except (nx.NetworkXNoPath,nx.NetworkXError):
        return {"exercised":False,"restored":False,"reason":"service already unreachable or cut unavailable"}
    if not cut:
        return {"exercised":False,"restored":False,"reason":"empty service cut"}
    controller=OfflineControllerSimulator(graph); prior=controller.blocked_edges
    controller.install(compile_openflow13_policy(graph,cut))
    broken=service_preservation(controller.effective_graph(),required_service_pairs)<1.0-1e-12
    if broken: controller.rollback()
    return {"exercised":bool(broken),"restored":bool(broken and controller.blocked_edges==prior),"edges":[list(e) for e in cut]}

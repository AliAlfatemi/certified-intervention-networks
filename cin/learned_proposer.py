"""Calibrated learned edge ranker with a proposal-independent admission checker.

Training labels are membership in the MILP's canonically tie-broken selected
intervention on disjoint graphs. The learned probabilities guide only proposal
ranking; the existing deterministic capsule checker remains the admission
authority.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import math
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from .capsule import DefenseCapsule, EXPERIMENT_MAX_CAPSULE_AGE_SECONDS, EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS, build_capsule, experiment_issuance_time
from .repair import SubsetMinimalRepair
from .verifier import canonical_action_set
from .risk_model import RiskScoreFitResult
from .simulator import PathSearchScope, all_attack_routes_blocked, apply_interventions, enumerate_attack_paths, path_edges, reachable_pairs, residual_attack_witnesses, service_preservation

Edge=Tuple[str,str]
RELATIONS=("comm","auth","dependency","privilege","data","admin","shadow","iot_bridge","backbone","access","other")
FEATURE_NAMES=("path_coverage","anomaly_score","risk_prob","uncertainty","cross_department","positive_tier_delta","asset_proximity","source_out_degree","destination_in_degree","edge_betweenness","action_cost")+tuple(f"relation_{r}" for r in RELATIONS)


def extract_edge_features(graph:nx.DiGraph,attack_sources:Iterable[str],protected_assets:Iterable[str],*,cutoff:int=8):
    edges=sorted((str(u),str(v)) for u,v in graph.edges)
    paths=enumerate_attack_paths(graph,attack_sources,protected_assets,cutoff=cutoff)
    coverage={e:0.0 for e in edges}
    for p in paths:
        for e in zip(p[:-1],p[1:]):
            if e in coverage: coverage[e]+=1.0
    max_cov=max(coverage.values(),default=1.0) or 1.0
    reverse=graph.reverse(copy=False)
    dist:Dict[str,int]={}
    queue=[]
    for asset in sorted(set(map(str,protected_assets))):
        if asset in graph and asset not in dist: dist[asset]=0; queue.append(asset)
    for node in queue:
        for nb in sorted(reverse.successors(node)):
            if nb not in dist: dist[nb]=dist[node]+1; queue.append(nb)
    centrality=nx.edge_betweenness_centrality(graph,normalized=True)
    max_out=max((graph.out_degree(n) for n in graph),default=1) or 1
    max_in=max((graph.in_degree(n) for n in graph),default=1) or 1
    rows=[]
    for u,v in edges:
        data=graph[u][v]; rel=str(data.get("kind","other")); rel=rel if rel in RELATIONS else "other"
        d=dist.get(v); proximity=0.0 if d is None else 1.0/(1.0+d)
        row=[coverage[(u,v)]/max_cov,float(data.get("anomaly_score",0.0)),float(data.get("risk_prob",0.05)),float(data.get("uncertainty",0.0)),
             float(graph.nodes[u].get("dept")!=graph.nodes[v].get("dept")),max(0.0,float(graph.nodes[v].get("tier",0))-float(graph.nodes[u].get("tier",0)))/3.0,
             proximity,float(graph.out_degree(u))/max_out,float(graph.in_degree(v))/max_in,float(centrality.get((u,v),0.0)),float(data.get("action_cost",1.0))]
        row.extend(float(rel==candidate) for candidate in RELATIONS); rows.append(row)
    return edges,np.asarray(rows,dtype=float)


@dataclass(frozen=True)
class CalibratedEdgeModel:
    schema:str; feature_names:Tuple[str,...]; scaler_mean:Tuple[float,...]; scaler_scale:Tuple[float,...]
    base_coef:Tuple[float,...]; base_intercept:float; calibration_coef:float; calibration_intercept:float
    training_graph_count:int; calibration_graph_count:int; positive_edge_fraction:float
    def to_dict(self)->Dict[str,object]:
        p=asdict(self)
        for k in ("feature_names","scaler_mean","scaler_scale","base_coef"): p[k]=list(p[k])
        return p
    @classmethod
    def from_dict(cls,p:Mapping[str,object]):
        return cls(schema=str(p["schema"]),feature_names=tuple(map(str,p["feature_names"])),scaler_mean=tuple(map(float,p["scaler_mean"])),
                   scaler_scale=tuple(map(float,p["scaler_scale"])),base_coef=tuple(map(float,p["base_coef"])),base_intercept=float(p["base_intercept"]),
                   calibration_coef=float(p["calibration_coef"]),calibration_intercept=float(p["calibration_intercept"]),
                   training_graph_count=int(p["training_graph_count"]),calibration_graph_count=int(p["calibration_graph_count"]),positive_edge_fraction=float(p["positive_edge_fraction"]))
    def predict_proba(self,features:np.ndarray)->np.ndarray:
        mean=np.asarray(self.scaler_mean); scale=np.asarray(self.scaler_scale); scale=np.where(scale==0,1.0,scale)
        logits=((features-mean)/scale)@np.asarray(self.base_coef)+self.base_intercept
        logits=np.clip(self.calibration_coef*logits+self.calibration_intercept,-40,40)
        return 1.0/(1.0+np.exp(-logits))


def fit_calibrated_edge_model(training_examples,calibration_examples)->CalibratedEdgeModel:
    if not training_examples or not calibration_examples: raise ValueError("training and calibration sets must be non-empty")
    def materialize(examples):
        xs=[]; ys=[]
        for graph,sources,assets,selected in examples:
            edges,x=extract_edge_features(graph,sources,assets); selected=set((str(u),str(v)) for u,v in selected)
            xs.append(x); ys.append(np.asarray([int(e in selected) for e in edges],dtype=int))
        return np.vstack(xs),np.concatenate(ys)
    xtr,ytr=materialize(training_examples); xcal,ycal=materialize(calibration_examples)
    if len(np.unique(ytr))<2: raise ValueError("training labels contain one class")
    scaler=StandardScaler().fit(xtr)
    base=LogisticRegression(class_weight="balanced",max_iter=4000,solver="lbfgs",random_state=17).fit(scaler.transform(xtr),ytr)
    score=base.decision_function(scaler.transform(xcal)).reshape(-1,1)
    if len(np.unique(ycal))<2: a,b=1.0,0.0
    else:
        cal=LogisticRegression(max_iter=2000,solver="lbfgs",random_state=23).fit(score,ycal)
        a,b=float(cal.coef_[0,0]),float(cal.intercept_[0])
    return CalibratedEdgeModel("cin.calibrated-edge-model.v1",FEATURE_NAMES,tuple(map(float,scaler.mean_)),tuple(map(float,scaler.scale_)),
                               tuple(map(float,base.coef_[0])),float(base.intercept_[0]),a,b,len(training_examples),len(calibration_examples),float(np.mean(ytr)))


def expected_calibration_error(labels:np.ndarray,probabilities:np.ndarray,bins:int=10)->float:
    if bins<=0: raise ValueError("bins must be positive")
    if len(labels)==0: return float("nan")
    boundaries=np.linspace(0,1,bins+1); ece=0.0
    for i in range(bins):
        lo,hi=boundaries[i],boundaries[i+1]
        mask=(probabilities>=lo)&((probabilities<hi) if i<bins-1 else (probabilities<=hi))
        if np.any(mask): ece+=float(np.mean(mask))*abs(float(np.mean(labels[mask]))-float(np.mean(probabilities[mask])))
    return ece


def evaluate_edge_model(model:CalibratedEdgeModel,examples)->Dict[str,float]:
    ys=[]; ps=[]
    for graph,sources,assets,selected in examples:
        edges,x=extract_edge_features(graph,sources,assets); selected=set((str(u),str(v)) for u,v in selected)
        ys.extend(int(e in selected) for e in edges); ps.extend(map(float,model.predict_proba(x)))
    y=np.asarray(ys,dtype=int); p=np.asarray(ps,dtype=float)
    return {"roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float("nan"),
            "average_precision":float(average_precision_score(y,p)) if np.any(y==1) else float("nan"),
            "brier_score":float(brier_score_loss(y,p)),"ece_10":float(expected_calibration_error(y,p,10)),
            "positive_edge_fraction":float(np.mean(y)),"edge_count":float(len(y))}


class LearnedEdgeScorer:
    def __init__(self,model:CalibratedEdgeModel,cutoff:int=8): self.model=model; self.cutoff=cutoff; self.fit_result=None
    def fit(self,graph,attack_sources,protected_assets):
        edges,x=extract_edge_features(graph,attack_sources,protected_assets,cutoff=self.cutoff)
        probs=self.model.predict_proba(x); paths=enumerate_attack_paths(graph,attack_sources,protected_assets,cutoff=self.cutoff)
        self.fit_result=RiskScoreFitResult({e:float(p) for e,p in zip(edges,probs)},{e:float(graph[e[0]][e[1]].get("uncertainty",0.0)) for e in edges},paths)
        return self.fit_result
    def scores(self):
        if self.fit_result is None: raise RuntimeError("fit must be called")
        return self.fit_result.edge_scores


class LearnedServiceAwarePlanner:
    def __init__(self,model:CalibratedEdgeModel,*,cutoff:int=8,learned_weight:float=0.8):
        self.model=model; self.scope=PathSearchScope(cutoff_edges=cutoff); self.learned_weight=float(learned_weight)
    def synthesize(self,graph,attack_sources,protected_assets,required_service_pairs,*,action_budget:int,candidate_actions:Sequence[Edge])->DefenseCapsule:
        allowed=set(canonical_action_set(candidate_actions,field="candidate_actions"))
        if any(not graph.has_edge(*edge) for edge in allowed): raise ValueError("candidate_actions contains an edge absent from the graph")
        scorer=LearnedEdgeScorer(self.model,self.scope.cutoff_edges); scorer.fit(graph,attack_sources,protected_assets); probs=scorer.scores(); selected=[]
        while len(selected)<action_budget:
            current=apply_interventions(graph,selected)
            if all_attack_routes_blocked(current,attack_sources,protected_assets): break
            paths=enumerate_attack_paths(current,attack_sources,protected_assets,cutoff=self.scope.cutoff_edges,
                                         max_paths_per_pair=self.scope.max_paths_per_pair,max_expansions_per_pair=self.scope.max_expansions_per_pair)
            paths=list(dict.fromkeys(paths+residual_attack_witnesses(current,attack_sources,protected_assets,limit=max(20,len(attack_sources)*len(protected_assets)))))
            candidates=sorted({e for p in paths for e in path_edges(p) if e in allowed and e not in selected})
            current_pairs=len(reachable_pairs(current,attack_sources,protected_assets)); best=None; best_value=-math.inf
            for edge in candidates:
                trial=current.copy(); trial.remove_edge(*edge)
                if service_preservation(trial,required_service_pairs)<1.0-1e-12: continue
                gain=current_pairs-len(reachable_pairs(trial,attack_sources,protected_assets))
                hit=sum(edge in path_edges(p) for p in paths)/max(1,len(paths)); cost=float(graph[edge[0]][edge[1]].get("action_cost",1.0))
                value=2.0*gain+hit+self.learned_weight*probs.get(edge,0.0)-0.03*cost
                if value>best_value+1e-12 or (abs(value-best_value)<=1e-12 and (best is None or edge<best)): best,best_value=edge,value
            if best is None: break
            selected.append(best)
        transformation=SubsetMinimalRepair.apply(graph,selected,attack_sources,protected_assets,required_service_pairs,tau=1.0)
        return build_capsule(method="Learned-CIN",graph=graph,proposal_transformation=transformation,candidate_actions=sorted(allowed),attack_sources=attack_sources,protected_assets=protected_assets,
                             required_service_pairs=required_service_pairs,action_budget=action_budget,tau=1.0,path_scope=self.scope,
                             issued_at=experiment_issuance_time(graph),max_snapshot_age_at_issue_seconds=EXPERIMENT_MAX_SNAPSHOT_AGE_AT_ISSUE_SECONDS,
                             max_capsule_age_seconds=EXPERIMENT_MAX_CAPSULE_AGE_SECONDS)

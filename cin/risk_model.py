"""Transparent observed-edge risk scorer used to order intervention proposals.

This module performs neither causal discovery nor temporal learning.  It combines
bounded path coverage with observed attributes using fixed, inspectable weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np

from .simulator import enumerate_attack_paths

Edge = Tuple[str, str]


@dataclass
class RiskScoreFitResult:
    edge_scores: Dict[Edge, float]
    uncertainty: Dict[Edge, float]
    attack_paths: List[Tuple[str, ...]]


class ObservedRiskScorer:
    """Fixed-form scorer over observed graph telemetry and bounded paths."""

    def __init__(self, cutoff: int = 8):
        self.cutoff = cutoff
        self.fit_result: Optional[RiskScoreFitResult] = None

    def fit(
        self,
        graph: nx.DiGraph,
        attack_sources: Iterable[str],
        protected_assets: Iterable[str],
    ) -> RiskScoreFitResult:
        paths = enumerate_attack_paths(graph, attack_sources, protected_assets, cutoff=self.cutoff)
        coverage = {edge: 0 for edge in graph.edges}
        for path in paths:
            for edge in zip(path[:-1], path[1:]):
                if edge in coverage:
                    coverage[edge] += 1
        max_coverage = max(coverage.values(), default=1) or 1

        min_distance_to_asset: Dict[str, float] = {}
        assets = list(protected_assets)
        for node in graph.nodes:
            distances = []
            for asset in assets:
                try:
                    distances.append(nx.shortest_path_length(graph, node, asset))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
            min_distance_to_asset[node] = min(distances) if distances else float("inf")

        raw_scores: Dict[Edge, float] = {}
        uncertainty: Dict[Edge, float] = {}
        for source, destination, data in graph.edges(data=True):
            edge = (source, destination)
            relation_risk = {
                "privilege": 0.95,
                "admin": 0.90,
                "shadow": 0.88,
                "iot_bridge": 0.86,
                "data": 0.70,
                "auth": 0.66,
                "comm": 0.45,
                "dependency": 0.40,
            }.get(data.get("kind", "comm"), 0.35)
            anomaly = float(data.get("anomaly_score", 0.0))
            risk_probability = float(data.get("risk_prob", 0.05))
            path_coverage = coverage.get(edge, 0) / max_coverage
            cross_department = float(graph.nodes[source].get("dept") != graph.nodes[destination].get("dept"))
            tier_delta = max(
                0.0,
                float(graph.nodes[destination].get("tier", 0) - graph.nodes[source].get("tier", 0)),
            ) / 3.0
            distance = min_distance_to_asset.get(destination, float("inf"))
            proximity = 0.0 if distance == float("inf") else 1.0 / (1.0 + distance)
            appearance_delta = float(data.get("temporal_delta", 0.0))
            edge_uncertainty = float(data.get("uncertainty", 0.0))
            score = (
                0.32 * path_coverage
                + 0.20 * anomaly
                + 0.14 * relation_risk
                + 0.13 * risk_probability
                + 0.08 * cross_department
                + 0.08 * tier_delta
                + 0.07 * proximity
                + 0.03 * appearance_delta
                - 0.10 * edge_uncertainty
            )
            raw_scores[edge] = float(np.clip(score, 0.0, 1.0))
            uncertainty[edge] = edge_uncertainty

        self.fit_result = RiskScoreFitResult(
            raw_scores,
            uncertainty,
            paths,
        )
        return self.fit_result

    def scores(self) -> Dict[Edge, float]:
        if self.fit_result is None:
            raise RuntimeError("fit must be called before scores")
        return self.fit_result.edge_scores

    def uncertainty(self) -> Dict[Edge, float]:
        if self.fit_result is None:
            raise RuntimeError("fit must be called before uncertainty")
        return self.fit_result.uncertainty

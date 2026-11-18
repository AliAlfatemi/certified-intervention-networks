"""Canonical paper figures generated only from canonical result CSVs."""
from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiments import CANONICAL_METHOD, SERVICE_AWARE_BASELINE, WEIGHTED_RANDOM_ROLLOUTS

LEGACY_WEIGHTED_RANDOM_ROLLOUTS = "RandomSequentialDefender"

METHOD_ORDER = [
    CANONICAL_METHOD,
    SERVICE_AWARE_BASELINE,
    "GreedyRisk",
    "AnomalyBlock",
    "GraphAnomalyCentralityBlock",
    "StaticMicrosegmentation",
    WEIGHTED_RANDOM_ROLLOUTS,
    LEGACY_WEIGHTED_RANDOM_ROLLOUTS,
]
COLORS = {
    CANONICAL_METHOD: "#185FA5",
    SERVICE_AWARE_BASELINE: "#D97706",
    "GreedyRisk": "#438D80",
    "AnomalyBlock": "#7A6E9F",
    "GraphAnomalyCentralityBlock": "#B85C5C",
    "StaticMicrosegmentation": "#77818A",
    WEIGHTED_RANDOM_ROLLOUTS: "#9A7B4F",
    LEGACY_WEIGHTED_RANDOM_ROLLOUTS: "#9A7B4F",
}
DISPLAY_LABELS = {
    CANONICAL_METHOD: "CIN",
    SERVICE_AWARE_BASELINE: "Service-Aware",
    "GreedyRisk": "Risk Greedy",
    "AnomalyBlock": "Anomaly",
    "GraphAnomalyCentralityBlock": "Graph+Anomaly",
    "StaticMicrosegmentation": "Static Segmentation",
    WEIGHTED_RANDOM_ROLLOUTS: "Best-of-8 Weighted Random Rollouts",
    LEGACY_WEIGHTED_RANDOM_ROLLOUTS: "Best-of-8 Weighted Random Rollouts",
    "ExactBoundedCandidateOracle": "Pool Oracle",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11.0,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.0,
            "legend.fontsize": 10.5,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(
        directory / name,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def _ordered(methods):
    values = list(methods)
    return [method for method in METHOD_ORDER if method in values] + sorted(set(values) - set(METHOD_ORDER))


def make_plots(
    results_dir: str,
    paper_fig_dir: str | None = None,
    figure_dir: str | None = None,
) -> None:
    _style()
    results = Path(results_dir)
    figure_directory = Path(figure_dir) if figure_dir else results / "figures"
    raw = pd.read_csv(results / "raw_results.csv")
    ablations = pd.read_csv(results / "ablation_results.csv")
    oracle = pd.read_csv(results / "oracle_diagnostics.csv")

    # Main trade-off: each method is one point; marker size encodes policy size.
    main = raw.groupby("method", as_index=False).agg(
        attack_reduction=("attack_pair_isolation_true", "mean"),
        service=("service_pair_preservation_true", "mean"),
        actions=("policy_size", "mean"),
    )
    figure, axis = plt.subplots(figsize=(6.8, 4.1))
    for method in _ordered(main.method):
        row = main[main.method == method].iloc[0]
        axis.scatter(
            row.attack_reduction,
            row.service,
            s=38 + 11 * row.actions,
            color=COLORS.get(method, "#555555"),
            edgecolor="white",
            linewidth=0.7,
            label=DISPLAY_LABELS.get(method, method),
            zorder=3,
        )
    axis.set(xlabel="Exact hidden source–asset pair isolation", ylabel="True endpoint-service preservation", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.legend(loc="lower left", frameon=False, ncol=2)
    _save(figure, figure_directory, "canonical_tradeoff.pdf")

    # Robustness with seed-level uncertainty bands for the two strongest planners.
    robust = raw[raw.method.isin([CANONICAL_METHOD, SERVICE_AWARE_BASELINE])]
    seed_means = robust.groupby(["method", "noise", "seed"], as_index=False).attack_pair_isolation_true.mean()
    summary = seed_means.groupby(["method", "noise"]).attack_pair_isolation_true.agg(["mean", "sem"]).reset_index()
    figure, axis = plt.subplots(figsize=(6.6, 3.7))
    for method in [CANONICAL_METHOD, SERVICE_AWARE_BASELINE]:
        subset = summary[summary.method == method]
        axis.plot(
            subset.noise,
            subset["mean"],
            marker="o",
            color=COLORS[method],
            label=DISPLAY_LABELS.get(method, method),
        )
        axis.fill_between(
            subset.noise.to_numpy(),
            (subset["mean"] - 1.96 * subset["sem"].fillna(0)).to_numpy(),
            (subset["mean"] + 1.96 * subset["sem"].fillna(0)).to_numpy(),
            color=COLORS[method],
            alpha=0.14,
        )
    axis.set(xlabel="Telemetry noise", ylabel="Exact hidden pair isolation", ylim=(-0.02, 1.02))
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.legend(frameon=False)
    _save(figure, figure_directory, "canonical_robustness.pdf")

    # Coverage/precision separates modeled acceptance from hidden-graph validity.
    selectivity_rows = []
    for method, group in raw.groupby("method"):
        accepted = group[group.certificate_pass > 0.5]
        selectivity_rows.append(
            {
                "method": method,
                "coverage": group.certificate_pass.mean(),
                "precision": accepted.true_feasible.mean() if len(accepted) else np.nan,
            }
        )
    selectivity = pd.DataFrame(selectivity_rows).set_index("method").reindex(_ordered(raw.method.unique()))
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    y = np.arange(len(selectivity))
    axes[0].barh(y, selectivity.coverage, color=[COLORS.get(method, "#777777") for method in selectivity.index])
    finite_precision = selectivity.precision.notna()
    axes[1].barh(
        y[finite_precision],
        selectivity.loc[finite_precision, "precision"],
        color=[COLORS.get(method, "#777777") for method in selectivity.index[finite_precision]],
    )
    for position in y[~finite_precision]:
        axes[1].text(
            0.02,
            position,
            "N/A (no accepted capsules)",
            va="center",
            ha="left",
            color="#667085",
            fontsize=11.0,
        )
    axes[0].set(
        yticks=y,
        yticklabels=[DISPLAY_LABELS.get(method, method) for method in selectivity.index],
        xlabel="Observed-model admission rate",
        xlim=(0, 1.02),
    )
    axes[1].set(xlabel="Hidden-state feasibility\namong admitted actions", xlim=(0, 1.02))
    for axis in axes:
        axis.grid(axis="x", alpha=0.18, linewidth=0.6)
    _save(figure, figure_directory, "canonical_certificate_selectivity.pdf")

    scalability = raw.groupby(["method", "size"], as_index=False).runtime_s.mean()
    size_order = [size for size in ["small", "medium", "large"] if size in set(raw["size"])]
    figure, axis = plt.subplots(figsize=(6.6, 3.8))
    for method in _ordered(scalability.method):
        subset = scalability[scalability.method == method].set_index("size").reindex(size_order)
        axis.plot(
            size_order,
            subset.runtime_s,
            marker="o",
            color=COLORS.get(method, "#777777"),
            label=DISPLAY_LABELS.get(method, method),
        )
    axis.set(xlabel="Graph size", ylabel="Runtime (seconds, log scale)", yscale="log")
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.legend(frameon=False, ncol=2)
    _save(figure, figure_directory, "canonical_scalability.pdf")

    ablation_order = [
        "full",
        "uniform_scores",
        "no_service_guard",
        "bounded_only",
        "random_tiebreak",
        "no_reverse_delete",
    ]
    ablation_labels = {
        "full": "Full",
        "uniform_scores": "Uniform scores",
        "no_service_guard": "No service guard",
        "bounded_only": "Bounded paths only",
        "random_tiebreak": "Random tie-break",
        "no_reverse_delete": "No reverse-delete",
    }
    ablation_summary = ablations.groupby("ablation", as_index=False)[
        ["attack_pair_isolation_true", "service_pair_preservation_true", "certificate_pass"]
    ].mean().set_index("ablation").reindex(ablation_order).dropna(how="all").reset_index()
    figure, axis = plt.subplots(figsize=(6.8, 3.9))
    positions = np.arange(len(ablation_summary))
    width = 0.25
    metrics = [
        ("attack_pair_isolation_true", "Hidden pair isolation"),
        ("service_pair_preservation_true", "True service preservation"),
        ("certificate_pass", "Observed-model admission"),
    ]
    for index, (metric, label) in enumerate(metrics):
        axis.bar(positions + (index - 1) * width, ablation_summary[metric], width=width, label=label)
    axis.set(
        xticks=positions,
        xticklabels=[ablation_labels[value] for value in ablation_summary.ablation],
        ylabel="Mean score",
        ylim=(0, 1.02),
    )
    axis.tick_params(axis="x", rotation=18)
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    axis.grid(axis="y", alpha=0.18, linewidth=0.6)
    _save(figure, figure_directory, "canonical_ablation.pdf")

    gap = oracle.groupby("method", as_index=False).optimality_gap_residual_attack_pairs_observed.mean()
    figure, axis = plt.subplots(figsize=(6.4, 3.5))
    gap = gap.sort_values("optimality_gap_residual_attack_pairs_observed")
    axis.barh(gap.method, gap.optimality_gap_residual_attack_pairs_observed, color=[COLORS.get(method, "#555555") for method in gap.method])
    axis.set(xlabel="Mean residual-pair gap to exact declared-pool oracle")
    axis.grid(axis="x", alpha=0.18, linewidth=0.6)
    _save(figure, figure_directory, "canonical_oracle_gap.pdf")

    if paper_fig_dir:
        destination = Path(paper_fig_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for source in figure_directory.glob("canonical_*.pdf"):
            shutil.copy2(source, destination / source.name)

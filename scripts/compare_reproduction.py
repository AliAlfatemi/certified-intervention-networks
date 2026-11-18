#!/usr/bin/env python3
"""Compare a regenerated result directory with the archived corrected evidence.

The comparison is deliberately scientific rather than byte-for-byte: wall-clock
runtimes, run timestamps, manifests, and PDF bytes are environment-dependent.
All deterministic non-timing rows, benchmark audits, aggregates, and statistical
results are checked with strict keys and a small floating-point tolerance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

TIMING_TOKENS = (
    "runtime",
    "model_fit_s",
    "planner_and_verifier_s",
    "serialized_replay_verifier_s",
)

FILE_SPECS: Dict[str, Sequence[str]] = {
    "raw_results.csv": (
        "seed",
        "size",
        "noise",
        "hidden",
        "density",
        "action_budget",
        "method",
    ),
    "summary_results.csv": ("method", "size"),
    "benchmark_observation_audit.csv": (
        "seed",
        "size",
        "noise",
        "hidden",
        "density",
        "action_budget",
    ),
    "benchmark_observation_summary.csv": ("noise", "hidden"),
    "ablation_results.csv": (
        "seed",
        "size",
        "noise",
        "hidden",
        "density",
        "action_budget",
        "ablation",
    ),
    "ablation_statistical_tests.csv": ("ablation", "metric"),
    "topology_stress.csv": (
        "seed",
        "size",
        "noise",
        "hidden",
        "density",
        "action_budget",
        "topology_regime",
        "method",
    ),
    "topology_stress_summary.csv": ("topology_regime", "method"),
    "topology_stress_statistical_tests.csv": (
        "contrast",
        "topology_regime",
        "method",
        "metric",
    ),
    "oracle_diagnostics.csv": ("seed", "noise", "hidden", "method"),
    "budget_sensitivity.csv": (
        "seed",
        "size",
        "density",
        "budget_regime",
        "method",
    ),
    "statistical_tests.csv": ("baseline", "metric"),
    "paper_main_table.csv": ("method",),
}


def _is_timing_column(column: str) -> bool:
    lowered = column.lower()
    return any(token in lowered for token in TIMING_TOKENS)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if pd.isna(value):
        return None
    return value


def _compare_frame(
    reference_path: Path,
    candidate_path: Path,
    keys: Sequence[str],
    *,
    atol: float,
) -> Dict[str, Any]:
    reference = pd.read_csv(reference_path)
    candidate = pd.read_csv(candidate_path)

    # Runtime-derived inferential rows are intentionally not reproducible across
    # machines; all scientific metrics remain in the comparison.
    if "metric" in reference.columns:
        reference = reference[reference["metric"] != "runtime_s"].copy()
        candidate = candidate[candidate["metric"] != "runtime_s"].copy()

    missing_keys = [key for key in keys if key not in reference or key not in candidate]
    if missing_keys:
        return {
            "passed": False,
            "reason": "missing key columns",
            "missing_keys": missing_keys,
        }

    reference_columns = [column for column in reference.columns if not _is_timing_column(column)]
    candidate_columns = [column for column in candidate.columns if not _is_timing_column(column)]
    missing_in_candidate = sorted(set(reference_columns) - set(candidate_columns))
    unexpected_in_candidate = sorted(set(candidate_columns) - set(reference_columns))
    common_columns = [column for column in reference_columns if column in candidate_columns]

    reference = reference[common_columns].sort_values(list(keys), kind="mergesort").reset_index(drop=True)
    candidate = candidate[common_columns].sort_values(list(keys), kind="mergesort").reset_index(drop=True)

    result: Dict[str, Any] = {
        "passed": True,
        "reference_rows": len(reference),
        "candidate_rows": len(candidate),
        "compared_columns": common_columns,
        "excluded_timing_columns": sorted(
            set(reference_path and pd.read_csv(reference_path, nrows=0).columns)
            - set(reference_columns)
        ),
        "missing_columns_in_candidate": missing_in_candidate,
        "unexpected_columns_in_candidate": unexpected_in_candidate,
        "mismatch_count": 0,
        "mismatch_examples": [],
    }
    if missing_in_candidate or unexpected_in_candidate or len(reference) != len(candidate):
        result["passed"] = False

    if len(reference) != len(candidate):
        result["reason"] = "row-count mismatch"
        return result

    mismatches: List[Dict[str, Any]] = []
    for column in common_columns:
        left = reference[column]
        right = candidate[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            left_values = left.to_numpy(dtype=float)
            right_values = right.to_numpy(dtype=float)
            equal = np.isclose(left_values, right_values, rtol=0.0, atol=atol, equal_nan=True)
        else:
            left_values = left.astype("string").fillna("<NA>").to_numpy()
            right_values = right.astype("string").fillna("<NA>").to_numpy()
            equal = left_values == right_values
        indices = np.flatnonzero(~equal)
        for index in indices[: max(0, 10 - len(mismatches))]:
            key_payload = {
                key: _json_scalar(reference.iloc[index][key])
                for key in keys
                if key in reference.columns
            }
            mismatches.append(
                {
                    "key": key_payload,
                    "column": column,
                    "reference": _json_scalar(reference.iloc[index][column]),
                    "candidate": _json_scalar(candidate.iloc[index][column]),
                }
            )
        result["mismatch_count"] += int(len(indices))

    if result["mismatch_count"]:
        result["passed"] = False
    result["mismatch_examples"] = mismatches
    return result


def compare_directories(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    atol: float = 1e-12,
) -> Dict[str, Any]:
    files: Dict[str, Any] = {}
    for filename, keys in FILE_SPECS.items():
        reference_path = reference_dir / filename
        candidate_path = candidate_dir / filename
        if not reference_path.exists() or not candidate_path.exists():
            files[filename] = {
                "passed": False,
                "reason": "missing file",
                "reference_exists": reference_path.exists(),
                "candidate_exists": candidate_path.exists(),
            }
            continue
        files[filename] = _compare_frame(
            reference_path,
            candidate_path,
            keys,
            atol=atol,
        )
    return {
        "schema": "cin.reproduction-comparison.v1",
        "reference": str(reference_dir.resolve()),
        "candidate": str(candidate_dir.resolve()),
        "absolute_tolerance": atol,
        "timing_fields_excluded": list(TIMING_TOKENS),
        "passed": all(item.get("passed", False) for item in files.values()),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()

    if not args.reference.is_dir() or not args.candidate.is_dir():
        parser.error("--reference and --candidate must be existing directories")
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be a finite non-negative number")

    report = compare_directories(args.reference, args.candidate, atol=args.atol)
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

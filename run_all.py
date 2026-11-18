#!/usr/bin/env python3
"""Run the single canonical experiment-to-paper-artifact pipeline.

Examples:
    python run_all.py --seeds 30 --output results/ --paper-dir ../paper/
    python run_all.py --quick --seeds 3 --output results_smoke_test/ --no-paper-figures
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from cin.experiments import run_all, write_manifest
from cin.plotting import make_plots


def require_absent_or_empty_output(path: Path) -> None:
    """Fail before execution if a supposedly fresh result root has contents."""

    if path.is_symlink():
        raise ValueError("fresh output directory must not be a symbolic link")
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError("fresh output path exists and is not a directory")
    members = sorted(child.name for child in path.iterdir())
    if members:
        preview = ", ".join(members[:5])
        suffix = "" if len(members) <= 5 else ", ..."
        raise ValueError(
            "fresh output directory is not empty; refusing stale-result reuse: "
            f"{preview}{suffix}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic CIN experiments.")
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of random seeds. The submitted paper run uses 30; default is 10 for local speed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for CSVs, JSON diagnostics, and figures.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small smoke-test grid for installation validation.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Parallel workers; 0 uses bounded auto-parallelism.",
    )
    parser.add_argument(
        "--paper-dir",
        type=str,
        default="paper/",
        help="Paper directory whose figures/ subdirectory should receive regenerated PDF figures.",
    )
    parser.add_argument(
        "--no-paper-figures",
        action="store_true",
        help="Do not copy regenerated figures into the paper directory.",
    )
    parser.add_argument(
        "--require-empty-output",
        action="store_true",
        help=(
            "Fail before execution unless the output path is absent or an empty "
            "directory; used by provenance-critical full reruns."
        ),
    )
    args = parser.parse_args()
    if args.require_empty_output:
        require_absent_or_empty_output(Path(args.output))
    started_at = datetime.now(timezone.utc)

    paper_tables_dir = None
    if not args.no_paper_figures and args.paper_dir:
        paper_root = Path(args.paper_dir)
        if paper_root.exists():
            paper_tables_dir = str(paper_root / "tables")

    run_all(
        args.seeds,
        args.output,
        quick=args.quick,
        jobs=args.jobs,
        paper_tables_dir=paper_tables_dir,
        started_at=started_at,
    )

    paper_fig_dir = None
    if not args.no_paper_figures and args.paper_dir:
        candidate = Path(args.paper_dir) / "figures"
        if candidate.parent.exists():
            paper_fig_dir = str(candidate)
    make_plots(args.output, paper_fig_dir=paper_fig_dir)
    # Refresh after plotting so the manifest binds canonical figures too.
    invocation = (
        f"python run_all.py --seeds {args.seeds} --output <OUTPUT> "
        f"{'--quick ' if args.quick else ''}--jobs {args.jobs} "
        f"{'--require-empty-output ' if args.require_empty_output else ''}"
        + ("--no-paper-figures" if args.no_paper_figures else "--paper-dir <PAPER_DIR>")
    )
    write_manifest(
        args.output,
        seeds=args.seeds,
        quick=args.quick,
        jobs=args.jobs,
        started_at=started_at,
        invocation=invocation,
    )
    print(f"Wrote results to {args.output}")
    if paper_fig_dir:
        print(f"Updated canonical paper figures in {paper_fig_dir}")
    if paper_tables_dir:
        print(f"Updated generated paper tables in {paper_tables_dir}")


if __name__ == "__main__":
    main()

"""Regressions for provenance-critical fresh primary execution roots."""

from __future__ import annotations

from pathlib import Path

import pytest

import run_all as pipeline


def test_fresh_output_accepts_absent_or_empty_directory(tmp_path: Path) -> None:
    pipeline.require_absent_or_empty_output(tmp_path / "absent")
    empty = tmp_path / "empty"
    empty.mkdir()
    pipeline.require_absent_or_empty_output(empty)


def test_fresh_output_rejects_existing_file_or_nonempty_directory(
    tmp_path: Path,
) -> None:
    existing_file = tmp_path / "result.csv"
    existing_file.write_text("stale\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        pipeline.require_absent_or_empty_output(existing_file)

    nonempty = tmp_path / "results"
    nonempty.mkdir()
    (nonempty / "stale.csv").write_text("stale\n", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing stale-result reuse"):
        pipeline.require_absent_or_empty_output(nonempty)


def test_fresh_output_rejects_symbolic_link(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked-results"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        pipeline.require_absent_or_empty_output(link)

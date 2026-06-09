"""Discover route sources from an input directory.

One source per file: every ``*.routes.jsonl`` becomes a "no-probe" source and
every ``*.probed.jsonl`` becomes a "(+probe)" source, so a classifier that has
both yields two leaderboard rows. The loader is classifier-agnostic, so T5,
RoBERTa and DeBERTa files are all picked up automatically. Optionally appends
the six synthetic baselines.
"""

from __future__ import annotations

from pathlib import Path

from . import sources as src_mod


def find_route_files(input_dir: Path) -> list:
    return sorted(Path(input_dir).glob("*.routes.jsonl"))


def find_probed_files(input_dir: Path) -> list:
    return sorted(Path(input_dir).glob("*.probed.jsonl"))


def discover_sources(input_dir: Path, grid: dict, fit_grid: dict,
                     include_baselines: bool = True) -> list:
    """Return one RouteSource per route/probed file in ``input_dir``, plus the
    synthetic baselines unless disabled."""
    sources = []
    for path in find_route_files(input_dir):
        sources.append(src_mod.from_routes_file(path))
    for path in find_probed_files(input_dir):
        sources.append(src_mod.from_probed_file(path))
    if include_baselines:
        sources.extend(src_mod.synthetic_baselines(grid, fit_grid))
    return sources

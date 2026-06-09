"""Offline evaluation harness for the Adaptive-RAG router.

Maps each question to a chosen route (from a classifier's ``*.routes.jsonl`` /
``*.probed.jsonl`` file, or a synthetic baseline), looks up the precomputed
metrics in ``data/labeled/{split}/*.jsonl``, and aggregates the slide's three
metric groups plus baselines, an oracle ceiling, and an optional τ-sweep. No
LLM/GPU/JDK — pure offline join.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\evaluate.py `
      --input-dir outputs\\routes `
      --output-dir outputs\\eval `
      --split eval `
      --sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import config
from src.eval_lib import discover, grid as grid_mod, metrics, probe_sweep, report


def parse_tau_grid(spec: str) -> list:
    """``"start:stop:step"`` (inclusive) -> list of taus. Single value allowed."""
    parts = spec.split(":")
    if len(parts) == 1:
        return [float(parts[0])]
    if len(parts) != 3:
        raise ValueError(f"--tau-grid must be 'start:stop:step', got {spec!r}")
    start, stop, step = (float(p) for p in parts)
    if step <= 0:
        raise ValueError("--tau-grid step must be > 0")
    taus = []
    x = start
    # accumulate via integer count to avoid float drift
    n = int(round((stop - start) / step))
    for i in range(n + 1):
        taus.append(round(start + i * step, 10))
    return taus


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labeled-dir", type=Path, default=config.LABELED_DIR,
                   help="Root holding {split}/{dataset}.jsonl (default: config.LABELED_DIR).")
    p.add_argument("--split", choices=("eval", "train"), default="eval")
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory of *.routes.jsonl / *.probed.jsonl files.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Subset of datasets (default: all six).")
    p.add_argument("--tau-grid", default="0:1:0.1",
                   help="Probe sweep grid 'start:stop:step' (default 0:1:0.1).")
    p.add_argument("--no-baselines", action="store_true",
                   help="Skip the six synthetic baselines.")
    p.add_argument("--sweep", action="store_true",
                   help="Also write a τ-sweep table per *.probed.jsonl file.")
    p.add_argument("--fit-split", choices=("eval", "train"), default="train",
                   help="Split whose oracle-label distribution defines the majority baseline.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    grid = grid_mod.load_grid(args.labeled_dir, args.split, args.datasets)
    print(f"loaded grid: {len(grid)} rows from {args.split} split")

    if args.no_baselines:
        fit_grid = {}
    elif args.fit_split == args.split:
        fit_grid = grid
    else:
        fit_grid = grid_mod.load_grid(args.labeled_dir, args.fit_split, args.datasets)

    ceiling_em, ceiling_f1 = metrics.oracle_ceiling(grid)
    print(f"oracle ceiling: {ceiling_em * 100:.1f} EM / {ceiling_f1 * 100:.1f} F1")

    sources = discover.discover_sources(
        args.input_dir, grid, fit_grid, include_baselines=not args.no_baselines
    )
    print(f"discovered {len(sources)} sources")

    results = []
    for source in sources:
        result = metrics.evaluate_source(grid, source, ceiling_em, ceiling_f1)
        results.append(result)
        cov = result.coverage
        if cov["matched"] < cov["grid_total"]:
            print(
                f"  warn: {source.name}: coverage {cov['matched']}/{cov['grid_total']} "
                f"(missing_in_source={cov['missing_in_source']}, "
                f"missing_in_grid={cov['missing_in_grid']})"
            )

    sweeps = []
    if args.sweep:
        taus = parse_tau_grid(args.tau_grid)
        for path in discover.find_probed_files(args.input_dir):
            sweeps.append(probe_sweep.sweep_probe(grid, path, taus))
        print(f"swept {len(sweeps)} probed file(s) over {len(taus)} tau values")

    paths = report.write_reports(
        results, sweeps, ceiling_em, ceiling_f1, args.output_dir, args.split
    )
    print(f"wrote {paths['leaderboard']}")
    print(f"wrote {paths['results']}")
    print(f"wrote {paths['sweeps']}")


if __name__ == "__main__":
    main()

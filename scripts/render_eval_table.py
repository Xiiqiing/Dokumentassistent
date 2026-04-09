"""Render an evaluation matrix as a Markdown table from RAGAS result JSONs.

Reads one or more ``eval/runs/<ts>_<cell>.json`` (or ``.rejudged.json``)
files, extracts the per-cell ``config`` + ``aggregate`` blocks, and prints a
single comparison table suitable for pasting into a README.

The cell with the highest score for each metric is bolded in the rendered
table, so the "winning" configuration on each axis is visible at a glance.

Usage:
    # Pass file paths directly (preserves the order you provide):
    python -m scripts.render_eval_table \\
        eval/runs/20260409_031715_fixed_react.rejudged.json \\
        eval/runs/20260409_031853_recursive_react.rejudged.json \\
        eval/runs/20260409_031911_semantic_react.rejudged.json \\
        eval/runs/20260409_031933_recursive_pipeline.rejudged.json

    # Or pass a shell glob via --pattern (sorted by filename):
    python -m scripts.render_eval_table --pattern 'eval/runs/20260409_03*.rejudged.json'
"""

import argparse
import glob
import json
import os
import sys
from typing import Any

# Display labels for metric columns, in the order they should appear in the
# rendered table. Maps the RAGAS metric column name to a short header.
# Metrics not in this map are skipped from the table; the raw JSON still
# contains them.
_METRIC_DISPLAY: list[tuple[str, str]] = [
    ("faithfulness", "Faith"),
    ("answer_relevancy", "Ans.Rel"),
    ("llm_context_precision_with_reference", "Ctx.Prec"),
    ("context_recall", "Ctx.Recall"),
    ("answer_correctness", "Ans.Corr"),
    ("factual_correctness(mode=f1)", "Fact.Corr"),
]

_FIXED_HEADERS: list[str] = ["Config", "Chunking", "Router", "top_k"]


def _load_cells(paths: list[str]) -> list[dict[str, Any]]:
    """Load and validate the requested result JSON files.

    Args:
        paths: List of JSON file paths.

    Returns:
        List of cell dicts with keys ``path``, ``config``, ``aggregate``.
        Files that are missing or malformed are skipped with a warning.
    """
    cells: list[dict[str, Any]] = []
    for path in paths:
        if not os.path.exists(path):
            print(f"WARN: skipping missing file {path}", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "config" not in data or "aggregate" not in data:
            print(
                f"WARN: {path} is missing 'config' or 'aggregate' — skipping",
                file=sys.stderr,
            )
            continue
        cells.append(
            {
                "path": path,
                "config": data["config"],
                "aggregate": data["aggregate"],
            }
        )
    return cells


def _format_table(cells: list[dict[str, Any]]) -> str:
    """Format cells as a Markdown comparison table with per-metric winners bolded.

    Args:
        cells: List of cell dicts produced by ``_load_cells``.

    Returns:
        Markdown string ready to paste into a README.
    """
    if not cells:
        return "(no cells to render)"

    header = list(_FIXED_HEADERS) + [display for _, display in _METRIC_DISPLAY]

    # Build rows AND track the best score per metric so we can bold the winner.
    rows: list[list[str]] = []
    best_score: dict[str, float] = {}
    for cell in cells:
        cfg = cell["config"]
        agg = cell["aggregate"]
        row: list[str] = [
            str(cfg.get("name", "?")),
            str(cfg.get("chunking", "?")),
            str(cfg.get("router", "?")),
            str(cfg.get("top_k", "?")),
        ]
        for metric_key, _display in _METRIC_DISPLAY:
            value = agg.get(metric_key)
            if isinstance(value, (int, float)):
                row.append(f"{value:.3f}")
                if value > best_score.get(metric_key, float("-inf")):
                    best_score[metric_key] = value
            else:
                row.append("—")
        rows.append(row)

    # Bold the winning cell for each metric column.
    for row in rows:
        for col_idx, (metric_key, _display) in enumerate(_METRIC_DISPLAY):
            cell_idx = len(_FIXED_HEADERS) + col_idx
            value_str = row[cell_idx]
            if value_str == "—":
                continue
            if float(value_str) == best_score.get(metric_key):
                row[cell_idx] = f"**{value_str}**"

    lines: list[str] = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render a RAGAS evaluation matrix as a Markdown table.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Result JSON files to include (preserves the order given).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="",
        help="Shell glob to match files (sorted by filename).",
    )
    return parser.parse_args()


def main() -> None:
    """Render the requested cells as a Markdown table to stdout."""
    args = parse_args()
    if args.pattern:
        files = sorted(glob.glob(args.pattern))
    else:
        files = list(args.files)
    if not files:
        print(
            "ERROR: no files provided. Pass file paths as arguments or use --pattern.",
            file=sys.stderr,
        )
        sys.exit(1)
    cells = _load_cells(files)
    if not cells:
        print("ERROR: no valid cells loaded.", file=sys.stderr)
        sys.exit(1)
    print(_format_table(cells))


if __name__ == "__main__":
    main()

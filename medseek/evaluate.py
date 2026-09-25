from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from medseek.metrics import compute_group_metrics, compute_metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute MedSeek metrics from a prediction CSV.")
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args(argv)
    with args.predictions.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = [
        {
            "gold_label": int(row["gold_label"]),
            "risk_score": float(row["risk_score"]),
            "center": row.get("center", ""),
        }
        for row in source_rows
        if row.get("gold_label", "") != "" and row.get("risk_score", "") != ""
    ]
    metrics = compute_metrics(rows)
    metrics["by_center"] = compute_group_metrics(rows)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

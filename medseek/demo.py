from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from medseek.data import load_records, split_records
from medseek.retrieval import run_balanced_retrieval, write_outputs


DEFAULT_MODEL = "Yjian1998/medseek"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MedSeek retrieval on sample or user-provided clinical records.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=4)
    return parser


def run_demo(
    *,
    data_path: Path,
    output_dir: Path,
    encoder,
    top_k: int = 4,
) -> dict:
    records = load_records(data_path)
    train_records, test_records = split_records(records)
    predictions, metrics, indexes, record_id_maps = run_balanced_retrieval(
        train_records=train_records,
        test_records=test_records,
        encoder=encoder,
        top_k=top_k,
    )
    written = write_outputs(output_dir, predictions, metrics, indexes, record_id_maps)
    return {
        "train_records": len(train_records),
        "test_records": len(test_records),
        "metrics": metrics,
        **written,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from medseek.encoder import E5TextEncoder
    encoder = E5TextEncoder(
        args.model_path,
        device=args.device,
        max_length=args.max_length,
    )
    summary = run_demo(
        data_path=args.data,
        output_dir=args.output_dir,
        encoder=encoder,
        top_k=args.top_k,
    )
    summary["model"] = args.model_path
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


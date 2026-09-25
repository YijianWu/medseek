from __future__ import annotations

import argparse
import json
from pathlib import Path

from medseek.data import load_records, validate_dataset, write_jsonl_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and normalize MedSeek CSV/JSONL data.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = load_records(args.input)
    summary = validate_dataset(records)
    write_jsonl_records(records, args.output)
    print(json.dumps({"output": str(args.output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

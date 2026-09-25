from __future__ import annotations

import argparse
import json
from pathlib import Path

from medseek.config import load_experiment_config, model_config_from_experiment
from medseek.data import load_records, validate_dataset
from medseek.models import MedSeekDualEncoder
from medseek.trainer import train_model


def main() -> int:
    """Run MedSeek training from a YAML configuration and prepared dataset."""
    # Parse the input, configuration, and output locations.
    parser = argparse.ArgumentParser(description="Train the compact MedSeek dual encoder.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training"))
    args = parser.parse_args()

    # Load and validate the experiment data before creating the model.
    experiment = load_experiment_config(args.config)
    records = load_records(args.data)
    data_summary = validate_dataset(records)

    # Use the test split for validation only when explicitly allowed by the configuration.
    train_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "validation"]
    if not validation_records and bool(experiment["data"].get("allow_test_as_validation", False)):
        validation_records = [record for record in records if record.split == "test"]
    if not train_records or not validation_records:
        raise ValueError("training requires train and validation records")

    # Check that enabled auxiliary tasks have the required supervision.
    model_config = model_config_from_experiment(experiment)
    if model_config.diagnosis_aux_enabled and not model_config.diagnosis_prototype_path:
        raise ValueError("diagnosis_aux.enabled requires diagnosis_aux.prototype_path; use configs/demo.yaml for sample data")
    if model_config.secondary_endpoint_cls_enabled and not any(
        record.secondary_endpoint is not None for record in train_records
    ):
        raise ValueError("secondary_endpoint_aux.enabled requires secondary_endpoint values in training data")
    # Train the model and print a machine-readable run summary.
    model = MedSeekDualEncoder(model_config)
    summary = train_model(model, train_records, validation_records, experiment["training"], args.output_dir)
    summary["data"] = data_summary
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

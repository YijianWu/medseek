# MedSeek

## Description

Timely identification of high-risk patients is important for clinical decision support. Here, we introduce MedSeek, a medical retrieval and representation-learning system that identifies clinically similar high-risk and non-high-risk cases from clinical records.

This repository provides the code used for data processing, model training, case-index construction, retrieval, risk scoring, and evaluation. A small fictional dataset is included for software testing and demonstration.

Clinical datasets and study outputs are not included. The public inference checkpoint is available on Hugging Face.

## Code Structure

- `configs/`: Configurations for the study workflow and sample-data testing.
- `data/data.csv`: Fictional sample data for running and testing the code.
- `medseek/`: Main source code.
  - `data/`: Data loading, preprocessing, validation, and sampling.
  - `models/`: Model definitions and training components.
  - `encoder.py`: Text encoding for retrieval.
  - `retrieval.py`: Case indexing, retrieval, and scoring.
  - `trainer.py`: Training and validation framework.
  - `evaluate.py`: Evaluation utilities.
- `prompts/case_to_query.txt`: Optional preprocessing template. It is not called by the runtime.
- `prepare_data.py`: Validate and normalize CSV or JSONL data.
- `run_train.py`: Train MedSeek.
- `run_retrieval.py`: Run case retrieval and risk scoring.
- `run_evaluate.py`: Evaluate exported predictions.
- `tests/`: Offline software tests.

## Quick Start

Install MedSeek and its runtime dependencies:

```bash
python -m pip install -e .
```

For development and testing, install the optional development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the example with the fictional sample data:

```bash
python run_retrieval.py \
  --data data/data.csv \
  --model-path Yjian1998/medseek \
  --device auto \
  --output-dir outputs/demo
```

The first run downloads `Yjian1998/medseek` from Hugging Face.

## Tests

```bash
python -m pytest -q
```

The tests use a deterministic test encoder and do not download model weights.

## Data

The included `data/data.csv` contains only fictional records. See `DATA.md` for the input schema and data-use requirements.

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.

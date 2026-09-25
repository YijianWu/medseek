from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from medseek.data import ClinicalRecord
from medseek.metrics import compute_group_metrics, compute_metrics


def _as_float32(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype="float32")
    if array.ndim != 2:
        raise ValueError(f"expected a 2D embedding matrix, got shape={array.shape}")
    return np.ascontiguousarray(array)


def _build_bucket(records: Sequence[ClinicalRecord], embeddings: np.ndarray, label: int):
    positions = [index for index, record in enumerate(records) if record.label == label]
    if not positions:
        raise ValueError(f"training corpus has no label={label} records")
    bucket_embeddings = _as_float32(embeddings[positions])
    index = faiss.IndexFlatIP(bucket_embeddings.shape[1])
    index.add(bucket_embeddings)
    return index, [records[position] for position in positions]


def run_balanced_retrieval(
    *,
    train_records: Sequence[ClinicalRecord],
    test_records: Sequence[ClinicalRecord],
    encoder,
    top_k: int = 4,
) -> tuple[list[dict], dict, dict[int, object], dict[int, list[str]]]:
    if top_k < 2:
        raise ValueError("top_k must be at least 2 to retrieve both risk classes")
    positive_k = (top_k + 1) // 2
    negative_k = top_k // 2
    train_embeddings = _as_float32(
        encoder.encode_numpy([record.retrieval_text for record in train_records], role="document")
    )
    query_embeddings = _as_float32(
        encoder.encode_numpy([record.query_text for record in test_records], role="query")
    )
    label_k = {1: positive_k, 0: negative_k}
    buckets = {label: _build_bucket(train_records, train_embeddings, label) for label in (0, 1)}
    predictions: list[dict] = []
    for row_index, record in enumerate(test_records):
        neighbors: list[dict] = []
        means: dict[int, float] = {}
        for label in (0, 1):
            index, bucket_records = buckets[label]
            limit = min(label_k[label], len(bucket_records))
            scores, positions = index.search(query_embeddings[row_index : row_index + 1], limit)
            means[label] = float(np.mean(scores[0]))
            for score, position in zip(scores[0], positions[0]):
                neighbor = bucket_records[int(position)]
                neighbors.append({"record_id": neighbor.record_id, "label": label, "score": float(score)})
        predictions.append(
            {
                "record_id": record.record_id,
                "gold_label": record.label,
                "center": record.center,
                "secondary_endpoint": record.secondary_endpoint,
                "risk_score": means[1] - means[0],
                "positive_mean_similarity": means[1],
                "negative_mean_similarity": means[0],
                "neighbors": neighbors,
            }
        )
    metrics = compute_metrics(predictions)
    metrics["by_center"] = compute_group_metrics(predictions)
    indexes = {label: buckets[label][0] for label in (0, 1)}
    record_id_maps = {label: [r.record_id for r in buckets[label][1]] for label in (0, 1)}
    return predictions, metrics, indexes, record_id_maps


def write_outputs(
    output_dir: Path,
    predictions: Sequence[dict],
    metrics: dict,
    indexes: dict[int, object],
    record_id_maps: dict[int, list[str]],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_dir = output_dir / "artifacts"
    index_dir.mkdir(parents=True, exist_ok=True)
    for label, index in indexes.items():
        ids = record_id_maps.get(label)
        if ids is None:
            raise ValueError(f"missing record ID map for label {label}")
        if index.ntotal != len(ids):
            raise ValueError(
                f"FAISS/ID map size mismatch for label {label}: "
                f"{index.ntotal} vectors but {len(ids)} record IDs"
            )

    id_map_paths: dict[int, Path] = {}
    for label, index in indexes.items():
        ids = record_id_maps[label]
        faiss.write_index(index, str(index_dir / f"label_{label}.faiss"))
        map_path = index_dir / f"label_{label}_ids.json"
        map_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
        id_map_paths[label] = map_path
    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "record_id",
            "gold_label",
            "center",
            "secondary_endpoint",
            "risk_score",
            "positive_mean_similarity",
            "negative_mean_similarity",
            "neighbors",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            exported = dict(row)
            exported["neighbors"] = json.dumps(row["neighbors"], ensure_ascii=False)
            writer.writerow(exported)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "positive_index_path": str(index_dir / "label_1.faiss"),
        "negative_index_path": str(index_dir / "label_0.faiss"),
        "positive_id_map_path": str(id_map_paths[1]),
        "negative_id_map_path": str(id_map_paths[0]),
    }


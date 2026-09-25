import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from medseek.data import load_records, split_records
from medseek.demo import DEFAULT_MODEL, build_parser, run_demo
from medseek.retrieval import run_balanced_retrieval, write_outputs


EXAMPLE = Path(__file__).parents[1] / "data" / "data.csv"


class DeterministicEncoder:
    """Small offline encoder used only to test retrieval plumbing."""

    def encode_numpy(self, texts, *, role):
        rows = []
        high_risk_terms = ("shock", "hypotension", "rigid", "sepsis", "bleeding", "instability")
        for text in texts:
            lowered = text.lower()
            risk = float(sum(term in lowered for term in high_risk_terms))
            vector = np.asarray([risk + 0.1, 1.0, float(len(lowered) % 17) / 17.0], dtype="float32")
            vector /= np.linalg.norm(vector)
            rows.append(vector)
        return np.stack(rows).astype("float32")


def test_public_checkpoint_is_the_demo_default() -> None:
    args = build_parser().parse_args(["--data", str(EXAMPLE)])
    assert DEFAULT_MODEL == "Yjian1998/medseek"
    assert args.model_path == DEFAULT_MODEL


def test_end_to_end_pipeline_without_model_download(tmp_path: Path) -> None:
    summary = run_demo(data_path=EXAMPLE, output_dir=tmp_path, encoder=DeterministicEncoder())
    assert summary["train_records"] == 80
    assert summary["test_records"] == 5
    assert Path(summary["positive_index_path"]).is_file()
    assert Path(summary["negative_index_path"]).is_file()
    assert Path(summary["predictions_path"]).is_file()
    assert Path(summary["metrics_path"]).is_file()
    assert summary["metrics"]["count"] == 5
    negative_map_path = Path(summary["negative_id_map_path"])
    positive_map_path = Path(summary["positive_id_map_path"])
    assert negative_map_path.is_file()
    assert positive_map_path.is_file()

    train, _ = split_records(load_records(EXAMPLE))
    expected = {label: [row.record_id for row in train if row.label == label] for label in (0, 1)}
    for label, map_path in ((0, negative_map_path), (1, positive_map_path)):
        ids = json.loads(map_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(tmp_path / "artifacts" / f"label_{label}.faiss"))
        assert ids == expected[label]
        assert len(ids) == index.ntotal


@pytest.mark.parametrize("top_k", [2, 3, 4, 5])
def test_top_k_is_total_neighbor_count(top_k: int) -> None:
    train, test = split_records(load_records(EXAMPLE))
    predictions, _, _, _ = run_balanced_retrieval(
        train_records=train,
        test_records=test[:1],
        encoder=DeterministicEncoder(),
        top_k=top_k,
    )
    neighbors = predictions[0]["neighbors"]
    assert len(neighbors) == top_k
    assert {neighbor["label"] for neighbor in neighbors} == {0, 1}
    assert sum(neighbor["label"] == 1 for neighbor in neighbors) == (top_k + 1) // 2


def test_top_k_one_is_rejected() -> None:
    train, test = split_records(load_records(EXAMPLE))
    with pytest.raises(ValueError, match="at least 2"):
        run_balanced_retrieval(
            train_records=train,
            test_records=test[:1],
            encoder=DeterministicEncoder(),
            top_k=1,
        )


def test_index_mapping_mismatch_fails_before_writing(tmp_path: Path) -> None:
    train, test = split_records(load_records(EXAMPLE))
    predictions, metrics, indexes, record_id_maps = run_balanced_retrieval(
        train_records=train,
        test_records=test[:1],
        encoder=DeterministicEncoder(),
        top_k=2,
    )
    record_id_maps[1] = record_id_maps[1][:-1]
    with pytest.raises(ValueError, match="FAISS/ID map size mismatch"):
        write_outputs(tmp_path, predictions, metrics, indexes, record_id_maps)
    assert not list((tmp_path / "artifacts").glob("*.faiss"))

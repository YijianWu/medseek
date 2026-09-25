from pathlib import Path

import pytest

from medseek.data import load_records, split_records


EXAMPLE = Path(__file__).parents[1] / "data" / "data.csv"


def test_sample_dataset_is_balanced_and_split() -> None:
    records = load_records(EXAMPLE)
    train, test = split_records(records)
    assert len(train) == 80
    assert len(test) == 5
    assert {record.label for record in train} == {0, 1}
    assert {record.label for record in test} == {0, 1}
    assert all(record.record_id.startswith("SYN-") for record in records)


def test_loader_rejects_non_binary_label(tmp_path: Path) -> None:
    source = tmp_path / "bad.csv"
    source.write_text(
        "record_id,split,label,query_text,retrieval_text\nX,train,2,query,document\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="label must be 0 or 1"):
        load_records(source)


def test_csv_accepts_label_high_risk_column(tmp_path: Path) -> None:
    source = tmp_path / "alt.csv"
    source.write_text(
        "record_id,split,label_high_risk,query_text,retrieval_text\n"
        "A,train,1,query a,document a\n"
        "B,test,0,query b,document b\n",
        encoding="utf-8",
    )
    records = load_records(source)
    assert records[0].label == 1
    assert records[1].label == 0


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [("1", "1", 1), ("", "0", 0), ("1", "", 1)],
)
def test_csv_resolves_two_label_columns(
    tmp_path: Path, first: str, second: str, expected: int
) -> None:
    source = tmp_path / "labels.csv"
    source.write_text(
        "record_id,split,label,label_high_risk,query_text,retrieval_text\n"
        f"A,train,{first},{second},query,document\n",
        encoding="utf-8",
    )
    assert load_records(source)[0].label == expected


def test_csv_rejects_conflicting_label_columns(tmp_path: Path) -> None:
    source = tmp_path / "conflict.csv"
    source.write_text(
        "record_id,split,label,label_high_risk,query_text,retrieval_text\n"
        "A,train,0,1,query,document\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicting label values"):
        load_records(source)


def test_jsonl_uses_nonempty_label_alias(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(
        '{"record_id":"A","split":"test","label":"",'
        '"label_high_risk":1,"query_text":"query","retrieval_text":"document"}\n',
        encoding="utf-8",
    )
    assert load_records(source)[0].label == 1

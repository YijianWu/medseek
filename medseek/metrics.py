from __future__ import annotations

from typing import Sequence


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positive_count = sum(label == 1 for label in labels)
    if positive_count == 0:
        return None
    ranked = sorted(zip(scores, labels), reverse=True)
    true_positives = 0
    false_positives = 0
    result = 0.0
    start = 0
    while start < len(ranked):
        score = ranked[start][0]
        end = start
        group_positives = 0
        group_negatives = 0
        while end < len(ranked) and ranked[end][0] == score:
            if ranked[end][1] == 1:
                group_positives += 1
            else:
                group_negatives += 1
            end += 1

        true_positives += group_positives
        false_positives += group_negatives
        precision = true_positives / (true_positives + false_positives)
        result += precision * (group_positives / positive_count)
        start = end
    return result


def compute_metrics(rows: Sequence[dict]) -> dict[str, float | int | None]:
    labels = [int(row["gold_label"]) for row in rows]
    scores = [float(row["risk_score"]) for row in rows]
    return {
        "count": len(rows),
        "positive_count": sum(labels),
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": average_precision(labels, scores),
    }


def reciprocal_rank(relevance: Sequence[int]) -> float:
    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(relevance: Sequence[int], k: int, *, relevant_total: int) -> float:
    if k <= 0 or relevant_total <= 0:
        return 0.0
    return float(sum(relevance[:k])) / float(relevant_total)


def compute_group_metrics(rows: Sequence[dict], group_field: str = "center") -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        name = str(row.get(group_field) or "unknown")
        groups.setdefault(name, []).append(row)
    return {name: compute_metrics(group_rows) for name, group_rows in sorted(groups.items())}


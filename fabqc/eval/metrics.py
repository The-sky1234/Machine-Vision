from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray


def auroc(y_true: NDArray[np.integer] | list[int], y_score: NDArray[np.floating] | list[float]) -> float:
    """Mann-Whitney AUROC. 0.5 is chance, 1.0 is perfect ranking."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # P(score_pos > score_neg) + 0.5 P(tie)
    gt = np.sum(pos[:, None] > neg[None, :])
    eq = np.sum(pos[:, None] == neg[None, :])
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


@dataclass(slots=True)
class AttributionSummary:
    n_cases: int
    n_decided: int
    n_abstain: int
    n_correct: int
    n_wrong: int
    accuracy_among_decided: float | None
    abstain_rate: float
    correct_on_single_cause: float | None
    missed_single_cause: float | None
    wrong_cause_on_single: float | None
    abstain_on_ambiguous: float | None
    abstain_on_unexplained: float | None
    false_attribution_unexplained: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def summarize_attribution(rows: list[dict]) -> AttributionSummary:
    n = len(rows)
    decided = [r for r in rows if r["predicted"] != "abstain"]
    abstain = [r for r in rows if r["predicted"] == "abstain"]
    correct = [r for r in decided if r["predicted"] == r["true_cause"]]
    wrong = [r for r in decided if r["predicted"] != r["true_cause"]]

    single = [r for r in rows if r["bucket"] == "single_cause"]
    amb = [r for r in rows if r["bucket"] == "ambiguous"]
    unexp = [r for r in rows if r["bucket"] == "unexplained"]

    def _rate(subset: list[dict], pred: str) -> float | None:
        if not subset:
            return None
        return float(np.mean([r["predicted"] == pred for r in subset]))

    def _acc(subset: list[dict]) -> float | None:
        if not subset:
            return None
        return float(np.mean([r["predicted"] == r["true_cause"] for r in subset]))

    return AttributionSummary(
        n_cases=n,
        n_decided=len(decided),
        n_abstain=len(abstain),
        n_correct=len(correct),
        n_wrong=len(wrong),
        accuracy_among_decided=(len(correct) / len(decided) if decided else None),
        abstain_rate=(len(abstain) / n if n else 0.0),
        correct_on_single_cause=_acc(single),
        missed_single_cause=(
            float(np.mean([r["predicted"] == "abstain" for r in single])) if single else None
        ),
        wrong_cause_on_single=(
            float(np.mean([r["predicted"] not in {r["true_cause"], "abstain"} for r in single])) if single else None
        ),
        abstain_on_ambiguous=_rate(amb, "abstain"),
        abstain_on_unexplained=_rate(unexp, "abstain"),
        false_attribution_unexplained=(
            float(np.mean([r["predicted"] != "abstain" for r in unexp])) if unexp else None
        ),
    )

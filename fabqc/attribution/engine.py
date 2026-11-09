"""Deterministic attribution: temporal precedence plus noisy-OR.

The LLM is not used here. Scores come from FMEA edges, lag windows, and
excursion magnitude. The engine abstains when evidence is weak or two
causes are too close.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from fabqc.graph import LEAK_PROBABILITY, CausalEdge, edges_for_defect

Z_THRESH = 2.6
OSC_Z_THRESH = 7.5  # rolling-std z; nominal noise sits ~2–4, injected osc ~30+
MIN_DURATION_S = 3.0
MIN_SCORE = 0.34
AMBIGUITY_MARGIN = 0.10
AMBIGUITY_FLOOR = 0.25
BASELINE_S = 18.0


@dataclass(slots=True)
class Excursion:
    signal: str
    mode: str
    t_start: float
    t_end: float
    t_peak: float
    z_peak: float


@dataclass(slots=True)
class CauseScore:
    cause_id: str
    score: float
    lag_s: float | None
    z_peak: float | None
    edge_strength: float
    reason: str


@dataclass(slots=True)
class Attribution:
    decision: str  # cause_id or "abstain"
    abstain_reason: str | None
    scores: list[CauseScore] = field(default_factory=list)
    noisy_or_p: float = 0.0

    @property
    def cause_id(self) -> str | None:
        return None if self.decision == "abstain" else self.decision


def _mad_sigma(x: NDArray[np.float64]) -> tuple[float, float]:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return med, max(mad, 1e-6)


def _events_from_mask(t: NDArray[np.float64], z: NDArray[np.float64], mask: NDArray[np.bool_], min_s: float) -> list[tuple[int, int]]:
    events: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if t[min(j, n - 1)] - t[i] >= min_s or (j - i) >= min_s:
            events.append((i, j))
        i = j
    return events


def detect_excursions(
    t: NDArray[np.float64],
    series: dict[str, NDArray[np.float64]],
    signal: str,
    mode: str,
) -> list[Excursion]:
    x = np.asarray(series[signal], dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n_base = max(8, int(np.searchsorted(t, BASELINE_S)))
    base = x[:n_base]
    med, sig = _mad_sigma(base)
    z = (x - med) / sig

    if mode == "high":
        mask = z > Z_THRESH
        z_for_peak = z
    elif mode == "low":
        mask = z < -Z_THRESH
        z_for_peak = -z
    elif mode == "oscillation":
        win = 8
        kernel = np.ones(win) / win
        # Population std via rolling second moment.
        pad = np.pad(x, (win - 1, 0), mode="edge")
        roll_mean = np.convolve(pad, kernel, mode="valid")
        roll_sq = np.convolve(pad**2, kernel, mode="valid")
        roll_std = np.sqrt(np.clip(roll_sq - roll_mean**2, 0.0, None))
        bmed, bsig = _mad_sigma(roll_std[:n_base])
        z = (roll_std - bmed) / bsig
        mask = z > OSC_Z_THRESH
        z_for_peak = z
    else:
        raise ValueError(mode)

    found: list[Excursion] = []
    for i, j in _events_from_mask(t, z_for_peak, mask, MIN_DURATION_S):
        sl = slice(i, j)
        peak_i = i + int(np.argmax(z_for_peak[sl]))
        found.append(
            Excursion(
                signal=signal,
                mode=mode,
                t_start=float(t[i]),
                t_end=float(t[min(j - 1, len(t) - 1)]),
                t_peak=float(t[peak_i]),
                z_peak=float(z_for_peak[peak_i]),
            )
        )
    return found


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _temporal_kernel(lag: float, edge: CausalEdge) -> float:
    if lag < edge.min_lag_s or lag > edge.max_lag_s:
        return 0.0
    mid = 0.5 * (edge.min_lag_s + edge.max_lag_s)
    span = max(edge.max_lag_s - edge.min_lag_s, 1.0)
    return float(np.clip(1.0 - 0.25 * abs(lag - mid) / (span / 2.0), 0.55, 1.0))


def _score_edge(
    edge: CausalEdge,
    t: NDArray[np.float64],
    series: dict[str, NDArray[np.float64]],
    t_defect: float,
) -> CauseScore:
    events = detect_excursions(t, series, edge.signal, edge.mode)
    best = 0.0
    best_lag = None
    best_z = None
    best_reason = "no excursion in lag window"
    for ev in events:
        # Cause must precede the defect. Use event start (onset), not peak.
        lag = t_defect - ev.t_start
        if lag < 0:
            continue
        temporal = _temporal_kernel(lag, edge)
        if temporal <= 0:
            continue
        mag = _sigmoid((ev.z_peak - 2.4) / 1.1)
        score = edge.strength * temporal * mag
        if score > best:
            best = score
            best_lag = lag
            best_z = ev.z_peak
            best_reason = (
                f"{edge.signal} {edge.mode} onset at t={ev.t_start:.1f}s, "
                f"lag={lag:.1f}s, z={ev.z_peak:.2f}"
            )
    return CauseScore(
        cause_id=edge.cause_id,
        score=float(best),
        lag_s=best_lag,
        z_peak=best_z,
        edge_strength=edge.strength,
        reason=best_reason,
    )


def attribute_defect(
    defect_type: str,
    t_defect: float,
    t: NDArray[np.float64],
    series: dict[str, NDArray[np.float64]],
) -> Attribution:
    edges = edges_for_defect(defect_type)
    scores = [_score_edge(e, t, series, t_defect) for e in edges]
    scores.sort(key=lambda s: s.score, reverse=True)

    prod = 1.0
    for s in scores:
        prod *= 1.0 - min(s.score, 0.999)
    noisy_or_p = 1.0 - (1.0 - LEAK_PROBABILITY) * prod

    if not scores or scores[0].score < MIN_SCORE:
        return Attribution(
            decision="abstain",
            abstain_reason="insufficient_evidence",
            scores=scores,
            noisy_or_p=noisy_or_p,
        )
    # Two modelled causes both clearing the bar: refuse to pick a winner.
    n_viable = sum(1 for s in scores if s.score >= MIN_SCORE)
    close_second = (
        len(scores) >= 2
        and scores[1].score >= AMBIGUITY_FLOOR
        and (scores[0].score - scores[1].score) < AMBIGUITY_MARGIN
    )
    if n_viable >= 2 or close_second:
        return Attribution(
            decision="abstain",
            abstain_reason="ambiguous_causes",
            scores=scores,
            noisy_or_p=noisy_or_p,
        )
    return Attribution(
        decision=scores[0].cause_id,
        abstain_reason=None,
        scores=scores,
        noisy_or_p=noisy_or_p,
    )

"""Simulated process line: telemetry plus wafers, with injected known faults.

Ground-truth causes are recorded so attribution can be scored. The engine
does not receive these labels, only the raw series and the FMEA graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fabqc.graph import CAUSAL_GRAPH, NOMINAL, SIGNALS, CausalEdge, edge_by_cause
from fabqc.simulator.images import render_part, save_png

WAFER_CYCLE_S = 12.0
N_WAFERS = 16
PRE_S = 20.0
POST_S = 8.0
SAMPLE_HZ = 1.0


@dataclass(slots=True)
class Wafer:
    wafer_id: str
    process_start: float
    process_end: float
    defect_type: str | None
    true_cause: str | None
    image: NDArray[np.uint8] | None = None
    image_path: str | None = None


@dataclass(slots=True)
class InjectedFault:
    cause_id: str
    t_start: float
    t_end: float
    magnitude: float


@dataclass(slots=True)
class SimulatedLot:
    lot_id: str
    scenario: str
    t: NDArray[np.float64]
    series: dict[str, NDArray[np.float64]]
    wafers: list[Wafer]
    injected: list[InjectedFault] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0


SCENARIOS = (
    "coolant_drift",
    "temp_spike",
    "pressure_oscillation",
    "rpm_drop",
    "ambiguous_scratch",
    "unexplained_scratch",
    "healthy",
    "distractor_pressure_on_scratch",
)


def _time_grid(rng: np.random.Generator) -> tuple[NDArray[np.float64], list[tuple[float, float]]]:
    duration = PRE_S + N_WAFERS * WAFER_CYCLE_S + POST_S
    n = int(duration * SAMPLE_HZ) + 1
    t = np.arange(n, dtype=np.float64) / SAMPLE_HZ
    wafers = []
    for i in range(N_WAFERS):
        start = PRE_S + i * WAFER_CYCLE_S + float(rng.uniform(0.0, 0.4))
        end = start + WAFER_CYCLE_S * 0.75
        wafers.append((start, end))
    return t, wafers


def _nominal_series(t: NDArray[np.float64], rng: np.random.Generator) -> dict[str, NDArray[np.float64]]:
    series = {}
    for name in SIGNALS:
        mu = NOMINAL[name]
        sig = NOMINAL["noise"][name]
        # Slow drift plus white noise so z-scores are well defined.
        slow = 0.15 * sig * np.sin(2 * np.pi * t / (len(t) * 0.8) + rng.uniform(0, 6))
        series[name] = mu + slow + rng.normal(0.0, sig, size=t.shape)
    return series


def _apply_high(series: NDArray[np.float64], t: NDArray[np.float64], t0: float, t1: float, delta: float) -> None:
    mask = (t >= t0) & (t <= t1)
    ramp = np.clip((t[mask] - t0) / 4.0, 0.0, 1.0)
    series[mask] += delta * ramp


def _apply_low(series: NDArray[np.float64], t: NDArray[np.float64], t0: float, t1: float, delta: float) -> None:
    mask = (t >= t0) & (t <= t1)
    ramp = np.clip((t[mask] - t0) / 3.0, 0.0, 1.0)
    series[mask] -= delta * ramp


def _apply_oscillation(
    series: NDArray[np.float64],
    t: NDArray[np.float64],
    t0: float,
    t1: float,
    amplitude: float,
    rng: np.random.Generator,
) -> None:
    mask = (t >= t0) & (t <= t1)
    freq = rng.uniform(0.35, 0.55)
    series[mask] += amplitude * np.sin(2 * np.pi * freq * (t[mask] - t0))


def _inject(edge: CausalEdge, t: NDArray[np.float64], series: dict[str, NDArray[np.float64]], rng: np.random.Generator) -> InjectedFault:
    t0 = float(rng.uniform(PRE_S + 4.0, PRE_S + 6 * WAFER_CYCLE_S))
    t1 = t0 + float(rng.uniform(14.0, 28.0))
    sig = series[edge.signal]
    if edge.mode == "high":
        mag = 6.5 * NOMINAL["noise"][edge.signal] + (8.0 if edge.signal.endswith("temp") else 4.0)
        if edge.signal == "coolant_temp":
            mag = rng.uniform(6.5, 9.0)
        elif edge.signal == "chamber_temp":
            mag = rng.uniform(7.0, 10.0)
        _apply_high(sig, t, t0, t1, mag)
    elif edge.mode == "low":
        mag = rng.uniform(180.0, 280.0)
        _apply_low(sig, t, t0, t1, mag)
    elif edge.mode == "oscillation":
        mag = rng.uniform(3.5, 6.0) * NOMINAL["noise"][edge.signal] if edge.signal != "spindle_rpm" else rng.uniform(90.0, 140.0)
        if edge.signal == "pressure":
            mag = rng.uniform(3.8, 5.5)
        _apply_oscillation(sig, t, t0, t1, mag, rng)
    else:
        raise ValueError(edge.mode)
    return InjectedFault(cause_id=edge.cause_id, t_start=t0, t_end=t1, magnitude=float(mag))


def _defect_from_injections(
    process_end: float,
    injected: list[InjectedFault],
    rng: np.random.Generator,
    force_unexplained: bool = False,
) -> tuple[str | None, str | None]:
    if force_unexplained:
        return "scratch", None

    candidates: list[tuple[float, str, str]] = []
    for fault in injected:
        for edge in CAUSAL_GRAPH:
            if edge.cause_id != fault.cause_id:
                continue
            lag = process_end - fault.t_start
            if edge.min_lag_s <= lag <= edge.max_lag_s and process_end >= fault.t_start:
                # Later wafers in the window still count if the fault is active or recently ended.
                if process_end <= fault.t_end + edge.max_lag_s:
                    weight = edge.strength * (1.0 - 0.15 * abs(lag - (edge.min_lag_s + edge.max_lag_s) / 2) / edge.max_lag_s)
                    candidates.append((weight, edge.defect_type, fault.cause_id))
    if not candidates:
        return None, None
    # Strongest matching cause wins for ground truth. If two causes share the
    # same defect type, mark true_cause as ambiguous (None) only when caller asks.
    candidates.sort(reverse=True)
    _, defect, cause = candidates[0]
    return defect, cause


def generate_lot(
    lot_id: str,
    scenario: str,
    rng: np.random.Generator,
    render_images: bool = False,
) -> SimulatedLot:
    t, wafer_windows = _time_grid(rng)
    series = _nominal_series(t, rng)
    injected: list[InjectedFault] = []

    def _must_inject(cause_id: str, defect_type: str) -> InjectedFault:
        edge = edge_by_cause(cause_id, defect_type)
        if edge is None:
            raise KeyError(cause_id)
        return _inject(edge, t, series, rng)

    if scenario == "coolant_drift":
        injected.append(_must_inject("coolant_drift", "scratch"))
    elif scenario == "temp_spike":
        injected.append(_must_inject("temp_spike", "stain"))
    elif scenario == "pressure_oscillation":
        injected.append(_must_inject("pressure_oscillation", "particle"))
    elif scenario == "rpm_drop":
        injected.append(_must_inject("rpm_drop", "uneven"))
    elif scenario == "ambiguous_scratch":
        coolant = edge_by_cause("coolant_drift", "scratch")
        assert coolant is not None
        first = _inject(coolant, t, series, rng)
        injected.append(first)
        second = InjectedFault(
            cause_id="spindle_vibration",
            t_start=first.t_start + 2.0,
            t_end=first.t_end + 2.0,
            magnitude=120.0,
        )
        _apply_oscillation(series["spindle_rpm"], t, second.t_start, second.t_end, 120.0, rng)
        injected.append(second)
    elif scenario == "unexplained_scratch":
        pass
    elif scenario == "healthy":
        pass
    elif scenario == "distractor_pressure_on_scratch":
        coolant = edge_by_cause("coolant_drift", "scratch")
        assert coolant is not None
        injected.append(_inject(coolant, t, series, rng))
        t0 = injected[0].t_start + 1.0
        # Wrong-parent blip (pressure is not an FMEA parent of scratch).
        _apply_oscillation(series["pressure"], t, t0, t0 + 8.0, 0.55, rng)
        # Same-parent but sub-threshold spindle wiggle; detector should ignore it.
        _apply_oscillation(series["spindle_rpm"], t, t0, t0 + 6.0, 18.0, rng)
    else:
        raise ValueError(f"unknown scenario {scenario}")

    wafers: list[Wafer] = []
    unexplained_assigned = False
    for i, (start, end) in enumerate(wafer_windows):
        force = False
        if scenario == "unexplained_scratch" and not unexplained_assigned and i == N_WAFERS // 2:
            force = True
            unexplained_assigned = True
        defect, cause = _defect_from_injections(end, injected, rng, force_unexplained=force)
        if scenario == "ambiguous_scratch" and defect == "scratch":
            cause = None  # two viable parents; ground truth is "abstain"
            true_flag = "__ambiguous__"
        else:
            true_flag = cause
        image = render_part(defect, rng) if render_images else None
        wafers.append(
            Wafer(
                wafer_id=f"{lot_id}-W{i:02d}",
                process_start=start,
                process_end=end,
                defect_type=defect,
                true_cause=true_flag,
                image=image,
            )
        )

    return SimulatedLot(lot_id=lot_id, scenario=scenario, t=t, series=series, wafers=wafers, injected=injected)


def generate_dataset(
    n_per_scenario: dict[str, int],
    seed: int = 7,
    render_images: bool = False,
    image_dir: str | Path | None = None,
) -> list[SimulatedLot]:
    rng = np.random.default_rng(seed)
    lots: list[SimulatedLot] = []
    out = Path(image_dir) if image_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    k = 0
    for scenario, n in n_per_scenario.items():
        for _ in range(n):
            lot = generate_lot(f"LOT{k:04d}", scenario, rng, render_images=render_images)
            if out and render_images:
                lot_dir = out / lot.lot_id
                lot_dir.mkdir(parents=True, exist_ok=True)
                for wafer in lot.wafers:
                    if wafer.image is None:
                        continue
                    path = lot_dir / f"{wafer.wafer_id}.png"
                    save_png(wafer.image, str(path))
                    wafer.image_path = str(path)
            lots.append(lot)
            k += 1
    return lots

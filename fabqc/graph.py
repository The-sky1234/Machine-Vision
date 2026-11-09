"""Process FMEA: which equipment excursion can produce which visual defect.

This graph is known process knowledge. The attribution engine is allowed to
use it. It is not allowed to see the simulator's injection timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CausalEdge:
    cause_id: str
    signal: str
    mode: str  # high | low | oscillation
    defect_type: str
    strength: float  # noisy-OR causal strength in (0, 1)
    min_lag_s: float
    max_lag_s: float
    description: str


# One-to-many is allowed so ambiguous lots can have two viable parents.
CAUSAL_GRAPH: tuple[CausalEdge, ...] = (
    CausalEdge(
        cause_id="coolant_drift",
        signal="coolant_temp",
        mode="high",
        defect_type="scratch",
        strength=0.88,
        min_lag_s=8.0,
        max_lag_s=80.0,
        description="Coolant temperature high vs baseline; scratches appear on later wafers.",
    ),
    CausalEdge(
        cause_id="spindle_vibration",
        signal="spindle_rpm",
        mode="oscillation",
        defect_type="scratch",
        strength=0.62,
        min_lag_s=6.0,
        max_lag_s=70.0,
        description="Spindle RPM unstable; weaker second parent of scratches.",
    ),
    CausalEdge(
        cause_id="temp_spike",
        signal="chamber_temp",
        mode="high",
        defect_type="stain",
        strength=0.84,
        min_lag_s=5.0,
        max_lag_s=70.0,
        description="Chamber temperature spike; thermal stain on subsequent wafers.",
    ),
    CausalEdge(
        cause_id="pressure_oscillation",
        signal="pressure",
        mode="oscillation",
        defect_type="particle",
        strength=0.80,
        min_lag_s=6.0,
        max_lag_s=75.0,
        description="Chamber pressure oscillating; particle contamination.",
    ),
    CausalEdge(
        cause_id="rpm_drop",
        signal="spindle_rpm",
        mode="low",
        defect_type="uneven",
        strength=0.78,
        min_lag_s=5.0,
        max_lag_s=70.0,
        description="Spindle RPM drop; uneven surface / thickness.",
    ),
)

SIGNALS: tuple[str, ...] = ("coolant_temp", "chamber_temp", "pressure", "spindle_rpm")
DEFECT_TYPES: tuple[str, ...] = ("scratch", "stain", "particle", "uneven")

NOMINAL = {
    "coolant_temp": 22.0,
    "chamber_temp": 65.0,
    "pressure": 45.0,
    "spindle_rpm": 2400.0,
    "noise": {
        "coolant_temp": 0.15,
        "chamber_temp": 0.25,
        "pressure": 0.20,
        "spindle_rpm": 8.0,
    },
}

LEAK_PROBABILITY = 0.04  # P(defect | no modelled cause)


def edges_for_defect(defect_type: str) -> tuple[CausalEdge, ...]:
    return tuple(e for e in CAUSAL_GRAPH if e.defect_type == defect_type)


def edge_by_cause(cause_id: str, defect_type: str | None = None) -> CausalEdge | None:
    for edge in CAUSAL_GRAPH:
        if edge.cause_id != cause_id:
            continue
        if defect_type is None or edge.defect_type == defect_type:
            return edge
    return None

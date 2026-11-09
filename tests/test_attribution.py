import numpy as np

from fabqc.attribution.engine import attribute_defect
from fabqc.simulator.line import generate_lot


def test_single_cause_recovers_coolant_drift():
    rng = np.random.default_rng(3)
    lot = generate_lot("C1", "coolant_drift", rng)
    scratches = [w for w in lot.wafers if w.defect_type == "scratch"]
    assert scratches
    result = attribute_defect("scratch", scratches[0].process_end, lot.t, lot.series)
    assert result.decision == "coolant_drift"


def test_unexplained_defect_abstains():
    rng = np.random.default_rng(4)
    lot = generate_lot("U1", "unexplained_scratch", rng)
    wafer = next(w for w in lot.wafers if w.defect_type)
    result = attribute_defect("scratch", wafer.process_end, lot.t, lot.series)
    assert result.decision == "abstain"
    assert result.abstain_reason == "insufficient_evidence"


def test_does_not_use_future_excursions():
    rng = np.random.default_rng(5)
    lot = generate_lot("C2", "coolant_drift", rng)
    fault = lot.injected[0]
    # Query a time strictly before the cause exists.
    t_query = fault.t_start - 1.0
    result = attribute_defect("scratch", t_query, lot.t, lot.series)
    assert result.decision == "abstain"


def test_ambiguous_parents_abstain():
    rng = np.random.default_rng(6)
    lot = generate_lot("A1", "ambiguous_scratch", rng)
    scratches = [w for w in lot.wafers if w.defect_type == "scratch"]
    assert scratches
    result = attribute_defect("scratch", scratches[-1].process_end, lot.t, lot.series)
    assert result.decision == "abstain"
    assert result.abstain_reason == "ambiguous_causes"

import numpy as np

from fabqc.simulator.line import generate_lot


def test_healthy_lot_has_no_defects():
    rng = np.random.default_rng(0)
    lot = generate_lot("H0", "healthy", rng)
    assert all(w.defect_type is None for w in lot.wafers)
    assert lot.injected == []


def test_coolant_drift_creates_later_scratches():
    rng = np.random.default_rng(1)
    lot = generate_lot("C0", "coolant_drift", rng)
    assert len(lot.injected) == 1
    fault = lot.injected[0]
    scratches = [w for w in lot.wafers if w.defect_type == "scratch"]
    assert scratches, "injected coolant drift should mark some later wafers"
    assert all(w.process_end >= fault.t_start for w in scratches)
    assert all(w.true_cause == "coolant_drift" for w in scratches)
    early = [w for w in lot.wafers if w.process_end < fault.t_start]
    assert all(w.defect_type is None for w in early)


def test_unexplained_scratch_has_no_equipment_cause():
    rng = np.random.default_rng(2)
    lot = generate_lot("U0", "unexplained_scratch", rng)
    defects = [w for w in lot.wafers if w.defect_type]
    assert len(defects) == 1
    assert defects[0].true_cause is None
    assert lot.injected == []

import numpy as np

from fabqc.eval.metrics import auroc, summarize_attribution


def test_auroc_perfect_and_chance():
    y = [0, 0, 1, 1]
    assert auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert 0.4 < auroc(y, [0.5, 0.5, 0.5, 0.5]) < 0.6


def test_summary_counts_abstain_and_accuracy():
    rows = [
        {"predicted": "coolant_drift", "true_cause": "coolant_drift", "bucket": "single_cause"},
        {"predicted": "abstain", "true_cause": "abstain", "bucket": "ambiguous"},
        {"predicted": "temp_spike", "true_cause": "coolant_drift", "bucket": "single_cause"},
    ]
    s = summarize_attribution(rows)
    assert s.n_decided == 2
    assert s.n_correct == 1
    assert s.abstain_rate == np.float64(1 / 3) or abs(s.abstain_rate - 1 / 3) < 1e-9

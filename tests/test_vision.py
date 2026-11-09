import numpy as np
import pytest

from fabqc.simulator.images import render_part

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from fabqc.vision.patchcore import LightweightPatchCore  # noqa: E402


def test_patchcore_ranks_synthetic_defects_above_good_parts():
    rng = np.random.default_rng(8)
    train = [render_part(None, rng) for _ in range(12)]
    good = [render_part(None, rng) for _ in range(6)]
    bad = [render_part("scratch", rng) for _ in range(6)]
    model = LightweightPatchCore(image_size=128, coreset_ratio=0.4)
    model.fit(train, rng=rng)
    good_scores = [model.predict(img).score for img in good]
    bad_scores = [model.predict(img).score for img in bad]
    assert np.mean(bad_scores) > np.mean(good_scores)

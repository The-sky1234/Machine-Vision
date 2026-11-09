"""Run simulator, attribution, and PatchCore. Writes numbers that were actually computed."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from fabqc.attribution.engine import attribute_defect
from fabqc.eval.metrics import auroc, summarize_attribution
from fabqc.simulator.images import render_part
from fabqc.simulator.line import generate_dataset
from fabqc.vision.patchcore import LightweightPatchCore


DEFAULT_SCENARIOS = {
    "coolant_drift": 18,
    "temp_spike": 18,
    "pressure_oscillation": 18,
    "rpm_drop": 18,
    "distractor_pressure_on_scratch": 12,
    "ambiguous_scratch": 16,
    "unexplained_scratch": 16,
    "healthy": 8,
}


def _bucket(true_cause: str | None, defect_type: str | None) -> str | None:
    if defect_type is None:
        return None
    if true_cause == "__ambiguous__":
        return "ambiguous"
    if true_cause is None:
        return "unexplained"
    return "single_cause"


def evaluate_attribution(lots, seed: int) -> dict:
    rows = []
    for lot in lots:
        for wafer in lot.wafers:
            if wafer.defect_type is None:
                continue
            result = attribute_defect(wafer.defect_type, wafer.process_end, lot.t, lot.series)
            bucket = _bucket(wafer.true_cause, wafer.defect_type)
            true_cause = wafer.true_cause if wafer.true_cause not in (None, "__ambiguous__") else None
            expected = "abstain" if bucket in {"ambiguous", "unexplained"} else true_cause
            rows.append(
                {
                    "lot_id": lot.lot_id,
                    "wafer_id": wafer.wafer_id,
                    "scenario": lot.scenario,
                    "defect_type": wafer.defect_type,
                    "true_cause": true_cause,
                    "expected": expected,
                    "predicted": result.decision,
                    "abstain_reason": result.abstain_reason,
                    "noisy_or_p": result.noisy_or_p,
                    "top_score": result.scores[0].score if result.scores else 0.0,
                    "top_cause": result.scores[0].cause_id if result.scores else None,
                    "bucket": bucket,
                    "correct": result.decision == expected,
                }
            )
    summary = summarize_attribution(
        [
            {
                "predicted": r["predicted"],
                "true_cause": r["expected"],
                "bucket": r["bucket"],
            }
            for r in rows
        ]
    )
    payload = {
        "seed": seed,
        "n_lots": len(lots),
        "n_defective_wafers": len(rows),
        "summary": summary.as_dict(),
        "by_scenario": {},
        "disclaimer": (
            "Telemetry and the defect-to-fault mapping are synthetic. "
            "These numbers describe the constructed line, not a real factory."
        ),
    }
    for scenario in sorted({r["scenario"] for r in rows}):
        sub = [r for r in rows if r["scenario"] == scenario]
        payload["by_scenario"][scenario] = {
            "n": len(sub),
            "accuracy": float(np.mean([r["correct"] for r in sub])),
            "abstain_rate": float(np.mean([r["predicted"] == "abstain" for r in sub])),
        }
    payload["cases"] = rows
    return payload


def evaluate_vision(seed: int, n_train_good: int = 48, n_test: int = 32) -> dict:
    rng = np.random.default_rng(seed)
    train_good = [render_part(None, rng) for _ in range(n_train_good)]
    calib_good = [render_part(None, rng) for _ in range(24)]
    test_good = [render_part(None, rng) for _ in range(n_test)]
    test_bad = []
    labels_bad = []
    for kind in ("scratch", "stain", "particle", "uneven"):
        for _ in range(n_test // 4):
            test_bad.append(render_part(kind, rng))
            labels_bad.append(kind)

    model = LightweightPatchCore(image_size=128, coreset_ratio=0.2)
    model.fit(train_good, rng=rng)
    threshold = model.calibrate_threshold(calib_good, quantile=0.90)

    good_scores = [model.predict(img).score for img in test_good]
    bad_scores = [model.predict(img).score for img in test_bad]
    y = [0] * len(good_scores) + [1] * len(bad_scores)
    s = good_scores + bad_scores
    auc = auroc(y, s)

    pred_good = [sc >= threshold for sc in good_scores]
    pred_bad = [sc >= threshold for sc in bad_scores]
    tp = int(np.sum(pred_bad))
    fn = len(pred_bad) - tp
    fp = int(np.sum(pred_good))
    tn = len(pred_good) - fp

    return {
        "seed": seed,
        "n_train_good": n_train_good,
        "n_test_good": len(test_good),
        "n_test_defect": len(test_bad),
        "threshold_p90_on_holdout_good": threshold,
        "auroc": auc,
        "tpr": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "mean_score_good": float(np.mean(good_scores)),
        "mean_score_defect": float(np.mean(bad_scores)),
        "good_scores": good_scores,
        "bad_scores": bad_scores,
        "bad_labels": labels_bad,
        "test_good_images": test_good,
        "test_bad_images": test_bad,
        "model": model,
        "disclaimer": (
            "Images are synthetic textures, not MVTec and not factory photos. "
            "AUROC is for this generator only."
        ),
    }


def _plot_lot(lot, path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    colors = {
        "coolant_temp": "#1f77b4",
        "chamber_temp": "#d62728",
        "pressure": "#2ca02c",
        "spindle_rpm": "#9467bd",
    }
    defect_times = [w.process_end for w in lot.wafers if w.defect_type]
    for ax, (name, y) in zip(axes, lot.series.items()):
        ax.plot(lot.t, y, color=colors.get(name, "black"), lw=1.1)
        ax.set_ylabel(name)
        for f in lot.injected:
            ax.axvspan(f.t_start, f.t_end, color="orange", alpha=0.18)
        for td in defect_times:
            ax.axvline(td, color="crimson", ls="--", lw=0.8, alpha=0.7)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(f"{lot.lot_id} scenario={lot.scenario}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_scores(vision: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vision["good_scores"], bins=12, alpha=0.65, label="good", color="#1f77b4")
    ax.hist(vision["bad_scores"], bins=12, alpha=0.65, label="defect", color="#d62728")
    ax.axvline(vision["threshold_p90_on_holdout_good"], color="black", ls="--", label="threshold")
    ax.set_xlabel("PatchCore image score")
    ax.set_ylabel("count")
    ax.set_title(f"synthetic-image AUROC={vision['auroc']:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_heatmap_examples(vision: dict, path: Path) -> None:
    model = vision["model"]
    good = vision["test_good_images"][0]
    bad = vision["test_bad_images"][0]
    hg = model.predict(good).heatmap
    hb = model.predict(bad).heatmap
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    axes[0, 0].imshow(good)
    axes[0, 0].set_title("good part")
    axes[0, 1].imshow(hg, cmap="magma")
    axes[0, 1].set_title("good heatmap")
    axes[1, 0].imshow(bad)
    axes[1, 0].set_title(f"defect ({vision['bad_labels'][0]})")
    axes[1, 1].imshow(hb, cmap="magma")
    axes[1, 1].set_title("defect heatmap")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _upsample_overlay(image: np.ndarray, heat: np.ndarray) -> np.ndarray:
    heat_img = Image.fromarray((heat / (heat.max() + 1e-9) * 255).astype(np.uint8)).resize(
        (image.shape[1], image.shape[0]), Image.BILINEAR
    )
    h = np.asarray(heat_img).astype(np.float32) / 255.0
    overlay = image.astype(np.float32) / 255.0
    overlay[..., 0] = np.clip(overlay[..., 0] * 0.55 + h * 0.9, 0, 1)
    return (overlay * 255).astype(np.uint8)


def run(out_dir: str | Path, seed: int = 7) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lots = generate_dataset(DEFAULT_SCENARIOS, seed=seed, render_images=False)
    attr = evaluate_attribution(lots, seed=seed)

    # Example lots for the report
    examples = {}
    for scenario in ("coolant_drift", "ambiguous_scratch", "unexplained_scratch"):
        lot = next(l for l in lots if l.scenario == scenario)
        examples[scenario] = lot.lot_id
        _plot_lot(lot, out / f"telemetry_{scenario}.png")

    vision = evaluate_vision(seed=seed)
    _plot_scores(vision, out / "vision_score_hist.png")
    _plot_heatmap_examples(vision, out / "vision_heatmaps.png")

    # Overlay PNG for the report
    overlay = _upsample_overlay(vision["test_bad_images"][0], vision["model"].predict(vision["test_bad_images"][0]).heatmap)
    Image.fromarray(overlay).save(out / "defect_overlay.png")
    Image.fromarray(vision["test_good_images"][0]).save(out / "good_example.png")
    Image.fromarray(vision["test_bad_images"][0]).save(out / "defect_example.png")

    pipeline_lots = generate_dataset(
        {"coolant_drift": 6, "unexplained_scratch": 6, "healthy": 4},
        seed=seed + 11,
        render_images=True,
    )
    model = vision["model"]
    threshold = vision["threshold_p90_on_holdout_good"]
    pipe_rows = []
    for lot in pipeline_lots:
        for wafer in lot.wafers:
            if wafer.image is None:
                continue
            vis = model.predict(wafer.image)
            flagged = vis.score >= threshold
            predicted_cause = None
            if flagged:
                if wafer.defect_type is None:
                    predicted_cause = "false_vision_flag"
                else:
                    att = attribute_defect(wafer.defect_type, wafer.process_end, lot.t, lot.series)
                    predicted_cause = att.decision
            expected_flag = wafer.defect_type is not None
            pipe_rows.append(
                {
                    "scenario": lot.scenario,
                    "flagged": flagged,
                    "true_defect": expected_flag,
                    "true_cause": wafer.true_cause,
                    "predicted_cause": predicted_cause,
                    "vision_score": vis.score,
                }
            )
    n_def = sum(r["true_defect"] for r in pipe_rows)
    n_good = len(pipe_rows) - n_def
    vision_tp = sum(r["flagged"] and r["true_defect"] for r in pipe_rows)
    vision_fp = sum(r["flagged"] and not r["true_defect"] for r in pipe_rows)
    rca_ok = 0
    rca_n = 0
    for r in pipe_rows:
        if not (r["flagged"] and r["true_defect"]):
            continue
        rca_n += 1
        if r["true_cause"] in (None, "__ambiguous__"):
            rca_ok += r["predicted_cause"] == "abstain"
        else:
            rca_ok += r["predicted_cause"] == r["true_cause"]
    pipeline = {
        "n_wafers": len(pipe_rows),
        "vision_tpr": vision_tp / max(n_def, 1),
        "vision_fpr": vision_fp / max(n_good, 1),
        "attribution_correct_given_detected_defect": (rca_ok / rca_n) if rca_n else None,
        "n_detected_defects": rca_n,
        "disclaimer": (
            "Same synthetic generator as the standalone evals. "
            "Vision AUROC/TPR on painted defects is not an industrial claim."
        ),
    }

    report = {
        "seed": seed,
        "disclaimer": attr["disclaimer"],
        "attribution": {k: v for k, v in attr.items() if k != "cases"},
        "vision": {k: v for k, v in vision.items() if k not in {"model", "test_good_images", "test_bad_images", "good_scores", "bad_scores", "bad_labels"}},
        "pipeline": pipeline,
        "example_lots": examples,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    (out / "attribution_cases.json").write_text(json.dumps(attr["cases"], indent=2))
    return report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    report = run(args.out, seed=args.seed)
    print(json.dumps({k: report[k] for k in ("seed", "disclaimer", "attribution", "vision") if k != "x"}, indent=2))

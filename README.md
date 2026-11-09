# Machine-Vision

Visual defect detection plus **deterministic** equipment-timeline root-cause attribution.

This is not a LangGraph demo and not a MongoDB Atlas demo. A lightweight PatchCore flags a defective part; a noisy-OR engine with temporal precedence then scores equipment excursions that **started before** the defect. The LLM does not choose the cause. Corrective actions are not executed in this revision.

**The line is simulated.** Wafer images are painted textures. Equipment traces and the defect-to-fault mapping are injected with known labels so attribution can be scored. Numbers below are from that constructed scene. They are not factory performance.

The MongoDB reference app that was imported earlier remains in `backend/` and `frontend/` (MIT, see `LICENSE` and `README.upstream.md`). It is not required to run `fabqc`.

## Results (seed=7, actually executed)

Command: `python -m fabqc.cli eval --out results --seed 7`

| Metric | Value | What it means |
|---|---|---|
| Attribution cases | 584 defective wafers / 124 lots | Simulated CMP-like cycle, 1 Hz telemetry |
| Accuracy among decided | **97.5%** (434/445) | Exact cause match when the engine did not abstain |
| Wrong cause on single-fault lots | **0%** | Errors on decided rows were "picked the stronger of two true parents" on ambiguous lots, not a wrong FMEA edge |
| Single-cause recall | 91.9% | Remaining 8.1% abstained (mostly lag-window edge or competing spindle oscillation) |
| Abstain on unexplained defects | **100%** (16/16) | Random scratch, no equipment fault |
| Abstain on two-parent lots | 88.5% (85/96) | Coolant drift and spindle vibration both in the window |
| PatchCore AUROC (synthetic images) | **1.000** | 48 good train / 32 good test / 32 defect test, ResNet18 CPU |
| PatchCore TPR / FPR | 1.00 / 0.031 | Threshold = 90th percentile of hold-out good scores |
| Pipeline (vision then RCA, 256 wafers) | TPR 1.00, FPR 0.079, RCA 88.1% on 42 flagged defects | Different seed than the standalone split |

The vision AUROC is 1.0 because the painted scratches, stains, particles and banding are visually obvious against a smooth disk. That number must not be quoted as MVTec or line performance.

Plots written by the same command:

- `results/telemetry_coolant_drift.png` — coolant step, then later wafers marked defective
- `results/vision_score_hist.png` — good vs defect PatchCore scores
- `results/vision_heatmaps.png` — example disk and heatmap
- `results/metrics.json` — full JSON

## How attribution works

1. Detect excursions on each signal (robust z-score; rolling-std z for oscillation).
2. Keep events whose **onset precedes** the defect and whose lag sits in the FMEA window.
3. Score `strength × temporal kernel × magnitude sigmoid`.
4. Combine with noisy-OR. Abstain if the top score is weak, or if two causes both clear the bar.

The engine sees raw series and the FMEA graph in `fabqc/graph.py`. It does not see injection timestamps.

## Run

Python 3.11+, CPU is enough (no GPU in this environment).

```bash
uv venv .venv-fabqc && source .venv-fabqc/bin/activate
uv pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu
python -m pytest tests -q
python -m fabqc.cli eval --out results --seed 7
```

## Layout

```
fabqc/simulator/     # lots, telemetry, painted part images, injected faults
fabqc/attribution/   # noisy-OR + lag windows, abstain rules
fabqc/vision/        # reduced PatchCore (ResNet18 layer2, random coreset)
fabqc/eval/          # metrics that were actually run
tests/
results/             # last eval output
```

## Not in this revision

- LangGraph / report writing / human approval of parameter changes
- Real MVTec or factory images
- Atlas, Voyage, Bedrock

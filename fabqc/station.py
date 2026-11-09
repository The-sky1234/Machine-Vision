"""One inspection job: photograph the lot, then attribute flagged wafers."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

import numpy as np

from fabqc.attribution.engine import Attribution, attribute_defect
from fabqc.graph import edge_by_cause
from fabqc.labels import ACTIONS, SIGNAL_LABELS, cause_label, defect_label
from fabqc.simulator.images import render_part
from fabqc.simulator.line import SimulatedLot, generate_lot
from fabqc.vision.overlay import overlay_heatmap, png_data_url
from fabqc.vision.patchcore import LightweightPatchCore, VisionResult


@dataclass
class ProposedAction:
    action_id: str
    title: str
    detail: str
    status: str  # pending | approved | rejected
    cause_id: str | None


@dataclass
class StationRun:
    run_id: str
    lot: SimulatedLot
    vision: dict[str, VisionResult]
    attributions: dict[str, Attribution]
    action: ProposedAction
    report: str


_lock = threading.Lock()
_model: LightweightPatchCore | None = None
_runs: dict[str, StationRun] = {}


def ensure_model() -> LightweightPatchCore:
    global _model
    with _lock:
        if _model is None:
            rng = np.random.default_rng(0)
            goods = [render_part(None, rng) for _ in range(28)]
            hold = [render_part(None, rng) for _ in range(12)]
            model = LightweightPatchCore(image_size=128, coreset_ratio=0.2)
            model.fit(goods, rng=rng)
            model.calibrate_threshold(hold, quantile=0.90)
            _model = model
        return _model


def _majority_cause(run_attributions: dict[str, Attribution], flagged: list[str]) -> str | None:
    counts: dict[str, int] = {}
    for wid in flagged:
        cause = run_attributions[wid].cause_id
        if cause:
            counts[cause] = counts.get(cause, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _propose(cause_id: str | None, n_flagged: int) -> ProposedAction:
    if n_flagged == 0:
        return ProposedAction(
            action_id="act-" + uuid.uuid4().hex[:8],
            title="不需要动作",
            detail="这一批相机判定全部合格，不下发任何设备改动。",
            status="not_required",
            cause_id=None,
        )
    if cause_id and cause_id in ACTIONS:
        return ProposedAction(
            action_id="act-" + uuid.uuid4().hex[:8],
            title=ACTIONS[cause_id],
            detail="这是设备参数或检修动作，必须有人点「批准」才会记下。系统不会自己改机器。",
            status="pending",
            cause_id=cause_id,
        )
    return ProposedAction(
        action_id="act-" + uuid.uuid4().hex[:8],
        title="转给工艺工程师看，先不要改设备",
        detail="相机看到了异常，但设备曲线对不上唯一原因。禁止自动改参数。",
        status="pending",
        cause_id=None,
    )


def _write_report(lot: SimulatedLot, flagged: list[str], cause_id: str | None, attrs: dict[str, Attribution]) -> str:
    n = len(lot.wafers)
    if not flagged:
        return f"本批 {n} 片，相机没有发现异常。设备曲线也没有用来下结论。"
    lines = [f"本批 {n} 片，相机标出 {len(flagged)} 片异常。"]
    decided = [wid for wid in flagged if attrs[wid].cause_id]
    abstained = [wid for wid in flagged if attrs[wid].decision == "abstain"]
    if cause_id:
        sample = next(attrs[wid] for wid in decided if attrs[wid].cause_id == cause_id)
        lines.append(f"系统结论：{cause_label(cause_id)}（{len(decided)} 片对得上设备曲线）。")
        if sample.scores and sample.scores[0].lag_s is not None:
            top = sample.scores[0]
            edge = edge_by_cause(top.cause_id)
            signal = SIGNAL_LABELS.get(edge.signal, edge.signal) if edge else "设备"
            lines.append(
                f"证据：{signal}在零件出问题前 {top.lag_s:.0f} 秒开始异常。"
            )
        if abstained:
            lines.append(f"另外 {len(abstained)} 片相机也亮了，但对不上唯一设备原因，没有算进结论。")
        lines.append("改设备的动作要等人批准。")
    else:
        reason = "两个设备原因同时说得通" if any(
            attrs[wid].abstain_reason == "ambiguous_causes" for wid in abstained
        ) else "设备曲线上找不到足够的异常"
        lines.append(f"系统结论：拿不准，不指定原因。{reason}。")
        lines.append("不要改设备参数，转给工艺工程师。")
    return "\n".join(lines)


def inspect_lot(scenario: str, seed: int | None = None) -> StationRun:
    rng = np.random.default_rng(seed if seed is not None else np.random.randint(1, 10_000_000))
    lot = generate_lot(f"RUN{uuid.uuid4().hex[:6].upper()}", scenario, rng, render_images=True)
    model = ensure_model()
    vision: dict[str, VisionResult] = {}
    attributions: dict[str, Attribution] = {}
    flagged: list[str] = []
    for wafer in lot.wafers:
        assert wafer.image is not None
        result = model.predict(wafer.image)
        result.is_anomaly = bool(model.threshold is not None and result.score >= model.threshold)
        vision[wafer.wafer_id] = result
        if result.is_anomaly:
            flagged.append(wafer.wafer_id)
            # Attribute using the visual class if the generator labelled one,
            # otherwise the most common defect family for this scenario's FMEA.
            defect = wafer.defect_type or "scratch"
            attributions[wafer.wafer_id] = attribute_defect(
                defect, wafer.process_end, lot.t, lot.series
            )
    cause = _majority_cause(attributions, flagged)
    action = _propose(cause, len(flagged))
    report = _write_report(lot, flagged, cause, attributions)
    run = StationRun(
        run_id=uuid.uuid4().hex[:12],
        lot=lot,
        vision=vision,
        attributions=attributions,
        action=action,
        report=report,
    )
    _runs[run.run_id] = run
    return run


def get_run(run_id: str) -> StationRun | None:
    return _runs.get(run_id)


def decide_action(run_id: str, action_id: str, approved: bool) -> ProposedAction:
    run = _runs[run_id]
    if run.action.action_id != action_id:
        raise KeyError("action")
    if run.action.status != "pending":
        return run.action
    run.action.status = "approved" if approved else "rejected"
    return run.action


def serialize_run(run: StationRun) -> dict:
    model = ensure_model()
    wafers = []
    for wafer in run.lot.wafers:
        vis = run.vision[wafer.wafer_id]
        overlay = overlay_heatmap(wafer.image, vis.heatmap) if wafer.image is not None else None
        attr = run.attributions.get(wafer.wafer_id)
        wafers.append(
            {
                "wafer_id": wafer.wafer_id,
                "process_end": wafer.process_end,
                "flagged": vis.is_anomaly,
                "vision_score": vis.score,
                "threshold": model.threshold,
                "image": png_data_url(wafer.image) if wafer.image is not None else None,
                "overlay": png_data_url(overlay) if overlay is not None else None,
                "defect_label": defect_label(wafer.defect_type) if vis.is_anomaly else "合格",
                "true_defect": wafer.defect_type,
                "attribution": None
                if attr is None
                else {
                    "decision": attr.decision,
                    "decision_label": "拿不准，先不指定原因"
                    if attr.decision == "abstain"
                    else cause_label(attr.cause_id),
                    "abstain_reason": attr.abstain_reason,
                    "scores": [
                        {
                            "cause_id": s.cause_id,
                            "label": cause_label(s.cause_id),
                            "score": s.score,
                            "reason": s.reason,
                            "lag_s": s.lag_s,
                        }
                        for s in attr.scores
                    ],
                },
            }
        )
    injected = [
        {"cause_id": f.cause_id, "label": cause_label(f.cause_id), "t_start": f.t_start, "t_end": f.t_end}
        for f in run.lot.injected
    ]
    unexplained = run.lot.scenario == "unexplained_scratch"
    drill_answer = (
        "设备正常，划痕原因未知"
        if unexplained
        else ("两个原因同时在" if run.lot.scenario == "ambiguous_scratch" else (injected[0]["label"] if injected else "没有注入故障"))
    )
    return {
        "run_id": run.run_id,
        "lot_id": run.lot.lot_id,
        "scenario": run.lot.scenario,
        "disclaimer": "这是模拟产线的样本数据，不是真实工厂。",
        "report": run.report,
        "drill_answer": drill_answer,
        "telemetry": {
            "t": run.lot.t.tolist(),
            "series": {k: v.tolist() for k, v in run.lot.series.items()},
        },
        "injected": injected,
        "wafers": wafers,
        "action": {
            "action_id": run.action.action_id,
            "title": run.action.title,
            "detail": run.action.detail,
            "status": run.action.status,
            "cause_id": run.action.cause_id,
        },
    }

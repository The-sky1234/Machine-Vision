"""User-facing Chinese copy. Keep technical ids in code, not in the UI."""

SCENARIO_OPTIONS = (
    ("coolant_drift", "冷却水突然变热（后面会出划痕）"),
    ("temp_spike", "腔室温度冲高（后面会出污渍）"),
    ("pressure_oscillation", "气压不稳（后面会有颗粒）"),
    ("rpm_drop", "主轴转速掉下去（表面会不均匀）"),
    ("distractor_pressure_on_scratch", "冷却水有问题，气压还有点杂讯"),
    ("ambiguous_scratch", "两个原因同时出现（应该说拿不准）"),
    ("unexplained_scratch", "零件有划痕，但设备是正常的（应该说不知道）"),
    ("healthy", "这一批全是好的"),
)

SCENARIO_HELP = {k: v for k, v in SCENARIO_OPTIONS}

CAUSE_LABELS = {
    "coolant_drift": "冷却水温度偏高",
    "spindle_vibration": "主轴转速不稳",
    "temp_spike": "腔室温度冲高",
    "pressure_oscillation": "腔室气压振荡",
    "rpm_drop": "主轴转速偏低",
}

DEFECT_LABELS = {
    "scratch": "划痕",
    "stain": "污渍",
    "particle": "颗粒",
    "uneven": "表面不均匀",
}

SIGNAL_LABELS = {
    "coolant_temp": "冷却水温度",
    "chamber_temp": "腔室温度",
    "pressure": "腔室气压",
    "spindle_rpm": "主轴转速",
}

SIGNAL_UNITS = {
    "coolant_temp": "°C",
    "chamber_temp": "°C",
    "pressure": "",
    "spindle_rpm": "rpm",
}

ACTIONS = {
    "coolant_drift": "把冷却水设定温度下调 2°C",
    "spindle_vibration": "安排主轴动平衡检查",
    "temp_spike": "把腔室加热功率降一档",
    "pressure_oscillation": "检查气路稳压阀",
    "rpm_drop": "检查主轴驱动，恢复额定转速",
}


def cause_label(cause_id: str | None) -> str:
    if not cause_id:
        return "没有确定原因"
    return CAUSE_LABELS.get(cause_id, cause_id)


def defect_label(defect_type: str | None) -> str:
    if not defect_type:
        return "合格"
    return DEFECT_LABELS.get(defect_type, defect_type)

const $ = (id) => document.getElementById(id);

const SIGNAL_LABEL = {
  coolant_temp: "冷却水温度 °C",
  chamber_temp: "腔室温度 °C",
  pressure: "腔室气压",
  spindle_rpm: "主轴转速 rpm",
};

let current = null;
let selectedId = null;

async function loadYoloStatus() {
  const res = await fetch("/api/yolo/status");
  const data = await res.json();
  $("yolo-run").disabled = !data.available;
  $("yolo-status").textContent = data.available
    ? `${data.model} 已就绪；训练数据：${data.dataset}`
    : "模型权重还没有放到 models/neudet_yolo11n.pt";
}

async function runYolo() {
  const file = $("yolo-file").files[0];
  if (!file) {
    $("yolo-status").textContent = "请先选择一张钢材表面图片。";
    return;
  }
  const button = $("yolo-run");
  button.disabled = true;
  $("yolo-status").textContent = "YOLO 正在检查…";
  const form = new FormData();
  form.append("image", file);
  try {
    const res = await fetch("/api/yolo/detect", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json()).detail || "检测失败");
    const data = await res.json();
    $("yolo-image").src = data.annotated_image;
    const rows = data.detections.length
      ? data.detections
          .map(
            (d) =>
              `<div class="detection-row"><b>${escapeHtml(d.class_name_zh)}</b><span>${(
                d.confidence * 100
              ).toFixed(1)}%</span></div>`
          )
          .join("")
      : "<p>没有检出六类已知钢材缺陷。</p>";
    $("yolo-detections").innerHTML = `
      <h3>检测到 ${data.count} 个缺陷框</h3>
      ${rows}
      <p>CPU 推理 ${data.inference_ms.toFixed(1)} ms</p>
      <p>${escapeHtml(data.notice)}</p>
    `;
    $("yolo-result").classList.remove("hidden");
    $("yolo-status").textContent = "检测完成。";
  } catch (err) {
    $("yolo-status").textContent = String(err.message || err);
  } finally {
    button.disabled = false;
  }
}

async function loadScenarios() {
  const res = await fetch("/api/scenarios");
  const data = await res.json();
  const sel = $("scenario");
  sel.innerHTML = "";
  for (const s of data.scenarios) {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = s.label;
    sel.appendChild(opt);
  }
}

function setStatus(text) {
  $("status").textContent = text;
}

async function runLot() {
  const btn = $("run");
  btn.disabled = true;
  setStatus("正在拍照、对照设备曲线，请稍等几秒…");
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario: $("scenario").value }),
    });
    if (!res.ok) throw new Error("检查失败");
    current = await res.json();
    selectedId = pickDefault(current);
    $("board").classList.remove("hidden");
    render();
    const n = current.wafers.filter((w) => w.flagged).length;
    setStatus(`本批 ${current.wafers.length} 片，相机标出 ${n} 片异常。下面可以点某一片看细节。`);
  } catch (err) {
    setStatus(String(err.message || err));
  } finally {
    btn.disabled = false;
  }
}

function pickDefault(run) {
  const decided = run.wafers.find(
    (w) => w.flagged && w.attribution && w.attribution.decision !== "abstain"
  );
  if (decided) return decided.wafer_id;
  const bad = run.wafers.find((w) => w.flagged);
  return (bad || run.wafers[0]).wafer_id;
}

function waferById(id) {
  return current.wafers.find((w) => w.wafer_id === id);
}

function render() {
  renderStrip();
  renderWafer();
  renderCharts();
}

function renderStrip() {
  const strip = $("strip");
  strip.innerHTML = "";
  for (const w of current.wafers) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "die" + (w.flagged ? " flagged" : "") + (w.wafer_id === selectedId ? " selected" : "");
    b.title = w.wafer_id + (w.flagged ? " 异常" : " 合格");
    b.innerHTML = `<img alt="" src="${w.image}" /><span class="dot"></span>`;
    b.addEventListener("click", () => {
      selectedId = w.wafer_id;
      render();
    });
    strip.appendChild(b);
  }
}

function renderWafer() {
  const w = waferById(selectedId);
  $("photo").src = w.image;
  $("overlay").src = w.overlay;
  const box = $("verdict");
  const attr = w.attribution;
  let causeHtml = "";
  if (!w.flagged) {
    causeHtml = `<h3>相机</h3><p class="big ok">合格</p><p>分数 ${w.vision_score.toFixed(3)}，低于门槛 ${Number(w.threshold).toFixed(3)}。这片不去做根因。</p>`;
  } else if (!attr) {
    causeHtml = `<h3>相机</h3><p class="big bad">${w.defect_label}</p>`;
  } else {
    const klass = attr.decision === "abstain" ? "wait" : "bad";
    const scores = attr.scores
      .map((s) => {
        const pct = Math.max(0, Math.min(100, s.score * 100));
        return `<li><span>${s.label}</span><span class="bar"><span style="width:${pct}%"></span></span></li>`;
      })
      .join("");
    causeHtml = `
      <h3>相机</h3>
      <p class="big bad">${w.defect_label}</p>
      <p>异常分数 ${w.vision_score.toFixed(3)}（门槛 ${Number(w.threshold).toFixed(3)}）</p>
      <h3>设备原因</h3>
      <p class="big ${klass}">${attr.decision_label}</p>
      <ul class="scores">${scores}</ul>
    `;
  }

  const act = current.action;
  let actionHtml = `<h3>建议动作</h3><p><b>${act.title}</b></p><p>${act.detail}</p>`;
  if (act.status === "pending") {
    actionHtml += `<div class="actions">
      <button type="button" class="ok" id="approve">批准</button>
      <button type="button" class="no" id="reject">驳回</button>
    </div>`;
  } else if (act.status === "approved") {
    actionHtml += `<div class="stamp approved">已批准（演习记录，没有真的改机器）</div>`;
  } else if (act.status === "rejected") {
    actionHtml += `<div class="stamp rejected">已驳回，不下发</div>`;
  }

  box.innerHTML = `
    ${causeHtml}
    <h3>给工艺员的几句话</h3>
    <p>${escapeHtml(current.report)}</p>
    ${actionHtml}
    <p class="drill">演习答案（系统做判断时看不到）：${escapeHtml(current.drill_answer)}</p>
  `;
  const approve = $("approve");
  const reject = $("reject");
  if (approve) approve.onclick = () => decide(true);
  if (reject) reject.onclick = () => decide(false);
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function decide(approved) {
  const act = current.action;
  const res = await fetch(`/api/runs/${current.run_id}/actions/${act.action_id}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approved }),
  });
  if (!res.ok) return;
  const data = await res.json();
  current.action.status = data.status;
  renderWafer();
}

function renderCharts() {
  const host = $("charts");
  host.innerHTML = "";
  const t = current.telemetry.t;
  const w = waferById(selectedId);
  for (const [key, values] of Object.entries(current.telemetry.series)) {
    const wrap = document.createElement("div");
    wrap.className = "chart";
    wrap.innerHTML = `<div class="name">${SIGNAL_LABEL[key] || key}</div>`;
    const canvas = document.createElement("canvas");
    wrap.appendChild(canvas);
    host.appendChild(wrap);
    drawChart(canvas, t, values, current.injected, w.process_end);
  }
}

function drawChart(canvas, t, y, injected, tMark) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 800;
  const height = 92;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  const minY = Math.min(...y);
  const maxY = Math.max(...y);
  const pad = (maxY - minY) * 0.15 || 1;
  const y0 = minY - pad;
  const y1 = maxY + pad;
  const xOf = (v) => ((v - t[0]) / (t[t.length - 1] - t[0] || 1)) * (width - 8) + 4;
  const yOf = (v) => height - 8 - ((v - y0) / (y1 - y0)) * (height - 16);

  ctx.fillStyle = "rgba(240,180,41,0.18)";
  for (const ev of injected) {
    const x1 = xOf(ev.t_start);
    const x2 = xOf(ev.t_end);
    ctx.fillRect(x1, 4, Math.max(2, x2 - x1), height - 12);
  }

  ctx.beginPath();
  ctx.strokeStyle = "#8eb4ff";
  ctx.lineWidth = 1.4;
  y.forEach((v, i) => {
    const x = xOf(t[i]);
    const yy = yOf(v);
    if (i === 0) ctx.moveTo(x, yy);
    else ctx.lineTo(x, yy);
  });
  ctx.stroke();

  if (tMark != null) {
    ctx.strokeStyle = "#e85d4c";
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(xOf(tMark), 4);
    ctx.lineTo(xOf(tMark), height - 8);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

$("run").addEventListener("click", runLot);
$("yolo-run").addEventListener("click", runYolo);
loadYoloStatus().catch((err) => ($("yolo-status").textContent = String(err)));
loadScenarios().catch((err) => setStatus(String(err)));
window.addEventListener("resize", () => {
  if (current) renderCharts();
});

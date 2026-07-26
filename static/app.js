/* Sparkle Sentinel dashboard */
const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n).toLocaleString("en-US");

const ANALYSIS_TOOLS = ["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"];
const STATUS_STEPS = ["Parsing intent…", "Building execution plan…", "Running tools…", "Scoring & explaining…"];

/* ---------- stats + volume chart ---------- */
async function loadStats() {
  try {
    const r = await fetch("/api/stats");
    const s = await r.json();
    $("kpi-txns").textContent = fmt(s.transactions);
    $("kpi-accounts").textContent = fmt(s.accounts);
    $("kpi-xb").textContent = s.cross_border_pct + "%";
    $("kpi-dates").textContent = s.date_min + " → " + s.date_max;
    $("api-pill").textContent = "data: live";
    $("api-pill").className = "pill pill-ok";
    drawVolume(s.daily_volume);
  } catch (e) {
    $("api-pill").textContent = "api unreachable";
  }
}

function drawVolume(days) {
  if (!days || !days.length) return;
  const W = 860, H = 190, padL = 46, padR = 12, padT = 14, padB = 26;
  const iw = W - padL - padR, ih = H - padT - padB;
  const max = Math.max(...days.map((d) => d.count));
  const x = (i) => padL + (i / (days.length - 1)) * iw;
  const y = (v) => padT + ih - (v / max) * ih;

  let grid = "", labels = "";
  for (let g = 0; g <= 3; g++) {
    const v = Math.round((max / 3) * g), gy = y(v);
    grid += `<line x1="${padL}" y1="${gy}" x2="${W - padR}" y2="${gy}" stroke="rgba(255,255,255,0.06)"/>`;
    labels += `<text x="${padL - 8}" y="${gy + 4}" text-anchor="end" font-size="10" fill="#8a897f">${fmt(v)}</text>`;
  }
  const stepN = Math.max(1, Math.floor(days.length / 6));
  for (let i = 0; i < days.length; i += stepN) {
    labels += `<text x="${x(i)}" y="${H - 8}" text-anchor="middle" font-size="10" fill="#8a897f">${days[i].date}</text>`;
  }
  const pts = days.map((d, i) => `${x(i).toFixed(1)},${y(d.count).toFixed(1)}`);
  const defs = `<defs><linearGradient id="volgrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="rgba(57,135,229,0.32)"/>
      <stop offset="1" stop-color="rgba(57,135,229,0)"/>
    </linearGradient></defs>`;
  const line = defs + `<polyline points="${pts.join(" ")}" fill="none" stroke="#3987e5" stroke-width="2" stroke-linejoin="round"/>`;
  const area = `<polygon points="${padL},${padT + ih} ${pts.join(" ")} ${W - padR},${padT + ih}" fill="url(#volgrad)"/>`;
  const hover = `<line id="vol-cross" y1="${padT}" y2="${padT + ih}" stroke="rgba(255,255,255,0.25)" stroke-dasharray="3 3" visibility="hidden"/>
                 <circle id="vol-dot" r="4" fill="#3987e5" stroke="#121211" stroke-width="2" visibility="hidden"/>`;
  const svg = $("vol-chart");
  svg.innerHTML = grid + area + line + labels + hover;
  $("vol-note").textContent = `${days.length} days · peak ${fmt(max)} txns/day`;

  const tip = $("chart-tip");
  svg.onmousemove = (ev) => {
    const rect = svg.getBoundingClientRect();
    const sx = ((ev.clientX - rect.left) / rect.width) * W;
    const i = Math.max(0, Math.min(days.length - 1, Math.round(((sx - padL) / iw) * (days.length - 1))));
    const cx = x(i), cy = y(days[i].count);
    const cross = $("vol-cross"), dot = $("vol-dot");
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", cx); dot.setAttribute("cy", cy); dot.setAttribute("visibility", "visible");
    tip.hidden = false;
    tip.textContent = `${days[i].date} · ${fmt(days[i].count)} txns`;
    tip.style.left = (cx / W) * rect.width + "px";
    tip.style.top = (cy / H) * rect.height + "px";
  };
  svg.onmouseleave = () => {
    tip.hidden = true;
    $("vol-cross").setAttribute("visibility", "hidden");
    $("vol-dot").setAttribute("visibility", "hidden");
  };

  const tbl = $("vol-table");
  tbl.innerHTML = "<tr><th>Date</th><th>Transactions</th></tr>" +
    days.map((d) => `<tr><td>${d.date}</td><td>${fmt(d.count)}</td></tr>`).join("");
}

/* ---------- query flow ---------- */
let statusTimer = null;
let running = false;

function setRunning(on) {
  running = on;
  $("run-btn").disabled = on;
  $("agent-status").hidden = !on;
  clearInterval(statusTimer);
  if (on) {
    let step = 0;
    $("agent-status-text").textContent = STATUS_STEPS[0];
    statusTimer = setInterval(() => {
      step = Math.min(step + 1, STATUS_STEPS.length - 1);
      $("agent-status-text").textContent = STATUS_STEPS[step];
    }, 2600);
  }
}

async function runQuery() {
  const q = $("query-input").value.trim();
  if (!q || running) return;
  setRunning(true);
  $("results").hidden = true;
  const abort = new AbortController();
  const kill = setTimeout(() => abort.abort(), 90_000);
  try {
    const r = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, sample_size: 20000 }),
      signal: abort.signal,
    });
    if (!r.ok) throw new Error("API " + r.status);
    render(await r.json());
  } catch (e) {
    $("results").hidden = false;
    ["risk-card", "agg-card", "eda-card"].forEach((id) => ($(id).hidden = true));
    $("error-card").hidden = false;
    $("error-text").textContent =
      e.name === "AbortError"
        ? "The agent took too long — is the API server up? Try again."
        : "Agent request failed: " + e.message + " — try again.";
  } finally {
    clearTimeout(kill);
    setRunning(false);
  }
}

function render(res) {
  $("results").hidden = false;
  $("error-card").hidden = !res.error;
  if (res.error) $("error-text").textContent = res.error;

  $("exec-intent").textContent = res.detected_intent || "—";
  $("exec-pattern").textContent = res.detected_pattern || "generic";
  $("exec-rows").textContent = res.working_rows ? fmt(res.working_rows) + " txns" : "—";
  $("exec-parser").textContent = res.parser_used || "—";
  $("exec-meta").textContent = '"' + res.query + '"';
  $("parser-pill").textContent = "parser: " + (res.parser_used || "?");
  $("parser-pill").className = "pill " + (res.parser_used === "gemini" ? "pill-accent" : "pill-muted");

  const invoked = res.tools_invoked || [];
  $("plan-chips").innerHTML = invoked
    .map((t, i) => `<span class="plan-step" style="animation-delay:${i * 0.12}s"><span class="n">${i + 1}</span>${t}</span>`)
    .join('<span class="plan-arrow" style="animation-delay:0.1s">→</span>');
  const invokedBase = invoked.map((t) => t.split("[")[0]);
  const skipped = ANALYSIS_TOOLS.filter((t) => !invokedBase.includes(t));
  $("plan-skipped").innerHTML = skipped.length
    ? "agent skipped: " + skipped.map((t) => `<span class="skip">${t}</span>`).join(" ")
    : "full analysis chain invoked";

  // flags
  const flags = res.flags || [];
  $("risk-card").hidden = !flags.length;
  if (flags.length) {
    const rb = res.risk_breakdown || {};
    const dotColor = { High: "#d03b3b", Medium: "#fab219", Low: "#0ca30c" };
    $("risk-pills").innerHTML = ["High", "Medium", "Low"]
      .filter((lv) => rb[lv])
      .map((lv) => `<span class="risk-pill"><span class="dot" style="background:${dotColor[lv]}"></span>${lv}: ${rb[lv]}</span>`)
      .join("");
    $("flags").innerHTML = flags
      .map(
        (f) => `<div class="flag flag-${f.risk_level}">
          <div class="flag-head">
            <span class="flag-route">${f.Sender_account}<span class="arr">→</span>${f.Receiver_account}</span>
            <span class="flag-amount">${fmt(f.Amount)}</span>
            <span class="badge badge-${f.risk_level}">${f.risk_level === "High" ? "▲" : f.risk_level === "Medium" ? "◆" : "●"} ${f.risk_level}</span>
          </div>
          <div class="flag-expl">${f.explanation}</div>
          <div class="flag-action">recommended action: <b>${f.recommended_action}</b></div>
        </div>`
      )
      .join("");
  }

  // aggregation
  const agg = res.aggregation_result || [];
  $("agg-card").hidden = !agg.length;
  if (agg.length) {
    $("agg-note").textContent = agg.length + " accounts (showing first " + Math.min(agg.length, 100) + ")";
    const cols = Object.keys(agg[0]);
    $("agg-table").innerHTML =
      "<tr>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr>" +
      agg.map((row) => "<tr>" + cols.map((c) => `<td>${typeof row[c] === "number" ? fmt(Math.round(row[c] * 100) / 100) : row[c]}</td>`).join("") + "</tr>").join("");
  }

  // eda
  const eda = res.eda_summary;
  $("eda-card").hidden = !eda;
  if (eda) {
    const st = eda.amount_stats || {};
    const cell = (label, val) => `<div><div class="kpi-label">${label}</div><div class="exec-val">${val}</div></div>`;
    $("eda-stats").innerHTML =
      cell("Rows profiled", fmt(eda.row_count)) +
      cell("Mean amount", st.mean ? fmt(Math.round(st.mean)) : "—") +
      cell("Median amount", st.median ? fmt(Math.round(st.median)) : "—") +
      cell("Max amount", st.max ? fmt(Math.round(st.max)) : "—");
    const pt = eda.payment_type_dist || null;
    $("pt-label").hidden = !pt;
    $("pt-bars").innerHTML = pt
      ? Object.entries(pt)
          .sort((a, b) => b[1] - a[1])
          .map(
            ([name, share]) => `<div class="pt-bar-row">
              <span class="pt-name">${name}</span>
              <div class="pt-track"><div class="pt-fill" style="width:${(share * 100).toFixed(1)}%"></div></div>
              <span class="pt-val">${(share * 100).toFixed(1)}%</span>
            </div>`
          )
          .join("")
      : "";
  }

  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------- wiring ---------- */
$("run-btn").addEventListener("click", runQuery);
$("query-input").addEventListener("keydown", (e) => e.key === "Enter" && runQuery());
document.querySelectorAll("#suggest-chips .chip").forEach((c) =>
  c.addEventListener("click", () => {
    $("query-input").value = c.textContent;
    runQuery();
  })
);
loadStats();

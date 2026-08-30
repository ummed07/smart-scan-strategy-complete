/* Smart Scan Strategy — Django dashboard client */

const METRIC_DEFS = {
  "Detection Rate (Pd)": "HIT / (HIT + MISS) — fraction of true transmissions detected.",
  "False Alarm Rate (Pfa)": "FALSE_ALARM / (FALSE_ALARM + CORRECT_NEGATIVE) — false positives on silent slots.",
  "Interception Rate": "Percentage of transmission events detected at least once while active.",
  "Avg Detection Delay": "Average slots between event start and first HIT.",
  "Miss Rate": "1 − Interception Rate — events never detected.",
  "Average Reward": "+1 per HIT, −1 per FALSE_ALARM, averaged over all scans.",
};

const PLOTLY_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(15,23,42,0.5)",
  font: { color: "#e2e8f0", family: "DM Sans, sans-serif" },
  margin: { l: 48, r: 24, t: 40, b: 48 },
  xaxis: { gridcolor: "rgba(148,163,184,0.1)", zerolinecolor: "rgba(148,163,184,0.2)" },
  yaxis: { gridcolor: "rgba(148,163,184,0.1)", zerolinecolor: "rgba(148,163,184,0.2)" },
};

let currentStatus = null;

function getCookie(name) {
  const match = document.cookie.match(new RegExp("(^| )" + name + "=([^;]+)"));
  return match ? decodeURIComponent(match[2]) : null;
}

function getConfigPayload() {
  const form = document.getElementById("config-form");
  const data = new FormData(form);
  const payload = {};
  for (const [key, value] of data.entries()) {
    payload[key] = value;
  }
  return payload;
}

function showLoading(show) {
  document.getElementById("loading").classList.toggle("hidden", !show);
}

function toast(message, type = "info") {
  const container = document.getElementById("toast-container");
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

async function apiPost(url, body = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": window.CSRF_TOKEN || getCookie("csrftoken"),
    },
    body: JSON.stringify(body),
  });
  return parseJsonResponse(res);
}

async function apiGet(url) {
  const res = await fetch(url);
  return parseJsonResponse(res);
}

async function parseJsonResponse(res) {
  const text = await res.text();
  if (!text) {
    throw new Error(`Empty response (${res.status})`);
  }
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(`Server returned invalid JSON (${res.status})`);
  }
  if (!res.ok) {
    throw new Error(data.message || data.error || `Request failed (${res.status})`);
  }
  return data;
}

function plotChart(elId, chartData) {
  const el = document.getElementById(elId);
  if (!el) return;
  if (!chartData) {
    el.innerHTML = '<div class="empty-state">No data yet — run a simulation first.</div>';
    return;
  }
  const fig = typeof chartData === "string" ? JSON.parse(chartData) : chartData;
  fig.layout = { ...PLOTLY_LAYOUT, ...(fig.layout || {}) };
  Plotly.newPlot(el, fig.data, fig.layout, { responsive: true, displayModeBar: false });
}

function renderTable(tableId, rows) {
  const table = document.getElementById(tableId);
  if (!table || !rows || !rows.length) {
    if (table) table.innerHTML = "<tbody><tr><td class='empty-state'>No data</td></tr></tbody>";
    return;
  }
  const cols = Object.keys(rows[0]);
  const thead = `<thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${rows
    .map((row) => `<tr>${cols.map((c) => `<td>${formatCell(row[c])}</td>`).join("")}</tr>`)
    .join("")}</tbody>`;
  table.innerHTML = thead + tbody;
}

function formatCell(val) {
  if (val === null || val === undefined) return "—";
  if (typeof val === "number") {
    if (Number.isInteger(val)) return val;
    return val.toFixed(4);
  }
  return val;
}

function updateRangeLabels() {
  document.querySelectorAll("#config-form input[type=range]").forEach((input) => {
    const span = document.getElementById(`val-${input.name}`);
    if (!span) return;
    const v = parseFloat(input.value);
    span.textContent = v < 1 && v > 0 ? v.toFixed(2) : String(v);
  });
}

function updateStatusPills(status) {
  const env = document.getElementById("pill-env");
  const model = document.getElementById("pill-model");
  const compare = document.getElementById("pill-compare");

  env.textContent = status.has_environment ? "Environment: Ready" : "Environment: —";
  env.className = `pill ${status.has_environment ? "pill-ok" : "pill-muted"}`;

  model.textContent = status.has_model ? "Model: Trained" : "Model: —";
  model.className = `pill ${status.has_model ? "pill-ok" : "pill-muted"}`;

  compare.textContent = status.has_comparison ? "Comparison: Done" : "Comparison: —";
  compare.className = `pill ${status.has_comparison ? "pill-ok" : "pill-muted"}`;
}

function updateHeroMetrics(status) {
  const s = status.stats;
  document.getElementById("m-bands-time").textContent = s
    ? `${s.num_bands} × ${s.num_time_slots}`
    : "—";
  document.getElementById("m-occupancy").textContent = s
    ? `${(s.occupancy_rate * 100).toFixed(1)}%`
    : "—";
  document.getElementById("m-active-bands").textContent = s ? s.active_bands : "—";
  document.getElementById("m-events").textContent = s ? s.transmission_events : "—";
}

function updateMlMetrics(status) {
  const t = status.training;
  document.getElementById("ml-acc").textContent = t ? `${(t.accuracy * 100).toFixed(1)}%` : "—";
  document.getElementById("ml-prec").textContent = t ? `${(t.precision * 100).toFixed(1)}%` : "—";
  document.getElementById("ml-rec").textContent = t ? `${(t.recall * 100).toFixed(1)}%` : "—";
  document.getElementById("ml-f1").textContent = t ? `${(t.f1 * 100).toFixed(1)}%` : "—";
}

function renderPerfMetrics(status) {
  const select = document.getElementById("perf-scheduler");
  const name = select.value;
  const m = status.metrics[name];
  const container = document.getElementById("perf-metrics");

  if (!m) {
    container.innerHTML = '<div class="empty-state">Run a simulation to see performance metrics.</div>';
    return;
  }

  const items = [
    ["Detection Rate (Pd)", `${(m.detection_rate * 100).toFixed(2)}%`],
    ["False Alarm Rate (Pfa)", `${(m.false_alarm_rate * 100).toFixed(2)}%`],
    ["Interception Rate", `${(m.interception_rate * 100).toFixed(2)}%`],
    ["Avg Detection Delay", `${m.avg_detection_delay.toFixed(2)} slots`],
    ["Miss Rate", `${(m.miss_rate * 100).toFixed(2)}%`],
    ["Average Reward", m.average_reward.toFixed(3)],
  ];

  container.innerHTML = items
    .map(
      ([label, val]) =>
        `<div class="metric-card"><span class="metric-label">${label}</span><span class="metric-value">${val}</span></div>`
    )
    .join("");
}

function renderMetricDefs() {
  const dl = document.getElementById("metric-defs");
  dl.innerHTML = Object.entries(METRIC_DEFS)
    .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`)
    .join("");
}

async function refreshStatus() {
  currentStatus = await apiGet("/api/status/");
  updateStatusPills(currentStatus);
  updateHeroMetrics(currentStatus);
  updateMlMetrics(currentStatus);
  renderPerfMetrics(currentStatus);
  return currentStatus;
}

async function refreshCharts(tab) {
  if (!currentStatus?.has_environment) return;

  if (tab === "environment" || tab === "overview") {
    const data = await apiGet("/api/charts/environment/");
    plotChart("chart-environment", data.chart);
    const emitters = await apiGet("/api/emitters/");
    renderTable("emitter-table", emitters.rows);
  }

  if (tab === "receiver") {
    const data = await apiGet("/api/charts/receiver/");
    plotChart("chart-receiver", data.chart);
    plotChart("chart-outcomes", data.outcomes);
    renderTable("receiver-table", data.history);
  }

  if (tab === "ml") {
    const data = await apiGet("/api/charts/ml/");
    if (data.charts) {
      plotChart("chart-cm", data.charts.confusion_matrix);
      plotChart("chart-importance", data.charts.feature_importance);
      plotChart("chart-probabilities", data.charts.band_probabilities);
    } else {
      ["chart-cm", "chart-importance", "chart-probabilities"].forEach((id) => {
        document.getElementById(id).innerHTML =
          '<div class="empty-state">Train the ML model first.</div>';
      });
    }
  }

  if (tab === "smart") {
    const data = await apiGet("/api/charts/smart/");
    if (data.charts) {
      plotChart("chart-smart-decisions", data.charts.decisions);
      plotChart("chart-smart-priority", data.charts.priorities);
      document.getElementById("sm-decisions").textContent = data.charts.decisions_logged;
      document.getElementById("sm-explore").textContent = data.charts.exploration_moves;
      document.getElementById("sm-ratio").textContent = `${(data.charts.exploration_ratio * 100).toFixed(1)}%`;
      renderTable("smart-table", data.history);
    } else {
      ["chart-smart-decisions", "chart-smart-priority"].forEach((id) => {
        document.getElementById(id).innerHTML =
          '<div class="empty-state">Run the Smart ML simulation first.</div>';
      });
    }
  }

  if (tab === "comparison") {
    const data = await apiGet("/api/charts/comparison/");
    if (data.data) {
      renderTable("comparison-table", data.data.table);
      const container = document.getElementById("comparison-charts");
      container.innerHTML = "";
      const metrics = Object.keys(data.data.charts);
      metrics.forEach((metric, i) => {
        const div = document.createElement("div");
        div.className = "chart-card";
        div.id = `cmp-chart-${i}`;
        container.appendChild(div);
        plotChart(`cmp-chart-${i}`, data.data.charts[metric]);
      });
      plotChart("chart-comparison-scatter", data.data.scatter);
    } else {
      document.getElementById("comparison-table").innerHTML =
        "<tbody><tr><td class='empty-state'>Run comparison first.</td></tr></tbody>";
    }
  }
}

async function runAction(action) {
  showLoading(true);
  try {
    let result;
    const payload = getConfigPayload();

    switch (action) {
      case "generate":
        result = await apiPost("/api/generate/", payload);
        break;
      case "sequential":
        result = await apiPost("/api/run/sequential/");
        break;
      case "random":
        result = await apiPost("/api/run/random/");
        break;
      case "train":
        result = await apiPost("/api/train/");
        break;
      case "smart":
        result = await apiPost("/api/run/smart/");
        break;
      case "compare":
        result = await apiPost("/api/compare/");
        break;
      case "demo-sweep":
        result = await apiPost("/api/demo-sweep/");
        break;
      case "reset":
        result = await apiPost("/api/reset/");
        break;
      default:
        return;
    }

    toast(result.message, result.ok ? "success" : "error");
    if (result.status) {
      currentStatus = result.status;
      updateStatusPills(currentStatus);
      updateHeroMetrics(currentStatus);
      updateMlMetrics(currentStatus);
      renderPerfMetrics(currentStatus);
    } else {
      await refreshStatus();
    }

    const activeTab = document.querySelector(".tab.active")?.dataset.tab || "overview";
    await refreshCharts(activeTab);
  } catch (err) {
    toast(err.message || "Request failed", "error");
  } finally {
    showLoading(false);
  }
}

function initTabs() {
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", async () => {
      tabs.forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      const panel = document.getElementById(`panel-${tab.dataset.tab}`);
      panel.classList.add("active");
      await refreshCharts(tab.dataset.tab);
    });
  });
}

function initActions() {
  document.querySelectorAll("[data-action]").forEach((btn) => {
    btn.addEventListener("click", () => runAction(btn.dataset.action));
  });

  document.getElementById("perf-scheduler").addEventListener("change", () => {
    renderPerfMetrics(currentStatus || { metrics: {} });
  });
}

function initForm() {
  document.querySelectorAll("#config-form input").forEach((input) => {
    input.addEventListener("input", updateRangeLabels);
  });
  updateRangeLabels();
}

async function init() {
  renderMetricDefs();
  initTabs();
  initActions();
  initForm();
  await refreshStatus();
  await refreshCharts("overview");
}

document.addEventListener("DOMContentLoaded", init);

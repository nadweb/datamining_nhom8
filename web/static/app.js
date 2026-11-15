// Member 8: asynchronous prediction and analytics dashboard.
// Flight form validation, presets and route summary are owned by the TV7 UI.
"use strict";

document.addEventListener("DOMContentLoaded", () => {
  initNavigationTabs();
  initSliders();
  initPredictionForm();
  // Allow the TV7 DOMContentLoaded handler to register validation first.
  setTimeout(() => document.getElementById("predict-form")?.requestSubmit(), 0);
});

const byId = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "--").replace(/[&<>"']/g, (ch) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[ch]);
const numberOrNull = (value) => value === null || value === undefined || value === "" || typeof value === "boolean"
  ? null : (Number.isFinite(Number(value)) ? Number(value) : null);
const formatNumber = (value, digits = 2) => {
  const n = numberOrNull(value);
  return n === null ? "--" : n.toFixed(digits);
};
const formatPercent = (value, digits = 2) => {
  const n = numberOrNull(value);
  return n === null ? "--" : `${n.toFixed(digits)}%`;
};

async function requestJson(url, options = {}, controller = new AbortController()) {
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    let body;
    try {
      body = await response.json();
    } catch (_) {
      throw new Error(`Máy chủ trả về dữ liệu không hợp lệ (HTTP ${response.status}).`);
    }
    if (!response.ok) {
      throw new Error(typeof body?.error === "string" ? body.error : `HTTP ${response.status}`);
    }
    return body;
  } finally {
    clearTimeout(timeout);
  }
}

function initNavigationTabs() {
  const buttons = document.querySelectorAll(".nav-btn");
  const tabs = document.querySelectorAll(".tab-content");
  buttons.forEach((button) => button.addEventListener("click", () => {
    buttons.forEach((item) => {
      const selected = item === button;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-selected", String(selected));
    });
    tabs.forEach((tab) => tab.classList.toggle("active", tab.id === button.dataset.tab));
    if (button.dataset.tab === "models-tab") loadModelMetrics();
    if (button.dataset.tab === "clusters-tab") loadClusterProfiles();
    if (button.dataset.tab === "rules-tab") loadAssociationRules();
  }));
}

function initSliders() {
  ["precip", "wind", "vis", "load"].forEach((id) => {
    const slider = byId(id);
    const label = byId(`${id}-val`);
    if (slider && label) slider.addEventListener("input", () => { label.textContent = slider.value; });
  });
}

function readFlightInputs() {
  return {
    airline_code: byId("airline").value,
    airport: byId("airport").value,
    direction: byId("direction").value,
    route_type: byId("route-type").value,
    scheduled_time: byId("sched-time").value,
    flight_date: byId("flight-date").value,
    precipitation: Number(byId("precip").value),
    wind_gusts_10m: Number(byId("wind").value),
    visibility: Number(byId("vis").value),
    weather_code: Number(byId("weather-code").value),
    airport_load: Number(byId("load").value),
    origin: byId("origin").value,
    destination: byId("destination").value,
  };
}

function clearPrediction(message, state = "CHƯA CÓ DỰ BÁO") {
  const header = byId("predict-result")?.querySelector(".result-header");
  if (header) header.className = "result-header";
  byId("res-risk-level").textContent = state;
  byId("res-prediction-text").textContent = message;
  byId("res-risk-score").textContent = "--";
  ["prob-ontime", "prob-delayed", "prob-cancelled"].forEach((id) => {
    byId(id).style.width = "0%";
    byId(`${id}-text`).textContent = "--";
  });
  byId("res-cluster-name").textContent = "--";
  byId("res-cluster-rate").textContent = "--";
  byId("res-rules-list").replaceChildren();
}

function validatePrediction(data) {
  const p = data?.probabilities;
  const values = p && [p.on_time, p.delayed, p.cancelled];
  const isProbability = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
  if (!data || !["on_time", "delayed", "cancelled"].includes(data.prediction) ||
      !["LOW", "MEDIUM", "HIGH"].includes(data.risk_level) || !isProbability(data.risk_score) ||
      !values || !values.every(isProbability) || Math.abs(values.reduce((a, b) => a + b, 0) - 1) > 0.001 ||
      Math.abs(data.risk_score - p.delayed - p.cancelled) > 0.001 ||
      typeof data.estimated_delay_min !== "number" || !Number.isFinite(data.estimated_delay_min) ||
      data.estimated_delay_min < 0 || !data.cluster_profile || typeof data.cluster_profile !== "object" ||
      !Array.isArray(data.matched_rules)) {
    throw new Error("Kết quả dự báo không đúng cấu trúc dữ liệu của dự án.");
  }
}

function initPredictionForm() {
  const form = byId("predict-form");
  if (!form) return;
  const button = form.querySelector('button[type="submit"]');
  let activeRequest = null;
  let sequence = 0;

  const unlock = () => {
    if (button) { button.disabled = false; button.textContent = "Dự Báo Ngay"; }
  };
  const invalidate = () => {
    sequence += 1;
    activeRequest?.abort();
    activeRequest = null;
    unlock();
    clearPrediction("Thông số đã thay đổi. Hãy bấm Dự Báo Ngay để xem kết quả mới.");
  };
  form.addEventListener("input", invalidate);
  form.addEventListener("change", invalidate);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!form.checkValidity()) { form.reportValidity(); return; }
    activeRequest?.abort();
    const requestId = ++sequence;
    const controller = new AbortController();
    activeRequest = controller;
    clearPrediction("Đang tải kết quả từ máy chủ...", "ĐANG DỰ BÁO");
    if (button) { button.disabled = true; button.textContent = "Đang dự báo..."; }
    try {
      const result = await requestJson("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(readFlightInputs()),
      }, controller);
      validatePrediction(result);
      if (requestId === sequence) renderPredictionResult(result);
    } catch (error) {
      if (requestId === sequence) {
        const detail = error.name === "AbortError" ? "Máy chủ phản hồi quá thời gian." : error.message;
        clearPrediction(`Không thể dự báo: ${detail}`, "LỖI DỰ BÁO");
      }
    } finally {
      if (requestId === sequence) { activeRequest = null; unlock(); }
    }
  });
}

function setBar(id, value) {
  const pct = Math.round(value * 100);
  byId(id).style.width = `${pct}%`;
  byId(`${id}-text`).textContent = `${pct}%`;
}

function renderPredictionResult(data) {
  byId("predict-result").querySelector(".result-header").className = `result-header risk-${data.risk_level.toLowerCase()}`;
  byId("res-risk-level").textContent = `RỦI RO: ${data.risk_level}`;
  const messages = {
    on_time: "Dự báo: ĐÚNG GIỜ",
    delayed: `Dự báo: TRỄ KHOẢNG ${data.estimated_delay_min} PHÚT`,
    cancelled: "Dự báo: CÓ NGUY CƠ HỦY CHUYẾN",
  };
  byId("res-prediction-text").textContent = messages[data.prediction];
  byId("res-risk-score").textContent = `${Math.round(data.risk_score * 100)}%`;
  setBar("prob-ontime", data.probabilities.on_time);
  setBar("prob-delayed", data.probabilities.delayed);
  setBar("prob-cancelled", data.probabilities.cancelled);

  const profile = data.cluster_profile;
  byId("res-cluster-name").textContent = profile.cluster_name ?? profile.name ?? "Không có tên cụm";
  byId("res-cluster-rate").textContent = profile.cluster_risk_avg ??
    (numberOrNull(profile.delay_rate_pct) === null ? "--" : `${profile.delay_rate_pct}%`);
  const list = byId("res-rules-list");
  list.replaceChildren();
  if (!data.matched_rules.length) {
    const empty = document.createElement("p");
    empty.textContent = "Không có luật phù hợp với các thông số hiện tại.";
    list.appendChild(empty);
  }
  data.matched_rules.forEach((rule) => {
    const row = document.createElement("div");
    row.className = "rule-item";
    const confidence = rule.confidence_pct ?? rule.confidence;
    row.innerHTML = `<div class="rule-ante">${escapeHtml(rule.antecedent)} ➔ <strong>${escapeHtml(rule.consequent)}</strong></div>
      <div class="rule-meta">Độ tin cậy: <strong>${formatPercent(confidence)}</strong> | Lift: <strong>${formatNumber(rule.lift)}x</strong></div>`;
    list.appendChild(row);
  });
}

function panelError(container, error, retry) {
  delete container.dataset.loaded;
  container.replaceChildren();
  const card = document.createElement("div");
  card.className = "card";
  card.setAttribute("role", "alert");
  const text = document.createElement("p");
  text.textContent = `Không tải được dữ liệu: ${error.name === "AbortError" ? "Máy chủ phản hồi quá thời gian." : error.message}`;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "btn-primary";
  button.textContent = "Thử lại";
  button.addEventListener("click", retry);
  card.append(text, button);
  container.appendChild(card);
}

async function loadPanel(id, url, render, retry) {
  const container = byId(id);
  if (!container || container.dataset.loaded || container.dataset.loading) return;
  container.dataset.loading = "true";
  container.textContent = "Đang tải dữ liệu từ máy chủ...";
  try {
    const data = await requestJson(url);
    render(data, container);
    container.dataset.loaded = "true";
  } catch (error) {
    panelError(container, error, retry);
  } finally {
    delete container.dataset.loading;
  }
}

function loadModelMetrics() { return loadPanel("metrics-content", "/api/stats", renderModelMetrics, loadModelMetrics); }
function loadClusterProfiles() { return loadPanel("clusters-content", "/api/clusters", renderClusterProfiles, loadClusterProfiles); }
function loadAssociationRules() { return loadPanel("rules-content", "/api/rules", renderAssociationRules, loadAssociationRules); }

function renderModelMetrics(data, container) {
  const sup = data?.supervised;
  if (!sup?.classification || !sup.regression) throw new Error("Thiếu thống kê mô hình trong API.");
  const clf = sup.classification;
  const selected = sup.model_selection?.selected ?? clf.best_classifier_name;
  const rows = Object.entries(clf).filter(([, metrics]) => metrics && typeof metrics === "object" && !Array.isArray(metrics));
  const split = sup.split || {};
  const imps = Array.isArray(sup.feature_importances) ? sup.feature_importances.slice(0, 10) : [];
  container.innerHTML = `
    <div class="card"><div class="card-header-row"><h3 class="card-title">So sánh mô hình phân loại</h3>
      <span class="card-step-badge">Test: ${escapeHtml(split.test_rows ?? "--")} mẫu</span></div>
      <div class="dashboard-table"><table class="data-table"><thead><tr><th>Mô hình</th><th>Accuracy</th><th>Macro F1</th><th>Macro Recall</th><th>ROC-AUC</th><th>Lựa chọn</th></tr></thead>
      <tbody>${rows.map(([name, m]) => `<tr><td><strong>${escapeHtml(name)}</strong></td><td>${formatPercent(numberOrNull(m.accuracy) === null ? null : m.accuracy * 100, 1)}</td>
      <td>${formatNumber(m.f1_macro, 4)}</td><td>${formatNumber(m.recall_macro, 4)}</td><td>${formatNumber(m.roc_auc_ovr, 4)}</td><td>${name === selected ? "Được chọn" : "--"}</td></tr>`).join("")}</tbody></table></div>
      <p class="dashboard-note">Chọn mô hình: ${escapeHtml(sup.model_selection?.method ?? "Chưa có thông tin")}. Mốc cắt: ${escapeHtml(split.cutoff_date ?? "--")}.</p>
      <p class="dashboard-note">Số chuyến hủy trong tập test: ${escapeHtml(split.n_cancelled_test ?? "--")}; cần thận trọng khi diễn giải xác suất.</p>
    </div>
    <div class="card dashboard-section"><h3 class="card-title">Mô hình hồi quy</h3><p>MAE: <strong>${formatNumber(sup.regression.MAE_minutes)} phút</strong> · RMSE: <strong>${formatNumber(sup.regression.RMSE_minutes)} phút</strong></p></div>
    <div class="card dashboard-section"><h3 class="card-title">10 đặc trưng quan trọng</h3><div class="dashboard-table"><table class="data-table"><thead><tr><th>Đặc trưng</th><th>Mức độ quan trọng</th><th>Phương pháp</th></tr></thead>
      <tbody>${imps.map((item) => `<tr><td>${escapeHtml(item.feature)}</td><td>${formatPercent(numberOrNull(item.importance) === null ? null : item.importance * 100)}</td><td>${escapeHtml(item.method)}</td></tr>`).join("")}</tbody></table></div></div>`;
}

function renderClusterProfiles(data, container) {
  if (!data?.profiles || typeof data.profiles !== "object" || !data.anomalies || typeof data.anomalies !== "object") {
    throw new Error("Thiếu hồ sơ cụm hoặc thống kê bất thường trong API.");
  }
  const profiles = Object.values(data.profiles);
  const anomalies = data.anomalies;
  container.innerHTML = `
    <div class="card"><h3 class="card-title">Hồ sơ cụm K-Means</h3><div class="dashboard-table"><table class="data-table"><thead><tr><th>Cụm</th><th>Tên</th><th>Số chuyến</th><th>Tỷ lệ trễ</th><th>Tỷ lệ hủy</th><th>Trễ trung bình</th><th>Rủi ro</th></tr></thead>
      <tbody>${profiles.map((c) => `<tr><td>${escapeHtml(c.cluster_id)}</td><td>${escapeHtml(c.name ?? c.cluster_name)}</td><td>${escapeHtml(c.flight_count)}</td><td>${formatPercent(c.delay_rate_pct)}</td><td>${formatPercent(c.cancellation_rate_pct)}</td><td>${formatNumber(c.avg_delay_minutes)} phút</td><td>${escapeHtml(c.risk_tier)}</td></tr>`).join("")}</tbody></table></div></div>
    <div class="card dashboard-section"><h3 class="card-title">Chuyến bay bất thường (Isolation Forest)</h3><p>Số chuyến: <strong>${escapeHtml(anomalies.anomaly_count)}</strong> · Tỷ lệ: <strong>${formatPercent(anomalies.anomaly_percentage)}</strong></p></div>`;
}

function renderAssociationRules(data, container) {
  if (!Array.isArray(data)) throw new Error("Tập luật trả về không hợp lệ.");
  // Backend returns up to 50 rules sorted by descending lift; the dashboard shows 15.
  const rules = data.slice(0, 15);
  container.innerHTML = `<div class="card"><h3 class="card-title">15 luật kết hợp có Lift cao nhất</h3>
    <div class="dashboard-table"><table class="data-table"><thead><tr><th>Tiền đề</th><th>Hệ quả</th><th>Support</th><th>Confidence</th><th>Lift</th></tr></thead>
    <tbody>${rules.map((r) => `<tr><td>${escapeHtml(r.Antecedent)}</td><td>${escapeHtml(r.Consequent)}</td><td>${formatPercent(r["Support (%)"])}</td><td>${formatPercent(r["Confidence (%)"])}</td><td>${formatNumber(r.Lift)}</td></tr>`).join("")}</tbody></table></div>
    <p class="dashboard-note">Luật thể hiện mối liên hệ trong dữ liệu, không tự chứng minh quan hệ nhân quả.</p></div>`;
}

// ==========================================================================
// FlightRisk VN - Web Client Application
// Modern Data Mining & Prediction Engine with Ground Truth Auditing (0 Emojis)
// ==========================================================================
"use strict";

let allAuditRecords = [];
let activeAuditRecord = null;
let allFidsFlights = [];

document.addEventListener("DOMContentLoaded", () => {
  initNavigationTabs();
  initSliders();
  initPredictionForm();
  initFidsIntegration();
  initAuditExplorer();
  
  // Trigger initial inference with clean default inputs
  setTimeout(() => document.getElementById("predict-form")?.requestSubmit(), 50);
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

// ==========================================================================
// 1. Navigation Tabs
// ==========================================================================
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

// ==========================================================================
// 2. Sliders & Live Value Readouts
// ==========================================================================
function initSliders() {
  ["precip", "wind", "vis", "load"].forEach((id) => {
    const slider = byId(id);
    const label = byId(`${id}-val`);
    if (slider && label) {
      slider.addEventListener("input", () => { label.textContent = slider.value; });
    }
  });

  const swapBtn = byId("swap-route-btn");
  if (swapBtn) {
    swapBtn.addEventListener("click", () => {
      const orig = byId("origin");
      const dest = byId("destination");
      if (orig && dest) {
        const tmp = orig.value;
        orig.value = dest.value;
        dest.value = tmp;
        updateBoardingSummary();
      }
    });
  }

  ["origin", "destination", "airline", "sched-time", "flight-date"].forEach(id => {
    const el = byId(id);
    if (el) el.addEventListener("change", updateBoardingSummary);
  });
}

function updateBoardingSummary() {
  const orig = byId("origin")?.value || "SGN";
  const dest = byId("destination")?.value || "HAN";
  const airline = byId("airline")?.value || "VJ";
  const time = byId("sched-time")?.value || "19:30";
  const date = byId("flight-date")?.value || "2026-11-06";

  if (byId("summary-origin")) byId("summary-origin").textContent = orig;
  if (byId("summary-dest")) byId("summary-dest").textContent = dest;
  if (byId("summary-airline")) byId("summary-airline").innerHTML = `Hãng: <strong>${escapeHtml(airline)}</strong>`;
  if (byId("summary-time")) byId("summary-time").innerHTML = `Giờ: <strong>${escapeHtml(time)}</strong> (${escapeHtml(date)})`;

  const isGolden = (orig === "SGN" && dest === "HAN") || (orig === "HAN" && dest === "SGN");
  if (byId("summary-tag")) {
    byId("summary-tag").textContent = isGolden ? "Trục Vàng SGN-HAN" : `${orig}-${dest}`;
  }
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

// ==========================================================================
// 3. Prediction Submission & Validation
// ==========================================================================
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
  
  const auditVerdict = byId("res-audit-verdict");
  if (auditVerdict) {
    auditVerdict.style.display = "none";
    auditVerdict.innerHTML = "";
  }
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
    if (button) { button.disabled = false; button.textContent = "Chạy Dự Báo & Đánh Giá Rủi Ro"; }
  };
  
  const invalidate = () => {
    sequence += 1;
    activeRequest?.abort();
    activeRequest = null;
    unlock();
    clearPrediction("Thông số đã thay đổi. Hãy bấm Chạy Dự Báo để xem kết quả mới.");
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
    
    clearPrediction("Đang tải kết quả từ mô hình khóa băm SHA-256...", "ĐANG DỰ BÁO");
    if (button) { button.disabled = true; button.textContent = "Đang suy luận..."; }
    
    try {
      const inputs = readFlightInputs();
      updateBoardingSummary();
      
      const result = await requestJson("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputs),
      }, controller);
      
      validatePrediction(result);
      if (requestId === sequence) {
        renderPredictionResult(result, inputs);
      }
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

// ==========================================================================
// 4. Render Prediction Result & Ground Truth Verification
// ==========================================================================
function renderPredictionResult(data, inputs = null) {
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
    empty.style.fontSize = "0.78rem";
    empty.style.color = "var(--text-muted)";
    empty.textContent = "Không có luật cảnh báo kích hoạt cho chuyến bay này.";
    list.appendChild(empty);
  } else {
    data.matched_rules.forEach((rule) => {
      const row = document.createElement("div");
      row.className = "rule-item";
      const confidence = rule.confidence_pct ?? rule.confidence;
      row.innerHTML = `<div class="rule-ante">${escapeHtml(rule.antecedent)} &rarr; <strong>${escapeHtml(rule.consequent)}</strong></div>
        <div class="rule-meta" style="font-size: 0.72rem; color: var(--text-dim); margin-top: 0.2rem;">Độ tin cậy: <strong>${formatPercent(confidence)}</strong> | Lift: <strong>${formatNumber(rule.lift)}x</strong></div>`;
      list.appendChild(row);
    });
  }

  // Check Ground Truth Audit Verification
  renderAuditVerdictBanner(data, inputs);
}

function renderAuditVerdictBanner(data, inputs) {
  const banner = byId("res-audit-verdict");
  if (!banner) return;

  let matchedAudit = activeAuditRecord;
  
  // If not explicitly set via audit table, look for a matching flight in allAuditRecords
  if (!matchedAudit && inputs && allAuditRecords.length > 0) {
    matchedAudit = allAuditRecords.find(a => 
      a.airline_code === inputs.airline_code &&
      a.airport === inputs.airport &&
      a.scheduled_time === inputs.scheduled_time
    );
  }

  if (matchedAudit) {
    const isCorrect = (data.prediction === matchedAudit.ground_truth.actual_label);
    const actualLabel = matchedAudit.ground_truth.actual_label;
    const actualDelay = matchedAudit.ground_truth.actual_delay_min;
    const actualTime = matchedAudit.ground_truth.actual_time;
    const predDelay = data.estimated_delay_min;
    const absDiff = Math.abs(predDelay - actualDelay).toFixed(1);

    banner.className = `audit-verdict-card ${isCorrect ? 'verdict-correct' : 'verdict-incorrect'}`;
    banner.style.display = "block";
    banner.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <span class="badge ${isCorrect ? 'badge-emerald' : 'badge-rose'}" style="font-size: 0.78rem;">
          ĐỐI SOÁT THỰC TẾ: ${isCorrect ? 'ĐÚNG (CORRECT)' : 'SAI / LỆCH (MISCLASSIFICATION)'}
        </span>
        <span style="font-size: 0.74rem; font-family: var(--font-mono); color: var(--text-dim);">Chuyến bay: ${escapeHtml(matchedAudit.flight_no)}</span>
      </div>
      <div style="font-size: 0.8rem; line-height: 1.5; color: var(--text);">
        <div>&bull; <strong>Dự báo của Web:</strong> <span style="font-weight: 700; color: ${data.prediction === 'on_time' ? '#059669' : '#d97706'};">${data.prediction.toUpperCase()}</span> (Trễ dự kiến: ${predDelay} phút)</div>
        <div>&bull; <strong>Thực tế ghi nhận:</strong> <span style="font-weight: 700; color: ${actualLabel === 'on_time' ? '#059669' : '#d97706'};">${actualLabel.toUpperCase()}</span> (Cất cánh thực tế: ${escapeHtml(actualTime)}, Trễ thực tế: ${actualDelay} phút)</div>
        <div>&bull; <strong>Sai số dự báo hồi quy:</strong> <span style="font-family: var(--font-mono); font-weight: 700;">|${predDelay} - ${actualDelay}| = ${absDiff} phút</span> (${Number(absDiff) <= 15 ? 'Đạt chuẩn ICAO &le; 15m' : 'Vượt ngưỡng sai số'})</div>
      </div>
      <div style="margin-top: 0.4rem; padding-top: 0.4rem; border-top: 1px dashed var(--border); font-size: 0.75rem; color: var(--text-muted);">
        <strong>Phân tích nguyên nhân:</strong> ${escapeHtml(matchedAudit.audit.note)}
      </div>
    `;
  } else {
    banner.style.display = "none";
    banner.innerHTML = "";
  }
}

// ==========================================================================
// 5. FIDS Live Feed Integration & Presets
// ==========================================================================
async function initFidsIntegration() {
  const countBadge = byId("fids-count-badge");
  const flightSelect = byId("fids-flight-select");
  const tableBody = byId("fids-table-body");
  const btnApply = byId("btn-fids-apply");
  const btnToggleTable = byId("btn-toggle-fids-table");
  const tableContainer = byId("fids-table-container");

  try {
    const flights = await requestJson("/api/fids-flights");
    if (!Array.isArray(flights) || flights.length === 0) {
      if (countBadge) countBadge.textContent = "Không có dữ liệu FIDS";
      return;
    }
    allFidsFlights = flights;
    if (countBadge) countBadge.textContent = `${flights.length} chuyến bay sẵn sàng`;

    populateFidsSelect(flights);
    populateFidsTable(flights);

    // Airport Filter
    const filterGroup = byId("fids-airport-filter");
    if (filterGroup) {
      filterGroup.querySelectorAll("button").forEach(btn => {
        btn.addEventListener("click", () => {
          filterGroup.querySelectorAll("button").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          const apt = btn.dataset.airport;
          const filtered = (apt === "ALL") ? allFidsFlights : allFidsFlights.filter(f => f.airport === apt);
          populateFidsSelect(filtered);
          populateFidsTable(filtered);
        });
      });
    }

    if (btnApply) {
      btnApply.addEventListener("click", () => {
        const val = flightSelect?.value;
        if (!val) return;
        const flight = allFidsFlights.find(f => f.flight_no === val || `${f.flight_no}_${f.airport}` === val);
        if (flight) applyFlightToForm(flight);
      });
    }

    if (btnToggleTable && tableContainer) {
      btnToggleTable.addEventListener("click", () => {
        const isHidden = tableContainer.style.display === "none";
        tableContainer.style.display = isHidden ? "block" : "none";
        btnToggleTable.textContent = isHidden ? "Thu Gọn Bảng FIDS" : "Xem Bảng FIDS";
      });
    }
  } catch (err) {
    if (countBadge) countBadge.textContent = "Lỗi FIDS";
  }

  // Presets
  document.querySelectorAll(".preset-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const p = chip.dataset.preset;
      applyPreset(p);
    });
  });
}

function populateFidsSelect(flights) {
  const select = byId("fids-flight-select");
  if (!select) return;
  select.innerHTML = '<option value="">-- Chọn một chuyến bay FIDS để nạp thông số tức thì --</option>' +
    flights.map(f => `<option value="${escapeHtml(f.flight_no)}_${escapeHtml(f.airport)}">${escapeHtml(f.flight_no)} &bull; ${escapeHtml(f.airline)} &bull; ${escapeHtml(f.origin)} &rarr; ${escapeHtml(f.destination)} &bull; ${escapeHtml(f.scheduled_time)} (FIDS: ${escapeHtml(f.status)})</option>`).join("");
}

function populateFidsTable(flights) {
  const tbody = byId("fids-table-body");
  if (!tbody) return;
  tbody.innerHTML = flights.slice(0, 25).map(f => `
    <tr>
      <td><strong>${escapeHtml(f.flight_no)}</strong></td>
      <td><span class="badge badge-slate">${escapeHtml(f.airport)}</span></td>
      <td>${escapeHtml(f.origin)} &rarr; ${escapeHtml(f.destination)}</td>
      <td>${escapeHtml(f.scheduled_time)}</td>
      <td>${escapeHtml(f.status)}</td>
      <td>${escapeHtml(f.precipitation)} mm</td>
      <td>${escapeHtml(f.wind_gusts_10m)} km/h</td>
      <td>${escapeHtml(f.visibility)} m</td>
      <td>${escapeHtml(f.airport_load)}</td>
      <td>
        <button type="button" class="btn-sm-action" onclick="applyFidsByNo('${escapeHtml(f.flight_no)}')">Dự báo</button>
      </td>
    </tr>
  `).join("");
}

window.applyFidsByNo = function(flightNo) {
  const flight = allFidsFlights.find(f => f.flight_no === flightNo);
  if (flight) applyFlightToForm(flight);
};

function applyFlightToForm(flight) {
  activeAuditRecord = null; // reset specific audit lock
  
  if (byId("airline")) byId("airline").value = flight.airline_code || "VN";
  if (byId("airport")) byId("airport").value = flight.airport || "SGN";
  if (byId("origin")) byId("origin").value = flight.origin || flight.airport || "SGN";
  if (byId("destination")) byId("destination").value = flight.destination || "HAN";
  if (byId("direction")) byId("direction").value = flight.direction || "DEPARTURES";
  if (byId("route-type")) byId("route-type").value = flight.route_type || "domestic";
  if (byId("flight-date")) byId("flight-date").value = flight.flight_date || "2026-11-06";
  if (byId("sched-time")) byId("sched-time").value = flight.scheduled_time ? flight.scheduled_time.substring(0, 5) : "12:00";
  
  const precip = flight.precipitation ?? 0.0;
  const wind = flight.wind_gusts_10m ?? 10.0;
  const vis = flight.visibility ?? 10000.0;
  const load = flight.airport_load ?? 20;

  if (byId("precip")) { byId("precip").value = precip; byId("precip-val").textContent = precip; }
  if (byId("wind")) { byId("wind").value = wind; byId("wind-val").textContent = wind; }
  if (byId("vis")) { byId("vis").value = vis; byId("vis-val").textContent = vis; }
  if (byId("load")) { byId("load").value = load; byId("load-val").textContent = load; }
  if (byId("weather-code")) byId("weather-code").value = String(flight.weather_code || 0);

  updateBoardingSummary();
  byId("predict-form")?.requestSubmit();
}

function applyPreset(presetKey) {
  const presets = {
    clear_day: {
      airline: "VN", airport: "SGN", origin: "SGN", destination: "HAN",
      direction: "DEPARTURES", route_type: "domestic", time: "08:30", date: "2026-11-06",
      precip: 0.0, wind: 10, vis: 10000, load: 18, weather_code: "0"
    },
    windy_afternoon: {
      airline: "QH", airport: "DAD", origin: "DAD", destination: "SGN",
      direction: "DEPARTURES", route_type: "domestic", time: "14:15", date: "2026-11-06",
      precip: 1.5, wind: 45, vis: 6000, load: 28, weather_code: "61"
    },
    stormy_night: {
      airline: "VJ", airport: "SGN", origin: "SGN", destination: "HAN",
      direction: "DEPARTURES", route_type: "domestic", time: "21:45", date: "2026-11-06",
      precip: 18.0, wind: 55, vis: 1500, load: 42, weather_code: "95"
    }
  };

  const p = presets[presetKey];
  if (!p) return;

  activeAuditRecord = null;
  if (byId("airline")) byId("airline").value = p.airline;
  if (byId("airport")) byId("airport").value = p.airport;
  if (byId("origin")) byId("origin").value = p.origin;
  if (byId("destination")) byId("destination").value = p.destination;
  if (byId("direction")) byId("direction").value = p.direction;
  if (byId("route-type")) byId("route-type").value = p.route_type;
  if (byId("flight-date")) byId("flight-date").value = p.date;
  if (byId("sched-time")) byId("sched-time").value = p.time;

  if (byId("precip")) { byId("precip").value = p.precip; byId("precip-val").textContent = p.precip; }
  if (byId("wind")) { byId("wind").value = p.wind; byId("wind-val").textContent = p.wind; }
  if (byId("vis")) { byId("vis").value = p.vis; byId("vis-val").textContent = p.vis; }
  if (byId("load")) { byId("load").value = p.load; byId("load-val").textContent = p.load; }
  if (byId("weather-code")) byId("weather-code").value = p.weather_code;

  updateBoardingSummary();
  byId("predict-form")?.requestSubmit();
}

// ==========================================================================
// 6. Ground Truth Audit Explorer (Truy cứu Đúng hay Sai)
// ==========================================================================
async function initAuditExplorer() {
  const tbody = byId("audit-table-body");
  const searchInput = byId("audit-search-input");

  try {
    const records = await requestJson("/api/audit-flights");
    if (!Array.isArray(records) || records.length === 0) {
      if (tbody) tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding: 1rem;">Không có dữ liệu đối soát thực tế.</td></tr>';
      return;
    }

    allAuditRecords = records;
    const correctCount = records.filter(r => r.audit.is_correct).length;
    const incorrectCount = records.length - correctCount;

    if (byId("audit-stat-correct")) byId("audit-stat-correct").textContent = `Đúng: ${correctCount} / ${records.length} (${(correctCount/records.length*100).toFixed(1)}%)`;
    if (byId("audit-stat-incorrect")) byId("audit-stat-incorrect").textContent = `Sai / Lệch: ${incorrectCount} / ${records.length} (${(incorrectCount/records.length*100).toFixed(1)}%)`;

    renderAuditTable(allAuditRecords);

    if (searchInput) {
      searchInput.addEventListener("input", () => {
        const query = searchInput.value.toLowerCase().trim();
        const activeChip = document.querySelector(".audit-filter-chips .chip-btn.active")?.dataset.filter || "all";
        filterAndRenderAudit(activeChip, query);
      });
    }
  } catch (err) {
    if (tbody) tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color: var(--danger); padding: 1rem;">Lỗi tải dữ liệu đối soát: ${err.message}</td></tr>`;
  }
}

function filterAndRenderAudit(filterType, query = "") {
  let list = allAuditRecords;
  if (filterType === "correct") {
    list = list.filter(r => r.audit.is_correct);
  } else if (filterType === "incorrect") {
    list = list.filter(r => !r.audit.is_correct);
  }

  if (query) {
    list = list.filter(r => 
      r.flight_no.toLowerCase().includes(query) ||
      r.origin.toLowerCase().includes(query) ||
      r.destination.toLowerCase().includes(query) ||
      r.airport.toLowerCase().includes(query) ||
      r.airline_code.toLowerCase().includes(query)
    );
  }

  renderAuditTable(list);
}

window.filterAuditTable = function(filterType, btn) {
  document.querySelectorAll(".audit-filter-chips .chip-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  const query = byId("audit-search-input")?.value.toLowerCase().trim() || "";
  filterAndRenderAudit(filterType, query);
};

function renderAuditTable(records) {
  const tbody = byId("audit-table-body");
  if (!tbody) return;

  if (records.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">Không tìm thấy chuyến bay phù hợp với tiêu chí lọc.</td></tr>';
    return;
  }

  tbody.innerHTML = records.map(r => {
    const isCorrect = r.audit.is_correct;
    const predLabel = r.prediction.predicted_label;
    const actualLabel = r.ground_truth.actual_label;
    
    return `
      <tr>
        <td>
          <strong>${escapeHtml(r.flight_no)}</strong>
          <div style="font-size: 0.72rem; color: var(--text-dim);">${escapeHtml(r.flight_date)}</div>
        </td>
        <td>
          <div>${escapeHtml(r.origin)} &rarr; ${escapeHtml(r.destination)}</div>
          <div style="font-size: 0.72rem; color: var(--text-dim); font-family: var(--font-mono);">Giờ bay: ${escapeHtml(r.scheduled_time)}</div>
        </td>
        <td>
          <span class="badge ${predLabel === 'on_time' ? 'badge-emerald' : (predLabel === 'delayed' ? 'badge-amber' : 'badge-rose')}">
            ${predLabel.toUpperCase()}
          </span>
          <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 0.15rem;">Dự kiến trễ: <strong>${r.prediction.predicted_delay_min}m</strong></div>
        </td>
        <td>
          <span class="badge ${actualLabel === 'on_time' ? 'badge-emerald' : (actualLabel === 'delayed' ? 'badge-amber' : 'badge-rose')}">
            ${actualLabel.toUpperCase()}
          </span>
          <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 0.15rem;">Cất cánh: <strong>${escapeHtml(r.actual_time)}</strong> (Trễ ${r.ground_truth.actual_delay_min}m)</div>
        </td>
        <td>
          <span class="badge ${isCorrect ? 'badge-emerald' : 'badge-rose'}">
            ${isCorrect ? 'ĐÚNG (CORRECT)' : 'SAI / LỆCH'}
          </span>
          <div style="font-size: 0.72rem; color: var(--text-dim); margin-top: 0.15rem;">Sai số: ${r.audit.delay_error_min}m</div>
        </td>
        <td style="font-size: 0.76rem; color: var(--text-muted); line-height: 1.4;">
          ${escapeHtml(r.audit.note)}
          <div style="margin-top: 0.35rem;">
            <button type="button" class="btn-sm-action" onclick="applyAuditFlight('${escapeHtml(r.id)}')">
              Thử Dự Báo Chuyến Này
            </button>
          </div>
        </td>
      </tr>
    `;
  }).join("");
}

window.applyAuditFlight = function(auditId) {
  const audit = allAuditRecords.find(a => a.id === auditId);
  if (!audit) return;

  activeAuditRecord = audit;

  if (byId("airline")) byId("airline").value = audit.airline_code || "VN";
  if (byId("airport")) byId("airport").value = audit.airport || "SGN";
  if (byId("origin")) byId("origin").value = audit.origin || "SGN";
  if (byId("destination")) byId("destination").value = audit.destination || "HAN";
  if (byId("direction")) byId("direction").value = audit.direction || "DEPARTURES";
  if (byId("route-type")) byId("route-type").value = audit.route_type || "domestic";
  if (byId("flight-date")) byId("flight-date").value = audit.flight_date || "2026-11-06";
  if (byId("sched-time")) byId("sched-time").value = audit.scheduled_time || "12:00";

  const w = audit.weather || {};
  if (byId("precip")) { byId("precip").value = w.precipitation ?? 0.0; byId("precip-val").textContent = w.precipitation ?? 0.0; }
  if (byId("wind")) { byId("wind").value = w.wind ?? 10.0; byId("wind-val").textContent = w.wind ?? 10.0; }
  if (byId("vis")) { byId("vis").value = w.visibility ?? 10000.0; byId("vis-val").textContent = w.visibility ?? 10000.0; }
  if (byId("load")) { byId("load").value = w.load ?? 20; byId("load-val").textContent = w.load ?? 20; }

  updateBoardingSummary();

  // Scroll smoothly to simulator form
  byId("simulator-tab")?.scrollIntoView({ behavior: "smooth" });

  // Trigger form prediction
  byId("predict-form")?.requestSubmit();
};

// ==========================================================================
// 7. Panel Loaders (Models, Clusters, Rules)
// ==========================================================================
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
  button.style.padding = "0.4rem 0.8rem";
  button.style.marginTop = "0.5rem";
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
    <div class="card" style="margin-top: 1rem;">
      <div class="card-header-row">
        <div>
          <h3 class="card-title">Bảng So Sánh Các Thuật Toán Phân Loại (Test Set)</h3>
          <p class="card-subtitle">Mô hình tốt nhất được chọn lọc dựa trên Balanced Accuracy và ROC-AUC</p>
        </div>
        <span class="card-step-badge">Test: ${escapeHtml(split.test_rows ?? "--")} mẫu</span>
      </div>
      <div style="overflow-x: auto;">
        <table class="data-table">
          <thead>
            <tr>
              <th>Thuật toán</th>
              <th>Accuracy</th>
              <th>Macro F1</th>
              <th>Macro Recall</th>
              <th>ROC-AUC</th>
              <th>Trạng thái</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map(([name, m]) => `
              <tr style="${name === selected ? 'background: #f0fdfa; font-weight: 600;' : ''}">
                <td><strong>${escapeHtml(name)}</strong></td>
                <td>${formatPercent(numberOrNull(m.accuracy) === null ? null : m.accuracy * 100, 1)}</td>
                <td>${formatNumber(m.f1_macro, 4)}</td>
                <td>${formatNumber(m.recall_macro, 4)}</td>
                <td>${formatNumber(m.roc_auc_ovr, 4)}</td>
                <td>${name === selected ? '<span class="badge badge-teal">Đã chọn</span>' : '--'}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </div>
    <div class="card" style="margin-top: 1rem;">
      <div class="card-header-row">
        <div>
          <h3 class="card-title">Mô hình Hồi quy Trễ (HistGradientBoosting Regressor)</h3>
          <p class="card-subtitle">Hàm mất mát L1 (MAE) giúp kháng outlier trễ cực đoan 300-400 phút</p>
        </div>
      </div>
      <p style="font-size: 0.88rem; color: var(--text);">
        MAE dự báo: <strong>${formatNumber(sup.regression.MAE_minutes)} phút</strong> &bull; RMSE: <strong>${formatNumber(sup.regression.RMSE_minutes)} phút</strong>
      </p>
    </div>
  `;
}

function renderClusterProfiles(data, container) {
  if (!data?.profiles || typeof data.profiles !== "object" || !data.anomalies || typeof data.anomalies !== "object") {
    throw new Error("Thiếu hồ sơ cụm hoặc thống kê bất thường trong API.");
  }
  const profiles = Object.values(data.profiles);
  const anomalies = data.anomalies;
  
  container.innerHTML = `
    <div class="card" style="margin-top: 1rem;">
      <div class="card-header-row">
        <div>
          <h3 class="card-title">Bảng Hồ Sơ 4 Cụm Chuyến Bay (K-Means k=4)</h3>
          <p class="card-subtitle">Silhouette đạt 0.300 cao nhất khi thử nghiệm k từ 2 đến 8</p>
        </div>
      </div>
      <div style="overflow-x: auto;">
        <table class="data-table">
          <thead>
            <tr>
              <th>Cụm</th>
              <th>Tên Cụm</th>
              <th>Số Chuyến</th>
              <th>Tỷ Lệ Trễ</th>
              <th>Tỷ Lệ Hủy</th>
              <th>Trễ Trung Bình</th>
              <th>Mức Rủi Ro</th>
            </tr>
          </thead>
          <tbody>
            ${profiles.map((c) => `
              <tr>
                <td><strong>#${escapeHtml(c.cluster_id)}</strong></td>
                <td><strong>${escapeHtml(c.name ?? c.cluster_name)}</strong></td>
                <td>${escapeHtml(c.flight_count)}</td>
                <td>${formatPercent(c.delay_rate_pct)}</td>
                <td>${formatPercent(c.cancellation_rate_pct)}</td>
                <td>${formatNumber(c.avg_delay_minutes)} phút</td>
                <td><span class="badge ${c.risk_tier === 'HIGH' ? 'badge-rose' : (c.risk_tier === 'MEDIUM' ? 'badge-amber' : 'badge-emerald')}">${escapeHtml(c.risk_tier)}</span></td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </div>
    <div class="card" style="margin-top: 1rem;">
      <h3 class="card-title">Phát Hiện Dị Biệt (Isolation Forest)</h3>
      <p style="font-size: 0.86rem; color: var(--text); margin-top: 0.3rem;">
        Đã cô lập <strong>${escapeHtml(anomalies.anomaly_count)} chuyến bay</strong> (${formatPercent(anomalies.anomaly_percentage)}) có dấu hiệu lỗi radar hoặc trễ đặc biệt nghiêm trọng.
      </p>
    </div>
  `;
}

function renderAssociationRules(data, container) {
  if (!Array.isArray(data)) throw new Error("Tập luật trả về không hợp lệ.");
  const rules = data.slice(0, 15);
  
  container.innerHTML = `
    <div class="card" style="margin-top: 1rem;">
      <div class="card-header-row">
        <div>
          <h3 class="card-title">15 Luật Kết Hợp Có Lift Cao Nhất (FP-Growth)</h3>
          <p class="card-subtitle">Lift cực đại đạt 2.15 ở các khung giờ cao điểm và điều kiện mưa dông</p>
        </div>
      </div>
      <div style="overflow-x: auto;">
        <table class="data-table">
          <thead>
            <tr>
              <th>Tiền Đề (Antecedent)</th>
              <th>Hệ Quả (Consequent)</th>
              <th>Support</th>
              <th>Confidence</th>
              <th>Lift</th>
            </tr>
          </thead>
          <tbody>
            ${rules.map((r) => `
              <tr>
                <td><code>${escapeHtml(r.Antecedent)}</code></td>
                <td><strong>${escapeHtml(r.Consequent)}</strong></td>
                <td>${formatPercent(r["Support (%)"])}</td>
                <td>${formatPercent(r["Confidence (%)"])}</td>
                <td><strong>${formatNumber(r.Lift)}x</strong></td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

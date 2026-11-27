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

  const DOMESTIC_AIRPORTS = new Set(["SGN", "HAN", "DAD", "CXR", "PQC"]);
  const syncFormValidity = () => {
    const orig = byId("origin");
    const dest = byId("destination");
    const route = byId("route-type");
    const dir = byId("direction");
    const ap = byId("airport");
    if (!orig || !dest || !route || !dir || !ap) return;

    dest.setCustomValidity(orig.value && dest.value && orig.value === dest.value
      ? "Sân bay đến không được trùng sân bay đi." : "");

    const isIntl = !DOMESTIC_AIRPORTS.has(orig.value) || !DOMESTIC_AIRPORTS.has(dest.value);
    if (isIntl && route.value === "domestic") {
      route.setCustomValidity("Đường bay quốc tế phải chọn Loại đường bay là Quốc tế.");
    } else if (!isIntl && route.value === "international") {
      route.setCustomValidity("Đường bay giữa 2 cảng nội địa Việt Nam phải chọn Nội địa.");
    } else {
      route.setCustomValidity("");
    }

    const expectedAirport = dir.value === "ARRIVALS" ? dest.value : orig.value;
    if (ap.value !== expectedAirport) {
      ap.setCustomValidity(`Chiều ${dir.value} yêu cầu cảng khảo sát là ${expectedAirport}.`);
    } else {
      ap.setCustomValidity("");
    }
  };

  const swapBtn = byId("swap-route-btn");
  if (swapBtn) {
    swapBtn.addEventListener("click", () => {
      const orig = byId("origin");
      const dest = byId("destination");
      const dir = byId("direction");
      const ap = byId("airport");
      if (orig && dest) {
        const tmp = orig.value;
        orig.value = dest.value;
        dest.value = tmp;
        if (ap && dir) {
          const targetAp = dir.value === "ARRIVALS" ? dest.value : orig.value;
          if ([...ap.options].some(o => o.value === targetAp)) {
            ap.value = targetAp;
          }
        }
        syncFormValidity();
        updateBoardingSummary();
        byId("predict-form")?.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });
  }

  ["origin", "destination", "airline", "sched-time", "flight-date", "direction", "route-type", "airport"].forEach(id => {
    const el = byId(id);
    if (el) {
      el.addEventListener("change", () => {
        syncFormValidity();
        updateBoardingSummary();
      });
    }
  });
  window.__syncFlightFormValidity = syncFormValidity;
  syncFormValidity();
  updateBoardingSummary();
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
    const bar = byId(id);
    const txt = byId(`${id}-text`);
    if (bar) bar.style.width = "0%";
    if (txt) txt.textContent = "--";
  });
  
  const fList = byId("res-factors-list");
  if (fList) fList.innerHTML = "<li>Chờ kết quả phân tích tín hiệu...</li>";
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
      !values || !values.every(isProbability) || Math.abs(values.reduce((a, b) => a + b, 0) - 1) > 0.01 ||
      typeof data.estimated_delay_min !== "number" || !Number.isFinite(data.estimated_delay_min) ||
      data.estimated_delay_min < 0 || !data.cluster_profile || typeof data.cluster_profile !== "object" ||
      !Array.isArray(data.matched_rules)) {
    throw new Error("Kết quả dự báo không đúng cấu trúc dữ liệu 3 lớp ICAO.");
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
    on_time: "Dự báo: ĐÚNG GIỜ (ON-TIME)",
    delayed: `Dự báo: NGUY CƠ TRỄ CHUYẾN (~${data.estimated_delay_min} PHÚT)`,
    cancelled: "Dự báo: CẢNH BÁO NGUY CƠ HỦY CHUYẾN (CANCELLED)",
  };
  byId("res-prediction-text").textContent = messages[data.prediction] || messages.delayed;
  byId("res-risk-score").textContent = `${Math.round(data.risk_score * 100)}%`;
  
  setBar("prob-ontime", data.probabilities.on_time);
  setBar("prob-delayed", data.probabilities.delayed);
  setBar("prob-cancelled", data.probabilities.cancelled);

  const fList = byId("res-factors-list");
  if (fList) {
    const factors = Array.isArray(data.factors) ? data.factors : [];
    fList.innerHTML = factors.length
      ? factors.map((f) => `<li>${escapeHtml(f)}</li>`).join("")
      : "<li>Điều kiện khai thác ổn định</li>";
  }

  const profile = data.cluster_profile;
  byId("res-cluster-name").textContent = profile.cluster_name ?? profile.name ?? "Không có tên cụm";
  byId("res-cluster-rate").textContent = profile.disruption_rate_pct !== undefined
    ? `${profile.disruption_rate_pct}% gián đoạn (Trễ ${profile.delay_rate_pct}%)`
    : (numberOrNull(profile.delay_rate_pct) === null ? "--" : `${profile.delay_rate_pct}%`);
  
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
  
  if (!matchedAudit && inputs && allAuditRecords.length > 0) {
    matchedAudit = allAuditRecords.find(a => 
      a.airline_code === inputs.airline_code &&
      a.airport === inputs.airport &&
      a.scheduled_time === inputs.scheduled_time
    );
  }

  if (matchedAudit) {
    const actualLabel = matchedAudit.ground_truth.actual_label;
    const isCorrect = (data.prediction === actualLabel);
    const actualDelay = matchedAudit.ground_truth.actual_delay_min;
    const actualTime = matchedAudit.ground_truth.actual_time;
    const predDelay = data.estimated_delay_min;
    const errText = actualDelay === null || actualDelay === undefined
      ? "Chuyến hủy không áp dụng hồi quy phút trễ (delay_min = NaN)"
      : `|${predDelay} - ${actualDelay}| = ${Math.abs(predDelay - actualDelay).toFixed(1)} phút`;

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
        <div>&bull; <strong>Thực tế ghi nhận:</strong> <span style="font-weight: 700; color: ${actualLabel === 'on_time' ? '#059669' : '#d97706'};">${actualLabel.toUpperCase()}</span> (Cất cánh thực tế: ${escapeHtml(actualTime)}${actualDelay !== null && actualDelay !== undefined ? `, Trễ thực tế: ${actualDelay} phút` : ''})</div>
        <div>&bull; <strong>Sai số dự báo hồi quy:</strong> <span style="font-family: var(--font-mono); font-weight: 700;">${errText}</span></div>
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

  window.__syncFlightFormValidity?.();
  updateBoardingSummary();
  byId("predict-form")?.requestSubmit();
}

function applyPreset(presetKey) {
  const safePreset = {
    airline: "VN", airport: "SGN", origin: "SGN", destination: "HAN",
    direction: "DEPARTURES", route_type: "domestic", time: "08:00", date: "2026-11-06",
    precip: 0, wind: 10, vis: 10000, load: 15, weather_code: "0"
  };
  const windyPreset = {
    airline: "QH", airport: "DAD", origin: "DAD", destination: "SGN",
    direction: "DEPARTURES", route_type: "domestic", time: "14:45", date: "2026-11-06",
    precip: 1, wind: 32, vis: 6000, load: 26, weather_code: "61"
  };
  const stormPreset = {
    airline: "VJ", airport: "SGN", origin: "SGN", destination: "HAN",
    direction: "DEPARTURES", route_type: "domestic", time: "19:30", date: "2026-11-06",
    precip: 15, wind: 45, vis: 2000, load: 35, weather_code: "95"
  };
  const presets = {
    safe: safePreset,
    windy: windyPreset,
    storm: stormPreset,
    clear_day: safePreset,
    windy_afternoon: windyPreset,
    stormy_night: stormPreset,
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

  window.__syncFlightFormValidity?.();
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
    if (byId("audit-chip-all")) byId("audit-chip-all").textContent = `Tất Cả (${records.length})`;
    if (byId("audit-chip-correct")) byId("audit-chip-correct").textContent = `Chỉ Chuyến ĐÚNG (${correctCount})`;
    if (byId("audit-chip-incorrect")) byId("audit-chip-incorrect").textContent = `Chỉ Chuyến SAI / LỆCH (${incorrectCount})`;

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
  const selected = sup.model_selection?.selected ?? "XGBoost";
  const rows = Object.entries(clf).filter(([, metrics]) => metrics && typeof metrics === "object" && !Array.isArray(metrics));
  const split = sup.split || {};
  const binComp = Object.entries(sup.binary_ablation?.test_comparison || {});
  const binSelected = sup.binary_ablation?.best_model || "XGBoost";
  const horizons = Array.isArray(sup.horizon_ablation?.horizons) ? sup.horizon_ablation.horizons : [];
  const regModels = sup.regression_models && typeof sup.regression_models === "object" ? Object.entries(sup.regression_models) : [];
  const regSelected = sup.regression_details?.selected || "CatBoost_MAE";

  container.innerHTML = `
    <div class="card" style="margin-top: 1rem;">
      <div class="card-header-row">
        <div>
          <h3 class="card-title">Bảng Đối Chiếu 3 Lớp ICAO (On-time / Delayed / Cancelled — Holdout Test)</h3>
          <p class="card-subtitle">Tuyển chọn trên tập Validation theo Macro-F1; kiểm định độc lập trên Holdout Test (T-${escapeHtml(data.primary_horizon_hours ?? 3)}h)</p>
        </div>
        <span class="card-step-badge">Train ${escapeHtml(split.train ?? "--")} | Val ${escapeHtml(split.val ?? "--")} | Test ${escapeHtml(split.test ?? "--")}</span>
      </div>
      <div style="overflow-x: auto;">
        <table class="data-table">
          <thead>
            <tr>
              <th>Thuật toán</th>
              <th>Accuracy</th>
              <th>BalAcc</th>
              <th>Macro F1</th>
              <th>Delayed Recall</th>
              <th>Cancelled Recall</th>
              <th>ROC-AUC OvR</th>
              <th>LogLoss</th>
              <th>Trạng thái</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map(([name, m]) => `
              <tr style="${name === selected ? 'background: #f0fdfa; font-weight: 600;' : ''}">
                <td><strong>${escapeHtml(name)}</strong></td>
                <td>${formatPercent(numberOrNull(m.accuracy) === null ? null : m.accuracy * 100, 2)}</td>
                <td>${formatPercent(numberOrNull(m.balanced_accuracy) === null ? null : m.balanced_accuracy * 100, 2)}</td>
                <td>${formatNumber(m.f1_macro, 4)}</td>
                <td>${formatPercent(numberOrNull(m.delayed_recall) === null ? null : m.delayed_recall * 100, 1)}</td>
                <td>${formatPercent(numberOrNull(m.cancelled_recall) === null ? null : m.cancelled_recall * 100, 1)}</td>
                <td>${formatNumber(m.roc_auc_ovr ?? m.roc_auc, 4)}</td>
                <td>${formatNumber(m.log_loss, 4)}</td>
                <td>${name === selected ? '<span class="badge badge-teal">Production</span>' : '--'}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </div>

    ${binComp.length ? `
    <div class="card" style="margin-top: 1rem;">
      <div class="card-header-row">
        <div>
          <h3 class="card-title">Thí Nghiệm Đối Chứng Nhị Phân (Ablation: On-time vs Disrupted)</h3>
          <p class="card-subtitle">Đo lường độ nhạy khi gộp trễ &amp; hủy thành 1 lớp gián đoạn (Được chọn Val: ${escapeHtml(binSelected)})</p>
        </div>
      </div>
      <div style="overflow-x: auto;">
        <table class="data-table">
          <thead>
            <tr><th>Thuật toán</th><th>Accuracy</th><th>BalAcc</th><th>Macro F1</th><th>ROC-AUC</th><th>PR-AUC (Disrupted)</th><th>Brier</th></tr>
          </thead>
          <tbody>
            ${binComp.map(([bName, bm]) => `
              <tr style="${bName === binSelected ? 'background: #f8fafc; font-weight: 600;' : ''}">
                <td><strong>${escapeHtml(bName)}</strong></td>
                <td>${formatPercent((bm.accuracy || 0) * 100, 2)}</td>
                <td>${formatPercent((bm.balanced_accuracy || 0) * 100, 2)}</td>
                <td>${formatNumber(bm.f1_macro, 4)}</td>
                <td>${formatNumber(bm.roc_auc, 4)}</td>
                <td>${formatNumber(bm.average_precision_disrupted ?? bm.pr_auc_macro, 4)}</td>
                <td>${formatNumber(bm.brier_disrupted, 4)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </div>` : ""}

    ${horizons.length ? `
    <div class="card" style="margin-top: 1rem;">
      <div class="card-header-row">
        <div>
          <h3 class="card-title">Thí Nghiệm Ngưỡng Thời Gian Dự Báo (Horizon Ablation: T-1h / T-3h / T-6h)</h3>
          <p class="card-subtitle">So sánh tín hiệu Causal Historical Lookback trước giờ khởi hành theo từng chân trời</p>
        </div>
      </div>
      <div style="overflow-x: auto;">
        <table class="data-table">
          <thead>
            <tr><th>Horizon</th><th>Thuật toán tốt nhất</th><th>Val Macro-F1</th><th>Test Macro-F1</th><th>Test BalAcc</th><th>Delayed Recall</th><th>Cancelled Recall</th><th>LogLoss</th></tr>
          </thead>
          <tbody>
            ${horizons.map((h) => `
              <tr style="${h.is_primary ? 'background: #f0fdfa; font-weight: 600;' : ''}">
                <td><strong>T-${escapeHtml(h.horizon_hours)}h ${h.is_primary ? '(Chính)' : ''}</strong></td>
                <td>${escapeHtml(h.best_algorithm)}</td>
                <td>${formatNumber(h.val_f1_macro, 4)}</td>
                <td>${formatNumber(h.test_f1_macro, 4)}</td>
                <td>${formatPercent((h.test_balanced_accuracy || 0) * 100, 2)}</td>
                <td>${formatPercent((h.test_delayed_recall || 0) * 100, 1)}</td>
                <td>${formatPercent((h.test_cancelled_recall || 0) * 100, 1)}</td>
                <td>${formatNumber(h.test_log_loss, 4)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </div>` : ""}

    <div class="card" style="margin-top: 1rem;">
      <div class="card-header-row">
        <div>
          <h3 class="card-title">Hồi Quy Số Phút Trễ (Tabular L1/MAE Regression — Chỉ Chuyến Hoàn Tất)</h3>
          <p class="card-subtitle">Loại bỏ tuyệt đối chuyến hủy/diverted (never delay_min=0); đối chiếu Baseline Median (${escapeHtml(sup.regression_details?.baseline_median_MAE_minutes ?? "--")} phút)</p>
        </div>
      </div>
      <div class="kpi-strip" style="margin-top: 0.75rem;">
        ${regModels.map(([rName, rm]) => `
          <div class="kpi-card" style="${rName === regSelected ? 'border-left-color: var(--teal-primary); background: #f0fdfa;' : ''}">
            <div class="kpi-label">${escapeHtml(rName)} ${rName === regSelected ? '(Production)' : ''}</div>
            <div class="kpi-value">MAE ${formatNumber(rm.MAE_minutes, 2)}p</div>
            <div class="kpi-sub">MedAE: ${formatNumber(rm.MedAE_minutes, 2)}p &bull; RMSE: ${formatNumber(rm.RMSE_minutes, 2)}p &bull; R&sup2;: ${formatNumber(rm.R2_Score, 4)}</div>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function renderClusterProfiles(data, container) {
  if (!data?.profiles || typeof data.profiles !== "object" || !data.anomalies || typeof data.anomalies !== "object") {
    throw new Error("Thiếu hồ sơ cụm hoặc thống kê bất thường trong API.");
  }
  const profiles = Object.values(data.profiles);
  const anomalies = data.anomalies;
  const stab = data.kmeans?.seed_stability || {};
  
  container.innerHTML = `
    <div class="card" style="margin-top: 1rem;">
      <div class="card-header-row">
        <div>
          <h3 class="card-title">Bảng Hồ Sơ 4 Cụm Vận Hành (K-Means k=4 &bull; Seed ARI = ${formatNumber(stab.ari_mean, 4)})</h3>
          <p class="card-subtitle">${escapeHtml(data.kmeans?.suitability_note || "")}</p>
        </div>
      </div>
      <div style="overflow-x: auto;">
        <table class="data-table">
          <thead>
            <tr>
              <th>Cụm</th>
              <th>Chế Độ Khai Thác</th>
              <th>Số Chuyến</th>
              <th>Tải TB (ch/h)</th>
              <th>Tỷ Lệ Trễ</th>
              <th>Tỷ Lệ Hủy</th>
              <th>Khuyến Nghị</th>
            </tr>
          </thead>
          <tbody>
            ${profiles.map((c) => `
              <tr>
                <td><strong>#${escapeHtml(c.cluster_id)}</strong></td>
                <td><strong>${escapeHtml(c.name)}</strong><div style="font-size:0.74rem;color:var(--text-dim)">${escapeHtml(c.dominant_airport)} &bull; ${escapeHtml(c.dominant_airline)} &bull; ${escapeHtml(c.dominant_time_slot)}</div></td>
                <td>${escapeHtml(c.count ?? c.flight_count)} (${formatPercent(c.share_pct, 1)})</td>
                <td>${formatNumber(c.avg_airport_load, 1)}</td>
                <td>${formatPercent(c.delay_rate_pct, 1)}</td>
                <td>${formatPercent(c.cancel_rate_pct ?? c.cancellation_rate_pct, 2)}</td>
                <td style="font-size:0.76rem;color:var(--text-muted)">${escapeHtml(c.recommended_action)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </div>
    <div class="card" style="margin-top: 1rem;">
      <h3 class="card-title">Phát Hiện Dị Thường Khai Thác (Isolation Forest — Post-hoc Evaluation)</h3>
      <p style="font-size: 0.86rem; color: var(--text); margin-top: 0.3rem;">
        Cô lập <strong>${escapeHtml(anomalies.anomaly_count)} chuyến (${formatPercent(anomalies.anomaly_rate_pct, 2)})</strong>:
        tỷ lệ gián đoạn trong nhóm dị thường <strong>${formatPercent(anomalies.anomaly_disruption_rate_pct, 1)}</strong> so với <strong>${formatPercent(anomalies.normal_disruption_rate_pct, 1)}</strong> ở nhóm bình thường.
      </p>
    </div>
  `;
}

function renderAssociationRules(data, container) {
  const rulesList = Array.isArray(data) ? data : (Array.isArray(data?.rules) ? data.rules : []);
  if (!rulesList.length) throw new Error("Không có luật kết hợp trong API.");
  const rules = rulesList.slice(0, 25);
  
  container.innerHTML = `
    <div class="card" style="margin-top: 1rem;">
      <div class="card-header-row">
        <div>
          <h3 class="card-title">Top ${rules.length} / ${rulesList.length} Luật FP-Growth Phi Trùng Lặp (Train &rarr; Holdout Test)</h3>
          <p class="card-subtitle">Đã lọc luật dư thừa (Redundancy Pruning) và kiểm chứng khả năng tái lập trên Holdout Test</p>
        </div>
      </div>
      <div style="overflow-x: auto;">
        <table class="data-table">
          <thead>
            <tr>
              <th>Tiền Đề (Antecedent)</th>
              <th>Hệ Quả</th>
              <th>Train Sup</th>
              <th>Train Conf</th>
              <th>Train Lift</th>
              <th>Test Conf</th>
              <th>Test Lift</th>
              <th>Holdout</th>
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
                <td>${formatPercent(r.test_confidence_pct)}</td>
                <td><strong>${formatNumber(r.test_lift)}x</strong></td>
                <td>${r.holdout_validated ? '<span class="badge badge-emerald">ĐẠT</span>' : '<span class="badge badge-amber">THẤP</span>'}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

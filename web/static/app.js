// Vietnam Flight Intelligence Web Client
document.addEventListener("DOMContentLoaded", () => {
  // Navigation Tabs
  const tabBtns = document.querySelectorAll(".nav-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabContents.forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add("active");

      // Lazy load tab data
      if (btn.dataset.tab === "models-tab") loadModelMetrics();
      if (btn.dataset.tab === "clusters-tab") loadClusterProfiles();
      if (btn.dataset.tab === "rules-tab") loadAssociationRules();
    });
  });

  // Slider value displays
  const sliders = ["precip", "wind", "vis", "load"];
  sliders.forEach((id) => {
    const el = document.getElementById(id);
    const valEl = document.getElementById(id + "-val");
    if (el && valEl) {
      el.addEventListener("input", () => {
        valEl.textContent = el.value;
      });
    }
  });

  // Prediction Form Submission
  const form = document.getElementById("predict-form");
  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const submitBtn = form.querySelector('button[type="submit"]');
      submitBtn.disabled = true;
      submitBtn.textContent = "Đang dự báo rủi ro...";

      const payload = {
        airline_code: document.getElementById("airline").value,
        airport: document.getElementById("airport").value,
        direction: document.getElementById("direction").value,
        route_type: document.getElementById("route-type").value,
        scheduled_time: document.getElementById("sched-time").value,
        flight_date: document.getElementById("flight-date").value,
        precipitation: parseFloat(document.getElementById("precip").value),
        wind_gusts_10m: parseFloat(document.getElementById("wind").value),
        visibility: parseFloat(document.getElementById("vis").value),
        weather_code: parseInt(document.getElementById("weather-code").value),
        airport_load: parseInt(document.getElementById("load").value),
        origin: document.getElementById("origin").value,
        destination: document.getElementById("destination").value,
      };

      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        renderPredictionResult(data);
      } catch (err) {
        alert("Lỗi kết nối máy chủ dự báo: " + err);
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = "Dự Báo Ngay";
      }
    });
  }

  // Load initial demo prediction
  if (form) {
    form.dispatchEvent(new Event("submit"));
  }
});

function renderPredictionResult(data) {
  const resultContainer = document.getElementById("predict-result");
  if (!resultContainer) return;

  const header = resultContainer.querySelector(".result-header");
  header.className = `result-header risk-${data.risk_level.toLowerCase()}`;

  document.getElementById("res-risk-level").textContent = `RỦI RO: ${data.risk_level}`;
  document.getElementById("res-prediction-text").textContent =
    data.prediction === "on_time"
      ? "Khả năng cao ĐÚNG GIỜ"
      : data.prediction === "delayed"
      ? `Dự báo TRỄ khoảng ~${data.estimated_delay_min} phút`
      : "NGUY CƠ HỦY CHUYẾN CAO";

  document.getElementById("res-risk-score").textContent = `${Math.round(data.risk_score * 100)}%`;

  // Probabilities bars
  const p = data.probabilities;
  setBar("prob-ontime", p.on_time);
  setBar("prob-delayed", p.delayed);
  setBar("prob-cancelled", p.cancelled);

  // Cluster profile
  const cp = data.cluster_profile;
  document.getElementById("res-cluster-name").textContent = cp.name || "N/A";
  document.getElementById("res-cluster-rate").textContent = `${cp.delay_rate_pct || 0}% trễ lịch sử`;

  // Rules list
  const rulesList = document.getElementById("res-rules-list");
  rulesList.innerHTML = "";
  if (data.matched_rules && data.matched_rules.length > 0) {
    data.matched_rules.forEach((r) => {
      const div = document.createElement("div");
      div.className = "rule-item";
      div.innerHTML = `
        <div class="rule-ante">${r.antecedent} ➔ <strong>${r.consequent}</strong></div>
        <div class="rule-meta">Độ tin cậy (Confidence): ${r.confidence_pct}% | Lift: ${r.lift}x</div>
      `;
      rulesList.appendChild(div);
    });
  } else {
    rulesList.innerHTML = `<p style="color: var(--text-muted); font-size: 0.85rem;">Không có luật cảnh báo đặc biệt nào qua ngưỡng.</p>`;
  }
}

function setBar(id, val) {
  const pct = Math.round((val || 0) * 100);
  const el = document.getElementById(id);
  const label = document.getElementById(id + "-text");
  if (el) el.style.width = pct + "%";
  if (label) label.textContent = pct + "%";
}

async function loadModelMetrics() {
  const container = document.getElementById("metrics-content");
  if (!container || container.dataset.loaded) return;

  try {
    const res = await fetch("/api/stats");
    const data = await res.json();
    const clf = data.supervised.classification;
    const reg = data.supervised.regression;
    const imps = data.supervised.feature_importances;

    let html = `
      <div class="card">
        <h3 class="card-title">So Sánh Mô Hình Phân Loại (Supervised Benchmark)</h3>
        <table class="data-table">
          <thead>
            <tr>
              <th>Thuật toán</th>
              <th>Độ chính xác (Accuracy)</th>
              <th>F1-Score (Macro)</th>
              <th>Recall (Macro)</th>
              <th>ROC-AUC (ovr)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Logistic Regression</strong></td>
              <td>${(clf.LogisticRegression.accuracy * 100).toFixed(1)}%</td>
              <td>${clf.LogisticRegression.f1_macro}</td>
              <td>${clf.LogisticRegression.recall_macro}</td>
              <td>${clf.LogisticRegression.roc_auc_ovr}</td>
            </tr>
            <tr>
              <td><strong>Random Forest (Balanced)</strong></td>
              <td>${(clf.RandomForest.accuracy * 100).toFixed(1)}%</td>
              <td>${clf.RandomForest.f1_macro}</td>
              <td>${clf.RandomForest.recall_macro}</td>
              <td><strong>${clf.RandomForest.roc_auc_ovr}</strong></td>
            </tr>
            <tr style="background: #f0fdf4;">
              <td><strong>HistGradientBoosting (Best F1)</strong></td>
              <td>${(clf.HistGradientBoosting.accuracy * 100).toFixed(1)}%</td>
              <td><strong>${clf.HistGradientBoosting.f1_macro}</strong></td>
              <td><strong>${clf.HistGradientBoosting.recall_macro}</strong></td>
              <td>${clf.HistGradientBoosting.roc_auc_ovr}</td>
            </tr>
          </tbody>
        </table>
        <p style="margin-top: 1rem; font-size: 0.9rem; color: var(--text-muted);">
          * Đánh giá trên tập kiểm thử độc lập (Test Set) gồm 2,566 chuyến bay chia theo mốc thời gian (Chronological Split).
        </p>
      </div>

      <div class="card">
        <h3 class="card-title">Mô Hình Hồi Quy Dự Đoán Số Phút Trễ (Delay Minutes)</h3>
        <p><strong>Sai số tuyệt đối trung bình (MAE):</strong> ${reg.MAE_minutes} phút</p>
        <p><strong>Sai số căn bậc hai (RMSE):</strong> ${reg.RMSE_minutes} phút</p>
      </div>
    `;

    if (imps && imps.length > 0) {
      html += `
        <div class="card">
          <h3 class="card-title">Top Đặc Trưng Quan Trọng Nhất (Feature Importance)</h3>
          <table class="data-table">
            <thead>
              <tr><th>Hạng</th><th>Đặc trưng</th><th>Tầm quan trọng</th></tr>
            </thead>
            <tbody>
              ${imps.slice(0, 10).map((item, idx) => `
                <tr>
                  <td>#${idx + 1}</td>
                  <td><code>${item.feature}</code></td>
                  <td>${(item.importance * 100).toFixed(2)}%</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      `;
    }

    container.innerHTML = html;
    container.dataset.loaded = "true";
  } catch (err) {
    container.innerHTML = `<p style="color: var(--danger)">Lỗi tải dữ liệu mô hình: ${err}</p>`;
  }
}

async function loadClusterProfiles() {
  const container = document.getElementById("clusters-content");
  if (!container || container.dataset.loaded) return;

  try {
    const res = await fetch("/api/clusters");
    const data = await res.json();

    let html = `
      <div class="card">
        <h3 class="card-title">Hồ Sơ Cụm Vận Hành Không Giám Sát (K-Means Clustering, k=4)</h3>
        <table class="data-table">
          <thead>
            <tr>
              <th>Cụm</th>
              <th>Tên nhóm vận hành</th>
              <th>Số chuyến</th>
              <th>Tỷ lệ trễ (%)</th>
              <th>Tỷ lệ hủy (%)</th>
              <th>Trễ TB (phút)</th>
              <th>Mức rủi ro</th>
            </tr>
          </thead>
          <tbody>
            ${Object.values(data.profiles).map((c) => `
              <tr>
                <td><strong>#${c.cluster_id}</strong></td>
                <td>${c.name}</td>
                <td>${c.flight_count.toLocaleString()} (${c.percentage}%)</td>
                <td>${c.delay_rate_pct}%</td>
                <td>${c.cancellation_rate_pct}%</td>
                <td>${c.avg_delay_minutes}</td>
                <td><span class="risk-badge" style="background: ${c.risk_tier === 'HIGH' ? 'var(--danger)' : c.risk_tier === 'MEDIUM' ? 'var(--warning)' : 'var(--success)'}; color: white;">${c.risk_tier}</span></td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
      <div class="card">
        <h3 class="card-title">Phát Hiện Dị Thường Không Giám Sát (Isolation Forest)</h3>
        <p>Thuật toán đã phát hiện <strong>${data.anomalies.anomaly_count.toLocaleString()} chuyến bay có đặc tính bất thường (${data.anomalies.anomaly_percentage}%)</strong> về thời tiết cực đoan, quá tải đường băng hoặc lịch trình sai lệch.</p>
      </div>
    `;

    container.innerHTML = html;
    container.dataset.loaded = "true";
  } catch (err) {
    container.innerHTML = `<p style="color: var(--danger)">Lỗi tải dữ liệu cụm: ${err}</p>`;
  }
}

async function loadAssociationRules() {
  const container = document.getElementById("rules-content");
  if (!container || container.dataset.loaded) return;

  try {
    const res = await fetch("/api/rules");
    const data = await res.json();

    let html = `
      <div class="card">
        <h3 class="card-title">Luật Kết Hợp Khai Phá Được (FP-Growth Association Rules)</h3>
        <table class="data-table">
          <thead>
            <tr>
              <th>Tiền đề (Điều kiện bối cảnh)</th>
              <th>Hệ quả</th>
              <th>Support (%)</th>
              <th>Confidence (%)</th>
              <th>Lift</th>
            </tr>
          </thead>
          <tbody>
            ${data.slice(0, 15).map((r) => `
              <tr>
                <td><strong>${r.Antecedent}</strong></td>
                <td><span class="risk-badge" style="background: var(--warning); color: white;">${r.Consequent}</span></td>
                <td>${r["Support (%)"]}%</td>
                <td>${r["Confidence (%)"]}%</td>
                <td><strong>${r.Lift}x</strong></td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    `;

    container.innerHTML = html;
    container.dataset.loaded = "true";
  } catch (err) {
    container.innerHTML = `<p style="color: var(--danger)">Lỗi tải tập luật: ${err}</p>`;
  }
}


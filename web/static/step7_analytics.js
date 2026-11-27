/* ==========================================================================
   [STEP 7/7 - 7-STEP CRISP-DM FRONTEND ANALYTICS EXTENSIONS]
   FlightRisk VN — Schema Evolution, Correlation Matrix, Backtest, Silhouette & PSI Drift
   Zero innerHTML with untrusted values (XSS-safe DOM builders only)
   ========================================================================== */
(function () {
  "use strict";

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function makeKpiCard(label, value, sub) {
    const card = el("div", "kpi-card");
    card.appendChild(el("div", "kpi-label", label));
    card.appendChild(el("div", "kpi-value", value));
    if (sub) card.appendChild(el("div", "kpi-sub", sub));
    return card;
  }

  function makeTable(headers, rows) {
    const wrap = el("div", "table-scroll");
    const table = el("table", "data-table");
    const thead = el("thead");
    const trH = el("tr");
    headers.forEach((h) => trH.appendChild(el("th", "", h)));
    thead.appendChild(trH);
    table.appendChild(thead);

    const tbody = el("tbody");
    rows.forEach((row) => {
      const tr = el("tr");
      row.forEach((cell) => {
        const td = el("td");
        if (cell instanceof Node) {
          td.appendChild(cell);
        } else {
          td.textContent = String(cell);
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  function heatCellStyle(val) {
    const v = Math.max(-1, Math.min(1, Number(val) || 0));
    const intensity = Math.min(0.88, Math.abs(v) * 0.9);
    if (v >= 0) {
      return {
        bg: `rgba(13, 92, 88, ${intensity.toFixed(2)})`,
        color: Math.abs(v) >= 0.45 ? "#ffffff" : "#0f172a",
      };
    }
    return {
      bg: `rgba(225, 29, 72, ${intensity.toFixed(2)})`,
      color: Math.abs(v) >= 0.45 ? "#ffffff" : "#0f172a",
    };
  }

  function buildHeatmap(cols, matrix) {
    const wrap = el("div", "heatmap-wrap");
    const table = el("table", "heatmap-table");
    const thead = el("thead");
    const hr = el("tr");
    hr.appendChild(el("th", "row-hdr", "Đặc trưng"));
    cols.forEach((c) => hr.appendChild(el("th", "col-hdr", c.replace(/_10m$/, "").replace("weather_hazard_score", "hazard").replace("is_golden_triangle", "golden_tri").replace("is_weekend_flag", "weekend"))));
    thead.appendChild(hr);
    table.appendChild(thead);

    const tbody = el("tbody");
    cols.forEach((rowName, i) => {
      const tr = el("tr");
      tr.appendChild(el("th", "row-hdr", rowName));
      const rowVals = matrix[i] || [];
      cols.forEach((_, j) => {
        const val = rowVals[j] ?? 0;
        const td = el("td", "", Number(val).toFixed(2));
        const st = heatCellStyle(val);
        td.style.backgroundColor = st.bg;
        td.style.color = st.color;
        td.title = `${rowName} vs ${cols[j]}: ${Number(val).toFixed(4)}`;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  async function fetchJsonPanel(container, url, builder) {
    if (!container || container.dataset.loaded === "true" || container.dataset.loading === "true") return;
    container.dataset.loading = "true";
    container.replaceChildren(el("p", "card-subtitle", "Đang tải dữ liệu thực đo từ server..."));
    try {
      const res = await fetch(url, { headers: { Accept: "application/json" } });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      container.replaceChildren();
      builder(data, container);
      container.dataset.loaded = "true";
    } catch (err) {
      const box = el("div", "app-box limit");
      box.setAttribute("role", "alert");
      box.appendChild(el("strong", "", `Không tải được ${url}: `));
      box.appendChild(el("span", "", err.message || String(err)));
      const btn = el("button", "btn-secondary", "Thử tải lại");
      btn.type = "button";
      btn.style.marginLeft = "0.75rem";
      btn.addEventListener("click", () => {
        delete container.dataset.loading;
        fetchJsonPanel(container, url, builder);
      });
      box.appendChild(btn);
      container.replaceChildren(box);
    } finally {
      delete container.dataset.loading;
    }
  }

  /* --------------------------------------------------------------------------
     STEP 2-4: EDA & Schema Evolution Matrix Panel
     -------------------------------------------------------------------------- */
  function renderEdaPanel(data, container) {
    const raw = data.raw_provenance_audit || {};
    const schema = data.schema_evolution || {};
    const corr = data.correlation_and_association || {};
    const eda = data.eda || {};
    const dist = eda.delay_min_distribution || {};
    const status = eda.status_distribution || {};

    const stack = el("div", "analytics-stack");

    // KPI strip
    const kpis = el("div", "kpi-strip");
    kpis.appendChild(makeKpiCard("Bước 1–2 · Bronze Mixed", `${raw.raw_file_count || 11} files · ${(raw.raw_mixed_rows || 0).toLocaleString()} dòng`, `${raw.raw_mixed_columns || 58} cột -> 12 FIELDS (${(raw.standardized_snapshot_rows || 0).toLocaleString()} chặng)`));
    kpis.appendChild(makeKpiCard("Bước 2–3 · Silver & Gold Causal", `${(eda.total_flights || 0).toLocaleString()} chặng bay`, `${eda.date_range ? `${eda.date_range.min} -> ${eda.date_range.max}` : "T-3h Causal Features"}`));
    kpis.appendChild(makeKpiCard("Phân phối 3 Lớp (ICAO)", `${status.delayed?.pct ?? 0}% Trễ · ${status.cancelled?.pct ?? 0}% Hủy`, `Đúng giờ: ${status.on_time?.pct ?? 0}% (${(status.on_time?.count || 0).toLocaleString()} chuyến)`));
    kpis.appendChild(makeKpiCard("Phân phối delay_min", `TB ${dist.mean ?? 0}m · TV ${dist.median ?? 0}m`, `Q75=${dist.q75 ?? 0}m · P95=${dist.p95 ?? 0}m · Max=${dist.max ?? 0}m`));
    stack.appendChild(kpis);

    // 5-Stage Schema Lineage Matrix
    const cSchema = el("div", "card");
    cSchema.appendChild(el("h3", "card-title", "Ma Trận Tiến Hóa Schema 5 Tầng (Bronze -> Silver -> Gold -> X)"));
    cSchema.appendChild(el("p", "card-subtitle", "Theo dõi biến đổi số cột/dòng và loại bỏ các cột hậu kiểm rò rỉ tương lai."));
    const stageRows = (schema.schema_stages || []).map((s) => [
      s.stage,
      String(s.columns),
      Number(s.rows || 0).toLocaleString(),
      s.why_changed,
      s.how_transformed,
    ]);
    cSchema.appendChild(makeTable(["Tầng Schema", "Số cột", "Số dòng", "Vấn đề tầng trước (TẠI SAO)", "Biến đổi áp dụng (NHƯ THẾ NÀO)"], stageRows));
    stack.appendChild(cSchema);

    // Feature derivation empirical justification
    const cDeriv = el("div", "card");
    cDeriv.appendChild(el("h3", "card-title", "Biện Minh Tạo Biến Mới Bằng Tương Quan Thực Nghiệm"));
    const derivRows = (schema.feature_derivations || []).map((d) => [
      d.new_feature,
      Array.isArray(d.source_columns) ? d.source_columns.join(", ") : String(d.source_columns || ""),
      d.empirical_reason,
      d.formula_summary,
    ]);
    cDeriv.appendChild(makeTable(["Nhóm biến mới / Bị chặn", "Cột nguồn gốc", "Bằng chứng tương quan & EDA", "Quy tắc / Công thức"], derivRows));
    stack.appendChild(cDeriv);

    // Interactive Heatmap + Cramér's V
    const cCorr = el("div", "card");
    const hdrRow = el("div");
    hdrRow.style.display = "flex";
    hdrRow.style.justifyContent = "space-between";
    hdrRow.style.alignItems = "center";
    hdrRow.style.flexWrap = "wrap";
    hdrRow.style.gap = "0.75rem";
    hdrRow.style.marginBottom = "0.75rem";
    const hTitleWrap = el("div");
    hTitleWrap.appendChild(el("h3", "card-title", "Ma Trận Tương Quan (Pearson & Spearman Heatmap)"));
    hTitleWrap.appendChild(el("p", "card-subtitle", `Bật chuyển đổi để kiểm tra tương quan tuyến tính (Pearson r) và đơn điệu phi tuyến (Spearman rho) trên ${(eda.total_flights || 0).toLocaleString()} chặng bay.`));
    hdrRow.appendChild(hTitleWrap);

    const pills = el("div", "toggle-pills");
    const btnPearson = el("button", "toggle-pill active", "Pearson (r)");
    const btnSpearman = el("button", "toggle-pill", "Spearman (rho)");
    btnPearson.type = "button";
    btnSpearman.type = "button";
    pills.appendChild(btnPearson);
    pills.appendChild(btnSpearman);
    hdrRow.appendChild(pills);
    cCorr.appendChild(hdrRow);

    const heatSlot = el("div");
    const hCols = corr.heatmap_columns || [];
    heatSlot.appendChild(buildHeatmap(hCols, corr.pearson_heatmap || []));
    btnPearson.addEventListener("click", () => {
      btnPearson.classList.add("active");
      btnSpearman.classList.remove("active");
      heatSlot.replaceChildren(buildHeatmap(hCols, corr.pearson_heatmap || []));
    });
    btnSpearman.addEventListener("click", () => {
      btnSpearman.classList.add("active");
      btnPearson.classList.remove("active");
      heatSlot.replaceChildren(buildHeatmap(hCols, corr.spearman_heatmap || []));
    });
    cCorr.appendChild(heatSlot);
    stack.appendChild(cCorr);

    // Grid: Cramér's V + Histogram delay_min
    const g2 = el("div", "analytics-grid-2");

    const cCramer = el("div", "card");
    cCramer.appendChild(el("h3", "card-title", "Liên Kết Biến Phân Loại — Cramér's V"));
    const crRows = (corr.categorical_cramers_v || []).map((r) => {
      const bar = el("div", "bar-track");
      const fill = el("div", "bar-fill");
      fill.style.width = `${Math.min(100, Math.round((r.cramers_v_vs_delayed || 0) * 400))}%`;
      bar.appendChild(fill);
      return [r.feature, String(r.n_unique), String(r.cramers_v_vs_delayed), String(r.cramers_v_vs_status), bar];
    });
    cCramer.appendChild(makeTable(["Thuộc tính phân loại", "Giá trị", "V (is_delayed)", "V (3 lớp)", "Mức độ"], crRows));
    g2.appendChild(cCramer);

    const cHist = el("div", "card");
    cHist.appendChild(el("h3", "card-title", `Phân Phối Số Phút Trễ (delay_min — ${(dist.count || 0).toLocaleString()} chuyến hoàn tất)`));
    const histRows = (eda.delay_min_histogram || []).map((b) => {
      const bar = el("div", "bar-track");
      const fill = el("div", "bar-fill" + (b.bucket.includes("60") || b.bucket.includes(">=120") ? " bar-fill-danger" : b.bucket.includes("15-") || b.bucket.includes("30-") ? " bar-fill-warn" : ""));
      fill.style.width = `${Math.min(100, Math.round(b.pct * 2))}%`;
      bar.appendChild(fill);
      return [b.bucket, Number(b.count).toLocaleString(), `${b.pct}%`, bar];
    });
    cHist.appendChild(makeTable(["Khoảng trễ", "Số chuyến", "Tỷ lệ", "Phân bố"], histRows));
    g2.appendChild(cHist);
    stack.appendChild(g2);

    // EDA Breakdowns: Airports & Airlines & Slots & Weather
    const gBreak1 = el("div", "analytics-grid-2");
    const cAp = el("div", "card");
    const minor = eda.minor_airports_summary || {};
    cAp.appendChild(el("h3", "card-title", "EDA Theo 5 Sân Bay Chủ Lực"));
    cAp.appendChild(el("p", "card-subtitle", `${minor.n_airports ?? 0} sân bay địa phương còn lại chiếm ${minor.count ?? 0} chuyến (${minor.share_pct ?? 0}%).`));
    cAp.appendChild(makeTable(
      ["Sân bay", "Chuyến", "Trễ (%)", "Hủy (%)", "TB Trễ"],
      (eda.by_major_airport || []).map((a) => [a.key, Number(a.count).toLocaleString(), `${a.delayed_pct}%`, `${a.cancelled_pct}%`, `${a.avg_delay_min}m`])
    ));
    gBreak1.appendChild(cAp);

    const cAl = el("div", "card");
    cAl.appendChild(el("h3", "card-title", "EDA Theo 7 Hãng Hàng Không Việt Nam"));
    cAl.appendChild(makeTable(
      ["Hãng", "Chuyến", "Trễ (%)", "Hủy (%)", "TB Trễ"],
      (eda.by_airline || []).map((a) => [a.key, Number(a.count).toLocaleString(), `${a.delayed_pct}%`, `${a.cancelled_pct}%`, `${a.avg_delay_min}m`])
    ));
    gBreak1.appendChild(cAl);
    stack.appendChild(gBreak1);

    const gBreak2 = el("div", "analytics-grid-2");
    const cSlot = el("div", "card");
    cSlot.appendChild(el("h3", "card-title", "EDA Theo Khung Giờ (Time Slot)"));
    cSlot.appendChild(makeTable(
      ["Khung giờ", "Chuyến", "Trễ (%)", "TB Trễ"],
      (eda.by_time_slot || []).map((s) => [s.key, Number(s.count).toLocaleString(), `${s.delayed_pct}%`, `${s.avg_delay_min}m`])
    ));
    gBreak2.appendChild(cSlot);

    const cWx = el("div", "card");
    cWx.appendChild(el("h3", "card-title", "EDA Theo Chế Độ Thời Tiết (Open-Meteo)"));
    cWx.appendChild(makeTable(
      ["Điều kiện thời tiết", "Chuyến", "Trễ (%)", "Hủy (%)", "TB Trễ"],
      (eda.by_weather_regime || []).map((w) => [w.key, Number(w.count).toLocaleString(), `${w.delayed_pct}%`, `${w.cancelled_pct}%`, `${w.avg_delay_min}m`])
    ));
    gBreak2.appendChild(cWx);
    stack.appendChild(gBreak2);

    container.appendChild(stack);
  }

  /* --------------------------------------------------------------------------
     STEP 6: Backtest by Airport/Airline + Permutation + Practical Applicability
     -------------------------------------------------------------------------- */
  function renderBacktestPanel(data, container) {
    const stack = el("div", "analytics-stack");
    const perm = data.permutation_test || {};
    const statCmp = data.statistical_comparisons || {};
    const vsHeur = statCmp.best_vs_heuristic || {};
    const mcn = vsHeur.mcnemar || {};
    const boot = vsHeur.bootstrap_f1_macro || {};
    const app = data.practical_applicability || {};

    const cPerm = el("div", "card");
    cPerm.appendChild(el("h3", "card-title", `Bước 5 · Kiểm Định Thống Kê & Backtest Phân Khúc (${data.classifier_evaluated || "XGBoost"} + ${data.regressor_evaluated || "CatBoost_MAE"})`));
    const kpis = el("div", "kpi-strip");
    kpis.appendChild(makeKpiCard("Permutation Test (200 vòng)", `p = ${perm.p_value ?? "--"}`, `BalAcc ${perm.model_score ?? "--"}% vs ngẫu nhiên ${perm.chance_level_mean ?? "--"}% ± ${perm.chance_level_std ?? "--"}%`));
    kpis.appendChild(makeKpiCard("McNemar & Bootstrap vs Heuristic", `p = ${mcn.p_value ?? "--"}`, `ΔF1 = ${boot.mean_diff_b_minus_a ?? "--"} (95% CI: ${(boot.ci_95 || []).join(", ")})`));
    kpis.appendChild(makeKpiCard("Causal Cutoff & Minor Airports", `${data.cutoff_date || "--"}`, `Sân bay nhỏ Test: ${data.minor_airports_test_count ?? 0} chuyến`));
    cPerm.appendChild(kpis);
    stack.appendChild(cPerm);

    const g2 = el("div", "analytics-grid-2");
    const cAp = el("div", "card");
    cAp.appendChild(el("h3", "card-title", "Backtest Holdout Test Theo 5 Sân Bay Trọng Điểm"));
    const apRows = Object.entries(data.backtest_by_airport || {}).map(([ap, v]) => [
      ap, String(v.count), `${v.accuracy}%`, `${v.balanced_accuracy}%`, `${v.macro_f1}%`, `${v.mae_model ?? "--"}m`
    ]);
    cAp.appendChild(makeTable(["Sân bay", "Test n", "Accuracy", "BalAcc", "Macro-F1", "MAE"], apRows));
    g2.appendChild(cAp);

    const cAl = el("div", "card");
    cAl.appendChild(el("h3", "card-title", "Backtest Holdout Test Theo Hãng Hàng Không"));
    const alRows = Object.entries(data.backtest_by_airline || {}).map(([al, v]) => [
      al, String(v.count) + (v.small_sample_warning ? " (n<30)" : ""), `${v.accuracy}%`, `${v.balanced_accuracy}%`, `${v.macro_f1}%`
    ]);
    cAl.appendChild(makeTable(["Hãng", "Test n", "Accuracy", "BalAcc", "Macro-F1"], alRows));
    g2.appendChild(cAl);
    stack.appendChild(g2);

    const cApp = el("div", "card");
    cApp.appendChild(el("h3", "card-title", "Đánh Giá Khả Năng Ứng Dụng Thực Tế & Giới Hạn Khoa Học"));
    const gridApp = el("div", "applicability-grid");

    const boxReady = el("div", "app-box ready");
    boxReady.appendChild(el("strong", "", "Sẵn sàng triển khai ngay:"));
    const ulReady = el("ul");
    (app.ready_for || []).forEach((item) => ulReady.appendChild(el("li", "", item)));
    boxReady.appendChild(ulReady);
    gridApp.appendChild(boxReady);

    const boxLimit = el("div", "app-box limit");
    boxLimit.appendChild(el("strong", "", "Giới hạn khách quan & Khoảng trống dữ liệu:"));
    const ulLimit = el("ul");
    (app.limitations_and_gaps || []).forEach((item) => ulLimit.appendChild(el("li", "", item)));
    boxLimit.appendChild(ulLimit);
    gridApp.appendChild(boxLimit);

    cApp.appendChild(gridApp);
    stack.appendChild(cApp);
    container.appendChild(stack);
  }

  /* --------------------------------------------------------------------------
     STEP 4: K-Means Sweep (Silhouette/DBI/CH), PCA 3D & Anomaly Breakdown
     -------------------------------------------------------------------------- */
  function renderSilhouetteComparison(data, container) {
    if (!container || container.dataset.loaded === "true") return;
    const km = data.kmeans || {};
    const pca = data.pca || {};
    const iso = data.anomalies || {};
    const kSweep = km.k_selection || [];

    const stack = el("div", "analytics-stack");
    const cSil = el("div", "card");
    cSil.appendChild(el("h3", "card-title", `Thẩm Định Số Cụm (k=2..8) & Độ Ổn Định Seed ARI = ${km.seed_stability?.ari_mean ?? "--"}`));
    cSil.appendChild(el("p", "card-subtitle", km.suitability_note || "K-Means dùng cho phân đoạn bối cảnh vận hành."));

    const rows = kSweep.map((r) => [
      `k = ${r.k}${r.k === (km.k_selected || 4) ? " (Đang chọn)" : ""}`,
      String(r.silhouette),
      String(r.davies_bouldin ?? "-"),
      Number(r.calinski_harabasz || 0).toLocaleString(),
      Number(r.inertia || 0).toLocaleString(),
    ]);
    cSil.appendChild(makeTable(["Số cụm k", "Silhouette (cao tốt)", "Davies-Bouldin (thấp tốt)", "Calinski-Harabasz", "Inertia (Elbow)"], rows));
    stack.appendChild(cSil);

    const g2 = el("div", "analytics-grid-2");
    const cPca = el("div", "card");
    cPca.appendChild(el("h3", "card-title", `PCA ${pca.n_components ?? 3} Thành Phần Chính (Tổng phương sai: ${pca.cumulative_variance_pct ?? 0}%)`));
    const pcaRows = (pca.components || []).map((c) => {
      const top = (c.top_features || []).slice(0, 3).map((x) => `${x.feature} (${x.loading})`).join(", ");
      return [c.component, `${c.explained_variance_pct}%`, top || "-"];
    });
    cPca.appendChild(makeTable(["Thành phần", "Phương sai", "Biến đóng góp chính (Top Loadings)"], pcaRows));
    g2.appendChild(cPca);

    const cIso = el("div", "card");
    cIso.appendChild(el("h3", "card-title", `Isolation Forest Theo Sân Bay (${iso.anomaly_count ?? 0} ca dị thường)`));
    const apRows = (iso.by_airport || []).slice(0, 6).map((f) => [
      f.airport,
      String(f.total_flights),
      `${f.anomaly_count} (${f.anomaly_rate_pct}%)`,
      f.disruption_rate_in_anomalies_pct !== null ? `${f.disruption_rate_in_anomalies_pct}%` : "-",
    ]);
    cIso.appendChild(makeTable(["Sân bay", "Tổng chuyến", "Số ca dị thường", "Gián đoạn trong dị thường"], apRows));
    g2.appendChild(cIso);
    stack.appendChild(g2);

    container.replaceChildren(stack);
    container.dataset.loaded = "true";
  }

  /* --------------------------------------------------------------------------
     STEP 6-7: 4-Tier Data Drift (PSI + KS) & Lock Manifest SHA-256 Panel
     -------------------------------------------------------------------------- */
  function renderDriftPanel(drift, manifest, container) {
    const stack = el("div", "analytics-stack");
    const psi = drift.feature_psi || {};
    const lbl = drift.label_drift || {};

    const kpis = el("div", "kpi-strip");
    kpis.appendChild(makeKpiCard("Tập tham chiếu vs Hiện tại", `${Number(drift.train_rows || 0).toLocaleString()} vs ${Number(drift.test_rows || 0).toLocaleString()}`, `Overall: ${drift.overall_status || "OK"} | Retrain: ${drift.retrain_recommended ? "KHUYẾN NGHỊ" : "Chưa cần"}`));
    const dis = lbl.disrupted || {};
    kpis.appendChild(makeKpiCard("Concept Drift (Disrupted)", `${dis.train_pct ?? 0}% -> ${dis.test_pct ?? 0}%`, `Biến động: ${dis.diff_pct_points ?? 0} điểm %`));
    kpis.appendChild(makeKpiCard("Feature Drift (PSI >= 0.10)", `${(drift.drifted_features || []).length} biến cảnh báo`, `High >=0.25: ${(drift.high_drift_features || []).join(", ") || "Không"}`));
    kpis.appendChild(makeKpiCard("Governance & Approval Gate", manifest?.pipeline_lifecycle?.phase_3_processing?.status || "LOCKED", `Model Registry: ${Object.keys(manifest?.model_registry?.production || {}).length} Production`));
    stack.appendChild(kpis);

    const cPsi = el("div", "card");
    cPsi.appendChild(el("h3", "card-title", "Giám Sát Data Drift (PSI + Kiểm Định 2-Sample KS)"));
    cPsi.appendChild(el("p", "card-subtitle", drift.retrain_policy || "Cảnh báo PSI >= 0.10 hoặc KS p < 0.05; kích hoạt retrain khi PSI >= 0.25."));
    const psiRows = Object.entries(psi).map(([feat, info]) => {
      const val = Number(info.psi || 0);
      const badgeCls = val >= 0.25 ? "tag-danger" : val >= 0.1 ? "tag-warn" : "tag-stable";
      const badge = el("span", `tag-badge ${badgeCls}`, info.status);
      const bar = el("div", "bar-track");
      const fill = el("div", "bar-fill" + (val >= 0.25 ? " bar-fill-danger" : val >= 0.1 ? " bar-fill-warn" : ""));
      fill.style.width = `${Math.min(100, Math.max(4, Math.round((Math.min(val, 1.0)) * 100)))}%`;
      bar.appendChild(fill);
      return [feat, val.toFixed(4), `KS=${info.ks_statistic ?? "-"} (p=${info.ks_p_value ?? "-"})`, badge, bar];
    });
    cPsi.appendChild(makeTable(["Đặc trưng số", "Chỉ số PSI", "Kolmogorov-Smirnov", "Trạng thái", "Mức độ"], psiRows));
    stack.appendChild(cPsi);

    const g2 = el("div", "analytics-grid-2");
    const cLbl = el("div", "card");
    cLbl.appendChild(el("h3", "card-title", "Label Drift (Dịch Chuyển Phân Phối 3 Lớp ICAO)"));
    const lblRows = Object.entries(lbl).map(([k, v]) => [
      k, `${v.train_pct}%`, `${v.test_pct}%`, `${v.diff_pct_points > 0 ? "+" : ""}${v.diff_pct_points} điểm %`
    ]);
    cLbl.appendChild(makeTable(["Nhãn trạng thái", "Ref (Train+Val)", "Current (Holdout Test)", "Biến động"], lblRows));
    g2.appendChild(cLbl);

    const cLock = el("div", "card");
    cLock.appendChild(el("h3", "card-title", "Hồ Sơ Niêm Phong SHA-256 (LOCK_pipeline_manifest.json)"));
    const checks = Object.entries((manifest && manifest.artifact_checksums_sha256) || {}).map(([name, info]) => [
      name,
      info.path,
      `${(Number(info.size_bytes || 0) / 1024).toFixed(1)} KB`,
      String(info.sha256 || "").slice(0, 16) + "...",
    ]);
    cLock.appendChild(makeTable(["Artifact", "Đường dẫn", "Dung lượng", "SHA-256 Prefix"], checks));
    g2.appendChild(cLock);
    stack.appendChild(g2);

    container.appendChild(stack);
  }

  /* --------------------------------------------------------------------------
     Wire Tab Clicks Without Breaking Lazy Playwright Response Listeners
     -------------------------------------------------------------------------- */
  document.addEventListener("DOMContentLoaded", () => {
    const edaBtn = document.querySelector('[data-tab="eda-tab"]');
    if (edaBtn) {
      edaBtn.addEventListener("click", () => {
        fetchJsonPanel(document.getElementById("eda-content"), "/api/eda", renderEdaPanel);
      });
    }

    const modelsBtn = document.querySelector('[data-tab="models-tab"]');
    if (modelsBtn) {
      modelsBtn.addEventListener("click", () => {
        fetchJsonPanel(document.getElementById("backtest-content"), "/api/backtest", renderBacktestPanel);
      });
    }

    const clustersBtn = document.querySelector('[data-tab="clusters-tab"]');
    if (clustersBtn) {
      clustersBtn.addEventListener("click", () => {
        fetchJsonPanel(document.getElementById("clusters-silhouette-panel"), "/api/clusters", renderSilhouetteComparison);
      });
    }

    const driftBtn = document.querySelector('[data-tab="drift-tab"]');
    if (driftBtn) {
      driftBtn.addEventListener("click", () => {
        const container = document.getElementById("drift-content");
        if (!container || container.dataset.loaded === "true" || container.dataset.loading === "true") return;
        container.dataset.loading = "true";
        container.replaceChildren(el("p", "card-subtitle", "Đang tải báo cáo PSI Drift và Checksum SHA-256..."));
        Promise.all([
          fetch("/api/drift").then((r) => r.json().then((d) => ({ ok: r.ok, d }))),
          fetch("/api/manifest").then((r) => r.json().then((d) => ({ ok: r.ok, d }))).catch(() => ({ ok: false, d: {} })),
        ]).then(([dr, mf]) => {
          if (!dr.ok) throw new Error(dr.d.error || "Lỗi tải /api/drift");
          container.replaceChildren();
          renderDriftPanel(dr.d, mf.ok ? mf.d : {}, container);
          container.dataset.loaded = "true";
        }).catch((err) => {
          const box = el("div", "app-box limit");
          box.setAttribute("role", "alert");
          box.textContent = `Không tải được dữ liệu PSI Drift: ${err.message}`;
          container.replaceChildren(box);
        }).finally(() => {
          delete container.dataset.loading;
        });
      });
    }
  });
})();

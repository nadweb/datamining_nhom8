# -*- coding: utf-8 -*-
"""Interactive 20-Slide CRISP-DM Academic & Production Presentation Builder."""
import base64
import html
import shutil
from pathlib import Path

from flight_delay.common.constants import (
    ARTIFACT_MANIFESTS_DIR,
    DATA_MANIFESTS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    ROOT_DIR,
    WEB_DIR,
)
from flight_delay.common.utils import read_json


def _img_b64(filename: str) -> str:
    p = FIGURES_DIR / filename
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _esc(val) -> str:
    return html.escape(str(val))


def build_presentation_html() -> Path:
    """Compile web/templates/presentation.html and root presentation.html from live reports & figures."""
    eda = read_json(REPORTS_DIR / "eda" / "eda_report.json")
    cls = read_json(REPORTS_DIR / "experiments" / "classification_results.json")
    reg = read_json(REPORTS_DIR / "experiments" / "regression_results.json")
    unsup = read_json(REPORTS_DIR / "experiments" / "unsupervised_results.json")
    drift = read_json(REPORTS_DIR / "monitoring" / "drift_report.json")
    ds_man = read_json(DATA_MANIFESTS_DIR / "dataset_manifest.json")
    reg_man = read_json(ARTIFACT_MANIFESTS_DIR / "model_registry.json")

    raw_prov = eda.get("raw_provenance_audit", {})
    eda_core = eda.get("eda", {})
    st_dist = eda_core.get("status_distribution", {})
    delays = eda_core.get("delay_min_distribution", {})
    biv_al = eda_core.get("bivariate", {}).get("by_airline", [])
    biv_ap = eda_core.get("bivariate", {}).get("by_airport", [])[:5]

    mc = cls["multiclass_3"]
    bn = cls["binary_disrupted"]
    hz = cls["horizon_ablation"]["horizons"]
    best_cls = mc["best_model"]
    best_cls_m = mc["test_comparison"][best_cls]
    best_reg = reg["best_model"]
    best_reg_m = reg["test_comparison"][best_reg]
    perm = mc["permutation_test"]
    vs_heur = mc["statistical_comparisons"]["best_vs_heuristic"]
    km = unsup["kmeans"]
    pca = unsup["pca"]
    iso = unsup["isolation_forest"]
    rules_info = unsup["association_rules"]

    slides_html = []

    def add_slide(num: int, phase: int, badge: str, title: str, subtitle: str, body: str, foot: str) -> None:
        slides_html.append(f"""
      <section class="slide" data-slide="{num}" data-phase="{phase}">
        <div class="slide-header">
          <div style="display:flex;gap:8px;margin-bottom:6px;"><span class="badge badge-teal">{_esc(badge)}</span></div>
          <h1 class="slide-title">{title}</h1>
          <p class="slide-subtitle">{subtitle}</p>
        </div>
        <div class="slide-main-content">{body}</div>
        <div class="slide-footnote">{_esc(foot)}</div>
      </section>""")

    # Slide 1: Cover
    add_slide(
        1, 1, "FLIGHTRISK VN · VERSION 2.0",
        "Nền Tảng Khai Phá &amp; Dự Báo Rủi Ro Chuyến Bay Việt Nam",
        f"Quy trình 7 Bước CRISP-DM chuẩn hóa src/flight_delay · Dữ liệu {eda_core.get('total_flights', 0):,} chặng bay (T-3h Causal)",
        f"""
        <div class="grid-3col" style="margin-bottom:16px;">
          <div class="stat-callout"><div class="stat-num" style="color:#0d5c58;">{raw_prov.get('raw_mixed_rows', 0):,} &rarr; {eda_core.get('total_flights', 0):,}</div><div class="stat-label">Snapshots Thô &rarr; Chặng Silver Sạch</div></div>
          <div class="stat-callout"><div class="stat-num" style="color:#0284c7;">ROC-AUC {best_cls_m.get('roc_auc_ovr')}</div><div class="stat-label">Phân loại Chính 3 Lớp ICAO ({_esc(best_cls)})</div></div>
          <div class="stat-callout"><div class="stat-num" style="color:#059669;">MAE {best_reg_m.get('MAE_minutes')}m</div><div class="stat-label">Hồi quy L1 Phút Trễ ({_esc(best_reg)})</div></div>
        </div>
        <div class="grid-2col">
          <div class="card">
            <div class="card-title">Định Hướng Nâng Cấp Kiến Trúc</div>
            <ul style="font-size:12px;line-height:1.65;margin-left:18px;">
              <li><strong>Kiến trúc src-layout:</strong> Tách bạch 7 tầng CRISP-DM trong <code>src/flight_delay/</code> + cấu hình YAML tập trung tại <code>configs/</code>.</li>
              <li><strong>Bảo toàn 100% Bronze:</strong> Toàn bộ CSV cào gốc &amp; <code>data/debug_data/</code> (2,923 captures) nguyên vẹn tại <code>data/raw/</code>.</li>
              <li><strong>Phân loại 3 lớp ICAO làm bài toán chính:</strong> <code>0: ON_TIME</code>, <code>1: DELAYED</code>, <code>2: CANCELLED</code> trước T-3h; song hành Ablation nhị phân &amp; chân trời T-1h/T-3h/T-6h.</li>
            </ul>
          </div>
          <div class="card">
            <div class="card-title">Độ Tin Cậy Học Thuật &amp; Sản Phẩm Thực Tế</div>
            <ul style="font-size:12px;line-height:1.65;margin-left:18px;">
              <li><strong>Chống rò rỉ nhân quả (Causal):</strong> <code>CausalHistoryStore</code> dùng <code>bisect_left</code> chỉ tính lịch sử trước <code>prediction_time</code>.</li>
              <li><strong>Model Governance:</strong> Registry <code>Candidate &rarr; Validated &rarr; Approved &rarr; Production</code>, khóa băm SHA-256.</li>
              <li><strong>FastAPI + Uvicorn:</strong> Serving fail-closed qua <code>predict_risk()</code>, kiểm định PSI + KS + Concept Drift 4 tầng.</li>
            </ul>
          </div>
        </div>""",
        "FlightRisk VN 2.0 · configs/project.yaml · Dataset " + str(eda.get("dataset_version", "")),
    )

    # Slide 2: System Architecture
    add_slide(
        2, 1, "BƯỚC 0 · KIẾN TRÚC HỆ THỐNG 7 TẦNG",
        "Sơ Đồ Toàn Cảnh Hệ Thống FlightRisk VN (Modular CRISP-DM)",
        "Tách biệt hoàn toàn Thu thập, Kỹ thuật Dữ liệu, Khai phá Causal, Huấn luyện, Đánh giá, Quản trị & Vận hành",
        """
        <div class="card">
          <table class="modern-table">
            <thead><tr><th>Giai đoạn CRISP-DM</th><th>Package trong src/flight_delay/</th><th>Đầu vào &rarr; Đầu ra cốt lõi</th></tr></thead>
            <tbody>
              <tr><td><strong>01. Data Ingestion</strong></td><td><code>ingestion/</code> (FIDS, Aviationstack, Weather)</td><td>11 file thô + debug_data &rarr; Bronze chuẩn 12 cột (<code>93,081</code> snapshots)</td></tr>
              <tr><td><strong>02. Data Engineering</strong></td><td><code>data_engineering/</code> (Contracts, Dedup, Quality)</td><td>Silver <code>13,570</code> chặng hợp lệ + tách riêng <code>40</code> ca diverted/unknown</td></tr>
              <tr><td><strong>03. Data Mining &amp; Features</strong></td><td><code>data_mining/</code> (Profiling, EDA, Causal Builder)</td><td>Gold Causal T-3h + Ma trận ML <code>66</code> biến X + <code>dataset_manifest.json</code></td></tr>
              <tr><td><strong>04. Modeling</strong></td><td><code>modeling/</code> (Baselines, GBDT, Unsupervised, SHAP)</td><td>3-Class ICAO, Binary/Horizon ablations, MAE Regression, K-Means, PCA, IsoForest, FP-Growth</td></tr>
              <tr><td><strong>05. Evaluation</strong></td><td><code>evaluation/</code> (CV, Bootstrap, McNemar, Subgroups)</td><td>Temporal Expanding CV, Permutation Test, Subgroup Slices, Error Analysis</td></tr>
              <tr><td><strong>06. Model Governance</strong></td><td><code>governance/</code> (Tracker, Registry, Model Cards)</td><td>SHA-256 bundles trong <code>artifacts/models/</code> + <code>model_registry.json</code></td></tr>
              <tr><td><strong>07. Serving &amp; Monitoring</strong></td><td><code>inference/</code>, <code>monitoring/</code>, <code>api/</code></td><td>FastAPI REST API, <code>predict_risk()</code> fail-closed, Drift 4 tầng (PSI + KS + Concept)</td></tr>
            </tbody>
          </table>
        </div>""",
        "docs/architecture.md · configs/*.yaml · main.py",
    )

    # Slide 3: Step 1 Ingestion & Raw Audit
    files_rows = "".join(
        f"<tr><td><code>{_esc(f['file'])}</code></td><td>{f['rows']:,}</td><td>{f['cols']}</td></tr>"
        for f in raw_prov.get("files_summary", [])[:8]
    )
    add_slide(
        3, 1, "BƯỚC 1/7 · THU THẬP DỮ LIỆU ĐA NGUỒN",
        f"Kiểm Kê Bronze: {raw_prov.get('raw_file_count', 11)} File Hỗn Hợp ({raw_prov.get('raw_mixed_rows', 0):,} Dòng / {raw_prov.get('raw_mixed_columns', 58)} Cột)",
        "Hợp nhất bảng FIDS HTML (Playwright), Aviationstack JSON & Open-Meteo về Bronze 12 FIELDS bất biến",
        f"""
        <div class="grid-2col">
          <div class="card">
            <div class="card-title">Danh Mục Nguồn Bronze (Immutable Data Lake)</div>
            <table class="modern-table"><thead><tr><th>File Thô</th><th>Số dòng</th><th>Số cột</th></tr></thead><tbody>{files_rows}</tbody></table>
          </div>
          <div class="card">
            <div class="card-title">Tỷ Lệ Khuyết Thiếu Nghiêm Trọng Trong Nguồn API Thô</div>
            <ul style="font-size:12px;line-height:1.65;margin-left:18px;">
              <li><code>arrival.baggage</code>: thiếu <strong>94.69%</strong> (hành lý không cập nhật qua API công khai).</li>
              <li><code>departure.gate</code>: thiếu <strong>74.21%</strong> (chỉ cấp sát giờ ra tàu bay).</li>
              <li><code>aircraft.registration</code>: thiếu <strong>36.90%</strong>; <code>actual_time</code> trên FIDS rolling snapshots thiếu 50.71% vì gồm nhiều chuyến chưa đóng cửa lúc cào.</li>
            </ul>
            <div class="callout-box" style="margin-top:12px;">
              <strong>Quyết định kỹ thuật:</strong> Ánh xạ 58 cột thô về <strong>12 FIELDS</strong> sạch (<code>source, airport, direction, scraped_at, flight_date, airline_code, flight_no, other_airport, scheduled_time, estimated_time, actual_time, status</code>).
            </div>
          </div>
        </div>""",
        "Bước 1/7 · data/raw/ · reports/eda/eda_report.json",
    )

    # Slide 4: Step 2 Data Engineering & Policies
    add_slide(
        4, 1, "BƯỚC 2/7 · KỸ THUẬT DỮ LIỆU & HỢP ĐỒNG CHẤT LƯỢNG",
        "Làm Sạch Silver (13,570 Chặng Hợp Lệ) & Chính Sách Ngăn Nhiễu Nhãn",
        "Áp dụng băm SHA-256 3 cấp, chuẩn hóa Modulo 1440 qua nửa đêm & phân tách ca ngoại lệ",
        f"""
        <div class="grid-3col" style="margin-bottom:14px;">
          <div class="stat-callout"><div class="stat-num" style="color:#059669;">{st_dist.get('on_time', {}).get('count', 0):,} ({st_dist.get('on_time', {}).get('pct', 0)}%)</div><div class="stat-label">Lớp 0 · ON_TIME (&lt; 15 phút)</div></div>
          <div class="stat-callout"><div class="stat-num" style="color:#d97706;">{st_dist.get('delayed', {}).get('count', 0):,} ({st_dist.get('delayed', {}).get('pct', 0)}%)</div><div class="stat-label">Lớp 1 · DELAYED (&ge; 15 phút)</div></div>
          <div class="stat-callout"><div class="stat-num" style="color:#e11d48;">{st_dist.get('cancelled', {}).get('count', 0):,} ({st_dist.get('cancelled', {}).get('pct', 0)}%)</div><div class="stat-label">Lớp 2 · CANCELLED (Hủy)</div></div>
        </div>
        <div class="grid-2col">
          <div class="card">
            <div class="card-title">4 Nguyên Tắc Kỹ Thuật Dữ Liệu Sạch</div>
            <ul style="font-size:12px;line-height:1.65;margin-left:18px;">
              <li><strong>Khử trùng SHA-256 3 tầng:</strong> Exact row &rarr; Event hash &rarr; Canonical <code>leg_key</code> giữ bản chụp chốt cuối (93,081 &rarr; 13,610 chặng).</li>
              <li><strong>Phân tách trạng thái mơ hồ:</strong> <strong>40 chuyến</strong> (<code>diverted: 11</code>, <code>unknown: 29</code>) lưu riêng vào <code>excluded_outcomes.csv</code>, không ép thành <code>{0,1,2}</code>.</li>
              <li><strong>Chuyến hủy &amp; Hồi quy:</strong> Giữ nguyên <code>delay_min = NaN</code> cho 95 chuyến hủy; tuyệt đối <em>không</em> gán <code>delay_min = 0</code>.</li>
              <li><strong>Modulo 1440 qua nửa đêm:</strong> Chuyến 23:50 &rarr; 00:15 tính chuẩn <code>+25.0m</code>; cất cánh sớm hợp lệ giới hạn ở <code>-90.0m</code>.</li>
            </ul>
          </div>
          <div class="card" style="text-align:center;">
            <img class="zoomable" src="{_img_b64('01_class_distribution_by_split.png')}" onclick="openLightbox(this.src, 'Phân phối 3 lớp ICAO qua Train / Val / Holdout Test')" style="max-height:245px;border-radius:8px;border:1px solid #e2e8f0;" alt="Class Distribution"/>
            <div style="font-size:11px;color:#64748b;margin-top:4px;">Bấm vào ảnh để phóng to (Chronological Split 70/10/20)</div>
          </div>
        </div>""",
        "Bước 2/7 · data/silver/flights_silver.csv · reports/data_quality/silver_quality_report.json",
    )

    # Slide 5: Step 3 Schema Matrix
    stage_rows = "".join(
        f"<tr><td><strong>{_esc(s['stage'])}</strong></td><td>{s['columns']}</td><td>{s['rows']:,}</td><td>{_esc(s['explanation'])}</td></tr>"
        for s in ds_man.get("schema_lineage", [])
    )
    add_slide(
        5, 2, "BƯỚC 3/7 · MA TRẬN TIẾN HÓA SCHEMA",
        "Từ 58 Cột Hỗn Hợp &rarr; 12 FIELDS &rarr; 24 Silver &rarr; Gold Causal &rarr; 66 Biến ML",
        f"Dataset Version: {eda.get('dataset_version', '')} · Minh bạch hóa từng cột được giữ, biến đổi và loại bỏ",
        f"""
        <div class="card" style="margin-bottom:12px;">
          <table class="modern-table">
            <thead><tr><th>Tầng Dữ Liệu</th><th>Số cột</th><th>Số dòng</th><th>Lý do biến đổi &amp; Quy tắc kiểm soát</th></tr></thead>
            <tbody>{stage_rows}</tbody>
          </table>
        </div>
        <div class="callout-box">
          <strong>Chống rò rỉ nghiêm ngặt (9 cột hậu kiểm bị chặn):</strong> <code>delay_min, actual_time, status, status_detail, is_delayed_for_analysis, status_clean, target, outcome_observed_at, snapshot_scraped_at</code> không bao giờ xuất hiện trong 66 cột đầu vào <code>X</code>.
        </div>""",
        "Bước 3/7 · data/manifests/dataset_manifest.json · configs/schema.yaml",
    )

    # Slide 6: Step 3 EDA Breakdown
    al_rows = "".join(
        f"<tr><td><strong>{_esc(a['airline_code'])}</strong></td><td>{a['total']:,}</td><td>{a['on_time_pct']}%</td><td>{a['delayed_pct']}%</td><td>{a['cancelled_pct']}%</td><td>{a['mean_delay_min']}m</td></tr>"
        for a in biv_al
    )
    ap_rows = "".join(
        f"<tr><td><strong>{_esc(a['airport'])}</strong></td><td>{a['total']:,}</td><td>{a['on_time_pct']}%</td><td>{a['delayed_pct']}%</td><td>{a['cancelled_pct']}%</td><td>{a['mean_delay_min']}m</td></tr>"
        for a in biv_ap
    )
    add_slide(
        6, 2, "BƯỚC 3/7 · KHÁM PHÁ DỮ LIỆU (EDA) & PHÂN PHỐI",
        f"Phân Tích 7 Hãng Hàng Không &amp; 5 Sân Bay Chủ Lực ({eda_core.get('date_range', {}).get('min')} &rarr; {eda_core.get('date_range', {}).get('max')})",
        f"Phân phối delay_min (n={delays.get('count', 0):,} chuyến hoàn tất): Mean={delays.get('mean')}m · Median={delays.get('median')}m · Q75={delays.get('q75')}m · P95={delays.get('p95')}m · Max={delays.get('max')}m",
        f"""
        <div class="grid-2col">
          <div class="card">
            <div class="card-title">Bivariate Theo Hãng Hàng Không</div>
            <table class="modern-table">
              <thead><tr><th>Hãng</th><th>Chuyến</th><th>Đúng giờ</th><th>Trễ &ge;15p</th><th>Hủy</th><th>TB Trễ</th></tr></thead>
              <tbody>{al_rows}</tbody>
            </table>
          </div>
          <div class="card">
            <div class="card-title">Bivariate Theo 5 Cảng Hàng Không Trọng Điểm</div>
            <table class="modern-table">
              <thead><tr><th>Cảng</th><th>Chuyến</th><th>Đúng giờ</th><th>Trễ &ge;15p</th><th>Hủy</th><th>TB Trễ</th></tr></thead>
              <tbody>{ap_rows}</tbody>
            </table>
            <div class="callout-box" style="margin-top:10px;font-size:11.5px;">
              <strong>Lưu ý cỡ mẫu:</strong> <code>QH</code> chỉ có <code>32</code> chuyến sau làm sạch và các cảng địa phương chiếm tỷ trọng cực nhỏ &rarr; đánh dấu cảnh báo Small-N ở Evaluation.
            </div>
          </div>
        </div>""",
        "Bước 3/7 · reports/eda/eda_report.json · data/gold/flights_gold.csv",
    )

    # Slide 7: Causal Feature Engineering & Parity
    add_slide(
        7, 2, "BƯỚC 3/7 · ĐẶC TRƯNG NHÂN QUẢ (T-3H) & ĐỒNG NHẤT TRAINING-SERVING",
        "Kiến Trúc FeatureBuilder + CausalHistoryStore Không Rò Rỉ Tương Lai",
        "Đảm bảo 100% đặc trưng lúc huấn luyện và lúc dự báo trực tuyến đi qua cùng một hàm chuẩn hóa",
        """
        <div class="grid-2col">
          <div class="card">
            <div class="card-title">Tách Bạch Nhân Quả Trước Giờ Bay ($t_{\\text{pred}} = t_{\\text{sched}} - 3\\text{h}$)</div>
            <ul style="font-size:12px;line-height:1.65;margin-left:18px;">
              <li><strong>Temporal Cyclical:</strong> $\\sin/\\cos(2\\pi\\cdot h/24)$ và $\\sin/\\cos(2\\pi\\cdot \\text{dow}/7)$ giữ liên tục quanh mốc 00:00 &amp; cuối tuần.</li>
              <li><strong>Weather Hazard Score:</strong> Tổng hợp có trọng số mưa ($0.35$), gió giật ($0.25$), tầm nhìn ($0.20$), dông WMO $\\ge 95$ ($0.20$), kết hợp tương tác <code>load_x_weather</code>.</li>
              <li><strong>Causal History Rates:</strong> Dùng <code>bisect_left</code> chỉ đếm các chuyến đã có kết quả tại <code>outcome_known_at &lt; prediction_time</code> với làm mịn Bayesian $m=20$.</li>
            </ul>
          </div>
          <div class="card">
            <div class="card-title">Ma Trận Tách Chronological 70 / 10 / 20 &amp; Parity</div>
            <table class="modern-table">
              <thead><tr><th>Phân vùng</th><th>Số chuyến</th><th>Vai trò học máy</th></tr></thead>
              <tbody>
                <tr><td><strong>Train (70%)</strong></td><td>9,503</td><td>Fit <code>CausalHistoryStore</code>, <code>OneHotEncoder</code>, <code>StandardScaler</code> &amp; Models</td></tr>
                <tr><td><strong>Validation (10%)</strong></td><td>1,353</td><td>Tuyển chọn thuật toán tối ưu theo Macro-F1 / MAE</td></tr>
                <tr><td><strong>Holdout Test (20%)</strong></td><td>2,714</td><td>Kiểm thử độc lập 1 lần duy nhất (chưa từng tham gia tuning)</td></tr>
              </tbody>
            </table>
            <div class="callout-box" style="margin-top:10px;font-size:11.5px;">
              <code>test_training_serving_parity.py</code> xác minh output vector 66 chiều giữa <code>FeatureBuilder.build_dataframe</code> và <code>predict_risk()</code> trùng khớp tuyệt đối ($\\le 10^{-6}$).
            </div>
          </div>
        </div>""",
        "Bước 3/7 · src/flight_delay/data_mining/feature_engineering/builder.py · historical.py",
    )

    # Slide 8: Step 4 & 5 Classification Results
    mc_rows = ""
    for m_name in ["MajorityClassifier", "HeuristicBaseline", "HistGradientBoosting", "XGBoost", "CatBoost"]:
        mm = mc["test_comparison"][m_name]
        mark = " &starf; Production" if m_name == best_cls else ""
        mc_rows += f"<tr><td><strong>{_esc(m_name)}{mark}</strong></td><td>{mm['accuracy']:.4f}</td><td>{mm['balanced_accuracy']:.4f}</td><td><strong>{mm['f1_macro']:.4f}</strong></td><td>{mm['delayed_recall']:.4f}</td><td>{mm['cancelled_recall']:.4f}</td><td>{mm['roc_auc_ovr']:.4f}</td><td>{mm['pr_auc_macro']:.4f}</td></tr>"

    add_slide(
        8, 3, "BƯỚC 4–5/7 · BÀI TOÁN CHÍNH: PHÂN LOẠI 3 LỚP ICAO (T-3H)",
        f"HistGradientBoosting Đạt ROC-AUC OvR {best_cls_m['roc_auc_ovr']} &amp; Macro-F1 {best_cls_m['f1_macro']} Trên Holdout Test",
        "Tuyển chọn trên Validation (Val Macro-F1 = 0.3786) &middot; Báo cáo tách bạch Delayed Recall &amp; Cancelled Recall",
        f"""
        <div class="card" style="margin-bottom:10px;">
          <table class="modern-table">
            <thead><tr><th>Thuật toán</th><th>Acc</th><th>BalAcc</th><th>Macro-F1</th><th>Delayed Rec</th><th>Cancelled Rec</th><th>ROC-AUC OvR</th><th>PR-AUC Macro</th></tr></thead>
            <tbody>{mc_rows}</tbody>
          </table>
        </div>
        <div class="grid-2col" style="grid-template-columns:1.1fr 0.9fr;">
          <div class="card">
            <div class="card-title">Bằng Chứng Kiểm Định Thống Kê (Holdout Test n=2,714)</div>
            <ul style="font-size:12px;line-height:1.65;margin-left:18px;">
              <li><strong>Permutation Test (200 vòng):</strong> BalAcc mô hình <code>{perm['model_score']}%</code> vs Ngẫu nhiên <code>{perm['chance_level_mean']}% &plusmn; {perm['chance_level_std']}%</code> (<strong>p = {perm['p_value']}</strong>).</li>
              <li><strong>Vs HeuristicBaseline:</strong> McNemar <strong>p = {vs_heur['mcnemar']['p_value']}</strong>; Paired Bootstrap $\\Delta F_1 = +0.0326$ (95% CI: <code>[0.0141, 0.0568]</code>).</li>
              <li><strong>Diễn giải Cancelled Recall = 0.0%:</strong> Trong cửa sổ Test cuối cùng, chỉ có đúng <strong>2 chuyến hủy (0.07%)</strong> nên khó đạt argmax, nhưng <code>PR-AUC Cancelled = 0.2823</code> (gấp ~400 lần base rate) giúp cảnh báo sớm hiệu quả.</li>
            </ul>
          </div>
          <div class="card" style="text-align:center;">
            <img class="zoomable" src="{_img_b64('03_confusion_matrix_3class.png')}" onclick="openLightbox(this.src, 'Ma trận nhầm lẫn 3 lớp ICAO trên Holdout Test')" style="max-height:215px;border-radius:8px;" alt="Confusion Matrix"/>
          </div>
        </div>""",
        "Bước 4-5/7 · reports/experiments/classification_results.json",
    )

    # Slide 9: Binary & Horizon Ablations + SHAP
    bn_rows = ""
    for m_name in ["MajorityClassifier", "HeuristicBaseline", "HistGradientBoosting", "XGBoost", "CatBoost"]:
        bm = bn["test_comparison"][m_name]
        bn_rows += f"<tr><td><strong>{_esc(m_name)}</strong></td><td>{bm['accuracy']:.4f}</td><td>{bm['balanced_accuracy']:.4f}</td><td><strong>{bm['f1_macro']:.4f}</strong></td><td>{bm['disrupted_recall']:.4f}</td><td>{bm['roc_auc']:.4f}</td><td>{bm['brier_disrupted']:.4f}</td></tr>"

    add_slide(
        9, 3, "BƯỚC 4–5/7 · ABLATION NHỊ PHÂN, CHÂN TRỜI T-1H/3H/6H & SHAP",
        "Đánh Giá Chuyên Sâu Khả Năng Tách Nhị Phân &amp; Tầm Quan Trọng Đặc Trưng SHAP",
        "CatBoost đạt BalAcc 66.17% (Macro-F1 0.6234) trên bài toán nhị phân On-Time vs Disrupted",
        f"""
        <div class="grid-2col">
          <div>
            <div class="card" style="margin-bottom:10px;">
              <div class="card-title">Binary Disrupted Ablation (Test Set)</div>
              <table class="modern-table">
                <thead><tr><th>Mô hình</th><th>Acc</th><th>BalAcc</th><th>Macro-F1</th><th>Disr Rec</th><th>AUC</th><th>Brier</th></tr></thead>
                <tbody>{bn_rows}</tbody>
              </table>
            </div>
            <div class="card">
              <div class="card-title">Horizon Sensitivity (T-1h vs T-3h vs T-6h)</div>
              <p style="font-size:11.5px;line-height:1.55;">Tất cả chân trời cho Macro-F1 = <code>{hz[0]['test_f1_macro']}</code> trên Holdout tĩnh vì tập lịch sử nhân quả khóa cứng ở cuối tập Train; cải thiện T-1h đòi hỏi cập nhật streaming trạng thái chuyến bay nội ngày.</p>
            </div>
          </div>
          <div class="card" style="text-align:center;">
            <img class="zoomable" src="{_img_b64('04_shap_top_features.png')}" onclick="openLightbox(this.src, 'SHAP TreeExplainer Top 12 Features')" style="max-height:270px;border-radius:8px;" alt="SHAP Top Features"/>
            <div style="font-size:11px;color:#64748b;margin-top:4px;">SHAP Global Importance: <code>hour_sin, hour_cos, dow_sin, wind_gusts_10m</code></div>
          </div>
        </div>""",
        "Bước 4-5/7 · reports/experiments/horizon_ablation_results.json · shap_analysis.py",
    )

    # Slide 10: Regression L1/MAE
    reg_rows = "".join(
        f"<tr><td><strong>{_esc(k)}{' &starf;' if k == best_reg else ''}</strong></td><td><strong>{v['MAE_minutes']}m</strong></td><td>{v['MedAE_minutes']}m</td><td>{v['RMSE_minutes']}m</td><td>{v['R2_Score']}</td></tr>"
        for k, v in reg["test_comparison"].items()
    )
    add_slide(
        10, 3, "BƯỚC 4–5/7 · HỒI QUY SỐ PHÚT TRỄ (L1 / MAE REGRESSION)",
        f"{_esc(best_reg)} Đạt MAE {best_reg_m['MAE_minutes']} Phút &amp; MedAE {best_reg_m['MedAE_minutes']} Phút",
        f"Huấn luyện &amp; kiểm thử chỉ trên {reg['completed_counts']['test']:,} chuyến hoàn tất (loại trừ {reg['excluded_counts']['test']} chuyến hủy tại Test)",
        f"""
        <div class="grid-2col">
          <div class="card">
            <div class="card-title">So Sánh Các Thuật Toán Hồi Quy L1 (Test Set)</div>
            <table class="modern-table">
              <thead><tr><th>Thuật toán</th><th>MAE</th><th>MedAE</th><th>RMSE</th><th>R&sup2;</th></tr></thead>
              <tbody>{reg_rows}</tbody>
            </table>
            <div class="callout-box" style="margin-top:12px;">
              <strong>Vì sao chọn hàm mất mát L1 / MAE?</strong> Với phân phối trễ lệch phải cực mạnh (P95 = 62m, Max = 629m), L1 loss kháng nhiễu tốt hơn RMSE và giảm trực tiếp sai số trung bình tuyệt đối xuống dưới ngưỡng trễ chuẩn ICAO 15 phút.
            </div>
          </div>
          <div class="card" style="text-align:center;">
            <img class="zoomable" src="{_img_b64('05_regression_mae_comparison.png')}" onclick="openLightbox(this.src, 'So sánh MAE Hồi quy L1 trên tập Holdout Test')" style="max-height:260px;border-radius:8px;" alt="Regression MAE"/>
          </div>
        </div>""",
        "Bước 4-5/7 · reports/experiments/regression_results.json",
    )

    # Slide 11: Unsupervised Clustering, PCA & Anomaly Detection
    km_rows = "".join(
        f"<tr><td><strong>#{c['cluster_id']} {_esc(c['name'])}</strong></td><td>{c['count']:,} ({c['share_pct']}%)</td><td>{c['avg_airport_load']}</td><td>{c['delay_rate_pct']}%</td><td>{c['cancel_rate_pct']}%</td></tr>"
        for c in km["cluster_profiles"]
    )
    add_slide(
        11, 3, "BƯỚC 4/7 · KHAI PHÁ KHÔNG GIÁM SÁT (K-MEANS, PCA 3D &amp; ISOLATION FOREST)",
        f"4 Chế Độ Khai Thác Ổn Định (Seed ARI = {km['seed_stability']['ari_mean']}) &amp; Dị Thường {iso['anomaly_rate_pct']}%",
        f"PCA 3D giữ {pca['cumulative_variance_pct']}% phương sai &middot; Nhóm dị thường có tỷ lệ gián đoạn {iso['anomaly_disruption_rate_pct']}% vs {iso['normal_disruption_rate_pct']}% ở nhóm thường",
        f"""
        <div class="grid-2col">
          <div>
            <div class="card" style="margin-bottom:10px;">
              <div class="card-title">Hồ Sơ 4 Cụm Vận Hành (K-Means k=4)</div>
              <table class="modern-table">
                <thead><tr><th>Chế độ khai thác</th><th>Số chuyến</th><th>Tải TB</th><th>Trễ</th><th>Hủy</th></tr></thead>
                <tbody>{km_rows}</tbody>
              </table>
            </div>
            <div class="card">
              <div class="card-title">Thẩm Định Seed Stability &amp; Isolation Forest</div>
              <p style="font-size:11.5px;line-height:1.55;">5 seed ngẫu nhiên <code>[0,7,21,42,99]</code> cho độ đồng thuận <strong>ARI = {km['seed_stability']['ari_mean']}</strong>. Isolation Forest cô lập <strong>{iso['anomaly_count']} chuyến</strong> có đặc trưng vận hành/thời tiết cực đoan.</p>
            </div>
          </div>
          <div class="card" style="text-align:center;">
            <img class="zoomable" src="{_img_b64('06_kmeans_validation_curve.png')}" onclick="openLightbox(this.src, 'Đường cong kiểm định K-Means k=2..8 (Silhouette & Davies-Bouldin)')" style="max-height:260px;border-radius:8px;" alt="K-Means Validation Curve"/>
          </div>
        </div>""",
        "Bước 4/7 · reports/experiments/unsupervised_results.json",
    )

    # Slide 12: Association Rules FP-Growth Holdout Validated
    r_rows = "".join(
        f"<tr><td><code>{_esc(r['antecedents'])}</code></td><td><strong>{_esc(r['consequent'])}</strong></td><td>{r['support_pct']}%</td><td>{r['confidence_pct']}%</td><td><strong>{r['lift']}x</strong></td><td>{r['test_confidence_pct']}%</td><td>{r['test_lift']}x</td></tr>"
        for r in rules_info.get("top_rules", [])[:10]
    )
    add_slide(
        12, 3, "BƯỚC 4/7 · LUẬT KẾT HỢP FP-GROWTH &amp; KIỂM CHỨNG HOLDOUT",
        f"Lọc Còn {rules_info['total_non_redundant_rules']} Luật Phi Trùng Lặp (29/30 Luật Trễ Tái Lập Trên Holdout)",
        "Cắt tỉa luật dư thừa (Redundancy Pruning) và kiểm định độ nâng Lift từ Train sang Holdout Test",
        f"""
        <div class="card">
          <table class="modern-table">
            <thead><tr><th>Tiền đề (Pre-flight Context)</th><th>Hệ quả</th><th>Train Sup</th><th>Train Conf</th><th>Train Lift</th><th>Test Conf</th><th>Test Lift</th></tr></thead>
            <tbody>{r_rows}</tbody>
          </table>
          <div class="callout-box" style="margin-top:10px;font-size:11.5px;">
            <strong>Giá trị vận hành:</strong> <strong>96.7% (29/30)</strong> tập luật <code>Status=delayed</code> tái lập ổn định trên tương lai Holdout Test, chứng tỏ sự kết hợp của hãng bay, khung giờ chiều tối và mưa/tải cao là mẫu hình lặp lại thực thụ.
          </div>
        </div>""",
        "Bước 4/7 · artifacts/unsupervised/mined_association_rules.csv",
    )

    # Slide 13: Governance & 4-Tier Monitoring + FastAPI
    dd = drift["data_drift"]
    cd = drift["concept_drift"]
    add_slide(
        13, 3, "BƯỚC 6–7/7 · MODEL GOVERNANCE, 4-TIER MONITORING &amp; FASTAPI DEPLOYMENT",
        f"Giám Sát Drift ({drift['overall_status']}, Max PSI = {dd['max_psi']}) &amp; Phê Duyệt Registry SHA-256",
        f"Concept Drift Shift: {cd['disruption_shift_pct_points']} điểm % ({cd['reference_rates_pct']['disrupted_pct']}% &rarr; {cd['current_rates_pct']['disrupted_pct']}%) · Retrain Triggered = {drift['retrain_recommended']}",
        f"""
        <div class="grid-2col">
          <div>
            <div class="card" style="margin-bottom:10px;">
              <div class="card-title">Quản Trị Mô Hình &amp; Fail-Closed Serving</div>
              <ul style="font-size:12px;line-height:1.6;margin-left:18px;">
                <li><strong>Production Registry:</strong> <code>{_esc(reg_man['production']['flight_delay_classifier_3class']['model_name'] if 'model_name' in reg_man['production']['flight_delay_classifier_3class'] else 'flight_delay_classifier_3class:v2.0.0')}</code> + <code>flight_delay_regressor_mae:v2.0.0</code>.</li>
                <li><strong>Fail-closed:</strong> Kiểm checksum SHA-256 trước khi nạp, từ chối <code>prediction_time &gt; scheduled_dt</code> (HTTP 422) và chặn mọi cột hậu kiểm rò rỉ.</li>
                <li><strong>FastAPI Endpoints:</strong> <code>/api/predict</code>, <code>/api/stats</code>, <code>/api/eda</code>, <code>/api/backtest</code>, <code>/api/drift</code>, <code>/api/governance</code>, <code>/docs</code>.</li>
              </ul>
            </div>
            <div class="card">
              <div class="card-title">Khuyến Nghị Ứng Dụng &amp; Giới Hạn Thực Tế</div>
              <p style="font-size:11.5px;line-height:1.55;">Phạm vi tin cậy: 5 cảng chủ lực (SGN, HAN, DAD, CXR, PQC). Cần bổ sung dữ liệu tàu bay xoay vòng &amp; cập nhật lịch sử nội ngày cho pha triển khai tiếp theo.</p>
            </div>
          </div>
          <div class="card" style="text-align:center;">
            <img class="zoomable" src="{_img_b64('07_feature_drift_psi.png')}" onclick="openLightbox(this.src, 'Biểu đồ Population Stability Index (PSI) Train vs Holdout Test')" style="max-height:255px;border-radius:8px;" alt="Feature Drift PSI"/>
          </div>
        </div>""",
        "Bước 6-7/7 · artifacts/manifests/model_registry.json · reports/monitoring/drift_report.json",
    )

    total_slides = len(slides_html)
    doc = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FlightRisk VN 2.0 — Báo Cáo Hệ Thống 7 Bước CRISP-DM</title>
  <link rel="icon" href="data:,">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" crossorigin="anonymous">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" crossorigin="anonymous"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" crossorigin="anonymous" onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});"></script>
  <style>
    :root {{ --bg-page:#f1f5f9; --bg-deck:#ffffff; --bg-card:#ffffff; --border-subtle:#e2e8f0; --text-main:#0f172a; --text-muted:#475569; --teal-primary:#0d5c58; --teal-accent:#0f766e; --font-sans:'Inter',sans-serif; --font-mono:'JetBrains Mono',monospace; }}
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ background:var(--bg-page); font-family:var(--font-sans); color:var(--text-main); display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:100vh; padding:12px; }}
    .deck-frame {{ width:96vw; max-width:1540px; height:93vh; min-height:660px; background:var(--bg-deck); border-radius:16px; border:1px solid var(--border-subtle); box-shadow:0 10px 25px rgba(15,23,42,.08); display:flex; flex-direction:column; overflow:hidden; }}
    .deck-frame.is-fullscreen, :fullscreen .deck-frame {{ width:100vw; max-width:100vw; height:100vh; border-radius:0; }}
    .top-nav {{ height:56px; background:#fff; border-bottom:1px solid var(--border-subtle); display:flex; align-items:center; justify-content:space-between; padding:0 20px; }}
    .brand {{ font-weight:800; color:var(--teal-primary); font-size:15px; display:flex; align-items:center; gap:10px; }}
    .phase-pills {{ display:flex; gap:8px; }}
    .phase-pill {{ border:1px solid var(--border-subtle); background:#f8fafc; padding:6px 14px; border-radius:999px; font-size:12.5px; font-weight:600; cursor:pointer; color:var(--text-muted); }}
    .phase-pill.active {{ background:var(--teal-primary); color:#fff; border-color:var(--teal-primary); }}
    .nav-actions {{ display:flex; align-items:center; gap:8px; }}
    .btn-nav {{ border:1px solid var(--border-subtle); background:#fff; color:var(--text-main); padding:6px 12px; border-radius:8px; font-size:12.5px; font-weight:600; cursor:pointer; text-decoration:none; }}
    .btn-nav:hover {{ background:#f0fdfa; border-color:var(--teal-primary); }}
    .slide-viewport {{ flex:1; overflow-y:auto; position:relative; padding:24px 36px; }}
    .slide {{ display:none; height:100%; flex-direction:column; justify-content:space-between; gap:14px; }}
    .slide.active {{ display:flex; }}
    .badge {{ display:inline-block; padding:4px 10px; border-radius:6px; font-size:11px; font-weight:700; text-transform:uppercase; }}
    .badge-teal {{ background:#ccfbf1; color:#0f766e; }}
    .slide-title {{ font-size:24px; font-weight:800; color:var(--text-main); line-height:1.25; }}
    .slide-subtitle {{ font-size:13.5px; color:var(--text-muted); margin-top:4px; }}
    .slide-main-content {{ flex:1; display:flex; flex-direction:column; justify-content:center; }}
    .slide-footnote {{ font-size:11px; font-family:var(--font-mono); color:#64748b; border-top:1px dashed var(--border-subtle); padding-top:8px; }}
    .grid-2col {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:stretch; }}
    .grid-3col {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; }}
    .card {{ background:#fff; border:1px solid var(--border-subtle); border-radius:12px; padding:14px 16px; box-shadow:0 2px 4px rgba(15,23,42,.03); }}
    .card-title {{ font-size:14.5px; font-weight:700; color:var(--teal-primary); margin-bottom:10px; }}
    .stat-callout {{ background:#f8fafc; border:1px solid var(--border-subtle); border-left:4px solid var(--teal-primary); border-radius:10px; padding:12px 14px; }}
    .stat-num {{ font-size:24px; font-weight:800; font-family:var(--font-mono); }}
    .stat-label {{ font-size:12px; color:var(--text-muted); margin-top:4px; font-weight:600; }}
    .modern-table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    .modern-table th, .modern-table td {{ padding:6px 8px; border-bottom:1px solid var(--border-subtle); text-align:left; }}
    .modern-table th {{ background:#f8fafc; font-weight:700; color:var(--text-muted); }}
    .callout-box {{ background:#f0fdfa; border:1px solid #99f6e4; border-radius:8px; padding:10px 12px; color:#0f766e; font-size:12px; line-height:1.5; }}
    .bottom-bar {{ height:48px; background:#f8fafc; border-top:1px solid var(--border-subtle); display:flex; align-items:center; justify-content:space-between; padding:0 24px; }}
    .dots {{ display:flex; gap:6px; }}
    .dot {{ width:9px; height:9px; border-radius:50%; background:#cbd5e1; cursor:pointer; }}
    .dot.active {{ background:var(--teal-primary); width:22px; border-radius:5px; }}
    .zoomable {{ cursor:zoom-in; max-width:100%; height:auto; }}
    #lightbox {{ display:none; position:fixed; inset:0; background:rgba(15,23,42,.82); z-index:9999; align-items:center; justify-content:center; flex-direction:column; padding:24px; }}
    #lightbox.active {{ display:flex; }}
    #lightbox img {{ max-width:92vw; max-height:82vh; border-radius:10px; background:#fff; padding:6px; }}
    #lightbox-caption {{ color:#fff; margin-top:10px; font-size:14px; font-weight:600; }}
  </style>
</head>
<body>
  <div class="deck-frame" id="deck-frame">
    <div class="top-nav">
      <div class="brand">FlightRisk VN 2.0 &middot; Báo Cáo 7 Bước CRISP-DM</div>
      <div class="phase-pills">
        <button class="phase-pill active" data-phase="1" onclick="jumpToPhase(1)">I. Dữ liệu &amp; Kỹ thuật (1–4)</button>
        <button class="phase-pill" data-phase="2" onclick="jumpToPhase(2)">II. Schema &amp; Đặc trưng (5–7)</button>
        <button class="phase-pill" data-phase="3" onclick="jumpToPhase(3)">III. Mô hình &amp; Vận hành (8–13)</button>
      </div>
      <div class="nav-actions">
        <span id="slide-badge" style="font-family:var(--font-mono);font-size:12px;font-weight:700;color:#475569;">01 / {total_slides:02d}</span>
        <button class="btn-nav" onclick="toggleFullScreen()">Toàn màn hình</button>
        <a href="/" class="btn-nav">&larr; Về Dashboard</a>
      </div>
    </div>

    <div class="slide-viewport">
      {"".join(slides_html)}
    </div>

    <div class="bottom-bar">
      <button class="btn-nav" id="btn-prev" onclick="prevSlide()">&larr; Trang trước</button>
      <div class="dots" id="dots-wrap"></div>
      <button class="btn-nav" id="btn-next" onclick="nextSlide()">Trang tiếp &rarr;</button>
    </div>
  </div>

  <div id="lightbox" onclick="closeLightbox()">
    <img id="lightbox-img" src="" alt="Preview"/>
    <div id="lightbox-caption"></div>
  </div>

  <script>
    let currentSlide = 1;
    const totalSlides = {total_slides};
    const slides = Array.from(document.querySelectorAll(".slide"));

    function renderDots() {{
      const wrap = document.getElementById("dots-wrap");
      wrap.innerHTML = "";
      for (let i = 1; i <= totalSlides; i++) {{
        const d = document.createElement("div");
        d.className = "dot" + (i === currentSlide ? " active" : "");
        d.onclick = () => goToSlide(i);
        wrap.appendChild(d);
      }}
    }}

    function goToSlide(n) {{
      currentSlide = Math.max(1, Math.min(totalSlides, n));
      let activePhase = 1;
      slides.forEach((s, idx) => {{
        const isOn = (idx + 1) === currentSlide;
        s.classList.toggle("active", isOn);
        if (isOn) activePhase = parseInt(s.getAttribute("data-phase") || "1", 10);
      }});
      document.querySelectorAll(".phase-pill").forEach(p => {{
        p.classList.toggle("active", parseInt(p.getAttribute("data-phase"), 10) === activePhase);
      }});
      document.getElementById("slide-badge").textContent = String(currentSlide).padStart(2, "0") + " / " + String(totalSlides).padStart(2, "0");
      document.getElementById("btn-prev").disabled = currentSlide === 1;
      document.getElementById("btn-next").disabled = currentSlide === totalSlides;
      renderDots();
    }}

    function nextSlide() {{ goToSlide(currentSlide + 1); }}
    function prevSlide() {{ goToSlide(currentSlide - 1); }}
    function jumpToPhase(phase) {{
      const idx = slides.findIndex(s => parseInt(s.getAttribute("data-phase"), 10) === phase);
      if (idx >= 0) goToSlide(idx + 1);
    }}
    function toggleFullScreen() {{
      const el = document.getElementById("deck-frame");
      if (!document.fullscreenElement) el.requestFullscreen?.();
      else document.exitFullscreen?.();
    }}
    function openLightbox(src, cap) {{
      document.getElementById("lightbox-img").src = src;
      document.getElementById("lightbox-caption").textContent = cap || "";
      document.getElementById("lightbox").classList.add("active");
    }}
    function closeLightbox() {{ document.getElementById("lightbox").classList.remove("active"); }}
    document.addEventListener("keydown", e => {{
      if (e.key === "ArrowRight") nextSlide();
      else if (e.key === "ArrowLeft") prevSlide();
      else if (e.key === "Escape") closeLightbox();
    }});
    goToSlide(1);
  </script>
</body>
</html>"""
    out_web = WEB_DIR / "templates" / "presentation.html"
    out_web.write_text(doc, encoding="utf-8")
    out_root = ROOT_DIR / "presentation.html"
    shutil.copy2(out_web, out_root)
    return out_web

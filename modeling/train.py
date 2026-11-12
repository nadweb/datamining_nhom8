# -*- coding: utf-8 -*-
"""Học có giám sát và chia dữ liệu (Part 5A - Thành viên 5).

Quy trình:
1. Dữ liệu train/test do Thành viên 4 sinh ra (chia THEO THỜI GIAN, không shuffle; bộ tiền xử lý chỉ fit trên Train).
   File này KHÔNG tự chia/fit lại để tránh hai nơi cùng ghi train.csv, test.csv và metadata.
2. Phân loại 3 lớp (on_time, delayed, cancelled) với 3 mô hình theo đặc tả:
   LogisticRegression, RandomForest, HistGradientBoosting (class_weight="balanced").
3. CHỌN mô hình bằng TimeSeriesSplit trên tập Train (không dùng Test để chọn), sau đó đánh giá trên Test đúng một lần.
   Kèm baseline "luôn đoán lớp phổ biến nhất" để biết mô hình có thật sự học được gì không.
4. Hồi quy số phút trễ: so sánh baseline trung vị, HistGradientBoosting, RandomForest bằng MAE.
5. Tầm quan trọng đặc trưng: HistGradientBoosting không có feature_importances_, nên dùng permutation importance
   trên Test (kèm gộp theo biến gốc).
6. Lưu models/best_classifier.joblib, models/best_regressor.joblib, models/supervised_metrics.json.

Cột rò rỉ (delay_min, actual_time, status, status_detail, is_delayed_for_analysis) không bao giờ nằm trong X;
delay_min chỉ là nhãn y của mô hình hồi quy.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, classification_report,
    confusion_matrix, f1_score, log_loss, make_scorer, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from config import BASE_DIR
from modeling.encoding import (  # noqa: F401  (split_by_date được giữ ở đây để mã cũ import không vỡ)
    CATEGORICAL_COLS, FILE_METADATA, FILE_TEST, FILE_TRAIN, INV_LABEL_MAP, LABEL_MAP, LEAKAGE_COLUMNS,
    NUMERIC_CONTINUOUS_COLS, PASSTHROUGH_BINARY_COLS, PROCESSED_DIR, build_and_save_encoded_dataset, split_by_date,
)

MODELS_DIR = BASE_DIR / "models"
FILE_CLASSIFIER = MODELS_DIR / "best_classifier.joblib"
FILE_REGRESSOR = MODELS_DIR / "best_regressor.joblib"
FILE_METRICS = MODELS_DIR / "supervised_metrics.json"

RANDOM_STATE = 42
LABELS = [0, 1, 2]
CLASS_NAMES = ["on_time", "delayed", "cancelled"]
CV_SPLITS = 3


# ======================================================================= dữ liệu
def feature_columns(df: pd.DataFrame) -> List[str]:
    """Cột đặc trưng X: lấy từ metadata của Thành viên 4 (nếu có), nếu không thì loại các cột nhãn/rò rỉ."""
    if FILE_METADATA.exists():
        names = json.loads(FILE_METADATA.read_text(encoding="utf-8")).get("feature_names") or []
        if names and all(c in df.columns for c in names):
            return list(names)
    drop = {"target", "delay_min"} | set(LEAKAGE_COLUMNS)
    return [c for c in df.columns if c not in drop]


def load_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Nạp train.csv và test.csv của Thành viên 4; nếu chưa có thì gọi bước mã hóa của Thành viên 4 để tạo."""
    if not FILE_TRAIN.exists() or not FILE_TEST.exists():
        print("Chưa có train.csv/test.csv. Chạy bước mã hóa của Thành viên 4 (chia theo thời gian, fit chỉ trên Train)...")
        build_and_save_encoded_dataset()
    train_df, test_df = pd.read_csv(FILE_TRAIN), pd.read_csv(FILE_TEST)
    cols = feature_columns(train_df)
    return (train_df[cols], test_df[cols], train_df["target"].astype(int), test_df["target"].astype(int),
            train_df["delay_min"], test_df["delay_min"])


def prepare_chronological_train_test(source_df: pd.DataFrame = None, train_ratio: float = 0.80):
    """Giữ lại cho mã cũ: việc chia và fit nay do Thành viên 4 thực hiện (một nguồn duy nhất)."""
    if abs(train_ratio - 0.80) > 1e-9:
        raise ValueError("Tỷ lệ chia 80/20 theo thời gian được cố định trong schema.md")
    build_and_save_encoded_dataset(source_df)
    return load_train_test_data()


# ======================================================================= phân loại
def _classifier_candidates() -> Dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced",
                                               random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, class_weight="balanced",
                                                               random_state=RANDOM_STATE),
    }


def _macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0))


def select_classifier_by_time_cv(X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = CV_SPLITS):
    """Chọn mô hình bằng TimeSeriesSplit trên Train (huấn luyện trên quá khứ, chấm trên tương lai gần).
    KHÔNG nhận dữ liệu Test, nên việc chọn không thể nhìn thấy Test.
    Trả về (tên mô hình tốt nhất theo macro-F1 trung bình, kết quả CV của từng mô hình + baseline)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = dict(_classifier_candidates())
    models["Baseline_MostFrequent"] = DummyClassifier(strategy="most_frequent")
    cv = {}
    for name, model in models.items():
        folds = []
        for tr, va in tscv.split(X_train):
            m = clone(model).fit(X_train.iloc[tr], y_train.iloc[tr])
            folds.append(_macro_f1(y_train.iloc[va], m.predict(X_train.iloc[va])))
        cv[name] = {"cv_f1_macro_mean": round(float(np.mean(folds)), 4), "cv_f1_macro_std": round(float(np.std(folds)), 4),
                    "cv_f1_macro_folds": [round(f, 4) for f in folds]}
    eligible = {k: v for k, v in cv.items() if k != "Baseline_MostFrequent"}
    best = max(eligible, key=lambda k: eligible[k]["cv_f1_macro_mean"])
    return best, cv


def _prob_column(model, y_prob: np.ndarray, cls: int) -> Optional[np.ndarray]:
    classes = list(getattr(model, "classes_", LABELS))
    return y_prob[:, classes.index(cls)] if cls in classes else None


def _evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[dict, np.ndarray]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    try:
        auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr", labels=LABELS))
    except Exception:
        auc = None
    p_del = _prob_column(model, y_prob, 1)
    ap = float(average_precision_score((y_test == 1).astype(int), p_del)) if p_del is not None else None
    report = classification_report(y_test, y_pred, labels=LABELS, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", labels=LABELS, zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", labels=LABELS, zero_division=0)), 4),
        "f1_macro": round(_macro_f1(y_test, y_pred), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", labels=LABELS, zero_division=0)), 4),
        "roc_auc_ovr": round(auc, 4) if auc is not None else None,
        "average_precision_delayed": round(ap, 4) if ap is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=LABELS).tolist(),
        "classification_report": report,
    }
    return metrics, y_prob


def calibration_table(y_true_binary: np.ndarray, p: np.ndarray, bins: int = 5) -> dict:
    """Xác suất dự đoán có đúng với tần suất thật không? (cần thiết vì class_weight="balanced" làm lệch xác suất)."""
    df = pd.DataFrame({"p": p, "y": y_true_binary})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    return {
        "brier_delayed": round(float(brier_score_loss(y_true_binary, p)), 4),
        "base_rate_delayed": round(float(np.mean(y_true_binary)), 4),
        "bins": [{"bin": i + 1, "n": int(r.n), "mean_predicted": round(float(r.mean_predicted), 4),
                  "observed_rate": round(float(r.observed_rate), 4)} for i, r in enumerate(g.itertuples())],
    }


def train_classification_models(X_train, X_test, y_train, y_test, details: Optional[dict] = None):
    """Huấn luyện 3 mô hình, chọn bằng TimeSeriesSplit trên Train, đánh giá trên Test một lần.
    Trả về (mô hình tốt nhất, kết quả theo tên mô hình + 'best_classifier_name'); thông tin phụ ghi vào `details`."""
    print("\n" + "=" * 65)
    print("      HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI 3 LỚP")
    print("=" * 65)
    best_name, cv = select_classifier_by_time_cv(X_train, y_train)
    print("\nChọn mô hình bằng TimeSeriesSplit trên Train (macro-F1 trung bình):")
    for name, r in cv.items():
        print(f"  {name:<24} {r['cv_f1_macro_mean']:.4f} (± {r['cv_f1_macro_std']:.4f})")
    print(f">>> Mô hình được chọn: {best_name}")

    results, fitted, probs = {}, {}, {}
    for name, model in _classifier_candidates().items():
        model.fit(X_train, y_train)
        results[name], probs[name] = _evaluate_classifier(model, X_test, y_test)
        fitted[name] = model
        r = results[name]
        print(f"\n--- {name} (Test) ---\n  Accuracy {r['accuracy']*100:.2f}% | Balanced Acc {r['balanced_accuracy']:.4f} | "
              f"Macro-F1 {r['f1_macro']:.4f} | Weighted-F1 {r['f1_weighted']:.4f} | ROC-AUC {r['roc_auc_ovr']}")

    base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    base_metrics, _ = _evaluate_classifier(base, X_test, y_test)
    print(f"\n--- Baseline (luôn đoán lớp phổ biến nhất) ---\n  Accuracy {base_metrics['accuracy']*100:.2f}% | "
          f"Macro-F1 {base_metrics['f1_macro']:.4f}")

    best_model = fitted[best_name]
    results["best_classifier_name"] = best_name
    if details is not None:
        p_del = _prob_column(best_model, probs[best_name], 1)
        details["model_selection"] = {
            "method": f"TimeSeriesSplit(n_splits={CV_SPLITS}) trên Train, tiêu chí macro-F1; Test không tham gia việc chọn",
            "selected": best_name, "cv": cv}
        details["baselines"] = {"classifier_most_frequent": base_metrics}
        if p_del is not None:
            details["probability_calibration"] = calibration_table((y_test == 1).astype(int).values, p_del)
        details["_probs_best"] = probs[best_name]
    return best_model, results


# ======================================================================= hồi quy
def _regressor_candidates() -> Dict[str, object]:
    return {
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, random_state=RANDOM_STATE),
        "HistGradientBoosting_MAE": HistGradientBoostingRegressor(loss="absolute_error", max_iter=150, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_leaf=5,
                                              random_state=RANDOM_STATE, n_jobs=-1),
    }


def train_delay_regression_model(X_train, X_test, delay_train, delay_test, details: Optional[dict] = None):
    """Hồi quy số phút trễ trên các chuyến có delay_min (chuyến hủy không có delay_min).
    Chọn mô hình theo MAE qua TimeSeriesSplit trên Train; so với baseline luôn đoán trung vị."""
    print("\n" + "=" * 65)
    print("      HUẤN LUYỆN MÔ HÌNH HỒI QUY SỐ PHÚT TRỄ")
    print("=" * 65)
    tr_mask, te_mask = delay_train.notna(), delay_test.notna()
    Xtr, ytr = X_train[tr_mask], delay_train[tr_mask]
    Xte, yte = X_test[te_mask], delay_test[te_mask]

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    cv = {}
    for name, model in {**_regressor_candidates(), "Baseline_Median": DummyRegressor(strategy="median")}.items():
        maes = [mean_absolute_error(ytr.iloc[va], clone(model).fit(Xtr.iloc[tr], ytr.iloc[tr]).predict(Xtr.iloc[va]))
                for tr, va in tscv.split(Xtr)]
        cv[name] = {"cv_mae_mean": round(float(np.mean(maes)), 3), "cv_mae_folds": [round(float(m), 3) for m in maes]}
    best_name = min((k for k in cv if k != "Baseline_Median"), key=lambda k: cv[k]["cv_mae_mean"])

    regressor = _regressor_candidates()[best_name].fit(Xtr, ytr)
    pred = regressor.predict(Xte)
    base = DummyRegressor(strategy="median").fit(Xtr, ytr)
    mae = mean_absolute_error(yte, pred)
    base_mae = mean_absolute_error(yte, base.predict(Xte))
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    r2 = float(r2_score(yte, pred))

    print(f"  Mô hình được chọn (CV MAE): {best_name}")
    print(f"  MAE {mae:.2f} phút | RMSE {rmse:.2f} phút | R2 {r2:.4f}")
    print(f"  Baseline trung vị: MAE {base_mae:.2f} phút -> mô hình cải thiện {(1 - mae / base_mae) * 100:.1f}% MAE")
    if details is not None:
        details["regression_details"] = {
            "selected": best_name, "cv": cv, "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "baseline_median_MAE_minutes": round(float(base_mae), 2),
            "mae_improvement_vs_baseline_pct": round(float((1 - mae / base_mae) * 100), 1),
            "test_delay_mean_minutes": round(float(yte.mean()), 2),
        }
    return regressor, {"MAE_minutes": round(float(mae), 2), "RMSE_minutes": round(rmse, 2), "R2_Score": round(r2, 4)}


# ======================================================================= tầm quan trọng đặc trưng
def _importance_table(model, feature_names, X=None, y=None, n_repeats: int = 5) -> pd.DataFrame:
    """Bảng tầm quan trọng đầy đủ. HistGradientBoosting không có feature_importances_ -> dùng permutation importance
    trên dữ liệu giữ lại (độ giảm log-loss khi xáo trộn từng cột)."""
    if hasattr(model, "feature_importances_"):
        raw, method = np.asarray(model.feature_importances_, dtype=float), "impurity"
    elif X is not None and y is not None:
        scorer = make_scorer(log_loss, greater_is_better=False, response_method="predict_proba", labels=LABELS)
        r = permutation_importance(model, X, y, scoring=scorer, n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1)
        raw, method = r.importances_mean, "permutation (log-loss, Test)"
    else:
        return pd.DataFrame(columns=["feature", "importance", "raw", "method"])
    share = np.clip(raw, 0, None)
    share = share / share.sum() if share.sum() > 0 else share
    out = pd.DataFrame({"feature": list(feature_names), "importance": share, "raw": raw, "method": method})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def extract_feature_importances(model, feature_names, X=None, y=None, top_n: int = 20) -> list:
    """Top đặc trưng quan trọng: [{'feature', 'importance' (tỷ trọng, tổng = 1), 'raw', 'method'}]."""
    t = _importance_table(model, feature_names, X, y).head(top_n)
    return [{"feature": r.feature, "importance": round(float(r.importance), 4), "raw": round(float(r.raw), 5),
             "method": r.method} for r in t.itertuples()]


def group_importances_by_variable(table: pd.DataFrame) -> list:
    """Gộp các cột One-Hot về biến gốc (airport_SGN, airport_HAN, ... -> airport)."""
    def base(name: str) -> str:
        if name in NUMERIC_CONTINUOUS_COLS or name in PASSTHROUGH_BINARY_COLS:     # vd. airport_load không thuộc nhóm airport
            return name
        return next((c for c in CATEGORICAL_COLS if name.startswith(c + "_")), name)
    if table.empty:
        return []
    g = table.assign(variable=table["feature"].map(base)).groupby("variable")["importance"].sum().sort_values(ascending=False)
    return [{"variable": k, "importance": round(float(v), 4)} for k, v in g.items()]


# ======================================================================= điều phối
def run_supervised_training():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, delay_train, delay_test = load_train_test_data()
    details: dict = {}

    best_clf, clf_results = train_classification_models(X_train, X_test, y_train, y_test, details=details)
    best_reg, reg_metrics = train_delay_regression_model(X_train, X_test, delay_train, delay_test, details=details)
    details.pop("_probs_best", None)

    table = _importance_table(best_clf, list(X_train.columns), X_test, y_test)
    top = extract_feature_importances(best_clf, list(X_train.columns), X_test, y_test)
    by_var = group_importances_by_variable(table)
    if top:
        print(f"\nTop 10 đặc trưng quan trọng nhất ({top[0]['method']}):")
        for rank, item in enumerate(top[:10], 1):
            print(f"  {rank:2d}. {item['feature']:<30}: {item['importance'] * 100:.2f}%")

    meta = json.loads(FILE_METADATA.read_text(encoding="utf-8")) if FILE_METADATA.exists() else {}
    n_canc_test = int((y_test == 2).sum())
    split = {
        "cutoff_date": meta.get("cutoff_date"), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "label_share_train": {k: round(float((y_train == v).mean()), 4) for k, v in LABEL_MAP.items()},
        "label_share_test": {k: round(float((y_test == v).mean()), 4) for k, v in LABEL_MAP.items()},
        "n_cancelled_test": n_canc_test,
    }
    notes = []
    if n_canc_test < 30:
        notes.append(f"Tập Test chỉ có {n_canc_test} chuyến hủy: chỉ số của lớp cancelled (và macro-F1) có độ bất định rất lớn; "
                     f"đừng so sánh mô hình dựa trên chênh lệch nhỏ của macro-F1.")
    if abs(split["label_share_train"]["delayed"] - split["label_share_test"]["delayed"]) > 0.03:
        notes.append("Tỷ lệ trễ ở Train và Test khác nhau rõ rệt (lệch phân phối theo thời gian); xác suất dự báo sẽ lệch theo.")
    notes.append("class_weight='balanced' làm xác suất dự báo cao hơn thực tế cho lớp hiếm; xem 'probability_calibration' trước khi dùng "
                 "làm risk_score hiển thị cho người dùng.")

    metrics = {
        "classification": clf_results, "regression": reg_metrics, "feature_importances": top,
        "feature_importance_by_variable": by_var, "split": split, **details,
        "environment": {"sklearn": sklearn.__version__, "random_state": RANDOM_STATE},
        "notes": notes,
    }
    joblib.dump(best_clf, FILE_CLASSIFIER)
    joblib.dump(best_reg, FILE_REGRESSOR)
    FILE_METRICS.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 65)
    print(f"Đã lưu mô hình phân loại : {FILE_CLASSIFIER}")
    print(f"Đã lưu mô hình hồi quy   : {FILE_REGRESSOR}")
    print(f"Đã lưu chỉ số            : {FILE_METRICS}")
    print("=" * 65)
    return metrics


if __name__ == "__main__":
    run_supervised_training()

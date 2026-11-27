# -*- coding: utf-8 -*-
"""FlightRisk VN — Unified 7-Step CRISP-DM CLI Entrypoint (`python main.py <command>`)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flight_delay.common.constants import DELAY_THRESHOLD_MIN, MONITORING_CFG, PRIMARY_HORIZON_HOURS  # noqa: E402
from flight_delay.common.logging import configure_logging  # noqa: E402


def cmd_collect(args) -> None:
    from flight_delay.ingestion.collector import collect_bronze_and_standardize
    from flight_delay.ingestion.weather_client import update_weather_cache

    df = collect_bronze_and_standardize(run_live_scrape=args.live)
    print(f"[STEP 1] Bronze standardized snapshots: {len(df)} rows")
    if args.weather:
        update_weather_cache(force=False)


def cmd_prepare(args) -> None:
    from flight_delay.data_engineering.processor import run_preparation_pipeline

    valid, excluded, quality = run_preparation_pipeline(
        delay_threshold=args.threshold,
        prefer_side=args.prefer_side,
        unit=args.unit,
        min_age_hours=args.min_age,
    )
    print(
        f"[STEP 2] Silver valid={len(valid)} rows | Excluded (diverted/scheduled/unknown)={len(excluded)} | "
        f"Outcome={quality['outcome_distribution_3class']}"
    )


def cmd_eda(_args) -> None:
    from flight_delay.data_mining.eda.report_builder import generate_full_eda_report

    rep = generate_full_eda_report()
    print(f"[STEP 3] EDA report saved (total_flights={rep['eda']['total_flights']}, dataset_version={rep['dataset_version']})")


def cmd_dataset_build(args) -> None:
    from flight_delay.data_mining.datasets.dataset_builder import build_analytical_and_modeling_datasets

    res = build_analytical_and_modeling_datasets(horizon_hours=args.horizon)
    sp = res["splits"]
    print(
        f"[STEP 3] Gold Dataset v{res['dataset_version']} (T-{args.horizon}h): "
        f"Train={sp['train_rows']} | Val={sp['val_rows']} | Test={sp['test_rows']}"
    )


def cmd_features(args) -> None:
    cmd_dataset_build(args)
    cmd_eda(args)


def cmd_train(args) -> None:
    from flight_delay.modeling.classification.trainer import run_classification_suite
    from flight_delay.modeling.regression.trainer import run_regression_suite
    from flight_delay.modeling.runner import run_all_modeling_and_evaluation, run_unsupervised_suite

    if args.task == "all":
        res = run_all_modeling_and_evaluation(include_horizon_ablation=not args.no_ablation)
        mc = res["classification"]["multiclass_3"]
        print(
            f"[STEP 4-5] Classification 3-class best={mc['best_model']} | "
            f"Regression MAE best={res['regression']['best_model']}"
        )
    elif args.task == "classification":
        res = run_classification_suite(include_horizon_ablation=not args.no_ablation)
        print(f"[STEP 4-5] Classification 3-class best={res['multiclass_3']['best_model']}")
    elif args.task == "regression":
        res = run_regression_suite()
        print(f"[STEP 4-5] Regression MAE best={res['best_model']}")
    elif args.task == "unsupervised":
        res = run_unsupervised_suite()
        print(f"[STEP 4] Unsupervised done (rules={res['association_rules']['total_non_redundant_rules']})")


def cmd_evaluate(_args) -> None:
    from flight_delay.common.constants import REPORTS_DIR
    from flight_delay.common.utils import read_json
    from flight_delay.inference.audit_builder import build_holdout_audit_benchmark

    eval_path = REPORTS_DIR / "experiments" / "evaluation_report.json"
    if not eval_path.exists():
        raise FileNotFoundError("No evaluation report yet. Run `python main.py train --task all` first.")
    ev = read_json(eval_path)
    audit = build_holdout_audit_benchmark(max_records=80)
    print(
        f"[STEP 5] Classifier={ev['classification_3class']['best_model']} "
        f"Macro-F1={ev['classification_3class']['test_metrics']['f1_macro']} "
        f"| Reg MAE={ev['regression_mae']['test_metrics']['MAE_minutes']}m "
        f"| Audit sample={len(audit)}"
    )


def cmd_backtest(_args) -> None:
    from flight_delay.modeling.classification.trainer import run_classification_suite

    res = run_classification_suite(include_horizon_ablation=False)
    mc = res["multiclass_3"]
    print(f"[STEP 5] Temporal Backtest complete (best={mc['best_model']}, folds={mc['temporal_cv'].get(mc['best_model'])})")


def cmd_register(args) -> None:
    from flight_delay.governance.approval import validate_and_approve_model

    entry = validate_and_approve_model(
        model_name=args.model,
        version=args.version,
        reviewer=args.reviewer,
        promote_to_production=args.promote,
    )
    print(f"[STEP 6] Approved & transitioned {entry['model_name']}:{entry['version']} -> {entry['status']}")


def cmd_monitor(_args) -> None:
    from flight_delay.monitoring.alerts import run_full_monitoring_suite

    rep = run_full_monitoring_suite()
    print(
        f"[STEP 7] Monitoring status={rep['overall_status']} | "
        f"Max PSI={rep['data_drift']['max_psi']} | Retrain={rep['retrain_recommended']}"
    )


def cmd_serve(args) -> None:
    import uvicorn

    host = args.host or str(MONITORING_CFG["serving"]["host"])
    port = args.port or int(MONITORING_CFG["serving"]["port"])
    uvicorn.run("flight_delay.api.app:app", host=host, port=port, reload=args.reload)


def cmd_all(args) -> None:
    from flight_delay.inference.audit_builder import build_holdout_audit_benchmark

    cmd_prepare(args)
    cmd_dataset_build(argparse.Namespace(horizon=PRIMARY_HORIZON_HOURS))
    cmd_eda(args)
    cmd_train(argparse.Namespace(task="all", no_ablation=args.no_ablation))
    cmd_monitor(args)
    build_holdout_audit_benchmark(max_records=80)
    print("[PIPELINE COMPLETE] All 7 CRISP-DM steps finished & governance artifacts locked.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FlightRisk VN — 7-Step CRISP-DM Airline Delay Platform")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_col = sub.add_parser("collect", help="Step 1: Collect & standardize Bronze snapshots")
    p_col.add_argument("--live", action="store_true", help="Also run live FIDS Playwright scrape")
    p_col.add_argument("--weather", action="store_true", help="Refresh Open-Meteo cache")
    p_col.set_defaults(func=cmd_collect)

    p_prep = sub.add_parser("prepare", help="Step 2: Clean, deduplicate & weather-integrate Silver dataset")
    p_prep.add_argument("--threshold", type=int, default=DELAY_THRESHOLD_MIN)
    p_prep.add_argument("--prefer-side", default="DEPARTURES", choices=["DEPARTURES", "ARRIVALS"])
    p_prep.add_argument("--unit", default="leg", choices=["leg", "observation"])
    p_prep.add_argument("--min-age", type=float, default=0.0)
    p_prep.set_defaults(func=cmd_prepare)

    p_eda = sub.add_parser("eda", help="Step 3: Build EDA, correlation & schema report")
    p_eda.set_defaults(func=cmd_eda)

    p_ds = sub.add_parser("dataset", help="Step 3: Dataset operations (e.g. `dataset build`)")
    ds_sub = p_ds.add_subparsers(dest="action", required=True)
    p_ds_b = ds_sub.add_parser("build", help="Build causal Gold dataset & Train/Val/Test matrices")
    p_ds_b.add_argument("--horizon", type=int, default=PRIMARY_HORIZON_HOURS)
    p_ds_b.set_defaults(func=cmd_dataset_build)

    p_feat = sub.add_parser("features", help="Step 3: Build Gold features + EDA report")
    p_feat.add_argument("--horizon", type=int, default=PRIMARY_HORIZON_HOURS)
    p_feat.set_defaults(func=cmd_features)

    p_tr = sub.add_parser("train", help="Step 4-5: Train classification / regression / unsupervised models")
    p_tr.add_argument("--task", default="all", choices=["all", "classification", "regression", "unsupervised"])
    p_tr.add_argument("--no-ablation", action="store_true")
    p_tr.set_defaults(func=cmd_train)

    p_ev = sub.add_parser("evaluate", help="Step 5: Refresh Holdout Audit Benchmark & print summary")
    p_ev.set_defaults(func=cmd_evaluate)

    p_bt = sub.add_parser("backtest", help="Step 5: Run chronological expanding fold backtesting")
    p_bt.set_defaults(func=cmd_backtest)

    p_reg = sub.add_parser("register", help="Step 6: Validate, approve & promote model bundle")
    p_reg.add_argument("--model", default="flight_delay_classifier_3class")
    p_reg.add_argument("--version", default="v2.0.0")
    p_reg.add_argument("--reviewer", default="automated-governance-gate")
    p_reg.add_argument("--promote", action="store_true", default=True)
    p_reg.set_defaults(func=cmd_register)

    p_mon = sub.add_parser("monitor", help="Step 7: Run 4-tier Data Quality / PSI+KS / Concept Drift monitor")
    p_mon.set_defaults(func=cmd_monitor)

    for alias in ("serve", "web"):
        p_srv = sub.add_parser(alias, help="Step 7: Start FastAPI + Uvicorn Dashboard Server")
        p_srv.add_argument("--host", default=None)
        p_srv.add_argument("--port", type=int, default=None)
        p_srv.add_argument("--reload", action="store_true")
        p_srv.set_defaults(func=cmd_serve)

    p_all = sub.add_parser("all", help="Run full CRISP-DM Steps 2 -> 7 end-to-end")
    p_all.add_argument("--threshold", type=int, default=DELAY_THRESHOLD_MIN)
    p_all.add_argument("--prefer-side", default="DEPARTURES", choices=["DEPARTURES", "ARRIVALS"])
    p_all.add_argument("--unit", default="leg", choices=["leg", "observation"])
    p_all.add_argument("--min-age", type=float, default=0.0)
    p_all.add_argument("--no-ablation", action="store_true")
    p_all.set_defaults(func=cmd_all)
    return p


if __name__ == "__main__":
    configure_logging()
    args = build_parser().parse_args()
    args.func(args)

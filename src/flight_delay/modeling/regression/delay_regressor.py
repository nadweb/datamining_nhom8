# -*- coding: utf-8 -*-
"""[STEP 4] Tabular L1/MAE Delay Duration Regressors & Median Baseline."""
from catboost import CatBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

from flight_delay.common.constants import EXPERIMENTS_CFG, RANDOM_STATE


def build_regression_candidates() -> dict:
    cfg = EXPERIMENTS_CFG["regression"]["models"]
    return {
        "MedianBaseline": DummyRegressor(strategy="median"),
        "HistGradientBoosting_MAE": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            **cfg["HistGradientBoosting_MAE"]["params"],
        ),
        "XGBoost_MAE": XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **cfg["XGBoost_MAE"]["params"],
        ),
        "CatBoost_MAE": CatBoostRegressor(
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            **cfg["CatBoost_MAE"]["params"],
        ),
    }

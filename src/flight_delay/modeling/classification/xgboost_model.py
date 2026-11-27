# -*- coding: utf-8 -*-
"""[STEP 4] XGBoost Tabular Benchmark Model."""
from xgboost import XGBClassifier

from flight_delay.common.constants import EXPERIMENTS_CFG, RANDOM_STATE


def build_xgboost_classifier(n_classes: int = 3, params: dict = None) -> XGBClassifier:
    cfg = dict(EXPERIMENTS_CFG["classification"]["models"]["XGBoost"]["params"])
    if params:
        cfg.update(params)
    if n_classes == 2:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **cfg,
        )
    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **cfg,
    )

# -*- coding: utf-8 -*-
"""[STEP 4] Scikit-Learn HistGradientBoosting Tabular Baseline."""
from sklearn.ensemble import HistGradientBoostingClassifier

from flight_delay.common.constants import EXPERIMENTS_CFG, RANDOM_STATE


def build_hgb_classifier(params: dict = None) -> HistGradientBoostingClassifier:
    cfg = dict(EXPERIMENTS_CFG["classification"]["models"]["HistGradientBoosting"]["params"])
    if params:
        cfg.update(params)
    return HistGradientBoostingClassifier(random_state=RANDOM_STATE, **cfg)

# -*- coding: utf-8 -*-
"""[STEP 4] CatBoost Tabular Benchmark Model."""
from catboost import CatBoostClassifier

from flight_delay.common.constants import EXPERIMENTS_CFG, RANDOM_STATE


def build_catboost_classifier(n_classes: int = 3, params: dict = None) -> CatBoostClassifier:
    cfg = dict(EXPERIMENTS_CFG["classification"]["models"]["CatBoost"]["params"])
    if params:
        cfg.update(params)
    loss = "Logloss" if n_classes == 2 else "MultiClass"
    return CatBoostClassifier(
        loss_function=loss,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        **cfg,
    )

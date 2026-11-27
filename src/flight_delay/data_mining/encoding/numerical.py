# -*- coding: utf-8 -*-
"""[STEP 3] Train-Median Numeric Imputer & StandardScaler Wrapper."""
from typing import Dict, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler


class SafeNumericScaler:
    def __init__(self, columns, scaler=None, medians: Optional[Dict[str, float]] = None):
        self.columns = list(columns)
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.medians_: Optional[pd.Series] = pd.Series(medians, dtype=float) if medians else None

    def _to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.columns].apply(pd.to_numeric, errors="coerce").astype(float)

    def _fill_values(self) -> pd.Series:
        if self.medians_ is not None:
            return self.medians_.reindex(self.columns).fillna(0.0)
        return pd.Series(self.scaler.mean_, index=self.columns)

    def fit(self, df: pd.DataFrame) -> "SafeNumericScaler":
        num = self._to_numeric(df)
        self.medians_ = num.median().fillna(0.0)
        self.scaler.fit(num.fillna(self.medians_))
        return self

    def transform(self, df: pd.DataFrame):
        num = self._to_numeric(df)
        return self.scaler.transform(num.fillna(self._fill_values()))

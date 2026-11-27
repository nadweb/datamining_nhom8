# -*- coding: utf-8 -*-
"""[STEP 3] Safe Categorical One-Hot Encoder Wrapper."""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class SafeCategoricalEncoder:
    def __init__(self, columns):
        self.columns = list(columns)
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoded_names = []

    def fit(self, df: pd.DataFrame) -> "SafeCategoricalEncoder":
        clean = df[self.columns].astype(object).fillna("missing").astype(str)
        self.encoder.fit(clean)
        self.encoded_names = list(self.encoder.get_feature_names_out(self.columns))
        return self

    def transform(self, df: pd.DataFrame):
        clean = df[self.columns].astype(object).fillna("missing").astype(str)
        return self.encoder.transform(clean)

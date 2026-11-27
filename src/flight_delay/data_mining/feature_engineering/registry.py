# -*- coding: utf-8 -*-
"""[STEP 3] Declarative Feature Registry backed by configs/features.yaml."""
from dataclasses import dataclass
from typing import Dict, List

from flight_delay.common.constants import FEATURES_CFG, FORBIDDEN_POST_FLIGHT_COLUMNS
from flight_delay.common.exceptions import LeakageError


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    dtype: str
    formula: str
    source: str
    missing_policy: str
    available_before_prediction: bool
    leakage_risk: str
    role: str


class FeatureRegistry:
    """Centralized metadata & leakage guard for all modeled and analytical features."""

    def __init__(self, cfg: dict = FEATURES_CFG):
        self.schema_version: str = str(cfg.get("schema_version", "2.0"))
        groups = cfg.get("feature_groups", {})
        self.categorical_specs = [FeatureSpec(role="categorical", **item) for item in groups.get("categorical_model_inputs", [])]
        self.numeric_specs = [FeatureSpec(role="numeric", **item) for item in groups.get("numeric_model_inputs", [])]
        self.binary_specs = [FeatureSpec(role="binary", **item) for item in groups.get("binary_passthrough_inputs", [])]
        self.auxiliary_specs = [FeatureSpec(role="auxiliary", **item) for item in groups.get("auxiliary_analytical_only", [])]
        self.all_specs: Dict[str, FeatureSpec] = {
            s.name: s for s in (self.categorical_specs + self.numeric_specs + self.binary_specs + self.auxiliary_specs)
        }

    @property
    def categorical_columns(self) -> List[str]:
        return [s.name for s in self.categorical_specs]

    @property
    def numeric_columns(self) -> List[str]:
        return [s.name for s in self.numeric_specs]

    @property
    def binary_columns(self) -> List[str]:
        return [s.name for s in self.binary_specs]

    @property
    def model_input_columns(self) -> List[str]:
        cols = self.categorical_columns + self.numeric_columns + self.binary_columns
        assert_no_leakage(cols)
        return cols

    def to_manifest(self) -> List[dict]:
        return [
            {
                "name": s.name,
                "role": s.role,
                "group": s.group,
                "dtype": s.dtype,
                "formula": s.formula,
                "source": s.source,
                "missing_policy": s.missing_policy,
                "available_before_prediction": s.available_before_prediction,
                "leakage_risk": s.leakage_risk,
            }
            for s in self.all_specs.values()
        ]


def assert_no_leakage(columns) -> None:
    bad = sorted(set(columns) & set(FORBIDDEN_POST_FLIGHT_COLUMNS))
    if bad:
        raise LeakageError(f"Post-flight leakage columns forbidden in model inputs: {bad}")


FEATURE_REGISTRY = FeatureRegistry()
assert_no_leakage(FEATURE_REGISTRY.model_input_columns)

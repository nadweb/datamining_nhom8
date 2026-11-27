# -*- coding: utf-8 -*-
"""Domain-specific exceptions for fail-closed validation, leakage, and serving."""


class FlightDelayError(RuntimeError):
    """Base error for FlightRisk VN."""


class SchemaValidationError(FlightDelayError, ValueError):
    """Raised when Bronze, Silver, or Gold dataset schema validation fails."""


class DataQualityError(FlightDelayError, ValueError):
    """Raised when critical data quality invariants are violated."""


class LeakageError(FlightDelayError, ValueError):
    """Raised when post-flight or future information leaks into model inputs."""


class InvalidPredictionTimeError(FlightDelayError, ValueError):
    """Raised when prediction_time is missing, unparseable, or later than departure."""


class ArtifactError(FlightDelayError):
    """Raised when model artifacts are missing, corrupted, or checksum-mismatched."""


class ApprovalError(ArtifactError):
    """Raised when inference attempts to load an unapproved or non-production model."""

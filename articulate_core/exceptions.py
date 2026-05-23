class ArticulateError(Exception):
    """Base exception for all articulate errors."""


class LLMError(ArticulateError):
    """LLM API call failure (retryable)."""


class ConfigError(ArticulateError):
    """Bad or missing configuration."""


class StageError(ArticulateError):
    """Pipeline stage execution failure."""

    def __init__(self, message: str, stage: int):
        super().__init__(message)
        self.stage = stage


class RoutingError(ArticulateError):
    """Router cannot classify sub-task."""


class GenError(ArticulateError):
    """Code generation failure."""


class SimError(ArticulateError):
    """Simulation execution failure."""


class ValidationError(ArticulateError):
    """Validation metric thresholds not met."""


class ConversionError(ArticulateError):
    """Deployment converter failure."""


class UserCancelledError(ArticulateError):
    """User aborted at confirmation point."""

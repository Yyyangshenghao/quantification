class DataSourceError(RuntimeError):
    """Raised when an upstream data source fails."""


class ConfigError(RuntimeError):
    """Raised when configuration is invalid."""


class UnsupportedMetricError(RuntimeError):
    """Raised when a requested metric cannot be handled."""

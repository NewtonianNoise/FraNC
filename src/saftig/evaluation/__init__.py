"""Tooling to automate evaluation of filtering techniques on datasets."""

from .common import rms, total_power
from .dataset import EvaluationDataset
from .metrics import (
    EvaluationMetric,
    EvaluationMetricScalar,
    EvaluationMetricPlottable,
    MSEMetric,
    RMSMetric,
    BandwidthPowerMetric,
    PSDMetric,
)
from .evaluation import (
    EvaluationRun,
    TestDataGenerator,
    residual_power_ratio,
    residual_amplitude_ratio,
    measure_runtime,
)

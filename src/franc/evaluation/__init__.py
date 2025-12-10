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
    TimeSeriesMetric,
    SpectrogramMetric,
    WavePacketFitMetric,
)
from .evaluation import (
    EvaluationRun,
    residual_power_ratio,
    residual_amplitude_ratio,
    measure_runtime,
)
from .signal_generation import (
    TestDataGenerator,
)
from .report_generation import ReportElement
from .filter_interface import FilterInterface, make_2d_array, handle_from_dict

__all__ = [
    "rms",
    "total_power",
    "EvaluationDataset",
    "EvaluationMetric",
    "EvaluationMetricScalar",
    "EvaluationMetricPlottable",
    "MSEMetric",
    "RMSMetric",
    "BandwidthPowerMetric",
    "PSDMetric",
    "TimeSeriesMetric",
    "SpectrogramMetric",
    "WavePacketFitMetric",
    "EvaluationRun",
    "TestDataGenerator",
    "residual_power_ratio",
    "residual_amplitude_ratio",
    "measure_runtime",
    "ReportElement",
    "FilterInterface",
    "make_2d_array",
    "handle_from_dict",
]

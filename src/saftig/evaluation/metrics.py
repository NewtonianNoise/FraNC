"""Methods of evaluation noise cancellation performance."""

# this enables postponed evaluation of type annotations
# this is required to use class type hints inside of the class definition
from __future__ import annotations

import sys
from typing import Any
from collections.abc import Sequence
import abc
import functools
import warnings

import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
from scipy.signal import welch

from .dataset import EvaluationDataset

# Self type was only added in 3.11; this ensures compatibility with older python versions
if sys.hexversion >= 0x30B0000:
    from typing import Self  # pylint: disable=ungrouped-imports
else:
    Self = Any  # type: ignore

#################
# Parent classes


def welch_multiple_sequences(
    arrays: Sequence[NDArray], nperseg, *args, **kwargs
) -> tuple[NDArray, NDArray]:
    """Apply scipy.signal.welch to a sequence of arrays

    Additional arguments are passed to the scipy Welch implementation.
    Spectra are combined with an average weighted by the array lengths.
    If sequences

    :param arrays: Sequence of arrays
    :param nperseg: Length of FFT segments
    """
    norm = 0
    S_rr = np.zeros(int((nperseg + 2) / 2))
    skipped = False
    f = None

    for res in arrays:
        if len(res) >= nperseg:
            f, S_rr_i = welch(res, nperseg=nperseg, *args, **kwargs)
            S_rr += S_rr_i * len(res)
            norm += len(res)
        else:
            skipped = True
    if f is None:
        raise ValueError("All sequences are shorter than the fft block size.")

    if skipped:
        warnings.warn(
            "Skipped one or more sequences in spectral estimation as they were to short!"
        )

    S_rr /= norm
    return f, S_rr


class EvaluationMetric(abc.ABC):
    """Parent class for evaluation metrics"""

    # indicates whether data is available
    applied = False
    prediction: Sequence[NDArray]
    dataset: EvaluationDataset
    residual: Sequence[NDArray]
    parameters: dict = {}
    name: str

    # idea: initialize with the configuration, then call apply() to set data
    # apply() then returns a new instance of the metric that is configured with the data
    def apply(
        self,
        prediction: Sequence[NDArray],
        dataset: EvaluationDataset,
    ) -> Self:
        """Apply this filter"""
        # check input data shapes
        if len(prediction) != len(dataset.target_evaluation):
            raise ValueError("prediciton and target must have same length")
        for p, t in zip(prediction, dataset.target_evaluation):
            if len(p) != len(t):
                raise ValueError("all signals must have similar length")

        new_instance = type(self)(**self.parameters)
        new_instance.prediction = prediction
        new_instance.dataset = dataset

        if dataset.signal_evaluation is not None:
            new_instance.residual = [
                t - s - p
                for p, t, s in zip(
                    prediction, dataset.target_evaluation, dataset.signal_evaluation
                )
            ]
        else:
            new_instance.residual = [
                t - p for p, t in zip(prediction, dataset.target_evaluation)
            ]

        new_instance.applied = True
        return new_instance

    @property
    @abc.abstractmethod
    def result_full(self) -> tuple:
        """The raw data of the result"""

    @property
    def result(self) -> Any:
        """The result of the metric evaluation"""
        return self.result_full[0]

    @classmethod
    def result_to_text(cls, result_full: tuple[float, ...]) -> str:
        """String indicating the evaluation result"""
        del result_full  # mark non-used variable
        return f"{cls.name}"

    @property
    def text(self):
        """The text representation of the evaluation result"""
        return self.result_to_text(self.result_full)

    @staticmethod
    def result_full_wrapper(func):
        """A decorator for the result_full member function.

        Raises an exception if result is accessed on an object that was not applied to data.
        Caches the result to prevent double calculation.
        """

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.applied:
                raise RuntimeError(
                    "This functionality is only available after applying the metric to data."
                )

            if not hasattr(self, "cached_result"):
                self.cached_result = func(self, *args, **kwargs)
            return self.cached_result

        return wrapper


class EvaluationMetricScalar(EvaluationMetric):
    """Parent class for evaluation metrics that yield a scalar value"""

    @property
    def result(self) -> float:
        """The raw data of the result"""
        return self.result_full[0]

    @classmethod
    def result_to_text(cls, result_full: tuple[float, ...]) -> str:
        """String indicating the evaluation result"""
        # this default implementation works for floats
        return f"{cls.name}: {result_full[0]:f}"


class EvaluationMetricPlottable(EvaluationMetric):
    """Parent class for evaluation metrics that provide a plotting feature"""

    @abc.abstractmethod
    def plot(self, ax: Axes):
        """Generate a result plot on the given axes object"""


##########
# Metrics


class RMSMetric(EvaluationMetricScalar):
    """The RMS of the residual signal"""

    name = "Residual RMS"

    @property
    @EvaluationMetric.result_full_wrapper
    def result_full(self) -> tuple[float]:
        rms = np.sqrt(np.mean(np.square(np.concatenate(self.residual))))  # type: ignore[arg-type]
        return (rms,)

    @staticmethod
    def result_to_text(result_full: tuple[float, ...]) -> str:
        return f"Residual RMS: {result_full[0]:f}"


class MSEMetric(EvaluationMetricScalar):
    """The MSE of the residual signal"""

    name = "Residual MSE"

    @property
    @EvaluationMetric.result_full_wrapper
    def result_full(self) -> tuple[float]:
        mse = np.mean(np.square(np.concatenate(self.residual)))
        return (mse,)


class BandwidthPowerMetric(EvaluationMetricScalar):
    """The signal power on a given frequency range

    The spectrum is calculated with welch on each sequence.
    An average weighted by the sequence length is used to combine spectra from the sequences.
    The closes bins to f_start and f_stop is chosen as the integration borders.

    :param f_start: The frequency at which the power integration starts
    :param f_stop: The frequency at which the power integration stops
    :param n_fft: Sample count per FFT block used by welch
    :param window: The FFT window type
    """

    name = "Residual power on frequency range"

    def __init__(self, f_start: float, f_stop: float, n_fft: int = 1024, window="hann"):
        if f_start <= 0 or f_stop <= 0:
            raise ValueError("Frequencies must be positive")
        if n_fft < 2:
            raise ValueError("n_fft must be greater than 1")

        self.f_start = f_start
        self.f_stop = f_stop
        self.n_fft = n_fft
        self.window = window

        self.parameters = {
            "f_start": f_start,
            "f_stop": f_stop,
            "n_fft": n_fft,
            "window": window,
        }

        self.name = f"Residual power ({f_start}-{f_stop} Hz)"

    @property
    @EvaluationMetric.result_full_wrapper
    def result_full(self):
        f, S_rr = welch_multiple_sequences(
            self.residual,
            nperseg=self.n_fft,
            fs=self.dataset.sample_rate,
            window=self.window,
            scaling="density",
        )

        start_idx = np.argmin(f - self.f_start)
        stop_idx = np.argmin(f - self.f_stop)
        df = f[1] - f[0]
        power = np.sum(S_rr[start_idx : stop_idx + 1]) * df

        return (power, f[start_idx], f[stop_idx])


class PSDMetric(EvaluationMetricPlottable):
    """Plots the PSD of the given signal

    The spectrum is calculated with Welch on each sequence.
    An average weighted by the sequence length is used to combine spectra from the sequences.
    The closes bins to f_start and f_stop is chosen as the integration borders.

    :param n_fft: Sample count per FFT block used by Welch's method
    :param window: fft window type
    :param logx: Logarithmic x scale
    :param logy: Logarithmic y scale
    """

    name = "Power spectral density"

    def __init__(
        self,
        n_fft: int = 1024,
        window: str = "hann",
        logx: bool = True,
        logy: bool = True,
    ):
        if n_fft < 2:
            raise ValueError("n_fft must be greater than 1")

        self.n_fft = n_fft
        self.window = window
        self.logx = logx
        self.logy = logy

        self.parameters = {
            "n_fft": n_fft,
            "window": window,
            "logx": logx,
            "logy": logy,
        }

    @property
    @EvaluationMetric.result_full_wrapper
    def result_full(self) -> tuple[NDArray, NDArray]:
        f, S_rr = welch_multiple_sequences(
            self.residual,
            nperseg=self.n_fft,
            fs=self.dataset.sample_rate,
            window=self.window,
            scaling="density",
        )
        return (S_rr, f)

    def plot(self, ax: Axes):
        """Get the evaluation result"""
        ax.plot(*self.result_full)
        if self.logx:
            ax.set_xscale("log")
        if self.logy:
            ax.set_yscale("log")

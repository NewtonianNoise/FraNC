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
import inspect
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from scipy.signal import welch

from .dataset import EvaluationDataset
from ..common import hash_object_list, hash_function, bytes2str

# Self type was only added in 3.11; this ensures compatibility with older python versions
if sys.hexversion >= 0x30B0000:
    from typing import Self  # pylint: disable=ungrouped-imports
else:
    Self = Any  # type: ignore

#################
# Parent classes


def welch_multiple_sequences(
    arrays: Sequence[NDArray] | NDArray, nperseg, *args, **kwargs
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Apply scipy.signal.welch to a sequence of arrays

    Additional arguments are passed to the scipy Welch implementation.
    Spectra are combined with an average weighted by the array lengths.
    If sequences

    :param arrays: Sequence of arrays
    :param nperseg: Length of FFT segments

    :return: frequencies, spectrum mean, spectrum min, spectrum max
    """
    norm = 0
    S_rr = np.zeros(int((nperseg + 2) / 2))
    S_rr_all = []
    skipped = False
    f = None

    for res in arrays:
        if len(res) >= nperseg:
            f, S_rr_i = welch(res, nperseg=nperseg, *args, **kwargs)
            S_rr += S_rr_i * len(res)
            S_rr_all.append(S_rr_i)
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
    return f, S_rr, np.min(S_rr_all, axis=0), np.max(S_rr_all, axis=0)


class EvaluationMetric(abc.ABC):
    """Parent class for evaluation metrics"""

    # indicates whether data is available
    applied = False
    prediction: Sequence[NDArray] | NDArray
    dataset: EvaluationDataset
    residual: Sequence[NDArray]
    parameters: dict = {}
    name: str
    method_hash_value: bytes

    unit = "AU"

    @staticmethod
    def init_wrapper(func):
        """A decorator for the __init__function
        Saves a hash value for the configuration
        """

        @functools.wraps(func)
        def wrapper(self, **kwargs):
            # save init parameters
            self.parameters = {key: kwargs[key] for key in sorted(kwargs)}

            # calculate method hash
            hashes = self._file_hash()  # pylint: disable=[protected-access]
            hashes += hash_object_list(list(self.parameters.keys()))
            hashes += hash_object_list(list(self.parameters.values()))
            self.method_hash_value = hash_function(hashes)
            return func(self, **kwargs)

        return wrapper

    @init_wrapper
    def __init__(self, **kwargs):
        """Placeholder init function to ensure a hash is calculated"""
        del kwargs  # mark as unused

    # idea: initialize with the configuration, then call apply() to set data
    # apply() then returns a new instance of the metric that is configured with the data
    def apply(
        self,
        prediction: Sequence[NDArray] | NDArray,
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

        new_instance.unit = dataset.target_unit
        return new_instance

    @abc.abstractmethod
    def result_full(self) -> tuple:
        """The raw data of the result"""

    @property
    def result(self) -> Any:
        """The result of the metric evaluation"""
        return self.result_full()[0]

    @classmethod
    def result_to_text(cls, result_full: tuple[float | np.floating, ...]) -> str:
        """String indicating the evaluation result

        :param result_full: The return value of metric.result_full()
        """
        return f"{cls.name}: {str(result_full[0])}"

    @property
    def text(self):
        """The text representation of the evaluation result"""
        return self.result_to_text(self.result_full())

    @classmethod
    def _file_hash(cls) -> bytes:
        """Calculates a hash value based on the file in which this method was defined."""
        try:
            with open(inspect.getfile(cls), "rb") as f:
                script = f.read()
        except TypeError:
            try:
                script = inspect.getsource(cls).encode()
            except TypeError:
                script = cls.name.encode()
                warnings.warn(f"Could not include source code in hash for {cls.name}")

        return hash_function(script)

    @property
    def method_hash(self) -> bytes:
        """A hash representing the configured metric as a bytes object"""
        if not hasattr(self, "method_hash_value") or self.method_hash_value is None:
            raise NotImplementedError(
                f"The metric {type(self)} __init__() function is missing the @init_wrapper decorator."
            )
        return self.method_hash_value

    @property
    def method_hash_str(self) -> str:
        """A hash representing the configured metric as a base64 like string"""
        return bytes2str(self.method_hash)

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

    unit: str

    @property
    def result(self) -> float:
        """The raw data of the result"""
        return self.result_full()[0]

    @classmethod
    def result_to_text(cls, result_full: tuple[float | np.floating, ...]) -> str:
        """String indicating the evaluation result"""
        # this default implementation works for floats
        return f"{cls.name}: {result_full[0]:f}"


class EvaluationMetricPlottable(EvaluationMetric):
    """Parent class for evaluation metrics that provide a plotting feature"""

    plot_path: str | Path | None = None

    @classmethod
    def result_to_text(cls, result_full: tuple[float | np.floating, ...]) -> str:
        """String indicating the evaluation result"""
        return f"{cls.name}"

    @abc.abstractmethod
    def plot(self, ax: Axes):
        """Generate a result plot on the given axes object"""

    def save_plot(
        self,
        fname: str | Path,
        figsize: tuple[int, int] = (10, 4),
        tight_layout: bool = True,
    ):
        """Save the plot to a file"""
        # set serif font globally for matplotlib
        plt.rcParams["font.family"] = "serif"

        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(True, zorder=-1)
        self.plot(ax)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(fname)
        plt.close(fig)

        self.plot_path = fname

    def filename(self, context: str) -> str:
        """Generate a filename that includes the given context string

        :param context: This string is included in the generated filename
        """
        return self.name + "_" + context + "_" + self.method_hash_str + ".pdf"


##########
# Metrics


class RMSMetric(EvaluationMetricScalar):
    """The RMS of the residual signal"""

    name = "Residual RMS"

    @EvaluationMetric.result_full_wrapper
    def result_full(self) -> tuple[np.floating | float, str]:
        rms = np.sqrt(np.mean(np.square(np.concatenate(self.residual))))
        return (rms, self.unit)

    @classmethod
    def result_to_text(cls, result_full: tuple[float | np.floating, ...]) -> str:
        return f"{cls.name}: {result_full[0]:f} {result_full[1]}"


class MSEMetric(EvaluationMetricScalar):
    """The MSE of the residual signal"""

    name = "Residual MSE"

    def apply(self, *args, **kwargs):
        new_instance = super().apply(*args, **kwargs)

        new_instance.unit = f"({new_instance.unit})²"
        return new_instance

    @EvaluationMetric.result_full_wrapper
    def result_full(self) -> tuple[np.floating | float, str]:
        mse = np.mean(np.square(np.concatenate(self.residual)))
        return (mse, self.unit)

    @classmethod
    def result_to_text(cls, result_full: tuple[float | np.floating, ...]) -> str:
        return f"{cls.name}: {result_full[0]:f} {result_full[1]}"


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
        super().__init__(f_start=f_start, f_stop=f_stop, n_fft=n_fft, window=window)

        if f_start <= 0 or f_stop <= 0:
            raise ValueError("Frequencies must be positive")
        if n_fft < 2:
            raise ValueError("n_fft must be greater than 1")

        self.f_start = f_start
        self.f_stop = f_stop
        self.n_fft = n_fft
        self.window = window

        self.name = f"Residual power ({f_start}-{f_stop} Hz)"

    @EvaluationMetric.result_full_wrapper
    def result_full(self):
        f, S_rr, _, _ = welch_multiple_sequences(
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
    :param window: FFT window type
    :param logx: Logarithmic x scale
    :param logy: Logarithmic y scale
    :param show_target: If True, also show spectrum of the target signal
    """

    name = "Power spectral density"

    @EvaluationMetric.init_wrapper
    def __init__(
        self,
        n_fft: int = 1024,
        window: str = "hann",
        logx: bool = True,
        logy: bool = True,
        show_target: bool = True,
    ):
        super().__init__(
            n_fft=n_fft, window=window, logx=logx, logy=logy, show_target=show_target
        )

        if n_fft < 2:
            raise ValueError("n_fft must be greater than 1")

        self.n_fft = n_fft
        self.window = window
        self.logx = logx
        self.logy = logy
        self.show_target = show_target

    def _welch_multiple_sequences(self, signal: Sequence[NDArray]):
        """apply welch_multiple_sequences() with correct settings"""
        return welch_multiple_sequences(
            signal,
            nperseg=self.n_fft,
            fs=self.dataset.sample_rate,
            window=self.window,
            scaling="density",
        )

    @EvaluationMetric.result_full_wrapper
    def result_full(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        f, S_rr, S_rr_min, S_rr_max = self._welch_multiple_sequences(self.residual)
        return (S_rr, f, S_rr_min, S_rr_max)

    def plot(self, ax: Axes):
        """Plot to the given Axes object"""
        freq = self.result_full()[1]
        ax.fill_between(
            freq, self.result_full()[2], self.result_full()[3], fc="C0", alpha=0.3
        )
        ax.plot(freq, self.result, label="Residual", c="C0")
        if self.show_target:
            f, Stt, Stt_min, Stt_max = self._welch_multiple_sequences(
                self.dataset.target_evaluation
            )
            ax.fill_between(freq, Stt_min, Stt_max, fc="C1", alpha=0.3)
            ax.plot(f, Stt, label="Target", c="C1")
            plt.legend()

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(f"PSD [({self.dataset.target_unit})$^2$/Hz]")

        if self.logx:
            ax.set_xscale("log")
            ax.set_xlim(min(freq[freq > 0]), max(freq))
        else:
            ax.set_xlim(min(freq), max(freq))
        if self.logy:
            ax.set_yscale("log")


class TimeSeriesMetric(EvaluationMetricPlottable):
    """Plots the signal as a time series"""

    name = "Time series"

    @EvaluationMetric.init_wrapper
    def __init__(
        self,
        show_target: bool = True,
    ):
        super().__init__(show_target=show_target)
        self.show_target = show_target

    @EvaluationMetric.result_full_wrapper
    def result_full(self) -> tuple[Sequence[NDArray],]:
        return (self.residual,)

    def plot(self, ax: Axes):
        """Plot to the given Axes object"""
        if self.show_target:
            y = np.concatenate(self.dataset.target_evaluation)
            ax.plot(
                np.arange(len(y)) / self.dataset.sample_rate,
                y,
                label="Target",
                rasterized=True,
            )

        y = np.concatenate(self.result_full()[0])
        x = np.arange(len(y)) / self.dataset.sample_rate
        ax.plot(
            x,
            y,
            label="Residual",
            rasterized=True,
        )

        x_marker = 0.0
        for section in self.result_full()[0]:
            x_marker += len(section) / self.dataset.sample_rate
            plt.axvline(x_marker, c="k")

        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Target/residual signal [{self.dataset.target_unit}]")

        ax.legend()

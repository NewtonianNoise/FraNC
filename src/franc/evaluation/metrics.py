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
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram

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

    applied = False
    """indicates whether data is available"""
    prediction: Sequence[NDArray] | NDArray
    dataset: EvaluationDataset
    residual: Sequence[NDArray]
    """Residual without the signal (=zero for perfect filter)."""
    residual_signal: Sequence[NDArray]
    """Residual including the signal (=signal for perfect filter)."""
    parameters: dict = {}
    """The parameters with which the metric was initialized.

    Needed to re instantiate the filter during apply() and for hashing.
    """
    name: str
    method_hash_value: bytes

    unit = "AU"
    """unit of the target and prediction channels"""

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
            new_instance.residual_signal = [
                t - p for p, t in zip(prediction, dataset.target_evaluation)
            ]
        else:
            new_instance.residual = [
                t - p for p, t in zip(prediction, dataset.target_evaluation)
            ]
            new_instance.residual_signal = new_instance.residual

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
        dpi: float = 200,
        replot=False,
    ):
        """Save the plot to a file

        :param fname: Output file name
        :param figsize: A matplotlib figure size parameter
        :param tight_layout: Whether to use matplotlib tight figure command
        :param dpi: Output figure dpi value
        :param replot: If false, no new plot will be generated if a file with the same name already exists.
        """
        self.plot_path = fname

        if replot or (not Path(fname).exists()):
            # set serif font globally for matplotlib
            plt.rcParams["font.family"] = "serif"

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.grid(True, zorder=-1)
            self.plot(ax)
            if tight_layout:
                plt.tight_layout()
            plt.savefig(fname)
            plt.close(fig)

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


class PSDMetric(
    EvaluationMetricPlottable
):  # pylint: disable=too-many-instance-attributes
    """Plots the PSD of the given signal

    The spectrum is calculated with Welch on each sequence.
    An average weighted by the sequence length is used to combine spectra from the sequences.
    The closes bins to f_start and f_stop is chosen as the integration borders.

    :param n_fft: Sample count per FFT block used by Welch's method
    :param window: FFT window type
    :param logx: Logarithmic x scale
    :param logy: Logarithmic y scale
    :param show_target: If True, also show spectrum of the target channel
    :param show_target_minus_signal: If True, also show spectrum of the target channel minus the signal
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
        show_target_minus_signal: bool = True,
        show_signal: bool = False,
        autoscale: bool = False,
    ):
        super().__init__(
            n_fft=n_fft,
            window=window,
            logx=logx,
            logy=logy,
            show_target=show_target,
            show_target_minus_signal=show_target_minus_signal,
            show_signal=show_signal,
            autoscale=autoscale,
        )

        if n_fft < 2:
            raise ValueError("n_fft must be greater than 1")

        self.n_fft = n_fft
        self.window = window
        self.logx = logx
        self.logy = logy
        self.show_target = show_target
        self.show_target_minus_signal = show_target_minus_signal
        self.show_signal = show_signal
        self.autoscale = autoscale

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

    def _plot_channel(
        self,
        ax: Axes,
        signal: Sequence[NDArray],
        label: str,
        color=None,
        ls="-",
        zorder=10,
    ):
        """Plot spectrum of the signal onto the axes object"""
        f, Stt, Stt_min, Stt_max = self._welch_multiple_sequences(signal)
        ax.fill_between(f, Stt_min, Stt_max, fc=color, alpha=0.3)
        ax.plot(f, Stt, label=label, c=color, ls=ls, zorder=zorder)
        plt.legend()

    def plot(self, ax: Axes):
        """Plot to the given Axes object"""
        freq = self.result_full()[1]

        # setup up axes
        if self.logx:
            ax.set_xscale("log")
            ax.set_xlim(min(freq[freq > 0]), max(freq))
        else:
            ax.set_xlim(min(freq), max(freq))
        if self.logy:
            ax.set_yscale("log")
        ax.fill_between(
            freq, self.result_full()[2], self.result_full()[3], fc="C0", alpha=0.3
        )

        # plotting
        ax.plot(freq, self.result, label="Residual", c="C0")

        if self.show_target:
            self._plot_channel(ax, self.dataset.target_evaluation, "Target", "C1")

        if self.show_target_minus_signal and self.dataset.has_signal:
            difference = [
                t - s
                for t, s in zip(
                    self.dataset.target_evaluation,
                    self.dataset.signal_evaluation,  # type: ignore[arg-type]
                )
            ]
            self._plot_channel(ax, difference, "Target - signal", "C2")

        if self.dataset.has_signal:
            self._plot_channel(
                ax, self.residual_signal, "Residual w/ signal", "C3", ls="--"
            )

        if not self.autoscale:
            ax.autoscale(
                False
            )  # to prevent very low values of signal from significantly altering the result plot
        if self.show_signal and self.dataset.has_signal:
            self._plot_channel(ax, self.dataset.signal_evaluation, "Signal", "C4", zorder=9)  # type: ignore[arg-type]

        # labels
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(f"PSD [({self.dataset.target_unit})$^2$/Hz]")


class TimeSeriesMetric(EvaluationMetricPlottable):
    """Plots the signal as a time series

    :param show_target: if True, display the target channel
    :param show_target_minus_signal: if True, display the target channel minus the signal channel
    :param start: Start of the shown data as a sample index to the concatenated evaluation sequences.
    :param stop: Stop of the shown data as a sample index to the concatenated evaluation sequences.
    """

    name = "Time series"

    @EvaluationMetric.init_wrapper
    def __init__(
        self,
        show_target: bool = True,
        show_target_minus_signal: bool = True,
        show_signal: bool = False,
        residual_with_signal=True,
        start: int = 0,
        stop: int = -1,
    ):
        super().__init__(
            show_target=show_target,
            show_target_minus_signal=show_target_minus_signal,
            show_signal=show_signal,
            residual_with_signal=residual_with_signal,
            stop=stop,
            start=start,
        )
        self.show_target = show_target
        self.show_target_minus_signal = show_target_minus_signal
        self.show_signal = show_signal
        self.residual_with_signal = residual_with_signal
        self.start = start
        self.stop = stop

    @EvaluationMetric.result_full_wrapper
    def result_full(self) -> tuple[Sequence[NDArray],]:
        if self.residual_with_signal:
            return (self.residual_signal,)
        return (self.residual,)

    def plot(self, ax: Axes):
        """Plot to the given Axes object"""
        t = np.concatenate(self.dataset.target_evaluation)[self.start : self.stop]
        x = np.arange(len(t)) / self.dataset.sample_rate

        if self.show_target:
            ax.plot(x, t, label="Target", rasterized=True)

        r = np.concatenate(self.result_full()[0])[self.start : self.stop]
        label_residual = (
            "Residual w/ signal" if self.residual_with_signal else "Residual w/o signal"
        )
        ax.plot(x, r, label=label_residual, rasterized=True)

        if self.dataset.has_signal:
            s = np.concatenate(self.dataset.signal_evaluation)[self.start : self.stop]  # type: ignore[arg-type]
            if self.show_target_minus_signal:
                ax.plot(x, t - s, label="Target - Signal", rasterized=True, ls="--")

            if self.show_signal:
                ax.plot(x, s, label="Signal", rasterized=True, ls="--")

        x_marker = 0.0
        for section in self.result_full()[0]:
            x_marker += len(section) / self.dataset.sample_rate
            plt.axvline(x_marker, c="k")

        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Target/residual signal [{self.dataset.target_unit}]")

        ax.legend()


class SpectrogramMetric(EvaluationMetricPlottable):
    """Plots a spectrogram (waterfall diagram) for the residual signal

    :param n_fft: Sample count per FFT block used by Welch's method
    :param window: FFT window type
    """

    name = "Spectrogram"

    @EvaluationMetric.init_wrapper
    def __init__(
        self,
        n_fft: int = 4096,
        window: str = "hann",
        with_signal: bool = True,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        asd: bool = True,
    ):
        super().__init__(
            n_fft=n_fft,
            window=window,
            with_signal=with_signal,
            xlim=xlim,
            ylim=ylim,
            asd=asd,
        )

        if n_fft < 2:
            raise ValueError("n_fft must be greater than 1")

        self.n_fft = n_fft
        self.window = window
        self.with_signal = with_signal
        self.xlim = xlim
        self.ylim = ylim
        self.asd = asd

    @EvaluationMetric.result_full_wrapper
    def result_full(self) -> tuple[NDArray, tuple[float, float, float, float], str]:
        """The spectrogram and additional information
        :return: (spectrogram, spectrogram extent, figure_label)
            The spectrogram extent is given in the format that `matplotlib.pyplot.imshow` requires.
        """
        residual = self.residual_signal if self.with_signal else self.residual
        residual = np.concatenate(residual)
        f, t, Sxx = spectrogram(
            residual,
            fs=self.dataset.sample_rate,
            window=self.window,
            nperseg=self.n_fft,
            scaling="density",
        )

        figure_label = " with signal" if self.with_signal else ""
        if self.xlim is not None:
            figure_label += f", {self.xlim[0]}-{self.xlim[1]} s"
        if self.ylim is not None:
            figure_label += f", {self.ylim[0]}-{self.ylim[1]} Hz"

        if self.asd:
            Sxx = np.sqrt(Sxx)
            figure_label = "ASD " + figure_label

        return (
            Sxx,
            (t[0], t[-1], f[0], f[-1]),
            figure_label,
        )

    @classmethod
    def result_to_text(cls, result_full: tuple[float | np.floating, ...]) -> str:
        """String indicating the evaluation result

        :param result_full: The return value of metric.result_full()
        """
        return f"{cls.name} {result_full[2]}"

    def plot(self, ax: Axes):
        result_full = self.result_full()
        plt.imshow(
            result_full[0],
            norm=LogNorm(),
            extent=result_full[1],
            aspect="auto",
            origin="lower",
        )
        c_label = (
            f"Residual signal [{self.dataset.target_unit}/√Hz]"
            if self.asd
            else f"Residual signal [({self.dataset.target_unit})²/Hz]"
        )
        plt.colorbar(ax=ax, label=c_label)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")

        if self.xlim is not None:
            ax.set_xlim(*self.xlim)
        if self.ylim is not None:
            ax.set_ylim(*self.ylim)

        x_marker = 0.0
        for section in self.residual:
            x_marker += len(section) / self.dataset.sample_rate
            plt.axvline(x_marker, c="k")

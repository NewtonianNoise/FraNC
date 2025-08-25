"""Collection of tools for the evaluation and testing of filters"""

from typing import Any, Optional
from collections.abc import Sequence
from timeit import timeit

import numpy as np
from numpy.typing import NDArray

from .common import total_power
from ..filtering import FilterBase


class TestDataGenerator:
    """Generate simple test data for correlated noise mitigation techniques
    The channel count is implicitly defined by the shape of witness_noise_level

    :param witness_noise_level: amplitude ratio of the sensor noise
                to the correlated noise in the witness sensor
                Scalar or 1D-vector for multiple sensors
    :param target_noise_level: amplitude ratio of the sensor noise
                to the correlated noise in the target sensor
    :param transfer_functon: ratio between the amplitude in the target and witness signals
    :param sample_rate: The outputs are referenced
                to an ASD of 1/sqrt(Hz) if a sample rate is provided

    >>> import saftig as sg
    >>> # create data with two witness sensors with relative noise amplitudes of 0.1
    >>> tdg = sg.evaluation.TestDataGenerator(witness_noise_level=[0.1, 0.1])
    >>> # generate a dataset with 1000 samples
    >>> witness, target = tdg.generate(1000)
    >>> witness.shape, target.shape
    ((2, 1000), (1000,))

    """

    rng: Any

    def __init__(
        self,
        witness_noise_level: float | Sequence = 0.1,
        target_noise_level: float = 0,
        transfer_function: float = 1,
        sample_rate: float = 1.0,
        rng_seed: Optional[int] = None,
    ):
        self.witness_noise_level = np.array(witness_noise_level)
        self.target_noise_level = np.array(target_noise_level)
        self.transfer_function = np.array(transfer_function)
        self.sample_rate = sample_rate

        if rng_seed is None:
            self.rng = np.random
        else:
            self.rng = np.random.default_rng(rng_seed)

        if len(self.witness_noise_level.shape) == 0:
            self.witness_noise_level = np.array([self.witness_noise_level])

        assert (
            len(self.witness_noise_level.shape) == 1
        ), f"witness_noise_level.shape = {self.witness_noise_level.shape}"
        assert len(self.target_noise_level.shape) == 0
        assert len(self.transfer_function.shape) == 0
        assert self.sample_rate > 0

    def scaled_whitenoise(self, shape) -> NDArray:
        """Generate whitenoise with an ASD of one

        :param shape: shape of the new array

        :return: Array of white noise
        """
        return self.rng.normal(0, np.sqrt(self.sample_rate / 2), shape)

    def generate(self, n: int) -> tuple[NDArray, NDArray]:
        """Generate sequences of samples

        :param N: number of samples

        :return: witness signal, target signal

        """
        t_c = self.scaled_whitenoise(n)
        w_n = (
            self.scaled_whitenoise((len(self.witness_noise_level), n))
            * self.witness_noise_level[:, None]
        )
        t_n = self.scaled_whitenoise(n) * self.target_noise_level

        return (t_c + w_n) * self.transfer_function, (t_c + t_n)


def measure_runtime(
    filter_classes: Sequence[FilterBase],
    n_samples: int = int(1e4),
    n_filter: int = 128,
    idx_target: int = 0,
    n_channel: int = 1,
    additional_filter_settings: Sequence[dict] | None = None,
    repititions: int = 1,
) -> tuple[Sequence, Sequence]:
    """Measure the runtime of filers for a specific scenario
    Be aware that this gives no feedback upon how much multithreading is used!

    :param n_samples: Length of the test data
    :param n_filter: Length of the FIR filters / input block size
    :param idx_target: Position of the prediction
    :param n_channel: Number of witness sensor channels
    :param additional_filter_settings: optional settings passed to the filters
    :param repititions: how manu repititions to perform during the timing measurement

    :return: (time_conditioning, time_apply) each in seconds
    """
    filter_classes = list(filter_classes)
    if additional_filter_settings is None:
        additional_filter_settings = [{}] * len(filter_classes)
    additional_filter_settings = list(additional_filter_settings)
    assert len(additional_filter_settings) == len(filter_classes)

    witness, target = TestDataGenerator([0.1] * n_channel).generate(n_samples)

    times_conditioning = []
    times_apply = []

    def time_filter(filter_class, args):
        """wrapper function to make closures work correctly"""
        filt = filter_class(n_filter, idx_target, n_channel, **args)
        t_cond = timeit(lambda: filt.condition(witness, target), number=repititions)
        t_pred = timeit(lambda: filt.apply(witness, target), number=repititions)
        return t_cond / repititions, t_pred / repititions

    for fc, args in zip(filter_classes, additional_filter_settings):
        t_cond, t_pred = time_filter(fc, args)
        times_conditioning.append(t_cond)
        times_apply.append(t_pred)

    return times_conditioning, times_apply


def residual_power_ratio(
    target: Sequence,
    prediction: Sequence,
    start: int | None = None,
    stop: int | None = None,
    remove_dc: bool = True,
) -> float:
    """Calculate the ratio between residual power of the residual and the target signal

    :param target: target signal array
    :param prediction: prediction array (same length as target
    :param start: use only a section of the arrays, start at this index
    :param stop: use only a section of the arrays, stop at this index
    :param remove DC component: remove DC component before calculation
    """
    target_npy = np.array(target[start:stop]).astype(np.float64)
    prediction_npy = np.array(prediction[start:stop]).astype(np.float64)
    assert target_npy.shape == prediction_npy.shape

    if remove_dc:
        target_npy -= np.mean(target)
        prediction_npy -= np.mean(prediction_npy)

    residual = prediction_npy - target_npy

    return float(total_power(residual) / total_power(target_npy))


def residual_amplitude_ratio(*args, **kwargs) -> float:
    """Calculate the ratio between residual amplitude of the residual and the target signal

    :param target: target signal array
    :param prediction: prediction array (same length as target
    :param start: use only a section of the arrays, start at this index
    :param stop: use only a section of the arrays, stop at this index
    :param remove DC component: remove DC component before calculation
    """
    return float(np.sqrt(residual_power_ratio(*args, **kwargs)))

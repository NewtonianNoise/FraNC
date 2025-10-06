"""Tools to generate test data and EvaluationDatasets"""

from typing import Any
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
import numba

from .dataset import EvaluationDataset

NDArrayF = NDArray[np.floating]
NDArrayU = NDArray[np.uint]


@numba.jit
def generate_wave_packet(
    offset: float,
    width: float,
    amplitude: float,
    frequency: float,
    phase: float,
    generation_width: int = 10,
    peak_scaling: bool = True,
) -> NDArrayF:
    """Generate a Gaussian wave packet. Time related parameters (width and frequency)
    can be interpreted as number of samples or seconds at a sampling rate of 1 Hz.

    The amplitude normalization is chosen so that the sum of squares of the samples is one,
    so that the amplitude defines a fixed total energy of the signal.

    :param offset: Timing of the wave packet (negative values shift the packet to the past)
    :param width: The standard deviation σ parameter of the hull curve
    :param amplitude: Amplitude of the signal
    :param frequency: Frequency of the sinusoidal signal component
    :param phase: Phase of the sinusoidal signal component in rad. Zero indicates that a maximum is in phase with the peak of the hull curve
    :param generation_width: How many samples to generate, indicated in multiples of the width parameter to one side.
        Example: width=5 and generation_width=10 will result in 2*5*10=100+1 samples total.
    :param peak_scaling: If `True`, amplitude will determine the potential maximum value of the wave packet.
        If set to `False`, the wave packet will be scaled so that the sum of the squares of all samples is one.

    >>> import franc
    >>> franc.eval.signal_generation.generate_wave_packet(400, 100, 1, 0.02, 0)
    array([-2.49881816e-53,  1.67054194e-40,  3.81228489e-40, ...,
           -1.79993484e-05, -8.54421912e-06, -1.88708852e-19],
          shape=(2001,))
    """
    half_length = int(np.ceil(width * generation_width))
    T = np.arange(2 * half_length + 1) - half_length - offset
    sinusoidal = np.sin(2 * np.pi * frequency * T + phase) * np.sqrt(2)

    # First option: does not work with numba.jit
    # hull = scipy.stats.norm.pdf(T, 0, width)
    # Gaussian and normalization written separately
    # hull = np.exp(-((T / width) ** 2) / 2) / width / np.sqrt(2 * np.pi)
    # hull *= np.sqrt(width) * 2
    hull = np.exp(-((T / width) ** 2) / 2) * np.sqrt(2 / np.pi / width)
    if peak_scaling:
        hull /= hull[half_length]

    return sinusoidal * hull * amplitude


def generate_wave_packet_signal(
    n_samples: int,
    n_wave_packets: int,
    width_range: tuple[float, float],
    amplitude_range: tuple[float, float],
    frequency_range: tuple[float, float],
    rng: np.random.Generator,
    generation_width: int = 10,
    peak_scaling: bool = True,
    offset_range: float | None = None,
) -> tuple[NDArrayF, NDArrayF]:
    """Generate a signal consisting of wave packets with `generate_wave_packet`.
    All values are generated based on uniform distributions with the given ranges.

    :param n_samples: Length of the generated sequence
    :param n_wave_packets: Number of generated wave packets
    :param width_range: Tuple with lower and upper bound for width parameter
    :param amplitude_range: Tuple with lower and upper bound for amplitude parameter
    :param frequency_range: Tuple with lower and upper bound for frequency parameter
    :param rng: A numpy random number generator instance
    :param generator_width: How many standard deviations of the hull curve will be generated per packet
        Higher numbers will increase calculation time
    :param peak_scaling: Passed to generate_wave_packet()
    :param offset_range: The range of time offsets generated for the wave packets
        If no value is provided, width_range[1] is used.

    :return: Generated sequence, Sequence[(position, width, amplitude, frequency, phase)]
    """
    if frequency_range[1] > 0.5:
        raise ValueError(
            "Maximum value of frequencey_range is above niquist limit of 0.5."
        )
    offset_range = offset_range if offset_range is not None else width_range[1]

    # the sequence is first generated on a longer array with padding
    # this allows simpler adding of new values
    padding = int(np.ceil(width_range[1] * generation_width + 1))

    offsets = rng.uniform(-offset_range, offset_range, n_wave_packets)
    positions = rng.integers(padding, n_samples + padding, n_wave_packets)
    widths = rng.uniform(*width_range, n_wave_packets)
    amplitudes = rng.uniform(*amplitude_range, n_wave_packets)
    frequencies = rng.uniform(*frequency_range, n_wave_packets)
    phases = rng.uniform(0, 2 * np.pi, n_wave_packets)

    packet_properties = np.array(
        (offsets, positions, widths, amplitudes, frequencies, phases),
        dtype=np.float64,
    ).T

    sequence = np.zeros(n_samples + 2 * padding)
    for offsets, position, width, amplitude, frequency, phase in packet_properties:
        packet = generate_wave_packet(
            offsets, width, amplitude, frequency, phase, generation_width, peak_scaling
        )

        half_width = int(len(packet) / 2)
        position = int(position)
        sequence[position - half_width : position + half_width + 1] += packet

    return (sequence[padding:-padding], packet_properties)


class TestDataGenerator:
    """Generate simple test data for correlated noise mitigation techniques
    The channel count is implicitly defined by the shape of witness_noise_level

    :param witness_noise_level: Amplitude ratio of the sensor noise
                to the correlated noise in the witness sensor
    :param target_noise_level: Amplitude ratio of the sensor noise
                Scalar or 1D-vector for multiple sensors
                to the correlated noise in the target sensor
    :param transfer_function: Ratio between the amplitude in the target and witness signals
    :param sample_rate: The outputs are referenced
                to an ASD of 1/sqrt(Hz) if a sample rate is provided
    :param rng_seed: Optional value to generate the dataset based on a fixed seed for reproducible results.
                If not set, the randomly seeded global numpy rng is used.

    >>> import franc as fnc
    >>> # create data with two witness sensors with relative noise amplitudes of 0.1
    >>> tdg = fnc.evaluation.TestDataGenerator(witness_noise_level=[0.1, 0.1])
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
        rng_seed: int | None = None,
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

    def scaled_whitenoise(self, shape) -> NDArrayF:
        """Generate white noise with an ASD of one

        :param shape: shape of the new array

        :return: Array of white noise
        """
        return self.rng.normal(0, np.sqrt(self.sample_rate / 2), shape)

    def generate(self, n: int) -> tuple[NDArrayF, NDArrayF]:
        """Generate sequences of samples

        :param n: number of samples

        :return: witness signal, target signal
        """
        t_c = self.scaled_whitenoise(n)
        w_n = (
            self.scaled_whitenoise((len(self.witness_noise_level), n))
            * self.witness_noise_level[:, None]
        )
        t_n = self.scaled_whitenoise(n) * self.target_noise_level

        return (t_c + w_n) * self.transfer_function, (t_c + t_n)

    def generate_multiple(
        self, n: Sequence[int] | NDArrayU
    ) -> tuple[Sequence, Sequence]:
        """Generate sequences of samples

        :param n: Tuple with the length of the sequences

        :return: witness signals, target signals
        """
        witness = []
        target = []
        for w, t in (self.generate(n_i) for n_i in n):
            witness.append(w)
            target.append(t)
        return witness, target

    def dataset(
        self,
        n_condition: Sequence[int] | NDArray[np.uint],
        n_evaluation: Sequence[int] | NDArray[np.uint],
        generate_signal: bool = False,
        signal_amplitude: float = 1.0,
        sample_rate: float = 1.0,
        name: str | None = None,
    ) -> EvaluationDataset:
        """Generate an EvaluationDataset

        :param n_condition:  Sequence of integers indicating the number of conditioning samples generated per sample sequence
        :param n_evaluation: Number of evaluation samples
        :param sample_rate: (Optional) Sample rate for the generate EvaluationDataset
        :param name: (Optional) Specify the name of the EvaluationDataset

        Example:
        >>> # generate two sequences of 100 samples each of conditioning data and one 100 sample sequence of evaluation data
        >>> import franc as fnc
        >>> ds = fnc.evaluation.TestDataGenerator().dataset((100, 100), (100,))
        """
        # ensure the input parameters are 1D arrays of unsigned integers
        n_condition = np.array(n_condition, dtype=np.uint)
        n_evaluation = np.array(n_evaluation, dtype=np.uint)
        if len(n_condition.shape) != 1 or len(n_evaluation.shape) != 1:
            raise ValueError("Parameters must be sequences of integers. ")

        cond_data = self.generate_multiple(n_condition)
        eval_data = self.generate_multiple(n_evaluation)

        if generate_signal:
            cond_signal = [
                self.scaled_whitenoise(n) * signal_amplitude for n in n_condition
            ]
            eval_signal = [
                self.scaled_whitenoise(n) * signal_amplitude for n in n_evaluation
            ]
        else:
            cond_signal = None
            eval_signal = None

        return EvaluationDataset(
            sample_rate,
            cond_data[0],
            cond_data[1],
            eval_data[0],
            eval_data[1],
            cond_signal,
            eval_signal,
            name=name if name else "Unnamed",
        )

"""Collection of tools for the evaluation and testing of filters"""

from typing import Any, Optional, Tuple
from collections.abc import Sequence
from timeit import timeit
from dataclasses import dataclass
import struct

import numpy as np
from numpy.typing import NDArray

from .common import total_power
from ..common import hash_function
from ..filtering import FilterBase


@dataclass
class EvaluationDataset:
    """A representation of a dataset for the evaluation of noise mitigation methods.

    Provided sequences will be stored as immutable float64 numpy arrays.

    :param sample_rate: Sample rate in Hz
    :param witness_conditioning: witness channel data for the conditioning
        format: witness_conditioning[sequence_idx][channel_idx][sample_idx]
    :param target_conditioning: target channel data for the conditioning
        format: witness_conditioning[sequence_idx][sample_idx]
    :param witness_conditioning: witness channel data for the evaluation
    :param target_conditioning: target channel data for the evaluation
    :param name: (Optional) a string describing the dataset
    """

    sample_rate: float
    witness_conditioning: Sequence[Sequence[NDArray]]
    target_conditioning: Sequence[NDArray]
    witness_evaluation: Sequence[Sequence[NDArray]]
    target_evaluation: Sequence[NDArray]
    name: str

    def __init__(
        self,
        sample_rate: float,
        witness_conditioning: Sequence[Sequence[NDArray]],
        target_conditioning: Sequence[NDArray],
        witness_evaluation: Sequence[Sequence[NDArray]],
        target_evaluation: Sequence[NDArray],
        name: str = "Unnamed",
    ):
        self.sample_rate = float(sample_rate)
        self.witness_conditioning, self.target_conditioning = self._prepare_dataset(
            witness_conditioning, target_conditioning
        )
        self.witness_evaluation, self.target_evaluation = self._prepare_dataset(
            witness_evaluation, target_evaluation
        )
        self.name = name

        if not isinstance(name, str):
            raise ValueError("name must be a string")

    @staticmethod
    def _prepare_dataset(
        witness_inp, target_inp
    ) -> Tuple[Sequence[Sequence[NDArray]], Sequence[NDArray]]:
        """Convert input to immutable np.float64 arrays and check shape"""
        witness = tuple(
            tuple(np.array(j, dtype=np.float64, copy=True) for j in i)
            for i in witness_inp
        )
        target = tuple(np.array(i, dtype=np.float64, copy=True) for i in target_inp)

        # make numpy arrays immutable
        for w_sequence in witness:
            for channel in w_sequence:
                channel.flags.writeable = False
        for t_sequence in target:
            t_sequence.flags.writeable = False

        # check that sequence lengths match
        assert len(witness) > 0, "Creation of empty datasets is not allowd"
        assert len(target) == len(
            witness
        ), "Target and witness data must hold same number of sequences"
        for idx_sequence, (w, t) in enumerate(zip(witness, target)):
            assert len(w) > 0, "Creation of empty datasets is not allowed"

            for idx_channel, wi in enumerate(w):
                assert len(t) == len(
                    wi
                ), f"Witness channel {idx_channel} int sequence {idx_sequence} has {len(wi)} sequences, but target has {len(t)}!"
        return witness, target

    def get_min_sequence_len(self, separate=False) -> int | Tuple[int, int]:
        """Get the length of the shortest squence in the dataset"""
        min_cond = min(len(i) for i in self.target_conditioning)
        min_eval = min(len(i) for i in self.target_evaluation)
        if separate:
            return min_cond, min_eval
        return min(min_cond, min_eval)

    @staticmethod
    def _hash_wt_pair(witness: Sequence[Sequence], target: Sequence):
        """Calcualte a hash value for a pair of witness and target values"""
        hashes = 0
        for w_sequence in witness:
            for channel in w_sequence:
                hashes ^= hash_function(channel)
        for t_sequence in target:
            hashes ^= hash_function(t_sequence)
        return hashes

    def __hash__(self) -> int:
        # Python built-in hash() is randomly seeded, thus using a custom hash function is required
        hashes = hash_function(struct.pack("d", self.sample_rate) + self.name.encode())
        hashes ^= self._hash_wt_pair(
            self.witness_conditioning, self.target_conditioning
        )
        hashes ^= self._hash_wt_pair(self.witness_evaluation, self.target_evaluation)
        return hashes


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

    def generate_multiple(self, n: Sequence | NDArray) -> tuple[Sequence, Sequence]:
        """Generate sequences of samples

        :param N: Tuple with the length of the sequences

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
        n_condition: Sequence | NDArray,
        n_evaluation: Sequence | NDArray,
        sample_rate: float = 1.0,
        name: Optional[str] = None,
    ) -> EvaluationDataset:
        """Generate an EvaluationDataset

        :param n_condition:  Sequence of integers indicating the number of conditioning samples generated per sample sequence
        :param n_evaluation: Number of evaluation samples
        :param sample_rate: (Optional) Sample rate for the generate EvaluationDataset
        :param name: (Optional) Specify the name of the EvaluationDataset

        Example:
        >>> # generate two sequences of 100 samples each of conditioning data and one 100 sample sequence of evaluation data
        >>> import saftig as sg
        >>> ds = sg.evaluation.TestDataGenerator().dataset((100, 100), (100,))
        """
        # ensure the input parameters are 1D arrays of unsigned integers
        n_condition = np.array(n_condition, dtype=np.uint)
        n_evaluation = np.array(n_evaluation, dtype=np.uint)
        if len(n_condition.shape) != 1 or len(n_evaluation.shape) != 1:
            raise ValueError("Parameters must be sequences of integers. ")

        cond_data = self.generate_multiple(n_condition)
        eval_data = self.generate_multiple(n_evaluation)

        return EvaluationDataset(
            sample_rate,
            cond_data[0],
            cond_data[1],
            eval_data[0],
            eval_data[1],
            name=name if name else "Unnamed",
        )


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

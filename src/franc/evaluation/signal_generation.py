"""Tools to generate test data and EvaluatoinDatasets"""

from typing import Any
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .dataset import EvaluationDataset

NDArrayF = NDArray[np.floating]
NDArrayU = NDArray[np.uint]


class TestDataGenerator:
    """Generate simple test data for correlated noise mitigation techniques
    The channel count is implicitly defined by the shape of witness_noise_level

    :param witness_noise_level: Amplitude ratio of the sensor noise
                to the correlated noise in the witness sensor
                Scalar or 1D-vector for multiple sensors
    :param target_noise_level: Amplitude ratio of the sensor noise
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
        """Generate whitenoise with an ASD of one

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

        return EvaluationDataset(
            sample_rate,
            cond_data[0],
            cond_data[1],
            eval_data[0],
            eval_data[1],
            name=name if name else "Unnamed",
        )

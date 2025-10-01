"""A representation of a dataset for the evaluation of noise mitigation methods."""

from collections.abc import Sequence
from dataclasses import dataclass
import struct

import numpy as np
from numpy.typing import NDArray

from ..common import hash_function, hash_object_list, bytes2int, bytes2str

NDArrayF = NDArray[np.floating]


@dataclass
class EvaluationDataset:  # pylint: disable=too-many-instance-attributes
    """A representation of a dataset for the evaluation of noise mitigation methods.

    Provided sequences will be stored as immutable float64 numpy arrays.

    :param sample_rate: Sample rate in Hz
    :param witness_conditioning: witness channel data for the conditioning
        format: witness_conditioning[sequence_idx][channel_idx][sample_idx]
    :param target_conditioning: target channel data for the conditioning
        format: witness_conditioning[sequence_idx][sample_idx]
    :param witness_evaluation: witness channel data for the evaluation
    :param target_evaluation: target channel data for the evaluation
    :param signal_conditioning: (Optional) A signal that can be subtracted from the target for performance metrics
    :param signal_evaluation: (Optional) A signal that can be subtracted from the target for performance metrics
    :param name: (Optional) a string describing the dataset
    """

    sample_rate: float
    witness_conditioning: Sequence[Sequence[NDArrayF]]
    target_conditioning: Sequence[NDArrayF]
    witness_evaluation: Sequence[Sequence[NDArrayF]]
    target_evaluation: Sequence[NDArrayF]
    signal_conditioning: Sequence[NDArrayF] | None
    signal_evaluation: Sequence[NDArrayF] | None
    name: str
    target_unit: str  # unit of the target signal

    def __init__(
        self,
        sample_rate: float,
        witness_conditioning: Sequence[Sequence[NDArrayF]],
        target_conditioning: Sequence[NDArrayF],
        witness_evaluation: Sequence[Sequence[NDArrayF]],
        target_evaluation: Sequence[NDArrayF],
        signal_conditioning: Sequence[NDArrayF] | None = None,
        signal_evaluation: Sequence[NDArrayF] | None = None,
        name: str = "Unnamed",
        target_unit: str = "1",
    ):
        self.sample_rate = float(sample_rate)
        (
            self.witness_conditioning,
            self.target_conditioning,
            self.signal_conditioning,
        ) = self._prepare_dataset(
            witness_conditioning, target_conditioning, signal_conditioning
        )
        self.witness_evaluation, self.target_evaluation, self.signal_evaluation = (
            self._prepare_dataset(
                witness_evaluation, target_evaluation, signal_evaluation
            )
        )
        self.name = name
        self.target_unit = target_unit

        if not isinstance(name, str):
            raise ValueError("name must be a string")

    @staticmethod
    def _prepare_dataset(
        witness_inp: Sequence[Sequence[NDArrayF]],
        target_inp: Sequence[NDArrayF],
        signal_inp: Sequence[NDArrayF] | None = None,
    ) -> tuple[
        Sequence[Sequence[NDArrayF]], Sequence[NDArrayF], Sequence[NDArrayF] | None
    ]:
        """Convert input to immutable np.float64 arrays and check shape"""
        witness = tuple(
            tuple(np.array(j, dtype=np.float64, copy=True) for j in i)
            for i in witness_inp
        )
        target = tuple(np.array(i, dtype=np.float64, copy=True) for i in target_inp)
        if signal_inp is not None:
            signal = tuple(np.array(i, dtype=np.float64, copy=True) for i in signal_inp)
        else:
            signal = None

        # make numpy arrays immutable
        for w_sequence in witness:
            for channel in w_sequence:
                channel.flags.writeable = False
        for t_sequence in target:
            t_sequence.flags.writeable = False
        if signal is not None:
            for s_sequence in signal:
                s_sequence.flags.writeable = False

        # check that sequence lengths match
        checked_signals = [("target", target)]
        if signal is not None:
            checked_signals.append(("signal", signal))

        for input_name, input_data in checked_signals:
            assert len(witness) > 0, "Creation of empty datasets is not allowed"
            assert len(input_data) == len(
                witness
            ), f"Target and {input_name} data must hold same number of sequences"
            for idx_sequence, (w, t) in enumerate(zip(witness, input_data)):
                assert len(w) > 0, "Creation of empty datasets is not allowed"

                for idx_channel, wi in enumerate(w):
                    assert len(t) == len(
                        wi
                    ), f"Witness channel {idx_channel} in sequence {idx_sequence} has {len(wi)} length, but {input_name} has {len(t)}!"
        return witness, target, signal

    @property
    def channel_count(self) -> int:
        """Number of witness channels"""
        return len(self.witness_conditioning[0])

    def sequence_lengths(self, which: str) -> list[int]:
        """Returns the lengths of the evaluation or conditioning sequences

        :param which: A string selecting which sequence will be analyzed.
            Must be one of the following values: "cond", "conditioning", "eval", or "evaluation".
        """
        if which in {"conditioning", "cond"}:
            lengths = [len(sequence) for sequence in self.target_conditioning]
        elif which in {"evaluation", "eval"}:
            lengths = [len(sequence) for sequence in self.target_evaluation]
        else:
            raise ValueError("which must be either 'conditioning' or 'evaluation'")
        return lengths

    def get_min_sequence_len(self, separate: bool = False) -> int | tuple[int, int]:
        """Get the length of the shortest sequence in the dataset

        :param separate: If True, returns the minimum separately for conditioning and evaluation data.
        """
        min_cond = min(len(i) for i in self.target_conditioning)
        min_eval = min(len(i) for i in self.target_evaluation)
        if separate:
            return min_cond, min_eval
        return min(min_cond, min_eval)

    @staticmethod
    def _hash_wts_data(
        witness: Sequence[Sequence[NDArrayF]],
        target: Sequence[NDArrayF],
        signal: Sequence[NDArrayF] | None = None,
    ):
        """Calculate a hash value for a set of witness, target, signal data"""
        hashes = b""
        for w_sequence in witness:
            hashes += hash_object_list(w_sequence)
        hashes += hash_object_list(target)
        if signal is not None:
            hashes += hash_object_list(signal)
        return hashes

    def hash_bytes(self) -> bytes:
        """return a hash over the dataset data as a bytes object"""
        # Python built-in hash() is randomly seeded, thus using a custom hash function is required
        hashes = hash_function(struct.pack("d", self.sample_rate) + self.name.encode())
        hashes += self._hash_wts_data(
            self.witness_conditioning,
            self.target_conditioning,
            self.signal_conditioning,
        )
        hashes += self._hash_wts_data(
            self.witness_evaluation, self.target_evaluation, self.signal_evaluation
        )
        return hash_function(hashes)

    def __hash__(self) -> int:
        return bytes2int(self.hash_bytes())

    def description(self) -> str:
        """Generate a description of the dataset"""
        description = self.name + f" (Hash: {bytes2str(self.hash_bytes())})\n"
        description += f"Sample rate: {self.sample_rate} Hz\n"
        description += f"{self.channel_count} witness channels\n"
        for label in ["conditioning", "evaluation"]:
            min_samples = min(self.sequence_lengths(label))
            max_samples = max(self.sequence_lengths(label))
            if max_samples == min_samples:
                description += f"{len(self.target_conditioning)} {label} sequence(s) ({min(self.sequence_lengths('cond'))} samples each)\n"
            else:
                description += f"{len(self.target_conditioning)} {label} sequence(s) (min samples: {min_samples}, max samples: {max_samples})\n"
        return description

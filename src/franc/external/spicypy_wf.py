"""A wrapper for spicypy.signal.WienerFilter with the franc.common.FilterBase interface.
This is intended to allow comparisons between the implementations.
"""

from typing import overload
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
import warnings
import numpy as np
from numpy.typing import NDArray
import spicypy

from ..filtering.common import FilterBase, handle_from_dict


@dataclass
class SpicypyWienerFilter(FilterBase):
    """A wrapper for the spicypy WF implementation

    :param n_filter: Length of the FIR filter
        (how many samples are in the input window per output sample)
    :param idx_target: Position of the prediction
    :param n_channel: Number of witness sensor channels

    >>> import franc as fnc
    >>> n_filter = 10
    >>> witness, target = fnc.evaluation.TestDataGenerator(0.1).generate(int(1e3))
    >>> filt = fnc.external.SpicypyWienerFilter(1, n_filter, 0)
    >>> sp_filt = filt.condition(witness, target)
    >>> prediction = filt.apply(witness, target) # apply to training data

    """

    supports_multi_sequence = False
    filter_name = "SpicypyWF"

    @handle_from_dict
    def __init__(
        self,
        n_channel: int,
        n_filter: int,
        idx_target: int,
    ):
        super().__init__(n_channel, n_filter, idx_target)

        self.conditioned = False
        self.filter_state: spicypy.signal.WienerFilter | None

    @staticmethod
    def supports_saving_loading():
        """Indicates whether saving and loading is supported."""
        return False

    @overload
    @staticmethod
    def make_spicypy_time_series(
        witness: Sequence | NDArray,
        target: None,
        sample_rate: float = 1.0,
    ) -> tuple[Sequence[spicypy.signal.TimeSeries], None]: ...

    @overload
    @staticmethod
    def make_spicypy_time_series(
        witness: Sequence | NDArray,
        target: Sequence | NDArray,
        sample_rate: float = 1.0,
    ) -> tuple[Sequence[spicypy.signal.TimeSeries], spicypy.signal.TimeSeries]: ...

    @staticmethod
    def make_spicypy_time_series(
        witness: Sequence | NDArray,
        target: Sequence | NDArray | None,
        sample_rate: float = 1.0,
    ) -> tuple[Sequence[spicypy.signal.TimeSeries], spicypy.signal.TimeSeries | None]:
        """Convert the given witness and target signals to the format required by spicypy.

        :param witness: Witness sensor data
        :param target: Target sensor data
        :param sample_rate: The sample rate of the time series
        """
        witness_ts = [
            spicypy.signal.TimeSeries(wi, sample_rate=sample_rate) for wi in witness
        ]
        if target is None:
            target_ts = None
        else:
            target_ts = spicypy.signal.TimeSeries(target, sample_rate=sample_rate)
        return witness_ts, target_ts

    def condition(
        self,
        witness: Sequence | NDArray,
        target: Sequence | NDArray,
        sample_rate: float = 1.0,
        use_multiprocessing: bool = False,
    ) -> spicypy.signal.WienerFilter:
        """Use an input dataset to condition the filter

        :param witness: Witness sensor data
        :param target: Target sensor data
        :param sample_rate: The sample rate of the time series
        """
        witness, target = self.check_data_dimensions(witness, target)
        assert (
            self.n_filter <= target.shape[0]
        ), "Input data must be at least one filter length"

        witness, target = self.make_spicypy_time_series(witness, target, sample_rate)
        self.filter_state = spicypy.signal.WienerFilter(
            target,
            witness,
            n_taps=self.n_filter,
            use_multiprocessing=use_multiprocessing,
            use_norm_factor=False,
        )

        # spicypy.signal.WienerFilter uses a lot of print statements
        # this stops it from spamming stdout
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with redirect_stdout(StringIO()):
                self.filter_state.create_filters()

        self.conditioned = True

        return self.filter_state

    def condition_multi_sequence(
        self,
        witness: Sequence | Sequence[Sequence] | NDArray,
        target: Sequence | NDArray,
    ):
        """Not supported"""
        del witness, target  # mark parameters as unused
        raise NotImplementedError("Mulit sequence input is not supported.")

    def apply(
        self,
        witness: Sequence | NDArray,
        target: Sequence | NDArray | None = None,
        pad: bool = True,
        update_state: bool = False,
        sample_rate: float = 1.0,
    ) -> NDArray:
        """Apply the filter to input data

        :param witness: Witness sensor data
        :param target: Target sensor data (is ignored)
        :param pad: if True, apply padding zeros so that the length matches the target signal
        :param update_state: ignored

        :return: prediction
        """
        if target is None:
            raise ValueError("A target signal must be supplied")
        if not self.conditioned:
            raise RuntimeError(
                "This filter must be conditioned before calling apply()!"
            )

        witness, _target = self.check_data_dimensions(witness, target)
        assert (
            self.filter_state is not None
        ), "The filter must be conditioned before calling apply()"

        witness, target = self.make_spicypy_time_series(witness, target, sample_rate)
        prediction = self.filter_state.apply(witness, zero_padding=pad)

        if not pad:
            # append a zero to match length in case no filter is used
            prediction = np.concatenate([prediction, [0]])

        return prediction

    def apply_multi_sequence(
        self,
        witness: Sequence | NDArray,
        target: Sequence | NDArray | None,
        pad: bool = True,
        update_state: bool = False,
    ) -> Sequence[NDArray]:
        """Not supported"""
        del witness, target, pad, update_state  # mark parameters as unused
        raise NotImplementedError("Mulit sequence input is not supported.")

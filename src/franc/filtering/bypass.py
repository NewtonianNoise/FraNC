"""A filtering method that does conditioning"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .common import FilterBase, handle_from_dict


@dataclass
class BypassFilter(FilterBase):
    """Implementation of a filtering method that just predicts zeros.

    Intended for testing and to get apply metrics to input data data.

    :param n_channel: Number of witness sensor channels
    all other parameters are ignored

    >>> import franc as fnc
    >>> n_filter = 128
    >>> witness, target = fnc.evaluation.TestDataGenerator(0.1).generate(int(1e5))
    >>> filt = fnc.filtering.BypassFilter(1, n_filter, 0)
    >>> filt.condition(witness, target)
    >>> prediction = filt.apply(witness, target) # check on the data used for conditioning
    >>> float(sum(prediction))
    0.0

    """

    #: The FIR coefficients of the WF
    filter_name: str = "Bypass"

    @handle_from_dict
    def __init__(
        self,
        n_channel: int,
        n_filter: int = 1,
        idx_target: int = 0,
    ):
        super().__init__(n_channel, n_filter, idx_target)
        self.requires_apply_target = False

    def condition_multi_sequence(
        self,
        witness: Sequence | Sequence[Sequence] | NDArray,
        target: Sequence | NDArray,
    ) -> None:
        """Use an input dataset to condition the filter

        :param witness: Witness sensor data
        :param target: Target sensor data
        """
        del witness, target

    def apply_multi_sequence(
        self,
        witness: Sequence | NDArray,
        target: Sequence | NDArray | None = None,
        pad: bool = True,
        update_state: bool = False,
    ) -> list[NDArray]:
        """Apply the filter to input data

        :param witness: Witness sensor data
        :param target: Target sensor data (is ignored)
        :param pad: if True, apply padding zeros so that the length matches the target signal
        :param update_state: ignored

        :return: prediction
        """
        del update_state  # mark as unused

        witness, target = self.check_data_dimensions_multi_sequence(witness, target)

        predictions: list = []
        for w_sequence in witness:
            prediction_sequence = np.zeros(
                max(len(w_sequence[0]) - self.n_filter + 1, 0)
            )
            if pad:
                prediction_sequence = np.concatenate(
                    [
                        np.zeros(self.n_filter - 1 - self.idx_target),
                        prediction_sequence,
                        np.zeros(self.idx_target),
                    ]
                )
            predictions.append(prediction_sequence)
        return predictions

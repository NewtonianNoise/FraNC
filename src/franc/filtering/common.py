"""Shared functionality for all filtering techniques"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..evaluation import FilterInterface, make_2d_array, handle_from_dict


@dataclass
class FilterBase(FilterInterface):
    """common interface definition for Filter implementations

    :param n_filter: Length of the FIR filter
                     (how many samples are in the input window per output sample)
    :param idx_target: Position of the prediction
    :param n_channel: Number of witness sensor channels
    """

    n_filter: int
    idx_target: int
    default_args = [None, None, None]

    def __init__(self, n_channel: int, n_filter: int, idx_target: int, _from_dict=None):
        super().__init__(n_channel, _from_dict=_from_dict)
        self.n_filter = n_filter
        self.idx_target = idx_target

        if self.n_filter <= 0:
            raise ValueError("n_filter must be a positive integer")
        if self.n_channel <= 0:
            raise ValueError("n_channel must be a positive integer")
        if not 0 <= self.idx_target < self.n_filter:
            raise ValueError(
                "idx_target must not be negative and smaller than n_filter"
            )

    @property
    def method_filename_part(self) -> str:
        """string that can be used in a file name"""
        return f"{self.filter_name}_{self.n_filter}_{self.n_channel}_{self.idx_target}"


def pad_prediction(
    prediction: Sequence | NDArray,
    n_filter: int,
    idx_target: int,
    trailing_padding: int = 0,
) -> NDArray:
    """Pad a prediction with zeros so that its length matches the target signal it was
    calculated from.

    :param prediction: Prediction calculated with the given n_filter/idx_target
    :param n_filter: Length of the FIR filter used to calculate the prediction
    :param idx_target: Position of the prediction
    :param trailing_padding: Extra zeros appended at the end, used when the prediction
        ends early (e.g. an incomplete final block)
    """
    return np.concatenate(
        [
            np.zeros(n_filter - 1 - idx_target),
            prediction,
            np.zeros(idx_target + trailing_padding),
        ]
    )


class AdaptiveFilterBase(FilterBase):
    """Shared interface for filters that continuously adapt their state while apply() runs.

    Subclasses only need to provide a stateful apply(); condition() and the
    multi-sequence variants are expressed in terms of it.
    """

    def condition(
        self,
        witness: Sequence | NDArray,
        target: Sequence | NDArray,
    ) -> None:
        """Use an input dataset to condition the filter

        :param witness: Witness sensor data
        :param target: Target sensor data
        """
        _ = self.apply(witness, target, update_state=True)

    def condition_multi_sequence(
        self,
        witness: Sequence | Sequence[Sequence] | NDArray,
        target: Sequence | NDArray,
    ) -> None:
        """Similar to condition(), but expects multiple sequences"""
        for w, t in zip(witness, target):
            self.condition(w, t)

    def apply_multi_sequence(
        self,
        witness: Sequence | NDArray,
        target: Sequence | NDArray | None = None,
        pad: bool = True,
        update_state: bool = False,
    ) -> Sequence[NDArray]:
        if target is None:
            raise ValueError("A target signal must be supplied")
        return [self.apply(w, t, pad, update_state) for w, t in zip(witness, target)]


# include make_2d_array and handle_from_dict so all objects that
# are potentially required to create a compatible filter are in one place
__all__ = [
    "FilterBase",
    "AdaptiveFilterBase",
    "pad_prediction",
    "make_2d_array",
    "handle_from_dict",
]

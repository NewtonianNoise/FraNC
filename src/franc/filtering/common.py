"""Shared functionality for all filtering techniques"""

from __future__ import annotations

from dataclasses import dataclass

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


# include make_2d_array and handle_from_dict so all objects that
# are potentially required to create a compatible filter are in one place
__all__ = ["FilterBase", "make_2d_array", "handle_from_dict"]

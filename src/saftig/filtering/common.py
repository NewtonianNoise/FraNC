"""Shared functionality for all other modules"""

from typing import Optional
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def make_2d_array(A: Sequence | Sequence[Sequence] | NDArray) -> NDArray:
    """add a dimension to 1D arrays and leave 2D arrays as they are
    This is intended to allow 1D array input for single channel application

    :param A: input array

    :return: exteneded array

    :raises: ValueError if the input shape is not compatible

    >>> make_2d_array([1, 2])
    array([[1, 2]])

    >>> make_2d_array([[1, 2], [3, 4]])
    array([[1, 2],
           [3, 4]])

    """
    A_npy = np.array(A)
    if len(A_npy.shape) == 1:
        return np.array([A_npy])
    if len(A_npy.shape) == 2:
        return A_npy
    raise ValueError("Input must be 1D or 2D array")


class FilterBase:
    """common interface definition for Filter implementations

    :param n_filter: Length of the FIR filter
                     (how many samples are in the input window per output sample)
    :param idx_target: Position of the prediction
    :param n_channel: Number of witness sensor channels
    """

    filter_name: str | None = None

    def __init__(self, n_filter: int, idx_target: int, n_channel: int = 1):
        self.n_filter = n_filter
        self.n_channel = n_channel
        self.idx_target = idx_target

        assert self.n_filter > 0, "n_filter must be a positive integer"
        assert self.n_channel > 0, "n_filter must be a positive integer"
        assert (
            self.idx_target >= 0 and self.idx_target < self.n_filter
        ), "idx_target must not be negative and smaller than n_filter"
        assert self.filter_name is not None, "BaseFilter childs must set their name"

        self.requries_apply_target = True

    def condition(
        self,
        witness: Sequence | Sequence[Sequence],
        target: Sequence,
    ) -> None:
        """Use an input dataset to condition the filter

        :param witness: Witness sensor data
        :param target: Target sensor data
        """
        raise NotImplementedError(
            "This function must be implemented by the child class!"
        )

    def apply(
        self,
        witness: Sequence | NDArray,
        target: Sequence | NDArray,
        pad: bool = True,
        update_state: bool = False,
    ) -> NDArray:
        """Apply the filter to input data

        :param witness: Witness sensor data (1D or 2D array)
        :param target: Target sensor data (1D array)
        :param pad: if True, apply padding zeros so that the length matches the target signal
        :param update_state: if True, the filter state will be changed. If false, the filter state will remain

        :return: prediction
        """
        raise NotImplementedError(
            "This function must be implemented by the child class!"
        )

    def check_data_dimensions(
        self,
        witness: Sequence | NDArray,
        target: Optional[Sequence | NDArray] = None,
    ) -> tuple[NDArray, NDArray]:
        """Check the dimensions of the provided input data and apply make_2d_array()

        :param witness: Witness sensor data
        :param target: Target sensor data

        :return: data as (target, witness)

        :raises: AssertionError
        """
        target_npy = np.array(target)
        witness_npy = make_2d_array(witness)
        assert (
            witness_npy.shape[0] == self.n_channel
        ), "witness data shape does not match configured channel count"

        if self.requries_apply_target:
            assert (
                target is None or target_npy.shape[0] == witness_npy.shape[1]
            ), "Missmatch between target and witness data shapes"

        return witness_npy, target_npy

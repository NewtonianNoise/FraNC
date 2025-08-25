"""Clasical static Wiener filter"""

from typing import Tuple, Optional
from collections.abc import Sequence
from dataclasses import dataclass
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate

from .common import FilterBase, make_2d_array, handle_from_dict


def mean_cross_correlation_offset(
    A: Sequence | NDArray, B: Sequence | NDArray, N: int, offset: int
) -> Sequence:
    """estimate the cross-correlation between A and B
    :param A: First input array
    :param B: Second input array
    :param N: Number of steps to test. Defines length of output
    :param offset: Offset for the cross correlation
    """
    assert len(A) == len(B)
    assert offset < N

    if offset < N - 1:
        return correlate(A, B[offset : -N + 1 + offset], mode="valid")
    return correlate(A, B[offset:], mode="valid")


def wf_calculate(
    witness: Sequence | NDArray,
    target: Sequence | NDArray,
    n_filter: int,
    idx_target: int = 0,
) -> Tuple[NDArray, bool]:
    """caluclate the FIR coefficients for a wiener filter

    :param witness: Witness sensor data
    :param witness: Target sensor data
    :param n_filter: Length of the FIR filter (how many samples are in the input window per output sample)
    :param idx_target: offset of the prediction relative to the end of the array

    :return: filter coefficients, full_rank (bool)
    """
    target_npy: NDArray = np.array(target)
    witness_npy: NDArray = make_2d_array(witness)
    assert (
        witness_npy.shape[1] == target_npy.shape[0]
    ), "Missmatch between witness_npy and target_npy data shape"
    assert (
        n_filter <= target_npy.shape[0]
    ), "Input data must be at least one filter length"

    # calculate input autocorrelation and cross-correlation to target_npy
    # R_ws[channel, time]
    R_ws = np.array(
        [
            mean_cross_correlation_offset(target_npy, A, n_filter, idx_target)
            for A in witness_npy
        ]
    ).flatten(order="C")

    def calc_r_matrix(A, B, n_filter):
        """calculate the cross correlation matrix of a and b"""
        cc = correlate(A, B[: -n_filter + 1], mode="valid")
        return np.array(
            [np.concatenate([cc[i::-1], cc[1 : n_filter - i]]) for i in range(n_filter)]
        )

    def calc_r_matrix_symmetric(A, B, n_filter):
        """calculate the cross correlation matrix of a and b and average positive and negative lag
        to make the result symmetric (as is expected for an autocorrelation)
        """
        cc = correlate(A, B[n_filter:-n_filter], mode="valid")
        cc = np.concatenate(
            [[cc[n_filter]], (cc[n_filter + 1 :] + cc[n_filter - 1 :: -1]) / 2]
        )
        return np.array(
            [np.concatenate([cc[i::-1], cc[1 : n_filter - i]]) for i in range(n_filter)]
        )

    if (
        len(target_npy) >= 3 * n_filter
    ):  # using both sides is only possible if enough data is provided
        R_ww = np.block(
            [
                [calc_r_matrix_symmetric(A, B, n_filter) for B in witness_npy]
                for A in witness_npy
            ]
        )
    else:
        R_ww = np.block(
            [[calc_r_matrix(A, B, n_filter) for B in witness_npy] for A in witness_npy]
        )

    # calculate pseudo-inverse correlation matrix of inputs and the filter coefficients
    # for some reason the scipy.linalg implementations were extremely slow on white noise test case => using numpy
    full_rank = bool(np.linalg.matrix_rank(R_ww, hermitian=True) == len(R_ww[0]))
    R_ww_inv = np.linalg.pinv(R_ww, hermitian=True)
    WFC = R_ww_inv.dot(np.array(R_ws))

    # unwrap into seperate FIR filters
    WFC = WFC.reshape((len(witness), n_filter))
    WFC = np.array([np.flip(i) for i in WFC])

    assert (
        len(WFC[0]) == n_filter
    ), "input data was to short resulting in an incompatible filter"

    return WFC, full_rank


def wf_apply(
    WFC: Sequence | NDArray,
    witness: Sequence | NDArray,
) -> NDArray:
    """apply the WF to witness data

    :param witness: Witness sensor data
    :param target: Target sensor data

    :return: prediction
    """
    assert len(witness[0]) >= len(WFC[0]), "Input minimum lenght is one filter length"
    witness_npy = np.array(witness).astype(np.longdouble)
    return np.sum(
        [correlate(A, WF, mode="valid") for A, WF in zip(witness_npy, WFC)], axis=0
    )


@dataclass
class WienerFilter(FilterBase):
    """Satic Wiener filter implementation

    :param n_filter: Length of the FIR filter (how many samples are in the input window per output sample)
    :param idx_target: Position of the prediction
    :param n_channel: Number of witness sensor channels

    >>> import saftig as sg
    >>> n_filter = 128
    >>> witness, target = sg.evaluation.TestDataGenerator(0.1).generate(int(1e5))
    >>> filt = sg.filtering.WienerFilter(n_filter, 0, 1)
    >>> _coefficients, full_rank = filt.condition(witness, target)
    >>> full_rank
    True
    >>> prediction = filt.apply(witness, target) # check on the data used for conditioning
    >>> residual_rms = sg.evaluation.rms(target-prediction)
    >>> residual_rms > 0.05 and residual_rms < 0.15 # the expected RMS in this test scenario is 0.1
    True

    """

    #: The FIR coefficients of the WF
    filter_state: NDArray | None = None
    filter_name: str = "WF"

    @handle_from_dict
    def __init__(
        self,
        n_filter: int,
        idx_target: int,
        n_channel: int = 1,
    ):
        super().__init__(n_filter, idx_target, n_channel)
        self.requires_apply_target = False

    def condition(
        self,
        witness: Sequence,
        target: Sequence,
    ):
        """Use an input dataset to condition the filter

        :param witness: Witness sensor data
        :param target: Target sensor data
        """
        witness_npy, target_npy = self.check_data_dimensions(witness, target)

        self.filter_state, full_rank = wf_calculate(
            witness_npy, target_npy, self.n_filter, idx_target=self.idx_target
        )

        if not full_rank:
            warn("Warning: Filter is not of full rank", RuntimeWarning)
        return self.filter_state, full_rank

    def apply(
        self,
        witness: Sequence | NDArray,
        target: Optional[Sequence | NDArray] = None,
        pad: bool = True,
        update_state: bool = False,
    ) -> NDArray:
        """Apply the filter to input data

        :param witness: Witness sensor data
        :param target: Target sensor data (is ignored)
        :param pad: if True, apply padding zeros so that the length matches the target signal
        :param update_state: ignored

        :return: prediction
        """
        witness, target = self.check_data_dimensions(witness, target)
        if self.filter_state is None:
            raise RuntimeError(
                "The filter must be conditioned before apply() can be used."
            )

        prediction = wf_apply(self.filter_state, witness)
        if pad:
            prediction = np.concatenate(
                [
                    np.zeros(self.n_filter - 1 - self.idx_target),
                    prediction,
                    np.zeros(self.idx_target),
                ]
            )
        return prediction

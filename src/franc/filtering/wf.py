"""Classical static Wiener filter"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from warnings import warn

import numpy as np
from numpy.typing import NDArray
import scipy
from scipy.signal import correlate

from .common import FilterBase, make_2d_array, handle_from_dict, pad_prediction


def mean_cross_correlation_offset(
    A: Sequence | NDArray, B: Sequence | NDArray, N: int, offset: int
) -> NDArray:
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


def wf_calculate_correlations(
    witness: Sequence | NDArray,
    target: Sequence | NDArray,
    n_filter: int,
    idx_target: int = 0,
) -> tuple[NDArray, NDArray]:
    """calculate the cross- and auto-correlation statistics of the Wiener-Hopf equations
    for a single witness/target sequence pair

    Statistics of multiple sequences can be combined by summing R_ws and R_ww of each
    sequence before solving the resulting equations with wf_solve().

    :param witness: Witness sensor data
    :param target: Target sensor data
    :param n_filter: Length of the FIR filter (how many samples are in the input window per output sample)
    :param idx_target: offset of the prediction relative to the end of the array

    :return: cross-correlation vector R_ws, auto-correlation matrix R_ww
    """
    target_npy: NDArray = np.array(target)
    witness_npy: NDArray = make_2d_array(witness)
    if witness_npy.shape[1] != target_npy.shape[0]:
        raise ValueError("Missmatch between witness_npy and target_npy data shape")
    if n_filter > target_npy.shape[0]:
        raise ValueError("Input data must be at least one filter length")

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

    return R_ws, R_ww


def invert_R_ww(
    R_ww: Sequence | NDArray,
    inversion_method: str = "np_pinv",
    regularization: float = 0.0,
) -> tuple[NDArray, bool]:
    """invert an auto-correlation matrix of the Wiener-Hopf equations, as computed by
    wf_calculate_correlations() or bwf_calculate_correlations()

    :param R_ww: auto-correlation matrix (or the sum of several, when pooling statistics
        of multiple sequences)
    :param inversion_method: matrix inversion method used for filter coefficient calculation. Check WienerFilter class dock string for possible values
    :param regularization: Tikhonov regularization strength added to the diagonal of the input
        autocorrelation matrix before inversion. 0 disables regularization. Larger values trade
        fit accuracy for a better conditioned, more stable filter

    :return: inverted auto-correlation matrix, full_rank (bool)
    """
    R_ww = np.array(R_ww)

    if regularization:
        R_ww = R_ww + regularization * np.eye(len(R_ww))

    # calculate pseudo-inverse correlation matrix of inputs and the filter coefficients
    # for some reason the scipy.linalg implementations were extremely slow on white noise test case => using numpy
    matrix_rank = np.linalg.matrix_rank(R_ww, hermitian=True)
    full_rank = bool(matrix_rank == len(R_ww[0]))
    try:
        if inversion_method == "np_pinv":
            R_ww_inv = np.linalg.pinv(R_ww, hermitian=True)
        elif inversion_method == "np_inv":
            R_ww_inv = np.linalg.inv(R_ww)
        elif inversion_method == "sp_pinv":
            R_ww_inv = scipy.linalg.pinvh(R_ww)
        elif inversion_method == "sp_inv":
            R_ww_inv = scipy.linalg.inv(R_ww)
        else:
            raise ValueError(f"Undefined inversion_method value {inversion_method}")
    except np.linalg.LinAlgError as e:
        print(
            f"{len(R_ww[0])}x{len(R_ww[0])} input cross correlation matrix is of rank {matrix_rank}"
        )
        print(R_ww)
        raise e

    return R_ww_inv, full_rank


def wf_solve(
    R_ws: Sequence | NDArray,
    R_ww: Sequence | NDArray,
    n_filter: int,
    n_channel: int,
    inversion_method: str = "np_pinv",
    regularization: float = 0.0,
) -> tuple[NDArray, bool]:
    """solve the Wiener-Hopf equations R_ww @ w = R_ws for the FIR filter coefficients

    :param R_ws: cross-correlation vector, as returned by wf_calculate_correlations()
    :param R_ww: auto-correlation matrix, as returned by wf_calculate_correlations()
        (or the sum of several, when pooling statistics of multiple sequences)
    :param n_filter: Length of the FIR filter (how many samples are in the input window per output sample)
    :param n_channel: Number of witness sensor channels
    :param inversion_method: matrix inversion method used for filter coefficient calculation. Check WienerFilter class dock string for possible values
    :param regularization: Tikhonov regularization strength added to the diagonal of the input
        autocorrelation matrix before inversion. 0 disables regularization. Larger values trade
        fit accuracy for a better conditioned, more stable filter

    :return: filter coefficients, full_rank (bool)
    """
    R_ww_inv, full_rank = invert_R_ww(R_ww, inversion_method, regularization)
    WFC = R_ww_inv.dot(np.array(R_ws))

    # unwrap into seperate FIR filters
    WFC = WFC.reshape((n_channel, n_filter))
    WFC = np.array([np.flip(i) for i in WFC])

    assert (
        len(WFC[0]) == n_filter
    ), "input data was to short resulting in an incompatible filter"

    return WFC, full_rank


def wf_calculate(
    witness: Sequence | NDArray,
    target: Sequence | NDArray,
    n_filter: int,
    idx_target: int = 0,
    inversion_method: str = "np_pinv",
    regularization: float = 0.0,
) -> tuple[NDArray, bool]:
    """caluclate the FIR coefficients for a wiener filter

    :param witness: Witness sensor data
    :param witness: Target sensor data
    :param n_filter: Length of the FIR filter (how many samples are in the input window per output sample)
    :param idx_target: offset of the prediction relative to the end of the array
    :param inversion_method: matrix inversion method used for filter coefficient calculation. Check WienerFilter class dock string for possible values
    :param regularization: Tikhonov regularization strength added to the diagonal of the input
        autocorrelation matrix before inversion. 0 disables regularization. Larger values trade
        fit accuracy for a better conditioned, more stable filter


    :return: filter coefficients, full_rank (bool)
    """
    n_channel = make_2d_array(witness).shape[0]
    R_ws, R_ww = wf_calculate_correlations(witness, target, n_filter, idx_target)
    return wf_solve(
        R_ws,
        R_ww,
        n_filter,
        n_channel,
        inversion_method=inversion_method,
        regularization=regularization,
    )


def wf_apply(
    WFC: Sequence | NDArray,
    witness: Sequence | NDArray,
) -> NDArray:
    """apply the WF to witness data

    :param witness: Witness sensor data
    :param target: Target sensor data

    :return: prediction
    """
    witness_npy = make_2d_array(witness).astype(np.float64)
    if witness_npy.shape[1] < len(WFC[0]):
        raise ValueError("Input minimum lenght is one filter length")
    return np.sum(
        [correlate(A, WF, mode="valid") for A, WF in zip(witness_npy, WFC)], axis=0
    )


@dataclass
class WienerFilter(FilterBase):
    """Satic Wiener filter implementation

    :param n_channel: Number of witness sensor channels
    :param n_filter: Length of the FIR filter (how many samples are in the input window per output sample)
    :param idx_target: Position of the prediction
    :param inversion_method: Matrix inversion method used for filter coefficient calculation
        'np_pinv' np.linalg.pinv()
        'np_inv' np.linalg.inv()
        'sp_pinv' scipy.linalg.pinv()
        'sp_inv' scipy.linalg.inv()
    :param regularization: Tikhonov regularization strength added to the diagonal of the input
        autocorrelation matrix before inversion. 0 disables regularization. Larger values trade
        fit accuracy for a better conditioned, more stable filter

    >>> import franc as fnc
    >>> n_filter = 128
    >>> witness, target = fnc.evaluation.TestDataGenerator(0.1).generate(int(1e5))
    >>> filt = fnc.filtering.WienerFilter(1, n_filter, 0)
    >>> _coefficients, full_rank = filt.condition(witness, target)
    >>> full_rank
    True
    >>> prediction = filt.apply(witness, target) # check on the data used for conditioning
    >>> residual_rms = fnc.evaluation.rms(target-prediction)
    >>> residual_rms > 0.05 and residual_rms < 0.15 # the expected RMS in this test scenario is 0.1
    True

    """

    #: The FIR coefficients of the WF
    filter_state: NDArray | None = None
    filter_name: str = "WF"
    inversion_method: str = "np_pinv"
    regularization: float = 0.0

    @handle_from_dict
    def __init__(
        self,
        n_channel: int,
        n_filter: int,
        idx_target: int,
        inversion_method: str = "np_pinv",
        regularization: float = 0.0,
    ):
        super().__init__(n_channel, n_filter, idx_target)
        self.requires_apply_target = False
        self.inversion_method = inversion_method
        self.regularization = regularization

    def condition_multi_sequence(
        self,
        witness: Sequence | Sequence[Sequence] | NDArray,
        target: Sequence | NDArray,
    ) -> tuple[NDArray, bool]:
        """Use an input dataset to condition the filter

        The cross- and auto-correlation statistics of all sequences are pooled into a
        single set of Wiener-Hopf equations, which are then solved once. This is both
        cheaper and more accurate than solving per sequence and averaging the resulting
        filters, since it needs only one matrix inversion and lets data from all
        sequences jointly constrain the solution.

        :param witness: Witness sensor data
        :param target: Target sensor data
        """
        witness_npy, target_npy = self.check_data_dimensions_multi_sequence(
            witness, target
        )

        R_ws_total: NDArray | None = None
        R_ww_total: NDArray | None = None
        for witness_npy_i, target_npy_i in zip(witness_npy, target_npy):
            R_ws, R_ww = wf_calculate_correlations(
                witness_npy_i,
                target_npy_i,
                self.n_filter,
                idx_target=self.idx_target,
            )
            R_ws_total = R_ws if R_ws_total is None else R_ws_total + R_ws
            R_ww_total = R_ww if R_ww_total is None else R_ww_total + R_ww

        assert (
            R_ws_total is not None and R_ww_total is not None
        ), "at least one witness/target sequence must be provided"

        self.filter_state, full_rank = wf_solve(
            R_ws_total,
            R_ww_total,
            self.n_filter,
            self.n_channel,
            inversion_method=self.inversion_method,
            regularization=self.regularization,
        )

        if not full_rank:
            warn("Warning: Filter is not of full rank", RuntimeWarning)
        return self.filter_state, full_rank

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
        if self.filter_state is None:
            raise RuntimeError(
                "The filter must be conditioned before apply() can be used."
            )

        predictions: list = []
        for w_sequence in witness:
            prediction_sequence = wf_apply(self.filter_state, w_sequence)
            if pad:
                prediction_sequence = pad_prediction(
                    prediction_sequence, self.n_filter, self.idx_target
                )
            predictions.append(prediction_sequence)
        return predictions

"""Block-wise Wiener filter"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from numpy.lib.stride_tricks import sliding_window_view

from ..evaluation import FilterInterface, make_2d_array, handle_from_dict
from .wf import invert_R_ww


def bwf_calculate_correlations(
    witness: Sequence | NDArray,
    target: Sequence | NDArray,
    n_filter: int,
    train_step: int = 1,
) -> tuple[NDArray, NDArray]:
    """calculate the cross- and auto-correlation statistics of a block-wise Wiener filter
    for a single witness/target sequence pair

    Unlike the sliding-window WienerFilter, here a single length n_filter witness window
    is used to predict all n_filter samples of the aligned target window ("block"). This
    can be understood as n_filter independent scalar-output Wiener filters that all share
    the same input window, one per position in the block.

    Statistics of multiple sequences can be combined by summing R_ws and R_ww of each
    sequence before solving the resulting equations with bwf_solve().

    :param witness: Witness sensor data
    :param target: Target sensor data
    :param n_filter: Length of the FIR filter / block
    :param train_step: distance between consecutive training windows. 1 uses every
        possible window (maximal data reuse, identical statistics to the sliding-window
        WienerFilter). n_filter uses only non-overlapping, block-aligned windows,
        matching how the filter is applied

    :return: cross-correlation matrix R_ws (n_channel*n_filter, n_filter), auto-correlation
        matrix R_ww (n_channel*n_filter, n_channel*n_filter)
    """
    target_npy: NDArray = np.array(target)
    witness_npy: NDArray = make_2d_array(witness)
    if witness_npy.shape[1] != target_npy.shape[0]:
        raise ValueError("Missmatch between witness_npy and target_npy data shape")
    if n_filter > target_npy.shape[0]:
        raise ValueError("Input data must be at least one filter length")
    if train_step < 1:
        raise ValueError("train_step must be a positive integer")

    # witness_windows[channel] has shape (n_windows, n_filter); concatenating along the
    # channel axis matches the channel-major layout used by the resulting WFC coefficients
    witness_windows = np.concatenate(
        [sliding_window_view(A, n_filter)[::train_step] for A in witness_npy], axis=1
    )
    target_windows = sliding_window_view(target_npy, n_filter)[::train_step]

    R_ww = witness_windows.T.dot(witness_windows)
    R_ws = witness_windows.T.dot(target_windows)

    return R_ws, R_ww


def bwf_solve(
    R_ws: Sequence | NDArray,
    R_ww: Sequence | NDArray,
    n_filter: int,
    n_channel: int,
    inversion_method: str = "np_pinv",
    regularization: float = 0.0,
) -> tuple[NDArray, bool]:
    """solve the block-wise Wiener-Hopf equations R_ww @ w = R_ws for the block filter
    coefficients

    :param R_ws: cross-correlation matrix, as returned by bwf_calculate_correlations()
    :param R_ww: auto-correlation matrix, as returned by bwf_calculate_correlations()
        (or the sum of several, when pooling statistics of multiple sequences)
    :param n_filter: Length of the FIR filter / block
    :param n_channel: Number of witness sensor channels
    :param inversion_method: matrix inversion method used for filter coefficient calculation.
        Check BlockWienerFilter class dock string for possible values
    :param regularization: Tikhonov regularization strength added to the diagonal of the input
        autocorrelation matrix before inversion. 0 disables regularization.

    :return: block filter coefficients, shape (n_channel, n_filter, n_filter)
        (channel, input lag within the block, output position within the block), full_rank (bool)
    """
    R_ww_inv, full_rank = invert_R_ww(R_ww, inversion_method, regularization)
    WFC = R_ww_inv.dot(np.array(R_ws))
    WFC = WFC.reshape((n_channel, n_filter, n_filter))

    assert WFC.shape == (
        n_channel,
        n_filter,
        n_filter,
    ), "input data was to short resulting in an incompatible filter"

    return WFC, full_rank


def bwf_calculate(
    witness: Sequence | NDArray,
    target: Sequence | NDArray,
    n_filter: int,
    train_step: int = 1,
    inversion_method: str = "np_pinv",
    regularization: float = 0.0,
) -> tuple[NDArray, bool]:
    """calculate the coefficients for a block-wise wiener filter

    :param witness: Witness sensor data
    :param target: Target sensor data
    :param n_filter: Length of the FIR filter / block
    :param train_step: distance between consecutive training windows. Check
        BlockWienerFilter class dock string for details
    :param inversion_method: matrix inversion method used for filter coefficient calculation.
        Check BlockWienerFilter class dock string for possible values
    :param regularization: Tikhonov regularization strength added to the diagonal of the input
        autocorrelation matrix before inversion. 0 disables regularization.

    :return: block filter coefficients, full_rank (bool)
    """
    n_channel = make_2d_array(witness).shape[0]
    R_ws, R_ww = bwf_calculate_correlations(witness, target, n_filter, train_step)
    return bwf_solve(
        R_ws,
        R_ww,
        n_filter,
        n_channel,
        inversion_method=inversion_method,
        regularization=regularization,
    )


def bwf_apply(
    WFC: NDArray,
    witness: Sequence | NDArray,
) -> NDArray:
    """apply the block-wise WF to witness data

    Trailing samples that do not fill a whole n_filter block are dropped.

    :param WFC: block filter coefficients, as returned by bwf_solve()
    :param witness: Witness sensor data

    :return: prediction, truncated to a whole number of n_filter blocks
    """
    witness_npy = make_2d_array(witness).astype(np.float64)
    n_channel, n_filter, _ = WFC.shape
    if witness_npy.shape[1] < n_filter:
        raise ValueError("Input minimum lenght is one filter length")

    n_blocks = witness_npy.shape[1] // n_filter
    # (n_channel, n_blocks, n_filter) -> (n_blocks, n_channel, n_filter), matching the
    # channel-major flattening used by bwf_calculate_correlations()
    blocks = witness_npy[:, : n_blocks * n_filter].reshape(
        n_channel, n_blocks, n_filter
    )
    blocks_flat = blocks.transpose(1, 0, 2).reshape(n_blocks, n_channel * n_filter)

    prediction = blocks_flat.dot(WFC.reshape(n_channel * n_filter, n_filter))
    return prediction.reshape(n_blocks * n_filter)


@dataclass
class BlockWienerFilter(FilterInterface):
    """Block-wise Wiener filter implementation

    Splits the input into non-overlapping blocks of length n_filter. Each block's
    n_filter target samples are predicted from the single aligned n_filter witness
    window (per channel), i.e. this can be understood as n_filter independent
    scalar-output Wiener filters that share the same input window. Due to a higher
    non-logically motivated parameter count, this is expected to take more input
    data to train and perform worse at the edges of the blocks that a standard WF.
    It is intended for datasets that contain many short sequences of fixed length,
    where sliding filters are not able to provide long enough prediction sequences.

    :param n_channel: Number of witness sensor channels
    :param n_filter: Length of the FIR filter / block
    :param train_step: distance between consecutive training windows. 1 (default) uses
        every possible window (maximal data reuse, identical statistics to
        WienerFilter). n_filter uses only non-overlapping, block-aligned windows,
        matching how the filter is applied
    :param inversion_method: Matrix inversion method used for filter coefficient calculation
        'np_pinv' np.linalg.pinv()
        'np_inv' np.linalg.inv()
        'sp_pinv' scipy.linalg.pinv()
        'sp_inv' scipy.linalg.inv()
    :param regularization: Tikhonov regularization strength added to the diagonal of the input
        autocorrelation matrix before inversion. 0 disables regularization.

    >>> import franc as fnc
    >>> n_filter = 128
    >>> witness, target = fnc.evaluation.TestDataGenerator(0.1).generate(int(1e5))
    >>> filt = fnc.filtering.BlockWienerFilter(1, n_filter)
    >>> _coefficients, full_rank = filt.condition(witness, target)
    >>> full_rank
    True
    >>> prediction = filt.apply(witness, target) # check on the data used for conditioning
    >>> residual_rms = fnc.evaluation.rms(target[: len(prediction)] - prediction)
    >>> residual_rms > 0.05 and residual_rms < 0.3 # worse than the sliding-window WF
    True

    """

    #: Length of the FIR filter / block
    n_filter: int
    #: The block filter coefficients, shape (n_channel, n_filter, n_filter)
    filter_state: NDArray | None = None
    filter_name: str = "BWF"
    train_step: int = 1
    inversion_method: str = "np_pinv"
    regularization: float = 0.0

    default_args = [None, None]

    @handle_from_dict
    def __init__(
        self,
        n_channel: int,
        n_filter: int,
        train_step: int = 1,
        inversion_method: str = "np_pinv",
        regularization: float = 0.0,
    ):
        super().__init__(n_channel)
        self.requires_apply_target = False
        self.n_filter = n_filter
        self.train_step = train_step
        self.inversion_method = inversion_method
        self.regularization = regularization

        if self.n_filter <= 0:
            raise ValueError("n_filter must be a positive integer")
        if self.n_channel <= 0:
            raise ValueError("n_channel must be a positive integer")
        if self.train_step <= 0:
            raise ValueError("train_step must be a positive integer")

    @property
    def method_filename_part(self) -> str:
        """string that can be used in a file name"""
        return f"{self.filter_name}_{self.n_filter}_{self.n_channel}_{self.train_step}"

    def condition_multi_sequence(
        self,
        witness: Sequence | Sequence[Sequence] | NDArray,
        target: Sequence | NDArray,
    ) -> tuple[NDArray, bool]:
        """Use an input dataset to condition the filter

        The cross- and auto-correlation statistics of all sequences are pooled into a
        single set of block-wise Wiener-Hopf equations, which are then solved once.

        :param witness: Witness sensor data
        :param target: Target sensor data
        """
        witness_npy, target_npy = self.check_data_dimensions_multi_sequence(
            witness, target
        )

        R_ws_total: NDArray | None = None
        R_ww_total: NDArray | None = None
        for witness_npy_i, target_npy_i in zip(witness_npy, target_npy):
            R_ws, R_ww = bwf_calculate_correlations(
                witness_npy_i,
                target_npy_i,
                self.n_filter,
                train_step=self.train_step,
            )
            R_ws_total = R_ws if R_ws_total is None else R_ws_total + R_ws
            R_ww_total = R_ww if R_ww_total is None else R_ww_total + R_ww

        assert (
            R_ws_total is not None and R_ww_total is not None
        ), "at least one witness/target sequence must be provided"

        self.filter_state, full_rank = bwf_solve(
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

        Sequences whose length is not a multiple of n_filter have their trailing,
        incomplete block dropped from the prediction (or zero-padded if pad is True).

        :param witness: Witness sensor data
        :param target: Target sensor data (is ignored)
        :param pad: if True, apply padding zeros so that the length matches the input signal
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
            prediction_sequence = bwf_apply(self.filter_state, w_sequence)
            if pad:
                trailing_padding = w_sequence.shape[1] - len(prediction_sequence)
                prediction_sequence = np.concatenate(
                    [prediction_sequence, np.zeros(trailing_padding)]
                )
            predictions.append(prediction_sequence)
        return predictions

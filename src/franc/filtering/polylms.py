"""experiment in building a polynomial lms filter"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import numba

from .common import FilterBase, handle_from_dict


@numba.njit
def _lms_loop(
    witness: NDArray,
    target: NDArray,
    n_filter: int,
    idx_target: int,
    filter_state: NDArray,
    normalized: bool,
    step_scale: float,
    coefficient_clipping: float,
    order: int,
) -> tuple[NDArray, NDArray, int, int]:
    """Run an LMS filter over intput sequences.

    :param witness: Witness sensor data
    :param target: Target sensor data (is ignored)
    :param n_filter: Length of the FIR filter (how many samples are in the input window per output sample)
    :param idx_target: Position of the prediction
    :param filter_state: The initial FIR filter state
    :param normalized: if True: NLMS, else LMS
    :param step_scale: the learning rate of the LMS filter

    :param n_channel: Number of witness sensor channels
    :param order: polynomial order of the filter
    :param pad: if True, apply padding zeros so that the length matches the target signal
    :param update_state: if True, the filter state will be changed. If false, the filter state will remain

    :return: Prediction, Filter state, Target offset, Prediction length
    """
    offset_target = n_filter - idx_target - 1
    pred_length = len(target) - n_filter + 1

    prediction = []
    for idx in range(0, pred_length):
        # make prediction
        w_sel = witness[:, idx : idx + n_filter]  # input to predcition
        pred = 0
        for i in range(order):
            pred += np.sum(filter_state[i] * w_sel ** (i + 1))
        err = target[idx + offset_target] - pred

        prediction.append(pred)

        # update filter
        if normalized:
            norm = np.sum(w_sel * w_sel)
            if norm < 0:
                raise ValueError(
                    "Overflow! You are probably passing integers of insufficient precision to this function."
                )

            for i in range(order):
                # NOTE: this might not be the correct/optimal normalization
                filter_state[i] += (
                    2 * step_scale * err * w_sel ** (i + 1) / norm ** ((i + 2) / 2)
                )
        else:
            for i in range(order):
                filter_state[i] += 2 * step_scale * err * w_sel ** (i + 1)

        if not np.isnan(coefficient_clipping):
            filter_state = np.clip(
                filter_state, -coefficient_clipping, coefficient_clipping
            )

    prediction_npy = np.array(prediction, dtype=np.float64)
    return prediction_npy, filter_state, offset_target, pred_length


@dataclass
class PolynomialLMSFilter(FilterBase):
    r"""Experimental non-linear LMS-like filter implementation
    Implements: :math:`x[n] = \sum_p\sum_i\sum_t {w_i[n-t]}^pH_{it}` where p is the polynomial order, i the channel and t the index within the filter

    :param n_filter: Length of the FIR filter (how many samples are in the input window per output sample)
    :param idx_target: Position of the prediction
    :param n_channel: Number of witness sensor channels
    :param normalized: If True: NLMS, else LMS
    :param step_scale: The learning rate of the LMS filter
    :param coefficient_clipping: If set to a positive float, FIR filter coefficients
           will be limited to this value. This can increase filter stability.
    :param order: Polynomial order of the filter

    >>> import franc as fnc
    >>> n_filter = 128
    >>> witness, target = fnc.evaluation.TestDataGenerator(0.1).generate(int(1e5))
    >>> filt = fnc.filtering.PolynomialLMSFilter(1, n_filter, 0, step_scale=0.1, order=2, coefficient_clipping=4)
    >>> filt.condition(witness, target)
    >>> prediction = filt.apply(witness, target) # check on the data used for conditioning
    >>> residual_rms = fnc.evaluation.rms((target-prediction)[1000:])
    >>> residual_rms > 0.05 and residual_rms < 0.15 # the expected RMS in this test scenario is 0.1
    True

    """

    #: The current FIR coefficients of the LMS filter
    filter_state: NDArray
    normalized: bool
    step_scale: float
    coefficient_clipping: float
    order: int

    filter_name: str = "PolyLMS"

    @handle_from_dict
    def __init__(
        self,
        n_channel: int,
        n_filter: int,
        idx_target: int,
        normalized: bool = True,
        step_scale: float = 0.5,
        coefficient_clipping: float = np.nan,
        order: int = 1,
    ):
        super().__init__(n_channel, n_filter, idx_target)
        self.normalized = normalized
        self.step_scale = step_scale
        self.coefficient_clipping = coefficient_clipping
        self.order = order

        assert self.step_scale > 0, "Step scale must be positive"
        assert (
            np.isnan(self.coefficient_clipping) or self.coefficient_clipping > 0
        ), "coefficient_clipping must be positive"
        assert self.order > 0

        self.reset()

    def reset(self):
        """reset the filter coefficients to zero"""
        self.filter_state = np.zeros((self.order, self.n_channel, self.n_filter))

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

    def apply(
        self,
        witness: Sequence | NDArray,
        target: Sequence | NDArray | None = None,
        pad: bool = True,
        update_state: bool = False,
    ) -> NDArray:
        """Apply the filter to input data

        :param witness: Witness sensor data
        :param target: Target sensor data (is ignored)
        :param pad: if True, apply padding zeros so that the length matches the target signal
        :param update_state: if True, the filter state will be changed. If false, the filter state will remain

        :return: prediction
        """
        if target is None:
            raise ValueError("A target signal must be supplied")
        witness, target = self.check_data_dimensions(witness, target)
        assert target is not None, "Target data must be supplied"

        # numba jit and numpy don't always work correctly with numpy arrays scalars
        # casting is required to prevent problems
        prediction, filter_state, offset_target, pred_length = _lms_loop(
            witness,
            target,
            self.n_filter,
            self.idx_target,
            np.array(self.filter_state),
            self.normalized,
            self.step_scale,
            np.float64(self.coefficient_clipping),
            int(self.order),
        )

        if update_state:
            self.filter_state = filter_state

        prediction = np.array(prediction)
        if pad:
            prediction = np.concatenate(
                [
                    np.zeros(offset_target),
                    prediction,
                    np.zeros(len(target) - pred_length - offset_target),
                ]
            )

        return prediction

    def apply_multi_sequence(
        self,
        witness: Sequence | NDArray,
        target: Sequence | NDArray | None = None,
        pad: bool = True,
        update_state: bool = False,
    ) -> Sequence[NDArray]:
        if target is None:
            raise ValueError("A target signal must be supplied")
        predictions = [
            self.apply(w, t, pad, update_state) for w, t in zip(witness, target)
        ]
        return predictions

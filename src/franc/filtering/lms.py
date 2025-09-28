"""Least Mean Squares filter"""

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
) -> tuple[NDArray, NDArray, int, int]:
    offset_target = n_filter - idx_target - 1
    pred_length = len(target) - n_filter + 1

    prediction = []
    for idx in range(0, pred_length):
        # make prediction
        w_sel = witness[:, idx : idx + n_filter]  # input to predcition
        pred = np.sum(filter_state * w_sel)
        err = target[idx + offset_target] - pred

        prediction.append(pred)

        # update filter
        if normalized:
            norm = np.sum(w_sel * w_sel)
            if norm < 0:
                raise ValueError(
                    "Overflow! You are probably passing integers of insufficient precision to this function."
                )
            filter_state += 2 * step_scale * err * w_sel / norm
        else:
            filter_state += 2 * step_scale * err * w_sel

        if not np.isnan(coefficient_clipping):
            filter_state = np.clip(
                filter_state, -coefficient_clipping, coefficient_clipping
            )
    return np.array(prediction), filter_state, offset_target, pred_length


@dataclass
class LMSFilter(FilterBase):
    """LMS filter implementation

    :param n_filter: Length of the FIR filter (how many samples are in the input window per output sample)
    :param idx_target: Position of the prediction
    :param n_channel: Number of witness sensor channels
    :param normalized: if True: NLMS, else LMS
    :param coefficient_clipping: If set to a positive float, FIR filter coefficients
           will be limited to this value. This can increase filter stability.
    :param step_scale: the learning rate of the LMS filter

    >>> import franc as fnc
    >>> n_filter = 128
    >>> witness, target = fnc.evaluation.TestDataGenerator(0.1).generate(int(1e5))
    >>> filt = fnc.filtering.LMSFilter(1, n_filter, 0)
    >>> filt.condition(witness, target)
    >>> prediction = filt.apply(witness, target) # check on the data used for conditioning
    >>> residual_rms = fnc.evaluation.rms(target-prediction)
    >>> residual_rms > 0.05 and residual_rms < 0.15 # the expected RMS in this test scenario is 0.1
    True

    """

    #: The current FIR coefficients of the LMS filter
    filter_state: NDArray
    normalized: bool
    step_scale: float
    coefficient_clipping: float

    filter_name: str = "LMS"

    @handle_from_dict
    def __init__(
        self,
        n_channel: int,
        n_filter: int,
        idx_target: int,
        normalized: bool = True,
        step_scale: float = 0.1,
        coefficient_clipping: float = np.nan,
    ):
        super().__init__(n_channel, n_filter, idx_target)
        self.normalized = normalized
        self.step_scale = step_scale
        self.coefficient_clipping = coefficient_clipping

        assert self.step_scale > 0, "Step scale must be positive"
        assert (
            np.isnan(self.coefficient_clipping) or self.coefficient_clipping > 0
        ), "coefficient_clipping must be positive"

        self.reset()

    def reset(self):
        """reset the filter coefficients to zero"""
        self.filter_state = np.zeros((self.n_channel, self.n_filter))

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

        # addition of zero is used to convert numpy scalars into standard python objects
        # numba jit and numpy don't work otherwise
        prediction, filter_state, offset_target, pred_length = _lms_loop(
            witness,
            target,
            self.n_filter,
            self.idx_target,
            np.array(self.filter_state),
            self.normalized,
            self.step_scale,
            0 + self.coefficient_clipping,
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

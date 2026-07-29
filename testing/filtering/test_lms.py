"""Tests for filtering.LMSFilter"""

from typing import Iterable

import numpy as np

import franc as fnc
from franc.filtering.lms import LMSFilter

from .test_filters import TestFilter

RNG_SEED = 113510
ZERO_NORM_WARNING = "Zero normalization encountered"


class TestLMSFilter(TestFilter.TestFilter[LMSFilter]):
    """tests for the LeastMeanSquares filter implementation"""

    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        test_configurations = [
            {"normalized": True},
            {"normalized": True, "coefficient_clipping": 2},
            {"normalized": False, "step_scale": 0.001},
        ]
        self.set_target(fnc.filtering.LMSFilter, test_configurations)

    # mark return type of instantiate_filters correctly
    def instantiate_filters(
        self, n_channel=1, n_filter=128, idx_target=0
    ) -> Iterable[LMSFilter]:
        return_value: Iterable[LMSFilter] = super().instantiate_filters(
            n_channel, n_filter, idx_target
        )
        return return_value

    def test_update_state_setting(self):
        """check that the filter reaches a WF-Like performance on a simple static test case"""
        witness, target = fnc.evaluation.TestDataGenerator(
            [0.1] * 2, rng_seed=RNG_SEED
        ).generate(int(2e4))

        for filt in self.instantiate_filters(n_channel=2, n_filter=32):
            # check for no changes when False
            filt.apply(witness, target, update_state=False)
            self.assertTrue(bool(np.all(filt.filter_state == 0)))

            # check for no changes when True
            filt.apply(witness, target, update_state=True)
            self.assertTrue(bool(np.any(filt.filter_state != 0)))

    def test_zero_norm_witness_window(self):
        """A witness window that is exactly zero must not corrupt the filter with NaN/Inf.

        normalized LMS/PolyLMS divides by the window norm; an all-zero
        window can result in a 0/0 division that permanently poisones the filter_state
        with NaN for the remainder of the run. This checks,
        that these updates are being skipped.
        """
        n_filter = 8
        witness, target = fnc.evaluation.TestDataGenerator(
            [0.1] * 2, rng_seed=RNG_SEED
        ).generate(int(1e3))
        witness = np.array(witness)
        # force norm == 0 for the witness window starting at sample 100
        witness[:, 100 : 100 + n_filter] = 0.0

        for filt in self.instantiate_filters(n_channel=2, n_filter=n_filter):
            if not filt.normalized:
                continue

            with self.assertWarns(RuntimeWarning) as cm:
                filt.condition(witness, target)
            self.assertIn(ZERO_NORM_WARNING, str(cm.warning))

            self.assertTrue(np.isfinite(filt.filter_state).all())

            prediction = filt.apply(witness, target)
            self.assertTrue(np.isfinite(prediction).all())

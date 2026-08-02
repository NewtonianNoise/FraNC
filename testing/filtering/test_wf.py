"""Tests for WienerFilter"""

import numpy as np

import franc as fnc
from franc.filtering.wf import WienerFilter, wf_calculate, wf_apply

from .test_filters import TestFilter, RNG_SEED


class TestWienerFilter(TestFilter.TestFilter[WienerFilter]):
    """Tests for the WF"""

    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        test_configurations = [
            {"inversion_method": "np_pinv"},
            {"inversion_method": "np_inv", "regularization": 1e-10},
            {"inversion_method": "sp_pinv"},
            {"inversion_method": "sp_inv", "regularization": 1e-10},
            {"regularization": 1e-3},
        ]
        self.set_target(fnc.filtering.WienerFilter, test_configurations)

    def test_conditioning_warning(self):
        """check that a warning is thrown if the autocorrelation array does not have full rank"""
        n_filter = 128
        witness, target = fnc.evaluation.TestDataGenerator([0.1]).generate(int(1e4))

        # using two identical input datasets produces non-full-rank autocorrelation matrices
        witness = [witness[0], witness[0]]

        for filt in self.instantiate_filters(n_channel=2, n_filter=n_filter):
            if filt.regularization == 0:
                # with regularization on, we expect the warning to not appear
                self.assertWarns(RuntimeWarning, filt.condition, witness, target)

    def test_regularization_fixes_rank_deficiency(self):
        """check that Tikhonov regularization resolves a rank-deficient autocorrelation matrix"""
        n_filter = 128
        witness, target = fnc.evaluation.TestDataGenerator([0.1]).generate(int(1e4))

        # using two identical input datasets produces non-full-rank autocorrelation matrices
        witness = [witness[0], witness[0]]

        filt = fnc.filtering.WienerFilter(2, n_filter, 0, regularization=1e-2)
        _, full_rank = filt.condition(witness, target)
        self.assertTrue(full_rank)

    def test_impulse_response_recovery(self):
        """check that a known FIR response is recovered from a noiseless system"""
        impulse_response = np.array([0.5, -0.25, 0.75, 0.1])
        witness = np.random.default_rng(RNG_SEED).normal(0, 1, 20000)
        target = np.convolve(witness, impulse_response)[: len(witness)]

        filt = fnc.filtering.WienerFilter(1, len(impulse_response), 0)
        coefficients, _ = filt.condition(witness, target)

        # the coefficients hold the impulse response in reverse order
        np.testing.assert_allclose(
            coefficients[0], np.flip(impulse_response), atol=1e-2
        )

    def test_module_functions_with_1d_witness(self):
        """check that the wf functions treat 1D and 2D single channel input the same"""
        witness, target = fnc.evaluation.TestDataGenerator(
            0.1, rng_seed=RNG_SEED
        ).generate(2000)

        coefficients_1d = wf_calculate(witness[0], target, 16)[0]
        coefficients_2d = wf_calculate(witness, target, 16)[0]
        prediction_1d = wf_apply(coefficients_1d, witness[0])
        prediction_2d = wf_apply(coefficients_2d, witness)

        np.testing.assert_array_equal(coefficients_1d, coefficients_2d)
        np.testing.assert_array_equal(prediction_1d, prediction_2d)

    def test_no_target_for_apply(self):
        """check that the filter can be applied without a target signal"""
        n_filter = 128
        witness, target = fnc.evaluation.TestDataGenerator(0.1).generate(n_filter * 2)

        for filt in self.instantiate_filters(n_channel=1, n_filter=n_filter):
            filt.condition(witness, target)
            filt.apply(witness)

"""Tests for BlockWienerFilter

BlockWienerFilter does not extend FilterBase (it has no idx_target), so it cannot use
the shared TestFilter.TestFilter parent class the other filters use.
"""

import unittest

import numpy as np

import franc as fnc
from franc.filtering.bwf import (
    BlockWienerFilter,
    bwf_calculate,
    bwf_apply,
)

RNG_SEED = 113510
TEST_FILE = "testing/test_outputs/filter_serialization_test_file"


class TestBlockWienerFilter(unittest.TestCase):
    """Tests for the block-wise WF"""

    def test_conditioning_returns_expected_shape(self):
        """check that the conditioned coefficients have shape (n_channel, n_filter, n_filter)"""
        n_filter = 32
        witness, target = fnc.evaluation.TestDataGenerator(
            [0.1] * 2, rng_seed=RNG_SEED
        ).generate(int(2e4))

        filt = BlockWienerFilter(2, n_filter)
        coefficients, full_rank = filt.condition(witness, target)

        self.assertEqual(coefficients.shape, (2, n_filter, n_filter))
        self.assertTrue(full_rank)

    def test_conditioning_warning(self):
        """check that a warning is thrown if the autocorrelation array does not have full rank"""
        n_filter = 32
        witness, target = fnc.evaluation.TestDataGenerator([0.1]).generate(int(1e4))

        # using two identical input datasets produces non-full-rank autocorrelation matrices
        witness = [witness[0], witness[0]]

        filt = BlockWienerFilter(2, n_filter)
        self.assertWarns(RuntimeWarning, filt.condition, witness, target)

    def test_regularization_fixes_rank_deficiency(self):
        """check that Tikhonov regularization resolves a rank-deficient autocorrelation matrix"""
        n_filter = 32
        witness, target = fnc.evaluation.TestDataGenerator([0.1]).generate(int(1e4))

        # using two identical input datasets produces non-full-rank autocorrelation matrices
        witness = [witness[0], witness[0]]

        filt = BlockWienerFilter(2, n_filter, regularization=1e-2)
        _, full_rank = filt.condition(witness, target)
        self.assertTrue(full_rank)

    def test_apply_output_shapes(self):
        """check padded and un-padded output lengths for a sequence that isn't a whole number of blocks"""
        n_filter = 32
        witness, target = fnc.evaluation.TestDataGenerator(
            0.1, rng_seed=RNG_SEED
        ).generate(int(1e4) + 5)

        filt = BlockWienerFilter(1, n_filter)
        filt.condition(witness, target)

        prediction = filt.apply(witness, target)
        self.assertEqual(prediction.shape, target.shape)

        prediction_nopad = filt.apply(witness, target, pad=False)
        self.assertEqual(len(prediction_nopad), (len(target) // n_filter) * n_filter)

    def test_no_target_for_apply(self):
        """check that the filter can be applied without a target signal"""
        n_filter = 32
        witness, target = fnc.evaluation.TestDataGenerator(0.1).generate(n_filter * 4)

        filt = BlockWienerFilter(1, n_filter)
        filt.condition(witness, target)
        filt.apply(witness)

    def test_module_functions_with_1d_witness(self):
        """check that the bwf functions treat 1D and 2D single channel input the same"""
        witness, target = fnc.evaluation.TestDataGenerator(
            0.1, rng_seed=RNG_SEED
        ).generate(2000)

        coefficients_1d = bwf_calculate(witness[0], target, 16)[0]
        coefficients_2d = bwf_calculate(witness, target, 16)[0]
        prediction_1d = bwf_apply(coefficients_1d, witness[0])
        prediction_2d = bwf_apply(coefficients_2d, witness)

        np.testing.assert_array_equal(coefficients_1d, coefficients_2d)
        np.testing.assert_array_equal(prediction_1d, prediction_2d)

    def test_multi_sequence_pools_statistics(self):
        """check that conditioning on multiple sequences works and produces a usable filter"""
        n_filter = 32
        witness, target = fnc.evaluation.TestDataGenerator(
            0.1, rng_seed=RNG_SEED
        ).generate_multiple([int(1e4), int(2e4)])

        filt = BlockWienerFilter(1, n_filter, train_step=n_filter)
        _, full_rank = filt.condition_multi_sequence(witness, target)
        self.assertTrue(full_rank)

        prediction = filt.apply_multi_sequence(witness, target)
        for p, t in zip(prediction, target):
            self.assertEqual(p.shape, t.shape)

    def test_performance(self):
        """check that the filter reaches a reasonable (WF-like, if somewhat worse) performance"""
        n_filter = 32
        witness, target = fnc.evaluation.TestDataGenerator(
            [0.1], rng_seed=RNG_SEED
        ).generate(int(3e4))

        for train_step in (1, n_filter):
            filt = BlockWienerFilter(1, n_filter, train_step=train_step)
            filt.condition(witness, target)
            prediction = filt.apply(witness, target)

            residual = fnc.evaluation.rms((target - prediction)[4000:-n_filter])
            self.assertGreater(residual, 0.05)
            self.assertLess(residual, 0.3)

    def test_saving_loading(self):
        """check that saving and loading a conditioned filter round-trips"""
        n_filter = 16
        witness, target = fnc.evaluation.TestDataGenerator(
            [0.1] * 2, rng_seed=RNG_SEED
        ).generate(int(2e4))

        filt = BlockWienerFilter(2, n_filter)
        filt.condition(witness, target)
        filt.save(TEST_FILE)
        loaded_filter = BlockWienerFilter.load(TEST_FILE)

        self.assertEqual(filt.method_hash, loaded_filter.method_hash)

        prediction_orig = filt.apply(witness, target)
        prediction_loaded = loaded_filter.apply(witness, target)
        np.testing.assert_array_equal(prediction_orig, prediction_loaded)

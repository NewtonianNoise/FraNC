"""Tests for TestUpdatingWienerFilter"""

import franc as fnc
from franc.filtering.uwf import UpdatingWienerFilter

from .test_filters import TestFilter


class TestUpdatingWienerFilter(TestFilter.TestFilter[UpdatingWienerFilter]):
    """Test cases for the UWF"""

    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_target(
            fnc.filtering.UpdatingWienerFilter,
            [{"context_pre": 128 * 40, "context_post": 128 * 40}],
        )

    def test_conditioning_warning(self):
        """check that a warning is thrown if the conditioning function is caled"""
        n_filter = 128
        witness, target = fnc.evaluation.TestDataGenerator([0.1]).generate(int(1e4))

        for filt in self.instantiate_filters(n_channel=2, n_filter=n_filter):
            self.assertWarns(RuntimeWarning, filt.condition, witness, target)

    def test_non_full_rank_warning(self):
        """check that a warning is thrown if a filter does not have full rank"""
        n_filter = 128
        n_channel = 2
        witness, target = fnc.evaluation.TestDataGenerator([0.1]).generate(int(1e4))

        # using two identical input datasets produces non-full-rank autocorrelation matrices
        witness = [witness[0]] * n_channel

        for filt in self.instantiate_filters(n_channel=n_channel, n_filter=n_filter):
            self.assertWarns(RuntimeWarning, filt.apply, witness, target)

    def test_acceptance_of_minimum_input_length_different_context_length(self):
        """test that the minimum input length is accepter for different context values"""
        n_filter = 128
        witness, target = fnc.evaluation.TestDataGenerator([0.1]).generate(n_filter * 2)

        for context_len in [0, 10000]:
            filt = self.target_filter(
                n_channel=1, n_filter=n_filter, idx_target=0, context_pre=context_len
            )
            pred = filt.apply(witness, target)
            self.assertEqual(len(pred), len(target))

            filt = self.target_filter(
                n_channel=1, n_filter=n_filter, idx_target=0, context_post=context_len
            )
            pred = filt.apply(witness, target)
            self.assertEqual(len(pred), len(target))

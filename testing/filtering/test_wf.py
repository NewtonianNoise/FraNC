"""Tests for WienerFilter"""

import franc as fnc
from franc.filtering.wf import WienerFilter

from .test_filters import TestFilter


class TestWienerFilter(TestFilter.TestFilter[WienerFilter]):
    """Tests for the WF"""

    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_target(fnc.filtering.WienerFilter)

    def test_conditioning_warning(self):
        """check that a warning is thrown if the autocorrelation array does not have full rank"""
        n_filter = 128
        witness, target = fnc.evaluation.TestDataGenerator([0.1]).generate(int(1e4))

        # using two identical input datasets produces non-full-rank autocorrelation matrices
        witness = [witness[0], witness[0]]

        for filt in self.instantiate_filters(n_channel=2, n_filter=n_filter):
            self.assertWarns(RuntimeWarning, filt.condition, witness, target)

    def test_no_target_for_apply(self):
        """check that the filter can be applied without a target signal"""
        n_filter = 128
        witness, target = fnc.evaluation.TestDataGenerator(0.1).generate(n_filter * 2)

        for filt in self.instantiate_filters(n_channel=1, n_filter=n_filter):
            filt.condition(witness, target)
            filt.apply(witness)

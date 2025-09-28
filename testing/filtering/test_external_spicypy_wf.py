"""Tests for the spicypy WF implementation"""

from warnings import warn

import franc
from .test_filters import TestFilter


class TestSpicypyWienerFilter(
    TestFilter.TestFilter[franc.external.SpicypyWienerFilter]
):
    """Tests for the spicypy WF implementation"""

    __test__ = True

    expected_performance = {
        # noise level, (acceptance min, acceptance_max)
        0.0: (0, 0.05),
        0.1: (0.05, 0.2),  # typically worse performance
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_target(franc.external.SpicypyWienerFilter)

        warn("Running spicypy WF tests. These are quite slow.")

    def test_performance(self):
        warn(
            "The performance test is disabled for spicypy WF, because it is very slow."
        )

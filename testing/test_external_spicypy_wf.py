import unittest
from warnings import warn

import saftig

from .test_filters import TestFilter


class TestSpicypyWienerFilter(unittest.TestCase, TestFilter):
    """Tests for the WF"""

    __test__ = True

    expected_performance = {
        # noise level, (acceptance min, acceptance_max)
        0.0: (0, 0.05),
        0.1: (0.05, 0.2),  # typically worse performance
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_target(saftig.external.SpicypyWienerFilter)

        warn("Running spicypy WF tests. These are quite slow.")

    def test_performance(self):
        warn(
            "The performance test is disabled for spicypy WF, because it is very slow."
        )

"""Tests for BypassFilter"""

import franc as fnc
from franc.filtering.bypass import BypassFilter

from .test_filters import TestFilter


class TestBypassFilter(TestFilter.TestFilter[BypassFilter]):
    """Tests for the BypassFilter"""

    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_target(fnc.filtering.BypassFilter)

    def test_performance(self):
        """disable this test as it is expected that this filter has no effect"""

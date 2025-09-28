"""Tests for PolynomialLMSFilter class"""

import franc as fnc

# from .test_filters import TestFilter
from .test_lms import TestLMSFilter


class TestPolynomialLMSFilter(TestLMSFilter):
    """tests for the polynomial variant of a LeastMeanSquares filter implementation"""

    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        test_configurations = [
            {"order": 1},
            {"order": 1, "coefficient_clipping": 2},
            {"order": 2},
            {"order": 1, "normalized": False, "step_scale": 0.001},
        ]
        self.set_target(fnc.filtering.PolynomialLMSFilter, test_configurations)

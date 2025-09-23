"""Tests for EvaluationMetric classes"""

from typing import Generator, Type
import unittest
from collections.abc import Sequence

import numpy as np
import matplotlib.pyplot as plt

import franc as fnc

N_TEST_DATASET = 1000
test_dataset = fnc.evaluation.TestDataGenerator(witness_noise_level=[1] * 3).dataset(
    [N_TEST_DATASET], [N_TEST_DATASET]
)
test_prediction = [
    np.array(seq, copy=True) + 1 for seq in test_dataset.target_evaluation
]


class TestEvaluationMetric:  # pylint: disable=too-few-public-methods
    """Outer class to prevent loading of the parent class by test frameworks"""

    class TestEvaluationMetric(unittest.TestCase):
        """Parent class for evaluation metric testing"""

        __test__ = False
        expected_results: list | None = None
        tested_metric: type[fnc.evaluation.EvaluationMetric]
        parameter_sets: Sequence[dict]

        def set_tested_metric(
            self,
            tested_metric: Type[fnc.evaluation.EvaluationMetric],
            parameter_sets: Sequence[dict],
        ):
            """Must be called by child __init__ to set target and parameter sets"""
            self.tested_metric = tested_metric
            self.parameter_sets = parameter_sets

        def instantiate_filters(
            self,
        ) -> Generator[fnc.evaluation.EvaluationMetric, None, None]:
            """instantiate the target filter for all configurations"""
            for parameters in self.parameter_sets:
                yield self.tested_metric(**parameters)

        def test_basic_functionality(self):
            """Check that instantiation works"""
            assert self.expected_results is None or (
                len(self.expected_results) == len(self.parameter_sets)
            )

            for idx, metric in enumerate(self.instantiate_filters()):
                metric = metric.apply(test_prediction, test_dataset)

                self.assertIsInstance(metric.text, str)

                # optional check that the result is matching the expectation
                if self.expected_results is not None:
                    self.assertAlmostEqual(metric.result, self.expected_results[idx])

                if issubclass(
                    self.tested_metric, fnc.evaluation.EvaluationMetricPlottable
                ):
                    fig, ax = plt.subplots()
                    metric.plot(ax)
                    plt.close(fig)


class TestRMSMetric(TestEvaluationMetric.TestEvaluationMetric):
    """Tests for RMSMetric"""

    __test__ = True

    expected_results = [1.0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_tested_metric(fnc.evaluation.RMSMetric, [{}])


class TestMSEMetric(TestEvaluationMetric.TestEvaluationMetric):
    """Tests for MSEMetric"""

    __test__ = True

    expected_results = [1.0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_tested_metric(fnc.evaluation.MSEMetric, [{}])


class TestBandwidthPowerMetric(TestEvaluationMetric.TestEvaluationMetric):
    """Tests for BandwidthPowerMetric"""

    __test__ = True

    expected_results = [0.0, 0.0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_tested_metric(
            fnc.evaluation.BandwidthPowerMetric,
            [
                {"f_start": 0.1, "f_stop": 0.2, "n_fft": 15},
                {"f_start": 0.1, "f_stop": 0.2, "n_fft": 16, "window": "boxcar"},
            ],
        )


class TestPSDMetric(TestEvaluationMetric.TestEvaluationMetric):
    """Tests for PSDMetric"""

    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_tested_metric(
            fnc.evaluation.PSDMetric,
            [
                {"n_fft": 15},
                {"n_fft": 16, "logx": False, "logy": False, "window": "boxcar"},
            ],
        )

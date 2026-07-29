"""Tests for EvaluationMetric classes"""

from __future__ import annotations

from typing import Generator, Type
import unittest
from collections.abc import Sequence

import numpy as np

import franc as fnc

N_TEST_DATASET = 1000
test_dataset = fnc.evaluation.TestDataGenerator(
    witness_noise_level=[1] * 3, rng_seed=0xDEADBEAF
).dataset([N_TEST_DATASET], [N_TEST_DATASET])
test_dataset_zero_signal = fnc.evaluation.TestDataGenerator(
    witness_noise_level=[1] * 3, rng_seed=0xDEADBEAF
).dataset([N_TEST_DATASET], [N_TEST_DATASET], generate_signal=True, signal_amplitude=0)
test_prediction = [
    np.array(seq, copy=True) + 1 for seq in test_dataset.target_evaluation
]

# test_dataset_zero_signal above uses signal_amplitude=0, so residual (signal subtracted)
# and residual_signal (signal kept) are always numerically identical there. This
# deterministic dataset has a real, nonzero signal so the two can be told apart.
test_dataset_signal = fnc.evaluation.EvaluationDataset(
    sample_rate=1.0,
    witness_conditioning=[[np.zeros(4)]],
    target_conditioning=[np.zeros(4)],
    witness_evaluation=[[np.zeros(4)]],
    target_evaluation=[np.full(4, 5.0)],
    signal_conditioning=[np.zeros(4)],
    signal_evaluation=[np.full(4, 2.0)],
)
test_dataset_signal_prediction = [np.full(4, 1.0)]


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

            for dataset in (test_dataset, test_dataset_zero_signal):
                for idx, metric in enumerate(self.instantiate_filters()):
                    self.assertRaises(RuntimeError, metric.result_full)

                    metric = metric.apply(test_prediction, dataset)
                    self.assertIsInstance(metric.result_full(), tuple)

                    self.assertIsInstance(metric.text, str)

                    # optional check that the result is matching the expectation
                    if self.expected_results is not None:
                        self.assertAlmostEqual(
                            metric.result, self.expected_results[idx]
                        )

                    # test functionality of plottable filters
                    if issubclass(
                        self.tested_metric, fnc.evaluation.EvaluationMetricPlottable
                    ):
                        metric.save_plot(
                            "testing/test_outputs/" + metric.filename("some_context")
                        )


class TestRMSMetric(TestEvaluationMetric.TestEvaluationMetric):
    """Tests for RMSMetric"""

    __test__ = True

    expected_results = [1.0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_tested_metric(fnc.evaluation.RMSMetric, [{}])

    def test_signal_is_subtracted_from_residual(self):
        """residual = target - signal - prediction = 5 - 2 - 1 = 2"""
        metric = self.tested_metric().apply(
            test_dataset_signal_prediction, test_dataset_signal
        )
        self.assertAlmostEqual(metric.result, 2.0)


class TestMSEMetric(TestEvaluationMetric.TestEvaluationMetric):
    """Tests for MSEMetric"""

    __test__ = True

    expected_results = [1.0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_tested_metric(fnc.evaluation.MSEMetric, [{}])


class TestRMetric(TestEvaluationMetric.TestEvaluationMetric):
    """Tests for RMetric"""

    __test__ = True

    expected_results = [1.9563013885714056]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_tested_metric(fnc.evaluation.RMetric, [{}])

    def test_pre_filter_power_subtracts_signal(self):
        """mse_pre must use (target - signal): (5-2)^2=9, mse_post=(5-2-1)^2=4, R=4/9"""
        metric = self.tested_metric().apply(
            test_dataset_signal_prediction, test_dataset_signal
        )
        self.assertAlmostEqual(metric.result, 4.0 / 9.0)


class TestSqrtRMetric(TestEvaluationMetric.TestEvaluationMetric):
    """Tests for SqrtRMetric"""

    __test__ = True

    expected_results = [1.3986784435928816]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_tested_metric(fnc.evaluation.SqrtRMetric, [{}])


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

    def test_band_selection(self):
        """A pure tone must be picked up by a metric whose band contains it,
        and not by a metric configured for a disjoint band."""
        sample_rate = 100.0
        n = 2000
        t = np.arange(n) / sample_rate
        f0 = 20.0
        target = np.zeros(n)
        # residual = target - prediction = sin(2*pi*f0*t), a pure tone at f0
        prediction = -np.sin(2 * np.pi * f0 * t)

        dataset = fnc.evaluation.EvaluationDataset(
            sample_rate=sample_rate,
            witness_conditioning=[[np.zeros(n)]],
            target_conditioning=[np.zeros(n)],
            witness_evaluation=[[np.zeros(n)]],
            target_evaluation=[target],
        )

        in_band = fnc.evaluation.BandwidthPowerMetric(
            f_start=f0 - 5, f_stop=f0 + 5, n_fft=256
        ).apply([prediction], dataset)
        out_of_band = fnc.evaluation.BandwidthPowerMetric(
            f_start=f0 + 15, f_stop=f0 + 25, n_fft=256
        ).apply([prediction], dataset)

        # a unit-amplitude sine has mean power 0.5; the in-band metric must recover it
        self.assertAlmostEqual(in_band.result, 0.5, places=2)
        # the disjoint band excludes the tone entirely, so it must see none of that power
        self.assertLess(out_of_band.result, 1e-6)


class TestPSDMetric(TestEvaluationMetric.TestEvaluationMetric):
    """Tests for PSDMetric"""

    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_tested_metric(
            fnc.evaluation.PSDMetric,
            [
                {"n_fft": 15},
                {
                    "n_fft": 16,
                    "logx": False,
                    "logy": False,
                    "window": "boxcar",
                    "show_signal": True,
                },
            ],
        )

    def test_low_n_fft(self):
        """Check that a too low n_fft value creates raises an exception"""
        self.assertRaises(ValueError, self.tested_metric, n_fft=1)


class TestASDMetric(TestPSDMetric):
    """Tests for ASDMetric"""

    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_tested_metric(
            fnc.evaluation.ASDMetric,
            [
                {"n_fft": 15},
                {
                    "n_fft": 16,
                    "logx": False,
                    "logy": False,
                    "window": "boxcar",
                    "show_signal": True,
                },
                {"n_fft": 15, "show_target_minus_signal": True},
            ],
        )


class TestTimeSeriesMEtric(TestEvaluationMetric.TestEvaluationMetric):
    """Tests for TimeSeriesMetric"""

    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_tested_metric(
            fnc.evaluation.TimeSeriesMetric,
            [
                {},
            ],
        )

    def test_residual_signal_keeps_signal(self):
        """residual_signal = target - prediction = 5 - 1 = 4 (signal not removed)"""
        metric = self.tested_metric(residual_with_signal=True).apply(
            test_dataset_signal_prediction, test_dataset_signal
        )
        np.testing.assert_array_almost_equal(
            metric.result_full()[0][0], np.full(4, 4.0)
        )


class TestSpectrogramMetric(TestEvaluationMetric.TestEvaluationMetric):
    """Tests for SpectrogramMetric"""

    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_tested_metric(
            fnc.evaluation.SpectrogramMetric,
            [
                {"n_fft": 32},
                {"n_fft": 128, "xlim": (0, 100), "ylim": (0, 0.1)},
                {"n_fft": 64, "with_signal": False, "asd": False},
            ],
        )


class TestMetricHashing(unittest.TestCase):
    """Tests for the metric hash values"""

    def test_hash_differs_between_metric_classes(self):
        """check that metrics defined in the same file with equal parameters differ"""
        self.assertNotEqual(
            fnc.evaluation.RMSMetric().method_hash,
            fnc.evaluation.MSEMetric().method_hash,
        )

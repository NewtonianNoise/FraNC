"""Test cases for different evaluation classes and functions"""

import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

import franc as fnc
from franc.evaluation.report_generation import HTMLReport

from ..toolbox import calc_mean_asd


class TestTestDataGenerator(
    unittest.TestCase
):  # yup, this is what my naming scheme yields :(
    """Test cases for the test data generator"""

    def test_output_shapes(self):
        """check that the generated data has the correct shape"""
        N_channels = 4
        tdg = fnc.evaluation.TestDataGenerator(witness_noise_level=[1] * N_channels)
        witness, target = tdg.generate(1000)

        self.assertEqual(witness.shape, (N_channels, 1000))
        self.assertEqual(target.shape, (1000,))

    def test_sr_scaling(self):
        """check that the generated noise ASD is correct"""
        sample_rate = 123.0
        w_noise_levels = [0.1, 1, 2, 3, 4]

        tdg = fnc.evaluation.TestDataGenerator(
            witness_noise_level=w_noise_levels, sample_rate=sample_rate
        )
        witness, target = tdg.generate(int(5e5))

        # test the amplitudes
        ASD_target = calc_mean_asd(target, sample_rate)
        ASD_witness = [calc_mean_asd(i, sample_rate) for i in witness]

        self.assertAlmostEqual(ASD_target, 1, places=1)
        for asd_witness, asd_expectation in zip(ASD_witness, w_noise_levels):
            self.assertAlmostEqual(
                asd_witness, np.sqrt(1 + asd_expectation**2), places=1
            )

    def test_transfer_function(self):
        """check that the transfer function amplitude is applied correctly"""
        transfer_amplitude = 3.14

        tdg = fnc.evaluation.TestDataGenerator(
            witness_noise_level=0, transfer_function=transfer_amplitude
        )
        witness, target = tdg.generate(10)

        self.assertTrue((target * transfer_amplitude == witness[0]).all())

    def test_dataset_generation(self):
        """check that generating a dataset is possible"""
        tdg = fnc.evaluation.TestDataGenerator(witness_noise_level=[1] * 3)
        self.assertIsInstance(tdg.dataset([10], [10]), fnc.evaluation.EvaluationDataset)


# there is no tesing for the residual_power_ratio function, as it is indirectly tested through the amplitude wrapper
class TestResidualAmplitudeRatio(unittest.TestCase):
    """tests for residual_amplitude_ratio() and indirectly for residual_power_ratio()"""

    def test_dc_removal(self):
        """test that the remove_dc parameter is habdled correctly"""
        a = np.array([3, 4])
        b = np.array([np.sqrt(0.5), -np.sqrt(0.5)])
        self.assertAlmostEqual(
            fnc.evaluation.residual_amplitude_ratio(a, a + b, remove_dc=False), 1 / 5
        )
        self.assertAlmostEqual(
            fnc.evaluation.residual_amplitude_ratio(a, a + b, remove_dc=True),
            np.sqrt(2),
        )


class TestMeasureRuntime(unittest.TestCase):
    """tests for residual_amplitude_ratio() and indirectly for residual_power_ratio()"""

    def test_causality(self):
        """check that results follow basic expectations"""
        result_100 = fnc.evaluation.measure_runtime(
            [fnc.filtering.WienerFilter], n_samples=int(1e4)
        )
        result_1000 = fnc.evaluation.measure_runtime(
            [fnc.filtering.WienerFilter], n_samples=int(1e5), repititions=2
        )
        result_1000_repeated = fnc.evaluation.measure_runtime(
            [fnc.filtering.WienerFilter], n_samples=int(1e5), repititions=4
        )

        self.assertLess(result_100[1][0], result_1000[1][0])
        self.assertLess(result_100[1][0], result_1000[1][0])
        for i in range(2):
            self.assertAlmostEqual(
                result_1000[i][0], result_1000_repeated[i][0], places=1
            )


class TestEvaluationRun(unittest.TestCase):
    """Test cases for the EvaluationRun class"""

    def generate_configuration(self, n_channel):
        """Generate some testing data"""
        dataset = fnc.evaluation.TestDataGenerator(
            witness_noise_level=[1] * n_channel,
        ).dataset([1000], [1000])

        method_configurations = [
            (
                fnc.filtering.WienerFilter,
                [
                    {"n_filter": 10, "idx_target": 0, "n_channel": n_channel},
                    {"n_filter": 12, "idx_target": 0},
                ],
            )
        ]
        return dataset, method_configurations

    def test_functionality(self):
        """Check basic functionality"""
        n_channel = 3

        dataset, method_configurations = self.generate_configuration(n_channel)

        # enter the same configuration twice. It should only show up once in the result
        er = fnc.evaluation.EvaluationRun(
            method_configurations * 2,
            dataset,
            fnc.evaluation.RMSMetric(),
            [fnc.evaluation.TimeSeriesMetric(stop=10)],
            directory="testing/test_outputs/",
        )

        def add_n_channel(parameters):
            return parameters | {"n_channel": n_channel}

        expected_result = [
            (filtering_method, add_n_channel(conf))
            for filtering_method, configurations in method_configurations
            for conf in configurations
        ]
        self.assertEqual(er.get_all_configurations(), expected_result)

        with redirect_stdout(StringIO()):
            results = er.run()

        # latex compiling cannot be tested without
        er.generate_report(results, compile_report=False)
        # html compilation has no effect, but running it results in better coverage
        er.generate_report(results, compile_report=True, report_type=HTMLReport)

    def test_wrong_parameters(self):
        """Check that wrong parameters lead to matching errors"""
        n_channel = 2
        dataset, method_configurations = self.generate_configuration(n_channel)

        # test passing not a filter instance
        self.assertRaises(
            TypeError,
            fnc.evaluation.EvaluationRun,
            [("not a filter instance", [{}])],
            dataset,
            fnc.evaluation.RMSMetric(),
            directory="testing/test_outputs/",
        )

        # test passing not a dataset
        self.assertRaises(
            TypeError,
            fnc.evaluation.EvaluationRun,
            method_configurations,
            "Not a dataset",
            fnc.evaluation.RMSMetric(),
            directory="testing/test_outputs/",
        )

        # test not optimization metric not being EvaluationMetricScalar
        self.assertRaises(
            TypeError,
            fnc.evaluation.EvaluationRun,
            method_configurations,
            dataset,
            fnc.evaluation.PSDMetric(),
            directory="testing/test_outputs/",
        )

        # test a metric not being EvaluationMetric
        self.assertRaises(
            TypeError,
            fnc.evaluation.EvaluationRun,
            method_configurations,
            dataset,
            fnc.evaluation.PSDMetric(),
            ["not a metric"],
            directory="testing/test_outputs/",
        )

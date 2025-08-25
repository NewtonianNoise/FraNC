import unittest
from copy import deepcopy
import numpy as np

import saftig as sg

from .toolbox import calc_mean_asd


class TestTestDataGenerator(
    unittest.TestCase
):  # yup, this is what my naming scheme yields :(
    """Test cases for the test data generator"""

    def test_output_shapes(self):
        """check that the generated data has the correct shape"""
        N_channels = 4
        tdg = sg.evaluation.TestDataGenerator(witness_noise_level=[1] * N_channels)
        witness, target = tdg.generate(1000)

        self.assertEqual(witness.shape, (N_channels, 1000))
        self.assertEqual(target.shape, (1000,))

    def test_sr_scaling(self):
        """check that the generated noise ASD is correct"""
        sample_rate = 123.0
        w_noise_levels = [0.1, 1, 2, 3, 4]

        tdg = sg.evaluation.TestDataGenerator(
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

        tdg = sg.evaluation.TestDataGenerator(
            witness_noise_level=0, transfer_function=transfer_amplitude
        )
        witness, target = tdg.generate(10)

        self.assertTrue((target * transfer_amplitude == witness[0]).all())

    def test_dataset_generation(self):
        """check that generating a dataset is possible"""
        tdg = sg.evaluation.TestDataGenerator(witness_noise_level=[1] * 3)
        self.assertIsInstance(tdg.dataset([10], [10]), sg.evaluation.EvaluationDataset)


# there is no tesing for the residual_power_ratio function, as it is indirectly tested through the amplitude wrapper
class TestResidualAmplitudeRatio(unittest.TestCase):
    """tests for residual_amplitude_ratio() and indirectly for residual_power_ratio()"""

    def test_dc_removal(self):
        """test that the remove_dc parameter is habdled correctly"""
        a = np.array([3, 4])
        b = np.array([np.sqrt(0.5), -np.sqrt(0.5)])
        self.assertAlmostEqual(
            sg.evaluation.residual_amplitude_ratio(a, a + b, remove_dc=False), 1 / 5
        )
        self.assertAlmostEqual(
            sg.evaluation.residual_amplitude_ratio(a, a + b, remove_dc=True), np.sqrt(2)
        )


class TestMeasureRuntime(unittest.TestCase):
    """tests for residual_amplitude_ratio() and indirectly for residual_power_ratio()"""

    def test_causality(self):
        """check that results follow basic expectations"""
        result_100 = sg.evaluation.measure_runtime(
            [sg.filtering.WienerFilter], n_samples=int(1e4)
        )
        result_1000 = sg.evaluation.measure_runtime(
            [sg.filtering.WienerFilter], n_samples=int(1e5), repititions=2
        )
        result_1000_repeated = sg.evaluation.measure_runtime(
            [sg.filtering.WienerFilter], n_samples=int(1e5), repititions=4
        )

        self.assertLess(result_100[1][0], result_1000[1][0])
        self.assertLess(result_100[1][0], result_1000[1][0])
        for i in range(2):
            self.assertAlmostEqual(
                result_1000[i][0], result_1000_repeated[i][0], places=1
            )


class TestEvaluationDataset(unittest.TestCase):
    """Tests for EvaluationDataset"""

    @staticmethod
    def simple_test_data(n_samples=100, n_sequences=4, n_channels=2):
        target = [np.ones(n_samples)] * n_sequences
        witness = [[sequence for channel in range(n_channels)] for sequence in target]
        return witness, target

    def test_functionality(self):
        """Check that the basic functionality works"""
        witness, target = self.simple_test_data()

        sg.evaluation.EvaluationDataset(1.0, witness, target, witness, target)
        sg.evaluation.EvaluationDataset(
            1.0, witness, target, witness, target, "Dataset Name"
        )

    def test_wrong_input(self):
        """Check that malformed input results in adequate errors"""
        from saftig.evaluation import EvaluationDataset

        witness, target = self.simple_test_data()

        self.assertRaises(
            ValueError,
            EvaluationDataset,
            "not_a_float",
            witness,
            target,
            witness,
            target,
        )
        self.assertRaises(
            ValueError,
            EvaluationDataset,
            1.0,
            witness,
            target,
            witness,
            target,
            {"not_a_string"},
        )

        self.assertRaises(
            AssertionError, EvaluationDataset, 1.0, witness, [], witness, target
        )
        self.assertRaises(
            AssertionError, EvaluationDataset, 1.0, [], target, witness, target
        )
        self.assertRaises(
            AssertionError, EvaluationDataset, 1.0, [[]], target, witness, target
        )
        self.assertRaises(
            AssertionError,
            EvaluationDataset,
            1.0,
            [witness[0][:-1]],
            target,
            witness,
            target,
        )

        self.assertRaises(
            AssertionError, EvaluationDataset, 1.0, witness, target, witness, []
        )
        self.assertRaises(
            AssertionError, EvaluationDataset, 1.0, witness, target, [], target
        )
        self.assertRaises(
            AssertionError, EvaluationDataset, 1.0, witness, target, [[]], target
        )
        self.assertRaises(
            AssertionError,
            EvaluationDataset,
            1.0,
            witness,
            target,
            [witness[0][:-1]],
            target,
        )

    def test_get_min_sequence_len(self):
        """Test get_min_sequence_len()"""
        test_data1 = [np.zeros(4), np.zeros(3), np.zeros(10)]
        test_data2 = [np.zeros(4), np.zeros(4), np.zeros(10)]

        for td1, td2 in [(test_data1, test_data2), (test_data2, test_data1)]:
            # list(zip(*x)) transposes the first two dimensions
            # using numpy arrays is not possible as the lengths of the last dimension are not consistent
            min_len = sg.evaluation.EvaluationDataset(
                1.0, list(zip(*[td1, td1])), td1, list(zip(*[td2, td2])), td2
            ).get_min_sequence_len()
            self.assertEqual(min_len, 3)

    def test_hash(self):
        """Test hashability of the object and that changes in each parameter affect the hash value."""
        from saftig.evaluation import EvaluationDataset

        # get hash for base paramters (also checks that hashing works at all)
        base_parameters = [
            1.0,
            [[np.zeros(10), np.zeros(10), np.zeros(10)]],
            [np.zeros(10)],
            [[np.zeros(10), np.zeros(10), np.zeros(10)]],
            [np.zeros(10)],
            "name",
        ]
        base_hash = hash(EvaluationDataset(*base_parameters))

        # check that hash changes for different input
        new_values = [
            2.0,
            [[np.zeros(10), np.ones(10), np.zeros(10)]],
            [np.ones(10)],
            [[np.zeros(10), np.ones(10), np.zeros(10)]],
            [np.ones(10)],
            "new_name",
        ]
        for idx, new_value in enumerate(new_values):
            new_parameters = deepcopy(base_parameters)
            new_parameters[idx] = new_value
            new_hash = hash(EvaluationDataset(*new_parameters))
            self.assertNotEqual(
                new_hash,
                base_hash,
                f"Changing parameter at position {idx} had no effect",
            )

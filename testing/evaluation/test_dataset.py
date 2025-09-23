"""Tests for EvaluationDataset"""

import unittest
from copy import deepcopy
import numpy as np

import franc as fnc


class TestEvaluationDataset(unittest.TestCase):
    """Tests for EvaluationDataset"""

    @staticmethod
    def simple_test_data(n_samples=100, n_sequences=4, n_channels=2):
        """Generate a simple dataset"""
        target = [np.ones(n_samples)] * n_sequences
        witness = [[sequence for _ in range(n_channels)] for sequence in target]
        return witness, target

    def test_functionality(self):
        """Check that the basic functionality works"""
        witness, target = self.simple_test_data()
        signal = target

        fnc.evaluation.EvaluationDataset(1.0, witness, target, witness, target)
        fnc.evaluation.EvaluationDataset(1.0, witness, target, witness, target, signal)
        fnc.evaluation.EvaluationDataset(
            1.0, witness, target, witness, target, signal, signal
        )
        fnc.evaluation.EvaluationDataset(
            1.0, witness, target, witness, target, name="Dataset Name"
        )
        fnc.evaluation.EvaluationDataset(
            1.0, witness, target, witness, target, signal, signal, "Dataset Name"
        )

    def test_wrong_input(self):
        """Check that malformed input results in adequate errors"""
        from franc.evaluation import (  # pylint: disable=import-outside-toplevel
            EvaluationDataset,
        )

        witness, target = self.simple_test_data()
        signal = target

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
            signal_evaluation="not list of npy array",
        )
        self.assertRaises(
            ValueError,
            EvaluationDataset,
            1.0,
            witness,
            target,
            witness,
            target,
            signal,
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
            min_len = fnc.evaluation.EvaluationDataset(
                1.0, list(zip(*[td1, td1])), td1, list(zip(*[td2, td2])), td2
            ).get_min_sequence_len()
            self.assertEqual(min_len, 3)

    def test_hash(self):
        """Test hashability of the object and that changes in each parameter affect the hash value."""
        from franc.evaluation import (  # pylint: disable=import-outside-toplevel
            EvaluationDataset,
        )

        # get hash for base paramters (also checks that hashing works at all)
        base_parameters = [
            1.0,
            [[np.zeros(10), np.zeros(10), np.zeros(10)]],
            [np.zeros(10)],
            [[np.zeros(10), np.zeros(10), np.zeros(10)]],
            [np.zeros(10)],
            [np.zeros(10)],
            [np.zeros(10)],
            "name",
        ]
        base_hash = hash(EvaluationDataset(*base_parameters))

        # check that hashing works with minimal parameter count
        self.assertIsInstance(hash(EvaluationDataset(*base_parameters[:5])), int)

        # check that hash changes for different input
        new_values = [
            2.0,
            [[np.zeros(10), np.ones(10), np.zeros(10)]],
            [np.ones(10)],
            [[np.zeros(10), np.ones(10), np.zeros(10)]],
            [np.ones(10)],
            [np.ones(10)],
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

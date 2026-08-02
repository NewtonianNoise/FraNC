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
        fnc.evaluation.EvaluationDataset(
            1.0, witness, target, witness, target, signal, signal
        )
        fnc.evaluation.EvaluationDataset(
            1.0, witness, target, witness, target, name="Dataset Name"
        )
        fnc.evaluation.EvaluationDataset(
            1.0, witness, target, witness, target, signal, signal, "Dataset Name"
        )

    def test_immutability(self):
        """EvaluationDataset must store data as read-only copies: mutating the stored
        arrays must fail, and mutating the caller's original input after construction
        must not be reflected in the dataset."""
        witness = [[np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]]
        target = [np.array([1.0, 2.0, 3.0])]
        signal = [np.array([1.0, 2.0, 3.0])]

        dataset = fnc.evaluation.EvaluationDataset(
            1.0, witness, target, witness, target, signal, signal
        )

        # the dataset's stored arrays must be read-only
        for array in [
            dataset.witness_conditioning[0][0],
            dataset.target_conditioning[0],
            dataset.witness_evaluation[0][0],
            dataset.target_evaluation[0],
            dataset.signal_conditioning[0],
            dataset.signal_evaluation[0],
        ]:
            with self.assertRaises(ValueError):
                array[0] = 42

        # mutating the caller's original input after construction must not leak in
        target[0][0] = 999
        witness[0][0][0] = 999
        signal[0][0] = 999
        self.assertEqual(dataset.target_conditioning[0][0], 1.0)
        self.assertEqual(dataset.witness_conditioning[0][0][0], 1.0)
        self.assertEqual(dataset.signal_conditioning[0][0], 1.0)

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
            ValueError, EvaluationDataset, 1.0, witness, [], witness, target
        )
        self.assertRaises(
            ValueError, EvaluationDataset, 1.0, [], target, witness, target
        )
        self.assertRaises(
            ValueError, EvaluationDataset, 1.0, [[]], target, witness, target
        )
        self.assertRaises(
            ValueError,
            EvaluationDataset,
            1.0,
            [witness[0][:-1]],
            target,
            witness,
            target,
        )

        self.assertRaises(
            ValueError, EvaluationDataset, 1.0, witness, target, witness, []
        )
        self.assertRaises(
            ValueError, EvaluationDataset, 1.0, witness, target, [], target
        )
        self.assertRaises(
            ValueError, EvaluationDataset, 1.0, witness, target, [[]], target
        )
        self.assertRaises(
            ValueError,
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
            "unit",
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
            "new_unit",
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

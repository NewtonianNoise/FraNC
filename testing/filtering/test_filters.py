"""Parent class to test filter implementations"""

import unittest
from typing import Iterable, TypeVar, Generic
import warnings

import numpy as np

import franc as fnc

# file used to test saving and loading of filters
TEST_FILE = "testing/filter_serialization_test_file"

RNG_SEED = 113510

T = TypeVar("T", bound=fnc.filtering.FilterBase)


class TestFilter:  # pylint: disable=too-few-public-methods
    """this wrapper class prevents automatic test detections
    from finding the parent class and attempting to run it
    """

    class TestFilter(unittest.TestCase, Generic[T]):
        """Parent class for all filter testing
        Contains common test cases and testing tools
        """

        __test__ = False

        # The to-be-tested filter class
        target_filter: type[T]
        # to-be-tested configurations
        default_filter_parameters: list = [{}]

        # settings to check performance
        expected_performance = {
            # noise level, (acceptance min, acceptance_max)
            0.0: (0, 0.05),
            0.1: (0.05, 0.15),
        }

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__test__ = False

        def set_target(
            self, target_filter: type[T], default_filter_parameters=None
        ) -> None:
            """set the target filter and configurations
            This is required to run the common tests

            :param target_filter: The to-be-tested filter class
            :param default_filter_parameters: A list of all configuration for which the tests will be run
            """
            self.target_filter = target_filter
            self.default_filter_parameters = (
                [{}]
                if (default_filter_parameters is None)
                else default_filter_parameters
            )
            assert isinstance(self.default_filter_parameters, list)

        def instantiate_filters(
            self, n_channel=1, n_filter=128, idx_target=0
        ) -> Iterable[T]:
            """instantiate the target filter for all configurations"""
            for parameters in self.default_filter_parameters:
                yield self.target_filter(
                    n_channel=n_channel,
                    n_filter=n_filter,
                    idx_target=idx_target,
                    **parameters,
                )

        def test_exception_on_missshaped_input(self):
            """Check that matching exceptions are thrown for obviously wrong input shapes"""
            n_filter = 128
            witness, target = fnc.evaluation.TestDataGenerator(
                0.1, rng_seed=RNG_SEED
            ).generate(int(1e4))

            for filt in self.instantiate_filters(n_filter=n_filter):
                with warnings.catch_warnings():  # warnings are expected here
                    warnings.simplefilter("ignore")
                    filt.condition(witness, target)
                self.assertRaises(ValueError, filt.apply, 1, 1)
                self.assertRaises(ValueError, filt.apply, [[[1], [1]]], [1])

        def test_acceptance_of_minimum_input_length(self):
            """Check that the filter works with the minimum input length of two filter lengths"""
            n_filter = 128
            witness, target = fnc.evaluation.TestDataGenerator(
                0.1, rng_seed=RNG_SEED
            ).generate(n_filter * 2)

            for filt in self.instantiate_filters(n_filter=n_filter):
                with warnings.catch_warnings():  # warnings are expected here
                    warnings.simplefilter("ignore")
                    filt.condition(witness, target)
                    filt.apply(witness, target)

        def test_acceptance_of_lists(self):
            """Check that the filter accepts inputs that are not np.ndarray"""
            n_filter = 128
            witness, target = fnc.evaluation.TestDataGenerator(
                0.1, rng_seed=RNG_SEED
            ).generate(n_filter * 2)

            for filt in self.instantiate_filters(n_filter=n_filter):
                with warnings.catch_warnings():  # warnings are expected here
                    warnings.simplefilter("ignore")
                    filt.condition(witness.tolist(), target.tolist())
                    filt.apply(witness.tolist(), target.tolist())

        def test_output_shapes(self):
            """Check output shapes"""
            n_filter = 128
            witness, target = fnc.evaluation.TestDataGenerator(
                0.1, rng_seed=RNG_SEED
            ).generate(int(1e4))

            for filt in self.instantiate_filters(n_filter=n_filter):
                with warnings.catch_warnings():  # warnings are expected here
                    warnings.simplefilter("ignore")
                    filt.condition(witness, target)

                # with padding
                prediction = filt.apply(witness, target)
                self.assertEqual(prediction.shape, target.shape)

                # without padding
                prediction = filt.apply(witness, target, pad=False)
                self.assertEqual(len(prediction), len(target) - n_filter + 1)

        def test_multi_sequence(self):
            """Test that multi sequence functions work is supports_multi_sequence is set"""
            if not self.target_filter.supports_multi_sequence:
                return
            n_filter = 128
            witness, target = fnc.evaluation.TestDataGenerator(
                0.1, rng_seed=RNG_SEED
            ).generate_multiple([int(1e4), int(2e4)])

            for filt in self.instantiate_filters(n_filter=n_filter):
                with warnings.catch_warnings():  # warnings are expected here
                    warnings.simplefilter("ignore")
                    filt.condition_multi_sequence(witness, target)

                # with padding
                prediction = filt.apply_multi_sequence(witness, target)
                for p, t in zip(prediction, target):
                    self.assertEqual(p.shape, t.shape)

                # without padding
                prediction = filt.apply_multi_sequence(witness, target, pad=False)
                for p, t in zip(prediction, target):
                    self.assertEqual(len(p), len(t) - n_filter + 1)

        def test_apply_on_unconditioned_filter(self):
            """Check that calling apply() on an unconditioned filter either works or throws an RuntimeError"""
            n_filter = 128
            witness, target = fnc.evaluation.TestDataGenerator(
                0.1, rng_seed=RNG_SEED
            ).generate(int(1e4))

            for filt in self.instantiate_filters(n_filter=n_filter):
                try:
                    filt.apply(witness, target)
                except RuntimeError:
                    pass

        def test_performance(self):
            """Check that the filter reaches a WF-Like performance on a simple static test case"""
            for noise_level, acceptable_residual in self.expected_performance.items():
                n_filter = 32
                witness, target = fnc.evaluation.TestDataGenerator(
                    [noise_level] * 2, rng_seed=RNG_SEED
                ).generate(int(2e4))

                for idx_target in [0, int(n_filter / 2), n_filter - 1]:
                    for filt in self.instantiate_filters(
                        n_channel=2, n_filter=n_filter, idx_target=idx_target
                    ):
                        with warnings.catch_warnings():  # warnings are expected here
                            warnings.simplefilter("ignore")
                            filt.condition(witness, target)
                            prediction = filt.apply(witness, target)

                        residual = fnc.evaluation.rms((target - prediction)[4000:])

                        self.assertGreater(residual, acceptable_residual[0])
                        self.assertLess(residual, acceptable_residual[1])

        def test_from_wrong_dict(self):
            """Check that attemting to load from an incompatible dict fails"""
            if self.target_filter.supports_saving_loading():
                self.assertRaises(ValueError, self.target_filter.from_dict, {})

                ref_dict = self.target_filter(1, 10, 0).as_dict()
                ref_dict["filter_name"] = "not_a_filter_name"
                self.assertRaises(ValueError, self.target_filter.from_dict, ref_dict)
            else:
                self.assertRaises(NotImplementedError, self.target_filter.from_dict, {})

        def test_saving_loading(self):
            """Test that saving and loading works correctly.
            Or check that NotImplemented errors are thrown if self.do_saving_loading_tests is False.
            """
            # generate test data
            n_filter = 32
            witness, target = fnc.evaluation.TestDataGenerator(
                [0.1] * 2, rng_seed=RNG_SEED
            ).generate(int(2e4))

            for filt in self.instantiate_filters(n_channel=2, n_filter=n_filter):

                if filt.supports_saving_loading():
                    filt.condition(witness, target)
                    filt.save(TEST_FILE)
                    loaded_filter = self.target_filter.load(TEST_FILE)

                    self.assertEqual(
                        filt.method_hash,
                        loaded_filter.method_hash,
                    )

                    prediction_orig = filt.apply(witness, target)
                    prediction_loaded = loaded_filter.apply(witness, target)

                    self.assertAlmostEqual(
                        np.sum(prediction_orig), np.sum(prediction_loaded)
                    )
                else:
                    self.assertRaises(NotImplementedError, filt.save, TEST_FILE)
                    self.assertRaises(
                        NotImplementedError, self.target_filter.load, TEST_FILE
                    )

        def test_hashing(self):
            """Check that hashing works for the filter instances"""
            old_hash = None
            for filt in self.instantiate_filters(n_channel=2, n_filter=10):
                new_hash = filt.method_hash

                self.assertIsInstance(new_hash, bytes)
                self.assertNotEqual(new_hash, old_hash)
                old_hash = new_hash

from typing import Iterable
import warnings

import saftig as sg


class TestFilter:
    """Parent class for all filter testingr
    Contains common test cases and testing tools
    """

    __test__ = False

    # The to-be-tested filter class
    target_filter: type[sg.filtering.FilterBase]
    # to-be-tested configurations
    default_filter_parameters: list = [{}]

    # settings to check performance
    expected_performance = {
        # noise level, (acceptance min, acceptance_max)
        0.0: (0, 0.05),
        0.1: (0.05, 0.15),
    }

    def __init__(self):
        raise RuntimeError("This only functions as a parent class!")

    def set_target(self, target_filter, default_filter_parameters=[{}]) -> None:
        """set the target filter and configurtions
        This is required to run the common tests

        :param target_filter: The to-be-tested filter class
        :param default_filter_parameters: A list of all configuration for which the tests will be run
        """
        assert isinstance(default_filter_parameters, list)
        self.target_filter = target_filter
        self.default_filter_parameters = default_filter_parameters

    def instantiate_filters(
        self, n_filter=128, idx_target=0, n_channel=1
    ) -> Iterable[sg.filtering.FilterBase]:
        """instantiate the target filter for all configurations"""
        for parameters in self.default_filter_parameters:
            yield self.target_filter(n_filter, idx_target, n_channel, **parameters)

    def test_exception_on_missshaped_input(self):
        """Check that matching exceptions are thrown for obviously wrong input shapes"""
        n_filter = 128
        witness, target = sg.evaluation.TestDataGenerator(0.1).generate(int(1e4))

        for filt in self.instantiate_filters(n_filter):
            with warnings.catch_warnings():  # warnings are expected here
                warnings.simplefilter("ignore")
                filt.condition(witness, target)
            self.assertRaises(ValueError, filt.apply, 1, 1)
            self.assertRaises(ValueError, filt.apply, [[[1], [1]]], [1])

    def test_acceptance_of_minimum_input_length(self):
        """Check that the filter works with the minimum input length of two filter lengths"""
        n_filter = 128
        witness, target = sg.evaluation.TestDataGenerator(0.1).generate(n_filter * 2)

        for filt in self.instantiate_filters(n_filter):
            with warnings.catch_warnings():  # warnings are expected here
                warnings.simplefilter("ignore")
                filt.condition(witness, target)
                filt.apply(witness, target)

    def test_acceptance_of_lists(self):
        """Check that the filter accepts inputs that are not np.ndarray"""
        n_filter = 128
        witness, target = sg.evaluation.TestDataGenerator(0.1).generate(n_filter * 2)

        for filt in self.instantiate_filters(n_filter):
            with warnings.catch_warnings():  # warnings are expected here
                warnings.simplefilter("ignore")
                filt.condition(witness.tolist(), target.tolist())
                filt.apply(witness.tolist(), target.tolist())

    def test_output_shapes(self):
        """Check output shapes"""
        n_filter = 128
        witness, target = sg.evaluation.TestDataGenerator(0.1).generate(int(1e4))

        for filt in self.instantiate_filters(n_filter):
            with warnings.catch_warnings():  # warnings are expected here
                warnings.simplefilter("ignore")
                filt.condition(witness, target)

            # with padding
            prediction = filt.apply(witness, target)
            self.assertEqual(prediction.shape, target.shape)

            # without padding
            prediction = filt.apply(witness, target, pad=False)
            self.assertEqual(len(prediction), len(target) - n_filter + 1)

    def test_apply_on_unconditioned_filter(self):
        """Check that calling apply() on an unconditioned filter either works or throws an RuntimeError"""
        n_filter = 128
        witness, target = sg.evaluation.TestDataGenerator(0.1).generate(int(1e4))

        for filt in self.instantiate_filters(n_filter):
            try:
                filt.apply(witness, target)
            except RuntimeError:
                pass

    def test_performance(self):
        """Check that the filter reaches a WF-Like performance on a simple static test case"""
        for noise_level, acceptable_residual in self.expected_performance.items():
            n_filter = 32
            witness, target = sg.evaluation.TestDataGenerator(
                [noise_level] * 2
            ).generate(int(2e4))

            for idx_target in [0, int(n_filter / 2), n_filter - 1]:
                for filt in self.instantiate_filters(
                    n_filter, n_channel=2, idx_target=idx_target
                ):
                    with warnings.catch_warnings():  # warnings are expected here
                        warnings.simplefilter("ignore")
                        filt.condition(witness, target)
                        prediction = filt.apply(witness, target)

                    residual = sg.evaluation.RMS((target - prediction)[4000:])

                    self.assertGreater(residual, acceptable_residual[0])
                    self.assertLess(residual, acceptable_residual[1])

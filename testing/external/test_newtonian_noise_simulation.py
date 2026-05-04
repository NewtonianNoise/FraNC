"""Tests for the NewtonianNoiseDataGenerator"""

import unittest
import franc as fnc


class TestNewtonianNoiseDataGenerator(unittest.TestCase):
    """Tests for the NewtonianNoiseDataGenerator"""

    def test_functionality(self):
        """Check that the basic functionality works"""

        generator = fnc.eval.NewtonianNoiseDataGenerator(
            folder="testing/test_outputs",
            NoR=3,
        )

        try:
            generator.generateEventSet(tag="test")
            generator.generateDataset(tag="test")
            generator.deleteEventSet(tag="test")
        except NameError:
            pass

        try:
            generator.generateEventSet(
                tag="test", isMonochromatic=True, anisotropy="p50"
            )
            generator.generateDataset(tag="test")
            generator.deleteEventSet(tag="test")
        except NameError:
            pass


# obj=TestNewtonianNoiseDataGenerator()
# obj.test_functionality()

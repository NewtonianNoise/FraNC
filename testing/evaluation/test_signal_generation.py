"""Parent class to test filter implementations"""

import unittest
import numpy as np

import franc as fnc


class TestWavePacketGeneration(unittest.TestCase):
    """Test cases for generate_wave_packet_signal and generate_wave_packet"""

    def test_functionality(self):
        """Test that the function is callable"""
        values = fnc.eval.signal_generation.generate_wave_packet_signal(
            1000,
            10,
            (10, 20),
            (1, 10),
            (0.03, 0.2),
            np.random.default_rng(0xABCDE),
        )
        self.assertIsInstance(values[0], np.ndarray)

    def test_peak_scaling(self):
        """Test that the amplitude bounds the packet independent of the offset"""
        for offset in [0, 100, 400, -250]:
            packet = fnc.eval.signal_generation.generate_wave_packet(
                offset, 100.0, 3.0, 0.02, np.pi / 2
            )
            self.assertAlmostEqual(np.max(np.abs(packet)), 3.0)

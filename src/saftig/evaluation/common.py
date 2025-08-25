"""Shared functionality for all other modules"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def total_power(a: Sequence | NDArray) -> float:
    """calculate the total power of a signal (square or RMS)

    >>> import saftig, numpy
    >>> signal = numpy.ones(10) * 2
    >>> saftig.evaluation.total_power(signal)
    4.0

    """
    a_npy: NDArray = np.array(a)
    return float(np.mean(np.square(a_npy)))


def rms(a: Sequence | NDArray) -> float:
    """Calculate the root mean square value of an array"""
    a_npy: NDArray = np.array(a)

    # float() is used to convert this into a standard float instead of a 0D numpy array
    # this simplifies writing doctests
    return float(np.sqrt(np.mean(np.square(a_npy))))

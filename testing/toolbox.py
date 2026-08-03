"""shared tools for testing"""

from __future__ import annotations

from typing import Iterable
import numpy as np
from scipy.signal import welch


def calc_mean_asd(
    A: Iterable[float],
    sample_rate: float = 1.0,
    f_band: tuple[float, float] | None = None,
    nperseg: int | None = None,
):
    """calculate the mean ASD for a given time series A at given sample_rate

    :param f_band: (Optional) restrict the calculation to [f_band[0], f_band[1]) Hz
    :param nperseg: (Optional) passed to scipy.signal.welch; samples per FFT
    """
    freqs, psd = welch(A, fs=sample_rate, nperseg=nperseg)
    if f_band is not None:
        psd = psd[(freqs >= f_band[0]) & (freqs < f_band[1])]
    return np.sqrt(np.mean(psd))

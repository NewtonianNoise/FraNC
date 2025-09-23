"""Static & Adaptive Filtering In Gravitational-wave-research
Implementations of prediction techniques with a unified interface.
"""

from franc import evaluation
from franc import filtering
from franc import external

eval = evaluation  # pylint: disable=redefined-builtin
filt = filtering

__all__ = [
    "eval",
    "filt",
    "external",
    "evaluation",
    "filtering",
]

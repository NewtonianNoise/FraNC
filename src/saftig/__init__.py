"""Static & Adaptive Filtering In Gravitational-wave-research
Implementations of prediction techniques with a unified interface.
"""

from saftig import evaluation
from saftig import filtering
from saftig import external

eval = evaluation  # pylint: disable=redefined-builtin
filt = filtering

__all__ = [
    "eval",
    "filt",
    "external",
    "evaluation",
    "filtering",
]

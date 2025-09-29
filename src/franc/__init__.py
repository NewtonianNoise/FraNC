"""Static & Adaptive Filtering In Gravitational-wave-research
Implementations of prediction techniques with a unified interface.
"""

from franc import evaluation
from franc import filtering
from franc import external

eval = evaluation  # pylint: disable=redefined-builtin
"""shorthand for franc.evaluation"""

filt = filtering
"""shorthand for franc.filtering"""

__all__ = [
    "eval",
    "filt",
    "external",
    "evaluation",
    "filtering",
]


# expose package version
def get_package_version():
    """wrapping function to hide import statement from doctest discovery"""
    import importlib.metadata  # pylint: disable=import-outside-toplevel

    return importlib.metadata.version(__name__)


__version__ = get_package_version()

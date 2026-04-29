"""Wrappers for other implementations of filtering techniques."""

__all__ = []

try:
    from .spicypy_wf import SpicypyWienerFilter

    __all__.append("SpicypyWienerFilter")
except ImportError:
    import warnings

    warnings.warn("To use spicipy filters, install spicypy", RuntimeWarning)

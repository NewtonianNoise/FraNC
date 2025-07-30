"""Implementations of filtering techniques with a common interface."""

from .common import FilterBase
from .wf import WienerFilter
from .uwf import UpdatingWienerFilter
from .lms import LMSFilter
from .polylms import PolynomialLMSFilter

from .lms_c import LMSFilterC

#: A list of all filters for automated testing and comparisons
all_filters = [
    FilterBase,
    WienerFilter,
    UpdatingWienerFilter,
    LMSFilter,
    LMSFilterC,
    PolynomialLMSFilter,
]

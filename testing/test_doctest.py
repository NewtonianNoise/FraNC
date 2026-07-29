"""Load doctests in the main module as unittests"""

import doctest
import numpy as np

import franc

# numpy's array repr line-wraps trailing annotations (e.g. "shape=(...)") differently
# across python and numpy versions in ways that are whitespace-only, not a real
# behavioural difference. The same flag is set for the documentation build in
# doc/source/conf.py and for pytest --doctest-modules in pyproject.toml.
OPTION_FLAGS = doctest.NORMALIZE_WHITESPACE


# NOTE: pytest does not support the load_tests() paradigm
# See: https://docs.pytest.org/en/7.1.x/how-to/unittest.html
# Therefore, running the doctests separately is imperative if this test suite is run through pytest
def load_tests(_loader, tests, _ignore):
    """load doctests as unittests"""
    tests.addTests(doctest.DocTestSuite(franc, optionflags=OPTION_FLAGS))

    if np.__version__[0] == "1":
        # skip doctests for old numpy versions
        # as they match the recent releases
        return tests

    for submodule_name in dir(franc):
        submodule = getattr(franc, submodule_name)
        if "__file__" in dir(submodule):
            tests.addTests(doctest.DocTestSuite(submodule, optionflags=OPTION_FLAGS))

            for subsubmodule_name in dir(submodule):
                subsubmodule = getattr(submodule, subsubmodule_name)
                if "__file__" in dir(subsubmodule):
                    tests.addTests(
                        doctest.DocTestSuite(subsubmodule, optionflags=OPTION_FLAGS)
                    )
    return tests

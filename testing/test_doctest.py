"""Load doctests in the main module as unittests"""

import doctest
import franc


# NOTE: pytest does not support the load_tests() paradigm
# See: https://docs.pytest.org/en/7.1.x/how-to/unittest.html
# Therefore, running the doctests separately is imperative if this test suite is run through pytest
def load_tests(_loader, tests, _ignore):
    """load doctests as unittests"""
    tests.addTests(doctest.DocTestSuite(franc))

    for submodule_name in dir(franc):
        submodule = getattr(franc, submodule_name)
        if "__file__" in dir(submodule):
            tests.addTests(doctest.DocTestSuite(submodule))

            for subsubmodule_name in dir(submodule):
                subsubmodule = getattr(submodule, subsubmodule_name)
                if "__file__" in dir(subsubmodule):
                    tests.addTests(doctest.DocTestSuite(subsubmodule))
    return tests

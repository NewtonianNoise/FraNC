import doctest
import saftig


def load_tests(_loader, tests, _ignore):
    """load doctests as unittests"""
    tests.addTests(doctest.DocTestSuite(saftig))

    for submodule_name in dir(saftig):
        submodule = getattr(saftig, submodule_name)
        if "__file__" in dir(submodule):
            tests.addTests(doctest.DocTestSuite(submodule))

            for subsubmodule_name in dir(submodule):
                subsubmodule = getattr(submodule, subsubmodule_name)
                if "__file__" in dir(subsubmodule):
                    tests.addTests(doctest.DocTestSuite(subsubmodule))
    return tests

#!/bin/sh
pylint --rcfile=tooling/pylint_testing.rc --fail-under=10 $(git ls-files 'src/testing/*.py' 'tooling/*.py' 'tooling/*/*.py')

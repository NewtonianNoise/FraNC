#!/bin/sh
pylint --rcfile=pylint.rc --fail-under=10 $(git ls-files 'src/saftig/*.py' 'tooling/*.py' 'tooling/*/*.py')

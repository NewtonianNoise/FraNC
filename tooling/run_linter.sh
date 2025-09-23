#!/bin/sh
pylint --rcfile=pylint.rc --fail-under=10 $(git ls-files 'src/*.py' 'tooling/*.py' 'testing/*.py')

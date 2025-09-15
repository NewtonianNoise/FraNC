#!/bin/sh
mypy $(git ls-files 'src/saftig/*.py' 'tooling/*.py' 'testing/*.py' 'examples/*.py')

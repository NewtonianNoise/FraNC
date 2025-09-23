#!/bin/sh
mypy $(git ls-files 'src/*.py' 'tooling/*.py' 'testing/*.py' 'examples/*.py')

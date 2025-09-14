#!/bin/sh
basedpyright $(git ls-files 'src/saftig/*.py' 'tooling/*.py' 'tooling/*/*.py')

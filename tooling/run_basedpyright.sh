#!/bin/sh
basedpyright $(git ls-files 'src/*.py' 'tooling/*.py' 'testing/*.py')

SHELL := /bin/bash

all: ie doc coverage linter mypy
ie:
	pip install --no-build-isolation -e ".[dev]"

test:
	python -m unittest discover .
pytest:
	pytest --doctest-modules src/franc/
	pytest
test_time:
	python -m unittest discover . --duration=5
test_nojit:
	export NUMBA_DISABLE_JIT=1 && python -m unittest discover .
test_coverage:
	export NUMBA_DISABLE_JIT=1 && coverage run -m unittest discover .
coverage: test_coverage
	coverage report
cweb: test_coverage
	coverage html && open htmlcov/index.html

linter:
	./tooling/run_linter.sh
pylint: linter

mypy:
	./tooling/run_mypy.sh

documentationtest:
	-rm -r doc/test_outputs/*
	touch doc/test_outputs/.gitkeep
	cd doc/ && $(MAKE) doctest
doc: documentationtest
	cd doc/ && $(MAKE) html

view: doc
	open doc/build/html/index.html

clean:
	-rm -r build/
	-rm -r dist/
	-rm -r franc/__pycache__/
	-rm *.so
	-rm franc/*.so
	-rm -r FRANC.egg-info/
	-rm -r htmlcov

testpublish:
	python -m build -s
	twine upload --repository testpypi dist/*

.PHONY: all, ie, test, pytest, test_time, test_nojit, test_coverage, coverage, cweb, linter, pylint, lt, mypy, doc, documentationtest, view, clean, testpublish

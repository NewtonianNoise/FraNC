SHELL := /bin/bash

all: build doc coverage linter mypy
ie:
	pip install --no-build-isolation -e .

test:
	python -m unittest discover .
pytest:
	pytest --doctest-modules src/saftig/
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
linter_testing:
	./tooling/run_linter_tests.sh
lt: linter_testing

mypy:
	mypy .

doc: doc/source/* doc/*
	cd doc/ && $(MAKE) html

view: doc
	open doc/build/html/index.html

clean:
	-rm -r build/
	-rm -r dist/
	-rm -r saftig/__pycache__/
	-rm *.so
	-rm saftig/*.so
	-rm -r SAFTIG.egg-info/
	-rm -r htmlcov

testpublish:
	python -m build -s
	twine upload --repository testpypi dist/*

.PHONY: all, doc, view, test, test_time, linter, coverage, cweb, linter_testing, lt, mypy, build, clean, testpublish, ie, test_coverage

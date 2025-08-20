# Development Environment

To install the package as editable and install all developement dependencies run
```bash
pip install --no-build-isolation -e ".[dev]"
```

To use the git pre-commit hooks run
```bash
pre-commit install
```

## Useful commands
```bash
make # run linter, testing and generate documentation
make test # run just the tests
make view # build and open documentation
make coverage # report test coverage in terminal
make cweb # full test coverage report with html

make ie # install as editable package
make testpublish # build and push to test pypi

# run an individual test
python -m unittest testing.test_wf
```

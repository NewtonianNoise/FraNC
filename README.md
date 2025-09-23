# FraNC - Framework for Noise Cancellation in gravitational wave detection

![Test status](https://github.com/NewtonianNoise/franc/actions/workflows/testing.yml/badge.svg)
![Linting status](https://github.com/NewtonianNoise/franc/actions/workflows/pylint.yml/badge.svg)
![Static type check status](https://github.com/NewtonianNoise/franc/actions/workflows/mypy.yml/badge.svg)

A framework to develop and evaluate noise cancellation techniques.
Includes python implementations of different static and adaptive filtering techniques.
The techniques for the prediction of a correlated signal component from witness signals provide a unified interface.

[Documentation](https://franc.readthedocs.io/en/latest/)

## Filtering methods

Static:

* Wiener Filter (WF)

Adaptive

* Updating Wiener Fitler (UWF)
* Least-Mean-Squares Filter (LMS)

Non-Linear:

* Experimental non-linear LMS Filter variant (PolynomialLMS)

## Install

From pypi: `pip install franc`

From repository: `pip install .`

From repository (editable): `make ie`

## Minimal example

```python
>>> import franc as fnc
>>>
>>> # generate data
>>> n_channel = 2
>>> witness, target = fnc.eval.TestDataGenerator([0.1]*n_channel).generate(int(1e5))
>>>
>>> # instantiate the filter and apply it
>>> filt = fnc.filt.LMSFilter(n_filter=128, idx_target=0, n_channel=n_channel)
>>> filt.condition(witness, target)
>>> prediction = filt.apply(witness, target) # check on the data used for conditioning
>>>
>>> # success
>>> fnc.eval.RMS(target-prediction) / fnc.eval.RMS(prediction)
0.08221177645361015
```

## Terminology

* Witness signal w: One or multiple sensors that are used to make a prediction
* Target signal s: The goal for the prediction

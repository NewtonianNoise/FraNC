Filtering techniques
*********************

This framework provides implementations of some noise prediction techniques.

Built-in filtering techniques
==============================

**Static:**

* Wiener Filter (WF)

**Adaptive:**

* Updating Wiener Fitler (UWF)
* Least-Mean-Squares Filter (LMS)

**Non-Linear:**

* Experimental non-linear LMS Filter variant (PolynomialLMS)

Minimal example
================

A minimal example of how the filtering techniques can be used. All techniques follow the same interface concept.

.. doctest::

    >>> import franc as fnc
    >>>
    >>> # generate data
    >>> n_channel = 2
    >>> witness, target = fnc.eval.TestDataGenerator([0.1]*n_channel, rng_seed=123).generate(int(1e5))
    >>>
    >>> # instantiate the filter and apply it
    >>> filt = fnc.filt.LMSFilter(n_filter=128, idx_target=0, n_channel=n_channel)
    >>> filt.condition(witness, target)
    >>> prediction = filt.apply(witness, target) # check on the data used for conditioning
    >>>
    >>> # success
    >>> round(fnc.eval.rms(target-prediction) / fnc.eval.rms(prediction), 10)
    0.0815971935

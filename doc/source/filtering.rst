Filtering techniques
*********************

Features
=========

**Static:**

* Wiener Filter (WF)

**Adaptive:**

* Updating Wiener Fitler (UWF)
* Least-Mean-Squares Filter (LMS)

**Non-Linear:**

* Experimental non-linear LMS Filter variant (PolynomialLMS)

Minimal example
================


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
    >>> fnc.eval.rms(target-prediction) / fnc.eval.rms(prediction)
    0.08159719348131059





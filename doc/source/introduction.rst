Introduction
*************

This is a python implementations of different static and adaptive filtering techniques for the prediction of a correlated signal component from witness signals. The main goal is to provide a unified interface for the different filtering techniques.

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


.. code-block:: python

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

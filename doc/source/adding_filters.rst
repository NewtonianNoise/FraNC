Adding filtering techniques 
*************************************

Filters must inherit from :py:class:`franc.evaluation.FilterInterface` to be usable with this framework.
The filter interface includes the following aspects:

* Must have the `@dataclass` decorator

  * This enables saving all class variables through the `dataclasses.asdict()` interface.
  * All values used here must be exportable through the `numpy.save` interface without using `pickle` to support saving / loading

* Must declare a `filter_name`
* Can overwrite the :py:func:`franc.evaluation.FilterInterface.supports_saving_loading` function

  * By default, it is assumed that saving/loading is supported. Overwriting the function to return `False` disables all related functionality and arbitrary class variables can be used. 

* Must provide a :py:func:`franc.evaluation.FilterInterface.condition_multi_sequence` function

  * This function receives the conditioning data and initializes the filter
  * If a method does not require/support conditioning, this method must be implemented and can just do nothing

* Must provide a :py:func:`franc.evaluation.FilterInterface.apply_multi_sequence` function

  * This function applies the filter to the given data and returns the predicted sequences.


Minimal example
================

This is a minimal example implementing a simple multichannel `FIR <https://en.wikipedia.org/wiki/Finite_impulse_response>`_ filter.


.. testcode::

   from dataclasses import dataclass
   import numpy as np
   from scipy.signal import correlate
   import franc
 
   @dataclass
   class FIRFilter(franc.evaluation.FilterInterface):
       filter_coefficients:np.typing.NDArray
       filter_name = "FIR Filter"
   
       # the handle_from_dict decorator is required to support saving and loading
       @franc.evaluation.handle_from_dict
       def __init__(self, n_channel, filter_coefficients):
           # parameters must be passed to parent constructor for hashing
           super().__init__(n_channel, filter_coefficients)
 
           if len(filter_coefficients) != n_channel:
               raise ValueError( "filter_coefficitnes length of " + \
                   f"{len(filter_coefficients)} does not match n_channel={n_channel}" )
           self.filter_coefficients = np.array(filter_coefficients).astype(np.longdouble)
           self.n_filter = len(self.filter_coefficients[0])
   
       def condition_multi_sequence(self, witness, target):
           # not needed here
           pass
   
       def apply_multi_sequence(self, witness, target, pad, update_state):
           """Apply the filter to input data
   
           :param witness: Witness sensor data
           :param target: Target sensor data (is ignored)
           :param pad: if True, apply padding zeros so that the length matches the target signal
           :param update_state: ignored
   
           :return: prediction
           """
           witness, target = self.check_data_dimensions_multi_sequence(witness, target)
   
           predictions = []
           for w_sequence in witness:
               prediction_sequence = np.sum(
                   [ correlate(inpt, coeffcients, mode="valid")
                       for inpt, coeffcients
                       in zip(w_sequence, self.filter_coefficients) ],
                   axis=0,
               )
               if pad:
                   prediction_sequence = np.concatenate(
                       [
                           np.zeros(self.n_filter - 1),
                           prediction_sequence,
                       ]
                   )
               predictions.append(prediction_sequence)
           return predictions

.. testcleanup::

   # code to test that the filter defined above actually works as intended
   n_channel = 3
   n_filter = 10
   generator = franc.eval.TestDataGenerator([0.1]*n_channel, rng_seed=123)
   dataset = generator.dataset([int(1e3)], [int(1e4)])

   filt = FIRFilter(n_channel, np.ones((n_channel, n_filter)))
   filt.condition(dataset.witness_conditioning[0], dataset.target_conditioning[0])
   prediction = filt.apply(dataset.witness_evaluation[0], dataset.target_evaluation[0], pad=True)

   assert len(dataset.target_evaluation[0]) == len(prediction)

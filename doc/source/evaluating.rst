Evaluating techniques on a dataset
***********************************

Defining a dataset
===================

A :py:class:`franc.evaluation.dataset` can be instantiated by providing the required sequences of samples and a sampling rate.
The format is intended to support multiple measurement sequences of different lengths.

Target data must be provided as a sequence of sequences. The first index is the measurement sequence; the second index is the time axis within the sequence.

Witness data has three indices. First sequence, then witness channel, and last the time axis.

The following example generates a dataset with completely random data to explain the interface.

.. testcode::
   
   import numpy as np
   import franc


   # define test data
   n_channel = 3
   sequence_lengths = [100, 200]
   sampling_rate = 1.

   data_generator = franc.evaluation.TestDataGenerator()

   generator = franc.eval.TestDataGenerator([0.1]*n_channel, rng_seed=123)
   witness_conditioning, target_conditioning= generator.generate_multiple(sequence_lengths)
   witness_evaluation, target_evaluation = generator.generate_multiple(sequence_lengths)

   print('witness shapes', [i.shape for i in witness_conditioning])
   print('target shapes', [i.shape for i in target_conditioning])


   # create the dataset object
   dataset = franc.evaluation.EvaluationDataset(
      sampling_rate,
      witness_conditioning,
      target_conditioning,
      witness_evaluation,
      target_evaluation
   )

Output:

.. testoutput::

    witness shapes [(3, 100), (3, 200)]
    target shapes [(100,), (200,)]

Executing an evaluation run
============================

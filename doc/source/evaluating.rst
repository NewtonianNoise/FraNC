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

The following is a minimal example to execute a :py:class:`franc.evaluation.EvaluationRun`.
A pdf report can be generated with `compile_report=True` (this requires a latex installation and `pdflatex` to be on the path).
By default the evaluation run creates a folder structure for the generated files in the current working directory.
The location can be changed through the `directory` parameter.

.. testcode::

   import franc as fnc
 
 
   # create a simple dataset
   n_channel = 2
   dataset = fnc.eval.TestDataGenerator(
       [0.1] * n_channel, rng_seed=0xdeadbeef
   ).dataset([int(1e4), int(1e4)], [int(1e4), int(2e4)])
   dataset.target_unit = "AU"

   print('dataset hash', franc.common.hash_function_str(dataset.hash_bytes()))
 
   # define evaluation run
   filter_configurations = [
       (
           fnc.filt.WienerFilter,
           [{"n_filter": 16, "idx_target": 0, "n_channel": n_channel}],
       ),
       (
           fnc.filt.LMSFilter,
           [{"n_filter": 16, "idx_target": 0, "n_channel": n_channel}],
       ),
   ]
 
   eval_run = fnc.eval.EvaluationRun(
       filter_configurations,
       dataset,
       fnc.eval.RMSMetric(),
       [fnc.eval.MSEMetric(), fnc.eval.PSDMetric()],
       directory="test_outputs/", # either change this or create a directory with that name
   )
 
   # execute evaluation run
   results = eval_run.run()
 
   # set compile_report to True to generate a pdf file
   eval_run.generate_report(results, compile_report=False)

.. testoutput::

    dataset hash dMaPUtgfVvNz6m4a8dZBcwBE5Dg=
    WF (ran conditioning and calculated prediction)
        target:  Residual RMS: 0.055742 AU
          Residual MSE: 0.003107 (AU)²
          Power spectral density
    LMS (ran conditioning and calculated prediction)
        target:  Residual RMS: 0.056846 AU
          Residual MSE: 0.003231 (AU)²
          Power spectral density

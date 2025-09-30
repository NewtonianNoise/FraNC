Adding evaluation metrics
**************************

Evaluation metrics are applied to the prediction of a cancellation method and the given dataset.
They represent some kind of insight into the effectiveness of the method.
An example is :py:class:`franc.evaluation.RMSMetric` which calculates the Root Mean Square value of the residual signals.
The generated values, texts, and/or plots can be included in the report generation.

Any new evaluation metric must inherit directly or indirectly from the To be usable as an evaluation metric, :py:class:`franc.evaluation.EvaluationMetric` base class.

Adding scalar metrics
======================

Scalar metrics yield a floating point value as the result.
They can be set as the optimization metric to generate additional plots.
Scalar metrics inherit from :py:class:`franc.evaluation.EvaluationMetricScalar`.

An example for a minimal evaluation metric is:


.. testsetup:: *

  import franc
  import matplotlib.pyplot as plt

.. testcode::

   import numpy as np

   class SumMetric(franc.evaluation.EvaluationMetricScalar):
       """A metric that calculates the sum of the residual signal"""

       name = "Sum of residual signal"

       @franc.evaluation.EvaluationMetric.result_full_wrapper
       def result_full(self):
           residual_sum = np.sum(np.concatenate(self.residual))
           return (residual_sum, self.dataset.target_unit)

       # adding a result_to_text function is optional
       @classmethod
       def result_to_text(cls, result_full):
           return f"{cls.name}: {result_full[0]:f} {result_full[1]}"

.. testcleanup::

   # code to test that the metric defined above actually works as intended
   def generate_test_data():
       generator = franc.eval.TestDataGenerator([0.1], rng_seed=123)
       dataset = generator.dataset([int(1e4)], [int(1e4)])
       prediction = dataset.target_evaluation
       return prediction, dataset

   metric = SumMetric().apply(*generate_test_data())
   assert metric.result == 0, "This is expected to be zero because the prediction is perfect"
   assert isinstance(metric.text, str)

What happens here:

* New class definition that derives from :py:class:`franc.evaluation.EvaluationMetricScalar`.
* The `name` is declared which will be used to print pretty representation of the metric.
* The method `result_full` is defined.

  * For a scalar metric, the first value of the returned tuple must the the primary float value.
  * Additional values in the tuple can be used. In this case, the unit of the target channel is passed to print it as part of the resulting string.

* The `@EvaluationMetric.result_full_wrapper` decorator handles error messages and caches the returned value to reduce computation times
* The optional `result_to_text` function is added to create a nicer text of the result

Apart from the residual signal, `self.prediction` and `self.dataset` are available.
A more complex example is shown in the following:

.. testcode::

  import numpy as np

  class RelativeResidualPowerMetric(franc.evaluation.EvaluationMetricScalar):
      """A metric that calculates the power ratio of the residual and target signal"""

      name = "Relative residual power"

      @franc.evaluation.EvaluationMetric.result_full_wrapper
      def result_full(self):
          pwr_residual = np.mean(np.concatenate(self.residual)**2)

          if self.dataset.signal_evaluation is not None:
              useful_signal = np.concatenate(self.dataset.target_evaluation) - np.concatenate(self.dataset.signal_evaluation)
          else:
              useful_signal = np.concatenate(self.dataset.target_evaluation)
          pwr_useful_signal = np.mean(useful_signal**2)
          return (pwr_residual/pwr_useful_signal,)

.. testcleanup::

   # code to test that the metric defined above actually works as intended
   metric = RelativeResidualPowerMetric().apply(*generate_test_data())
   assert metric.result == 0, "This is expected to be zero because the prediction is perfect"
   assert isinstance(metric.text, str)

Adding plotable metrics
=========================

Evaluation metrics can be derived from :py:class:`franc.evaluation.EvaluationMetricPlottable`.
This requires an additional plotting function according to the interface defined in :py:func:`franc.evaluation.EvaluationMetricPlottable.plot`.
The text generated through :py:func:`franc.evaluation.EvaluationMetric.result_to_text` is used as a caption for the figure in the report.
Plots are generated with `matplotlib <https://matplotlib.org/>`_.

A simple example that just plots a time series:

.. testcode::

  import numpy as np

  class SimplePlotMetric(franc.evaluation.EvaluationMetricPlottable):
      """A metric that calculates the power ratio of the residual and target signal"""

      name = "Simple plot of the residual signals"

      @franc.evaluation.EvaluationMetric.result_full_wrapper
      def result_full(self):
          return (self.residual,)

      def plot(self, ax):
          for y_data in self.result:
              ax.plot(y_data)

.. testcleanup::

   # code to test that the metric defined above actually works as intended
   metric = SimplePlotMetric().apply(*generate_test_data())
   assert isinstance(metric.text, str)
   fig, ax = plt.subplots()
   metric.plot(ax)
   plt.close(fig)

Here :py:attr:`franc.evaluation.EvaluationMetric.result`, a shorthand for the first element of :py:func:`franc.evaluation.EvaluationMetric.result_full`, is used.

Adding other metrics
=====================

Other metrics can directly inherit from :py:class:`franc.evaluation.EvaluationMetric`. In a report they can create arbitrary text elements.

Parameterizing metrics
=======================

Evaluation metrics can be made adjustable by adding a custom `__init__` function.
The following is an extension of the previous `SumMetric` example.

.. testcode::

   import numpy as np

   class ScaledSumMetric(franc.evaluation.EvaluationMetricScalar):
       """A metric that calculates the sum of the residual signal"""

       name = "Scaled sum of residual signal"

       def __init__(self, scaling_factor):
           # parameters must be passed to parent init function this way
           # to make the hashing process work correctly
           super().__init__(scaling_factor=scaling_factor)

           self.scaling_factor = scaling_factor

       @franc.evaluation.EvaluationMetric.result_full_wrapper
       def result_full(self):
           residual_sum = np.sum(np.concatenate(self.residual))
           residual_sum *= self.scaling_factor
           return (residual_sum, )

.. testcleanup::

   # code to test that the metric defined above actually works as intended
   metric = ScaledSumMetric(2.).apply(*generate_test_data())
   assert metric.result == 0, "This is expected to be zero because the prediction is perfect"
   assert isinstance(metric.text, str)

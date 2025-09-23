"""Collection of tools for the evaluation and testing of filters"""

from typing import Any
from collections.abc import Sequence, Generator
import os
from pathlib import Path
from timeit import timeit

import numpy as np
from numpy.typing import NDArray

from .common import total_power
from .dataset import EvaluationDataset
from .metrics import EvaluationMetric, EvaluationMetricScalar, EvaluationMetricPlottable
from ..filtering import FilterBase
from ..common import hash_function_str

NDArrayF = NDArray[np.floating]
NDArrayU = NDArray[np.uint]


class TestDataGenerator:
    """Generate simple test data for correlated noise mitigation techniques
    The channel count is implicitly defined by the shape of witness_noise_level

    :param witness_noise_level: amplitude ratio of the sensor noise
                to the correlated noise in the witness sensor
                Scalar or 1D-vector for multiple sensors
    :param target_noise_level: amplitude ratio of the sensor noise
                to the correlated noise in the target sensor
    :param transfer_functon: ratio between the amplitude in the target and witness signals
    :param sample_rate: The outputs are referenced
                to an ASD of 1/sqrt(Hz) if a sample rate is provided

    >>> import franc as fnc
    >>> # create data with two witness sensors with relative noise amplitudes of 0.1
    >>> tdg = fnc.evaluation.TestDataGenerator(witness_noise_level=[0.1, 0.1])
    >>> # generate a dataset with 1000 samples
    >>> witness, target = tdg.generate(1000)
    >>> witness.shape, target.shape
    ((2, 1000), (1000,))

    """

    rng: Any

    def __init__(
        self,
        witness_noise_level: float | Sequence = 0.1,
        target_noise_level: float = 0,
        transfer_function: float = 1,
        sample_rate: float = 1.0,
        rng_seed: int | None = None,
    ):
        self.witness_noise_level = np.array(witness_noise_level)
        self.target_noise_level = np.array(target_noise_level)
        self.transfer_function = np.array(transfer_function)
        self.sample_rate = sample_rate

        if rng_seed is None:
            self.rng = np.random
        else:
            self.rng = np.random.default_rng(rng_seed)

        if len(self.witness_noise_level.shape) == 0:
            self.witness_noise_level = np.array([self.witness_noise_level])

        assert (
            len(self.witness_noise_level.shape) == 1
        ), f"witness_noise_level.shape = {self.witness_noise_level.shape}"
        assert len(self.target_noise_level.shape) == 0
        assert len(self.transfer_function.shape) == 0
        assert self.sample_rate > 0

    def scaled_whitenoise(self, shape) -> NDArrayF:
        """Generate whitenoise with an ASD of one

        :param shape: shape of the new array

        :return: Array of white noise
        """
        return self.rng.normal(0, np.sqrt(self.sample_rate / 2), shape)

    def generate(self, n: int) -> tuple[NDArrayF, NDArrayF]:
        """Generate sequences of samples

        :param N: number of samples

        :return: witness signal, target signal
        """
        t_c = self.scaled_whitenoise(n)
        w_n = (
            self.scaled_whitenoise((len(self.witness_noise_level), n))
            * self.witness_noise_level[:, None]
        )
        t_n = self.scaled_whitenoise(n) * self.target_noise_level

        return (t_c + w_n) * self.transfer_function, (t_c + t_n)

    def generate_multiple(
        self, n: Sequence[int] | NDArrayU
    ) -> tuple[Sequence, Sequence]:
        """Generate sequences of samples

        :param N: Tuple with the length of the sequences

        :return: witness signals, target signals
        """
        witness = []
        target = []
        for w, t in (self.generate(n_i) for n_i in n):
            witness.append(w)
            target.append(t)
        return witness, target

    def dataset(
        self,
        n_condition: Sequence[int] | NDArray[np.uint],
        n_evaluation: Sequence[int] | NDArray[np.uint],
        sample_rate: float = 1.0,
        name: str | None = None,
    ) -> EvaluationDataset:
        """Generate an EvaluationDataset

        :param n_condition:  Sequence of integers indicating the number of conditioning samples generated per sample sequence
        :param n_evaluation: Number of evaluation samples
        :param sample_rate: (Optional) Sample rate for the generate EvaluationDataset
        :param name: (Optional) Specify the name of the EvaluationDataset

        Example:
        >>> # generate two sequences of 100 samples each of conditioning data and one 100 sample sequence of evaluation data
        >>> import franc as fnc
        >>> ds = fnc.evaluation.TestDataGenerator().dataset((100, 100), (100,))
        """
        # ensure the input parameters are 1D arrays of unsigned integers
        n_condition = np.array(n_condition, dtype=np.uint)
        n_evaluation = np.array(n_evaluation, dtype=np.uint)
        if len(n_condition.shape) != 1 or len(n_evaluation.shape) != 1:
            raise ValueError("Parameters must be sequences of integers. ")

        cond_data = self.generate_multiple(n_condition)
        eval_data = self.generate_multiple(n_evaluation)

        return EvaluationDataset(
            sample_rate,
            cond_data[0],
            cond_data[1],
            eval_data[0],
            eval_data[1],
            name=name if name else "Unnamed",
        )


def measure_runtime(
    filter_classes: Sequence[FilterBase],
    n_samples: int = int(1e4),
    n_filter: int = 128,
    idx_target: int = 0,
    n_channel: int = 1,
    additional_filter_settings: Sequence[dict] | None = None,
    repititions: int = 1,
) -> tuple[Sequence, Sequence]:
    """Measure the runtime of filers for a specific scenario
    Be aware that this gives no feedback upon how much multithreading is used!

    :param n_samples: Length of the test data
    :param n_filter: Length of the FIR filters / input block size
    :param idx_target: Position of the prediction
    :param n_channel: Number of witness sensor channels
    :param additional_filter_settings: optional settings passed to the filters
    :param repititions: how manu repititions to perform during the timing measurement

    :return: (time_conditioning, time_apply) each in seconds
    """
    filter_classes = list(filter_classes)
    if additional_filter_settings is None:
        additional_filter_settings = [{}] * len(filter_classes)
    additional_filter_settings = list(additional_filter_settings)
    assert len(additional_filter_settings) == len(filter_classes)

    witness, target = TestDataGenerator([0.1] * n_channel).generate(n_samples)

    times_conditioning = []
    times_apply = []

    def time_filter(filter_class, args):
        """wrapper function to make closures work correctly"""
        filt = filter_class(n_filter, idx_target, n_channel, **args)
        t_cond = timeit(lambda: filt.condition(witness, target), number=repititions)
        t_pred = timeit(lambda: filt.apply(witness, target), number=repititions)
        return t_cond / repititions, t_pred / repititions

    for fc, args in zip(filter_classes, additional_filter_settings):
        t_cond, t_pred = time_filter(fc, args)
        times_conditioning.append(t_cond)
        times_apply.append(t_pred)

    return times_conditioning, times_apply


class EvaluationRun:  # pylint: disable=too-many-instance-attributes
    """
    Representation of an evaluation run

    :param method_configurations: A list of tuples with the following format
        [(filter_technique, [{'n_filter': 1024, ..}, ..]), ..]
    :param dataset: An EvaluationDataset instance
    :param optimization_metric: The optimization metric by which the optimum is selected
    :param metrics: All metrics which will be exported
    :param name: (optional) name of the evaluation run
    :param directory: (optional) the directory in which results are saved
        If results are saved, the required folder structure will be created
    """

    def __init__(
        self,
        method_configurations: Sequence[tuple[type[FilterBase], Sequence]],
        dataset: EvaluationDataset,
        optimization_metric: EvaluationMetricScalar,
        metrics: Sequence[EvaluationMetric] | None = None,
        name: str = "unnamed",
        directory: str = ".",
    ):
        # check that input data format
        self.multi_sequence_support = (
            True  # indicates whether all methods support multiple sequences
        )
        for filter_technique, configurations in method_configurations:
            if not issubclass(filter_technique, FilterBase):
                raise TypeError(
                    "Only filtering techniques with the FilterBase interface are supported."
                )
            if len(configurations) < 0:
                raise TypeError(
                    "At least one parameter configuration must be supported."
                )
            for config in configurations:
                if not isinstance(config, dict):
                    raise TypeError("Filter configurations must be dictionaries.")

            if not filter_technique.supports_multi_sequence:
                self.multi_sequence_support = False

        if not isinstance(dataset, EvaluationDataset):
            raise TypeError("Dataset must be an EvaluationDataset instance.")
        if not isinstance(optimization_metric, EvaluationMetricScalar):
            raise TypeError(
                "The optimization_metric must be an instance of an EvaluationMetricScalar."
            )
        if metrics is not None:
            for metric in metrics:
                if not isinstance(metric, EvaluationMetric):
                    raise TypeError(
                        "The metrics must be instances of an EvaluationMetric."
                    )

        self.method_configurations = method_configurations
        self.dataset = dataset
        self.optimization_metric = optimization_metric
        self.metrics = metrics if metrics else []
        self.name = name
        self.directory = Path(directory)

        self.all_configurations_list: list | None = None

    def get_all_configurations(self) -> list:
        """Returns a list of all unique (filter_technique, configuration) pairs."""
        if self.all_configurations_list is None:
            self.all_configurations_list = []

            for filter_technique, configurations in self.method_configurations:
                for conf in configurations:
                    new_value = (filter_technique, conf)
                    if new_value not in self.all_configurations_list:
                        self.all_configurations_list.append(new_value)

        return self.all_configurations_list

    def _create_folder_structure(self) -> None:
        """Create standardized folder structure for results"""
        folders = [
            self.directory / "conditioned_filters",
            self.directory / "predictions",
            self.directory / "report",
            self.directory / "report" / "plots",
            self.directory / "report" / "texts",
        ]
        for path in folders:
            try:
                os.mkdir(path)
            except FileExistsError:
                pass

    @staticmethod
    def save_np_array_list(
        data: Sequence[Sequence[NDArrayF]] | Sequence[NDArrayF] | NDArrayF,
        filename: str | Path,
    ) -> None:
        """Save a list of numpy arrays to a .npz file"""
        np.savez(filename, allow_pickle=False, *data)

    @staticmethod
    def load_np_array_list(filename: str | Path) -> Sequence[NDArrayF]:
        """Load a list of numpy arrays from a .npz file"""
        data = np.load(filename, allow_pickle=False)
        keys = list(sorted(data, key=lambda x: int(x[4:])))
        for key in keys:
            if not key.startswith("arr_"):
                raise ValueError("Numpy file does not match expected format.")
        print(keys)
        return [data[key] for key in keys]

    def get_prediction(self, filter_technique: type[FilterBase], conf: dict[str, Any]):
        """Load the prediction created by applying the given filter and configuration to the dataset"""
        self._create_folder_structure()

        filt = filter_technique(**conf)

        result_hash = hash_function_str(filt.method_hash + self.dataset.hash_bytes())
        result_filename = filt.method_filename_part + "_" + result_hash
        conditioned_filter_path: Path = (
            self.directory / "conditioned_filters" / filt.make_filename(result_filename)
        )
        prediction_path = self.directory / "predictions" / (result_filename + ".npz")

        status = "loaded from file"
        if prediction_path.exists():
            pred: Sequence | NDArrayF = self.load_np_array_list(prediction_path)
        else:
            status = "calculated from loaded filter"
            # load conditioned filter or run conditioning
            if conditioned_filter_path.exists():
                filt = filter_technique.load(conditioned_filter_path)
            else:
                status = "ran conditioning and calculated prediction"
                if self.multi_sequence_support:
                    filt.condition_multi_sequence(
                        self.dataset.witness_conditioning,
                        self.dataset.target_conditioning,
                    )
                else:
                    filt.condition(
                        self.dataset.witness_conditioning[0],
                        self.dataset.target_conditioning[0],
                    )
                if filt.supports_saving_loading():
                    filt.save(conditioned_filter_path)

            # create prediction
            if self.multi_sequence_support:
                pred = filt.apply_multi_sequence(
                    self.dataset.witness_evaluation,
                    self.dataset.target_evaluation,
                )
            else:
                pred_single = filt.apply(
                    self.dataset.witness_evaluation[0],
                    self.dataset.target_evaluation[0],
                )
                pred = [pred_single]
            self.save_np_array_list(pred, prediction_path)
        print(filter_technique.filter_name, f"({status})")
        return pred, result_filename

    def run(self, select: int | None = None) -> Generator[
        tuple[Sequence[NDArrayF], EvaluationMetricScalar, Sequence[EvaluationMetric]],
        None,
        None,
    ]:
        """Execute the evaluation run

        :param select: select one specific run from the list yielded by get_all_configurations

        :return: generates (Prediction, optimization_metric, other_metrics) objects
        """
        # generate list of to-be-executed evaluations
        configurations = self.get_all_configurations()
        if select is not None:
            configurations = [configurations[select]]

        if len(self.dataset.target_evaluation) != 1 and not self.multi_sequence_support:
            raise NotImplementedError(
                "At least one filter method does not support multi sequence input, but the dataset contains multiple sequences."
            )

        # run evaluations
        for filter_technique, conf in configurations:
            pred, result_filename = self.get_prediction(filter_technique, conf)

            optimization_metric_result = self.optimization_metric.apply(
                pred, self.dataset
            )
            print("\ttarget: ", optimization_metric_result.text)

            metric_results = [
                metric.apply(pred, self.dataset) for metric in self.metrics
            ]

            for metric in metric_results:
                if isinstance(metric, EvaluationMetricPlottable):
                    plot_filename = (
                        metric.name
                        + "_"
                        + result_filename
                        + "_"
                        + metric.method_hash_str
                        + ".pdf"
                    )
                    save_path = self.directory / "report" / "plots" / plot_filename
                    metric.save_plot(save_path)
                print("\t", metric.text)

            yield pred, optimization_metric_result, metric_results


def residual_power_ratio(
    target: Sequence,
    prediction: Sequence,
    start: int | None = None,
    stop: int | None = None,
    remove_dc: bool = True,
) -> float:
    """Calculate the ratio between residual power of the residual and the target signal

    :param target: target signal array
    :param prediction: prediction array (same length as target
    :param start: use only a section of the arrays, start at this index
    :param stop: use only a section of the arrays, stop at this index
    :param remove DC component: remove DC component before calculation
    """
    target_npy = np.array(target[start:stop]).astype(np.float64)
    prediction_npy = np.array(prediction[start:stop]).astype(np.float64)
    assert target_npy.shape == prediction_npy.shape

    if remove_dc:
        target_npy -= np.mean(target)
        prediction_npy -= np.mean(prediction_npy)

    residual = prediction_npy - target_npy

    return float(total_power(residual) / total_power(target_npy))


def residual_amplitude_ratio(*args, **kwargs) -> float:
    """Calculate the ratio between residual amplitude of the residual and the target signal

    :param target: target signal array
    :param prediction: prediction array (same length as target
    :param start: use only a section of the arrays, start at this index
    :param stop: use only a section of the arrays, stop at this index
    :param remove DC component: remove DC component before calculation
    """
    return float(np.sqrt(residual_power_ratio(*args, **kwargs)))

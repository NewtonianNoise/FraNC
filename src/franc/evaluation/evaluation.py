"""Collection of tools for the evaluation and testing of filters"""

from typing import Any
from collections.abc import Sequence
import sys
import os
from pathlib import Path
from timeit import timeit
from copy import deepcopy
import importlib.metadata
from datetime import datetime
import inspect

import numpy as np
from numpy.typing import NDArray
import scipy
import numba
import matplotlib.pyplot as plt

from .common import total_power
from .dataset import EvaluationDataset
from .metrics import EvaluationMetric, EvaluationMetricScalar, EvaluationMetricPlottable
from .filter_interface import FilterInterface
from .report_generation import Report, ReportElement, ReportTable, ReportFigure
from ..common import hash_function_str, get_platform_info, bytes2str

NDArrayF = NDArray[np.floating]
NDArrayU = NDArray[np.uint]


class TestDataGenerator:
    """Generate simple test data for correlated noise mitigation techniques
    The channel count is implicitly defined by the shape of witness_noise_level

    :param witness_noise_level: Amplitude ratio of the sensor noise
                to the correlated noise in the witness sensor
                Scalar or 1D-vector for multiple sensors
    :param target_noise_level: Amplitude ratio of the sensor noise
                to the correlated noise in the target sensor
    :param transfer_function: Ratio between the amplitude in the target and witness signals
    :param sample_rate: The outputs are referenced
                to an ASD of 1/sqrt(Hz) if a sample rate is provided
    :param rng_seed: Optional value to generate the dataset based on a fixed seed for reproducible results.
                If not set, the randomly seeded global numpy rng is used.

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

        :param n: number of samples

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

        :param n: Tuple with the length of the sequences

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
    filter_classes: Sequence[FilterInterface],
    n_samples: int = int(1e4),
    n_channel: int = 1,
    n_filter: int = 128,
    idx_target: int = 0,
    additional_filter_settings: Sequence[dict[str, Any]] | None = None,
    repititions: int = 1,
) -> tuple[Sequence, Sequence]:
    """Measure the runtime of filers for a specific scenario
    Be aware that this gives no feedback upon how much multithreading is used!

    :param n_samples: Length of the test data
    :param n_channel: Number of witness sensor channels
    :param n_filter: Length of the FIR filters / input block size
    :param idx_target: Position of the prediction
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
        filt = filter_class(n_channel, n_filter, idx_target, **args)
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
        method_configurations: Sequence[tuple[type[FilterInterface], Sequence]],
        dataset: EvaluationDataset,
        optimization_metric: EvaluationMetricScalar,
        metrics: Sequence[EvaluationMetric] | None = None,
        name: str = "unnamed",
        directory: str = ".",
        figsize: tuple[float, float] = (10, 4),
    ):
        self.multi_sequence_support = self._check_method_configurations(
            method_configurations
        )

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

        self.method_configurations = deepcopy(method_configurations)
        self.dataset = dataset
        self.optimization_metric = optimization_metric
        self.metrics = metrics if metrics else []
        self.name = name
        self.directory = Path(directory)
        self.figsize = figsize

        # add n_channel values in case they were not supplied
        def add_n_channel(conf: dict):
            if "n_channel" in conf:
                return conf
            return conf | {"n_channel": self.dataset.channel_count}

        self.method_configurations = [
            (fm, [add_n_channel(conf) for conf in configurations])
            for fm, configurations in self.method_configurations
        ]

        for _, configurations in method_configurations:
            for conf in configurations:
                if "n_channel" not in conf:
                    conf = conf | {"n_channel": self.dataset.channel_count}

        self.all_configurations_list: list | None = None

    def _check_method_configurations(
        self,
        method_configurations: Sequence[tuple[type[FilterInterface], Sequence]],
    ) -> bool:
        """Throw meaningful errors for problems with the configurations"""
        multi_sequence_support = True

        for filter_technique, configurations in method_configurations:
            if not issubclass(filter_technique, FilterInterface):
                raise TypeError(
                    "Only filtering techniques with the FilterInterface interface are supported."
                )
            if len(configurations) < 0:
                raise TypeError(
                    "At least one parameter configuration must be supported."
                )
            for config in configurations:
                if not isinstance(config, dict):
                    raise TypeError("Filter configurations must be dictionaries.")

            if not filter_technique.supports_multi_sequence:
                multi_sequence_support = False
        return multi_sequence_support

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
            self.directory / "report" / "tex",
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
        return [data[key] for key in keys]

    @staticmethod
    def software_version_report() -> list[str]:
        """generate a list of strings indicating the important software versions"""
        this_package_name = __name__.split(".", maxsplit=1)[0]

        version_strings = [
            f"Python: {sys.version}\n",
            f"{this_package_name}: {importlib.metadata.version(this_package_name)}\n",
        ]
        for package in [np, scipy, numba]:
            version_strings += [f"{package.__name__}: {package.__version__}\n"]
        return version_strings

    @staticmethod
    def platform_info_report() -> list[str]:
        """generate a list of strings indicating platform information (cpu, OS, ..)"""
        return [get_platform_info()]

    def generate_overview_plots(
        self,
        results: list[tuple[type[FilterInterface], list]],
    ):
        """Generate overview plots
        Returns a Report section with the generated plot
        """
        report_section = []

        optimization_metric_values = []
        for idx, (filter_technique, configurations) in enumerate(results):
            performance_results = [conf[2].result for conf in configurations]
            optimization_metric_values.append(
                (filter_technique, min(performance_results), performance_results, idx)
            )
        optimization_metric_ranges = list(
            sorted(optimization_metric_values, key=lambda x: x[1])
        )

        fig, ax = plt.subplots(figsize=self.figsize)
        for idx, (filter_technique, _lowest, all_values, _idx) in enumerate(
            optimization_metric_ranges
        ):
            plt.scatter([idx] * len(all_values), all_values)
        ax.set_xticks(
            range(len(optimization_metric_ranges)),
            list(
                map(
                    lambda x: x[0].filter_name + f" {x[3]+1}",
                    optimization_metric_ranges,
                )
            ),
        )

        y_min = min(min(i[2]) for i in optimization_metric_ranges)
        y_max = max(max(i[2]) for i in optimization_metric_ranges)
        y_limits = [
            y_min - (y_max - y_min) * 0.05,
            y_max + (y_max - y_min) * 0.05,
        ]
        if y_limits[0] != y_limits[1]:
            ax.set_ylim(*y_limits)

        # get an evaluation metric instantiated on the dataset
        optimization_metric = results[0][1][0][2]
        ax.set_ylabel(f"{optimization_metric.name} [{optimization_metric.unit}]")

        figure_fname = f"comparison_{self.hash_str()}.pdf"
        fig.savefig(self.directory / "report/plots" / figure_fname)
        plt.close(fig)

        report_section += [
            ReportFigure(
                str(Path("../plots") / figure_fname),
                caption="Optimizaiton metric overview",
            )
        ]

        return report_section

    def generate_parameter_scan_plots(
        self,
        results: list[
            tuple[
                type[FilterInterface],
                list[
                    tuple[
                        dict,
                        NDArray,
                        EvaluationMetricScalar,
                        list[EvaluationMetric],
                        str,
                    ]
                ],
            ]
        ],
    ) -> list[list[ReportFigure]]:
        """Generate a plot of the optimization_metric values over all varied parameters
        Returns a report section with the generated plots
        """
        report_sections = []

        # identify changed values
        for filter_method, configurations in results:
            parameter_values: dict[str, Any] = {}
            parameterset_hash = ""

            for conf, _, _, _, configuration_hash in configurations:
                parameterset_hash += configuration_hash
                for parameter, value in conf.items():
                    if parameter not in parameter_values:
                        parameter_values[parameter] = set()
                    parameter_values[parameter] |= {value}
            parameterset_hash = hash_function_str(parameterset_hash.encode())

            section = []
            for parameter, values in parameter_values.items():
                if len(values) > 1:
                    fig, ax = plt.subplots(figsize=self.figsize)
                    ax.scatter(
                        [conf[0][parameter] for conf in configurations],
                        [conf[2].result for conf in configurations],
                    )
                    ax.set_xlabel(parameter)
                    ax.set_ylabel(
                        f"{configurations[0][2].name} {configurations[0][2].unit}"
                    )
                    fig.tight_layout()

                    # save and add to report
                    fname = f"{filter_method.filter_name}_{parameter}_{hash}.pdf"
                    fig.savefig(self.directory / "report/plots" / fname)
                    plt.close(fig)

                    section.append(ReportFigure(str(Path("../plots/") / fname)))

            report_sections.append(section)

        return report_sections

    def generate_report(
        self,
        results: list[tuple[type[FilterInterface], list]],
        compile_report: bool = False,
    ):
        """Generate a report for the given results object from run()"""
        report = Report()

        # generate overview page
        report_generation_timestamp = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        report["Overview"] = {
            "General": [
                f"Report generated: {report_generation_timestamp}\n",
                f"EvaluationRun hash: {self.hash_str()}",
            ],
            "Platform": self.platform_info_report(),
            "Software versions": self.software_version_report(),
            "Evaluation dataset": self.dataset.description().replace("\n", "\n\n"),
        }

        report["Results overview"] = self.generate_overview_plots(results)
        report["Results detailed"] = {}

        parameter_scan_sections = self.generate_parameter_scan_plots(results)

        # generate report entries for every tested configuration
        for filter_idx, (filter_technique, configurations) in enumerate(results):
            docstring = inspect.getdoc(filter_technique)
            if docstring is not None:
                docstring = docstring.split(">>>", maxsplit=1)[0]
            else:
                docstring = ""

            filter_hash = (
                filter_technique.__module__
                + "."
                + filter_technique.__name__
                + " (File hash: "
                + bytes2str(filter_technique.file_hash())
                + ")"
            )
            detailed_report_entries: dict[str, list | dict | str | ReportElement] = {
                "Overview": [
                    filter_hash,
                    f"\\begin{{lstlisting}}{docstring}\\end{{lstlisting}}",
                ]
                + parameter_scan_sections[filter_idx],
            }

            for conf_idx, (
                conf,
                _pred,
                optimization_metric,
                metrics,
                _hash,
            ) in enumerate(configurations):
                entry: list[ReportElement | str] = []
                entry.append(
                    ReportTable(
                        [(str(key), str(value)) for key, value in conf.items()],
                        caption="Configuration values",
                    )
                )
                entry.append(optimization_metric.text + "\n\n")
                for metric in metrics:
                    if (
                        isinstance(metric, EvaluationMetricPlottable)
                        and metric.plot_path is not None
                    ):
                        path = Path(metric.plot_path)
                        entry.append(
                            ReportFigure("../plots/" + path.name, caption=metric.text)
                        )
                    else:
                        entry.append(metric.text + "\n\n")
                detailed_report_entries[f"Configuration {conf_idx+1}"] = entry

            report["Results detailed"][
                filter_technique.filter_name + f" {filter_idx+1}"
            ] = detailed_report_entries

        if compile_report:
            report.compile(self.directory / "report" / "tex" / "report")
        else:
            report.save(self.directory / "report" / "tex" / "report")

    def get_prediction(
        self, filter_technique: type[FilterInterface], conf: dict[str, Any]
    ) -> tuple[Sequence[NDArray] | NDArray, str, str]:
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
            pred: Sequence[NDArrayF] | NDArrayF = self.load_np_array_list(
                prediction_path
            )
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
        return (
            pred,
            result_hash,
            result_filename,
        )

    def run(self) -> list[tuple[type[FilterInterface], list]]:
        """Execute the evaluation run

        :return: list of (Prediction, optimization_metric, other_metrics) objects
        """
        if len(self.dataset.target_evaluation) != 1 and not self.multi_sequence_support:
            raise NotImplementedError(
                "At least one filter method does not support multi sequence input, but the dataset contains multiple sequences."
            )

        # run evaluations
        results: list[tuple[type[FilterInterface], list]] = []
        for filter_technique, filt_configs in self.method_configurations:
            results.append((filter_technique, []))
            for conf in filt_configs:

                pred, result_hash, result_filename = self.get_prediction(
                    filter_technique, conf
                )

                optimization_metric_result = self.optimization_metric.apply(
                    pred, self.dataset
                )
                print("    target: ", optimization_metric_result.text)

                metric_results = [
                    metric.apply(pred, self.dataset) for metric in self.metrics
                ]

                for metric in metric_results:
                    if isinstance(metric, EvaluationMetricPlottable):
                        save_path = (
                            self.directory
                            / "report"
                            / "plots"
                            / metric.filename(result_filename)
                        )
                        metric.save_plot(save_path)
                    print("     ", metric.text)

                results[-1][1].append(
                    (
                        conf,
                        pred,
                        optimization_metric_result,
                        metric_results,
                        result_hash,
                    )
                )
        return results

    def hash_str(self) -> str:
        """returns a hash over the dataset and filtering configurations as a string"""
        hash_value = self.dataset.hash_bytes()
        for filter_technique, configurations in self.method_configurations:
            for conf in configurations:
                hash_value += filter_technique(**conf).method_hash
        return hash_function_str(hash_value)


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
    :param remove_dc: if true, the mean is subtracted from each array to remove the DC component before the calculations
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

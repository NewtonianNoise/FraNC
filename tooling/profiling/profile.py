"""Tooling to measure data throughput of filter implementations."""

from collections.abc import Sequence
import time
import platform
import subprocess
from warnings import warn

import psutil
import numpy as np
from numpy.typing import NDArray

import franc as fnc

MULTITHREAD = True

DEBUG = False
IGNORE_FILTER_OPTIONS = {"coefficient_clipping", "step_scale"}

# filter, additional_filter_config, skip_conditioning
FILTER_CONFIGURATIONS = [
    (fnc.filtering.WienerFilter, {}, False),
    (fnc.filtering.UpdatingWienerFilter, {"context_pre": 3000}, True),
    (fnc.filtering.LMSFilter, {"normalized": True, "coefficient_clipping": 10}, False),
    (
        fnc.filtering.PolynomialLMSFilter,
        {"order": 1, "coefficient_clipping": 10},
        False,
    ),
    (
        fnc.filtering.PolynomialLMSFilter,
        {"order": 3, "coefficient_clipping": 10},
        False,
    ),
]

if DEBUG:
    FILTER_CONFIGURATIONS = [
        (fnc.filtering.WienerFilter, {}, False),
        (fnc.filtering.LMSFilter, {"normalized": True}, False),
        (fnc.filtering.UpdatingWienerFilter, {"context_pre": 1000}, True),
    ]


def get_platform_info():
    """Get a string that describes the operating system and CPU model."""
    os = platform.system()

    cpu = "-"
    try:
        if os == "Linux":
            cpu = subprocess.check_output("cat /proc/cpuinfo", shell=True).decode()
            cpu = filter(lambda i: "model name" in i, cpu.split("\n"))
            cpu = next(cpu).split(":")[1].strip()
        elif os == "Darwin":
            cpu = (
                subprocess.check_output(
                    "sysctl -n machdep.cpu.brand_string", shell=True
                )
                .decode()
                .strip()
            )
            cpu += (
                " "
                + subprocess.check_output(
                    "sysctl -n machdep.cpu.core_count", shell=True
                )
                .decode()
                .strip()
                + " Cores"
            )
    except Exception as e:  # pylint: disable=broad-exception-caught
        cpu = "-"
        warn(f"Could not get platform information ({repr(e)})")

    return os + ", " + cpu


def get_git_hash() -> str:
    """Get the current short git commit hash and add a + if there are uncommitted changes"""
    git_hash = (
        subprocess.check_output("git log --pretty=format:'%h' -n 1", shell=True)
        .decode()
        .strip()
    )
    git_local_changes = subprocess.run(
        "git diff --quiet --exit-code", shell=True, check=False
    ).returncode
    return git_hash + ("+" if git_local_changes else "")


def filter_configs_to_str(configs):
    """build documenting strings for the given list of configuations"""

    def additional_filter_config_to_str(conf):
        if len(conf) == 0:
            return ""
        return (
            "("
            + ", ".join(
                f"{k}={v}" for k, v in conf.items() if k not in IGNORE_FILTER_OPTIONS
            )
            + ")"
        )

    return [
        f"{fc.filter_name} {additional_filter_config_to_str(settings)}"
        for fc, settings, _ignore_conditioning in configs
    ]


filter_configuration_strings = filter_configs_to_str(FILTER_CONFIGURATIONS)


def run_profiling(config, n_samples, n_filter, n_channel, idx_target=0):
    """execute a profiling run for specific configuration for a list of filter configurations
    to unwrap config list and replace irrelevant results with NaNs

    :param config: filter configurations as a list of (filter_instance, additional_filter_params, clear_conditioning_runtime)
                    setting clear_conditioning_runtime to True will set the conditioning runtime to np.nan
    :params n_samples, n_filter, n_channel, idx_target: passed on to franc.measure_runtime()
    """
    filters = list(map(lambda x: x[0], config))
    additional_settings = list(map(lambda x: x[1], config))
    skip_conditioning = list(map(lambda x: x[2], config))

    results = fnc.evaluation.measure_runtime(
        filters,
        n_samples=n_samples,
        n_channel=n_channel,
        n_filter=n_filter,
        idx_target=idx_target,
        additional_filter_settings=additional_settings,
        repititions=1,
    )
    results = np.array(results)
    results[0, skip_conditioning] = np.nan
    return results


def profiling_scan(
    target: str, target_values: Sequence | NDArray, other_values: dict, filter_configs
) -> dict:
    """scan through one variable and record the runtime of the selected filters

    :param target: the target variable; one of 'n_filter', 'n_samples', 'idx_target', 'n_channel'
    :param target_values: the list of values for the target parameter
    :param other_values: dict of remaining values (may contain the target, but that value will be overwritten)
    :param filter_configs: The filter configuration as required by run_profiling

    :return: list of run_profiling results as processing rate in Sps
            array dimensions: (target_value, stage, filter_method)
            stage is conditioning=0, applying=1
    :raises: AssertionError
    """
    all_values = ["n_filter", "n_samples", "idx_target", "n_channel"]
    assert target in all_values, f"target must be one of {all_values}"
    for key in all_values:
        assert (key == target) or (
            key in other_values
        ), f"{key} must be the target value or provided through other_values"

    results = []
    for target_value in target_values:
        other_values[target] = target_value

        result = run_profiling(filter_configs, **other_values)
        results.append(other_values["n_samples"] / result)
    return {
        "target": target,
        "target_values": target_values,
        "results": np.array(results),
        "filter_configs": filter_configs_to_str(filter_configs),
        "filter_names": [i[0].filter_name for i in filter_configs],
        "other_values": tuple((k, v) for k, v in other_values.items() if k != target),
    }


def measure_cpu_load(function, *args, **kwargs):
    """Run function with given args and get CPU load.
    Cpu load is measured before, during, and after execution.
    Values are reported relative to the whole system.
    A load of one means all cpu cores are fully loaded.
    """
    psutil.cpu_percent()  # start measurement
    time.sleep(1)  # delay for accurate cpu load measurement
    cpu_load = [psutil.cpu_percent() * 1e-2]

    results = function(*args, **kwargs)
    cpu_load += [psutil.cpu_percent() * 1e-2]

    time.sleep(1)  # delay for accurate cpu load measurement
    cpu_load += [psutil.cpu_percent() * 1e-2]

    print("\tCPU load " + " ".join(f"{i*100:.1f}%" for i in cpu_load))
    return results, cpu_load


def run_and_save_scan(target, values, default_values, filter_config, **file_additions):
    """Run a throughput measurement scan and save the result with context information in a numpy dump."""
    # run profiling and measure total duration
    start = time.time()
    results, cpu_load = measure_cpu_load(
        profiling_scan, target, values, default_values, filter_config
    )
    end = time.time()

    # save results to numpy dump
    additional_info = {
        "platform": get_platform_info(),
        "git_hash": get_git_hash(),
        "cpu_load": cpu_load,
    }
    np.savez(
        f'results/results_{target}_{"mt" if MULTITHREAD else "st"}.npz',
        multithreaded=MULTITHREAD,
        **results,
        **file_additions,
        **additional_info,
    )

    return end - start


def main():
    """Main function, runs a default profiling job."""
    # target, target_values, default_values, log_scale, save_results
    parameter_scans = [
        (
            "n_filter",
            [10],
            {"n_samples": int(1e4), "n_channel": 1, "n_filter": 16, "idx_target": 0},
            True,
            False,
        ),
        (
            "n_filter",
            [10, 30, 100, 300, 1000],
            {"n_samples": int(1e4), "n_channel": 1, "n_filter": 128, "idx_target": 0},
            True,
            True,
        ),
        (
            "n_channel",
            [1, 2, 3],
            {"n_samples": int(1e4), "n_channel": 1, "n_filter": 128, "idx_target": 0},
            False,
            True,
        ),
        (
            "n_samples",
            (10 ** np.arange(2, 5.6, 0.5)).astype(int),
            {"n_samples": int(1e4), "n_channel": 1, "n_filter": 32, "idx_target": 0},
            True,
            True,
        ),
    ]

    for target, target_values, default_values, x_log, save_results in parameter_scans:
        if save_results:
            print(target)
            runtime = run_and_save_scan(
                target,
                target_values,
                default_values,
                FILTER_CONFIGURATIONS,
                x_log=x_log,
            )
            print(f"\tdone in {runtime:.1f} s")
        else:
            # this triggers e.g. numba compiles that would otherwise slow down the first exection
            print("priming filters")
            profiling_scan(target, target_values, default_values, FILTER_CONFIGURATIONS)


if __name__ == "__main__":
    main()

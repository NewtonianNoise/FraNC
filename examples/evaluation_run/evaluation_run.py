import franc as fnc

n_channel = 3

if __name__ == "__main__":
    # create evaluation dataset
    dataset = fnc.eval.TestDataGenerator(
        [0.1] * n_channel, rng_seed=831011041148397102116105103
    ).dataset([int(1e5), int(1e5)], [int(1e5), int(2e5)])
    dataset.target_unit = "AU"

    # define evaluation run
    eval_run = fnc.eval.EvaluationRun(
        [
            (
                fnc.filt.WienerFilter,
                [
                    {"n_filter": 16, "idx_target": 0, "n_channel": n_channel},
                    {"n_filter": 32, "idx_target": 0, "n_channel": n_channel},
                    {"n_filter": 64, "idx_target": 0, "n_channel": n_channel},
                ],
            ),
            (
                fnc.filt.LMSFilter,
                [{"n_filter": 16, "idx_target": 0, "n_channel": n_channel}],
            ),
        ],
        dataset,
        fnc.eval.RMSMetric(),
        [fnc.eval.MSEMetric(), fnc.eval.PSDMetric(), fnc.eval.TimeSeriesMetric()],
    )

    # execute evaluation run
    results = eval_run.run()
    eval_run.generate_report(results, True)

    print("done")

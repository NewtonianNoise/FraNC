import saftig as sg

n_channel = 3

if __name__ == "__main__":
    # create evaluation dataset
    dataset = sg.eval.TestDataGenerator(
        [0.1] * n_channel, rng_seed=831011041148397102116105103
    ).dataset([int(1e5)], [int(1e5)])

    # define evaluation run
    eval_run = sg.eval.EvaluationRun(
        [
            (
                sg.filt.WienerFilter,
                [{"n_filter": 16, "idx_target": 0, "n_channel": n_channel}],
            ),
            (
                sg.filt.LMSFilter,
                [{"n_filter": 16, "idx_target": 0, "n_channel": n_channel}],
            ),
        ],
        dataset,
        sg.eval.RMSMetric(),
        [sg.eval.MSEMetric()],
    )

    # execute evaluation run
    for entry in eval_run.run():
        pass

    print("done")

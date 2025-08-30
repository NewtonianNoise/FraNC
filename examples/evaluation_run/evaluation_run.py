import saftig as sg

if __name__ == "__main__":
    # create evaluation dataset
    dataset = sg.eval.TestDataGenerator([0.1] * 3).dataset([int(1e5)], [int(1e5)])

    # define evaluation run
    eval_run = sg.eval.EvaluationRun(
        [
            (sg.filt.WienerFilter, [{"n_filter": 16, "idx_target": 0}]),
            (sg.filt.LMSFilter, [{"n_filter": 16, "idx_target": 0}]),
        ],
        dataset,
        sg.eval.RMSMetric(),
    )

    # execute evaluation run
    for entry in eval_run.run():
        pass

    print("done")

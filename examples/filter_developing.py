import franc
from icecream import ic
import matplotlib.pyplot as plt
import numpy as np
import scipy
import cProfile
import timeit

# settings
N = int(1e4)
N_filter = 128
N_channel = 1

PROFILE = True

A = np.random.rand(N_filter, N_filter)

if __name__ == "__main__":

    w, t = franc.evaluation.TestDataGenerator([0.1] * N_channel).generate(N)

    # filt = franc.filtering.WienerFilter(N_filter, 0, N_channel)
    # filt = franc.filtering.UpdatingWienerFilter(N_filter, 0, N_channel, 20*N_filter, 20*N_filter)
    # filt = franc.filtering.LMSFilter(N_filter, 0, N_channel, step_scale=0.1)
    filt = franc.filtering.PolynomialLMSFilter(
        N_filter, 0, N_channel, step_scale=0.1, order=3, coefficient_clipping=5
    )
    # filt = franc.filtering.external.SpicypyWienerFilter(N_filter, 0, N_channel)

    filt.condition(w, t)
    fs_before = np.array(filt.filter_state)
    if PROFILE:
        cProfile.run(
            "pred = filt.apply(w, t, pad=True, update_state=True)", sort="tottime"
        )
        exit()
    else:
        pred = filt.apply(w, t, pad=True, update_state=True)
    fs_after = np.array(filt.filter_state)

    ic((fs_before == fs_after).all())

    ic(filt.filter_state.shape)
    ic(pred.shape)
    ic(pred.shape[0] - t.shape[0])

    ic(franc.evaluation.rms(t[2000:]))
    ic(franc.evaluation.rms((t - pred)[2000:]))
    ic(franc.evaluation.residual_amplitude_ratio(t, pred, start=2000))

    plt.figure()
    plt.plot(t, label="target")
    plt.plot(pred, label="prediction")
    plt.plot(pred - t, label="residual")
    plt.legend()

    plt.show()

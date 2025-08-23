"""Run profiling with numpy set to singlethreading."""

import os

# limit numy to a single thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import profile  # pylint: disable=wrong-import-position

if __name__ == "__main__":
    profile.MULTITHREAD = False
    profile.main()

"""Step 7c, speed: element_mean runs once in every run of the ROM, so it must not loop over the triangles in Python.

    from twin.mechanics import element_mean

On the plate of the benchmark (64 x 32 cells: 4096 triangles, 2145 nodes) the mean of the nodal values of all the
triangles is a handful of NumPy operations on whole arrays, about 0.1 ms. A Python loop over the 4096 triangles costs
from 3 to 15 ms, depending on the machine, and in the first benchmark of the ROM it was 90% of the time of a run. The
other tests of element_mean check the numbers and cannot see a loop; this one can. The limit is 2 ms and the time is the
best of 20 calls, so a busy computer does not fail the test by bad luck.
"""

import time

import numpy as np
import pytest

# Any warning is an error in these tests.
pytestmark = pytest.mark.filterwarnings("error")

N_TRIANGLES = 4096
N_NODES = 2145
LIMIT_SECONDS = 2e-3
REPEATS = 20


# The function is imported inside the test and not at the top: while twin/mechanics.py does not exist, or does not have
# the function yet, only the test fails, not the whole session.
def test_element_mean_does_not_loop_over_the_triangles_in_python():
    from twin.mechanics import element_mean

    rng = np.random.default_rng(0)
    triangles = rng.integers(0, N_NODES, size=(N_TRIANGLES, 3))
    values = rng.random((1, N_NODES))  # one state: what a run of the ROM that keeps only the last state gives

    element_mean(triangles, values)  # a first call, not timed

    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        element_mean(triangles, values)
        best = min(best, time.perf_counter() - start)

    assert best < LIMIT_SECONDS, "the best of %d calls took %.2f ms, the limit is %.1f ms" % (
        REPEATS,
        1e3 * best,
        1e3 * LIMIT_SECONDS,
    )
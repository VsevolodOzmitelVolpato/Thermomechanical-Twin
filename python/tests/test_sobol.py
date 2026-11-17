"""Step 8a: Sobol sensitivity indices of a model with uncertain parameters, from twin/sobol.py.

    from twin.sobol import sobol_indices

The question. A model has d parameters that are uncertain: each one is a random number, uniform on an interval and
independent of the others, so the output of the model (a number, or a few numbers) is random too. The Sobol indices say
how much each parameter is responsible for that randomness:

    S_i    the first-order index of the parameter i: the share of the variance of the output that i explains by itself
    ST_i   the total index of i: the share of the variance that is lost if i is fixed to one value, that is i by itself
           and i together with the others (the interactions)

An example: f = 2 x1 + x2 with x1 and x2 uniform on the same interval. The weight of x1 is twice that of x2, so its
share of the variance is four times larger: S_1 = ST_1 = 4/5 and S_2 = ST_2 = 1/5. Another: f = x1 x2 with x1 and x2
uniform on [-1, 1]. Each parameter alone says nothing about f (S_1 = S_2 = 0), but nothing is left without it
(ST_1 = ST_2 = 1): all the variance is interaction.

The estimate needs only values of the model at well chosen points (Saltelli, 2010; Jansen, 1999). Two tables of points A
and B (n rows and d columns) are drawn and, for every parameter i, a third table AB_i: the table A where the column i is
taken from B. Comparing f(A) with f(AB_i) shows what happens to the output when only the parameter i changes.

sobol_indices(model, low, high, n, seed=0, n_bootstrap=200, confidence=0.95)
    model         a function, model(points). The argument is a float array (m, d), one point per row, and the function
                  returns the outputs of the m points: an array (m,) for one output, or (m, q) for q outputs. The output
                  of a point does not depend on the other points. The function may change the array it receives and may
                  return the same array at every call.
    low, high     the bounds of the d parameters: two sequences, or arrays, of d numbers with low < high. The parameter
                  j is uniform on [low[j], high[j]] and independent of the others
    n             the number of rows of A and B: a power of 2, at least 2
    seed          an integer >= 0. The same seed gives the same answer
    n_bootstrap   the number of resamples of the confidence intervals: an integer >= 1
    confidence    the level of the confidence intervals: a number with 0 < confidence < 1

    The points. A and B are the first d columns and the last d columns of the first n points of a scrambled Sobol
    sequence of dimension 2 d (scipy.stats.qmc.Sobol), mapped from the unit cube onto the box. Every point is inside the
    box, and, because the sequence is scrambled, never on its boundary. The model is called d + 2 times, always with n
    points, in this order: A, B, AB_1, AB_2, ..., AB_d. That is n (d + 2) evaluations in all, and no other.

    The formulas. fA, fB and fAB_i are the outputs of the model at A, B and AB_i. f0 is the mean, and V the variance
    (the divisor is 2 n, not 2 n - 1), of the 2 n values of fA and fB together. With a = fA - f0, b = fB - f0 and
    c_i = fAB_i - f0, row by row,

        S_i  = mean(b (c_i - a)) / V               (Saltelli)
        ST_i = mean((a - c_i)^2) / (2 V)           (Jansen)

    where the mean is over the n rows. With several outputs every column has its own f0, V and indices.

    The confidence intervals (bootstrap). n_bootstrap times: n rows are drawn at random, with replacement, among the n
    rows of fA, fB and fAB (the same rows in all of them) and S_i and ST_i are computed again on them. The interval goes
    from the percentile 100 (1 - confidence) / 2 to the percentile 100 (1 + confidence) / 2 of the n_bootstrap values,
    as numpy.percentile computes them (linear interpolation between the sorted values). The model is not called again.

    Returns a SobolResult with the attributes (d parameters, q outputs):
        first_order      S_i                                    shape (d,) or (d, q)
        total_order      ST_i                                   shape (d,) or (d, q)
        first_order_ci   the interval of S_i                    shape (2, d) or (2, d, q)
        total_order_ci   the interval of ST_i                   shape (2, d) or (2, d, q)
                         (index 0 of the first axis: the lower bound, index 1: the upper bound)
        mean             f0                                     shape () or (q,)
        variance         V                                      shape () or (q,)
        n_evaluations    n (d + 2), an int
    The shapes with q are those of a model that returns (m, q), even if q = 1; the others are those of a model that
    returns (m,). The arrays are new float64 arrays.

    ValueError if low and high are not two sequences of the same length d >= 1 of finite numbers with low < high, if n
    is not a power of 2 or is less than 2, if seed is negative, if n_bootstrap is less than 1, if confidence is not
    between 0 and 1 (NaN is not), and for an output of the model that has not the shape (m,) or (m, q) with q >= 1, that
    has a NaN or an infinity, that is constant (its variance is zero, so its indices are not defined) or whose shape
    changes from one call to the next. TypeError if model is not callable, if n, seed or n_bootstrap is not an
    integer (a bool is not one) and if confidence is not a number. The model is not called if the arguments are
    refused. An error of the model goes through unchanged.
"""
import math

import numpy as np
import pytest

# Any warning is an error in these tests.
pytestmark = pytest.mark.filterwarnings("error")

PI = math.pi
NAN = float("nan")
INF = float("inf")


# The function is imported inside this wrapper and not at the top: while twin/sobol.py does not exist, or is not
# complete, only the tests that need it fail, not the whole session.


def sobol(*args, **kwargs):
    from twin.sobol import sobol_indices
    return sobol_indices(*args, **kwargs)


# ---------------------------------------------------------------------------------------------------------------------
# Models with known indices, and helpers
# ---------------------------------------------------------------------------------------------------------------------


def linear(weights):
    """f = sum of weights[j] x_j: S_j = ST_j = (weights[j] width[j])^2 / sum of the same numbers."""
    weights = np.array(weights, dtype=float)
    return lambda points: points @ weights


# The linear model of most tests: the parameters have very different widths and the last one is ignored.
LIN_WEIGHTS = [2.0, 1.0, -3.0, 0.0]
LIN_LOW = [0.0, -5.0, 100.0, 0.0]
LIN_HIGH = [1.0, 5.0, 102.0, 7.0]
_lin_variances = (np.array(LIN_WEIGHTS) * (np.array(LIN_HIGH) - np.array(LIN_LOW))) ** 2 / 12.0
LIN_INDICES = _lin_variances / _lin_variances.sum()    # 0.0286, 0.7143, 0.2571, 0


def ishigami(points, a=7.0, b=0.1):
    return np.sin(points[:, 0]) + a * np.sin(points[:, 1]) ** 2 + b * points[:, 2] ** 4 * np.sin(points[:, 0])


# The exact indices of the function of Ishigami, for three parameters uniform on [-pi, pi] (a = 7, b = 0.1).
_V = 49.0 / 8 + 0.1 * PI ** 4 / 5 + 0.01 * PI ** 8 / 18 + 0.5
_V1 = 0.1 * PI ** 4 / 5 + 0.01 * PI ** 8 / 50 + 0.5
_V2 = 49.0 / 8
_V13 = 0.01 * PI ** 8 / 18 - 0.01 * PI ** 8 / 50
ISHIGAMI_S = np.array([_V1, _V2, 0.0]) / _V             # 0.3139, 0.4424, 0
ISHIGAMI_ST = np.array([_V1 + _V13, _V2, _V13]) / _V    # 0.5576, 0.4424, 0.2437
ISHIGAMI_LOW = [-PI] * 3
ISHIGAMI_HIGH = [PI] * 3

# The g-function of Sobol on [0, 1]^5: the larger the number a_j, the less the parameter j matters.
G_OFFSETS = np.array([0.0, 1.0, 4.5, 9.0, 99.0])


def g_function(points):
    return np.prod((np.abs(4.0 * points - 2.0) + G_OFFSETS) / (1.0 + G_OFFSETS), axis=1)


_g_partial = 1.0 / (3.0 * (1.0 + G_OFFSETS) ** 2)
_g_total = np.prod(1.0 + _g_partial) - 1.0
G_S = _g_partial / _g_total
G_ST = np.array([_g_partial[i] * np.prod(np.delete(1.0 + _g_partial, i)) for i in range(5)]) / _g_total


def wiggle(points, q):
    """A smooth function with interactions and q different outputs, for the checks against the formulas."""
    others = points[:, 1:].sum(axis=1)
    return np.stack([np.sin(3.0 * points[:, 0]) + (c + 1.0) * others ** 2 + c * points[:, 0] * points[:, -1]
                     for c in range(q)], axis=1)


class Recorder:
    """A model that remembers the points it received and the outputs it gave, one entry for each call."""

    def __init__(self, function):
        self.function = function
        self.calls = []
        self.outputs = []

    def __call__(self, points):
        self.calls.append(np.array(points))
        output = np.array(self.function(points))
        self.outputs.append(output)
        return output


def textbook(fA, fB, fAB):
    """The formulas of the specification, with loops: S, ST (d, q), and f0, V (q,). fA, fB: (n, q); fAB: d of (n, q)."""
    n, q = fA.shape
    d = len(fAB)
    first = np.zeros((d, q))
    total = np.zeros((d, q))
    means = np.zeros(q)
    variances = np.zeros(q)
    for j in range(q):
        pooled = [float(v) for v in fA[:, j]] + [float(v) for v in fB[:, j]]
        f0 = sum(pooled) / (2 * n)
        variance = sum((v - f0) ** 2 for v in pooled) / (2 * n)
        for i in range(d):
            numerator_first = sum((fB[r, j] - f0) * (fAB[i][r, j] - fA[r, j]) for r in range(n)) / n
            numerator_total = sum((fA[r, j] - fAB[i][r, j]) ** 2 for r in range(n)) / n
            first[i, j] = numerator_first / variance
            total[i, j] = numerator_total / (2.0 * variance)
        means[j] = f0
        variances[j] = variance
    return first, total, means, variances


def assert_same_result(result, expected):
    """Every number of the two results is identical, bit for bit."""
    for name in ("first_order", "total_order", "first_order_ci", "total_order_ci", "mean", "variance"):
        np.testing.assert_array_equal(getattr(result, name), getattr(expected, name), err_msg=name)
    assert result.n_evaluations == expected.n_evaluations


def linear_case(n, **kwargs):
    return sobol(linear(LIN_WEIGHTS), LIN_LOW, LIN_HIGH, n, **kwargs)


def widths(interval):
    return interval[1] - interval[0]


def bootstrap_widths(fA, fB, fAB, confidence, resamples, seed):
    """The widths of the bootstrap intervals of S_i and ST_i, made here with their own random numbers.

    fA and fB are (n, 1) arrays and fAB is a list of d such arrays: the outputs of one run, with one output."""
    rng = np.random.default_rng(seed)
    n, d = fA.shape[0], len(fAB)
    first = np.empty((resamples, d))
    total = np.empty((resamples, d))
    for r in range(resamples):
        rows = rng.integers(0, n, size=n)
        a, b = fA[rows, 0], fB[rows, 0]
        f0 = np.concatenate([a, b]).mean()
        variance = np.concatenate([a, b]).var()
        for i in range(d):
            c = fAB[i][rows, 0]
            first[r, i] = np.mean((b - f0) * (c - a)) / variance
            total[r, i] = np.mean((a - c) ** 2) / (2.0 * variance)
    tail = 100.0 * (1.0 - confidence) / 2.0
    low, high = np.percentile(first, [tail, 100.0 - tail], axis=0)
    low_total, high_total = np.percentile(total, [tail, 100.0 - tail], axis=0)
    return high - low, high_total - low_total


@pytest.fixture
def global_random_state():
    """Leaves the global random state of NumPy as it was found."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


# ---------------------------------------------------------------------------------------------------------------------
# By hand: the examples of the introduction
# ---------------------------------------------------------------------------------------------------------------------


def test_sobol_hand_two_x1_plus_x2_gives_four_fifths_and_one_fifth():
    result = sobol(lambda points: 2.0 * points[:, 0] + points[:, 1], [0.0, 0.0], [1.0, 1.0], 1024, n_bootstrap=5)
    np.testing.assert_allclose(result.first_order, [0.8, 0.2], atol=0.01)
    np.testing.assert_allclose(result.total_order, [0.8, 0.2], atol=0.01)


def test_sobol_hand_a_product_of_two_centred_parameters_is_all_interaction():
    result = sobol(lambda points: points[:, 0] * points[:, 1], [-1.0, -1.0], [1.0, 1.0], 4096, n_bootstrap=5)
    np.testing.assert_allclose(result.first_order, [0.0, 0.0], atol=0.04)
    np.testing.assert_allclose(result.total_order, [1.0, 1.0], atol=0.04)


def test_sobol_hand_a_parameter_that_the_model_ignores_has_exactly_zero_indices_and_intervals():
    result = linear_case(256)
    last = len(LIN_WEIGHTS) - 1
    np.testing.assert_allclose(result.first_order[last], 0.0, atol=1e-12)
    np.testing.assert_allclose(result.total_order[last], 0.0, atol=1e-12)
    np.testing.assert_allclose(result.first_order_ci[:, last], 0.0, atol=1e-12)
    np.testing.assert_allclose(result.total_order_ci[:, last], 0.0, atol=1e-12)


def test_sobol_hand_with_one_parameter_the_whole_variance_is_its_own():
    result = sobol(lambda points: 3.0 * points[:, 0] + 1.0, [0.0], [1.0], 1024, n_bootstrap=5)
    np.testing.assert_allclose(result.first_order, [1.0], atol=0.02)
    np.testing.assert_allclose(result.total_order, [1.0], atol=0.02)


def test_sobol_hand_the_mean_and_the_variance_of_a_uniform_parameter():
    result = sobol(lambda points: points[:, 0], [2.0], [5.0], 1024, n_bootstrap=5)
    assert float(result.mean) == pytest.approx(3.5, abs=0.01)
    assert float(result.variance) == pytest.approx(9.0 / 12.0, rel=0.01)


# ---------------------------------------------------------------------------------------------------------------------
# The calls of the model: how many, with which points
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("d, n", [(1, 4), (2, 8), (3, 16), (5, 32), (8, 2)])
def test_sobol_calls_there_are_d_plus_two_calls_of_n_points_of_d_floats(d, n):
    recorder = Recorder(linear(range(1, d + 1)))
    result = sobol(recorder, [0.0] * d, [1.0] * d, n, n_bootstrap=3)
    assert len(recorder.calls) == d + 2
    for points in recorder.calls:
        assert points.shape == (n, d)
        assert points.dtype == np.float64
    assert result.n_evaluations == n * (d + 2)
    assert type(result.n_evaluations) is int


def test_sobol_calls_the_confidence_intervals_do_not_call_the_model_again():
    recorder = Recorder(linear([1.0, 2.0]))
    sobol(recorder, [0.0, 0.0], [1.0, 1.0], 16, n_bootstrap=500)
    assert len(recorder.calls) == 4


def test_sobol_calls_the_points_are_inside_the_box_and_not_on_its_boundary():
    low = np.array([-2.0, 0.0, 10.0])
    high = np.array([3.0, 1.0, 14.0])
    recorder = Recorder(linear([1.0, 1.0, 1.0]))
    sobol(recorder, low, high, 64, n_bootstrap=1)
    for points in recorder.calls:
        assert (points > low).all()
        assert (points < high).all()


def test_sobol_calls_AB_i_is_the_table_A_with_its_column_i_taken_from_B():
    d = 4
    recorder = Recorder(linear(range(1, d + 1)))
    sobol(recorder, [0.0] * d, [1.0] * d, 16, n_bootstrap=1)
    A, B = recorder.calls[0], recorder.calls[1]
    for i in range(d):
        expected = A.copy()
        expected[:, i] = B[:, i]
        np.testing.assert_array_equal(recorder.calls[2 + i], expected)


def test_sobol_calls_A_and_B_have_no_column_in_common():
    recorder = Recorder(linear([1.0, 1.0, 1.0]))
    sobol(recorder, [0.0] * 3, [1.0] * 3, 32, n_bootstrap=1)
    A, B = recorder.calls[0], recorder.calls[1]
    for j in range(3):
        for k in range(3):
            assert not np.array_equal(A[:, j], B[:, k])
        assert len(set(A[:, j])) == 32


def test_sobol_calls_every_column_of_A_and_B_has_one_point_in_each_of_the_n_equal_slices_of_the_box():
    low = np.array([-2.0, 0.0, 10.0])
    high = np.array([3.0, 1.0, 14.0])
    recorder = Recorder(linear([1.0, 1.0, 1.0]))
    sobol(recorder, low, high, 64, n_bootstrap=1)
    for points in recorder.calls[:2]:
        slices = np.floor((points - low) / (high - low) * 64).astype(int)
        for j in range(3):
            assert sorted(slices[:, j]) == list(range(64))


def test_sobol_calls_the_points_depend_on_the_seed_and_on_nothing_else():
    first = Recorder(linear([1.0, 2.0]))
    second = Recorder(lambda points: points[:, 0] ** 2 - points[:, 1])
    other = Recorder(linear([1.0, 2.0]))
    sobol(first, [0.0, 0.0], [1.0, 1.0], 16, seed=3, n_bootstrap=1)
    sobol(second, [0.0, 0.0], [1.0, 1.0], 16, seed=3, n_bootstrap=7, confidence=0.5)
    sobol(other, [0.0, 0.0], [1.0, 1.0], 16, seed=4, n_bootstrap=1)
    for points, same, different in zip(first.calls, second.calls, other.calls):
        np.testing.assert_array_equal(points, same)
        assert not np.array_equal(points, different)


def test_sobol_calls_the_bounds_may_be_given_as_lists_tuples_arrays_or_integers():
    expected = linear_case(64, n_bootstrap=5)
    variants = [
        (tuple(LIN_LOW), tuple(LIN_HIGH)),
        (np.array(LIN_LOW), np.array(LIN_HIGH)),
        ([0, -5, 100, 0], [1, 5, 102, 7]),
        (np.array([0, -5, 100, 0]), np.array([1, 5, 102, 7])),
        (np.array(LIN_LOW, dtype=np.float32), np.array(LIN_HIGH, dtype=np.float32)),
    ]
    for low, high in variants:
        assert_same_result(sobol(linear(LIN_WEIGHTS), low, high, 64, n_bootstrap=5), expected)


def test_sobol_calls_the_bounds_are_not_modified():
    low, high = np.array(LIN_LOW), np.array(LIN_HIGH)
    sobol(linear(LIN_WEIGHTS), low, high, 16, n_bootstrap=2)
    np.testing.assert_array_equal(low, LIN_LOW)
    np.testing.assert_array_equal(high, LIN_HIGH)


def test_sobol_calls_the_model_may_change_the_points_it_receives():
    def spoiler(points):
        result = linear(LIN_WEIGHTS)(points)
        points[:] = 0.5 * (points + 17.0)
        return result

    assert_same_result(sobol(spoiler, LIN_LOW, LIN_HIGH, 128, n_bootstrap=5),
                       linear_case(128, n_bootstrap=5))


def test_sobol_calls_the_model_may_return_the_same_array_at_every_call():
    buffer = np.empty(128)

    def economical(points):
        buffer[:] = linear(LIN_WEIGHTS)(points)
        return buffer

    assert_same_result(sobol(economical, LIN_LOW, LIN_HIGH, 128, n_bootstrap=5), linear_case(128, n_bootstrap=5))


def test_sobol_calls_the_model_may_return_a_read_only_array():
    def frozen(points):
        result = linear(LIN_WEIGHTS)(points)
        result.setflags(write=False)
        return result

    assert_same_result(sobol(frozen, LIN_LOW, LIN_HIGH, 128, n_bootstrap=5), linear_case(128, n_bootstrap=5))


def test_sobol_calls_the_model_may_return_lists_and_integers():
    expected = sobol(lambda points: np.floor(100.0 * linear(LIN_WEIGHTS)(points)), LIN_LOW, LIN_HIGH, 128,
                     n_bootstrap=5)
    as_list = sobol(lambda points: [float(v) for v in np.floor(100.0 * linear(LIN_WEIGHTS)(points))], LIN_LOW,
                    LIN_HIGH, 128, n_bootstrap=5)
    as_int = sobol(lambda points: np.floor(100.0 * linear(LIN_WEIGHTS)(points)).astype(int), LIN_LOW, LIN_HIGH, 128,
                   n_bootstrap=5)
    assert_same_result(as_list, expected)
    assert_same_result(as_int, expected)


# ---------------------------------------------------------------------------------------------------------------------
# The formulas, checked on the very points and outputs of the run: no statistics in these tests
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("d, q, n", [(1, 1, 8), (2, 1, 16), (3, 2, 32), (4, 3, 8), (2, 4, 64), (6, 1, 4)])
def test_sobol_formula_the_indices_are_those_of_the_formulas_on_the_outputs_of_the_run(d, q, n):
    recorder = Recorder(lambda points: wiggle(points, q))
    result = sobol(recorder, [0.0] * d, [1.0] * d, n, n_bootstrap=3)
    outputs = [np.asarray(values).reshape(n, q) for values in recorder.outputs]
    first, total, means, variances = textbook(outputs[0], outputs[1], outputs[2:])
    np.testing.assert_allclose(result.first_order, first, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.total_order, total, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.mean, means, rtol=1e-12)
    np.testing.assert_allclose(result.variance, variances, rtol=1e-12)


@pytest.mark.parametrize("n", [2, 4, 16])
def test_sobol_formula_a_model_with_one_output_follows_the_formulas_too(n):
    recorder = Recorder(lambda points: np.exp(points[:, 0]) * (1.0 + points[:, 1] ** 2) + points[:, 2])
    result = sobol(recorder, [0.0] * 3, [1.0] * 3, n, n_bootstrap=3)
    outputs = [np.asarray(values).reshape(n, 1) for values in recorder.outputs]
    first, total, means, variances = textbook(outputs[0], outputs[1], outputs[2:])
    np.testing.assert_allclose(result.first_order, first[:, 0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.total_order, total[:, 0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.mean, means[0], rtol=1e-12)
    np.testing.assert_allclose(result.variance, variances[0], rtol=1e-12)


@pytest.mark.parametrize("shift, scale", [(1e6, 1.0), (0.0, 1e8), (-3e4, 7e5), (0.0, 1e-9), (0.0, -2.0), (-1e4, -1e-2)])
def test_sobol_formula_the_indices_do_not_change_if_the_output_is_shifted_and_scaled(shift, scale):
    reference = sobol(lambda points: wiggle(points, 2), [0.0] * 3, [1.0] * 3, 64, n_bootstrap=20)
    changed = sobol(lambda points: shift + scale * wiggle(points, 2), [0.0] * 3, [1.0] * 3, 64, n_bootstrap=20)
    np.testing.assert_allclose(changed.first_order, reference.first_order, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(changed.total_order, reference.total_order, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(changed.first_order_ci, reference.first_order_ci, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(changed.mean, shift + scale * reference.mean, rtol=1e-9, atol=1e-9 * abs(scale))
    np.testing.assert_allclose(changed.variance, scale ** 2 * reference.variance, rtol=1e-6)


def test_sobol_formula_every_output_has_its_own_indices():
    both = sobol(lambda points: wiggle(points, 2), [0.0] * 3, [1.0] * 3, 128, n_bootstrap=7)
    for c in range(2):
        alone = sobol(lambda points: wiggle(points, 2)[:, c], [0.0] * 3, [1.0] * 3, 128, n_bootstrap=7)
        np.testing.assert_allclose(both.first_order[:, c], alone.first_order, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(both.total_order[:, c], alone.total_order, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(both.mean[c], alone.mean, rtol=1e-12)
        np.testing.assert_allclose(both.variance[c], alone.variance, rtol=1e-12)


def test_sobol_formula_the_order_of_the_outputs_does_not_matter():
    straight = sobol(lambda points: wiggle(points, 3), [0.0] * 3, [1.0] * 3, 64, n_bootstrap=11)
    reversed_ = sobol(lambda points: wiggle(points, 3)[:, ::-1], [0.0] * 3, [1.0] * 3, 64, n_bootstrap=11)
    for name in ("first_order", "total_order", "first_order_ci", "total_order_ci", "mean", "variance"):
        np.testing.assert_allclose(getattr(reversed_, name), getattr(straight, name)[..., ::-1], rtol=1e-12, atol=1e-14,
                                   err_msg=name)


# ---------------------------------------------------------------------------------------------------------------------
# Models with known indices
# ---------------------------------------------------------------------------------------------------------------------


def test_sobol_exact_a_linear_model_with_very_different_widths():
    result = linear_case(4096, n_bootstrap=5)
    np.testing.assert_allclose(result.first_order, LIN_INDICES, atol=0.01)
    np.testing.assert_allclose(result.total_order, LIN_INDICES, atol=0.01)
    assert result.first_order.sum() == pytest.approx(1.0, abs=0.01)


def test_sobol_exact_the_mean_and_the_variance_of_a_linear_model():
    result = linear_case(4096, n_bootstrap=5)
    weights, low, high = np.array(LIN_WEIGHTS), np.array(LIN_LOW), np.array(LIN_HIGH)
    assert float(result.mean) == pytest.approx(weights @ (low + high) / 2.0, rel=1e-3)
    assert float(result.variance) == pytest.approx(_lin_variances.sum(), rel=0.01)


def test_sobol_exact_the_function_of_ishigami():
    result = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 2 ** 14, n_bootstrap=5)
    np.testing.assert_allclose(result.first_order, ISHIGAMI_S, atol=0.03)
    np.testing.assert_allclose(result.total_order, ISHIGAMI_ST, atol=0.03)


def test_sobol_exact_the_function_of_ishigami_has_a_parameter_with_no_first_order_effect_but_a_total_one():
    result = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 2 ** 14, n_bootstrap=5)
    assert abs(result.first_order[2]) < 0.015
    assert result.total_order[2] > 0.2


def test_sobol_exact_the_g_function_of_sobol_with_five_parameters():
    result = sobol(g_function, [0.0] * 5, [1.0] * 5, 2 ** 14, n_bootstrap=5)
    np.testing.assert_allclose(result.first_order, G_S, atol=0.015)
    np.testing.assert_allclose(result.total_order, G_ST, atol=0.015)


def test_sobol_exact_the_product_of_two_positive_parameters_has_a_small_interaction():
    # f = x1 x2 on [1, 2]^2: S = 0.4909 for both parameters, ST = 0.5091.
    variance = (7.0 / 3.0) ** 2 - 1.5 ** 4
    first = (1.0 / 12.0) * 1.5 ** 2 / variance
    total = first + (1.0 / 12.0) ** 2 / variance
    result = sobol(lambda points: points[:, 0] * points[:, 1], [1.0, 1.0], [2.0, 2.0], 4096, n_bootstrap=5)
    np.testing.assert_allclose(result.first_order, [first, first], atol=0.005)
    np.testing.assert_allclose(result.total_order, [total, total], atol=0.005)
    assert (result.total_order > result.first_order).all()


def test_sobol_exact_the_parameters_are_not_mixed_up_when_the_model_is_not_symmetric():
    weights = [0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 0.0]
    result = sobol(linear(weights), [0.0] * 7, [1.0] * 7, 1024, n_bootstrap=5)
    np.testing.assert_allclose(result.first_order, [0.0, 0.0, 0.9, 0.0, 0.1, 0.0, 0.0], atol=0.02)
    np.testing.assert_allclose(result.total_order, [0.0, 0.0, 0.9, 0.0, 0.1, 0.0, 0.0], atol=0.02)


def test_sobol_exact_twenty_parameters_work():
    weights = np.arange(1.0, 21.0)
    result = sobol(linear(weights), [0.0] * 20, [1.0] * 20, 2048, n_bootstrap=3)
    expected = weights ** 2 / (weights ** 2).sum()
    assert result.first_order.shape == (20,)
    np.testing.assert_allclose(result.first_order, expected, atol=0.03)
    np.testing.assert_allclose(result.total_order, expected, atol=0.03)


def test_sobol_exact_more_points_give_a_smaller_error():
    def mean_error(n):
        errors = [np.abs(sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, n, seed=seed, n_bootstrap=1).first_order
                         - ISHIGAMI_S).max() for seed in range(4)]
        return np.mean(errors)

    assert mean_error(2 ** 14) < mean_error(2 ** 6) / 10.0


def test_sobol_exact_the_smallest_runs_give_finite_numbers():
    for d in (1, 2, 3):
        result = sobol(linear(range(1, d + 1)), [0.0] * d, [1.0] * d, 2, n_bootstrap=5)
        for name in ("first_order", "total_order", "first_order_ci", "total_order_ci", "mean", "variance"):
            assert np.isfinite(getattr(result, name)).all()
        assert result.n_evaluations == 2 * (d + 2)


# ---------------------------------------------------------------------------------------------------------------------
# The shape of the answer
# ---------------------------------------------------------------------------------------------------------------------


def test_sobol_result_one_output_gives_arrays_without_the_axis_of_the_outputs():
    result = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 16, n_bootstrap=4)
    assert result.first_order.shape == (3,)
    assert result.total_order.shape == (3,)
    assert result.first_order_ci.shape == (2, 3)
    assert result.total_order_ci.shape == (2, 3)
    assert np.shape(result.mean) == ()
    assert np.shape(result.variance) == ()


@pytest.mark.parametrize("q", [1, 2, 5])
def test_sobol_result_q_outputs_give_arrays_with_a_last_axis_of_length_q(q):
    result = sobol(lambda points: wiggle(points, q), [0.0] * 3, [1.0] * 3, 16, n_bootstrap=4)
    assert result.first_order.shape == (3, q)
    assert result.total_order.shape == (3, q)
    assert result.first_order_ci.shape == (2, 3, q)
    assert result.total_order_ci.shape == (2, 3, q)
    assert np.shape(result.mean) == (q,)
    assert np.shape(result.variance) == (q,)


def test_sobol_result_the_numbers_are_float64_and_the_arrays_are_new():
    result = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 16, n_bootstrap=4)
    for name in ("first_order", "total_order", "first_order_ci", "total_order_ci", "mean", "variance"):
        values = getattr(result, name)
        assert np.asarray(values).dtype == np.float64
    again = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 16, n_bootstrap=4)
    result.first_order[:] = 99.0
    result.total_order_ci[:] = 99.0
    assert_same_result(sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 16, n_bootstrap=4), again)
    assert result.first_order is not result.total_order
    assert not np.shares_memory(result.first_order_ci, result.total_order_ci)


# ---------------------------------------------------------------------------------------------------------------------
# The seed, and the other optional arguments
# ---------------------------------------------------------------------------------------------------------------------


def test_sobol_seed_the_same_seed_gives_the_same_answer_bit_for_bit():
    first = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 256, seed=5, n_bootstrap=20)
    again = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 256, seed=5, n_bootstrap=20)
    assert_same_result(first, again)


def test_sobol_seed_another_seed_gives_other_estimates_that_are_both_good():
    one = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 2 ** 14, seed=1, n_bootstrap=20)
    two = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 2 ** 14, seed=2, n_bootstrap=20)
    assert not np.array_equal(one.first_order, two.first_order)
    assert not np.array_equal(one.first_order_ci, two.first_order_ci)
    for result in (one, two):
        np.testing.assert_allclose(result.first_order, ISHIGAMI_S, atol=0.03)
        np.testing.assert_allclose(result.total_order, ISHIGAMI_ST, atol=0.03)


def test_sobol_seed_the_defaults_are_seed_0_200_resamples_and_95_per_cent():
    plain = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 64)
    explicit = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 64, seed=0, n_bootstrap=200, confidence=0.95)
    assert_same_result(plain, explicit)


def test_sobol_seed_the_answer_does_not_depend_on_the_global_random_state(global_random_state):
    np.random.seed(1)
    one = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 64, n_bootstrap=20)
    np.random.seed(2)
    two = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 64, n_bootstrap=20)
    assert_same_result(one, two)


def test_sobol_seed_the_global_random_state_is_left_alone(global_random_state):
    np.random.seed(7)
    expected = np.random.random(3)
    np.random.seed(7)
    sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 64, n_bootstrap=20)
    np.testing.assert_array_equal(np.random.random(3), expected)


def test_sobol_seed_the_estimates_do_not_depend_on_n_bootstrap_nor_on_the_confidence_level():
    base = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 128, n_bootstrap=10)
    for kwargs in ({"n_bootstrap": 1}, {"n_bootstrap": 77}, {"n_bootstrap": 10, "confidence": 0.5}):
        other = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 128, **kwargs)
        np.testing.assert_array_equal(other.first_order, base.first_order)
        np.testing.assert_array_equal(other.total_order, base.total_order)
        np.testing.assert_array_equal(other.mean, base.mean)
        np.testing.assert_array_equal(other.variance, base.variance)


def test_sobol_seed_integers_of_numpy_are_accepted():
    expected = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 64, seed=3, n_bootstrap=5)
    result = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, np.int64(64), seed=np.int32(3), n_bootstrap=np.int64(5),
                   confidence=np.float32(0.95))
    np.testing.assert_array_equal(result.first_order, expected.first_order)
    np.testing.assert_allclose(result.first_order_ci, expected.first_order_ci, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------------------------------------------------
# The confidence intervals
# ---------------------------------------------------------------------------------------------------------------------


def test_sobol_ci_the_lower_bound_is_not_above_the_upper_bound():
    result = sobol(lambda points: wiggle(points, 2), [0.0] * 3, [1.0] * 3, 64, n_bootstrap=50)
    assert (result.first_order_ci[0] <= result.first_order_ci[1]).all()
    assert (result.total_order_ci[0] <= result.total_order_ci[1]).all()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_sobol_ci_the_interval_holds_the_estimate_and_the_exact_value(seed):
    result = linear_case(1024, seed=seed)
    for estimate, interval in ((result.first_order, result.first_order_ci),
                               (result.total_order, result.total_order_ci)):
        assert (interval[0] <= estimate).all() and (estimate <= interval[1]).all()
        assert (interval[0] <= LIN_INDICES).all() and (LIN_INDICES <= interval[1]).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sobol_ci_the_interval_holds_the_exact_value_for_a_model_with_interactions(seed):
    result = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 1024, seed=seed)
    assert (result.first_order_ci[0] <= ISHIGAMI_S).all() and (ISHIGAMI_S <= result.first_order_ci[1]).all()
    assert (result.total_order_ci[0] <= ISHIGAMI_ST).all() and (ISHIGAMI_ST <= result.total_order_ci[1]).all()


def test_sobol_ci_a_parameter_that_matters_has_an_interval_of_positive_width():
    result = linear_case(256)
    assert (widths(result.first_order_ci)[:3] > 0.0).all()
    assert (widths(result.total_order_ci)[:3] > 0.0).all()


def test_sobol_ci_a_larger_confidence_gives_a_larger_interval_that_holds_the_smaller_one():
    narrow = linear_case(256, confidence=0.5)
    wide = linear_case(256, confidence=0.99)
    for name in ("first_order_ci", "total_order_ci"):
        small_interval, large_interval = getattr(narrow, name), getattr(wide, name)
        assert (large_interval[0] <= small_interval[0]).all()
        assert (small_interval[1] <= large_interval[1]).all()
        assert (widths(large_interval)[:3] > widths(small_interval)[:3]).all()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_sobol_ci_the_95_per_cent_interval_is_about_twice_as_wide_as_the_68_per_cent_one(seed):
    wide = linear_case(1024, seed=seed, confidence=0.95)
    narrow = linear_case(1024, seed=seed, confidence=0.6827)
    ratio = np.mean(widths(wide.first_order_ci)[:3] / widths(narrow.first_order_ci)[:3])
    assert 1.6 < ratio < 2.35


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_sobol_ci_sixteen_times_more_points_make_the_interval_about_four_times_narrower(seed):
    few = linear_case(2 ** 8, seed=seed)
    many = linear_case(2 ** 12, seed=seed)
    ratio = np.mean(widths(few.first_order_ci)[:3] / widths(many.first_order_ci)[:3])
    assert 2.8 < ratio < 6.0


def test_sobol_ci_with_one_resample_the_interval_has_no_width():
    result = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 64, n_bootstrap=1)
    np.testing.assert_array_equal(result.first_order_ci[0], result.first_order_ci[1])
    np.testing.assert_array_equal(result.total_order_ci[0], result.total_order_ci[1])


def test_sobol_ci_a_confidence_near_zero_gives_almost_a_point_and_near_one_the_whole_range_of_the_resamples():
    # With 3 resamples the percentile 0 is the smallest value, 50 the middle one and 100 the largest.
    wide = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 64, n_bootstrap=3, confidence=0.999)
    narrow = sobol(ishigami, ISHIGAMI_LOW, ISHIGAMI_HIGH, 64, n_bootstrap=3, confidence=0.001)
    for name in ("first_order_ci", "total_order_ci"):
        assert (widths(getattr(narrow, name)) < 0.01 * widths(getattr(wide, name))).all()
        assert (widths(getattr(wide, name)) > 0.0).all()


@pytest.mark.parametrize("confidence", [0.95, 0.8])
def test_sobol_ci_the_width_is_that_of_a_bootstrap_made_by_hand_on_the_outputs_of_the_run(confidence):
    recorder = Recorder(ishigami)
    result = sobol(recorder, ISHIGAMI_LOW, ISHIGAMI_HIGH, 256, n_bootstrap=1000, confidence=confidence)
    outputs = [np.asarray(values).reshape(256, 1) for values in recorder.outputs]
    first, total = bootstrap_widths(outputs[0], outputs[1], outputs[2:], confidence, 1000, seed=123)
    for ratio in (widths(result.first_order_ci) / first, widths(result.total_order_ci) / total):
        assert (ratio > 0.8).all() and (ratio < 1.25).all()


def test_sobol_ci_every_output_has_its_own_interval():
    result = sobol(lambda points: wiggle(points, 3), [0.0] * 3, [1.0] * 3, 64, n_bootstrap=20)
    widths_per_output = widths(result.first_order_ci).mean(axis=0)
    assert len(set(np.round(widths_per_output, 12))) == 3
    assert (widths(result.first_order_ci) > 0.0).all()


# ---------------------------------------------------------------------------------------------------------------------
# What is refused
# ---------------------------------------------------------------------------------------------------------------------


def good_arguments(**change):
    arguments = {"model": linear([1.0, 2.0]), "low": [0.0, 0.0], "high": [1.0, 1.0], "n": 16, "n_bootstrap": 3}
    arguments.update(change)
    return arguments


def test_sobol_refused_the_arguments_of_the_other_tests_are_good():
    result = sobol(**good_arguments())
    assert result.first_order.shape == (2,)


@pytest.mark.parametrize("n", [0, 1, 3, 5, 6, 12, 100, 1000, -4, -1, -8])
def test_sobol_refused_n_must_be_a_power_of_two_and_at_least_two(n):
    recorder = Recorder(linear([1.0, 2.0]))
    with pytest.raises(ValueError):
        sobol(**good_arguments(model=recorder, n=n))
    assert recorder.calls == []


@pytest.mark.parametrize("n", [2, 4, 8, 1024])
def test_sobol_refused_the_powers_of_two_are_accepted(n):
    assert sobol(**good_arguments(n=n)).n_evaluations == 4 * n


@pytest.mark.parametrize("n", [16.0, 16.5, "16", None, True, [16], np.float64(16.0), 1 + 0j])
def test_sobol_refused_n_must_be_an_integer(n):
    recorder = Recorder(linear([1.0, 2.0]))
    with pytest.raises(TypeError):
        sobol(**good_arguments(model=recorder, n=n))
    assert recorder.calls == []


@pytest.mark.parametrize("seed", [-1, -100])
def test_sobol_refused_the_seed_must_not_be_negative(seed):
    recorder = Recorder(linear([1.0, 2.0]))
    with pytest.raises(ValueError):
        sobol(**good_arguments(model=recorder, seed=seed))
    assert recorder.calls == []


@pytest.mark.parametrize("seed", [0, 1, 12345, 2 ** 40])
def test_sobol_refused_every_seed_that_is_not_negative_is_accepted(seed):
    sobol(**good_arguments(seed=seed))


@pytest.mark.parametrize("seed", [1.0, 0.5, "0", None, False, True, [0]])
def test_sobol_refused_the_seed_must_be_an_integer(seed):
    recorder = Recorder(linear([1.0, 2.0]))
    with pytest.raises(TypeError):
        sobol(**good_arguments(model=recorder, seed=seed))
    assert recorder.calls == []


@pytest.mark.parametrize("n_bootstrap", [0, -1, -50])
def test_sobol_refused_n_bootstrap_must_be_at_least_one(n_bootstrap):
    recorder = Recorder(linear([1.0, 2.0]))
    with pytest.raises(ValueError):
        sobol(**good_arguments(model=recorder, n_bootstrap=n_bootstrap))
    assert recorder.calls == []


@pytest.mark.parametrize("n_bootstrap", [2.0, 10.5, "10", None, True, [3]])
def test_sobol_refused_n_bootstrap_must_be_an_integer(n_bootstrap):
    recorder = Recorder(linear([1.0, 2.0]))
    with pytest.raises(TypeError):
        sobol(**good_arguments(model=recorder, n_bootstrap=n_bootstrap))
    assert recorder.calls == []


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 1.5, 2.0, 95.0, NAN, INF, -INF])
def test_sobol_refused_the_confidence_must_be_between_zero_and_one(confidence):
    recorder = Recorder(linear([1.0, 2.0]))
    with pytest.raises(ValueError):
        sobol(**good_arguments(model=recorder, confidence=confidence))
    assert recorder.calls == []


@pytest.mark.parametrize("confidence", [0.001, 0.5, 0.9, 0.999999, np.float64(0.9), np.float32(0.9)])
def test_sobol_refused_the_confidence_levels_between_zero_and_one_are_accepted(confidence):
    sobol(**good_arguments(confidence=confidence))


@pytest.mark.parametrize("confidence", ["0.9", None, [0.9], True, 1 + 0j])
def test_sobol_refused_the_confidence_must_be_a_number(confidence):
    recorder = Recorder(linear([1.0, 2.0]))
    with pytest.raises(TypeError):
        sobol(**good_arguments(model=recorder, confidence=confidence))
    assert recorder.calls == []


@pytest.mark.parametrize("model", [None, 3.0, "model", [1.0, 2.0], np.ones(3)])
def test_sobol_refused_the_model_must_be_callable(model):
    with pytest.raises(TypeError):
        sobol(**good_arguments(model=model))


BAD_BOUNDS = [
    ([0.0, 0.0], [1.0]),                  # not the same length
    ([0.0], [1.0, 1.0]),
    ([[0.0, 0.0]], [[1.0, 1.0]]),         # two axes
    ([], []),                             # no parameter at all
    (0.0, 1.0),                           # numbers, not sequences
    ([0.0, NAN], [1.0, 1.0]),             # not finite
    ([0.0, 0.0], [1.0, NAN]),
    ([0.0, 0.0], [1.0, INF]),
    ([-INF, 0.0], [1.0, 1.0]),
    ([0.0, 1.0], [1.0, 1.0]),             # low = high
    ([0.0, 2.0], [1.0, 1.0]),             # low > high
    ([1.0, 1.0], [0.0, 0.0]),
    ([0.0, 0.0], [0.0, 0.0]),
]


@pytest.mark.parametrize("low, high", BAD_BOUNDS)
def test_sobol_refused_the_bounds_must_be_two_sequences_of_finite_numbers_with_low_below_high(low, high):
    recorder = Recorder(linear([1.0, 2.0]))
    with pytest.raises(ValueError):
        sobol(**good_arguments(model=recorder, low=low, high=high))
    assert recorder.calls == []


@pytest.mark.parametrize("low, high", [
    (None, None), ("ab", "cd"), ({}, {}), ([None, 0.0], [1.0, 1.0]), (["a", "b"], [1.0, 1.0])])
def test_sobol_refused_the_bounds_must_be_numbers(low, high):
    recorder = Recorder(linear([1.0, 2.0]))
    with pytest.raises((TypeError, ValueError)):
        sobol(**good_arguments(model=recorder, low=low, high=high))
    assert recorder.calls == []


def refusing_model(function):
    """A model that is good except for one output: the shape of the answer, a NaN, a constant..."""
    state = {"calls": 0}

    def model(points):
        state["calls"] += 1
        return function(points, state["calls"])

    return model


BAD_OUTPUTS = {
    "one row too many": lambda points, call: np.arange(len(points) + 1, dtype=float),
    "one row too few": lambda points, call: np.arange(len(points) - 1, dtype=float),
    "twice as many rows": lambda points, call: np.concatenate([points[:, 0], points[:, 0] ** 2]),
    "twice as many rows, two columns": lambda points, call: np.concatenate([points, points ** 2]),
    "a single number": lambda points, call: 1.0,
    "no output at all": lambda points, call: None,
    "three axes": lambda points, call: np.zeros((len(points), 2, 2)) + points[:, :1, None],
    "no column": lambda points, call: np.zeros((len(points), 0)),
    "text": lambda points, call: ["a"] * len(points),
    "a NaN in the first call": lambda points, call: np.where(np.arange(len(points)) == 3, NAN, points[:, 0]),
    "an infinity in the last call": lambda points, call: np.where((np.arange(len(points)) == 2) & (call == 4), INF,
                                                                  points[:, 0]),
    "a NaN in a second column": lambda points, call: np.stack(
        [points[:, 0], np.where(np.arange(len(points)) == 1, NAN, points[:, 1])], axis=1),
    "a constant output": lambda points, call: np.full(len(points), 3.0),
    "a constant column": lambda points, call: np.stack([points[:, 0], np.full(len(points), -1.0)], axis=1),
    "the number of axes changes": lambda points, call: points[:, 0] if call == 1 else np.stack([points[:, 0]] * 2, 1),
    "a column is added at the second call": lambda points, call: points[:, 0] if call == 1 else points[:, :1],
    "a column is dropped at the second call": lambda points, call: points[:, :1] if call == 1 else points[:, 0],
    "a column is added at the last call": lambda points, call: points[:, 0] if call < 4 else points[:, :1],
    "the number of columns changes": lambda points, call: np.stack([points[:, 0]] * (2 if call < 3 else 3), axis=1),
    "the last two calls have one column more": lambda points, call: np.stack(
        [points[:, 0]] * (1 if call < 3 else 2), axis=1),
    "the last two calls have one column less": lambda points, call: np.stack(
        [points[:, 0]] * (2 if call < 3 else 1), axis=1),
    "the number of rows changes": lambda points, call: points[:, 0] if call < 3 else points[:-1, 0],
}


@pytest.mark.parametrize("name", sorted(BAD_OUTPUTS))
def test_sobol_refused_a_model_whose_output_is_not_right_is_refused_with_a_value_error(name):
    with pytest.raises(ValueError):
        sobol(**good_arguments(model=refusing_model(BAD_OUTPUTS[name]), n=8))


def test_sobol_refused_an_error_of_the_model_goes_through_unchanged():
    def breaks(points):
        raise ZeroDivisionError("the model has failed")

    with pytest.raises(ZeroDivisionError, match="the model has failed"):
        sobol(**good_arguments(model=breaks))


def test_sobol_refused_an_error_of_the_model_in_a_later_call_goes_through_unchanged():
    state = {"calls": 0}

    def breaks_late(points):
        state["calls"] += 1
        if state["calls"] == 3:
            raise RuntimeError("third call")
        return points[:, 0] + points[:, 1]

    with pytest.raises(RuntimeError, match="third call"):
        sobol(**good_arguments(model=breaks_late))
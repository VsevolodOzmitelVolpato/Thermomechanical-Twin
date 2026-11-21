"""Step 7b: the Galerkin reduced model of the transient heat equation, with imposed temperatures.

    from twin.galerkin import dirichlet_lift, reduced_theta_scheme, reconstruct, solve_reduced_heat

The full model is M dT/dt + K T = f (M mass, K stiffness, f the load of the heat source) with the temperature g imposed
on the fixed nodes. With the lifting T_g (g on the fixed nodes, 0 elsewhere) the unknown is the deviation d = T - T_g,
which is zero on the fixed nodes and solves M dd/dt + K d = f - K T_g. The reduced model looks for d = B a, where the
columns of B (the basis) vanish on the fixed nodes, and asks that the equations are orthogonal to the modes (Galerkin):

    M_r da/dt + K_r a = f_r,    M_r = B^T M B,  K_r = B^T K B,  f_r = B^T (f - K T_g)

dirichlet_lift(num_nodes, fixed_nodes, fixed_values)
    The lifting T_g: a float64 array (num_nodes,) with fixed_values[q] at the node fixed_nodes[q] and 0 elsewhere. A new
    array. ValueError if the two sequences have different lengths, if a node is not between 0 and num_nodes - 1
    (negative nodes included), if a node appears twice or if num_nodes < 1. TypeError if num_nodes or a node is not an
    integer.

reduced_theta_scheme(mass_r, stiffness_r, load_r, a0, dt, n_steps, theta)
    The theta-scheme of the reduced system. mass_r and stiffness_r are (r, r), load_r and a0 are (r,). Returns a float64
    array (n_steps + 1, r): the row s holds the coordinates after s steps and the row 0 is a0. Every step solves

        (M_r + theta dt K_r) a_new = (M_r - (1 - theta) dt K_r) a_old + dt f_r

    ValueError if the shapes do not fit, if dt is not > 0 or theta is not between 0 and 1 (NaN is not), if n_steps < 0,
    or if M_r + theta dt K_r is singular (n_steps >= 1). TypeError if n_steps is not an integer.

reconstruct(basis, coordinates, lift)
    basis is (n, r), lift is (n,) and coordinates is (r,) or (m, r). Returns lift + basis a for the vector a, or for
    every row a of coordinates: a float64 array (n,) or (m, n), always a new one. ValueError if the shapes do not fit.

solve_reduced_heat(basis, mass, stiffness, load, fixed_nodes, fixed_values, T0, dt, n_steps, theta)
    The whole reduced model. basis is (n, r), mass and stiffness are (n, n) (NumPy arrays or scipy.sparse matrices),
    load and T0 are (n,). The result is a float64 array (n_steps + 1, n): the temperature after s steps in the row s.
    The row 0 is not T0 but what the model sees of it: the lifting plus the projection of T0 - T_g onto the span of
    the basis, for the inner product of M. The columns of the basis must vanish on the fixed nodes: ValueError if the
    largest entry of the basis on those rows is more than 1e-8 times the largest entry of the whole basis. ValueError
    also if mass or stiffness is not n x n, if load or T0 does not have n entries, if M_r is singular, and for every
    error of dirichlet_lift and of reduced_theta_scheme (n_steps not an integer is a TypeError).
"""
import math

import numpy as np
import pytest
import scipy.sparse as sp

import thermomech

# Any warning is an error in these tests.
pytestmark = pytest.mark.filterwarnings("error")

PI = math.pi


# The functions are imported inside these wrappers and not at the top: while twin/galerkin.py does not exist, or does
# not have the function yet, only the tests of that function fail, not the whole session.


def lifting(*args, **kwargs):
    from twin.galerkin import dirichlet_lift
    return dirichlet_lift(*args, **kwargs)


def scheme(*args, **kwargs):
    from twin.galerkin import reduced_theta_scheme
    return reduced_theta_scheme(*args, **kwargs)


def rebuild(*args, **kwargs):
    from twin.galerkin import reconstruct
    return reconstruct(*args, **kwargs)


def reduced_heat(*args, **kwargs):
    from twin.galerkin import solve_reduced_heat
    return solve_reduced_heat(*args, **kwargs)


# ---------------------------------------------------------------- helpers


def dense(matrix):
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def random_spd(n, seed, shift=0.5):
    b = np.random.default_rng(seed).standard_normal((n, n))
    return b @ b.T / n + shift * np.eye(n)


def naive_scheme(mass_r, stiffness_r, load_r, a0, dt, n_steps, theta):
    """The equations of the scheme, one linear system per step."""
    states = [np.array(a0, dtype=float)]
    for _ in range(n_steps):
        rhs = (mass_r - (1.0 - theta) * dt * stiffness_r) @ states[-1] + dt * load_r
        states.append(np.linalg.solve(mass_r + theta * dt * stiffness_r, rhs))
    return np.array(states)


def every_second(values):
    wide = np.zeros(values.shape[:-1] + (2 * values.shape[-1],))
    wide[..., ::2] = values
    return wide[..., ::2]  # neighbouring entries are two entries apart in memory


def backwards(values):
    return values[::-1, ::-1].copy()[::-1, ::-1] if values.ndim == 2 else values[::-1].copy()[::-1]


def read_only(values):
    values = values.copy()
    values.setflags(write=False)
    return values


def fortran(values):
    return np.asfortranarray(values)


LAYOUTS = [every_second, backwards, read_only, fortran]


# ---------------------------------------------------------------- dirichlet_lift


def test_lift_hand_example_the_values_go_to_the_fixed_nodes():
    # 5 nodes, the node 3 at 7 and the node 0 at -2: the lifting is (-2, 0, 0, 7, 0)
    result = lifting(5, [3, 0], [7.0, -2.0])
    np.testing.assert_array_equal(result, [-2.0, 0.0, 0.0, 7.0, 0.0])


@pytest.mark.parametrize("nodes, values, expected", [
    ([0, 3], [1.5, -4.0], [1.5, 0.0, 0.0, -4.0]),  # the first and the last node
    ([2], [9.0], [0.0, 0.0, 9.0, 0.0]),
    ([1, 2], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),  # imposed values equal to zero
    ([3, 2, 1, 0], [4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]),  # all the nodes, in the reverse order
])
def test_lift_of_four_nodes(nodes, values, expected):
    np.testing.assert_array_equal(lifting(4, nodes, values), expected)


@pytest.mark.parametrize("nodes", [[], (), np.array([], dtype=np.int64), np.array([])],
                         ids=["list", "tuple", "int64", "float64"])
def test_lift_without_fixed_nodes_is_zero(nodes):
    result = lifting(4, nodes, [])
    np.testing.assert_array_equal(result, np.zeros(4))
    assert result.shape == (4,) and result.dtype == np.float64


def test_lift_is_a_new_float64_array():
    values = np.array([5.0, 6.0])
    result = lifting(6, [1, 4], values)
    assert isinstance(result, np.ndarray) and result.dtype == np.float64 and result.shape == (6,)
    assert result.flags["WRITEABLE"] and not np.shares_memory(result, values)
    result[1] = -1.0
    np.testing.assert_array_equal(values, [5.0, 6.0])
    again = lifting(6, [1, 4], values)
    assert again[1] == 5.0 and not np.shares_memory(again, result)


@pytest.mark.parametrize("make_nodes", [list, tuple, np.int8, np.int32, np.int64, np.uint8, np.uint64])
@pytest.mark.parametrize("make_values", [list, tuple, np.float64, np.float32, np.int64])
def test_lift_accepts_nodes_and_values_of_any_container_and_type(make_nodes, make_values):
    nodes = [4, 0, 2]
    values = [3, -1, 8]  # exact in every type
    nodes = nodes if make_nodes in (list, tuple) else np.array(nodes, dtype=make_nodes)
    nodes = tuple(nodes) if make_nodes is tuple else nodes
    values = values if make_values in (list, tuple) else np.array(values, dtype=make_values)
    values = tuple(values) if make_values is tuple else values
    result = lifting(5, nodes, values)
    np.testing.assert_array_equal(result, [-1.0, 0.0, 8.0, 0.0, 3.0])
    assert result.dtype == np.float64


def test_lift_num_nodes_can_be_a_numpy_integer():
    for make in (np.int8, np.int32, np.int64, np.uint8, np.uint64):
        np.testing.assert_array_equal(lifting(make(3), [1], [2.0]), [0.0, 2.0, 0.0])


def test_lift_does_not_depend_on_the_order_of_the_nodes():
    rng = np.random.default_rng(3)
    nodes = rng.permutation(40)[:15]
    values = rng.standard_normal(15)
    expected = np.zeros(40)
    for node, value in zip(nodes, values):
        expected[node] = value
    for seed in range(4):
        order = np.random.default_rng(seed).permutation(15)
        np.testing.assert_array_equal(lifting(40, nodes[order], values[order]), expected)


def test_lift_with_every_node_fixed():
    values = np.arange(1.0, 8.0)
    np.testing.assert_array_equal(lifting(7, np.arange(7), values), values)


def test_lift_arguments_can_be_given_by_name():
    result = lifting(num_nodes=3, fixed_nodes=[2], fixed_values=[1.0])
    np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])


LIFT_VALUE_ERRORS = {
    "a node equal to num_nodes": dict(num_nodes=4, fixed_nodes=[1, 4], fixed_values=[1.0, 2.0]),
    "a node beyond num_nodes": dict(num_nodes=4, fixed_nodes=[10], fixed_values=[1.0]),
    "the node -1": dict(num_nodes=4, fixed_nodes=[-1], fixed_values=[1.0]),
    "the node -num_nodes": dict(num_nodes=4, fixed_nodes=[-4], fixed_values=[1.0]),
    "a repeated node, other value": dict(num_nodes=4, fixed_nodes=[2, 2], fixed_values=[1.0, 2.0]),
    "a repeated node, same value": dict(num_nodes=4, fixed_nodes=[2, 0, 2], fixed_values=[1.0, 5.0, 1.0]),
    "more nodes than values": dict(num_nodes=4, fixed_nodes=[0, 1, 2], fixed_values=[1.0, 2.0]),
    "more values than nodes": dict(num_nodes=4, fixed_nodes=[0], fixed_values=[1.0, 2.0]),
    "one value for three nodes": dict(num_nodes=4, fixed_nodes=[0, 1, 2], fixed_values=[5.0]),
    "a number as the values": dict(num_nodes=4, fixed_nodes=[0, 1], fixed_values=5.0),
    "values with two dimensions": dict(num_nodes=4, fixed_nodes=[0, 1], fixed_values=[[1.0, 2.0]]),
    "no nodes and one value": dict(num_nodes=4, fixed_nodes=[], fixed_values=[1.0]),
    "num_nodes is zero": dict(num_nodes=0, fixed_nodes=[], fixed_values=[]),
    "num_nodes is negative": dict(num_nodes=-3, fixed_nodes=[], fixed_values=[]),
}


@pytest.mark.parametrize("arguments", list(LIFT_VALUE_ERRORS.values()), ids=list(LIFT_VALUE_ERRORS))
def test_lift_arguments_that_do_not_fit_are_a_value_error(arguments):
    with pytest.raises(ValueError):
        lifting(**arguments)


LIFT_TYPE_ERRORS = {
    "a float node": dict(num_nodes=4, fixed_nodes=[1.0], fixed_values=[1.0]),
    "a fractional node": dict(num_nodes=4, fixed_nodes=[1.5], fixed_values=[1.0]),
    "a node that is a string": dict(num_nodes=4, fixed_nodes=["1"], fixed_values=[1.0]),
    "a node that is None": dict(num_nodes=4, fixed_nodes=[None], fixed_values=[1.0]),
    "float nodes in an array": dict(num_nodes=4, fixed_nodes=np.array([1.0, 2.0]), fixed_values=[1.0, 2.0]),
    "num_nodes is a float": dict(num_nodes=4.0, fixed_nodes=[1], fixed_values=[1.0]),
    "num_nodes is a string": dict(num_nodes="4", fixed_nodes=[1], fixed_values=[1.0]),
    "num_nodes is None": dict(num_nodes=None, fixed_nodes=[1], fixed_values=[1.0]),
}


@pytest.mark.parametrize("arguments", list(LIFT_TYPE_ERRORS.values()), ids=list(LIFT_TYPE_ERRORS))
def test_lift_arguments_that_are_not_integers_are_a_type_error(arguments):
    with pytest.raises(TypeError):
        lifting(**arguments)


# ---------------------------------------------------------------- reduced_theta_scheme


def test_scheme_hand_example_one_unknown_and_no_load():
    # M = 1, K = 2, f = 0, a0 = 1, dt = 0.1: one step multiplies a by (1 - (1 - theta) 0.2) / (1 + theta 0.2), which is
    # 1 / 1.2 for theta = 1 (backward Euler), 0.9 / 1.1 for theta = 1/2 (Crank-Nicolson), 0.8 for theta = 0 (forward
    # Euler) and 0.85 / 1.05 for theta = 1/4
    for theta, factor in ((1.0, 1.0 / 1.2), (0.5, 0.9 / 1.1), (0.0, 0.8), (0.25, 0.85 / 1.05)):
        result = scheme([[1.0]], [[2.0]], [0.0], [1.0], 0.1, 2, theta)
        assert result.shape == (3, 1)
        np.testing.assert_allclose(result[:, 0], [1.0, factor, factor ** 2], rtol=1e-13, atol=0.0)


def test_scheme_hand_example_one_unknown_with_a_load():
    # M = 1, K = 2, f = 3, a0 = 0, dt = 0.1. Backward Euler: 1.2 a_new = a_old + 0.3 gives 0.25 and then 0.55 / 1.2.
    # Forward Euler: a_new = 0.8 a_old + 0.3 gives 0.3 and then 0.54
    result = scheme([[1.0]], [[2.0]], [3.0], [0.0], 0.1, 2, 1.0)
    np.testing.assert_allclose(result[:, 0], [0.0, 0.25, 0.55 / 1.2], rtol=1e-13, atol=0.0)
    result = scheme([[1.0]], [[2.0]], [3.0], [0.0], 0.1, 2, 0.0)
    np.testing.assert_allclose(result[:, 0], [0.0, 0.3, 0.54], rtol=1e-13, atol=0.0)


def test_scheme_hand_example_two_unknowns_backward_euler():
    # M = diag(2, 1), K = [[2, -1], [-1, 2]], f = (1, 0), a0 = 0, dt = 1, theta = 1: M + K = [[4, -1], [-1, 3]] has the
    # determinant 11 and the inverse [[3, 1], [1, 4]] / 11, so a1 = (3, 1) / 11. Then the right-hand side is
    # M a1 + f = (17, 1) / 11 and a2 = (51 + 1, 17 + 4) / 121 = (52, 21) / 121
    mass_r = np.diag([2.0, 1.0])
    stiffness_r = np.array([[2.0, -1.0], [-1.0, 2.0]])
    result = scheme(mass_r, stiffness_r, [1.0, 0.0], [0.0, 0.0], 1.0, 2, 1.0)
    np.testing.assert_allclose(result, [[0.0, 0.0], [3 / 11, 1 / 11], [52 / 121, 21 / 121]], rtol=1e-13, atol=1e-15)


def test_scheme_hand_example_two_unknowns_crank_nicolson():
    # the same M and K, f = 0, a0 = (1, 1), dt = 1, theta = 1/2: M + K / 2 = [[3, -1/2], [-1/2, 2]] has the determinant
    # 23/4 and the inverse (4/23) [[2, 1/2], [1/2, 3]]; M - K / 2 = [[1, 1/2], [1/2, 0]] gives the right-hand side
    # (3/2, 1/2), and a1 = (4/23) (3 + 1/4, 3/4 + 3/2) = (4/23) (13/4, 9/4) = (13, 9) / 23
    mass_r = np.diag([2.0, 1.0])
    stiffness_r = np.array([[2.0, -1.0], [-1.0, 2.0]])
    result = scheme(mass_r, stiffness_r, [0.0, 0.0], [1.0, 1.0], 1.0, 1, 0.5)
    np.testing.assert_allclose(result, [[1.0, 1.0], [13 / 23, 9 / 23]], rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("theta", [0.0, 0.3, 0.5, 1.0])
@pytest.mark.parametrize("r", [1, 2, 5, 12])
def test_scheme_agrees_with_one_linear_system_per_step(r, theta):
    rng = np.random.default_rng(r)
    mass_r, stiffness_r = random_spd(r, 1), random_spd(r, 2, shift=0.0)
    load_r, a0 = rng.standard_normal(r), rng.standard_normal(r)
    result = scheme(mass_r, stiffness_r, load_r, a0, 0.07, 9, theta)
    expected = naive_scheme(mass_r, stiffness_r, load_r, a0, 0.07, 9, theta)
    assert result.shape == (10, r)
    np.testing.assert_allclose(result, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("theta", [0.0, 0.4, 0.5, 1.0])
def test_scheme_modes_decay_by_the_closed_form_factor(theta):
    # with M = I, K = diag(lambda) and f = 0 the unknowns are independent, and one step multiplies the i-th one by
    # (1 - (1 - theta) lambda dt) / (1 + theta lambda dt)
    lam = np.array([0.5, 2.0, 7.5])
    dt, steps = 0.05, 12
    a0 = np.array([1.0, -2.0, 0.5])
    result = scheme(np.eye(3), np.diag(lam), np.zeros(3), a0, dt, steps, theta)
    factor = (1.0 - (1.0 - theta) * lam * dt) / (1.0 + theta * lam * dt)
    expected = a0 * factor[None, :] ** np.arange(steps + 1)[:, None]
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("theta", [0.0, 0.3, 0.5, 0.8, 1.0])
def test_scheme_keeps_the_steady_state(theta):
    # K a = f gives (M + theta dt K) a = (M - (1 - theta) dt K) a + dt f: the steady state does not move
    rng = np.random.default_rng(5)
    mass_r, stiffness_r, load_r = random_spd(4, 1), random_spd(4, 2), rng.standard_normal(4)
    steady = np.linalg.solve(stiffness_r, load_r)
    result = scheme(mass_r, stiffness_r, load_r, steady, 0.05, 6, theta)
    np.testing.assert_allclose(result, np.tile(steady, (7, 1)), rtol=0, atol=1e-12)


def test_scheme_backward_euler_goes_to_the_steady_state():
    rng = np.random.default_rng(6)
    mass_r, stiffness_r, load_r = random_spd(5, 1), random_spd(5, 2), rng.standard_normal(5)
    result = scheme(mass_r, stiffness_r, load_r, rng.standard_normal(5), 50.0, 60, 1.0)
    np.testing.assert_allclose(result[-1], np.linalg.solve(stiffness_r, load_r), rtol=0, atol=1e-10)


@pytest.mark.parametrize("theta, order", [(1.0, 1), (0.5, 2), (0.0, 1), (0.75, 1)])
def test_scheme_converges_to_the_exact_solution_with_the_order_of_the_theta_scheme(theta, order):
    # M = I and f = 0: the exact solution is exp(-K t) a0. Halving dt divides the error at t = 1 by 2 (Euler) or by 4
    # (Crank-Nicolson)
    rng = np.random.default_rng(1)
    b = rng.standard_normal((3, 3))
    stiffness_r = b @ b.T / 3 + 0.3 * np.eye(3)
    a0 = np.array([1.0, -0.5, 2.0])
    w, v = np.linalg.eigh(stiffness_r)
    exact = v @ (np.exp(-w) * (v.T @ a0))
    errors = [np.linalg.norm(scheme(np.eye(3), stiffness_r, np.zeros(3), a0, 1.0 / n, n, theta)[-1] - exact)
              for n in (20, 40, 80)]
    expected_ratio = 2.0 ** order
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert 0.85 * expected_ratio <= coarse / fine <= 1.15 * expected_ratio


def test_scheme_is_linear_in_the_initial_state_and_the_load():
    rng = np.random.default_rng(7)
    mass_r, stiffness_r = random_spd(6, 1), random_spd(6, 2)
    a1, a2, f1, f2 = (rng.standard_normal(6) for _ in range(4))
    for theta in (0.0, 0.5, 1.0):
        both = scheme(mass_r, stiffness_r, f1 + f2, a1 + a2, 0.04, 8, theta)
        separate = (scheme(mass_r, stiffness_r, f1, a1, 0.04, 8, theta)
                    + scheme(mass_r, stiffness_r, f2, a2, 0.04, 8, theta))
        np.testing.assert_allclose(both, separate, rtol=0, atol=1e-11)


@pytest.mark.parametrize("factor", [7.0, 1e-3])
def test_scheme_does_not_change_when_the_whole_system_is_scaled(factor):
    # c M da/dt + c K a = c f is the same equation
    rng = np.random.default_rng(8)
    mass_r, stiffness_r, load_r, a0 = random_spd(5, 1), random_spd(5, 2), rng.standard_normal(5), rng.standard_normal(5)
    result = scheme(mass_r, stiffness_r, load_r, a0, 0.1, 7, 0.6)
    scaled = scheme(factor * mass_r, factor * stiffness_r, factor * load_r, a0, 0.1, 7, 0.6)
    np.testing.assert_allclose(scaled, result, rtol=1e-11, atol=1e-12)


def test_scheme_more_steps_continue_the_same_trajectory():
    rng = np.random.default_rng(9)
    mass_r, stiffness_r, load_r, a0 = random_spd(4, 1), random_spd(4, 2), rng.standard_normal(4), rng.standard_normal(4)
    short = scheme(mass_r, stiffness_r, load_r, a0, 0.1, 5, 0.5)
    long = scheme(mass_r, stiffness_r, load_r, a0, 0.1, 12, 0.5)
    assert short.shape == (6, 4) and long.shape == (13, 4)
    np.testing.assert_allclose(long[:6], short, rtol=0, atol=1e-13)
    restarted = scheme(mass_r, stiffness_r, load_r, long[5], 0.1, 7, 0.5)
    np.testing.assert_allclose(restarted, long[5:], rtol=0, atol=1e-11)


def test_scheme_without_steps_returns_only_the_initial_state():
    result = scheme(np.eye(2), np.eye(2), [1.0, 2.0], [3.0, 4.0], 0.5, 0, 1.0)
    np.testing.assert_array_equal(result, [[3.0, 4.0]])
    assert result.shape == (1, 2)


def test_scheme_result_is_a_new_float64_array_and_the_arguments_are_not_modified():
    rng = np.random.default_rng(10)
    mass_r, stiffness_r, load_r, a0 = random_spd(3, 1), random_spd(3, 2), rng.standard_normal(3), rng.standard_normal(3)
    copies = [array.copy() for array in (mass_r, stiffness_r, load_r, a0)]
    result = scheme(mass_r, stiffness_r, load_r, a0, 0.1, 4, 0.5)
    assert isinstance(result, np.ndarray) and result.dtype == np.float64 and result.flags["WRITEABLE"]
    np.testing.assert_array_equal(result[0], a0)
    for array, copy in zip((mass_r, stiffness_r, load_r, a0), copies):
        np.testing.assert_array_equal(array, copy)
        assert not np.shares_memory(result, array)
    result[0, 0] = 99.0
    assert a0[0] != 99.0


@pytest.mark.parametrize("layout", LAYOUTS, ids=[layout.__name__ for layout in LAYOUTS])
def test_scheme_arrays_of_any_memory_layout_give_the_same_answer(layout):
    rng = np.random.default_rng(11)
    mass_r, stiffness_r, load_r, a0 = random_spd(4, 1), random_spd(4, 2), rng.standard_normal(4), rng.standard_normal(4)
    expected = scheme(mass_r, stiffness_r, load_r, a0, 0.1, 5, 0.5)
    result = scheme(layout(mass_r), layout(stiffness_r), layout(load_r), layout(a0), 0.1, 5, 0.5)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-14)


def test_scheme_accepts_lists_and_integers():
    result = scheme([[2, 0], [0, 1]], [[1, 0], [0, 1]], [1, 0], [0, 0], 1, 1, 1)
    np.testing.assert_allclose(result, [[0.0, 0.0], [1 / 3, 0.0]], rtol=1e-14, atol=0.0)


def test_scheme_accepts_the_limits_of_theta_and_a_numpy_number_of_steps():
    for theta in (0, 1, 0.0, 1.0, np.float32(0.5)):
        assert scheme(np.eye(2), np.eye(2), [0.0, 0.0], [1.0, 1.0], 0.01, 2, theta).shape == (3, 2)
    for make in (np.int8, np.int32, np.int64, np.uint8, np.uint64):
        assert scheme(np.eye(2), np.eye(2), [0.0, 0.0], [1.0, 1.0], 0.1, make(3), 1.0).shape == (4, 2)


def test_scheme_arguments_can_be_given_by_name():
    by_name = scheme(mass_r=np.eye(2), stiffness_r=np.eye(2), load_r=[1.0, 0.0], a0=[0.0, 0.0], dt=1.0, n_steps=1,
                     theta=1.0)
    np.testing.assert_allclose(by_name, [[0.0, 0.0], [0.5, 0.0]], rtol=1e-14, atol=0.0)


def a_good_scheme(**change):
    arguments = dict(mass_r=np.eye(3), stiffness_r=np.eye(3), load_r=np.ones(3), a0=np.zeros(3), dt=0.1, n_steps=2,
                     theta=0.5)
    arguments.update(change)
    return arguments


SCHEME_VALUE_ERRORS = {
    "dt is zero": dict(dt=0.0),
    "dt is negative": dict(dt=-0.1),
    "dt is NaN": dict(dt=math.nan),
    "theta is below 0": dict(theta=-0.1),
    "theta is above 1": dict(theta=1.1),
    "theta is NaN": dict(theta=math.nan),
    "n_steps is negative": dict(n_steps=-1),
    "mass_r is not square": dict(mass_r=np.ones((3, 2))),
    "mass_r is a column": dict(mass_r=np.ones((3, 1))),
    "mass_r is a row": dict(mass_r=np.ones((1, 3))),
    "mass_r is one-dimensional": dict(mass_r=np.ones(3)),
    "stiffness_r is too small": dict(stiffness_r=np.eye(2)),
    "stiffness_r is too big": dict(stiffness_r=np.eye(4)),
    "stiffness_r is not square": dict(stiffness_r=np.ones((3, 4))),
    "stiffness_r is a column": dict(stiffness_r=np.ones((3, 1))),
    "stiffness_r is a row": dict(stiffness_r=np.ones((1, 3))),
    "stiffness_r has one entry": dict(stiffness_r=np.ones((1, 1))),
    "load_r is too short": dict(load_r=np.ones(2)),
    "load_r is too long": dict(load_r=np.ones(4)),
    "load_r has one entry": dict(load_r=np.ones(1)),
    "a0 is too short": dict(a0=np.zeros(2)),
    "a0 is too long": dict(a0=np.zeros(4)),
    "a0 has one entry": dict(a0=np.zeros(1)),
    "a0 is a column": dict(a0=np.zeros((3, 1))),
    "M_r + theta dt K_r is singular": dict(mass_r=np.zeros((3, 3)), stiffness_r=np.zeros((3, 3))),
    "M_r is singular and theta is 0": dict(mass_r=np.ones((3, 3)), theta=0.0),
}


@pytest.mark.parametrize("change", list(SCHEME_VALUE_ERRORS.values()), ids=list(SCHEME_VALUE_ERRORS))
def test_scheme_arguments_that_do_not_fit_are_a_value_error(change):
    with pytest.raises(ValueError):
        scheme(**a_good_scheme(**change))


@pytest.mark.parametrize("n_steps", [2.0, 2.5, "2", None, [2], 1j], ids=lambda value: repr(value))
def test_scheme_n_steps_that_is_not_an_integer_is_a_type_error(n_steps):
    with pytest.raises(TypeError):
        scheme(**a_good_scheme(n_steps=n_steps))


ONE_MODE_ERRORS = {
    "load_r is a 1 x 1 array": dict(load_r=np.ones((1, 1))),
    "load_r is a number": dict(load_r=1.0),
    "a0 is a 1 x 1 array": dict(a0=np.ones((1, 1))),
    "a0 is a number": dict(a0=1.0),
}


@pytest.mark.parametrize("change", list(ONE_MODE_ERRORS.values()), ids=list(ONE_MODE_ERRORS))
def test_scheme_with_one_mode_the_load_and_the_start_are_still_vectors(change):
    arguments = a_good_scheme(mass_r=np.eye(1), stiffness_r=np.eye(1), load_r=np.ones(1), a0=np.ones(1))
    arguments.update(change)
    with pytest.raises(ValueError):
        scheme(**arguments)


# ---------------------------------------------------------------- reconstruct


def test_reconstruct_hand_example_the_lifting_plus_a_combination_of_the_modes():
    # basis columns (1, 0, 1) and (0, 1, 1), coordinates (2, -1), lifting (10, 20, 30):
    # 2 (1, 0, 1) - (0, 1, 1) = (2, -1, 1), plus the lifting (12, 19, 31)
    basis = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = rebuild(basis, [2.0, -1.0], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(result, [12.0, 19.0, 31.0])
    assert result.shape == (3,)


def test_reconstruct_hand_example_one_row_for_every_state():
    basis = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, -1.0]])
    result = rebuild(basis, coordinates, [10.0, 20.0, 30.0])
    expected = [[10.0, 20.0, 30.0], [11.0, 20.0, 31.0], [10.0, 22.0, 32.0], [12.0, 19.0, 31.0]]
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (4, 3)


def test_reconstruct_a_vector_and_a_matrix_of_coordinates_agree():
    rng = np.random.default_rng(1)
    basis, lift, coordinates = rng.standard_normal((9, 3)), rng.standard_normal(9), rng.standard_normal((5, 3))
    together = rebuild(basis, coordinates, lift)
    for row in range(5):
        single = rebuild(basis, coordinates[row], lift)
        assert single.shape == (9,)
        np.testing.assert_allclose(single, together[row], rtol=0, atol=1e-14)
        np.testing.assert_allclose(single, lift + sum(coordinates[row, i] * basis[:, i] for i in range(3)), rtol=0,
                                   atol=1e-14)


def test_reconstruct_with_zero_coordinates_gives_the_lifting_and_with_zero_lifting_the_combination():
    rng = np.random.default_rng(2)
    basis, lift = rng.standard_normal((6, 2)), rng.standard_normal(6)
    np.testing.assert_array_equal(rebuild(basis, np.zeros((3, 2)), lift), np.tile(lift, (3, 1)))
    coordinates = rng.standard_normal((4, 2))
    np.testing.assert_allclose(rebuild(basis, coordinates, np.zeros(6)), coordinates @ basis.T, rtol=0, atol=1e-14)


def test_reconstruct_a_single_mode_and_a_single_state():
    result = rebuild([[1.0], [2.0], [3.0]], [[2.0]], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(result, [[3.0, 5.0, 7.0]])
    assert rebuild([[1.0], [2.0], [3.0]], [2.0], [1.0, 1.0, 1.0]).shape == (3,)


def test_reconstruct_result_is_a_new_float64_array_and_the_arguments_are_not_modified():
    rng = np.random.default_rng(3)
    basis, lift, coordinates = rng.standard_normal((7, 3)), rng.standard_normal(7), rng.standard_normal((4, 3))
    copies = [array.copy() for array in (basis, lift, coordinates)]
    result = rebuild(basis, coordinates, lift)
    assert isinstance(result, np.ndarray) and result.dtype == np.float64 and result.flags["WRITEABLE"]
    for array, copy in zip((basis, lift, coordinates), copies):
        np.testing.assert_array_equal(array, copy)
        assert not np.shares_memory(result, array)
    zero = rebuild(basis, np.zeros(3), lift)
    assert not np.shares_memory(zero, lift)
    zero[0] = 123.0
    assert lift[0] != 123.0


@pytest.mark.parametrize("layout", LAYOUTS, ids=[layout.__name__ for layout in LAYOUTS])
def test_reconstruct_arrays_of_any_memory_layout_give_the_same_answer(layout):
    rng = np.random.default_rng(12)
    basis, lift, coordinates = rng.standard_normal((8, 3)), rng.standard_normal(8), rng.standard_normal((5, 3))
    expected = rebuild(basis, coordinates, lift)
    np.testing.assert_allclose(rebuild(layout(basis), layout(coordinates), layout(lift)), expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(rebuild(layout(basis), layout(coordinates[2]), layout(lift)), expected[2], rtol=0,
                               atol=1e-14)


def test_reconstruct_accepts_lists_integers_and_read_only_arrays():
    basis = np.array([[1, 0], [0, 1], [1, 1]])
    read_only = basis.copy()
    read_only.setflags(write=False)
    for arguments in ((basis.tolist(), [1, 2], [0, 0, 0]), (basis, np.array([1, 2]), np.zeros(3, dtype=np.int32)),
                      (read_only, (1, 2), (0, 0, 0))):
        result = rebuild(*arguments)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
        assert result.dtype == np.float64


def test_reconstruct_arguments_can_be_given_by_name():
    result = rebuild(basis=np.eye(2), coordinates=[1.0, 2.0], lift=[0.5, 0.5])
    np.testing.assert_array_equal(result, [1.5, 2.5])


def a_good_reconstruct(**change):
    arguments = dict(basis=np.ones((4, 2)), coordinates=np.ones((3, 2)), lift=np.ones(4))
    arguments.update(change)
    return arguments


RECONSTRUCT_ERRORS = {
    "basis is one-dimensional": dict(basis=np.ones(4)),
    "basis has three dimensions": dict(basis=np.ones((4, 2, 1))),
    "lift is too short": dict(lift=np.ones(3)),
    "lift is too long": dict(lift=np.ones(5)),
    "lift has one entry": dict(lift=np.ones(1)),
    "lift is a column": dict(lift=np.ones((4, 1))),
    "coordinates are too few": dict(coordinates=np.ones((3, 1))),
    "coordinates are too many": dict(coordinates=np.ones((3, 3))),
    "coordinates, one state, too few": dict(coordinates=np.ones(1)),
    "coordinates, one state, too many": dict(coordinates=np.ones(4)),
    "coordinates have three dimensions": dict(coordinates=np.ones((3, 2, 1))),
    "coordinates have three dimensions and the right last one": dict(coordinates=np.ones((3, 3, 2))),
    "coordinates are a number": dict(coordinates=np.array(1.0)),
}


@pytest.mark.parametrize("change", list(RECONSTRUCT_ERRORS.values()), ids=list(RECONSTRUCT_ERRORS))
def test_reconstruct_arguments_that_do_not_fit_are_a_value_error(change):
    with pytest.raises(ValueError):
        rebuild(**a_good_reconstruct(**change))


# ---------------------------------------------------------------- solve_reduced_heat: small problems

PATH_STIFFNESS = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])


def test_reduced_heat_hand_example_the_two_free_nodes_are_the_two_modes():
    # three nodes on a line, M = I, K = tridiag(-1, 2, -1), the heat load 1 on the node 1, the node 0 at 10.
    # Lifting (10, 0, 0), K T_g = (20, -10, 0), so f - K T_g = (-20, 11, 0). The modes are the unit vectors of
    # the nodes 1 and 2: M_r = I, K_r = [[2, -1], [-1, 2]], f_r = (11, 0).
    # Backward Euler with dt = 1, from T0 = T_g (so a0 = 0): [[3, -1], [-1, 3]] a1 = (11, 0), a1 = (33, 11) / 8,
    # and T1 = (10, 33/8, 11/8)
    basis = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = reduced_heat(basis, np.eye(3), PATH_STIFFNESS, [0.0, 1.0, 0.0], [0], [10.0], [10.0, 0.0, 0.0], 1.0, 1, 1.0)
    np.testing.assert_allclose(result, [[10.0, 0.0, 0.0], [10.0, 33 / 8, 11 / 8]], rtol=1e-13, atol=0.0)


def test_reduced_heat_hand_example_two_fixed_nodes_in_any_order():
    # four nodes on a line, M = I, K = tridiag(-1, 2, -1), no heat load, the node 3 at 8 and the node 0 at 2. Lifting
    # (2, 0, 0, 8), K T_g = (4, -2, -8, 16), so on the free nodes 1 and 2 the load is -K T_g = (2, 8). Backward Euler,
    # dt = 1, from T0 = T_g: [[3, -1], [-1, 3]] a1 = (2, 8), a1 = [[3, 1], [1, 3]] (2, 8) / 8 = (7/4, 13/4)
    stiffness = np.array([[2.0, -1.0, 0.0, 0.0], [-1.0, 2.0, -1.0, 0.0], [0.0, -1.0, 2.0, -1.0], [0.0, 0.0, -1.0, 2.0]])
    basis = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    start = [2.0, 0.0, 0.0, 8.0]
    result = reduced_heat(basis, np.eye(4), stiffness, np.zeros(4), [3, 0], [8.0, 2.0], start, 1.0, 1, 1.0)
    np.testing.assert_allclose(result, [[2.0, 0.0, 0.0, 8.0], [2.0, 1.75, 3.25, 8.0]], rtol=1e-13, atol=0.0)


def test_reduced_heat_hand_example_one_mode_with_a_mass_and_a_start_outside_the_span():
    # the same line of three nodes, M = diag(1, 2, 4), load (0, 1, 0), the node 0 at 10, one mode (0, 1, 1).
    # M_r = 2 + 4 = 6, K_r = (0, 1, 1) . K (0, 1, 1) = (0, 1, 1) . (-1, 1, 1) = 2, f_r = (0, 1, 1) . (-20, 11, 0) = 11.
    # T0 = (10, 3, 0) is not in the span: the deviation (0, 3, 0) has the coordinate (0, 1, 1) M (0, 3, 0) / M_r =
    # 6 / 6 = 1, so the model starts from (10, 1, 1).
    # Backward Euler, dt = 1: (6 + 2) a1 = 6 a0 + 11 gives a1 = 17/8, then 8 a2 = 6 a1 + 11 gives a2 = 95/32.
    # Crank-Nicolson: 7 a1 = 5 a0 + 11 gives a1 = 16/7, then 7 a2 = 5 a1 + 11 gives a2 = 157/49
    mass = np.diag([1.0, 2.0, 4.0])
    basis = np.array([[0.0], [1.0], [1.0]])
    arguments = (basis, mass, PATH_STIFFNESS, [0.0, 1.0, 0.0], [0], [10.0], [10.0, 3.0, 0.0], 1.0, 2)
    lifting_part = 10.0 * np.array([1.0, 0.0, 0.0])
    result = reduced_heat(*arguments, 1.0)
    coordinates = np.array([1.0, 17 / 8, 95 / 32])
    np.testing.assert_allclose(result, lifting_part + np.outer(coordinates, [0.0, 1.0, 1.0]), rtol=1e-13, atol=1e-13)
    result = reduced_heat(*arguments, 0.5)
    coordinates = np.array([1.0, 16 / 7, 157 / 49])
    np.testing.assert_allclose(result, lifting_part + np.outer(coordinates, [0.0, 1.0, 1.0]), rtol=1e-13, atol=1e-13)


def a_random_problem(n=14, r=4, seed=0, fixed=(2, 7, 11)):
    """Dense symmetric positive definite M and K, a basis that vanishes on the fixed nodes (not orthonormal), a random
    load, random imposed values and a random start T0 (not equal to the imposed values on the fixed nodes)."""
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((n, r))
    basis[list(fixed)] = 0.0
    return dict(basis=basis, mass=random_spd(n, seed + 100), stiffness=random_spd(n, seed + 200, shift=0.1),
                load=rng.standard_normal(n), fixed_nodes=list(fixed),
                fixed_values=list(rng.standard_normal(len(fixed))), T0=rng.standard_normal(n),
                dt=0.05, n_steps=6, theta=0.5)


def projection_of_the_start(problem):
    """lifting + the projection of T0 - lifting onto the span of the basis in the norm of M, from a least-squares
    problem: with M = L L^T the norm of M is the usual norm of L^T v."""
    lift = np.zeros(len(problem["load"]))
    lift[problem["fixed_nodes"]] = problem["fixed_values"]
    factor = np.linalg.cholesky(dense(problem["mass"]))
    coordinates = np.linalg.lstsq(factor.T @ problem["basis"], factor.T @ (problem["T0"] - lift), rcond=None)[0]
    return lift + problem["basis"] @ coordinates


@pytest.mark.parametrize("fixed", [(2, 7, 11), (), (0,), (13, 0)], ids=["three", "none", "first", "last and first"])
def test_reduced_heat_first_row_is_the_projection_of_the_start(fixed):
    problem = a_random_problem(fixed=fixed)
    problem["n_steps"] = 0
    result = reduced_heat(**problem)
    assert result.shape == (1, 14)
    np.testing.assert_allclose(result[0], projection_of_the_start(problem), rtol=0, atol=1e-11)


def test_reduced_heat_start_inside_the_span_is_kept_exactly():
    problem = a_random_problem()
    lift = np.zeros(14)
    lift[problem["fixed_nodes"]] = problem["fixed_values"]
    problem["T0"] = lift + problem["basis"] @ np.array([1.0, -2.0, 0.5, 3.0])
    result = reduced_heat(**problem)
    np.testing.assert_allclose(result[0], problem["T0"], rtol=0, atol=1e-11)


@pytest.mark.parametrize("theta, dt", [(0.0, 0.02), (0.5, 0.1), (0.7, 0.1), (1.0, 0.5)])
@pytest.mark.parametrize("fixed", [(2, 7, 11), ()], ids=["three fixed nodes", "no fixed nodes"])
def test_reduced_heat_every_step_is_orthogonal_to_the_modes_in_the_full_equations(fixed, theta, dt):
    # Galerkin: the residual A d_new - B d_old - dt (f - K T_g) of the full equations, with the deviations
    # d = T - T_g of the reduced states, is orthogonal to every mode. This defines the reduced model
    problem = a_random_problem(fixed=fixed)
    problem.update(theta=theta, dt=dt, n_steps=7)
    result = reduced_heat(**problem)
    assert result.shape == (8, 14)
    lift = np.zeros(14)
    lift[problem["fixed_nodes"]] = problem["fixed_values"]
    mass, stiffness = problem["mass"], problem["stiffness"]
    lhs = mass + theta * dt * stiffness
    rhs = mass - (1.0 - theta) * dt * stiffness
    for step in range(7):
        residual = (lhs @ (result[step + 1] - lift) - rhs @ (result[step] - lift)
                    - dt * (problem["load"] - stiffness @ lift))
        scale = np.abs(lhs @ (result[step + 1] - lift)).max() + np.abs(dt * problem["load"]).max()
        np.testing.assert_allclose(problem["basis"].T @ residual, 0.0, rtol=0, atol=1e-11 * scale)


def test_reduced_heat_keeps_the_imposed_temperatures():
    problem = a_random_problem()
    result = reduced_heat(**problem)
    np.testing.assert_allclose(result[:, problem["fixed_nodes"]], np.tile(problem["fixed_values"], (7, 1)), rtol=0,
                               atol=1e-13)


def test_reduced_heat_depends_on_the_span_of_the_basis_and_not_on_the_basis():
    problem = a_random_problem(seed=3)
    result = reduced_heat(**problem)
    change = np.random.default_rng(4).standard_normal((4, 4)) + 3.0 * np.eye(4)
    assert np.linalg.cond(change) < 20.0
    for other in (problem["basis"] @ change, 7.5 * problem["basis"], problem["basis"][:, ::-1], -problem["basis"]):
        np.testing.assert_allclose(reduced_heat(**{**problem, "basis": other}), result, rtol=0, atol=1e-8)


def test_reduced_heat_is_affine_in_the_start_and_the_load():
    problem = a_random_problem(seed=5)
    rng = np.random.default_rng(6)
    start_step, load_step = rng.standard_normal(14), rng.standard_normal(14)

    def run(start, load):
        return reduced_heat(**{**problem, "T0": problem["T0"] + start, "load": problem["load"] + load})

    np.testing.assert_allclose(run(start_step, load_step) + run(0.0 * start_step, 0.0 * load_step),
                               run(start_step, 0.0 * load_step) + run(0.0 * start_step, load_step), rtol=0, atol=1e-10)


@pytest.mark.parametrize("factor", [7.0, 1e-3])
def test_reduced_heat_does_not_change_when_the_whole_problem_is_scaled(factor):
    problem = a_random_problem(seed=7)
    result = reduced_heat(**problem)
    scaled = {**problem, "mass": factor * problem["mass"], "stiffness": factor * problem["stiffness"],
              "load": factor * problem["load"]}
    np.testing.assert_allclose(reduced_heat(**scaled), result, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo", "lil", "dense"])
def test_reduced_heat_mass_and_stiffness_can_be_dense_or_sparse_of_any_format(fmt):
    problem = a_random_problem(seed=8)
    expected = reduced_heat(**problem)
    make = (lambda matrix: matrix) if fmt == "dense" else (lambda matrix: sp.csr_matrix(matrix).asformat(fmt))
    other = {**problem, "mass": make(problem["mass"]), "stiffness": make(problem["stiffness"])}
    np.testing.assert_allclose(reduced_heat(**other), expected, rtol=0, atol=1e-12)


def test_reduced_heat_result_is_a_new_float64_array_and_the_arguments_are_not_modified():
    problem = a_random_problem(seed=9)
    copies = {name: np.array(value, copy=True) for name, value in problem.items() if isinstance(value, np.ndarray)}
    result = reduced_heat(**problem)
    assert isinstance(result, np.ndarray) and result.dtype == np.float64 and result.shape == (7, 14)
    assert result.flags["WRITEABLE"]
    for name, copy in copies.items():
        np.testing.assert_array_equal(problem[name], copy)
        assert not np.shares_memory(result, problem[name])
    assert problem["fixed_nodes"] == [2, 7, 11]


def test_reduced_heat_accepts_lists_and_arrays_of_integer_nodes():
    problem = a_random_problem(seed=10)
    expected = reduced_heat(**problem)
    given = problem["fixed_nodes"]
    for nodes in (tuple(given), np.array(given), np.array(given, dtype=np.int32)):
        other = {**problem, "fixed_nodes": nodes, "fixed_values": np.array(problem["fixed_values"])}
        np.testing.assert_array_equal(reduced_heat(**other), expected)
    other = {**problem, "basis": problem["basis"].tolist(), "load": problem["load"].tolist(),
             "T0": problem["T0"].tolist()}
    np.testing.assert_allclose(reduced_heat(**other), expected, rtol=0, atol=1e-13)


# ---------------------------------------------------------------- solve_reduced_heat against the full model (C++)

NX, NY = 6, 4
RHO_C, CONDUCTIVITY = 1.5, 0.8


def heat_spot(x, y):
    return 6.0 * math.exp(-((x - 1.2) ** 2 + (y - 0.4) ** 2) / 0.15)


class Plate:
    """A rectangle 2 x 1 of 7 x 5 nodes, heated by a spot, with the temperature imposed on the left side: 20 + 5 y."""

    def __init__(self):
        self.mesh = thermomech.rectangle_mesh(2.0, 1.0, NX, NY)
        self.n = self.mesh.num_nodes
        self.x, self.y = np.asarray(self.mesh.x), np.asarray(self.mesh.y)
        self.mass = thermomech.assemble_mass_2d(self.mesh, RHO_C)
        self.stiffness = thermomech.assemble_stiffness_2d(self.mesh, CONDUCTIVITY)
        self.load = thermomech.assemble_load_2d(self.mesh, heat_spot)
        self.left = [j * (NX + 1) for j in range(NY + 1)]
        self.border = [j * (NX + 1) + i for j in range(NY + 1) for i in range(NX + 1) if i in (0, NX) or j in (0, NY)]
        self.start = 20.0 + 3.0 * np.sin(PI * self.x / 4.0) * np.sin(PI * self.y)  # 20 on the left side

    def side(self, name):
        """(nodes, imposed temperatures, lifting, a start equal to the imposed temperatures on those nodes)."""
        nodes = {"left": self.left, "border": self.border, "none": []}[name]
        if name == "left":
            values = [20.0 + 5.0 * self.y[node] for node in nodes]
        else:
            values = [20.0 + self.x[node] for node in nodes]
        lift = np.zeros(self.n)
        lift[nodes] = values
        start = self.start.copy()
        start[nodes] = values
        return nodes, values, lift, start

    def full_model(self, start, load, dt, n_steps, theta, nodes, values):
        """The temperatures at the steps 0, 1, ..., n_steps of the C++ solver, one step at a time."""
        states = [np.array(start, dtype=float)]
        for _ in range(n_steps):
            states.append(thermomech.solve_transient_2d(self.mesh, RHO_C, CONDUCTIVITY, load, nodes, values,
                                                        states[-1], dt, 1, theta))
        return np.array(states)


@pytest.fixture(scope="module")
def plate():
    return Plate()


def orthonormal_basis(states, lift):
    """The columns of the QR factorization of the deviations T - T_g of the states."""
    return np.linalg.qr((states - lift).T)[0]


def pod_modes(states, lift, mass):
    """The POD modes (step 7a) of the deviations, as many as the singular values above 1e-6 of the first one."""
    from twin.pod import pod_basis
    snapshots = (states - lift).T
    sigma = pod_basis(snapshots, mass, 1)[1]
    return pod_basis(snapshots, mass, int(np.sum(sigma > 1e-6 * sigma[0])))[0]


STEP_FOR_THETA = [(0.0, 0.01), (0.5, 0.05), (2.0 / 3.0, 0.05), (1.0, 0.05)]  # forward Euler needs a small time step


@pytest.mark.parametrize("make_basis", [orthonormal_basis, pod_modes], ids=["qr", "pod"])
@pytest.mark.parametrize("theta, dt", STEP_FOR_THETA, ids=lambda value: format(value, ".3f"))
@pytest.mark.parametrize("side", ["left", "border"])
def test_reduced_heat_reproduces_the_full_model_when_the_basis_contains_its_trajectory(plate, side, theta, dt,
                                                                                       make_basis):
    # the reduced model is the Galerkin projection of the equations of the full model, so when every state of the full
    # model is a combination of the modes the two give the same states (up to the tolerance of the conjugate gradient)
    nodes, values, lift, start = plate.side(side)
    states = plate.full_model(start, plate.load, dt, 3, theta, nodes, values)
    basis = make_basis(states, lift) if make_basis is orthonormal_basis else make_basis(states, lift, plate.mass)
    assert basis.shape[1] == 4
    result = reduced_heat(basis, plate.mass, plate.stiffness, plate.load, nodes, values, start, dt, 3, theta)
    np.testing.assert_allclose(result, states, rtol=0, atol=1e-8)


def test_reduced_heat_without_fixed_nodes_reproduces_the_full_model(plate):
    # an insulated plate: the heat stays inside and the temperature rises everywhere
    nodes, values, lift, start = plate.side("none")
    states = plate.full_model(start, plate.load, 0.05, 4, 0.5, nodes, values)
    assert states[-1].mean() > states[0].mean() + 0.1
    basis = orthonormal_basis(states, lift)
    result = reduced_heat(basis, plate.mass, plate.stiffness, plate.load, nodes, values, start, 0.05, 4, 0.5)
    np.testing.assert_allclose(result, states, rtol=0, atol=1e-8)


@pytest.mark.parametrize("q", [2.0, -1.5, 5.0])
def test_reduced_heat_predicts_other_loads_when_the_states_are_affine_in_the_load(plate, q):
    # from the lifting, the deviations of the states are affine in the strength q of the load. Two runs, q = 1 and
    # q = 3, give modes that contain the states for every q
    nodes, values, lift, _ = plate.side("left")
    runs = [plate.full_model(lift, s * plate.load, 0.05, 3, 0.5, nodes, values) for s in (1.0, 3.0)]
    basis = np.linalg.qr(np.column_stack([(run[step] - lift) for run in runs for step in (1, 2, 3)]))[0]
    expected = plate.full_model(lift, q * plate.load, 0.05, 3, 0.5, nodes, values)
    result = reduced_heat(basis, plate.mass, plate.stiffness, q * plate.load, nodes, values, lift, 0.05, 3, 0.5)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-8)


def test_reduced_heat_steady_state_of_other_loads_is_exact_with_two_modes(plate):
    # the steady deviation is affine in the load q too: the steady states for q = 0 and q = 1 give two modes that
    # contain all of them, and a long run with backward Euler ends at the steady state of the full model
    nodes, values, lift, _ = plate.side("left")
    steady = [thermomech.solve_steady_2d(mesh=plate.mesh, k=CONDUCTIVITY, f=s * plate.load, fixed_nodes=nodes,
                                         fixed_values=values) for s in (0.0, 1.0)]
    basis = np.linalg.qr(np.column_stack([state - lift for state in steady]))[0]
    expected = thermomech.solve_steady_2d(mesh=plate.mesh, k=CONDUCTIVITY, f=4.5 * plate.load, fixed_nodes=nodes,
                                          fixed_values=values)
    result = reduced_heat(basis, plate.mass, plate.stiffness, 4.5 * plate.load, nodes, values, lift, 5.0, 40, 1.0)
    np.testing.assert_allclose(result[-1], expected, rtol=0, atol=1e-8)
    np.testing.assert_allclose(result[-1], result[-2], rtol=0, atol=1e-12)


def test_reduced_heat_steady_state_with_too_few_modes_is_the_galerkin_projection(plate):
    # with a single mode the steady state is far from the full one, but the error is orthogonal to the mode for the
    # stiffness (Galerkin): mode^T K (T_reduced - T_full) = 0. Only the mode itself would give a different answer
    nodes, values, lift, _ = plate.side("left")
    full = thermomech.solve_steady_2d(mesh=plate.mesh, k=CONDUCTIVITY, f=4.5 * plate.load, fixed_nodes=nodes,
                                      fixed_values=values)
    other = thermomech.solve_steady_2d(mesh=plate.mesh, k=CONDUCTIVITY, f=plate.load, fixed_nodes=nodes,
                                       fixed_values=values)
    basis = (other - lift)[:, None]
    result = reduced_heat(basis, plate.mass, plate.stiffness, 4.5 * plate.load, nodes, values, lift, 5.0, 40, 1.0)[-1]
    error = result - full
    assert np.abs(error).max() > 1.0  # the single mode does not contain the full steady state
    np.testing.assert_allclose(basis.T @ (plate.stiffness @ error), 0.0, rtol=0, atol=1e-9)


def test_reduced_heat_with_a_few_pod_modes_follows_a_long_run_of_the_full_model(plate):
    # 40 steps of Crank-Nicolson. The POD modes of the run are the best modes for its states: the reduced model with 8
    # of them is as good as they allow (its error is about their projection error) and is already below 1e-3
    nodes, values, lift, start = plate.side("left")
    states = plate.full_model(start, plate.load, 0.05, 40, 0.5, nodes, values)
    from twin.pod import pod_basis
    basis = pod_basis((states - lift).T, plate.mass, 8)[0]
    result = reduced_heat(basis, plate.mass, plate.stiffness, plate.load, nodes, values, start, 0.05, 40, 0.5)
    mass = dense(plate.mass)

    def size(fields):
        return max(math.sqrt(row @ mass @ row) for row in fields)

    scale = size(states - lift)
    assert size(result - states) <= 1e-3 * scale
    projection_error = size((states - lift) - (states - lift) @ mass @ basis @ basis.T)
    assert size(result - states) <= 3.0 * projection_error


# ---------------------------------------------------------------- solve_reduced_heat: wrong arguments


def a_good_heat(**change):
    problem = a_random_problem(seed=11)
    problem.update(change)
    return problem


def with_a_zero_column():
    basis = a_random_problem(seed=11)["basis"].copy()
    basis[:, 1] = 0.0
    return basis


HEAT_VALUE_ERRORS = {
    "basis is one-dimensional": dict(basis=np.ones(14)),
    "basis has three dimensions": dict(basis=np.ones((14, 4, 1))),
    "mass is too small": dict(mass=np.eye(13)),
    "mass is too big": dict(mass=np.eye(15)),
    "mass is not square": dict(mass=np.ones((14, 13))),
    "mass is a sparse matrix of the wrong size": dict(mass=sp.identity(13, format="csr")),
    "stiffness is too small": dict(stiffness=np.eye(13)),
    "stiffness is too big": dict(stiffness=np.eye(15)),
    "stiffness is not square": dict(stiffness=np.ones((14, 13))),
    "stiffness is a sparse matrix of the wrong size": dict(stiffness=sp.identity(15, format="csr")),
    "load is too short": dict(load=np.ones(13)),
    "load is too long": dict(load=np.ones(15)),
    "load has one entry": dict(load=np.ones(1)),
    "T0 is too short": dict(T0=np.ones(13)),
    "T0 is too long": dict(T0=np.ones(15)),
    "T0 has one entry": dict(T0=np.ones(1)),
    "a fixed node is equal to n": dict(fixed_nodes=[2, 14, 11]),
    "a fixed node is negative": dict(fixed_nodes=[2, 7, -1]),
    "a fixed node is repeated": dict(fixed_nodes=[2, 7, 2]),
    "fewer values than nodes": dict(fixed_values=[1.0, 2.0]),
    "more values than nodes": dict(fixed_values=[1.0, 2.0, 3.0, 4.0]),
    "one value for three nodes": dict(fixed_values=[1.0]),
    "dt is zero": dict(dt=0.0),
    "dt is negative": dict(dt=-0.05),
    "dt is NaN": dict(dt=math.nan),
    "theta is below 0": dict(theta=-0.1),
    "theta is above 1": dict(theta=1.5),
    "theta is NaN": dict(theta=math.nan),
    "n_steps is negative": dict(n_steps=-1),
    "the reduced mass matrix is singular": dict(basis=with_a_zero_column()),
}


@pytest.mark.parametrize("change", list(HEAT_VALUE_ERRORS.values()), ids=list(HEAT_VALUE_ERRORS))
def test_reduced_heat_arguments_that_do_not_fit_are_a_value_error(change):
    with pytest.raises(ValueError):
        reduced_heat(**a_good_heat(**change))


@pytest.mark.parametrize("n_steps", [2.0, 2.5, "2", None, [2], 1j], ids=lambda value: repr(value))
def test_reduced_heat_n_steps_that_is_not_an_integer_is_a_type_error(n_steps):
    with pytest.raises(TypeError):
        reduced_heat(**a_good_heat(n_steps=n_steps))


@pytest.mark.parametrize("node", [2, 7, 11])
def test_reduced_heat_a_basis_that_does_not_vanish_on_a_fixed_node_is_a_value_error(node):
    basis = a_random_problem(seed=11)["basis"].copy()
    basis[node, 2] = 1e-3 * np.abs(basis).max()
    with pytest.raises(ValueError):
        reduced_heat(**a_good_heat(basis=basis))


def test_reduced_heat_entries_of_the_basis_on_the_fixed_nodes_that_are_rounding_noise_are_accepted():
    problem = a_good_heat()
    noisy = problem["basis"].copy()
    noisy[[2, 7, 11], :] = 1e-13 * np.abs(noisy).max() * np.array([[1.0, -1.0, 1.0, -1.0]] * 3)
    np.testing.assert_allclose(reduced_heat(**{**problem, "basis": noisy}), reduced_heat(**problem), rtol=0, atol=1e-10)


@pytest.mark.parametrize("size, accepted", [(1e-9, True), (1e-7, False)], ids=["1e-9 accepted", "1e-7 rejected"])
def test_reduced_heat_the_limit_of_the_basis_rule_is_1e_minus_8_of_the_largest_entry(size, accepted):
    # one entry on a fixed node of the given size, relative to the largest entry of the whole basis
    problem = a_good_heat()
    basis = problem["basis"].copy()
    basis[7, 1] = size * np.abs(basis).max()
    if accepted:
        assert reduced_heat(**{**problem, "basis": basis}).shape == (problem["n_steps"] + 1, basis.shape[0])
    else:
        with pytest.raises(ValueError):
            reduced_heat(**{**problem, "basis": basis})


def test_reduced_heat_a_basis_made_from_the_temperatures_and_not_from_the_deviations_is_a_value_error(plate):
    # the snapshots T of the full model are not zero on the fixed nodes: the snapshots must be T - T_g
    from twin.pod import pod_basis
    nodes, values, lift, start = plate.side("left")
    states = plate.full_model(start, plate.load, 0.05, 6, 0.5, nodes, values)
    wrong = pod_basis(states.T, plate.mass, 3)[0]
    right = pod_basis((states - lift).T, plate.mass, 3)[0]
    with pytest.raises(ValueError):
        reduced_heat(wrong, plate.mass, plate.stiffness, plate.load, nodes, values, start, 0.05, 6, 0.5)
    result = reduced_heat(right, plate.mass, plate.stiffness, plate.load, nodes, values, start, 0.05, 6, 0.5)
    assert result.shape == (7, plate.n)
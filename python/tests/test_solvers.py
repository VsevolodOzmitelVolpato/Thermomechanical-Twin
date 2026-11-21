"""Step 6d: the heat solvers, from Python.

    solve_steady_2d(mesh, k, f, fixed_nodes, fixed_values)
    solve_transient_2d(mesh, rho_c, k, f, fixed_nodes, fixed_values, T0, dt, n_steps, theta)

Both return the temperature of the nodes: a NumPy array, float64, shape (n,), n = mesh.num_nodes. f, T0 and
fixed_values hold floats and fixed_nodes holds node numbers (integers); lists, tuples and arrays of any layout are
accepted, and none of them is modified. The node fixed_nodes[q] has the temperature fixed_values[q]. The steady solver
solves K T = f. The transient one marches the theta scheme
    (M + theta dt K) T_new = (M - (1 - theta) dt K) T_old + dt f
from T0 for n_steps steps. The errors of the C++ solvers arrive as ValueError (data that does not fit) and as
RuntimeError (the conjugate gradient does not converge, for instance with a NaN in the data).
"""
import math

import numpy as np
import pytest

import thermomech

PI = math.pi
CASES = [(1.0, 1.0, 1, 1), (2.0, 3.0, 4, 2), (4.0, 1.0, 8, 2), (1.0, 5.0, 2, 7), (0.1, 0.02, 5, 3),
         (10.0, 10.0, 16, 16)]
CASES_WITH_UNKNOWNS = CASES[1:]

# The numbers of the README tables, as the C++ tests of the verification problems fix them (relative tolerance 1e-6).
# Manufactured solution sin(pi x) sin(pi y) on the unit square: n, L2 error. It does not depend on k.
MANUFACTURED = [(4, 2.342328e-02), (8, 6.338199e-03), (16, 1.620920e-03)]
# Decaying sine: n, dt, n_steps, theta, L2 error at t = n_steps * dt (rho_c = 1, k = 1, no source, T = 0 on the border).
DECAYING_README = [(4, 2.5e-4, 200, 0.5, 2.452116e-02), (8, 2.5e-4, 200, 0.5, 6.894963e-03),
                   (16, 2.5e-4, 200, 0.5, 1.776746e-03),
                   (32, 0.05 / 4, 4, 1.0, 2.016970e-02), (32, 0.05 / 8, 8, 1.0, 1.035828e-02),
                   (32, 0.05 / 16, 16, 1.0, 5.092920e-03),
                   (32, 0.05 / 4, 4, 0.5, 1.386333e-03), (32, 0.05 / 8, 8, 0.5, 6.800985e-04),
                   (32, 0.05 / 16, 16, 0.5, 5.054105e-04)]
# The same problem, computed once more with dense matrices and Gaussian elimination (no conjugate gradient).
DECAYING_DENSE = [(4, 5e-4, 100, 0.5, 2.452250395721e-02), (8, 0.01, 5, 1.0, 1.018063914618e-02),
                  (8, 0.001, 10, 0.0, 4.073834024549e-03), (6, 0.002, 25, 0.3, 1.338984429158e-02)]


def make(case):
    length_x, length_y, nx, ny = case
    return thermomech.rectangle_mesh(length_x, length_y, nx, ny)


def border_nodes(nx, ny):
    """The nodes on the four sides of rectangle_mesh(.., nx, ny): the node of column i and row j is j*(nx+1) + i."""
    return np.array([j * (nx + 1) + i for j in range(ny + 1) for i in range(nx + 1) if i in (0, nx) or j in (0, ny)],
                    dtype=np.int64)


def left_and_right_nodes(nx, ny):
    left = [j * (nx + 1) for j in range(ny + 1)]
    right = [j * (nx + 1) + nx for j in range(ny + 1)]
    return np.array(left + right, dtype=np.int64)


def scale(values):
    return max(1.0, float(np.abs(values).max()))


def l2_error(mesh, T, T_exact):
    """sqrt(integral of (T - T_exact)^2), with the mass matrix of rho_c = 1, as the C++ l2_error_2d."""
    e = T - T_exact
    M = thermomech.assemble_mass_2d(mesh, 1.0)
    return math.sqrt(e @ (M @ e))


def reference_steady(mesh, k, f, fixed, values):
    """The same problem with dense matrices: eliminate the fixed nodes and solve the rest with numpy."""
    n = mesh.num_nodes
    K = thermomech.assemble_stiffness_2d(mesh, k).toarray()
    is_free = np.ones(n, dtype=bool)
    is_free[fixed] = False
    free = np.flatnonzero(is_free)
    T = np.zeros(n)
    T[fixed] = values
    T[free] = np.linalg.solve(K[np.ix_(free, free)], f[free] - K[np.ix_(free, fixed)] @ values)
    return T


def reference_transient(mesh, rho_c, k, f, fixed, values, T0, dt, n_steps, theta):
    n = mesh.num_nodes
    M = thermomech.assemble_mass_2d(mesh, rho_c).toarray()
    K = thermomech.assemble_stiffness_2d(mesh, k).toarray()
    A = M + theta * dt * K
    B = M - (1.0 - theta) * dt * K
    is_free = np.ones(n, dtype=bool)
    is_free[fixed] = False
    free = np.flatnonzero(is_free)
    T = np.array(T0, dtype=float)
    for _ in range(n_steps):
        rhs = B @ T + dt * f
        T_new = np.empty(n)
        T_new[fixed] = values
        T_new[free] = np.linalg.solve(A[np.ix_(free, free)], rhs[free] - A[np.ix_(free, fixed)] @ values)
        T = T_new
    return T


def manufactured_solution(n, k):
    """The temperature and the L2 error of the manufactured problem with n x n cells."""
    mesh = thermomech.rectangle_mesh(1.0, 1.0, n, n)
    f = thermomech.assemble_load_2d(mesh, lambda x, y: 2.0 * PI**2 * k * math.sin(PI * x) * math.sin(PI * y))
    fixed = border_nodes(n, n)
    T = thermomech.solve_steady_2d(mesh, k, f, fixed, np.zeros(len(fixed)))
    return T, l2_error(mesh, T, np.sin(PI * mesh.x) * np.sin(PI * mesh.y))


def decaying_error(n, dt, n_steps, theta):
    """The L2 error of the decaying sine exp(-2 pi^2 t) sin(pi x) sin(pi y) after n_steps steps."""
    mesh = thermomech.rectangle_mesh(1.0, 1.0, n, n)
    fixed = border_nodes(n, n)
    shape = np.sin(PI * mesh.x) * np.sin(PI * mesh.y)
    T = thermomech.solve_transient_2d(mesh, 1.0, 1.0, np.zeros(mesh.num_nodes), fixed, np.zeros(len(fixed)), shape,
                                      dt, n_steps, theta)
    return l2_error(mesh, T, math.exp(-2.0 * PI**2 * n_steps * dt) * shape)


def centre_of_a_2_by_2_plate(rho_c, k, q, dt, n_steps, theta, start):
    """The only unknown of the 2 x 2 plate (node 4, the centre) after n_steps steps, with the other 8 nodes at 0.
    K[4][4] = 4 k, M[4][4] = rho_c / 8 and the load of the centre is q / 4, so one step is
        (rho_c/8 + theta dt 4k) T_new = (rho_c/8 - (1 - theta) dt 4k) T_old + dt q/4."""
    left = rho_c / 8.0 + theta * dt * 4.0 * k
    right = rho_c / 8.0 - (1.0 - theta) * dt * 4.0 * k
    T = start
    for _ in range(n_steps):
        T = (right * T + dt * q / 4.0) / left
    return T


def test_the_module_has_the_solvers():
    for name in ("solve_steady_2d", "solve_transient_2d"):
        assert callable(getattr(thermomech, name))


# ---------------------------------------------------------------- the steady solver, by hand


def test_a_2_by_1_plate_with_a_cold_end_and_a_hot_end_by_hand():
    # nodes 0 1 2 below and 3 4 5 above; the ends 0, 3 at 0 and 2, 5 at 4. Row 1 of K is 2 T1 - T0/2 - T2/2 - T4 = 0
    # and row 4 is the same with T1 and T4 exchanged, so T1 = T4 = 2: the straight line between the two ends
    mesh = thermomech.rectangle_mesh(2.0, 1.0, 2, 1)
    T = thermomech.solve_steady_2d(mesh, 1.0, np.zeros(6), [0, 3, 2, 5], [0.0, 0.0, 4.0, 4.0])
    np.testing.assert_allclose(T, [0.0, 2.0, 4.0, 0.0, 2.0, 4.0], rtol=0, atol=1e-9)


@pytest.mark.parametrize("k, q", [(2.5, 10.0), (0.5, 3.0), (1.0, 16.0)])
def test_a_2_by_2_plate_with_a_source_has_one_unknown_by_hand(k, q):
    # 8 nodes on the border at 0 and the centre (node 4) free. 4 k T_c = f_c, and the load of the centre is q / 4
    # (6 triangles of area 1/8, a third of q A each), so T_c = q / (16 k)
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 2, 2)
    f = thermomech.assemble_load_2d(mesh, q)
    assert f[4] == pytest.approx(q / 4.0, rel=1e-12)
    T = thermomech.solve_steady_2d(mesh, k, f, border_nodes(2, 2), np.zeros(8))
    expected = np.zeros(9)
    expected[4] = q / (16.0 * k)
    np.testing.assert_allclose(T, expected, rtol=0, atol=1e-10)


def test_the_manufactured_solution_of_a_2_by_2_plate_by_hand():
    # T_c = pi^2 (1 + 2 sqrt 2) / 48 = 0.7872 (the exact value is 1) and, since M[4][4] = 1/8, the L2 error is
    # (1 - T_c) / (2 sqrt 2) = 0.0752
    T, error = manufactured_solution(2, 1.0)
    centre = PI**2 * (1.0 + 2.0 * math.sqrt(2.0)) / 48.0
    assert T[4] == pytest.approx(centre, rel=1e-10)
    assert error == pytest.approx((1.0 - centre) / (2.0 * math.sqrt(2.0)), rel=1e-9)


# ---------------------------------------------------------------- the steady solver, exact solutions


@pytest.mark.parametrize("case", CASES)
def test_a_linear_temperature_is_reproduced_exactly(case):
    length_x, length_y, nx, ny = case
    mesh = make(case)
    exact = 3.0 + 2.0 * mesh.x - 5.0 * mesh.y
    fixed = border_nodes(nx, ny)
    T = thermomech.solve_steady_2d(mesh, 2.5, np.zeros(mesh.num_nodes), fixed, exact[fixed])
    np.testing.assert_allclose(T, exact, rtol=0, atol=1e-8 * scale(exact))


@pytest.mark.parametrize("case", CASES)
def test_a_plate_with_insulated_sides_between_two_temperatures_is_a_straight_line(case):
    length_x, length_y, nx, ny = case
    mesh = make(case)
    fixed = left_and_right_nodes(nx, ny)
    values = np.where(mesh.x[fixed] < 0.5 * length_x, 20.0, 80.0)
    T = thermomech.solve_steady_2d(mesh, 3.0, np.zeros(mesh.num_nodes), fixed, values)
    np.testing.assert_allclose(T, 20.0 + 60.0 * mesh.x / length_x, rtol=0, atol=1e-7)


@pytest.mark.parametrize("case", CASES)
def test_a_plate_with_a_uniform_source_between_two_cold_ends_is_a_parabola(case):
    length_x, length_y, nx, ny = case
    mesh = make(case)
    k, q = 2.5, 7.0
    f = thermomech.assemble_load_2d(mesh, q)
    fixed = left_and_right_nodes(nx, ny)
    T = thermomech.solve_steady_2d(mesh, k, f, fixed, np.zeros(len(fixed)))
    exact = q * mesh.x * (length_x - mesh.x) / (2.0 * k)  # -k T'' = q, the nodes are exact for linear triangles
    np.testing.assert_allclose(T, exact, rtol=0, atol=1e-8 * scale(exact))


def test_the_fixed_nodes_get_exactly_the_imposed_temperatures():
    mesh = make(CASES[1])
    fixed = np.array([0, 7, 14, 3])
    values = np.array([0.1, 1.0 / 3.0, -2.5e-3, 1e5 + 0.123])
    T = thermomech.solve_steady_2d(mesh, 1.7, np.linspace(0.0, 9.0, mesh.num_nodes), fixed, values)
    np.testing.assert_array_equal(T[fixed], values)  # bit for bit, not "close"


def test_the_order_of_the_fixed_nodes_does_not_matter():
    mesh = make(CASES[3])
    n = mesh.num_nodes
    f = np.linspace(0.0, 2.0, n)
    fixed = np.array([0, 5, 9, 23, 17, 2])
    values = np.array([1.0, 2.0, -3.0, 4.0, 0.5, 6.0])
    expected = thermomech.solve_steady_2d(mesh, 1.3, f, fixed, values)
    order = [3, 0, 5, 1, 4, 2]
    T = thermomech.solve_steady_2d(mesh, 1.3, f, fixed[order], values[order])
    np.testing.assert_allclose(T, expected, rtol=1e-11, atol=1e-11)


def heated_plate():
    """A 2 x 3 plate (4 x 2 cells) with a source that depends on the place and temperatures imposed on its border."""
    mesh = make(CASES[1])
    f = thermomech.assemble_load_2d(mesh, lambda x, y: 1.0 + x * y)
    fixed = border_nodes(4, 2)
    return mesh, f, fixed, np.linspace(0.0, 1.0, len(fixed))


def test_multiplying_the_conductivity_and_the_source_by_the_same_number_changes_nothing():
    mesh, f, fixed, values = heated_plate()
    base = thermomech.solve_steady_2d(mesh, 1.0, f, fixed, values)
    scaled = thermomech.solve_steady_2d(mesh, 4.0, 4.0 * f, fixed, values)
    np.testing.assert_allclose(scaled, base, rtol=1e-9, atol=1e-9)


def test_the_steady_temperature_is_linear_in_the_source_and_in_the_imposed_temperatures():
    mesh, f, fixed, values = heated_plate()
    base = thermomech.solve_steady_2d(mesh, 1.0, f, fixed, values)
    twice = thermomech.solve_steady_2d(mesh, 1.0, 2.0 * f, fixed, 2.0 * values)
    np.testing.assert_allclose(twice, 2.0 * base, rtol=1e-9, atol=1e-9)


def test_halving_the_conductivity_doubles_the_temperature_of_a_plate_that_is_cold_on_its_border():
    mesh, f, fixed, _ = heated_plate()
    cold_border = np.zeros(len(fixed))
    normal = thermomech.solve_steady_2d(mesh, 1.0, f, fixed, cold_border)
    hotter = thermomech.solve_steady_2d(mesh, 0.5, f, fixed, cold_border)
    np.testing.assert_allclose(hotter, 2.0 * normal, rtol=1e-9, atol=1e-9)
    assert normal.max() > 0.0


@pytest.mark.parametrize("case", CASES_WITH_UNKNOWNS)
def test_the_steady_solution_matches_a_dense_solve_with_a_random_source_and_random_fixed_nodes(case):
    mesh = make(case)
    n = mesh.num_nodes
    rng = np.random.default_rng(2026)
    f = rng.uniform(0.0, 4.0, n)
    fixed = np.sort(rng.choice(n, size=max(2, n // 5), replace=False))
    values = rng.uniform(-2.0, 3.0, len(fixed))
    T = thermomech.solve_steady_2d(mesh, 1.7, f, fixed, values)
    expected = reference_steady(mesh, 1.7, f, fixed, values)
    np.testing.assert_allclose(T, expected, rtol=0, atol=1e-8 * scale(expected))


# ---------------------------------------------------------------- the steady solver, the README numbers


@pytest.mark.parametrize("n, want", MANUFACTURED)
def test_the_manufactured_solution_errors_match_the_readme_table(n, want):
    _, error = manufactured_solution(n, 1.0)
    assert error == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize("k", [7.5, 0.02])
def test_the_manufactured_solution_does_not_depend_on_the_conductivity(k):
    n, want = MANUFACTURED[1]
    _, error = manufactured_solution(n, k)
    assert error == pytest.approx(want, rel=1e-6)


# ---------------------------------------------------------------- the transient solver, by hand


CENTRE_CASES = [(1.0, 1.0, 0.0, 0.01, 2, 0.5), (1.0, 1.0, 0.0, 0.005, 4, 1.0), (1.0, 1.0, 0.0, 0.002, 5, 0.0),
                (1.0, 1.0, 0.0, 0.01, 3, 0.3), (1.0, 1.0, 0.0, 0.2, 3, 0.5),
                (2.0, 3.0, 0.0, 0.01, 4, 0.5), (2.0, 3.0, 5.0, 0.01, 6, 0.7), (0.5, 0.25, 10.0, 0.02, 7, 1.0)]


@pytest.mark.parametrize("rho_c, k, q, dt, n_steps, theta", CENTRE_CASES)
def test_a_2_by_2_plate_has_a_closed_form(rho_c, k, q, dt, n_steps, theta):
    # one unknown, so T_new = gamma T_old (+ a source term); for rho_c = k = 1 gamma = (1 - 32 (1-theta) dt) / (1 + 32
    # theta dt). With dt = 0.2 and theta = 1/2 gamma is negative: the temperature changes sign at every step
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 2, 2)
    fixed = np.array([0, 1, 2, 3, 5, 6, 7, 8])
    T0 = np.zeros(9)
    T0[4] = 3.0
    f = thermomech.assemble_load_2d(mesh, q)
    T = thermomech.solve_transient_2d(mesh, rho_c, k, f, fixed, np.zeros(8), T0, dt, n_steps, theta)
    assert T[4] == pytest.approx(centre_of_a_2_by_2_plate(rho_c, k, q, dt, n_steps, theta, 3.0), rel=1e-9, abs=1e-10)
    np.testing.assert_array_equal(T[fixed], np.zeros(8))


def test_zero_steps_give_back_the_initial_temperature():
    mesh = make(CASES[1])
    T0 = np.linspace(-3.0, 40.0, mesh.num_nodes)
    kept = T0.copy()
    T = thermomech.solve_transient_2d(mesh, 2.0, 3.0, np.ones(mesh.num_nodes), [0, 4], [100.0, 200.0], T0, 0.1, 0, 1.0)
    np.testing.assert_array_equal(T, kept)  # even the fixed nodes keep the values of T0: nothing is solved
    assert T is not T0 and not np.shares_memory(T, T0)
    np.testing.assert_array_equal(T0, kept)


# ---------------------------------------------------------------- the transient solver, physics that is exact


@pytest.mark.parametrize("theta", [0.0, 0.3, 0.5, 1.0])
def test_an_insulated_plate_with_a_uniform_source_heats_uniformly(theta):
    mesh = make(CASES[2])
    n = mesh.num_nodes
    rho_c, q, dt, n_steps = 4.0, 6.0, 0.01, 8
    f = thermomech.assemble_load_2d(mesh, q)
    T = thermomech.solve_transient_2d(mesh, rho_c, 1.5, f, [], [], np.full(n, 10.0), dt, n_steps, theta)
    np.testing.assert_allclose(T, 10.0 + q * n_steps * dt / rho_c, rtol=0, atol=1e-8)


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_an_insulated_plate_conserves_its_heat_and_a_source_adds_to_it(theta):
    mesh = make(CASES[1])
    n = mesh.num_nodes
    rho_c, dt, n_steps = 2.0, 0.02, 10
    M = thermomech.assemble_mass_2d(mesh, rho_c)
    T0 = 5.0 + np.sin(mesh.x) * np.cos(2.0 * mesh.y)
    heat0 = np.ones(n) @ (M @ T0)
    T = thermomech.solve_transient_2d(mesh, rho_c, 0.8, np.zeros(n), [], [], T0, dt, n_steps, theta)
    assert np.ones(n) @ (M @ T) == pytest.approx(heat0, rel=1e-10)
    assert T.max() - T.min() < T0.max() - T0.min()  # the heat spreads
    # with a uniform source q the heat grows by q * area * t (the plate is 2 x 3)
    f = thermomech.assemble_load_2d(mesh, 0.5)
    T = thermomech.solve_transient_2d(mesh, rho_c, 0.8, f, [], [], T0, dt, n_steps, theta)
    assert np.ones(n) @ (M @ T) == pytest.approx(heat0 + 0.5 * 6.0 * n_steps * dt, rel=1e-10)


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_a_temperature_that_is_already_steady_does_not_change(theta):
    length_x, length_y, nx, ny = CASES[1]
    mesh = make(CASES[1])
    exact = 3.0 + 2.0 * mesh.x - 5.0 * mesh.y
    fixed = border_nodes(nx, ny)
    f = np.zeros(mesh.num_nodes)
    T = thermomech.solve_transient_2d(mesh, 2.0, 1.5, f, fixed, exact[fixed], exact, 0.3, 5, theta)
    np.testing.assert_allclose(T, exact, rtol=0, atol=1e-8 * scale(exact))


def test_with_a_very_long_step_backward_euler_reaches_the_steady_temperature():
    mesh = make(CASES[1])
    f = thermomech.assemble_load_2d(mesh, lambda x, y: 1.0 + x * y)
    fixed = border_nodes(4, 2)
    values = np.linspace(0.0, 1.0, len(fixed))
    steady = thermomech.solve_steady_2d(mesh, 1.5, f, fixed, values)
    T = thermomech.solve_transient_2d(mesh, 2.0, 1.5, f, fixed, values, np.zeros(mesh.num_nodes), 1e6, 3, 1.0)
    np.testing.assert_allclose(T, steady, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("theta", [0.0, 0.3, 0.5, 0.75, 1.0])
def test_the_transient_solution_matches_a_dense_theta_scheme(theta):
    mesh = thermomech.rectangle_mesh(2.0, 1.0, 6, 3)
    n = mesh.num_nodes
    rng = np.random.default_rng(7)
    f = rng.uniform(0.0, 5.0, n)
    fixed = np.sort(rng.choice(n, size=6, replace=False))
    values = rng.uniform(-3.0, 3.0, len(fixed))
    T0 = rng.uniform(-1.0, 1.0, n)  # also on the fixed nodes: the first step starts from T0 as it is
    args = (mesh, 2.3, 1.7, f, fixed, values, T0, 0.002, 12, theta)
    expected = reference_transient(*args)
    T = thermomech.solve_transient_2d(*args)
    np.testing.assert_allclose(T, expected, rtol=0, atol=1e-8 * scale(expected))
    np.testing.assert_array_equal(T[fixed], values)


# ---------------------------------------------------------------- the transient solver, the README numbers


@pytest.mark.parametrize("n, dt, n_steps, theta, want", DECAYING_README)
def test_the_decaying_sine_errors_match_the_readme_tables(n, dt, n_steps, theta, want):
    assert decaying_error(n, dt, n_steps, theta) == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize("n, dt, n_steps, theta, want", DECAYING_DENSE)
def test_the_decaying_sine_errors_match_a_dense_reference(n, dt, n_steps, theta, want):
    assert decaying_error(n, dt, n_steps, theta) == pytest.approx(want, rel=1e-7)


# ---------------------------------------------------------------- the arguments


def steady_arguments(**change):
    """A valid call on the 2 x 2 plate, in the order of the parameters. No two arguments have the same value, so a
    function that mixes up two names gives another answer."""
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 2, 2)
    arguments = dict(mesh=mesh, k=2.0, f=np.linspace(1.0, 2.0, 9), fixed_nodes=[0, 1], fixed_values=[0.5, 1.5])
    arguments.update(change)
    return arguments


def transient_arguments(**change):
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 2, 2)
    arguments = dict(mesh=mesh, rho_c=2.0, k=3.0, f=np.linspace(1.0, 2.0, 9), fixed_nodes=[0, 1],
                     fixed_values=[0.5, 1.5], T0=np.linspace(2.0, 3.0, 9), dt=0.01, n_steps=2, theta=1.0)
    arguments.update(change)
    return arguments


def test_the_results_are_float64_arrays_that_belong_to_the_caller():
    steady = thermomech.solve_steady_2d(**steady_arguments())
    transient = thermomech.solve_transient_2d(**transient_arguments())
    for T in (steady, transient):
        assert isinstance(T, np.ndarray)
        assert T.dtype == np.float64
        assert T.shape == (9,)
        assert T.flags["C_CONTIGUOUS"] and T.flags["WRITEABLE"]
    # each call returns a new array: writing into one result changes nothing shared with the next call
    kept = steady.copy()
    steady[:] = 0.0
    again = thermomech.solve_steady_2d(**steady_arguments())
    np.testing.assert_array_equal(again, kept)
    assert not np.shares_memory(again, steady)


def test_the_arguments_are_not_modified():
    arguments = transient_arguments(f=np.linspace(0.0, 1.0, 9), fixed_nodes=np.array([0, 1]),
                                    fixed_values=np.array([0.5, 1.5]), T0=np.linspace(2.0, 3.0, 9))
    copies = {name: value.copy() for name, value in arguments.items() if isinstance(value, np.ndarray)}
    thermomech.solve_transient_2d(**arguments)
    for name, value in copies.items():
        np.testing.assert_array_equal(arguments[name], value, err_msg=name)


def test_lists_tuples_and_other_number_types_give_the_same_answer():
    n = 9
    base = dict(k=2.0, f=np.arange(n, dtype=float), fixed_nodes=np.array([0, 4, 8]),
                fixed_values=np.array([1.0, 2.0, 3.0]))
    reference = thermomech.solve_steady_2d(**steady_arguments(**base))
    variants = [dict(f=list(range(n))), dict(f=tuple(float(i) for i in range(n))), dict(f=np.arange(n)),
                dict(f=np.arange(n, dtype=np.float32)), dict(fixed_nodes=[0, 4, 8]), dict(fixed_nodes=(0, 4, 8)),
                dict(fixed_nodes=np.array([0, 4, 8], dtype=np.int32)),
                dict(fixed_nodes=np.array([0, 4, 8], dtype=np.uint8)), dict(fixed_values=[1, 2, 3]),
                dict(fixed_values=np.array([1.0, 2.0, 3.0], dtype=np.float32)), dict(k=2)]
    for change in variants:
        T = thermomech.solve_steady_2d(**steady_arguments(**dict(base, **change)))
        np.testing.assert_array_equal(T, reference, err_msg=str(change))


def test_the_initial_temperature_can_be_a_list_a_tuple_or_an_array_of_integers():
    reference = thermomech.solve_transient_2d(**transient_arguments(T0=np.arange(9, dtype=float)))
    for T0 in (list(range(9)), tuple(range(9)), np.arange(9), np.arange(9, dtype=np.float32)):
        T = thermomech.solve_transient_2d(**transient_arguments(T0=T0))
        np.testing.assert_array_equal(T, reference, err_msg=repr(T0))


def every_second(values):
    """The same numbers as a view of an array twice as long: neighbours are two elements apart in memory."""
    wide = np.zeros(2 * len(values), dtype=values.dtype)
    wide[::2] = values
    return wide[::2]


def backwards(values):
    """The same numbers as a view that runs backwards through memory (a negative stride)."""
    return values[::-1].copy()[::-1]


def read_only(values):
    values = values.copy()
    values.setflags(write=False)
    return values


LAYOUTS = [every_second, backwards, read_only]


@pytest.mark.parametrize("layout", LAYOUTS, ids=[layout.__name__ for layout in LAYOUTS])
def test_arrays_of_any_layout_are_read_correctly_by_the_steady_solver(layout):
    arguments = steady_arguments(f=np.linspace(1.0, 5.0, 9), fixed_nodes=np.array([0, 4, 8], dtype=np.uint64),
                                 fixed_values=np.array([0.5, 1.5, 2.5]))
    expected = thermomech.solve_steady_2d(**arguments)
    for name in ("f", "fixed_nodes", "fixed_values"):
        changed = dict(arguments)
        changed[name] = layout(arguments[name])
        np.testing.assert_array_equal(thermomech.solve_steady_2d(**changed), expected, err_msg=name)


@pytest.mark.parametrize("layout", LAYOUTS, ids=[layout.__name__ for layout in LAYOUTS])
def test_arrays_of_any_layout_are_read_correctly_by_the_transient_solver(layout):
    arguments = transient_arguments(f=np.linspace(1.0, 5.0, 9), fixed_nodes=np.array([0, 4, 8], dtype=np.uint64),
                                    fixed_values=np.array([0.5, 1.5, 2.5]), T0=np.linspace(-1.0, 3.0, 9))
    expected = thermomech.solve_transient_2d(**arguments)
    for name in ("f", "fixed_nodes", "fixed_values", "T0"):
        changed = dict(arguments)
        changed[name] = layout(arguments[name])
        np.testing.assert_array_equal(thermomech.solve_transient_2d(**changed), expected, err_msg=name)


def test_the_arrays_of_the_mesh_can_be_used_as_data():
    mesh = thermomech.rectangle_mesh(2.0, 1.0, 2, 1)  # the arrays x and y of a mesh are read-only
    T = thermomech.solve_transient_2d(mesh, 1.0, 1.0, mesh.y, [0], [0.0], mesh.x, 0.01, 1, 1.0)
    assert T.shape == (mesh.num_nodes,)
    assert T[0] == 0.0
    S = thermomech.solve_steady_2d(mesh, 1.0, mesh.x, mesh.triangles[0], mesh.y[mesh.triangles[0]])
    assert S.shape == (mesh.num_nodes,)


def test_the_arguments_can_be_given_by_name_and_as_numpy_integers():
    arguments = transient_arguments()
    by_position = thermomech.solve_transient_2d(*arguments.values())
    np.testing.assert_array_equal(thermomech.solve_transient_2d(**arguments), by_position)
    changed = transient_arguments(n_steps=np.int64(2), theta=1, dt=1e-2, k=3, rho_c=2)
    np.testing.assert_array_equal(thermomech.solve_transient_2d(**changed), by_position)
    arguments = steady_arguments()
    np.testing.assert_array_equal(thermomech.solve_steady_2d(**arguments),
                                  thermomech.solve_steady_2d(*arguments.values()))


def test_a_plate_without_fixed_nodes_is_allowed_only_when_it_is_transient():
    thermomech.solve_transient_2d(**transient_arguments(fixed_nodes=[], fixed_values=[]))
    thermomech.solve_transient_2d(**transient_arguments(fixed_nodes=np.array([]), fixed_values=np.array([])))
    with pytest.raises(ValueError):
        thermomech.solve_steady_2d(**steady_arguments(fixed_nodes=[], fixed_values=[]))


STEADY_BAD = {"k is zero": dict(k=0.0), "k is negative": dict(k=-1.0), "k is nan": dict(k=float("nan")),
              "f is too short": dict(f=np.ones(8)), "f is too long": dict(f=np.ones(10)),
              "f is empty": dict(f=np.ones(0)), "one value too few": dict(fixed_values=[0.0]),
              "one value too many": dict(fixed_values=[0.0, 1.0, 2.0]),
              "a node outside the mesh": dict(fixed_nodes=[0, 9]), "a node twice": dict(fixed_nodes=[1, 1])}
TRANSIENT_BAD = {"rho_c is zero": dict(rho_c=0.0), "rho_c is negative": dict(rho_c=-1.0),
                 "rho_c is nan": dict(rho_c=float("nan")), "k is zero": dict(k=0.0), "k is negative": dict(k=-1.0),
                 "k is nan": dict(k=float("nan")), "dt is zero": dict(dt=0.0), "dt is negative": dict(dt=-0.01),
                 "dt is nan": dict(dt=float("nan")), "theta is below 0": dict(theta=-0.1),
                 "theta is above 1": dict(theta=1.1), "theta is nan": dict(theta=float("nan")),
                 "T0 is too short": dict(T0=np.zeros(8)), "T0 is too long": dict(T0=np.zeros(10)),
                 "f is too short": dict(f=np.zeros(8)), "f is too long": dict(f=np.zeros(10)),
                 "T0 is empty": dict(T0=np.zeros(0)), "one value too few": dict(fixed_values=[0.0]),
                 "one value too many": dict(fixed_values=[0.0, 1.0, 2.0]),
                 "a node outside the mesh": dict(fixed_nodes=[0, 9]), "a node twice": dict(fixed_nodes=[1, 1]),
                 "a node twice, even with no step": dict(fixed_nodes=[1, 1], n_steps=0)}


@pytest.mark.parametrize("change", list(STEADY_BAD.values()), ids=list(STEADY_BAD))
def test_the_steady_solver_rejects_data_that_does_not_fit_with_a_value_error(change):
    with pytest.raises(ValueError):
        thermomech.solve_steady_2d(**steady_arguments(**change))


@pytest.mark.parametrize("change", list(TRANSIENT_BAD.values()), ids=list(TRANSIENT_BAD))
def test_the_transient_solver_rejects_data_that_does_not_fit_with_a_value_error(change):
    with pytest.raises(ValueError):
        thermomech.solve_transient_2d(**transient_arguments(**change))


def test_a_nan_in_the_data_is_never_returned_silently():
    # the conjugate gradient cannot converge when the data hold a NaN: the C++ solver throws, and Python gets the error
    with pytest.raises((ValueError, RuntimeError)):
        thermomech.solve_steady_2d(**steady_arguments(fixed_values=[0.5, float("nan")]))
    f = np.ones(9)
    f[4] = float("nan")
    with pytest.raises((ValueError, RuntimeError)):
        thermomech.solve_steady_2d(**steady_arguments(f=f))
    T0 = np.zeros(9)
    T0[4] = float("nan")
    with pytest.raises((ValueError, RuntimeError)):
        thermomech.solve_transient_2d(**transient_arguments(T0=T0))


@pytest.mark.parametrize("name, arguments", [("solve_steady_2d", steady_arguments),
                                             ("solve_transient_2d", transient_arguments)])
@pytest.mark.parametrize("change", [dict(mesh=None), dict(mesh=5), dict(k=None), dict(k="hot"), dict(f=None),
                                    dict(fixed_nodes=[0.5, 1]), dict(fixed_nodes=None), dict(fixed_values="ab")],
                         ids=["no mesh", "mesh is a number", "k is None", "k is a string", "f is None",
                              "a node that is not an integer", "fixed_nodes is None", "fixed_values is a string"])
def test_arguments_of_the_wrong_type_are_a_type_error(name, arguments, change):
    with pytest.raises(TypeError):
        getattr(thermomech, name)(**arguments(**change))


@pytest.mark.parametrize("change", [dict(n_steps=-1), dict(n_steps=2.5), dict(dt="slow"), dict(theta=None),
                                    dict(T0=None)], ids=["negative n_steps", "n_steps is a float", "dt is a string",
                                                         "theta is None", "T0 is None"])
def test_the_transient_arguments_of_the_wrong_type_are_a_type_error(change):
    with pytest.raises(TypeError):
        thermomech.solve_transient_2d(**transient_arguments(**change))


def test_a_negative_node_number_is_refused():
    with pytest.raises((TypeError, ValueError)):
        thermomech.solve_steady_2d(**steady_arguments(fixed_nodes=[0, -1]))
    with pytest.raises((TypeError, ValueError)):
        thermomech.solve_transient_2d(**transient_arguments(fixed_nodes=[0, -1]))
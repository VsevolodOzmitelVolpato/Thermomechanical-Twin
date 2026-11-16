"""Step 7c: the reduced thermo-mechanics, from the thermal modes to displacements, stresses and von Mises.

    from twin.mechanics import element_mean, von_mises_plane_strain, thermoelastic_response, mechanical_modes

The plate is in plane strain (as in the C++ solver) and in quasi-static equilibrium: at every instant K u = F(T), with
F(T) linear in the nodal temperatures T (the thermal strain of a triangle is (1 + nu) alpha (T_e - T_ref) (1, 1, 0),
with T_e the mean of the three nodal temperatures). So the displacements u(T) and the stresses sigma(T) = D (B u -
eps_th) are affine in T, and for a temperature T = T_g + B a (the lifting plus a combination of the thermal modes) the
superposition is exact:

    u(T_g + B a) = u(T_g) + U a,        U[:, i] = u(T_g + B[:, i]) - u(T_g)        (the same for sigma)

element_mean(triangles, values)
    triangles is an integer array (n_tri, 3) with the nodes of each triangle (mesh.triangles) and values holds one
    number per node, in the last axis: (n,) or (m, n). Returns the mean of the three nodal values of every triangle,
    a new float64 array (n_tri,) or (m, n_tri). ValueError if triangles is not (n_tri, 3), if values has not one or
    two dimensions, or if a triangle has a node that does not exist (negative nodes included).

von_mises_plane_strain(sigma, temperature_rise, E, nu, alpha)
    sigma has the plane stresses (xx, yy, xy) in its last axis, temperature_rise (T_e - T_ref of every triangle) has the
    shape of sigma without that axis. The stress out of the plane is sigma_zz = nu (xx + yy) - E alpha temperature_rise
    and the result is the von Mises stress

        sqrt(1/2 ((xx - yy)^2 + (yy - zz)^2 + (zz - xx)^2) + 3 xy^2)

    with the shape of temperature_rise (a new float64 array, or a number for a single stress). ValueError if E is not
    > 0, if nu is not between -1 and 1/2 (NaN is neither), if the last axis of sigma is not 3 or if temperature_rise has
    not the shape of sigma without its last axis.

thermoelastic_response(mesh, temperature, E, nu, alpha, T_ref, fixed_dofs, fixed_displacements)
    The displacements and the stresses of the plate in equilibrium with the nodal temperatures temperature (n,), for
    the material (E, nu, alpha, T_ref) and the constraints on the displacements (as in solve_thermomechanical_2d).
    Returns (u, sigma): new float64 arrays (2 n,) and (n_tri, 3). ValueError if temperature is not (n,) or if the data
    do not fit (as for the C++ solver, which also raises RuntimeError and TypeError).

mechanical_modes(mesh, basis, lift, E, nu, alpha, T_ref, fixed_dofs, fixed_displacements)
    The responses to the modes. basis is (n, r) and lift is (n,). Returns the new float64 arrays

        u_lift (2 n,)    u_modes (2 n, r)    sigma_lift (3 n_tri,)    sigma_modes (3 n_tri, r)

    where u_lift and sigma_lift are the response to the lifting and column i of u_modes and sigma_modes is the response
    to lift + basis[:, i] minus the response to the lifting. The stresses are flattened, the three components of the
    triangle e in the places 3 e, 3 e + 1 and 3 e + 2. ValueError if basis is not (n, r) or lift is not (n,), and the
    errors of thermoelastic_response.
"""
import math

import numpy as np
import pytest

import thermomech

# Any warning is an error in these tests.
pytestmark = pytest.mark.filterwarnings("error")

PI = math.pi


# The functions are imported inside these wrappers and not at the top: while twin/mechanics.py does not exist, or does
# not have the function yet, only the tests of that function fail, not the whole session.


def mean_of(*args, **kwargs):
    from twin.mechanics import element_mean
    return element_mean(*args, **kwargs)


def von_mises(*args, **kwargs):
    from twin.mechanics import von_mises_plane_strain
    return von_mises_plane_strain(*args, **kwargs)


def response(*args, **kwargs):
    from twin.mechanics import thermoelastic_response
    return thermoelastic_response(*args, **kwargs)


def modes(*args, **kwargs):
    from twin.mechanics import mechanical_modes
    return mechanical_modes(*args, **kwargs)


def reduced_heat(*args, **kwargs):
    from twin.galerkin import solve_reduced_heat
    return solve_reduced_heat(*args, **kwargs)


def rebuild(*args, **kwargs):
    from twin.galerkin import reconstruct
    return reconstruct(*args, **kwargs)


# Friendly numbers for hand calculations: E = 100, nu = 1/4, alpha = 1/100, T_ref = 20. Then lambda = mu = 40,
# D = [[120, 40, 0], [40, 120, 0], [0, 0, 40]], E alpha = 1 and a triangle that is heated by dT expands by
# (1 + nu) alpha dT = dT / 80 in x and y.
HAND = dict(E=100.0, nu=0.25, alpha=0.01, T_ref=20.0)


def relative_error(result, expected):
    """The largest difference, relative to the largest absolute value of the expected array."""
    return np.abs(np.asarray(result) - np.asarray(expected)).max() / np.abs(expected).max()


# ---------------------------------------------------------------- independent computations


def plane_strain_matrix(E, nu):
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return np.array([[lam + 2.0 * mu, lam, 0.0], [lam, lam + 2.0 * mu, 0.0], [0.0, 0.0, mu]])


def strain_displacement(mesh, e):
    """B (3 x 6) and the area of the triangle e: (du/dx, dv/dy, du/dy + dv/dx) = B (u0, v0, u1, v1, u2, v2)."""
    t = mesh.triangles[e]
    x, y = mesh.x[t], mesh.y[t]
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    area = 0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
    B = np.zeros((3, 6))
    B[0, 0::2] = b
    B[1, 1::2] = c
    B[2, 0::2] = c
    B[2, 1::2] = b
    return B / (2.0 * area), area


def dense_response(mesh, T, E, nu, alpha, T_ref, fixed_dofs, fixed_displacements):
    """u and sigma of the plate at the temperature T: K u = F with dense matrices, the fixed degrees of freedom
    eliminated, then sigma = D (B u - eps_th) in each triangle."""
    n, n_tri = mesh.num_nodes, mesh.num_triangles
    D = plane_strain_matrix(E, nu)
    K = np.zeros((2 * n, 2 * n))
    F = np.zeros(2 * n)
    elements = []
    for e in range(n_tri):
        B, area = strain_displacement(mesh, e)
        dofs = np.ravel([[2 * i, 2 * i + 1] for i in mesh.triangles[e]])
        eps_th = (1.0 + nu) * alpha * (np.asarray(T)[mesh.triangles[e]].mean() - T_ref) * np.array([1.0, 1.0, 0.0])
        K[np.ix_(dofs, dofs)] += area * B.T @ D @ B
        F[dofs] += area * B.T @ D @ eps_th
        elements.append((B, dofs, eps_th))
    fixed = np.asarray(fixed_dofs)
    values = np.asarray(fixed_displacements, dtype=float)
    is_free = np.ones(2 * n, dtype=bool)
    is_free[fixed] = False
    free = np.flatnonzero(is_free)
    u = np.zeros(2 * n)
    u[fixed] = values
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free] - K[np.ix_(free, fixed)] @ values)
    sigma = np.array([D @ (B @ u[dofs] - eps_th) for B, dofs, eps_th in elements])
    return u, sigma


def von_mises_by_the_deviator(sigma, rise, E, nu, alpha):
    """The von Mises stress from the 3 x 3 stress tensor: sqrt(3/2 s : s) with s = sigma - trace(sigma) / 3 I."""
    sigma = np.asarray(sigma, dtype=float)
    rise = np.asarray(rise, dtype=float)
    flat = sigma.reshape(-1, 3)
    out = np.empty(flat.shape[0])
    for i, (xx, yy, xy) in enumerate(flat):
        zz = nu * (xx + yy) - E * alpha * rise.reshape(-1)[i]
        tensor = np.array([[xx, xy, 0.0], [xy, yy, 0.0], [0.0, 0.0, zz]])
        deviator = tensor - np.trace(tensor) / 3.0 * np.eye(3)
        out[i] = math.sqrt(1.5 * np.sum(deviator * deviator))
    return out.reshape(sigma.shape[:-1])


def unit_square():
    """Four nodes (0, 0), (1, 0), (0, 1), (1, 1) and two triangles (0, 1, 3) and (0, 3, 2)."""
    return thermomech.rectangle_mesh(1.0, 1.0, 1, 1)


# u and v of the node 0 and v of the node 1: a plate that is free to expand
PINS_UNIT_SQUARE = dict(fixed_dofs=[0, 1, 3], fixed_displacements=[0.0, 0.0, 0.0])
# every displacement is zero
CLAMPED_UNIT_SQUARE = dict(fixed_dofs=list(range(8)), fixed_displacements=[0.0] * 8)


# ---------------------------------------------------------------- memory layouts


def every_second(values):
    wide = np.zeros(values.shape[:-1] + (2 * values.shape[-1],), dtype=values.dtype)
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
LAYOUT_IDS = [layout.__name__ for layout in LAYOUTS]


# ---------------------------------------------------------------- element_mean

TWO_TRIANGLES = [[0, 1, 2], [1, 3, 2]]  # the triangles (0, 1, 2) and (1, 3, 2) of four nodes


def test_element_mean_hand_example_two_triangles():
    # (3 + 6 + 9) / 3 = 6 and (6 + 12 + 9) / 3 = 9
    result = mean_of(TWO_TRIANGLES, [3.0, 6.0, 9.0, 12.0])
    assert result.shape == (2,)
    np.testing.assert_allclose(result, [6.0, 9.0], rtol=0, atol=1e-14)


def test_element_mean_hand_example_one_row_of_values_for_every_state():
    # the last axis is the nodes. Row 1: (0 + 0 + 0) / 3 = 0 and (0 + 3 + 0) / 3 = 1. Row 2: (1 - 1 + 1) / 3 = 1/3 and
    # (-1 - 1 + 1) / 3 = -1/3. The nodes go up to 3, more than the number of rows
    values = [[3.0, 6.0, 9.0, 12.0], [0.0, 0.0, 0.0, 3.0], [1.0, -1.0, 1.0, -1.0]]
    result = mean_of(TWO_TRIANGLES, values)
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result, [[6.0, 9.0], [0.0, 1.0], [1.0 / 3.0, -1.0 / 3.0]], rtol=0, atol=1e-14)


def test_element_mean_the_order_of_the_nodes_of_a_triangle_does_not_matter():
    values = np.array([0.5, -3.0, 7.0, 2.25, 11.0])
    base = np.array([[0, 2, 4], [1, 3, 4], [4, 0, 1]])
    expected = mean_of(base, values)
    for order in ([0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
        np.testing.assert_allclose(mean_of(base[:, order], values), expected, rtol=0, atol=1e-14)


def test_element_mean_every_entry_of_a_triangle_counts_once_even_when_a_node_is_repeated():
    # (a + a + b) / 3 and (c + c + c) / 3
    result = mean_of([[0, 0, 1], [2, 2, 2], [1, 3, 3]], [3.0, 6.0, 9.0, 12.0])
    np.testing.assert_allclose(result, [4.0, 9.0, 10.0], rtol=0, atol=1e-14)


@pytest.mark.parametrize("nx, ny", [(1, 1), (3, 2), (6, 4)])
def test_element_mean_of_a_linear_field_is_its_value_at_the_centroid(nx, ny):
    mesh = thermomech.rectangle_mesh(2.0, 1.5, nx, ny)
    x, y, tri = np.asarray(mesh.x), np.asarray(mesh.y), np.asarray(mesh.triangles)
    centroid_x = (x[tri[:, 0]] + x[tri[:, 1]] + x[tri[:, 2]]) / 3.0
    centroid_y = (y[tri[:, 0]] + y[tri[:, 1]] + y[tri[:, 2]]) / 3.0
    result = mean_of(mesh.triangles, 3.0 - 2.0 * x + 5.0 * y)
    assert result.shape == (mesh.num_triangles,)
    np.testing.assert_allclose(result, 3.0 - 2.0 * centroid_x + 5.0 * centroid_y, rtol=0, atol=1e-13)


def test_element_mean_of_a_constant_field_is_that_constant():
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 4, 3)
    result = mean_of(mesh.triangles, np.full(mesh.num_nodes, 7.5))
    np.testing.assert_allclose(result, 7.5, rtol=0, atol=1e-14)
    assert result.shape == (mesh.num_triangles,)


def test_element_mean_agrees_with_a_loop_over_the_triangles_for_every_state():
    mesh = thermomech.rectangle_mesh(2.0, 1.0, 5, 3)
    values = np.random.default_rng(3).normal(size=(4, mesh.num_nodes))
    expected = np.array([[row[t[0]] / 3.0 + row[t[1]] / 3.0 + row[t[2]] / 3.0 for t in mesh.triangles]
                         for row in values])
    result = mean_of(mesh.triangles, values)
    assert result.shape == (4, mesh.num_triangles)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-14)
    for row in range(4):  # a single state gives the same numbers as the same row of many
        np.testing.assert_allclose(mean_of(mesh.triangles, values[row]), result[row], rtol=0, atol=1e-14)


def test_element_mean_is_linear_in_the_values():
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 3, 3)
    rng = np.random.default_rng(5)
    v, w = rng.normal(size=(2, mesh.num_nodes))
    both = mean_of(mesh.triangles, 2.0 * v - 3.0 * w)
    np.testing.assert_allclose(both, 2.0 * mean_of(mesh.triangles, v) - 3.0 * mean_of(mesh.triangles, w), rtol=0,
                               atol=1e-13)


def test_element_mean_one_state_in_a_matrix_has_one_row():
    result = mean_of(TWO_TRIANGLES, [[3.0, 6.0, 9.0, 12.0]])
    assert result.shape == (1, 2)
    np.testing.assert_allclose(result, [[6.0, 9.0]], rtol=0, atol=1e-14)


def test_element_mean_result_is_a_new_float64_array_and_the_arguments_are_not_modified():
    triangles = np.array(TWO_TRIANGLES)
    values = np.array([3.0, 6.0, 9.0, 12.0])
    result = mean_of(triangles, values)
    assert isinstance(result, np.ndarray) and result.dtype == np.float64 and result.flags["WRITEABLE"]
    assert not np.shares_memory(result, values) and not np.shares_memory(result, triangles)
    result[:] = -1.0  # writing in it changes nothing else
    np.testing.assert_array_equal(triangles, TWO_TRIANGLES)
    np.testing.assert_array_equal(values, [3.0, 6.0, 9.0, 12.0])
    np.testing.assert_allclose(mean_of(triangles, values), [6.0, 9.0], rtol=0, atol=1e-14)


def container(data, make):
    """The nested list data as a list, as a tuple (of tuples) or as an array of the type make."""
    if make is list:
        return data
    if make is tuple:
        return tuple(tuple(row) if isinstance(row, list) else row for row in data)
    return np.array(data, dtype=make)


@pytest.mark.parametrize("make_triangles", [list, tuple, np.int8, np.int32, np.int64, np.uint8, np.uint64])
@pytest.mark.parametrize("make_values", [list, tuple, np.float64, np.float32, np.int64, np.int32])
def test_element_mean_accepts_triangles_and_values_of_any_container_and_type(make_triangles, make_values):
    triangles = container(TWO_TRIANGLES, make_triangles)
    values = container([3, 6, 9, 12], make_values)  # exact in every type
    result = mean_of(triangles, values)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [6.0, 9.0], rtol=0, atol=1e-14)


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_element_mean_arrays_of_any_memory_layout_give_the_same_answer(layout):
    mesh = thermomech.rectangle_mesh(2.0, 1.0, 4, 3)
    rng = np.random.default_rng(11)
    values = rng.normal(size=(3, mesh.num_nodes))
    triangles = np.asarray(mesh.triangles)
    expected = mean_of(triangles, values)
    np.testing.assert_allclose(mean_of(layout(triangles.astype(np.int64)), layout(values)), expected, rtol=0, atol=0)
    np.testing.assert_allclose(mean_of(triangles, layout(values[0])), expected[0], rtol=0, atol=0)


def test_element_mean_arguments_can_be_given_by_name():
    result = mean_of(triangles=TWO_TRIANGLES, values=[3.0, 6.0, 9.0, 12.0])
    np.testing.assert_allclose(result, [6.0, 9.0], rtol=0, atol=1e-14)


MEAN_ERRORS = {
    "triangles with two columns": dict(triangles=[[0, 1], [1, 2]], values=[1.0, 2.0, 3.0, 4.0]),
    "triangles with four columns": dict(triangles=[[0, 1, 2, 3]], values=[1.0, 2.0, 3.0, 4.0]),
    "triangles that are one row": dict(triangles=[0, 1, 2], values=[1.0, 2.0, 3.0, 4.0]),
    "triangles that are a number": dict(triangles=2, values=[1.0, 2.0, 3.0, 4.0]),
    "triangles in three dimensions": dict(triangles=[[[0, 1, 2]]], values=[1.0, 2.0, 3.0, 4.0]),
    "a node that is the number of nodes": dict(triangles=[[0, 1, 4]], values=[1.0, 2.0, 3.0, 4.0]),
    "a node far beyond the last": dict(triangles=[[0, 1, 400]], values=[1.0, 2.0, 3.0, 4.0]),
    "a negative node": dict(triangles=[[0, 1, -1]], values=[1.0, 2.0, 3.0, 4.0]),
    "a negative node in the first column": dict(triangles=[[-4, 1, 2]], values=[1.0, 2.0, 3.0, 4.0]),
    "values that are a number": dict(triangles=TWO_TRIANGLES, values=5.0),
    "values in three dimensions": dict(triangles=TWO_TRIANGLES, values=np.ones((2, 2, 4))),
    "a node that does not exist in a matrix of values with more rows than nodes":
        dict(triangles=[[0, 1, 4]], values=np.ones((5, 3))),
    "a node that does not exist in a matrix of values": dict(triangles=[[0, 1, 4]], values=np.ones((2, 4))),
}


@pytest.mark.parametrize("arguments", list(MEAN_ERRORS.values()), ids=list(MEAN_ERRORS))
def test_element_mean_arguments_that_do_not_fit_are_a_value_error(arguments):
    with pytest.raises(ValueError):
        mean_of(**arguments)


def test_element_mean_the_last_node_and_the_node_zero_are_valid():
    # the limits of the rule: 0 and the number of nodes minus 1 exist
    result = mean_of([[0, 3, 3]], [3.0, 6.0, 9.0, 12.0])
    np.testing.assert_allclose(result, [9.0], rtol=0, atol=1e-14)


# ---------------------------------------------------------------- von_mises_plane_strain

HAND_PLANE = dict(E=100.0, nu=0.25, alpha=0.01)
STEEL = dict(E=2.1e5, nu=0.3, alpha=1.2e-5)
MATERIALS = [HAND_PLANE, STEEL, dict(E=7.0e4, nu=0.33, alpha=2.3e-5), dict(E=1.0, nu=-0.5, alpha=3.0)]
MATERIAL_IDS = ["hand", "steel", "aluminium", "auxetic"]
NAN = float("nan")


@pytest.mark.parametrize("sigma, rise, expected", [
    ([8.0, 0.0, 0.0], 0.0, math.sqrt(52.0)),  # sigma_zz = 1/4 8 = 2: sqrt(1/2 (8^2 + 2^2 + 6^2)) = sqrt(52)
    ([0.0, 0.0, 5.0], 0.0, 5.0 * math.sqrt(3.0)),  # pure shear: sqrt(3) 5
    ([0.0, 0.0, 0.0], 10.0, 10.0),  # no stress in the plane, sigma_zz = -E alpha 10 = -10: sqrt(1/2 (10^2 + 10^2)) = 10
    ([6.0, 6.0, 0.0], 0.0, 3.0),  # sigma_zz = 1/4 12 = 3: sqrt(1/2 (0 + 3^2 + 3^2)) = 3
    ([6.0, -2.0, 3.0], 4.0, 10.0),  # sigma_zz = 1 - 4 = -3: sqrt(1/2 (8^2 + 1^2 + 9^2) + 3 3^2) = sqrt(73 + 27) = 10
    ([0.0, 0.0, 0.0], 0.0, 0.0),
    ([-8.0, 0.0, 0.0], 0.0, math.sqrt(52.0)),
], ids=["pull", "shear", "heated", "equibiaxial", "mixed", "nothing", "push"])
def test_von_mises_hand_examples(sigma, rise, expected):
    result = von_mises(sigma, rise, **HAND_PLANE)
    assert np.shape(result) == ()
    assert float(result) == pytest.approx(expected, rel=1e-13, abs=1e-13)


@pytest.mark.parametrize("material", MATERIALS, ids=MATERIAL_IDS)
def test_von_mises_a_hydrostatic_state_has_no_von_mises_stress(material):
    # the three normal stresses are equal when sigma_zz = nu (s + s) - E alpha rise = s, that is for
    # rise = (2 nu - 1) s / (E alpha)
    s = 40.0
    rise = (2.0 * material["nu"] - 1.0) * s / (material["E"] * material["alpha"])
    assert float(von_mises([s, s, 0.0], rise, **material)) == pytest.approx(0.0, abs=1e-6 * s)


@pytest.mark.parametrize("rise", [10.0, -3.5, 0.25])
@pytest.mark.parametrize("material", MATERIALS, ids=MATERIAL_IDS)
def test_von_mises_a_heated_plate_without_stress_in_the_plane_has_the_stress_E_alpha_rise(material, rise):
    # sigma_zz = -E alpha rise is the only stress: the von Mises stress is its size
    expected = material["E"] * material["alpha"] * abs(rise)
    assert float(von_mises([0.0, 0.0, 0.0], rise, **material)) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("sigma", [[50.0, 0.0, 0.0], [0.0, -50.0, 0.0], [7.0, 7.0, 0.0]], ids=["xx", "yy", "equi"])
@pytest.mark.parametrize("material", MATERIALS, ids=MATERIAL_IDS)
def test_von_mises_when_the_stress_out_of_the_plane_vanishes_it_is_the_textbook_formula(material, sigma):
    # the rise that makes sigma_zz = nu (xx + yy) - E alpha rise vanish: then sqrt(xx^2 - xx yy + yy^2 + 3 xy^2)
    xx, yy, xy = sigma
    rise = material["nu"] * (xx + yy) / (material["E"] * material["alpha"])
    expected = math.sqrt(xx**2 - xx * yy + yy**2 + 3.0 * xy**2)
    assert float(von_mises(sigma, rise, **material)) == pytest.approx(expected, rel=1e-12)


def random_states(shape, seed):
    rng = np.random.default_rng(seed)
    return 30.0 * rng.standard_normal(shape), 5.0 * rng.standard_normal(shape[:-1])


@pytest.mark.parametrize("shape", [(3,), (1, 3), (6, 3), (2, 5, 3), (2, 1, 4, 3)], ids=str)
@pytest.mark.parametrize("material", MATERIALS, ids=MATERIAL_IDS)
def test_von_mises_agrees_with_the_deviator_of_the_three_dimensional_stress(material, shape):
    sigma, rise = random_states(shape, 17)
    result = von_mises(sigma, rise, **material)
    assert np.shape(result) == shape[:-1]
    expected = von_mises_by_the_deviator(sigma, rise, **material)
    np.testing.assert_allclose(result, expected, rtol=1e-11, atol=0)


@pytest.mark.parametrize("angle", [0.3, 1.1, PI / 2.0, 2.4, -0.7])
def test_von_mises_does_not_depend_on_the_directions_of_the_axes(angle):
    sigma, rise = random_states((25, 3), 5)
    c, s = math.cos(angle), math.sin(angle)
    xx, yy, xy = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    turned = np.column_stack([c * c * xx + s * s * yy + 2.0 * c * s * xy,
                              s * s * xx + c * c * yy - 2.0 * c * s * xy,
                              (yy - xx) * c * s + (c * c - s * s) * xy])
    np.testing.assert_allclose(von_mises(turned, rise, **STEEL), von_mises(sigma, rise, **STEEL),
                               rtol=1e-12, atol=0)


def test_von_mises_does_not_change_with_the_sign_of_the_shear_and_of_the_whole_state():
    sigma, rise = random_states((25, 3), 6)
    expected = von_mises(sigma, rise, **STEEL)
    mirrored = sigma * np.array([1.0, 1.0, -1.0])
    np.testing.assert_allclose(von_mises(mirrored, rise, **STEEL), expected, rtol=1e-13, atol=0)
    np.testing.assert_allclose(von_mises(-sigma, -rise, **STEEL), expected, rtol=1e-13, atol=0)


@pytest.mark.parametrize("factor", [2.5, 1e-3, -0.4])
def test_von_mises_is_homogeneous_of_degree_one_in_the_stresses_and_the_rise(factor):
    sigma, rise = random_states((25, 3), 7)
    np.testing.assert_allclose(von_mises(factor * sigma, factor * rise, **STEEL),
                               abs(factor) * von_mises(sigma, rise, **STEEL), rtol=1e-12, atol=0)


def test_von_mises_depends_on_the_material_only_through_nu_and_the_product_E_alpha():
    sigma, rise = random_states((25, 3), 8)
    expected = von_mises(sigma, rise, E=100.0, nu=0.25, alpha=0.01)
    np.testing.assert_allclose(von_mises(sigma, rise, E=50.0, nu=0.25, alpha=0.02), expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(von_mises(sigma, rise, E=1.0, nu=0.25, alpha=1.0), expected, rtol=1e-12, atol=0)
    assert np.abs(von_mises(sigma, rise, E=100.0, nu=0.3, alpha=0.01) - expected).max() > 1e-3


def test_von_mises_agrees_with_every_state_of_the_cpp_solver(plate):
    # solve_thermomechanical_2d gives the stresses, the temperatures and the von Mises stress of every state: the same
    # von Mises stress follows from the stresses and the mean temperature of every triangle
    nodes, values, _, start = plate.side("left")
    for material, T_ref in ((HAND_PLANE, 20.0), (STEEL, 26.5)):
        full = plate.thermomechanical(start, nodes, values, 0.05, 3, 0.5, plate.constraints("pins"), material, T_ref)
        rise = np.asarray(full["T"])[:, np.asarray(plate.mesh.triangles)].mean(axis=2) - T_ref
        result = von_mises(full["sigma"], rise, **material)
        assert result.shape == full["von_mises"].shape
        assert relative_error(result, full["von_mises"]) < 1e-11


def test_von_mises_result_is_a_new_float64_array_and_the_arguments_are_not_modified():
    sigma = np.array([[8.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    rise = np.array([0.0, 0.0])
    result = von_mises(sigma, rise, **HAND_PLANE)
    assert isinstance(result, np.ndarray) and result.dtype == np.float64 and result.flags["WRITEABLE"]
    assert not np.shares_memory(result, sigma) and not np.shares_memory(result, rise)
    result[:] = -1.0
    np.testing.assert_array_equal(sigma, [[8.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    np.testing.assert_array_equal(rise, [0.0, 0.0])
    np.testing.assert_allclose(von_mises(sigma, rise, **HAND_PLANE), [math.sqrt(52.0), 5.0 * math.sqrt(3.0)],
                               rtol=1e-13, atol=0)


@pytest.mark.parametrize("make", [list, tuple, np.float32, np.int64])
def test_von_mises_accepts_lists_tuples_and_arrays_of_any_type(make):
    sigma = [[8, 0, 0], [0, 0, 5]]
    rise = [0, 0]
    if make is tuple:
        sigma, rise = tuple(tuple(row) for row in sigma), tuple(rise)
    elif make is not list:
        sigma, rise = np.array(sigma, dtype=make), np.array(rise, dtype=make)
    np.testing.assert_allclose(von_mises(sigma, rise, **HAND_PLANE), [math.sqrt(52.0), 5.0 * math.sqrt(3.0)],
                               rtol=1e-13, atol=0)


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_von_mises_arrays_of_any_memory_layout_give_the_same_answer(layout):
    sigma, rise = random_states((12, 3), 9)
    expected = von_mises(sigma, rise, **STEEL)
    np.testing.assert_allclose(von_mises(layout(sigma), layout(rise), **STEEL), expected, rtol=0, atol=0)


def test_von_mises_arguments_can_be_given_by_name():
    result = von_mises(sigma=[8.0, 0.0, 0.0], temperature_rise=0.0, E=100.0, nu=0.25, alpha=0.01)
    assert float(result) == pytest.approx(math.sqrt(52.0), rel=1e-13)


def a_good_von_mises(**change):
    arguments = dict(sigma=np.array([[8.0, 0.0, 0.0], [0.0, 0.0, 5.0]]), temperature_rise=np.zeros(2), **HAND_PLANE)
    arguments.update(change)
    return arguments


VON_MISES_ERRORS = {
    "E is zero": dict(E=0.0),
    "E is negative": dict(E=-100.0),
    "E is NaN": dict(E=NAN),
    "nu is one half": dict(nu=0.5),
    "nu is above one half": dict(nu=0.7),
    "nu is minus one": dict(nu=-1.0),
    "nu is below minus one": dict(nu=-1.5),
    "nu is NaN": dict(nu=NAN),
    "sigma has two components": dict(sigma=np.zeros((2, 2))),
    "sigma has four components": dict(sigma=np.zeros((2, 4))),
    "sigma is a number": dict(sigma=5.0, temperature_rise=0.0),
    "temperature_rise has an entry too many": dict(temperature_rise=np.zeros(3)),
    "temperature_rise has an entry too few": dict(temperature_rise=np.zeros(1)),
    "temperature_rise is a number for two stresses": dict(temperature_rise=0.0),
    "temperature_rise has an extra axis": dict(temperature_rise=np.zeros((2, 1))),
    "temperature_rise has the shape of sigma": dict(temperature_rise=np.zeros((2, 3))),
    "one stress and two rises": dict(sigma=[8.0, 0.0, 0.0], temperature_rise=[0.0, 0.0]),
}


@pytest.mark.parametrize("change", list(VON_MISES_ERRORS.values()), ids=list(VON_MISES_ERRORS))
def test_von_mises_arguments_that_do_not_fit_are_a_value_error(change):
    with pytest.raises(ValueError):
        von_mises(**a_good_von_mises(**change))


@pytest.mark.parametrize("nu", [-0.9999, -0.5, 0.0, 0.3, 0.4999])
def test_von_mises_accepts_every_poisson_ratio_between_minus_one_and_one_half(nu):
    result = von_mises(**a_good_von_mises(nu=nu))
    assert result.shape == (2,) and np.all(np.isfinite(result))


# ---------------------------------------------------------------- thermoelastic_response

SMALL = (2.0, 1.0, 4, 3)  # a rectangle 2 x 1 of 4 x 3 cells: 20 nodes, 24 triangles, 40 degrees of freedom
CASES = [(HAND_PLANE, 20.0), (STEEL, 26.5)]
CASE_IDS = ["hand", "steel"]
CONSTRAINTS = ["pins", "rollers", "stretch"]


def constraints_of(name, mesh, nx, ny):
    """Three ways to hold a rectangle of nx x ny cells (the node of the column i and the row j is j (nx + 1) + i):
    pins   the minimum: u and v of the corner (0, 0) and v of the corner (L, 0), all zero
    rollers   u = 0 on the left side and v = 0 on the bottom side (the symmetry planes of a quarter of a plate)
    stretch   the left side is clamped and the right side is pulled by 0.05 (1 + y), v free"""
    left = [j * (nx + 1) for j in range(ny + 1)]
    right = [j * (nx + 1) + nx for j in range(ny + 1)]
    if name == "pins":
        return dict(fixed_dofs=[0, 1, 2 * nx + 1], fixed_displacements=[0.0, 0.0, 0.0])
    if name == "rollers":
        dofs = [2 * node for node in left] + [2 * i + 1 for i in range(nx + 1)]
        return dict(fixed_dofs=dofs, fixed_displacements=[0.0] * len(dofs))
    dofs = [2 * node for node in left] + [2 * node + 1 for node in left] + [2 * node for node in right]
    pulls = [0.05 * (1.0 + mesh.y[node]) for node in right]
    return dict(fixed_dofs=dofs, fixed_displacements=[0.0] * (2 * len(left)) + pulls)


def small_case(name="pins"):
    mesh = thermomech.rectangle_mesh(*SMALL)
    return mesh, constraints_of(name, mesh, SMALL[2], SMALL[3])


def random_temperature(n, seed):
    return 20.0 + 10.0 * np.random.default_rng(seed).random(n)


def with_zero_displacements(constraints):
    return dict(fixed_dofs=constraints["fixed_dofs"], fixed_displacements=[0.0] * len(constraints["fixed_dofs"]))


def test_response_hand_example_a_free_plate_that_is_heated_expands_without_stress():
    # 10 above the reference: the strain (1 + nu) alpha 10 = 1/8 in both directions, u = (x, y) / 8, no stress
    u, sigma = response(unit_square(), np.full(4, 30.0), **HAND, **PINS_UNIT_SQUARE)
    np.testing.assert_allclose(u, [0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125], rtol=0, atol=1e-9)
    np.testing.assert_allclose(sigma, np.zeros((2, 3)), rtol=0, atol=1e-6)


def test_response_hand_example_a_clamped_plate_that_is_heated_is_compressed():
    # nothing moves, the thermal strain 1/8 is all elastic: sigma = -D (1/8, 1/8, 0) = (-20, -20, 0) in both triangles
    u, sigma = response(unit_square(), np.full(4, 30.0), **HAND, **CLAMPED_UNIT_SQUARE)
    np.testing.assert_allclose(u, np.zeros(8), rtol=0, atol=1e-12)
    np.testing.assert_allclose(sigma, [[-20.0, -20.0, 0.0], [-20.0, -20.0, 0.0]], rtol=0, atol=1e-9)


def test_response_hand_example_a_plate_that_is_stretched_at_the_reference_temperature():
    # the left side is held (u = 0), the right side moves by 0.1 (nodes 1 and 3), v = 0 on the bottom. With free
    # top side sigma_yy = 0 so the strain eps_y = -lambda / (lambda + 2 mu) 0.1 = -1/30 and sigma_xx = 120 0.1 - 40 / 30
    u, sigma = response(unit_square(), np.full(4, 20.0), fixed_dofs=[0, 1, 2, 3, 4, 6],
                        fixed_displacements=[0.0, 0.0, 0.1, 0.0, 0.0, 0.1], **HAND)
    np.testing.assert_allclose(u, [0.0, 0.0, 0.1, 0.0, 0.0, -1.0 / 30.0, 0.1, -1.0 / 30.0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(sigma, [[32.0 / 3.0, 0.0, 0.0]] * 2, rtol=0, atol=1e-6)


def test_response_hand_example_the_prescribed_displacement_of_a_dof_is_the_one_in_the_same_place():
    # the same constraints in another order, the values travel with their degrees of freedom
    dofs, values = [0, 1, 2, 3, 4, 6], [0.0, 0.0, 0.1, 0.0, 0.0, 0.1]
    u, sigma = response(unit_square(), np.full(4, 20.0), fixed_dofs=dofs, fixed_displacements=values, **HAND)
    for order in ([5, 2, 0, 4, 1, 3], [3, 4, 5, 0, 1, 2]):
        u_again, sigma_again = response(unit_square(), np.full(4, 20.0), fixed_dofs=[dofs[q] for q in order],
                                        fixed_displacements=[values[q] for q in order], **HAND)
        np.testing.assert_allclose(u_again, u, rtol=0, atol=1e-12)
        np.testing.assert_allclose(sigma_again, sigma, rtol=0, atol=1e-9)


def test_response_hand_example_a_translation_is_added_to_the_expansion():
    # u0 = 0.5, v0 = -0.25 and v1 = -0.25 move the whole plate by (0.5, -0.25) while it expands by 1/8
    u, sigma = response(unit_square(), np.full(4, 30.0), fixed_dofs=[0, 1, 3], fixed_displacements=[0.5, -0.25, -0.25],
                        **HAND)
    np.testing.assert_allclose(u, [0.5, -0.25, 0.625, -0.25, 0.5, -0.125, 0.625, -0.125], rtol=0, atol=1e-9)
    np.testing.assert_allclose(sigma, np.zeros((2, 3)), rtol=0, atol=1e-6)


@pytest.mark.parametrize("name", ["pins", "rollers"])
def test_response_at_the_reference_temperature_nothing_moves_and_there_is_no_stress(name):
    mesh, constraints = small_case(name)
    u, sigma = response(mesh, np.full(mesh.num_nodes, 26.5), **STEEL, T_ref=26.5, **constraints)
    np.testing.assert_allclose(u, 0.0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(sigma, 0.0, rtol=0, atol=1e-9)
    assert u.shape == (2 * mesh.num_nodes,) and sigma.shape == (mesh.num_triangles, 3)


def test_response_on_a_larger_mesh_a_free_plate_with_a_uniform_rise_expands_by_the_same_amount_everywhere():
    mesh, constraints = small_case("pins")
    e = (1.0 + STEEL["nu"]) * STEEL["alpha"] * 7.0  # a rise of 7
    u, sigma = response(mesh, np.full(mesh.num_nodes, 27.0), **STEEL, T_ref=20.0, **constraints)
    np.testing.assert_allclose(u, np.column_stack([e * mesh.x, e * mesh.y]).ravel(), rtol=0, atol=1e-9 * e)
    assert np.abs(sigma).max() < 1e-7 * STEEL["E"] * e


def test_response_on_a_larger_mesh_a_clamped_plate_with_a_uniform_rise_has_the_stress_E_alpha_rise_over_1_minus_2_nu():
    mesh = thermomech.rectangle_mesh(*SMALL)
    clamped = [dof for node in range(mesh.num_nodes) for dof in (2 * node, 2 * node + 1)]
    u, sigma = response(mesh, np.full(mesh.num_nodes, 27.0), **STEEL, T_ref=20.0, fixed_dofs=clamped,
                        fixed_displacements=[0.0] * len(clamped))
    expected = -STEEL["E"] * STEEL["alpha"] * 7.0 / (1.0 - 2.0 * STEEL["nu"])
    np.testing.assert_allclose(u, 0.0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(sigma[:, :2], expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(sigma[:, 2], 0.0, rtol=0, atol=1e-9)


@pytest.mark.parametrize("name", CONSTRAINTS)
@pytest.mark.parametrize("material, T_ref", CASES, ids=CASE_IDS)
def test_response_agrees_with_an_independent_computation_with_dense_matrices(material, T_ref, name):
    mesh, constraints = small_case(name)
    T = random_temperature(mesh.num_nodes, 4)
    u, sigma = response(mesh, T, **material, T_ref=T_ref, **constraints)
    u_expected, sigma_expected = dense_response(mesh, T, T_ref=T_ref, **material, **constraints)
    assert np.abs(u_expected).max() > 0.0 and np.abs(sigma_expected).max() > 0.0
    assert relative_error(u, u_expected) < 1e-8
    assert relative_error(sigma, sigma_expected) < 1e-7


@pytest.mark.parametrize("name", CONSTRAINTS)
@pytest.mark.parametrize("material, T_ref", CASES, ids=CASE_IDS)
def test_response_is_the_mechanics_of_every_state_of_the_cpp_solver(plate, material, T_ref, name):
    # the C++ solver computes the mechanics of the temperature of every state it saves: the same function of the
    # temperature, so the same numbers
    nodes, values, _, start = plate.side("left")
    constraints = plate.constraints(name)
    full = plate.thermomechanical(start, nodes, values, 0.05, 3, 0.5, constraints, material, T_ref)
    for state in range(4):
        u, sigma = response(plate.mesh, full["T"][state], **material, T_ref=T_ref, **constraints)
        assert relative_error(u, full["u"][state]) < 1e-9
        assert relative_error(sigma, full["sigma"][state]) < 1e-8


@pytest.mark.parametrize("name", ["pins", "rollers"])
def test_response_is_linear_in_the_temperature_rise_when_the_displacements_are_zero(name):
    mesh, constraints = small_case(name)
    n = mesh.num_nodes
    d1, d2 = np.random.default_rng(1).normal(size=(2, n))

    def of(rise):
        return response(mesh, 20.0 + rise, **STEEL, T_ref=20.0, **constraints)

    (u1, s1), (u2, s2), (u12, s12) = of(d1), of(d2), of(2.0 * d1 - 3.0 * d2)
    assert relative_error(u12, 2.0 * u1 - 3.0 * u2) < 1e-8
    assert relative_error(s12, 2.0 * s1 - 3.0 * s2) < 1e-7


def test_response_prescribed_displacements_add_the_response_of_the_reference_temperature():
    # u(T) = c + L (T - T_ref): the part c is what the displacements do at the uniform reference temperature
    mesh, constraints = small_case("stretch")
    T = random_temperature(mesh.num_nodes, 8)
    zero = with_zero_displacements(constraints)
    u, sigma = response(mesh, T, **STEEL, T_ref=20.0, **constraints)
    u_heat, sigma_heat = response(mesh, T, **STEEL, T_ref=20.0, **zero)
    u_pull, sigma_pull = response(mesh, np.full(mesh.num_nodes, 20.0), **STEEL, T_ref=20.0, **constraints)
    assert np.abs(u_pull).max() > 1e-3 and np.abs(u_heat).max() > 1e-6
    assert relative_error(u, u_heat + u_pull) < 1e-8
    assert relative_error(sigma, sigma_heat + sigma_pull) < 1e-7


@pytest.mark.parametrize("name", CONSTRAINTS)
def test_response_only_the_difference_between_the_temperature_and_the_reference_matters(name):
    mesh, constraints = small_case(name)
    T = random_temperature(mesh.num_nodes, 9)
    u, sigma = response(mesh, T, **STEEL, T_ref=20.0, **constraints)
    u_shifted, sigma_shifted = response(mesh, T + 100.0, **STEEL, T_ref=120.0, **constraints)
    assert relative_error(u_shifted, u) < 1e-8
    assert relative_error(sigma_shifted, sigma) < 1e-7


@pytest.mark.parametrize("name", ["pins", "rollers"])
def test_response_scales_with_the_material_the_displacements_with_alpha_and_the_stresses_with_E_alpha(name):
    mesh, constraints = small_case(name)
    T = random_temperature(mesh.num_nodes, 10)
    u, sigma = response(mesh, T, E=100.0, nu=0.25, alpha=0.01, T_ref=20.0, **constraints)
    for E, alpha in ((200.0, 0.01), (100.0, 0.03), (200.0, 0.03)):
        u_scaled, sigma_scaled = response(mesh, T, E=E, nu=0.25, alpha=alpha, T_ref=20.0, **constraints)
        assert relative_error(u_scaled, (alpha / 0.01) * u) < 1e-8
        assert relative_error(sigma_scaled, (E / 100.0) * (alpha / 0.01) * sigma) < 1e-7


def test_response_depends_on_the_poisson_ratio():
    mesh, constraints = small_case("rollers")
    T = random_temperature(mesh.num_nodes, 11)
    u, _ = response(mesh, T, E=100.0, nu=0.25, alpha=0.01, T_ref=20.0, **constraints)
    u_other, _ = response(mesh, T, E=100.0, nu=0.35, alpha=0.01, T_ref=20.0, **constraints)
    assert relative_error(u_other, u) > 1e-2


def test_response_is_deterministic():
    mesh, constraints = small_case("stretch")
    T = random_temperature(mesh.num_nodes, 12)
    first = response(mesh, T, **STEEL, T_ref=20.0, **constraints)
    second = response(mesh, T, **STEEL, T_ref=20.0, **constraints)
    for one, other in zip(first, second):
        np.testing.assert_array_equal(one, other)


def test_response_result_is_two_new_float64_arrays_and_the_arguments_are_not_modified():
    mesh = unit_square()
    T = np.array([30.0, 31.0, 29.0, 30.0])
    dofs = np.array([0, 1, 3])
    displacements = np.array([0.0, 0.0, 0.0])
    result = response(mesh, T, **HAND, fixed_dofs=dofs, fixed_displacements=displacements)
    assert len(result) == 2
    u, sigma = result
    for array, shape in ((u, (8,)), (sigma, (2, 3))):
        assert isinstance(array, np.ndarray) and array.dtype == np.float64 and array.shape == shape
        assert array.flags["WRITEABLE"] and not np.shares_memory(array, T) and not np.shares_memory(array, dofs)
    assert not np.shares_memory(u, sigma)
    expected_u, expected_sigma = u.copy(), sigma.copy()
    u[:] = -1.0
    sigma[:] = -1.0
    np.testing.assert_array_equal(T, [30.0, 31.0, 29.0, 30.0])
    np.testing.assert_array_equal(dofs, [0, 1, 3])
    np.testing.assert_array_equal(displacements, [0.0, 0.0, 0.0])
    again_u, again_sigma = response(mesh, T, **HAND, fixed_dofs=dofs, fixed_displacements=displacements)
    np.testing.assert_array_equal(again_u, expected_u)
    np.testing.assert_array_equal(again_sigma, expected_sigma)


def test_response_lists_tuples_and_other_number_types_give_the_same_answer():
    mesh = unit_square()
    base = dict(temperature=[30.0, 31.0, 29.0, 28.0], fixed_dofs=[0, 1, 3], fixed_displacements=[0.5, -0.25, 0.75])
    reference = response(mesh, **base, **HAND)
    variants = [dict(temperature=tuple(base["temperature"])), dict(temperature=np.array(base["temperature"])),
                dict(temperature=np.array(base["temperature"], dtype=np.float32)),
                dict(temperature=np.array([30, 31, 29, 28])),
                dict(fixed_dofs=(0, 1, 3)), dict(fixed_dofs=np.array([0, 1, 3], dtype=np.int32)),
                dict(fixed_dofs=np.array([0, 1, 3], dtype=np.uint8)),
                dict(fixed_dofs=np.array([0, 1, 3], dtype=np.uint64)),
                dict(fixed_displacements=(0.5, -0.25, 0.75)),
                dict(fixed_displacements=np.array([0.5, -0.25, 0.75], dtype=np.float32))]
    for change in variants:
        result = response(mesh, **dict(base, **change), **HAND)
        for got, want in zip(result, reference):
            np.testing.assert_array_equal(got, want, err_msg=str(sorted(change)))


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_response_arrays_of_any_memory_layout_give_the_same_answer(layout):
    mesh, constraints = small_case("stretch")
    T = random_temperature(mesh.num_nodes, 13)
    reference = response(mesh, T, **STEEL, T_ref=20.0, **constraints)
    dofs = layout(np.array(constraints["fixed_dofs"], dtype=np.int64))
    displacements = layout(np.array(constraints["fixed_displacements"]))
    for change in (dict(temperature=layout(T)), dict(fixed_dofs=dofs, fixed_displacements=displacements)):
        result = response(mesh, **dict(dict(temperature=T, **constraints), **change), **STEEL, T_ref=20.0)
        for got, want in zip(result, reference):
            np.testing.assert_array_equal(got, want)


def test_response_arguments_can_be_given_by_name():
    u, sigma = response(mesh=unit_square(), temperature=np.full(4, 30.0), E=100.0, nu=0.25, alpha=0.01, T_ref=20.0,
                        fixed_dofs=[0, 1, 3], fixed_displacements=[0.0, 0.0, 0.0])
    np.testing.assert_allclose(u, [0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125], rtol=0, atol=1e-9)


def a_good_response(**change):
    arguments = dict(mesh=unit_square(), temperature=np.full(4, 30.0), **HAND, **PINS_UNIT_SQUARE)
    arguments.update(change)
    return arguments


RESPONSE_VALUE_ERRORS = {
    "temperature has an entry too many": dict(temperature=np.zeros(5)),
    "temperature has an entry too few": dict(temperature=np.zeros(3)),
    "temperature is a number": dict(temperature=30.0),
    "temperature is a column": dict(temperature=np.zeros((4, 1))),
    "temperature has a row per state": dict(temperature=np.zeros((2, 4))),
    "temperature has a displacement per node": dict(temperature=np.zeros(8)),
    "E is zero": dict(E=0.0),
    "E is negative": dict(E=-100.0),
    "E is NaN": dict(E=NAN),
    "nu is one half": dict(nu=0.5),
    "nu is above one half": dict(nu=0.7),
    "nu is minus one": dict(nu=-1.0),
    "nu is NaN": dict(nu=NAN),
    "only two displacements are fixed": dict(fixed_dofs=[0, 1], fixed_displacements=[0.0, 0.0]),
    "one fixed displacement too few": dict(fixed_displacements=[0.0, 0.0]),
    "one fixed displacement too many": dict(fixed_displacements=[0.0, 0.0, 0.0, 0.0]),
    "a degree of freedom outside the plate": dict(fixed_dofs=[0, 1, 8]),
    "a degree of freedom twice": dict(fixed_dofs=[0, 1, 1]),
}


@pytest.mark.parametrize("change", list(RESPONSE_VALUE_ERRORS.values()), ids=list(RESPONSE_VALUE_ERRORS))
def test_response_arguments_that_do_not_fit_are_a_value_error(change):
    with pytest.raises(ValueError):
        response(**a_good_response(**change))


INF = float("inf")


@pytest.mark.parametrize("name, bad", [("temperature", NAN), ("temperature", INF), ("alpha", NAN), ("T_ref", NAN),
                                       ("fixed_displacements", NAN), ("fixed_displacements", INF)],
                         ids=["temperature NaN", "temperature inf", "alpha NaN", "T_ref NaN", "displacement NaN",
                              "displacement inf"])
def test_response_a_nan_or_an_infinity_is_never_returned_silently(name, bad):
    arguments = a_good_response()
    if isinstance(arguments[name], float):
        arguments[name] = bad
    else:
        value = np.array(arguments[name], dtype=float)
        value[-1] = bad
        arguments[name] = value
    with pytest.raises((ValueError, RuntimeError)):
        response(**arguments)


RESPONSE_TYPE_ERRORS = {
    "E is a string": dict(E="steel"),
    "E is None": dict(E=None),
    "alpha is None": dict(alpha=None),
    "T_ref is None": dict(T_ref=None),
    "a degree of freedom that is not an integer": dict(fixed_dofs=[0.5, 1, 3]),
    "fixed_dofs is None": dict(fixed_dofs=None),
    "fixed_displacements is None": dict(fixed_displacements=None),
}


@pytest.mark.parametrize("change", list(RESPONSE_TYPE_ERRORS.values()), ids=list(RESPONSE_TYPE_ERRORS))
def test_response_arguments_of_the_wrong_type_are_a_type_error(change):
    with pytest.raises(TypeError):
        response(**a_good_response(**change))


def test_response_a_negative_degree_of_freedom_is_refused():
    with pytest.raises((TypeError, ValueError)):
        response(**a_good_response(fixed_dofs=[0, 1, -1]))


# ---------------------------------------------------------------- mechanical_modes


def random_basis(n, r, seed):
    return np.random.default_rng(seed).normal(size=(n, r))


def test_modes_hand_example_a_pinned_plate_and_a_constant_mode():
    # the lifting 0 is 20 below the reference: the strain is -(1 + nu) alpha 20 = -1/4 and u = -(x, y) / 4. The mode
    # (a rise of 1) adds the strain 1/80: u = (x, y) / 80. A free plate has no stress
    xy = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    u_lift, u_modes, s_lift, s_modes = modes(unit_square(), np.ones((4, 1)), np.zeros(4), **HAND, **PINS_UNIT_SQUARE)
    np.testing.assert_allclose(u_lift, -0.25 * xy, rtol=0, atol=1e-9)
    np.testing.assert_allclose(u_modes, (xy / 80.0)[:, None], rtol=0, atol=1e-9)
    np.testing.assert_allclose(s_lift, np.zeros(6), rtol=0, atol=1e-6)
    np.testing.assert_allclose(s_modes, np.zeros((6, 1)), rtol=0, atol=1e-6)


def test_modes_hand_example_a_clamped_plate_and_a_constant_mode():
    # nothing moves and the stress is minus D times the thermal strain, (-160 eps, -160 eps, 0) in both triangles: the
    # lifting (eps = -1/4) gives (40, 40, 0) and the mode (eps = 1/80) gives (-2, -2, 0)
    u_lift, u_modes, s_lift, s_modes = modes(unit_square(), np.ones((4, 1)), np.zeros(4), **HAND,
                                             **CLAMPED_UNIT_SQUARE)
    np.testing.assert_allclose(u_lift, np.zeros(8), rtol=0, atol=1e-12)
    np.testing.assert_allclose(u_modes, np.zeros((8, 1)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(s_lift, [40.0, 40.0, 0.0, 40.0, 40.0, 0.0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(s_modes, [[-2.0], [-2.0], [0.0], [-2.0], [-2.0], [0.0]], rtol=0, atol=1e-9)


def test_modes_hand_example_two_modes_with_different_means_in_the_two_triangles():
    # the triangles are (0, 1, 3) and (0, 3, 2). The second mode (0, 2, 0, 2) has the mean 4/3 in the first and 2/3 in
    # the second: the stress is -2 times the mean in the two normal components of each triangle
    basis = np.array([[1.0, 0.0], [1.0, 2.0], [1.0, 0.0], [1.0, 2.0]])
    u_lift, u_modes, s_lift, s_modes = modes(unit_square(), basis, np.zeros(4), **HAND, **CLAMPED_UNIT_SQUARE)
    assert u_modes.shape == (8, 2) and s_modes.shape == (6, 2)
    np.testing.assert_allclose(s_modes[:, 0], [-2.0, -2.0, 0.0, -2.0, -2.0, 0.0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(s_modes[:, 1], [-8.0 / 3.0, -8.0 / 3.0, 0.0, -4.0 / 3.0, -4.0 / 3.0, 0.0], rtol=0,
                               atol=1e-9)


@pytest.mark.parametrize("name", CONSTRAINTS)
def test_modes_the_held_degrees_of_freedom_keep_their_prescribed_value_in_the_lifting_and_never_move_in_the_modes(name):
    mesh, constraints = small_case(name)
    n = mesh.num_nodes
    lift, basis = random_temperature(n, 30), random_basis(n, 3, 31)
    u_lift, u_modes, _, _ = modes(mesh, basis, lift, **STEEL, T_ref=20.0, **constraints)
    dofs = constraints["fixed_dofs"]
    np.testing.assert_allclose(u_lift[dofs], constraints["fixed_displacements"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(u_modes[dofs], 0.0, rtol=0, atol=1e-14)
    free = np.setdiff1d(np.arange(2 * n), dofs)
    assert np.abs(u_modes[free]).max() > 1e-6  # the other degrees of freedom do move


@pytest.mark.parametrize("name", CONSTRAINTS)
@pytest.mark.parametrize("material, T_ref", CASES, ids=CASE_IDS)
def test_modes_are_the_differences_between_the_responses_to_the_modes_and_to_the_lifting(material, T_ref, name):
    mesh, constraints = small_case(name)
    n = mesh.num_nodes
    lift, basis = random_temperature(n, 1), random_basis(n, 3, 2)
    u_lift, u_modes, s_lift, s_modes = modes(mesh, basis, lift, **material, T_ref=T_ref, **constraints)
    u0, s0 = response(mesh, lift, **material, T_ref=T_ref, **constraints)
    assert relative_error(u_lift, u0) < 1e-9
    assert relative_error(s_lift, s0.ravel()) < 1e-9
    for i in range(3):
        u_i, s_i = response(mesh, lift + basis[:, i], **material, T_ref=T_ref, **constraints)
        assert relative_error(u_modes[:, i], u_i - u0) < 1e-7
        assert relative_error(s_modes[:, i], (s_i - s0).ravel()) < 1e-6


@pytest.mark.parametrize("name", CONSTRAINTS)
@pytest.mark.parametrize("material, T_ref", CASES, ids=CASE_IDS)
def test_modes_give_the_response_to_any_temperature_of_the_form_lifting_plus_combination(material, T_ref, name):
    # this is what the reduced model uses: u(T_g + B a) = u_lift + U a and sigma(T_g + B a) = sigma_lift + S a
    mesh, constraints = small_case(name)
    n = mesh.num_nodes
    lift, basis = random_temperature(n, 3), random_basis(n, 4, 5)
    u_lift, u_modes, s_lift, s_modes = modes(mesh, basis, lift, **material, T_ref=T_ref, **constraints)
    for seed in range(3):
        a = np.random.default_rng(seed).normal(size=4)
        u, sigma = response(mesh, lift + basis @ a, **material, T_ref=T_ref, **constraints)
        assert relative_error(u_lift + u_modes @ a, u) < 1e-7
        assert relative_error(s_lift + s_modes @ a, sigma.ravel()) < 1e-6


@pytest.mark.parametrize("name", ["pins", "stretch"])
def test_modes_do_not_depend_on_the_reference_temperature_nor_on_the_prescribed_displacements(name):
    # the part of the response that does not depend on the temperature cancels in the differences
    mesh, constraints = small_case(name)
    n = mesh.num_nodes
    lift, basis = random_temperature(n, 8), random_basis(n, 3, 9)
    reference = modes(mesh, basis, lift, **STEEL, T_ref=20.0, **constraints)
    others = [modes(mesh, basis, lift, **STEEL, T_ref=85.0, **constraints),
              modes(mesh, basis, lift, **STEEL, T_ref=20.0, **with_zero_displacements(constraints))]
    for other in others:
        assert relative_error(other[1], reference[1]) < 1e-7
        assert relative_error(other[3], reference[3]) < 1e-6
    assert relative_error(others[0][0], reference[0]) > 1e-2  # the lifting alone does depend on the reference


def test_modes_do_not_depend_on_the_lifting():
    mesh, constraints = small_case("rollers")
    n = mesh.num_nodes
    basis = random_basis(n, 3, 10)
    first = modes(mesh, basis, np.zeros(n), **STEEL, T_ref=20.0, **constraints)
    second = modes(mesh, basis, random_temperature(n, 11), **STEEL, T_ref=20.0, **constraints)
    assert relative_error(second[1], first[1]) < 1e-7
    assert relative_error(second[3], first[3]) < 1e-6


@pytest.mark.parametrize("name", ["pins", "stretch"])
def test_modes_are_linear_in_the_columns_of_the_basis(name):
    mesh, constraints = small_case(name)
    n = mesh.num_nodes
    lift = random_temperature(n, 12)
    b1, b2 = random_basis(n, 2, 13).T
    basis = np.column_stack([b1, b2, b1 + 2.0 * b2, -3.5 * b1])
    _, u_modes, _, s_modes = modes(mesh, basis, lift, **STEEL, T_ref=20.0, **constraints)
    for result, tolerance in ((u_modes, 1e-7), (s_modes, 1e-6)):
        assert relative_error(result[:, 2], result[:, 0] + 2.0 * result[:, 1]) < tolerance
        assert relative_error(result[:, 3], -3.5 * result[:, 0]) < tolerance


def test_modes_every_mode_is_computed_on_its_own_and_in_the_order_of_the_basis():
    mesh, constraints = small_case("rollers")
    n = mesh.num_nodes
    lift, basis = random_temperature(n, 14), random_basis(n, 4, 15)
    together = modes(mesh, basis, lift, **STEEL, T_ref=20.0, **constraints)
    backwards_ = modes(mesh, basis[:, ::-1], lift, **STEEL, T_ref=20.0, **constraints)
    alone = modes(mesh, basis[:, [2]], lift, **STEEL, T_ref=20.0, **constraints)
    assert alone[1].shape == (2 * n, 1) and alone[3].shape == (3 * mesh.num_triangles, 1)
    for index, tolerance in ((1, 1e-9), (3, 1e-8)):
        assert relative_error(alone[index][:, 0], together[index][:, 2]) < tolerance
        assert relative_error(backwards_[index], together[index][:, ::-1]) < tolerance
    for index, tolerance in ((0, 1e-9), (2, 1e-8)):
        assert relative_error(backwards_[index], together[index]) < tolerance


def test_modes_a_mode_that_is_zero_has_no_response():
    mesh, constraints = small_case("pins")
    n = mesh.num_nodes
    basis = random_basis(n, 3, 16)
    basis[:, 1] = 0.0
    u_lift, u_modes, s_lift, s_modes = modes(mesh, basis, random_temperature(n, 17), **STEEL, T_ref=20.0, **constraints)
    assert np.abs(u_modes[:, 1]).max() < 1e-12 * np.abs(u_lift).max()
    assert np.abs(s_modes[:, 1]).max() < 1e-12 * np.abs(s_lift).max()
    assert np.abs(u_modes[:, 0]).max() > 1e-3 * np.abs(u_lift).max()  # the other modes do respond


@pytest.mark.parametrize("name", ["pins", "rollers"])
def test_modes_scale_with_the_material_the_displacements_with_alpha_and_the_stresses_with_E_alpha(name):
    mesh, constraints = small_case(name)
    n = mesh.num_nodes
    lift, basis = random_temperature(n, 18), random_basis(n, 3, 19)
    base = modes(mesh, basis, lift, E=100.0, nu=0.25, alpha=0.01, T_ref=20.0, **constraints)
    scaled = modes(mesh, basis, lift, E=200.0, nu=0.25, alpha=0.03, T_ref=20.0, **constraints)
    for got, want, factor in zip(scaled, base, (3.0, 3.0, 6.0, 6.0)):
        assert relative_error(got, factor * want) < 1e-6


def test_modes_returns_four_new_float64_arrays_with_these_shapes():
    mesh = thermomech.rectangle_mesh(2.0, 1.0, 5, 2)  # 18 nodes, 20 triangles
    n, n_tri = mesh.num_nodes, mesh.num_triangles
    basis, lift = random_basis(n, 3, 20), random_temperature(n, 21)
    result = modes(mesh, basis, lift, **STEEL, T_ref=20.0, **constraints_of("pins", mesh, 5, 2))
    assert len(result) == 4
    for array, shape in zip(result, [(2 * n,), (2 * n, 3), (3 * n_tri,), (3 * n_tri, 3)]):
        assert isinstance(array, np.ndarray) and array.dtype == np.float64 and array.shape == shape
        assert array.flags["WRITEABLE"] and not np.shares_memory(array, basis) and not np.shares_memory(array, lift)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.shares_memory(result[i], result[j])


def test_modes_with_one_mode_every_array_keeps_its_two_dimensions():
    mesh, constraints = small_case("pins")
    n = mesh.num_nodes
    result = modes(mesh, random_basis(n, 1, 22), random_temperature(n, 23), **STEEL, T_ref=20.0, **constraints)
    assert [array.shape for array in result] == [(2 * n,), (2 * n, 1), (3 * mesh.num_triangles,),
                                                 (3 * mesh.num_triangles, 1)]


def test_modes_the_arguments_are_not_modified_and_the_results_are_new_arrays():
    mesh, constraints = small_case("stretch")
    n = mesh.num_nodes
    basis, lift = random_basis(n, 3, 24), random_temperature(n, 25)
    basis_copy, lift_copy = basis.copy(), lift.copy()
    first = modes(mesh, basis, lift, **STEEL, T_ref=20.0, **constraints)
    np.testing.assert_array_equal(basis, basis_copy)
    np.testing.assert_array_equal(lift, lift_copy)
    expected = [array.copy() for array in first]
    for array in first:
        array[:] = -1.0
    second = modes(mesh, basis, lift, **STEEL, T_ref=20.0, **constraints)
    for got, want in zip(second, expected):
        np.testing.assert_array_equal(got, want)


def test_modes_accept_lists_integers_and_numbers_of_other_types():
    mesh = unit_square()
    reference = modes(mesh, np.array([[1.0, 0.0], [1.0, 2.0], [1.0, 0.0], [1.0, 2.0]]), np.zeros(4), **HAND,
                      **CLAMPED_UNIT_SQUARE)
    variants = [dict(basis=[[1, 0], [1, 2], [1, 0], [1, 2]]), dict(lift=[0, 0, 0, 0]), dict(lift=(0.0, 0.0, 0.0, 0.0)),
                dict(basis=np.array([[1, 0], [1, 2], [1, 0], [1, 2]], dtype=np.int32)),
                dict(basis=np.array([[1, 0], [1, 2], [1, 0], [1, 2]], dtype=np.float32))]
    base = dict(basis=[[1.0, 0.0], [1.0, 2.0], [1.0, 0.0], [1.0, 2.0]], lift=np.zeros(4))
    for change in variants:
        result = modes(mesh, **dict(base, **change), **HAND, **CLAMPED_UNIT_SQUARE)
        for got, want in zip(result, reference):
            np.testing.assert_array_equal(got, want, err_msg=str(sorted(change)))


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_modes_arrays_of_any_memory_layout_give_the_same_answer(layout):
    mesh, constraints = small_case("pins")
    n = mesh.num_nodes
    basis, lift = random_basis(n, 3, 26), random_temperature(n, 27)
    reference = modes(mesh, basis, lift, **STEEL, T_ref=20.0, **constraints)
    for change in (dict(basis=layout(basis)), dict(lift=layout(lift)), dict(basis=layout(basis), lift=layout(lift))):
        result = modes(mesh, **dict(dict(basis=basis, lift=lift), **change), **STEEL, T_ref=20.0, **constraints)
        for got, want in zip(result, reference):
            np.testing.assert_array_equal(got, want)


def test_modes_arguments_can_be_given_by_name():
    result = modes(mesh=unit_square(), basis=np.ones((4, 1)), lift=np.zeros(4), E=100.0, nu=0.25, alpha=0.01,
                   T_ref=20.0, fixed_dofs=list(range(8)), fixed_displacements=[0.0] * 8)
    np.testing.assert_allclose(result[2], [40.0, 40.0, 0.0, 40.0, 40.0, 0.0], rtol=0, atol=1e-9)


def a_good_modes(**change):
    arguments = dict(mesh=unit_square(), basis=np.ones((4, 2)), lift=np.zeros(4), **HAND, **PINS_UNIT_SQUARE)
    arguments.update(change)
    return arguments


MODES_VALUE_ERRORS = {
    "basis is one-dimensional": dict(basis=np.ones(4)),
    "basis is a number": dict(basis=1.0),
    "basis is three-dimensional": dict(basis=np.ones((4, 2, 1))),
    "basis has a row too many": dict(basis=np.ones((5, 2))),
    "basis has a row too few": dict(basis=np.ones((3, 2))),
    "basis has a single row": dict(basis=np.ones((1, 2))),
    "basis has a row per mode": dict(basis=np.ones((2, 4))),
    "lift has an entry too many": dict(lift=np.zeros(5)),
    "lift has an entry too few": dict(lift=np.zeros(3)),
    "lift is a number": dict(lift=0.0),
    "lift is a column": dict(lift=np.zeros((4, 1))),
    "lift has a row per state": dict(lift=np.zeros((2, 4))),
    "E is zero": dict(E=0.0),
    "E is negative": dict(E=-100.0),
    "nu is one half": dict(nu=0.5),
    "nu is minus one": dict(nu=-1.0),
    "only two displacements are fixed": dict(fixed_dofs=[0, 1], fixed_displacements=[0.0, 0.0]),
    "one fixed displacement too few": dict(fixed_displacements=[0.0, 0.0]),
    "a degree of freedom outside the plate": dict(fixed_dofs=[0, 1, 8]),
    "a degree of freedom twice": dict(fixed_dofs=[0, 1, 1]),
}


@pytest.mark.parametrize("change", list(MODES_VALUE_ERRORS.values()), ids=list(MODES_VALUE_ERRORS))
def test_modes_arguments_that_do_not_fit_are_a_value_error(change):
    with pytest.raises(ValueError):
        modes(**a_good_modes(**change))


@pytest.mark.parametrize("where", ["basis", "lift"])
def test_modes_a_nan_in_the_modes_or_in_the_lifting_is_never_returned_silently(where):
    arguments = a_good_modes()
    value = np.array(arguments[where])
    value.flat[-1] = NAN
    arguments[where] = value
    with pytest.raises((ValueError, RuntimeError)):
        modes(**arguments)


MODES_TYPE_ERRORS = {
    "E is a string": dict(E="steel"),
    "T_ref is None": dict(T_ref=None),
    "a degree of freedom that is not an integer": dict(fixed_dofs=[0.5, 1, 3]),
    "fixed_displacements is None": dict(fixed_displacements=None),
}


@pytest.mark.parametrize("change", list(MODES_TYPE_ERRORS.values()), ids=list(MODES_TYPE_ERRORS))
def test_modes_arguments_of_the_wrong_type_are_a_type_error(change):
    with pytest.raises(TypeError):
        modes(**a_good_modes(**change))


# ---------------------------------------------------------------- the whole chain against the full model (C++)

NX, NY = 6, 4
RHO_C, CONDUCTIVITY = 1.5, 0.8
T_REF = 22.0  # inside the range of the temperatures: the plate is both hotter and colder than the reference


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
        nodes = {"left": self.left, "border": self.border}[name]
        if name == "left":
            values = [20.0 + 5.0 * self.y[node] for node in nodes]
        else:
            values = [20.0 + self.x[node] for node in nodes]
        lift = np.zeros(self.n)
        lift[nodes] = values
        start = self.start.copy()
        start[nodes] = values
        return nodes, values, lift, start

    def constraints(self, name):
        return constraints_of(name, self.mesh, NX, NY)

    def thermomechanical(self, start, nodes, values, dt, n_steps, theta, constraints, material, T_ref, load=None):
        """The states of the C++ solver at the steps 0, 1, ..., n_steps."""
        return thermomech.solve_thermomechanical_2d(
            self.mesh, RHO_C, CONDUCTIVITY, self.load if load is None else load, nodes, values, start, dt, n_steps,
            theta, 1, material["E"], material["nu"], material["alpha"], T_ref, constraints["fixed_dofs"],
            constraints["fixed_displacements"])


@pytest.fixture(scope="module")
def plate():
    return Plate()


def pod_modes(states, lift, mass, count=None):
    """The POD modes (step 7a) of the deviations T - T_g: as many as the singular values above 1e-6 of the first one,
    or count."""
    from twin.pod import pod_basis
    snapshots = (states - lift).T
    if count is None:
        sigma = pod_basis(snapshots, mass, 1)[1]
        count = int(np.sum(sigma > 1e-6 * sigma[0]))
    return pod_basis(snapshots, mass, count)[0]


def chain_basis(kind, plate, states, lift):
    if kind == "qr":
        return np.linalg.qr((states - lift).T)[0]
    return pod_modes(states, lift, plate.mass)


def reduced_chain(plate, basis, lift, nodes, values, start, dt, n_steps, theta, constraints, material, T_ref,
                  load=None):
    """The reduced model: the temperatures of the reduced heat model (step 7b), then the displacements and the stresses
    from the mechanical modes, and the von Mises stress from the stresses and the mean temperatures of the triangles.
    Returns T (m, n), u (m, 2 n), sigma (m, n_tri, 3) and von Mises (m, n_tri)."""
    T = reduced_heat(basis, plate.mass, plate.stiffness, plate.load if load is None else load, nodes, values, start, dt,
                     n_steps, theta)
    coordinates = np.linalg.lstsq(basis, (T - lift).T, rcond=None)[0].T  # every state is the lifting + basis a
    u_lift, u_modes, s_lift, s_modes = modes(plate.mesh, basis, lift, **material, T_ref=T_ref, **constraints)
    u = rebuild(u_modes, coordinates, u_lift)
    sigma = rebuild(s_modes, coordinates, s_lift).reshape(len(T), plate.mesh.num_triangles, 3)
    rise = mean_of(plate.mesh.triangles, T) - T_ref
    return T, u, sigma, von_mises(sigma, rise, **material)


def assert_the_chains_agree(reduced, full, temperature_error=1e-8, error=1e-8, stress_error=1e-7):
    T, u, sigma, vm = reduced
    assert np.abs(T - full["T"]).max() < temperature_error, "the temperatures"
    assert relative_error(u, full["u"]) < error, "the displacements"
    assert relative_error(sigma, full["sigma"]) < stress_error, "the stresses"
    assert relative_error(vm, full["von_mises"]) < stress_error, "the von Mises stress"


STEP_FOR_THETA = [(0.0, 0.01), (0.5, 0.05), (2.0 / 3.0, 0.05), (1.0, 0.05)]  # forward Euler needs a small time step


@pytest.mark.parametrize("kind", ["qr", "pod"])
@pytest.mark.parametrize("theta, dt", STEP_FOR_THETA, ids=lambda value: format(value, ".3f"))
@pytest.mark.parametrize("name", CONSTRAINTS)
@pytest.mark.parametrize("side", ["left", "border"])
def test_chain_reproduces_the_full_model_when_the_basis_contains_its_trajectory(plate, side, name, theta, dt, kind):
    # the temperatures are those of the reduced heat model (exact when the basis contains the trajectory) and the
    # mechanics is affine in the temperature: the displacements, the stresses and the von Mises stress of the full model
    nodes, values, lift, start = plate.side(side)
    constraints = plate.constraints(name)
    full = plate.thermomechanical(start, nodes, values, dt, 3, theta, constraints, STEEL, T_REF)
    basis = chain_basis(kind, plate, full["T"], lift)
    assert basis.shape[1] == 4
    reduced = reduced_chain(plate, basis, lift, nodes, values, start, dt, 3, theta, constraints, STEEL, T_REF)
    assert_the_chains_agree(reduced, full)


@pytest.mark.parametrize("name", CONSTRAINTS)
def test_chain_the_hand_material_agrees_with_the_full_model_too(plate, name):
    nodes, values, lift, start = plate.side("left")
    constraints = plate.constraints(name)
    full = plate.thermomechanical(start, nodes, values, 0.05, 3, 0.5, constraints, HAND_PLANE, 20.0)
    basis = chain_basis("qr", plate, full["T"], lift)
    reduced = reduced_chain(plate, basis, lift, nodes, values, start, 0.05, 3, 0.5, constraints, HAND_PLANE, 20.0)
    assert_the_chains_agree(reduced, full)


@pytest.mark.parametrize("name", CONSTRAINTS)
def test_chain_with_a_poor_basis_it_is_still_the_exact_mechanics_of_the_reduced_temperatures(plate, name):
    # two modes for twenty steps: the reduced temperatures are far from those of the full model, but the displacements
    # and the stresses are those of the full mechanics for them: the mechanics adds no error of its own
    nodes, values, lift, start = plate.side("left")
    constraints = plate.constraints(name)
    full = plate.thermomechanical(start, nodes, values, 0.05, 20, 0.5, constraints, STEEL, T_REF)
    basis = np.linalg.qr((full["T"][[0, 10]] - lift).T)[0]
    T, u, sigma, vm = reduced_chain(plate, basis, lift, nodes, values, start, 0.05, 20, 0.5, constraints, STEEL, T_REF)
    assert np.abs(T - full["T"]).max() > 0.05
    triangles = np.asarray(plate.mesh.triangles)
    for state in (0, 1, 7, 20):
        u_full, sigma_full = response(plate.mesh, T[state], **STEEL, T_ref=T_REF, **constraints)
        assert relative_error(u[state], u_full) < 1e-8
        assert relative_error(sigma[state], sigma_full) < 1e-7
        rise = T[state][triangles].mean(axis=1) - T_REF
        assert relative_error(vm[state], von_mises(sigma_full, rise, **STEEL)) < 1e-7


@pytest.mark.parametrize("q", [2.0, -1.5, 5.0])
def test_chain_predicts_other_loads_with_the_same_modes(plate, q):
    # the deviations of the states are affine in the strength q of the load: two runs (q = 1 and q = 3) give modes that
    # contain the states for every q, and the mechanical modes do not depend on the load
    nodes, values, lift, _ = plate.side("left")
    constraints = plate.constraints("stretch")
    states = [plate.thermomechanical(lift, nodes, values, 0.05, 3, 0.5, constraints, STEEL, T_REF,
                                     load=s * plate.load)["T"] for s in (1.0, 3.0)]
    basis = np.linalg.qr(np.column_stack([(run[step] - lift) for run in states for step in (1, 2, 3)]))[0]
    full = plate.thermomechanical(lift, nodes, values, 0.05, 3, 0.5, constraints, STEEL, T_REF, load=q * plate.load)
    reduced = reduced_chain(plate, basis, lift, nodes, values, lift, 0.05, 3, 0.5, constraints, STEEL, T_REF,
                            load=q * plate.load)
    assert_the_chains_agree(reduced, full)


def test_chain_with_a_few_pod_modes_follows_a_long_run_of_the_full_model(plate):
    # 40 steps of Crank-Nicolson and 8 POD modes: the reduced displacements, stresses and von Mises stress are close to
    # those of the full model, as close as the reduced temperatures are
    nodes, values, lift, start = plate.side("left")
    constraints = plate.constraints("pins")
    full = plate.thermomechanical(start, nodes, values, 0.05, 40, 0.5, constraints, STEEL, T_REF)
    basis = pod_modes(full["T"], lift, plate.mass, 8)
    T, u, sigma, vm = reduced_chain(plate, basis, lift, nodes, values, start, 0.05, 40, 0.5, constraints, STEEL, T_REF)
    assert relative_error(T - lift, full["T"] - lift) < 5e-4
    assert relative_error(u, full["u"]) < 1.5e-3
    assert relative_error(sigma, full["sigma"]) < 3e-3
    assert relative_error(vm, full["von_mises"]) < 2.5e-3
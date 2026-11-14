"""Step 6c: the matrices K and M and the load vector, ready for Python.

    assemble_stiffness_2d(mesh, k)    K: scipy.sparse.csr_matrix, float64, n x n    (heat conduction)
    assemble_mass_2d(mesh, rho_c)     M: scipy.sparse.csr_matrix, float64, n x n    (consistent mass)
    assemble_load_2d(mesh, q)         f: NumPy array, float64, shape (n,)           (heat source)

n = mesh.num_nodes. In assemble_load_2d, q is a number (the same source everywhere) or a Python function q(x, y)
(a source that changes from point to point). The function is called from C++ at the midpoints of the sides of every
triangle: the midpoint rule of the C++ code. The matrices have no boundary conditions, so K is singular.
"""
import gc
import math
from functools import partial

import numpy as np
import pytest
import scipy.sparse as sp

import thermomech

CASES = [(1.0, 1.0, 1, 1), (2.0, 3.0, 4, 2), (4.0, 1.0, 8, 2), (1.0, 5.0, 2, 7), (0.1, 0.02, 5, 3),
         (10.0, 10.0, 16, 16)]
SMALL_CASES = CASES[:5]
CASES_WITH_INTERIOR_NODES = CASES[1:]
MATRICES = ["assemble_stiffness_2d", "assemble_mass_2d"]
ALL_FUNCTIONS = MATRICES + ["assemble_load_2d"]

# One unit square, cut in two triangles. The nodes are numbered row by row from the bottom: 0 = (0, 0), 1 = (1, 0),
# 2 = (0, 1), 3 = (1, 1). The triangles are (0, 1, 3) and (0, 3, 2), both with area 1/2, and the diagonal is 0-3.
#   K (k = 1): every row adds up to 0. The nodes 0 and 3, the ends of the diagonal, do not see each other: K[0, 3] = 0.
#   M (rho_c = 1): all the entries add up to the area 1. The nodes 1 and 2 share no triangle: M[1, 2] = 0.
K_UNIT_SQUARE = np.array([[1.0, -0.5, -0.5, 0.0],
                          [-0.5, 1.0, 0.0, -0.5],
                          [-0.5, 0.0, 1.0, -0.5],
                          [0.0, -0.5, -0.5, 1.0]])
M_UNIT_SQUARE = np.array([[4.0, 1.0, 1.0, 2.0],
                          [1.0, 2.0, 0.0, 1.0],
                          [1.0, 0.0, 2.0, 1.0],
                          [2.0, 1.0, 1.0, 4.0]]) / 24.0

# One rectangle 2 long and 1 high: the nodes 0 = (0, 0), 1 = (2, 0), 2 = (0, 1), 3 = (2, 1), triangles of area 1.
# The sides along x (0-1 and 2-3) are twice as long as the sides along y (0-2 and 1-3), so they conduct less: -1/4
# against -1.
K_RECTANGLE = np.array([[1.25, -0.25, -1.0, 0.0],
                        [-0.25, 1.25, 0.0, -1.0],
                        [-1.0, 0.0, 1.25, -0.25],
                        [0.0, -1.0, -0.25, 1.25]])
M_RECTANGLE = 2.0 * M_UNIT_SQUARE


def make(case):
    length_x, length_y, nx, ny = case
    return thermomech.rectangle_mesh(length_x, length_y, nx, ny)


def matrix(name, mesh, coefficient=1.0):
    return getattr(thermomech, name)(mesh, coefficient)


class SourceError(Exception):
    pass


class Ramp:
    """A source that is an object with a __call__ method, not a function."""

    def __call__(self, x, y):
        return 2.0 * x


def test_the_module_has_the_assembly_functions():
    for name in ALL_FUNCTIONS:
        assert callable(getattr(thermomech, name))


# ---------------------------------------------------------------- the matrices, by hand


def test_stiffness_of_the_unit_square_by_hand():
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 1, 1)
    K = thermomech.assemble_stiffness_2d(mesh, 1.0)
    np.testing.assert_allclose(K.toarray(), K_UNIT_SQUARE, rtol=0, atol=1e-14)


def test_mass_of_the_unit_square_by_hand():
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 1, 1)
    M = thermomech.assemble_mass_2d(mesh, 1.0)
    np.testing.assert_allclose(M.toarray(), M_UNIT_SQUARE, rtol=0, atol=1e-14)


def test_stiffness_and_mass_of_a_2_by_1_rectangle_by_hand():
    mesh = thermomech.rectangle_mesh(2.0, 1.0, 1, 1)
    K = thermomech.assemble_stiffness_2d(mesh, 1.0)
    M = thermomech.assemble_mass_2d(mesh, 1.0)
    np.testing.assert_allclose(K.toarray(), K_RECTANGLE, rtol=0, atol=1e-14)
    np.testing.assert_allclose(M.toarray(), M_RECTANGLE, rtol=0, atol=1e-14)


# ---------------------------------------------------------------- the matrices, type and shape


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", MATRICES)
def test_a_matrix_is_a_float64_csr_matrix_with_a_row_for_every_node(case, name):
    mesh = make(case)
    A = matrix(name, mesh)
    assert isinstance(A, sp.csr_matrix)
    assert A.format == "csr"
    assert A.dtype == np.float64
    assert A.shape == (mesh.num_nodes, mesh.num_nodes)
    assert A.has_canonical_format  # the columns of every row are sorted and no position is stored twice


@pytest.mark.parametrize("case", CASES)
def test_the_matrices_store_only_the_diagonal_and_the_sides_of_the_mesh(case):
    _, _, nx, ny = case
    mesh = make(case)
    most = mesh.num_nodes + 2 * (3 * nx * ny + nx + ny)  # the diagonal, plus two numbers for every side
    # every entry of M is positive: M stores exactly that many numbers
    assert matrix("assemble_mass_2d", mesh).nnz == most
    # K may store a 0 on the sides along the diagonals of the cells, but nothing outside the sides of the mesh
    assert matrix("assemble_stiffness_2d", mesh).nnz <= most


# ---------------------------------------------------------------- the matrices, properties


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", MATRICES)
def test_a_matrix_is_symmetric(case, name):
    A = matrix(name, make(case))
    assert abs(A - A.T).max() <= 1e-12 * abs(A).max()


@pytest.mark.parametrize("case", CASES)
def test_every_row_of_the_stiffness_adds_up_to_zero(case):
    mesh = make(case)
    K = matrix("assemble_stiffness_2d", mesh, 2.5)
    assert np.abs(K @ np.ones(mesh.num_nodes)).max() <= 1e-12 * K.diagonal().max()


@pytest.mark.parametrize("case", CASES)
def test_the_entries_of_the_mass_add_up_to_rho_c_times_the_area(case):
    length_x, length_y, _, _ = case
    mesh = make(case)
    M = matrix("assemble_mass_2d", mesh, 2.5)
    ones = np.ones(mesh.num_nodes)
    assert ones @ (M @ ones) == pytest.approx(2.5 * length_x * length_y, rel=1e-10)


@pytest.mark.parametrize("case", CASES)
def test_the_mass_gives_the_integrals_of_x_squared_and_x_times_y(case):
    length_x, length_y, _, _ = case
    mesh = make(case)
    M = matrix("assemble_mass_2d", mesh, 2.0)
    x, y = mesh.x, mesh.y
    assert x @ (M @ x) == pytest.approx(2.0 * length_x**3 * length_y / 3.0, rel=1e-10)
    assert x @ (M @ y) == pytest.approx(2.0 * length_x**2 * length_y**2 / 4.0, rel=1e-10)


@pytest.mark.parametrize("case", CASES)
def test_the_stiffness_gives_the_energy_of_a_linear_temperature(case):
    length_x, length_y, _, _ = case
    mesh = make(case)
    K = matrix("assemble_stiffness_2d", mesh, 1.5)
    T = 2.0 * mesh.x - 3.0 * mesh.y
    # the gradient of T is (2, -3) everywhere: T K T = k |grad T|^2 area
    assert T @ (K @ T) == pytest.approx(1.5 * 13.0 * length_x * length_y, rel=1e-10)


@pytest.mark.parametrize("case", CASES_WITH_INTERIOR_NODES)
def test_the_stiffness_of_an_interior_node_is_the_laplacian_of_the_temperature(case):
    length_x, length_y, nx, ny = case
    mesh = make(case)
    K = matrix("assemble_stiffness_2d", mesh, 1.5)
    interior = [j * (nx + 1) + i for j in range(1, ny) for i in range(1, nx)]
    # a straight line is not changed by the conduction inside the plate
    line = 2.0 * mesh.x - 3.0 * mesh.y
    inside = (K @ line)[interior]
    np.testing.assert_allclose(inside, 0.0, rtol=0, atol=1e-11 * K.diagonal().max() * np.abs(line).max())
    # x^2 + 3 y^2 has the Laplacian 8, and every interior node owns the area hx * hy: K T = -k * 8 * hx * hy
    T = mesh.x**2 + 3.0 * mesh.y**2
    hx, hy = length_x / nx, length_y / ny
    np.testing.assert_allclose((K @ T)[interior], -1.5 * 8.0 * hx * hy, rtol=1e-9)


@pytest.mark.parametrize("case", SMALL_CASES)
def test_the_stiffness_has_only_the_constants_in_its_kernel_and_the_mass_is_positive_definite(case):
    mesh = make(case)
    k_values = np.linalg.eigvalsh(matrix("assemble_stiffness_2d", mesh).toarray())
    m_values = np.linalg.eigvalsh(matrix("assemble_mass_2d", mesh).toarray())
    assert k_values[0] >= -1e-12 * k_values[-1]
    assert np.sum(np.abs(k_values) <= 1e-10 * k_values[-1]) == 1
    assert m_values[0] > 0.0


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", MATRICES)
def test_k_and_rho_c_multiply_the_matrix(case, name):
    mesh = make(case)
    one = matrix(name, mesh, 1.0).toarray()
    other = matrix(name, mesh, 2.5).toarray()
    np.testing.assert_allclose(other, 2.5 * one, rtol=1e-12, atol=1e-14 * np.abs(one).max())


# ---------------------------------------------------------------- the load vector


def test_the_load_vector_is_a_float64_array_with_one_number_per_node():
    mesh = thermomech.rectangle_mesh(2.0, 1.0, 3, 2)
    for q in (1.0, lambda x, y: x + y):
        f = thermomech.assemble_load_2d(mesh, q)
        assert isinstance(f, np.ndarray)
        assert f.dtype == np.float64
        assert f.shape == (mesh.num_nodes,)


def test_a_uniform_source_by_hand():
    # every triangle gives a third of its heat q * A to each of its corners; the nodes 0 and 3 are in both triangles
    square = thermomech.rectangle_mesh(1.0, 1.0, 1, 1)
    np.testing.assert_allclose(thermomech.assemble_load_2d(square, 1.0), [1 / 3, 1 / 6, 1 / 6, 1 / 3], atol=1e-14)
    np.testing.assert_allclose(thermomech.assemble_load_2d(square, 6.0), [2.0, 1.0, 1.0, 2.0], atol=1e-14)
    rectangle = thermomech.rectangle_mesh(2.0, 1.0, 1, 1)
    np.testing.assert_allclose(thermomech.assemble_load_2d(rectangle, 1.0), [2 / 3, 1 / 3, 1 / 3, 2 / 3], atol=1e-14)


def test_a_source_that_follows_x_on_the_unit_square_by_hand():
    # the integral of x * N_i over a triangle is A/12 * (x_i + the sum of the three x of the triangle):
    #   triangle (0, 1, 3): x = 0, 1, 1 gives 2/24, 3/24, 3/24;  triangle (0, 3, 2): x = 0, 1, 0 gives 1/24, 2/24, 1/24
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 1, 1)
    f = thermomech.assemble_load_2d(mesh, lambda x, y: x)
    np.testing.assert_allclose(f, np.array([3.0, 3.0, 1.0, 5.0]) / 24.0, atol=1e-14)


def test_a_source_that_follows_y_on_a_2_by_1_rectangle_by_hand():
    # the same with A = 1: triangle (0, 1, 3) has y = 0, 0, 1 and triangle (0, 3, 2) has y = 0, 1, 1
    mesh = thermomech.rectangle_mesh(2.0, 1.0, 1, 1)
    f = thermomech.assemble_load_2d(mesh, lambda x, y: y)
    np.testing.assert_allclose(f, np.array([3.0, 1.0, 3.0, 5.0]) / 12.0, atol=1e-14)


@pytest.mark.parametrize("case", CASES)
def test_a_uniform_source_is_the_mass_matrix_times_the_constant(case):
    mesh = make(case)
    M = matrix("assemble_mass_2d", mesh, 1.0)
    f = thermomech.assemble_load_2d(mesh, 5.0)
    np.testing.assert_allclose(f, M @ np.full(mesh.num_nodes, 5.0), rtol=1e-12)


@pytest.mark.parametrize("case", CASES)
def test_a_linear_source_is_the_mass_matrix_times_its_values_at_the_nodes(case):
    mesh = make(case)
    M = matrix("assemble_mass_2d", mesh, 1.0)
    f = thermomech.assemble_load_2d(mesh, lambda x, y: 2.0 * x - 3.0 * y + 1.0)
    expected = M @ (2.0 * mesh.x - 3.0 * mesh.y + 1.0)
    np.testing.assert_allclose(f, expected, rtol=0, atol=1e-11 * np.abs(expected).max())


@pytest.mark.parametrize("case", CASES)
def test_the_load_of_a_quadratic_source_adds_up_to_its_integral(case):
    length_x, length_y, _, _ = case
    mesh = make(case)
    f = thermomech.assemble_load_2d(mesh, lambda x, y: x * x + y * y)
    assert f.sum() == pytest.approx(length_y * length_x**3 / 3.0 + length_x * length_y**3 / 3.0, rel=1e-10)


def test_a_constant_function_gives_the_same_load_as_a_number():
    mesh = thermomech.rectangle_mesh(2.0, 3.0, 4, 2)
    from_function = thermomech.assemble_load_2d(mesh, lambda x, y: 4.0)
    np.testing.assert_allclose(from_function, thermomech.assemble_load_2d(mesh, 4.0), rtol=1e-13)


def test_the_python_function_is_called_at_the_midpoints_of_the_sides_of_every_triangle():
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 1, 1)
    calls = []

    def source(x, y):
        calls.append((x, y))
        return 1.0

    thermomech.assemble_load_2d(mesh, source)
    assert all(type(x) is float and type(y) is float for x, y in calls)
    # triangle (0, 1, 3): (0.5, 0) (1, 0.5) (0.5, 0.5);  triangle (0, 3, 2): (0.5, 0.5) (0.5, 1) (0, 0.5)
    expected = [(0.5, 0.0), (1.0, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 0.5)]
    assert len(calls) == 6
    np.testing.assert_allclose(sorted(calls), sorted(expected), rtol=0, atol=1e-14)


def test_any_python_callable_can_be_the_source():
    mesh = thermomech.rectangle_mesh(2.0, 1.0, 3, 2)
    expected = thermomech.assemble_load_2d(mesh, lambda x, y: 2.0 * x)

    def twice_x(x, y):
        return 2.0 * x

    def scaled_x(factor, x, y):
        return factor * x

    for source in (twice_x, partial(scaled_x, 2.0), Ramp()):
        np.testing.assert_allclose(thermomech.assemble_load_2d(mesh, source), expected, rtol=1e-13)
    # functions that are not written in Python work too
    np.testing.assert_allclose(thermomech.assemble_load_2d(mesh, np.hypot),
                               thermomech.assemble_load_2d(mesh, math.hypot), rtol=1e-13)


def test_the_python_function_may_return_an_int_or_a_numpy_float():
    mesh = thermomech.rectangle_mesh(2.0, 1.0, 3, 2)
    expected = thermomech.assemble_load_2d(mesh, 3.0)
    np.testing.assert_allclose(thermomech.assemble_load_2d(mesh, lambda x, y: 3), expected, rtol=1e-13)
    np.testing.assert_allclose(thermomech.assemble_load_2d(mesh, lambda x, y: np.float64(3.0)), expected, rtol=1e-13)


def test_an_error_in_the_python_function_reaches_the_caller_unchanged():
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 2, 2)

    def broken(x, y):
        raise SourceError("the source is broken")

    with pytest.raises(SourceError, match="the source is broken"):
        thermomech.assemble_load_2d(mesh, broken)


def test_the_load_vector_belongs_to_the_caller():
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 1, 1)
    f = thermomech.assemble_load_2d(mesh, 1.0)
    assert f.flags["WRITEABLE"]
    f[:] = 0.0
    np.testing.assert_allclose(thermomech.assemble_load_2d(mesh, 1.0), [1 / 3, 1 / 6, 1 / 6, 1 / 3], atol=1e-14)


# ---------------------------------------------------------------- how the functions are called


def test_the_arguments_can_be_given_by_name_and_as_integers():
    mesh = thermomech.rectangle_mesh(2.0, 1.0, 3, 2)
    K = thermomech.assemble_stiffness_2d(mesh=mesh, k=2)
    M = thermomech.assemble_mass_2d(mesh=mesh, rho_c=3)
    np.testing.assert_array_equal(K.toarray(), thermomech.assemble_stiffness_2d(mesh, 2.0).toarray())
    np.testing.assert_array_equal(M.toarray(), thermomech.assemble_mass_2d(mesh, 3.0).toarray())
    np.testing.assert_array_equal(thermomech.assemble_load_2d(mesh=mesh, q=4), thermomech.assemble_load_2d(mesh, 4.0))
    by_name = thermomech.assemble_load_2d(mesh=mesh, q=lambda x, y: x)
    np.testing.assert_array_equal(by_name, thermomech.assemble_load_2d(mesh, lambda x, y: x))


@pytest.mark.parametrize("name", ALL_FUNCTIONS)
@pytest.mark.parametrize("not_a_mesh", [None, 5, "mesh", [1.0, 2.0]])
def test_the_first_argument_must_be_a_mesh(name, not_a_mesh):
    with pytest.raises(TypeError):
        getattr(thermomech, name)(not_a_mesh, 1.0)


@pytest.mark.parametrize("name", ALL_FUNCTIONS)
@pytest.mark.parametrize("wrong", [None, "hot", [1.0]])
def test_the_second_argument_cannot_be_none_a_string_or_a_list(name, wrong):
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 1, 1)
    with pytest.raises(TypeError):
        getattr(thermomech, name)(mesh, wrong)


@pytest.mark.parametrize("name", ALL_FUNCTIONS)
def test_the_second_argument_is_required(name):
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 1, 1)
    with pytest.raises(TypeError):
        getattr(thermomech, name)(mesh)


def test_the_results_stay_valid_after_the_mesh_is_gone_and_more_work_is_done():
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 1, 1)
    f = thermomech.assemble_load_2d(mesh, 1.0)
    K = thermomech.assemble_stiffness_2d(mesh, 1.0)
    M = thermomech.assemble_mass_2d(mesh, 1.0)
    del mesh
    gc.collect()
    for _ in range(50):
        bigger = thermomech.rectangle_mesh(3.0, 2.0, 7, 5)
        thermomech.assemble_load_2d(bigger, 9.0)
        thermomech.assemble_stiffness_2d(bigger, 9.0)
        thermomech.assemble_mass_2d(bigger, 9.0)
    np.testing.assert_allclose(f, [1 / 3, 1 / 6, 1 / 6, 1 / 3], atol=1e-14)
    np.testing.assert_allclose(K.toarray(), K_UNIT_SQUARE, rtol=0, atol=1e-14)
    np.testing.assert_allclose(M.toarray(), M_UNIT_SQUARE, rtol=0, atol=1e-14)
"""Step 6b: the nodes and the triangles of a mesh as NumPy arrays.

    mesh.x, mesh.y    float64, shape (num_nodes,)        coordinates of the nodes
    mesh.triangles    int64,   shape (num_triangles, 3)  node numbers of every triangle, counter-clockwise

The numbering is the one fixed by the C++ tests of rectangle_mesh: nodes row by row from the bottom, the node of
column i and row j is j*(nx+1) + i. The cell k = j*nx + i has the corners a = j*(nx+1) + i, b = a + 1,
d = a + nx + 1, c = d + 1 (bottom left, bottom right, top left, top right) and gives the triangles
number 2k = (a, b, c) and number 2k+1 = (a, c, d).
"""
import gc
from collections import Counter

import numpy as np
import pytest

import thermomech

CASES = [(1.0, 1.0, 1, 1), (2.0, 3.0, 4, 2), (4.0, 1.0, 8, 2), (1.0, 5.0, 2, 7), (0.1, 0.02, 5, 3),
         (10.0, 10.0, 16, 16)]


def expected_x(length_x, nx, ny):
    return np.tile(np.linspace(0.0, length_x, nx + 1), ny + 1)


def expected_y(length_y, nx, ny):
    return np.repeat(np.linspace(0.0, length_y, ny + 1), nx + 1)


def expected_triangles(nx, ny):
    rows = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b = a + 1
            d = a + nx + 1
            c = d + 1
            rows.append([a, b, c])
            rows.append([a, c, d])
    return np.array(rows, dtype=np.int64)


def test_x_and_y_are_float64_arrays_with_one_value_per_node():
    mesh = thermomech.rectangle_mesh(4.0, 1.0, 8, 2)
    for coords in (mesh.x, mesh.y):
        assert isinstance(coords, np.ndarray)
        assert coords.dtype == np.float64
        assert coords.shape == (mesh.num_nodes,)
        assert mesh.num_nodes == 27


def test_triangles_are_an_int64_array_with_three_nodes_per_triangle():
    mesh = thermomech.rectangle_mesh(4.0, 1.0, 8, 2)
    triangles = mesh.triangles
    assert isinstance(triangles, np.ndarray)
    assert triangles.dtype == np.int64
    assert triangles.shape == (mesh.num_triangles, 3)
    assert mesh.num_triangles == 32


def test_the_arrays_are_c_contiguous():
    mesh = thermomech.rectangle_mesh(4.0, 1.0, 8, 2)
    assert mesh.x.flags["C_CONTIGUOUS"]
    assert mesh.y.flags["C_CONTIGUOUS"]
    assert mesh.triangles.flags["C_CONTIGUOUS"]


def test_two_by_one_cells_by_hand():
    # nodes 0 1 2 on the bottom row, 3 4 5 on the top row; every cell is cut from its bottom left to its top right
    mesh = thermomech.rectangle_mesh(2.0, 1.0, 2, 1)
    np.testing.assert_allclose(mesh.x, [0, 1, 2, 0, 1, 2], atol=1e-14)
    np.testing.assert_allclose(mesh.y, [0, 0, 0, 1, 1, 1], atol=1e-14)
    assert mesh.triangles.tolist() == [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]]


def test_three_by_two_cells_with_unequal_steps_by_hand():
    # dx = 1 and dy = 0.5, four nodes per row
    mesh = thermomech.rectangle_mesh(3.0, 1.0, 3, 2)
    np.testing.assert_allclose(mesh.x, [0, 1, 2, 3] * 3, atol=1e-14)
    np.testing.assert_allclose(mesh.y, [0.0] * 4 + [0.5] * 4 + [1.0] * 4, atol=1e-14)
    triangles = mesh.triangles
    assert triangles[0].tolist() == [0, 1, 5]
    assert triangles[1].tolist() == [0, 5, 4]
    assert triangles[6].tolist() == [4, 5, 9]
    assert triangles[7].tolist() == [4, 9, 8]
    assert triangles[10].tolist() == [6, 7, 11]
    assert triangles[11].tolist() == [6, 11, 10]


@pytest.mark.parametrize("length_x, length_y, nx, ny", CASES)
def test_the_nodes_are_numbered_row_by_row(length_x, length_y, nx, ny):
    mesh = thermomech.rectangle_mesh(length_x, length_y, nx, ny)
    np.testing.assert_allclose(mesh.x, expected_x(length_x, nx, ny), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(mesh.y, expected_y(length_y, nx, ny), rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("length_x, length_y, nx, ny", CASES)
def test_the_triangles_follow_the_numbering_rule(length_x, length_y, nx, ny):
    mesh = thermomech.rectangle_mesh(length_x, length_y, nx, ny)
    assert np.array_equal(mesh.triangles, expected_triangles(nx, ny))


@pytest.mark.parametrize("length_x, length_y, nx, ny", CASES)
def test_every_triangle_is_counter_clockwise_and_half_a_cell(length_x, length_y, nx, ny):
    mesh = thermomech.rectangle_mesh(length_x, length_y, nx, ny)
    x, y, t = mesh.x, mesh.y, mesh.triangles
    x0, x1, x2 = x[t[:, 0]], x[t[:, 1]], x[t[:, 2]]
    y0, y1, y2 = y[t[:, 0]], y[t[:, 1]], y[t[:, 2]]
    area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    assert np.all(area > 0.0)
    np.testing.assert_allclose(area, 0.5 * (length_x / nx) * (length_y / ny), rtol=1e-12)
    assert area.sum() == pytest.approx(length_x * length_y, rel=1e-12)


@pytest.mark.parametrize("length_x, length_y, nx, ny", CASES)
def test_the_node_numbers_are_valid_and_every_node_is_used(length_x, length_y, nx, ny):
    mesh = thermomech.rectangle_mesh(length_x, length_y, nx, ny)
    triangles = mesh.triangles
    assert triangles.min() == 0
    assert triangles.max() == mesh.num_nodes - 1
    assert set(triangles.ravel().tolist()) == set(range(mesh.num_nodes))
    assert all(len(set(row)) == 3 for row in triangles.tolist())


@pytest.mark.parametrize("length_x, length_y, nx, ny", CASES)
def test_every_side_belongs_to_two_triangles_or_to_one_on_the_boundary(length_x, length_y, nx, ny):
    mesh = thermomech.rectangle_mesh(length_x, length_y, nx, ny)
    counts = Counter()
    for a, b, c in mesh.triangles.tolist():
        for u, v in ((a, b), (b, c), (c, a)):
            counts[(min(u, v), max(u, v))] += 1
    assert set(counts.values()) <= {1, 2}
    assert sum(1 for n in counts.values() if n == 1) == 2 * (nx + ny)
    assert len(counts) == 3 * nx * ny + nx + ny


def test_the_arrays_cannot_be_modified():
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 2, 2)
    for array in (mesh.x, mesh.y, mesh.triangles):
        assert not array.flags["WRITEABLE"]
    with pytest.raises(ValueError):
        mesh.x[0] = 5.0
    with pytest.raises(ValueError):
        mesh.y[0] = 5.0
    with pytest.raises(ValueError):
        mesh.triangles[0, 0] = 5


def test_reading_the_arrays_twice_gives_the_same_values():
    mesh = thermomech.rectangle_mesh(3.0, 2.0, 3, 2)
    assert np.array_equal(mesh.x, mesh.x)
    assert np.array_equal(mesh.y, mesh.y)
    assert np.array_equal(mesh.triangles, mesh.triangles)


def test_the_arrays_outlive_the_mesh():
    x = thermomech.rectangle_mesh(1.0, 1.0, 3, 3).x
    triangles = thermomech.rectangle_mesh(1.0, 1.0, 3, 3).triangles
    gc.collect()
    np.testing.assert_allclose(x[:4], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], atol=1e-14)
    assert triangles.shape == (18, 3)
    assert triangles[0].tolist() == [0, 1, 5]
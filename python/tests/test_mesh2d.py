"""Step 6a: the module can be imported and a rectangular mesh can be built from Python.

CTest runs this file as the test "python_tests". By hand, from the project folder:
    python -m pytest python/tests -v
with the folder that contains the compiled module in PYTHONPATH (build/python).
"""
import pytest

import thermomech


def test_the_module_exposes_the_mesh_class_and_the_builder():
    assert hasattr(thermomech, "Mesh2D")
    assert hasattr(thermomech, "rectangle_mesh")


def test_rectangle_mesh_returns_a_mesh():
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 2, 2)
    assert isinstance(mesh, thermomech.Mesh2D)


def test_one_cell_is_four_nodes_and_two_triangles():
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 1, 1)
    assert mesh.num_nodes == 4
    assert mesh.num_triangles == 2


def test_two_by_two_cells_by_hand():
    # 3 x 3 nodes; 4 cells, each cut into 2 triangles
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 2, 2)
    assert mesh.num_nodes == 9
    assert mesh.num_triangles == 8


@pytest.mark.parametrize("nx, ny", [(1, 1), (2, 1), (1, 3), (4, 2), (7, 5), (16, 16)])
def test_the_counts_follow_the_formulas(nx, ny):
    mesh = thermomech.rectangle_mesh(2.0, 3.0, nx, ny)
    assert mesh.num_nodes == (nx + 1) * (ny + 1)
    assert mesh.num_triangles == 2 * nx * ny


def test_the_arguments_can_be_given_by_name():
    mesh = thermomech.rectangle_mesh(length_x=4.0, length_y=1.0, nx=8, ny=2)
    assert mesh.num_nodes == 27
    assert mesh.num_triangles == 32


def test_python_integers_are_accepted_as_lengths():
    mesh = thermomech.rectangle_mesh(4, 1, 8, 2)
    assert mesh.num_nodes == 27


def test_two_meshes_do_not_interfere():
    small = thermomech.rectangle_mesh(1.0, 1.0, 2, 2)
    large = thermomech.rectangle_mesh(1.0, 1.0, 4, 4)
    assert small.num_nodes == 9
    assert large.num_nodes == 25
    assert small.num_nodes == 9


def test_the_counts_cannot_be_assigned():
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 2, 2)
    with pytest.raises(AttributeError):
        mesh.num_nodes = 5
    with pytest.raises(AttributeError):
        mesh.num_triangles = 5


@pytest.mark.parametrize("length_x, length_y", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (1.0, -2.5), (float("nan"), 1.0)])
def test_a_side_that_is_not_positive_raises_value_error(length_x, length_y):
    with pytest.raises(ValueError):
        thermomech.rectangle_mesh(length_x, length_y, 2, 2)


@pytest.mark.parametrize("nx, ny", [(0, 3), (3, 0), (0, 0)])
def test_zero_cells_raise_value_error(nx, ny):
    with pytest.raises(ValueError):
        thermomech.rectangle_mesh(1.0, 1.0, nx, ny)


def test_a_negative_number_of_cells_is_rejected_by_python():
    with pytest.raises(TypeError):
        thermomech.rectangle_mesh(1.0, 1.0, -1, 2)


def test_a_fractional_number_of_cells_is_rejected_by_python():
    with pytest.raises(TypeError):
        thermomech.rectangle_mesh(1.0, 1.0, 2.0, 2)
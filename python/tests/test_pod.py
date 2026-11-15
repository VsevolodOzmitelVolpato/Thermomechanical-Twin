"""Step 7a: the POD basis of a set of snapshots, weighted with a mass matrix.

    from twin.pod import pod_basis
    basis, singular_values = pod_basis(snapshots, mass, n_modes)

snapshots is an array (n, m): the column j holds the snapshot j, the n nodal values of a field. mass is a symmetric
positive definite n x n matrix (a NumPy array or a scipy.sparse matrix) that defines the inner product (u, v) = u^T M v
and the norm |u| = sqrt(u^T M u). n_modes is an integer with 1 <= n_modes <= m.

The singular values of the snapshots (in the norm of M) are sigma_i = sqrt(lambda_i), where lambda_1 >= lambda_2 >= ...
are the eigenvalues of the m x m matrix S^T M S. The function returns

    basis            float64, (n, n_modes)   the first n_modes modes, one per column. They are orthonormal for M,
                                             basis^T M basis = I, and with P = basis basis^T M (the projection onto
                                             their span) the snapshots are as close to that span as possible:
                                             sum_k |s_k - P s_k|^2 = sigma_{n_modes+1}^2 + ... + sigma_m^2
    singular_values  float64, (m,)           all the singular values, descending, whatever n_modes is

Two sets of modes that differ by the sign of a column are the same basis: the sign is not fixed. A row that is zero in
every snapshot is zero in the basis. Both results are new arrays and the arguments are not modified. Data of any
real type or memory layout is accepted (lists too for the snapshots).

Errors: ValueError if snapshots is not 2-D or holds a NaN or an infinity, if mass is not n x n, if n_modes is not
between 1 and m, or if the snapshots do not have n_modes independent directions: sigma_{n_modes} <= 1e-6 sigma_1
(all-zero snapshots included). TypeError if n_modes is not an integer.
"""
import math

import numpy as np
import pytest
import scipy.sparse as sp

import thermomech

# Any warning, such as the square root of a tiny negative number, is an error in these tests.
pytestmark = pytest.mark.filterwarnings("error")

PI = math.pi


def pod(*args, **kwargs):
    # imported here and not at the top: while twin/pod.py does not exist only these tests fail, not the whole session
    from twin.pod import pod_basis
    return pod_basis(*args, **kwargs)


# ---------------------------------------------------------------- helpers


def dense(matrix):
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def build_mass(kind, n=None):
    """A symmetric positive definite matrix: the identity, a diagonal, a dense one, a tridiagonal sparse one or the
    consistent mass matrix of a mesh of 6 x 4 nodes (then n is not used)."""
    rng = np.random.default_rng(11)
    if kind == "identity":
        return np.eye(n)
    if kind == "diagonal":
        return np.diag(rng.uniform(0.5, 3.0, n))
    if kind == "dense":
        b = rng.standard_normal((n, n))
        return b @ b.T / n + 0.5 * np.eye(n)
    if kind == "sparse":
        return sp.diags([-np.ones(n - 1), 2.5 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format="csr")
    assert kind == "fem"
    return thermomech.assemble_mass_2d(thermomech.rectangle_mesh(2.0, 1.0, 5, 3), 1.5)


def planted(mass, sigma, seed=5):
    """Snapshots S = Q diag(sigma) W^T with Q^T M Q = I and W orthogonal. Their singular values are exactly sigma and
    their modes are the columns of Q: an answer that does not come out of any SVD."""
    rng = np.random.default_rng(seed)
    M = dense(mass)
    n, m = M.shape[0], len(sigma)
    lower = np.linalg.cholesky(M)
    q_euclid, _ = np.linalg.qr(rng.standard_normal((n, m)))
    modes = np.linalg.solve(lower.T, q_euclid)  # L^-T q: modes^T M modes = q^T L^-1 (L L^T) L^-T q = I
    w, _ = np.linalg.qr(rng.standard_normal((m, m)))
    return modes @ np.diag(sigma) @ w.T, modes


def reference_pod(snapshots, mass, n_modes):
    """Another way to the same result: M = L L^T turns the inner product of M into the usual one, so the POD in the
    norm of M is the SVD of L^T S, brought back with L^-T."""
    lower = np.linalg.cholesky(dense(mass))
    u, s, _ = np.linalg.svd(lower.T @ snapshots, full_matrices=False)
    return np.linalg.solve(lower.T, u[:, :n_modes]), s


def projector(basis, mass):
    """P = basis basis^T M, the projection onto the span of the modes (it does not depend on their signs)."""
    return basis @ (basis.T @ dense(mass))


def orthonormality_error(basis, mass):
    return np.abs(basis.T @ dense(mass) @ basis - np.eye(basis.shape[1])).max()


def truncation_error(snapshots, basis, mass):
    """sum_k |s_k - P s_k|^2, with the norm of M."""
    residual = snapshots - projector(basis, mass) @ snapshots
    return float(np.sum(residual * (dense(mass) @ residual)))


MASS_KINDS = [("identity", 20), ("diagonal", 25), ("dense", 30), ("sparse", 40), ("fem", None)]
KIND_IDS = [kind for kind, _ in MASS_KINDS]
SPECTRUM = np.array([1.0, 0.6, 0.35, 0.2, 0.1, 0.05])  # the singular values of the planted problems


@pytest.fixture(params=MASS_KINDS, ids=KIND_IDS)
def problem(request):
    kind, n = request.param
    mass = build_mass(kind, n)
    snapshots, modes = planted(mass, SPECTRUM)
    return mass, snapshots, modes


# ---------------------------------------------------------------- hand examples


def test_hand_example_the_mass_changes_the_answer():
    # two snapshots s1 = (3, 0) and s2 = (0, 1). With M = diag(1, 4): |s1|^2 = 9, |s2|^2 = 4, they are M-orthogonal,
    # so sigma = (3, 2) and the modes are s1 / 3 = (1, 0) and s2 / 2 = (0, 1/2), of norm 4 (1/2)^2 = 1. With M = I the
    # second one counts less: sigma = (3, 1) and the second mode is (0, 1).
    snapshots = np.array([[3.0, 0.0], [0.0, 1.0]])
    basis, sigma = pod(snapshots, np.diag([1.0, 4.0]), 2)
    np.testing.assert_allclose(sigma, [3.0, 2.0], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(np.abs(basis), [[1.0, 0.0], [0.0, 0.5]], rtol=0, atol=1e-14)
    basis, sigma = pod(snapshots, np.eye(2), 2)
    np.testing.assert_allclose(sigma, [3.0, 1.0], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(np.abs(basis), np.eye(2), rtol=0, atol=1e-14)


GOLDEN = (1.0 + math.sqrt(5.0)) / 2.0


def test_hand_example_the_golden_ratio():
    # S = [[1, 1], [1, 0]], M = I. S^T S = [[2, 1], [1, 1]] has the eigenvalues (3 +- sqrt 5) / 2 = 2.618 and 0.382,
    # so sigma = (1.618, 0.618) = (g, 1/g) with the golden ratio g. The eigenvector of 2.618 is (g, 1) / sqrt(g^2 + 1) =
    # (0.8507, 0.5257), the other one is (-0.5257, 0.8507)
    snapshots = np.array([[1.0, 1.0], [1.0, 0.0]])
    basis, sigma = pod(snapshots, np.eye(2), 2)
    np.testing.assert_allclose(sigma, [GOLDEN, 1.0 / GOLDEN], rtol=1e-14, atol=0.0)
    big, small = 0.8506508083520399, 0.5257311121191336
    np.testing.assert_allclose(np.abs(basis), [[big, small], [small, big]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), rtol=0, atol=1e-14)


def test_hand_example_one_mode_leaves_the_second_singular_value_squared():
    # with only the first mode (0.8507, 0.5257): s1 = (1, 1) has the coordinate 1.3764 and leaves the residual
    # (-0.1709, 0.2764), of squared norm 0.1056; s2 = (1, 0) has the coordinate 0.8507 and leaves (0.2764, -0.4472), of
    # squared norm 0.2764. The sum is 0.3820 = (3 - sqrt 5) / 2 = sigma_2^2
    snapshots = np.array([[1.0, 1.0], [1.0, 0.0]])
    basis, sigma = pod(snapshots, np.eye(2), 1)
    assert basis.shape == (2, 1) and sigma.shape == (2,)
    assert truncation_error(snapshots, basis, np.eye(2)) == pytest.approx(0.3819660112501051, rel=1e-12)
    assert truncation_error(snapshots, basis, np.eye(2)) == pytest.approx(sigma[1] ** 2, rel=1e-12)
    first = np.array([0.8506508083520399, 0.5257311121191336])
    assert abs(basis[:, 0] @ first) == pytest.approx(1.0, abs=1e-14)


def test_hand_example_more_snapshots_than_unknowns():
    # three snapshots in the plane: S = [[1, 0, 1], [0, 1, 1]], M = I. The third column is the sum of the others, S^T S
    # = [[1, 0, 1], [0, 1, 1], [1, 1, 2]] has the eigenvalues 3, 1 and 0 and sigma = (sqrt 3, 1, 0). The modes are
    # (1, 1) / sqrt 2 and (1, -1) / sqrt 2. There are only two directions in the plane: three modes are too many.
    snapshots = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    basis, sigma = pod(snapshots, np.eye(2), 2)
    assert sigma.shape == (3,)
    np.testing.assert_allclose(sigma[:2], [math.sqrt(3.0), 1.0], rtol=1e-14, atol=0.0)
    assert sigma[2] <= 1e-6 * sigma[0]
    np.testing.assert_allclose(np.abs(basis), np.full((2, 2), math.sqrt(0.5)), rtol=0, atol=1e-14)
    with pytest.raises(ValueError):
        pod(snapshots, np.eye(2), 3)


def test_hand_example_a_constant_field_has_the_norm_square_root_of_the_area():
    # on a mesh the sum of all the entries of M is the area times rho_c, so the constant field c has |c|^2 = c^2 A
    # rho_c.
    # The rectangle 2 x 3 with rho_c = 1 has A = 6: the snapshots 2 and -1 (everywhere) give a single direction and
    # sigma_1 = sqrt((4 + 1) 6) = sqrt 30, and the mode is the constant 1 / sqrt 6
    mesh = thermomech.rectangle_mesh(2.0, 3.0, 4, 3)
    mass = thermomech.assemble_mass_2d(mesh, 1.0)
    n = mesh.num_nodes
    snapshots = np.column_stack([np.full(n, 2.0), np.full(n, -1.0)])
    basis, sigma = pod(snapshots, mass, 1)
    assert sigma[0] == pytest.approx(math.sqrt(30.0), rel=1e-13)
    assert sigma[1] <= 1e-6 * sigma[0]
    np.testing.assert_allclose(np.abs(basis[:, 0]), np.full(n, 1.0 / math.sqrt(6.0)), rtol=0, atol=1e-13)
    assert len(set(np.sign(basis[:, 0]))) == 1  # one sign everywhere: the mode is the constant, not a mix


# ---------------------------------------------------------------- planted problems


def test_the_singular_values_are_the_planted_ones(problem):
    mass, snapshots, _ = problem
    _, sigma = pod(snapshots, mass, 3)
    np.testing.assert_allclose(sigma, SPECTRUM, rtol=0, atol=1e-12)


@pytest.mark.parametrize("n_modes", range(1, 7))
def test_the_number_of_modes_changes_the_basis_and_not_the_singular_values(problem, n_modes):
    mass, snapshots, _ = problem
    basis, sigma = pod(snapshots, mass, n_modes)
    assert basis.shape == (mass.shape[0], n_modes)
    assert sigma.shape == (6,)
    np.testing.assert_allclose(sigma, SPECTRUM, rtol=0, atol=1e-12)


@pytest.mark.parametrize("n_modes", range(1, 7))
def test_the_modes_are_orthonormal_for_the_mass(problem, n_modes):
    mass, snapshots, _ = problem
    basis, _ = pod(snapshots, mass, n_modes)
    assert orthonormality_error(basis, mass) < 1e-12


@pytest.mark.parametrize("n_modes", range(1, 7))
def test_the_modes_span_the_planted_modes(problem, n_modes):
    mass, snapshots, modes = problem
    basis, _ = pod(snapshots, mass, n_modes)
    np.testing.assert_allclose(projector(basis, mass), projector(modes[:, :n_modes], mass), rtol=0, atol=1e-10)


@pytest.mark.parametrize("n_modes", range(1, 7))
def test_the_error_of_the_truncation_is_the_sum_of_the_discarded_singular_values_squared(problem, n_modes):
    mass, snapshots, _ = problem
    basis, sigma = pod(snapshots, mass, n_modes)
    expected = float(np.sum(sigma[n_modes:] ** 2))
    assert truncation_error(snapshots, basis, mass) == pytest.approx(expected, rel=1e-9, abs=1e-20)


def test_the_first_modes_do_not_depend_on_how_many_are_asked(problem):
    mass, snapshots, _ = problem
    few, _ = pod(snapshots, mass, 2)
    many, _ = pod(snapshots, mass, 5)
    np.testing.assert_allclose(projector(many[:, :2], mass), projector(few, mass), rtol=0, atol=1e-10)
    assert orthonormality_error(many[:, :2], mass) < 1e-12


def test_each_mode_is_the_one_of_its_singular_value(problem):
    # the planted modes are in the order of sigma: the mode i is along the column i of Q (up to the sign)
    mass, snapshots, modes = problem
    basis, _ = pod(snapshots, mass, 6)
    overlaps = np.abs(basis.T @ dense(mass) @ modes)
    np.testing.assert_allclose(overlaps, np.eye(6), rtol=0, atol=1e-9)


# ---------------------------------------------------------------- random snapshots against another algorithm


@pytest.mark.parametrize("kind, n", MASS_KINDS, ids=KIND_IDS)
@pytest.mark.parametrize("n_modes", [1, 4, 9])
def test_random_snapshots_agree_with_the_cholesky_svd_algorithm(kind, n, n_modes):
    mass = build_mass(kind, n)
    snapshots = np.random.default_rng(3).standard_normal((mass.shape[0], 9))
    basis, sigma = pod(snapshots, mass, n_modes)
    expected_basis, expected_sigma = reference_pod(snapshots, mass, n_modes)
    np.testing.assert_allclose(sigma, expected_sigma, rtol=1e-10, atol=0)
    np.testing.assert_allclose(projector(basis, mass), projector(expected_basis, mass), rtol=0, atol=1e-9)
    assert orthonormality_error(basis, mass) < 1e-12


# ---------------------------------------------------------------- what does not change the answer


@pytest.mark.parametrize("factor", [-2.5, 1e-3, 1e5])
def test_scaling_the_snapshots_scales_the_singular_values_and_keeps_the_modes(problem, factor):
    mass, snapshots, _ = problem
    basis, sigma = pod(snapshots, mass, 4)
    scaled_basis, scaled_sigma = pod(factor * snapshots, mass, 4)
    np.testing.assert_allclose(scaled_sigma, abs(factor) * sigma, rtol=1e-10, atol=0)
    np.testing.assert_allclose(projector(scaled_basis, mass), projector(basis, mass), rtol=0, atol=1e-9)
    assert orthonormality_error(scaled_basis, mass) < 1e-12


@pytest.mark.parametrize("factor", [0.01, 7.0])
def test_scaling_the_mass_scales_the_norm(problem, factor):
    # |u|^2 is multiplied by the factor: sigma by its square root, and the modes, which have norm 1, are divided by it
    mass, snapshots, _ = problem
    basis, sigma = pod(snapshots, mass, 4)
    scaled_basis, scaled_sigma = pod(snapshots, factor * mass, 4)
    np.testing.assert_allclose(scaled_sigma, math.sqrt(factor) * sigma, rtol=1e-10, atol=0)
    np.testing.assert_allclose(projector(scaled_basis, factor * mass), projector(basis, mass), rtol=0, atol=1e-9)
    assert orthonormality_error(scaled_basis, factor * mass) < 1e-12
    np.testing.assert_allclose(np.abs(scaled_basis.T @ dense(mass) @ basis) * math.sqrt(factor), np.eye(4), rtol=0,
                               atol=1e-9)


def test_the_order_and_the_signs_of_the_snapshots_do_not_matter(problem):
    mass, snapshots, _ = problem
    basis, sigma = pod(snapshots, mass, 4)
    changed = snapshots[:, [3, 0, 5, 1, 4, 2]] * np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0])
    new_basis, new_sigma = pod(changed, mass, 4)
    np.testing.assert_allclose(new_sigma, sigma, rtol=0, atol=1e-12)
    np.testing.assert_allclose(projector(new_basis, mass), projector(basis, mass), rtol=0, atol=1e-10)


# ---------------------------------------------------------------- not enough independent snapshots


@pytest.mark.parametrize("kind, n", MASS_KINDS, ids=KIND_IDS)
def test_a_repeated_snapshot_and_a_zero_snapshot_add_no_mode(kind, n):
    mass = build_mass(kind, n)
    snapshots, modes = planted(mass, SPECTRUM)
    padded = np.column_stack([snapshots, snapshots[:, 0], np.zeros(mass.shape[0])])  # 8 columns of rank 6
    basis, sigma = pod(padded, mass, 6)
    assert sigma.shape == (8,)
    assert np.all(sigma[6:] <= 1e-6 * sigma[0])
    np.testing.assert_allclose(projector(basis, mass), projector(modes, mass), rtol=0, atol=1e-8)
    assert orthonormality_error(basis, mass) < 1e-12
    for too_many in (7, 8):
        with pytest.raises(ValueError):
            pod(padded, mass, too_many)


def test_snapshots_that_are_all_zero_have_no_mode():
    with pytest.raises(ValueError):
        pod(np.zeros((5, 3)), np.eye(5), 1)


def test_a_single_nonzero_snapshot_gives_one_mode():
    snapshots = np.zeros((4, 3))
    snapshots[:, 1] = [1.0, 2.0, 2.0, 0.0]  # |s|^2 = 9 with M = I
    basis, sigma = pod(snapshots, np.eye(4), 1)
    np.testing.assert_allclose(sigma, [3.0, 0.0, 0.0], rtol=0, atol=3e-6)  # 1e-6 sigma_1
    np.testing.assert_allclose(np.abs(basis[:, 0]), [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.0], rtol=0, atol=1e-14)
    with pytest.raises(ValueError):
        pod(snapshots, np.eye(4), 2)


# ---------------------------------------------------------------- singular values that decay fast


@pytest.mark.parametrize("kind, n", [("dense", 40), ("diagonal", 30), ("fem", None)])
def test_the_modes_stay_orthonormal_when_the_singular_values_fall_to_3e_minus_6(kind, n):
    # sigma from 1 down to 3e-6 in five steps of a factor 12.6. The matrix S^T M S has
    # eigenvalues down to 1e-11 and its eigenvectors are known to about 1e-16 / 1e-11 = 1e-5: modes computed with
    # them alone are not orthonormal to better than 1e-5, and one correction step of the wrong kind leaves 1e-10
    mass = build_mass(kind, n)
    sigma_true = 10.0 ** -np.linspace(0.0, 5.5, 6)
    snapshots, modes = planted(mass, sigma_true)
    basis, sigma = pod(snapshots, mass, 6)
    assert orthonormality_error(basis, mass) < 1e-12
    np.testing.assert_allclose(sigma, sigma_true, rtol=0, atol=1e-9)
    np.testing.assert_allclose(projector(basis, mass), projector(modes, mass), rtol=0, atol=1e-8)


# ---------------------------------------------------------------- the cases of a finite element mesh


def border_nodes(nx, ny):
    return [j * (nx + 1) + i for j in range(ny + 1) for i in range(nx + 1) if i in (0, nx) or j in (0, ny)]


def test_a_row_that_is_zero_in_every_snapshot_is_zero_in_the_basis():
    # the nodes of the border are at a fixed temperature: the snapshots are zero there, and so are the modes. The
    # problem is then the one of the inner nodes alone, with the inner block of M.
    nx, ny = 6, 4
    mesh = thermomech.rectangle_mesh(3.0, 2.0, nx, ny)
    mass = thermomech.assemble_mass_2d(mesh, 1.0)
    n = mesh.num_nodes
    border = border_nodes(nx, ny)
    inner = np.setdiff1d(np.arange(n), border)
    snapshots = np.random.default_rng(8).standard_normal((n, 7))
    snapshots[border, :] = 0.0
    basis, sigma = pod(snapshots, mass, 4)
    assert np.abs(basis[border]).max() <= 1e-14
    inner_mass = dense(mass)[np.ix_(inner, inner)]
    inner_basis, inner_sigma = pod(snapshots[inner], inner_mass, 4)
    np.testing.assert_allclose(sigma, inner_sigma, rtol=1e-10, atol=0)
    np.testing.assert_allclose(basis[inner] @ basis[inner].T @ inner_mass,
                               inner_basis @ inner_basis.T @ inner_mass, rtol=0, atol=1e-9)


def test_two_sine_modes_on_a_mesh_are_found_exactly():
    # snapshots a1 phi1 + a2 phi2 with phi1 = sin(pi x / 2) sin(pi y), phi2 = sin(pi x) sin(pi y) on the rectangle 2 x 1
    # and the amplitudes a1 = 0.7^k, a2 = 0.4^k: the snapshots have rank 2 and the modes span {phi1, phi2}
    mesh = thermomech.rectangle_mesh(2.0, 1.0, 6, 4)
    mass = thermomech.assemble_mass_2d(mesh, 1.0)
    phi1 = np.sin(PI * mesh.x / 2.0) * np.sin(PI * mesh.y)
    phi2 = np.sin(PI * mesh.x) * np.sin(PI * mesh.y)
    snapshots = np.column_stack([0.7**k * phi1 + 0.4**k * phi2 for k in range(8)])
    basis, sigma = pod(snapshots, mass, 2)
    assert np.all(sigma[2:] <= 1e-6 * sigma[0])
    M = dense(mass)
    for phi in (phi1, phi2):
        residual = phi - basis @ (basis.T @ (M @ phi))
        assert math.sqrt(residual @ M @ residual) <= 1e-10 * math.sqrt(phi @ M @ phi)
    with pytest.raises(ValueError):
        pod(snapshots, mass, 3)


# ---------------------------------------------------------------- the arguments and the results


def test_the_result_is_a_pair_of_new_float64_arrays(problem):
    mass, snapshots, _ = problem
    result = pod(snapshots, mass, 3)
    assert isinstance(result, tuple) and len(result) == 2
    basis, sigma = result
    for array in result:
        assert isinstance(array, np.ndarray) and array.dtype == np.float64 and array.flags["WRITEABLE"]
        assert not np.shares_memory(array, snapshots)
    assert not np.shares_memory(basis, sigma)
    assert basis.shape == (mass.shape[0], 3) and sigma.shape == (6,)
    assert np.all(np.isfinite(basis)) and np.all(np.diff(sigma) <= 0.0) and np.all(sigma >= 0.0)


def test_the_arguments_are_not_modified(problem):
    mass, snapshots, _ = problem
    snapshots_before = snapshots.copy()
    mass_before = mass.copy()
    pod(snapshots, mass, 4)
    np.testing.assert_array_equal(snapshots, snapshots_before)
    np.testing.assert_array_equal(dense(mass), dense(mass_before))


def test_the_arguments_can_be_given_by_name():
    snapshots = np.array([[3.0, 0.0], [0.0, 1.0]])
    by_name = pod(snapshots=snapshots, mass=np.diag([1.0, 4.0]), n_modes=2)
    by_position = pod(snapshots, np.diag([1.0, 4.0]), 2)
    for first, second in zip(by_name, by_position):
        np.testing.assert_array_equal(first, second)


def every_second(values):
    wide = np.zeros((values.shape[0], 2 * values.shape[1]))
    wide[:, ::2] = values
    return wide[:, ::2]  # neighbouring columns are two columns apart in memory


def backwards(values):
    return values[::-1, ::-1].copy()[::-1, ::-1]  # negative strides in both directions


def read_only(values):
    values = values.copy()
    values.setflags(write=False)
    return values


def fortran(values):
    return np.asfortranarray(values)


def nested_lists(values):
    return values.tolist()


SNAPSHOT_LAYOUTS = [every_second, backwards, read_only, fortran, nested_lists]


@pytest.mark.parametrize("layout", SNAPSHOT_LAYOUTS, ids=[layout.__name__ for layout in SNAPSHOT_LAYOUTS])
def test_snapshots_of_any_layout_give_the_same_answer(problem, layout):
    mass, snapshots, _ = problem
    basis, sigma = pod(snapshots, mass, 4)
    other_basis, other_sigma = pod(layout(snapshots), mass, 4)
    np.testing.assert_allclose(other_sigma, sigma, rtol=0, atol=1e-13)
    np.testing.assert_allclose(projector(other_basis, mass), projector(basis, mass), rtol=0, atol=1e-12)


def test_snapshots_of_other_number_types_give_the_same_answer():
    # integers, and float32 numbers that are exactly representable (multiples of 1/8)
    integers = np.array([[3, 0, 1], [0, 1, 2], [1, 1, 1], [2, 0, 5]])
    for variant in (integers, integers.astype(np.int32), integers.astype(np.int8), integers.astype(np.float32),
                    integers.astype(float)):
        basis, sigma = pod(variant, np.eye(4), 2)
        expected_basis, expected_sigma = pod(integers.astype(float), np.eye(4), 2)
        np.testing.assert_allclose(sigma, expected_sigma, rtol=0, atol=1e-13)
        np.testing.assert_allclose(projector(basis, np.eye(4)), projector(expected_basis, np.eye(4)), rtol=0,
                                   atol=1e-12)
    eighths = np.random.default_rng(2).integers(-40, 40, size=(6, 4)) / 8.0
    basis, sigma = pod(eighths.astype(np.float32), np.eye(6), 3)
    expected_basis, expected_sigma = pod(eighths, np.eye(6), 3)
    np.testing.assert_allclose(sigma, expected_sigma, rtol=0, atol=1e-13)
    np.testing.assert_allclose(projector(basis, np.eye(6)), projector(expected_basis, np.eye(6)), rtol=0, atol=1e-12)


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo", "dense"])
def test_the_mass_can_be_a_dense_or_a_sparse_matrix_of_any_format(fmt):
    sparse_mass = build_mass("sparse", 40)
    snapshots, _ = planted(sparse_mass, SPECTRUM)
    mass = dense(sparse_mass) if fmt == "dense" else sparse_mass.asformat(fmt)
    basis, sigma = pod(snapshots, mass, 4)
    expected_basis, expected_sigma = pod(snapshots, sparse_mass, 4)
    np.testing.assert_allclose(sigma, expected_sigma, rtol=0, atol=1e-13)
    np.testing.assert_allclose(projector(basis, sparse_mass), projector(expected_basis, sparse_mass), rtol=0,
                               atol=1e-12)


@pytest.mark.parametrize("make", [np.int8, np.int32, np.int64, np.uint8, np.uint64])
def test_n_modes_can_be_a_numpy_integer(make):
    snapshots = np.array([[3.0, 0.0], [0.0, 1.0]])
    basis, sigma = pod(snapshots, np.eye(2), make(1))
    assert basis.shape == (2, 1)


# ---------------------------------------------------------------- wrong arguments


def a_good_problem(**change):
    arguments = dict(snapshots=np.random.default_rng(4).standard_normal((6, 4)), mass=np.eye(6), n_modes=2)
    arguments.update(change)
    return arguments


NOT_FINITE = [("a NaN", np.nan), ("an infinity", np.inf), ("minus infinity", -np.inf)]


@pytest.mark.parametrize("text, bad", NOT_FINITE, ids=[text for text, _ in NOT_FINITE])
def test_a_snapshot_that_is_not_finite_is_a_value_error(text, bad):
    snapshots = np.random.default_rng(4).standard_normal((6, 4))
    snapshots[2, 1] = bad
    with pytest.raises(ValueError):
        pod(**a_good_problem(snapshots=snapshots))


BAD_SHAPES = {
    "snapshots is one-dimensional": dict(snapshots=np.ones(6)),
    "snapshots is three-dimensional": dict(snapshots=np.ones((6, 4, 1))),
    "snapshots is a number": dict(snapshots=np.array(3.0)),
    "no snapshots": dict(snapshots=np.zeros((6, 0))),
    "no nodes": dict(snapshots=np.zeros((0, 4)), mass=np.zeros((0, 0))),
    "mass has too few rows": dict(mass=np.eye(5)),
    "mass has too many rows": dict(mass=np.eye(7)),
    "mass is not square": dict(mass=np.ones((6, 5))),
    "mass is one-dimensional": dict(mass=np.ones(6)),
    "mass is a sparse matrix of the wrong size": dict(mass=sp.identity(5, format="csr")),
    "n_modes is zero": dict(n_modes=0),
    "n_modes is negative": dict(n_modes=-1),
    "n_modes is more than the snapshots": dict(n_modes=5),
    "n_modes is huge": dict(n_modes=10**6),
}


@pytest.mark.parametrize("change", list(BAD_SHAPES.values()), ids=list(BAD_SHAPES))
def test_arguments_that_do_not_fit_are_a_value_error(change):
    with pytest.raises(ValueError):
        pod(**a_good_problem(**change))


@pytest.mark.parametrize("n_modes", [2.0, 2.5, "2", None, [2], (2,), 1j], ids=lambda value: repr(value))
def test_n_modes_that_is_not_an_integer_is_a_type_error(n_modes):
    with pytest.raises(TypeError):
        pod(**a_good_problem(n_modes=n_modes))
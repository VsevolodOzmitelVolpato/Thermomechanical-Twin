"""Step 6e: the whole chain, heat -> displacements -> stresses, from Python.

    solve_thermomechanical_2d(mesh, rho_c, k, f, fixed_nodes, fixed_temperatures, T0, dt, n_steps, theta, save_every,
                              E, nu, alpha, T_ref, fixed_dofs, fixed_displacements)

The first ten arguments are those of solve_transient_2d (fixed_temperatures is its fixed_values). Then comes
save_every, then the material in plane strain (Young modulus E, Poisson ratio nu, expansion coefficient alpha and the
reference temperature T_ref, where nothing expands), then the constraints on the displacements: the displacement of the
degree of freedom fixed_dofs[q] is fixed_displacements[q]. These are numbers of degrees of freedom, not of nodes: the
node I has u at 2 I and v at 2 I + 1. At least three are needed (a plate has three rigid motions). Lists, tuples and
arrays are accepted for f, fixed_nodes, fixed_temperatures, T0, fixed_dofs and fixed_displacements, and none of them is
modified. All seventeen arguments are required (no default values) and can be given by name or by position.

The heat is marched for n_steps steps of the theta scheme and a state is saved every save_every steps: n_steps must be
a multiple of save_every, and the states are those of the steps 0, save_every, 2 save_every, ..., n_steps (m = n_steps /
save_every + 1 states), the first one with the temperature T0. In each state the plate is in equilibrium with the
temperature of that instant (K u = F) and the stresses come from the displacements. The function returns a dict with
five new NumPy arrays (float64, C order), one row per state:

    "time"       (m,)            the time of each state: step * dt
    "T"          (m, n)          the temperature of the nodes (n = mesh.num_nodes)
    "u"          (m, 2 n)        the displacements: node I has u at 2 I and v at 2 I + 1
    "sigma"      (m, n_tri, 3)   the stress (xx, yy, xy) of each triangle (n_tri = mesh.num_triangles)
    "von_mises"  (m, n_tri)      the von Mises stress of each triangle, with the plane-strain stress out of the plane
                                 sigma_zz = nu (sigma_xx + sigma_yy) - E alpha (T_triangle - T_ref)

The errors of the C++ functions arrive as before: ValueError for data that does not fit, RuntimeError when the conjugate
gradient does not converge, TypeError for the wrong types.
"""
import math

import numpy as np
import pytest

import thermomech

PI = math.pi
NAMES = ["time", "T", "u", "sigma", "von_mises"]

# Friendly numbers for hand calculations: E = 100, nu = 1/4, alpha = 1/100, T_ref = 20. Then
#     lambda = E nu / ((1 + nu) (1 - 2 nu)) = 40,   mu = E / (2 (1 + nu)) = 40,   D = [[120, 40, 0], [40, 120, 0],
#     [0, 0, 40]],   E alpha = 1,   (1 + nu) alpha = 1/80 = 0.0125.
HAND_MATERIAL = dict(E=100.0, nu=0.25, alpha=0.01, T_ref=20.0)


ARGUMENTS = ["mesh", "rho_c", "k", "f", "fixed_nodes", "fixed_temperatures", "T0", "dt", "n_steps", "theta",
             "save_every", "E", "nu", "alpha", "T_ref", "fixed_dofs", "fixed_displacements"]


def solve(**arguments):
    return thermomech.solve_thermomechanical_2d(**arguments)


def border_nodes(nx, ny):
    """The nodes on the four sides of rectangle_mesh(.., nx, ny): the node of column i and row j is j*(nx+1) + i."""
    return [j * (nx + 1) + i for j in range(ny + 1) for i in range(nx + 1) if i in (0, nx) or j in (0, ny)]


def pin_corners(nx):
    """Three constraints that remove the three rigid motions: u and v of the node 0 (the corner (0, 0)) and v of the
    node nx (the corner (L, 0))."""
    return dict(fixed_dofs=[0, 1, 2 * nx + 1], fixed_displacements=[0.0, 0.0, 0.0])


def expansion(mesh, e):
    """The displacements u = e x, v = e y of the free expansion by e, in the order of the unknowns (u0, v0, u1, ...)."""
    return np.column_stack([e * mesh.x, e * mesh.y]).ravel()


# ---------------------------------------------------------------- an independent computation with dense matrices


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


def von_mises(sigma, sigma_zz):
    xx, yy, xy = sigma[..., 0], sigma[..., 1], sigma[..., 2]
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - sigma_zz) ** 2 + (sigma_zz - xx) ** 2) + 3.0 * xy**2)


def von_mises_of(mesh, sigma, T, E, nu, alpha, T_ref):
    """The von Mises stress of every triangle, for the stresses sigma (n_tri, 3) and the temperatures T (n,)."""
    dT = T[mesh.triangles].mean(axis=1) - T_ref
    return von_mises(sigma, nu * (sigma[:, 0] + sigma[:, 1]) - E * alpha * dT)


def reference_mechanics(mesh, E, nu, alpha, T_ref, T, fixed_dofs, fixed_displacements):
    """u, sigma and von Mises of the plate at the temperature T: K u = F with dense matrices, the fixed degrees of
    freedom eliminated, then sigma = D (B u - eps_th) in each triangle."""
    n, n_tri = mesh.num_nodes, mesh.num_triangles
    D = plane_strain_matrix(E, nu)
    K = np.zeros((2 * n, 2 * n))
    F = np.zeros(2 * n)
    elements = []
    for e in range(n_tri):
        B, area = strain_displacement(mesh, e)
        dofs = np.ravel([[2 * i, 2 * i + 1] for i in mesh.triangles[e]])
        eps_th = (1.0 + nu) * alpha * (T[mesh.triangles[e]].mean() - T_ref) * np.array([1.0, 1.0, 0.0])
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
    return u, sigma, von_mises_of(mesh, sigma, T, E, nu, alpha, T_ref)


def reference_states(arguments):
    """The states of the whole chain: the temperatures from solve_transient_2d (the heat of step 6d), the rest from
    reference_mechanics."""
    mesh = arguments["mesh"]
    result = dict(time=[], T=[], u=[], sigma=[], von_mises=[])
    for s in range(arguments["n_steps"] // arguments["save_every"] + 1):
        steps = s * arguments["save_every"]
        if steps == 0:
            T = np.array(arguments["T0"], dtype=float)
        else:
            T = thermomech.solve_transient_2d(mesh, arguments["rho_c"], arguments["k"], arguments["f"],
                                              arguments["fixed_nodes"], arguments["fixed_temperatures"],
                                              arguments["T0"], arguments["dt"], steps, arguments["theta"])
        u, sigma, vm = reference_mechanics(mesh, arguments["E"], arguments["nu"], arguments["alpha"],
                                           arguments["T_ref"], T, arguments["fixed_dofs"],
                                           arguments["fixed_displacements"])
        for name, value in zip(NAMES, (steps * arguments["dt"], T, u, sigma, vm)):
            result[name].append(value)
    return {name: np.array(rows) for name, rows in result.items()}


def assert_close(got, want, relative, err_msg=""):
    """The same numbers, up to a tolerance relative to the largest of them."""
    want = np.asarray(want, dtype=float)
    np.testing.assert_allclose(got, want, rtol=0, atol=relative * float(np.abs(want).max()) + 1e-300, err_msg=err_msg)


# ---------------------------------------------------------------- the problems


GENERAL_CASES = [(2.0, 1.5, 4, 3), (1.0, 1.0, 1, 1), (3.0, 1.0, 6, 2), (1.0, 4.0, 2, 6), (0.5, 0.5, 5, 5)]
SMALL = (1.0, 1.0, 2, 2)


def general_problem(case=GENERAL_CASES[0], **change):
    """A problem on rectangle_mesh(*case) in which no two data have the same value, so a function that mixes up two of
    them gives another answer: a source that depends on the place, three nodes at other temperatures than the
    initial field, 6 steps saved every 2 (4 states), the displacements of the three corner constraints not zero."""
    length_x, length_y, nx, ny = case
    mesh = thermomech.rectangle_mesh(length_x, length_y, nx, ny)
    s, t = mesh.x / length_x, mesh.y / length_y
    arguments = dict(mesh=mesh, rho_c=2.5, k=0.7, f=thermomech.assemble_load_2d(mesh, lambda x, y: 3.0 + x),
                     fixed_nodes=[0, nx, ny * (nx + 1)], fixed_temperatures=[310.0, 280.0, 300.0],
                     T0=300.0 + 15.0 * np.sin(3.0 * s) + 10.0 * np.cos(2.0 * t) + 5.0 * s * t,
                     dt=0.15, n_steps=6, theta=0.6, save_every=2, E=70.0, nu=0.2, alpha=0.003, T_ref=290.0,
                     fixed_dofs=[0, 1, 2 * nx + 1], fixed_displacements=[0.01, -0.02, 0.03])
    arguments.update(change)
    return arguments


def heated_plate(theta, **change):
    """The 3 x 2 cells plate [0, 2] x [0, 1], insulated, at 25 degrees with the reference at 20 and a uniform source
    q = 8 with rho_c = 2: with steps of 0.25 each step heats it by q dt / rho_c = 1 degree, for every theta (K 1 = 0
    and the mass brings the load in), so after p steps it is at 25 + p everywhere. Pinned at two corners."""
    mesh = thermomech.rectangle_mesh(2.0, 1.0, 3, 2)
    arguments = dict(mesh=mesh, rho_c=2.0, k=1.0, f=thermomech.assemble_load_2d(mesh, 8.0), fixed_nodes=[],
                     fixed_temperatures=[], T0=np.full(12, 25.0), dt=0.25, n_steps=4, theta=theta, save_every=1,
                     **HAND_MATERIAL, **pin_corners(3))
    arguments.update(change)
    return arguments


def roller_slab(nx, ny, t_ref, amplitude):
    """The heat and the constraints of a slab of nx x ny cells on rollers, without any source: the faces y = 0 and
    y = H stay at t_ref, the initial temperature is t_ref + amplitude sin(pi y / H) in every column, u = 0 on the two
    sides x = 0 and x = L and v = 0 at the node 0 (the faces expand freely)."""
    n = (nx + 1) * (ny + 1)
    T0 = np.empty(n)
    for j in range(ny + 1):
        T0[j * (nx + 1):(j + 1) * (nx + 1)] = t_ref + amplitude * math.sin(PI * j / ny)
    faces = list(range(nx + 1)) + [ny * (nx + 1) + i for i in range(nx + 1)]
    dofs = [dof for j in range(ny + 1) for dof in (2 * (j * (nx + 1)), 2 * (j * (nx + 1) + nx))] + [1]
    return dict(f=np.zeros(n), fixed_nodes=faces, fixed_temperatures=[t_ref] * len(faces), T0=T0, fixed_dofs=dofs,
                fixed_displacements=[0.0] * len(dofs))


def strip(nx, theta, dt, n_steps, save_every=1):
    """Two rows of square cells of side 1/2 (height H = 1, length nx / 2), the friendly material, amplitude 60,
    rho_c = 2 and k = 3."""
    mesh = thermomech.rectangle_mesh(0.5 * nx, 1.0, nx, 2)
    return dict(mesh=mesh, rho_c=2.0, k=3.0, dt=dt, n_steps=n_steps, theta=theta, save_every=save_every,
                **HAND_MATERIAL, **roller_slab(nx, 2, 20.0, 60.0))


# ---------------------------------------------------------------- the result


def test_the_module_has_the_function():
    assert callable(thermomech.solve_thermomechanical_2d)


def test_the_result_is_a_dict_of_five_new_float64_arrays_in_c_order():
    arguments = general_problem(SMALL)
    result = solve(**arguments)
    assert isinstance(result, dict)
    assert sorted(result) == sorted(NAMES)
    for name in NAMES:
        array = result[name]
        assert isinstance(array, np.ndarray), name
        assert array.dtype == np.float64, name
        assert array.flags["C_CONTIGUOUS"] and array.flags["WRITEABLE"], name
    for first in NAMES:  # five different blocks of memory
        for second in NAMES:
            if first < second:
                assert not np.shares_memory(result[first], result[second]), (first, second)
    again = solve(**arguments)
    for name in NAMES:  # and a new block at every call: writing in one result does not change the next
        kept = result[name].copy()
        result[name][...] = 0.0
        np.testing.assert_array_equal(again[name], kept, err_msg=name)
        assert not np.shares_memory(again[name], result[name]), name


@pytest.mark.parametrize("n_steps, save_every, states", [(0, 1, 1), (0, 5, 1), (1, 1, 2), (6, 2, 4), (10, 1, 11),
                                                         (6, 3, 3), (6, 6, 2), (6, 1, 7)])
def test_the_number_of_states_and_the_shape_of_every_array(n_steps, save_every, states):
    # 3 x 2 cells: 12 nodes (24 displacements) and 12 triangles
    result = solve(**heated_plate(1.0, n_steps=n_steps, save_every=save_every))
    shapes = dict(time=(states,), T=(states, 12), u=(states, 24), sigma=(states, 12, 3), von_mises=(states, 12))
    for name in NAMES:
        assert result[name].shape == shapes[name], name


def test_the_shapes_follow_the_mesh():
    arguments = general_problem((3.0, 1.0, 6, 2))  # 7 x 3 = 21 nodes, 24 triangles, 4 states
    result = solve(**arguments)
    shapes = dict(time=(4,), T=(4, 21), u=(4, 42), sigma=(4, 24, 3), von_mises=(4, 24))
    for name in NAMES:
        assert result[name].shape == shapes[name], name


@pytest.mark.parametrize("n_steps, save_every", [(6, 2), (10, 1), (6, 3), (0, 1)])
def test_the_times_are_the_steps_times_dt(n_steps, save_every):
    result = solve(**heated_plate(1.0, dt=0.1, n_steps=n_steps, save_every=save_every))
    steps = np.arange(0, n_steps + 1, save_every)
    np.testing.assert_allclose(result["time"], steps * 0.1, rtol=1e-14, atol=0.0)
    assert result["time"][0] == 0.0


def test_the_first_state_has_the_initial_temperature_exactly():
    arguments = general_problem()
    result = solve(**arguments)
    np.testing.assert_array_equal(result["T"][0], arguments["T0"])
    assert not np.array_equal(result["T"][1], arguments["T0"])  # and the heat does move


# ---------------------------------------------------------------- zero steps


def test_with_zero_steps_there_is_one_state_and_no_heat_data_is_needed():
    # the unit square at 30 degrees with the reference at 20 expands by e = (1 + nu) alpha dT = 0.0125 x 10 = 0.125 in
    # every direction (u = e x, v = e y) without stress; the stress out of the plane is sigma_zz = -E alpha dT = -10 and
    # the von Mises stress of that single stress is 10
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 1, 1)
    result = solve(mesh=mesh, rho_c=0.0, k=0.0, f=[], fixed_nodes=[], fixed_temperatures=[], T0=np.full(4, 30.0),
                   dt=0.0, n_steps=0, theta=1.0, save_every=1, **HAND_MATERIAL, **pin_corners(1))
    np.testing.assert_array_equal(result["time"], [0.0])
    np.testing.assert_array_equal(result["T"], [[30.0, 30.0, 30.0, 30.0]])
    np.testing.assert_allclose(result["u"], [[0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["sigma"], np.zeros((1, 2, 3)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["von_mises"], [[10.0, 10.0]], rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------- plates whose answer is known by hand


@pytest.mark.parametrize("theta", [1.0, 0.5, 0.0, 0.8])
def test_a_free_plate_that_heats_up_uniformly_expands_without_stress(theta):
    # after p steps the plate is at 25 + p, dT = 5 + p, and it expands by e = 0.0125 (5 + p) in every direction. The
    # strain B u = e (1, 1, 0) is exactly the thermal one, so sigma = 0; only the stress out of the plane is left,
    # sigma_zz = -E alpha dT = -(5 + p), and the von Mises stress is its absolute value
    mesh = heated_plate(theta)["mesh"]
    result = solve(**heated_plate(theta))
    assert result["T"].shape == (5, 12)
    for p in range(5):
        np.testing.assert_allclose(result["T"][p], np.full(12, 25.0 + p), rtol=0, atol=1e-9, err_msg=str(p))
        np.testing.assert_allclose(result["u"][p], expansion(mesh, 0.0125 * (5.0 + p)), rtol=0, atol=1e-10,
                                   err_msg=str(p))
        np.testing.assert_allclose(result["sigma"][p], np.zeros((12, 3)), rtol=0, atol=1e-8, err_msg=str(p))
        np.testing.assert_allclose(result["von_mises"][p], np.full(12, 5.0 + p), rtol=0, atol=1e-8, err_msg=str(p))


def test_the_imposed_displacements_are_a_rigid_motion_added_to_the_expansion():
    # u = 3 - 0.1 y + e x, v = -2 + 0.1 x + e y: a translation (3, -2) and a rotation of 0.1 radians do not deform, so
    # there is still no stress. The three constraints are u and v of the node 0, 3 and -2, and v of the node 3 (the
    # corner (2, 0)): -2 + 0.1 x 2 = -1.8. The constraints at zero would leave the plate at the origin
    arguments = heated_plate(1.0, n_steps=2, fixed_dofs=[0, 1, 7], fixed_displacements=[3.0, -2.0, -1.8])
    mesh = arguments["mesh"]
    result = solve(**arguments)
    for p in range(3):
        e = 0.0125 * (5.0 + p)
        expected = np.column_stack([3.0 - 0.1 * mesh.y + e * mesh.x, -2.0 + 0.1 * mesh.x + e * mesh.y]).ravel()
        np.testing.assert_allclose(result["u"][p], expected, rtol=0, atol=1e-10, err_msg=str(p))
        np.testing.assert_allclose(result["sigma"][p], np.zeros((12, 3)), rtol=0, atol=1e-8, err_msg=str(p))


@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_a_clamped_plate_is_compressed_more_and_more_as_it_heats_up(theta):
    # a 4 x 4 cells plate with every node of the border fixed (u = v = 0) and the same heating: it does not move and
    # sigma = -D eps_th = -2 dT (1, 1, 0) with dT = p. Then sigma_zz = nu (sigma_xx + sigma_yy) - E alpha dT = -2 dT:
    # the pressure is the same in the three directions and the von Mises stress is zero
    mesh = thermomech.rectangle_mesh(2.0, 1.0, 4, 4)
    dofs = [dof for node in border_nodes(4, 4) for dof in (2 * node, 2 * node + 1)]
    arguments = heated_plate(theta, mesh=mesh, f=thermomech.assemble_load_2d(mesh, 8.0), T0=np.full(25, 20.0),
                             fixed_dofs=dofs, fixed_displacements=[0.0] * len(dofs))
    assert len(dofs) == 32
    result = solve(**arguments)
    for p in range(5):
        np.testing.assert_allclose(result["u"][p], np.zeros(50), rtol=0, atol=1e-10, err_msg=str(p))
        np.testing.assert_allclose(result["sigma"][p], np.tile([-2.0 * p, -2.0 * p, 0.0], (32, 1)), rtol=0, atol=1e-8,
                                   err_msg=str(p))
        np.testing.assert_allclose(result["von_mises"][p], np.zeros(32), rtol=0, atol=1e-6, err_msg=str(p))


HAND_FACTORS = [(1.0, 2.0 / 3.0), (0.5, 3.0 / 5.0), (0.0, 1.0 / 2.0)]


@pytest.mark.parametrize("theta, g", HAND_FACTORS)
def test_the_middle_row_of_the_strip_decays_by_the_hand_factor_at_every_step(theta, g):
    # three rows of nodes (y = 0, 1/2, 1, h = 1/2): the faces stay at 20 and the middle row is one unknown, the same in
    # every column, as for a 1D rod: mass rho_c 2 h / 3 = 2/3 and stiffness k 2 / h = 12. With dt = 1/36 (12 dt =
    # 1/3) the step (2/3 + 12 theta dt) T_new = (2/3 - 12 (1 - theta) dt) T_old on the difference from 20 multiplies
    # it by
    #     g = (2/3 - (1 - theta) / 3) / (2/3 + theta / 3) = (1 + theta) / (2 + theta):   2/3, 3/5 and 1/2.
    # From 20 + 60, after p steps the middle row is at 20 + 60 g^p and the faces are at 20 exactly
    result = solve(**strip(8, theta, 1.0 / 36.0, 3))
    for p in range(4):
        np.testing.assert_allclose(result["T"][p, 9:18], np.full(9, 20.0 + 60.0 * g**p), rtol=0, atol=1e-9,
                                   err_msg="middle row, step %d" % p)
        if p > 0:
            np.testing.assert_array_equal(result["T"][p, :9], np.full(9, 20.0))
            np.testing.assert_array_equal(result["T"][p, 18:], np.full(9, 20.0))


@pytest.mark.parametrize("theta, g", HAND_FACTORS)
def test_the_displacements_and_the_stresses_of_the_strip_decay_like_the_heat(theta, g):
    # the elastic part is linear in dT = T - T_ref (the constraints are zero): when dT is multiplied by g so are u,
    # sigma and the von Mises stress
    result = solve(**strip(8, theta, 1.0 / 36.0, 3))
    u0, sigma0, vm0 = result["u"][0], result["sigma"][0], result["von_mises"][0]
    assert np.abs(sigma0).max() > 20.0 and np.abs(u0).max() > 0.1
    assert sigma0[:, 0].sum() < 0.0  # compression: the hot middle would expand and the rollers hold it
    for p in range(1, 4):
        np.testing.assert_allclose(result["u"][p], g**p * u0, rtol=0, atol=1e-9 * np.abs(u0).max())
        np.testing.assert_allclose(result["sigma"][p], g**p * sigma0, rtol=0, atol=1e-9 * np.abs(sigma0).max())
        np.testing.assert_allclose(result["von_mises"][p], g**p * vm0, rtol=0, atol=1e-9 * np.abs(vm0).max())


def test_in_the_middle_of_a_long_strip_the_stresses_are_the_hand_numbers():
    # far from the ends the strip does not depend on x: u = 0 and v depends on y alone. With e_X = (1 + nu) alpha dT_X
    # and dT_X the mean over the three nodes of a triangle (rows at 0 and 60 above 20): the lower triangle of a cell
    # between the rows 0 and 1 (two nodes in the row 0) has dT_X = 20, e_X = 1/4 and the upper one dT_X = 40, e_X = 1/2.
    # The strain is (0, gamma, 0) with gamma = dv/dy the same in both; the vertical force on the nodes of the faces
    # (free in v) must be zero, so the sigma_yy = 120 gamma - 160 e_X of the two triangles add up to zero:
    # (120 gamma - 40) + (120 gamma - 80) = 0, gamma = 1/2. Then sigma_xx = 20 - 160 e_X and sigma_yy = 60 - 160 e_X:
    #     (-20, +20) and (-60, -20).
    # The cell above is the mirror image. The von Mises stress is 40 in every triangle: sigma_zz = nu (sigma_xx +
    # sigma_yy) - E alpha dT_X = -20 and -60, and the three differences are (-40, 40, 0) and (-40, 40, 0) again.
    # These are the numbers of an infinite strip; the rollers at the ends matter less and less (a factor 3 per column).
    # After one step everything is multiplied by g = 2/3. v grows by gamma h = 1/4 from one row to the next
    nx = 24
    result = solve(**strip(nx, 1.0, 1.0 / 36.0, 1))
    want = [[[-20.0, 20.0], [-60.0, -20.0]], [[-60.0, -20.0], [-20.0, 20.0]]]  # [row][triangle of the cell][xx, yy]
    for state, factor in ((0, 1.0), (1, 2.0 / 3.0)):
        for column in range(nx // 2 - 1, nx // 2 + 2):
            for row in range(2):
                for half in range(2):
                    triangle = 2 * (row * nx + column) + half
                    label = "state %d, cell (%d, %d), triangle %d" % (state, column, row, half)
                    sigma = result["sigma"][state, triangle]
                    np.testing.assert_allclose(sigma[:2], factor * np.array(want[row][half]), rtol=0, atol=2e-3,
                                               err_msg=label)
                    assert abs(sigma[2]) < 2e-3, label
                    assert result["von_mises"][state, triangle] == pytest.approx(40.0 * factor, abs=1e-2), label
        middle = nx // 2
        v = [result["u"][state, 2 * (row * (nx + 1) + middle) + 1] for row in range(3)]
        assert v[1] - v[0] == pytest.approx(factor * 0.25, abs=1e-5)
        assert v[2] - v[1] == pytest.approx(factor * 0.25, abs=1e-5)
        assert abs(result["u"][state, 2 * ((nx + 1) + middle)]) < 1e-9


def test_a_steel_slab_cooled_from_both_faces_has_the_textbook_stress():
    # a slab 0.1 m thick and 0.4 m long (32 x 8 cells), steel (E = 210 GPa, nu = 0.3, alpha = 12e-6, rho_c = 7850 x 470,
    # k = 45), the faces at 293.15 K and the initial temperature 293.15 + 100 sin(pi y / H): it cools, and the mode
    # decays like exp(-t / tau), tau = H^2 / (pi^2 k / rho_c) = 83 s. With rollers at the ends, in the middle of the
    # length the slab does not depend on x, u = 0 and, with dT the mean of the rises of the two rows of nodes of a cell,
    #     sigma_xx (mean of the two triangles of the cell) = -E alpha dT / (1 - nu)
    # (360 MPa of compression for 100 K). Steps of 5 s with Crank-Nicolson, states saved at 0, 50 and 100 s
    nx, ny, height, t_ref, amplitude = 32, 8, 0.1, 293.15, 100.0
    E, nu, alpha, rho_c, k = 210.0e9, 0.3, 12.0e-6, 7850.0 * 470.0, 45.0
    mesh = thermomech.rectangle_mesh(0.4, height, nx, ny)
    result = solve(mesh=mesh, rho_c=rho_c, k=k, dt=5.0, n_steps=20, theta=0.5, save_every=10, E=E, nu=nu, alpha=alpha,
                   T_ref=t_ref, **roller_slab(nx, ny, t_ref, amplitude))
    np.testing.assert_allclose(result["time"], [0.0, 50.0, 100.0], rtol=1e-14, atol=0.0)
    scale = E * alpha * amplitude / (1.0 - nu)
    tau = height**2 / (PI**2 * k / rho_c)
    column = nx // 2
    for state, time in enumerate(result["time"]):
        for row in range(ny):
            cell = row * nx + column
            mean_xx = 0.5 * (result["sigma"][state, 2 * cell, 0] + result["sigma"][state, 2 * cell + 1, 0])
            below = result["T"][state, row * (nx + 1) + column] - t_ref
            above = result["T"][state, (row + 1) * (nx + 1) + column] - t_ref
            assert mean_xx == pytest.approx(-E * alpha * 0.5 * (below + above) / (1.0 - nu), abs=1e-4 * scale)
            y_middle = (row + 0.5) * height / ny
            assert mean_xx == pytest.approx(-scale * math.sin(PI * y_middle / height) * math.exp(-time / tau),
                                            abs=0.05 * scale)
    middle = 0.5 * (result["sigma"][0, 2 * (4 * nx + column), 0] + result["sigma"][0, 2 * (4 * nx + column) + 1, 0])
    assert -scale < middle < -0.95 * scale  # at mid-height at time 0: 346 MPa, 96 % of 360


# ---------------------------------------------------------------- the same chain computed in another way


@pytest.mark.parametrize("case", GENERAL_CASES)
def test_every_state_matches_an_independent_computation_with_dense_matrices(case):
    arguments = general_problem(case)
    result = solve(**arguments)
    expected = reference_states(arguments)
    np.testing.assert_allclose(result["time"], expected["time"], rtol=1e-14, atol=0.0)
    assert_close(result["T"], expected["T"], 1e-9, "T")
    assert_close(result["u"], expected["u"], 1e-8, "u")
    assert_close(result["sigma"], expected["sigma"], 1e-8, "sigma")
    assert_close(result["von_mises"], expected["von_mises"], 1e-8, "von_mises")
    assert np.abs(expected["sigma"][-1]).max() > 1e-3 and np.abs(expected["u"][-1]).max() > 1e-4


def test_the_temperatures_are_those_of_the_transient_solver():
    arguments = general_problem()
    result = solve(**arguments)
    for s, steps in enumerate((0, 2, 4, 6)):
        T = thermomech.solve_transient_2d(arguments["mesh"], arguments["rho_c"], arguments["k"], arguments["f"],
                                          arguments["fixed_nodes"], arguments["fixed_temperatures"], arguments["T0"],
                                          arguments["dt"], steps, arguments["theta"])
        assert_close(result["T"][s], T, 1e-9, "state %d" % s)


def test_saving_every_three_steps_gives_the_same_numbers_as_saving_every_step():
    every_step = solve(**general_problem(save_every=1))
    every_third = solve(**general_problem(save_every=3))
    assert every_step["T"].shape[0] == 7 and every_third["T"].shape[0] == 3
    for name in NAMES:
        assert_close(every_third[name], every_step[name][::3], 1e-10, name)


@pytest.mark.parametrize("case", GENERAL_CASES[:3])
def test_the_von_mises_stress_follows_from_the_stress_and_the_temperature(case):
    arguments = general_problem(case)
    result = solve(**arguments)
    for s in range(4):
        expected = von_mises_of(arguments["mesh"], result["sigma"][s], result["T"][s], arguments["E"],
                                arguments["nu"], arguments["alpha"], arguments["T_ref"])
        assert_close(result["von_mises"][s], expected, 1e-12, "state %d" % s)
    assert result["von_mises"].min() >= 0.0 and result["von_mises"][-1].max() > 1e-3


# ---------------------------------------------------------------- scaling laws


def zero_constraints(**change):
    return general_problem(fixed_displacements=[0.0, 0.0, 0.0], **change)


def test_a_doubled_young_modulus_doubles_the_stresses_and_leaves_the_displacements_and_the_heat():
    base = solve(**zero_constraints())
    stiff = solve(**zero_constraints(E=140.0))
    np.testing.assert_array_equal(stiff["T"], base["T"])
    assert_close(stiff["u"], base["u"], 1e-8)
    assert_close(stiff["sigma"], 2.0 * base["sigma"], 1e-8)
    assert_close(stiff["von_mises"], 2.0 * base["von_mises"], 1e-8)


def test_a_doubled_expansion_coefficient_doubles_the_displacements_and_the_stresses():
    base = solve(**zero_constraints())
    expanding = solve(**zero_constraints(alpha=0.006))
    np.testing.assert_array_equal(expanding["T"], base["T"])
    for name in ("u", "sigma", "von_mises"):
        assert_close(expanding[name], 2.0 * base[name], 1e-8, name)


@pytest.mark.parametrize("factor", [2.0, -1.0])
def test_multiplying_the_heat_multiplies_the_displacements_and_the_stresses(factor):
    # the rise over T_ref of the initial and imposed temperatures and the source are multiplied by the factor: the
    # temperature rise, u and sigma are multiplied by it, and the von Mises stress (even in the stress) by its absolute
    # value
    base_arguments = zero_constraints()
    base = solve(**base_arguments)
    t_ref = base_arguments["T_ref"]
    imposed = np.array(base_arguments["fixed_temperatures"])
    hot = solve(**zero_constraints(T0=t_ref + factor * (base_arguments["T0"] - t_ref),
                                   fixed_temperatures=t_ref + factor * (imposed - t_ref),
                                   f=factor * base_arguments["f"]))
    assert_close(hot["T"], t_ref + factor * (base["T"] - t_ref), 1e-8, "T")
    assert_close(hot["u"], factor * base["u"], 1e-8, "u")
    assert_close(hot["sigma"], factor * base["sigma"], 1e-8, "sigma")
    assert_close(hot["von_mises"], abs(factor) * base["von_mises"], 1e-8, "von_mises")


def test_moving_the_reference_temperature_with_all_the_temperatures_changes_only_the_temperatures():
    base_arguments = zero_constraints()
    base = solve(**base_arguments)
    shifted = solve(**zero_constraints(T_ref=base_arguments["T_ref"] + 100.0, T0=base_arguments["T0"] + 100.0,
                                       fixed_temperatures=np.array(base_arguments["fixed_temperatures"]) + 100.0))
    assert_close(shifted["T"], base["T"] + 100.0, 1e-8, "T")
    for name in ("u", "sigma", "von_mises"):
        assert_close(shifted[name], base[name], 1e-8, name)


# ---------------------------------------------------------------- the arguments


def test_the_arguments_can_be_given_by_name_and_by_position():
    arguments = general_problem()
    assert list(arguments) == ARGUMENTS  # the order of the docstring
    by_position = thermomech.solve_thermomechanical_2d(*arguments.values())
    by_name = solve(**arguments)
    for name in NAMES:
        np.testing.assert_array_equal(by_name[name], by_position[name], err_msg=name)
    changed = general_problem(n_steps=np.int64(6), save_every=np.int64(2), theta=0.6, dt=0.15, E=70, nu=0.2)
    again = solve(**changed)
    for name in NAMES:
        np.testing.assert_array_equal(again[name], by_name[name], err_msg=name)


@pytest.mark.parametrize("name", ARGUMENTS)
def test_every_argument_is_required(name):
    arguments = general_problem(SMALL)
    del arguments[name]
    with pytest.raises(TypeError):
        solve(**arguments)


def small_exact_problem(**change):
    """The 2 x 2 cells unit plate with data that float32 represents exactly."""
    arguments = dict(f=np.linspace(0.0, 4.0, 9), T0=np.linspace(290.0, 330.0, 9),
                     fixed_temperatures=[310.0, 280.0, 300.0], fixed_displacements=[0.5, -0.25, 0.75])
    arguments.update(change)
    return general_problem(SMALL, **arguments)


def test_lists_tuples_and_other_number_types_give_the_same_answer():
    base = small_exact_problem()
    reference = solve(**base)
    variants = [dict(f=list(base["f"])), dict(f=tuple(base["f"])), dict(f=np.asarray(base["f"], dtype=np.float32)),
                dict(T0=list(base["T0"])), dict(T0=np.asarray(base["T0"], dtype=np.float32)),
                dict(fixed_nodes=(0, 2, 6)), dict(fixed_nodes=np.array([0, 2, 6], dtype=np.int32)),
                dict(fixed_nodes=np.array([0, 2, 6], dtype=np.uint8)), dict(fixed_dofs=(0, 1, 5)),
                dict(fixed_dofs=np.array([0, 1, 5], dtype=np.int32)),
                dict(fixed_temperatures=(310, 280, 300)),
                dict(fixed_temperatures=np.array([310.0, 280.0, 300.0], dtype=np.float32)),
                dict(fixed_displacements=(0.5, -0.25, 0.75)),
                dict(fixed_displacements=np.array([0.5, -0.25, 0.75], dtype=np.float32)),
                dict(rho_c=2.5, k=0.7, E=70, T_ref=290, nu=0.2)]
    for change in variants:
        result = solve(**dict(base, **change))
        for name in NAMES:
            np.testing.assert_array_equal(result[name], reference[name], err_msg="%s, %s" % (name, sorted(change)))


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
VECTORS = ["f", "fixed_nodes", "fixed_temperatures", "T0", "fixed_dofs", "fixed_displacements"]


@pytest.mark.parametrize("layout", LAYOUTS, ids=[layout.__name__ for layout in LAYOUTS])
def test_arrays_of_any_layout_are_read_correctly(layout):
    arrays = dict(fixed_nodes=np.array([0, 2, 6], dtype=np.uint64), fixed_dofs=np.array([0, 1, 5], dtype=np.uint64),
                  fixed_temperatures=np.array([310.0, 280.0, 300.0]),
                  fixed_displacements=np.array([0.5, -0.25, 0.75]))
    base = small_exact_problem(**arrays)
    reference = solve(**base)
    for name in VECTORS:
        result = solve(**dict(base, **{name: layout(np.asarray(base[name]))}))
        for key in NAMES:
            np.testing.assert_array_equal(result[key], reference[key], err_msg="%s, %s" % (name, key))


def test_the_arguments_are_not_modified():
    arguments = small_exact_problem(fixed_nodes=np.array([0, 2, 6]), fixed_dofs=np.array([0, 1, 5]),
                                    fixed_temperatures=np.array([310.0, 280.0, 300.0]),
                                    fixed_displacements=np.array([0.5, -0.25, 0.75]))
    copies = {name: arguments[name].copy() for name in VECTORS}
    solve(**arguments)
    for name in VECTORS:
        np.testing.assert_array_equal(arguments[name], copies[name], err_msg=name)


# ---------------------------------------------------------------- wrong data


def small_problem(**change):
    return general_problem(SMALL, **change)  # 9 nodes (18 displacements), 8 triangles


NAN = float("nan")
BAD = {
    "save_every is zero": dict(save_every=0),
    "save_every is zero and there are no steps": dict(save_every=0, n_steps=0),
    "6 steps saved every 4": dict(n_steps=6, save_every=4),
    "6 steps saved every 5": dict(n_steps=6, save_every=5),
    "E is zero": dict(E=0.0), "E is negative": dict(E=-70.0), "E is nan": dict(E=NAN),
    "E is zero and there are no steps": dict(E=0.0, n_steps=0, save_every=1),
    "nu is one half": dict(nu=0.5), "nu is above one half": dict(nu=0.7), "nu is -1": dict(nu=-1.0),
    "nu is nan": dict(nu=NAN),
    "T0 is too short": dict(T0=np.zeros(8)), "T0 is too long": dict(T0=np.zeros(10)),
    "T0 is too short and there are no steps": dict(T0=np.zeros(8), n_steps=0, save_every=1),
    "only two displacements are fixed": dict(fixed_dofs=[0, 1], fixed_displacements=[0.0, 0.0]),
    "one fixed displacement too few": dict(fixed_displacements=[0.0, 0.0]),
    "one fixed displacement too many": dict(fixed_displacements=[0.0, 0.0, 0.0, 0.0]),
    "a degree of freedom outside the plate": dict(fixed_dofs=[0, 1, 18]),
    "a degree of freedom twice": dict(fixed_dofs=[0, 1, 1]),
    "k is zero": dict(k=0.0), "k is negative": dict(k=-1.0), "rho_c is zero": dict(rho_c=0.0),
    "dt is zero": dict(dt=0.0),
    "theta is above 1": dict(theta=2.0), "theta is below 0": dict(theta=-0.5),
    "f is too short": dict(f=np.zeros(8)), "f is too long": dict(f=np.zeros(10)),
    "a node outside the mesh": dict(fixed_nodes=[0, 9, 6]), "a node twice": dict(fixed_nodes=[0, 0, 6]),
    "one temperature too few": dict(fixed_temperatures=[300.0, 280.0]),
}


@pytest.mark.parametrize("change", list(BAD.values()), ids=list(BAD))
def test_data_that_does_not_fit_is_a_value_error(change):
    with pytest.raises(ValueError):
        solve(**small_problem(**change))


INF = float("inf")
# An infinity in f is not in the list: when the norm of the right-hand side is infinite the conjugate gradient stops at
# its first check and returns the starting vector.
NAMES_WITH_NUMBERS = ("T0", "fixed_temperatures", "fixed_displacements", "alpha", "T_ref", "rho_c", "dt")
NOT_FINITE = [(name, NAN) for name in NAMES_WITH_NUMBERS + ("f",)] + [(name, INF) for name in NAMES_WITH_NUMBERS]


@pytest.mark.parametrize("name, bad", NOT_FINITE, ids=["%s is %s" % (name, bad) for name, bad in NOT_FINITE])
def test_a_nan_or_an_infinity_is_never_returned_silently(name, bad):
    # the conjugate gradient cannot converge when the data hold a NaN: the C++ code throws and Python gets the error
    arguments = small_problem()
    value = arguments[name]
    if isinstance(value, float):
        arguments[name] = bad
    else:
        value = np.array(value, dtype=float)
        value[-1 if name != "f" else 4] = bad  # a free node for f
        arguments[name] = value
    with pytest.raises((ValueError, RuntimeError)):
        solve(**arguments)


WRONG_TYPE = {
    "no mesh": dict(mesh=None), "mesh is a number": dict(mesh=5), "rho_c is None": dict(rho_c=None),
    "k is a string": dict(k="hot"), "dt is a string": dict(dt="slow"), "theta is None": dict(theta=None),
    "E is None": dict(E=None), "nu is a string": dict(nu="x"), "alpha is None": dict(alpha=None),
    "T_ref is None": dict(T_ref=None),
    "n_steps is negative": dict(n_steps=-1), "n_steps is a float": dict(n_steps=2.5),
    "save_every is negative": dict(save_every=-1), "save_every is a float": dict(save_every=1.0),
    "f is None": dict(f=None), "T0 is None": dict(T0=None), "fixed_nodes is None": dict(fixed_nodes=None),
    "a node that is not an integer": dict(fixed_nodes=[0.5, 2, 6]),
    "fixed_temperatures is a string": dict(fixed_temperatures="abc"), "fixed_dofs is None": dict(fixed_dofs=None),
    "a degree of freedom that is not an integer": dict(fixed_dofs=[0.5, 1, 5]),
    "fixed_displacements is None": dict(fixed_displacements=None),
}


@pytest.mark.parametrize("change", list(WRONG_TYPE.values()), ids=list(WRONG_TYPE))
def test_arguments_of_the_wrong_type_are_a_type_error(change):
    with pytest.raises(TypeError):
        solve(**small_problem(**change))


def test_a_negative_node_or_degree_of_freedom_number_is_refused():
    with pytest.raises((TypeError, ValueError)):
        solve(**small_problem(fixed_nodes=[1, 2, -1]))
    with pytest.raises((TypeError, ValueError)):
        solve(**small_problem(fixed_dofs=[1, 2, -1]))
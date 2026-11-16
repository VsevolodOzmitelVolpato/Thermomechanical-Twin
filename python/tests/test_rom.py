"""Step 7d: the parametric reduced-order model of the thermo-mechanical plate, offline and online.

    from twin.rom import ThermoMechanicalROM

The full model is the one of the C++ solver: the heat equation M dT/dt + K T = f with temperatures imposed on some nodes
and, at every instant, the plate in plane strain in equilibrium with the nodal temperatures. Five numbers change from
one run to the next (the parameters) and everything else is fixed once and for all:

    k       the conductivity                    K = k K1       (K1: the stiffness matrix for k = 1)
    rho_c   the volumetric heat capacity        M = rho_c M1   (M1: the mass matrix for rho_c = 1)
    q       the strength of the heat source     f = q f1       (f1: the load of the source for q = 1)
    E       the Young modulus                   the displacements are proportional to alpha,
    alpha   the thermal expansion               the stresses to E alpha

So every matrix of the reduced model of step 7b is a combination of matrices that are computed once: with the basis B
and the lifting T_g (the imposed temperatures on the fixed nodes, 0 elsewhere)

    M_r = rho_c (B^T M1 B)    K_r = k (B^T K1 B)    f_r = q (B^T f1) - k (B^T K1 T_g)

and the mechanical modes of step 7c for E = 1 and alpha = 1 give those of every other material (the supports hold the
displacements at zero): u = alpha (u_lift + U a) and sigma = E alpha (sigma_lift + S a), with a the reduced coordinates.

The constructor is the OFFLINE phase: all that does not depend on the parameters, so every call to the C++ solver. The
method solve is the ONLINE phase: matrices r x r, the reconstruction of the fields and the von Mises stress. After the
constructor there is no call to thermomech and no n x n matrix.

ThermoMechanicalROM(mesh, basis, load, fixed_nodes, fixed_values, T0, fixed_dofs, nu, T_ref)
    mesh                       the Mesh2D, with n nodes and n_tri triangles
    basis                      (n, r) with r >= 1: the thermal modes, one per column (for example from pod_basis). They
                               must vanish on the fixed nodes (as in solve_reduced_heat of step 7b) but need not be
                               orthonormal
    load                       (n,): the nodal load of the heat source of strength 1, for example
                               assemble_load_2d(mesh, shape). The source of strength q is q * load
    fixed_nodes, fixed_values  the nodes with an imposed temperature (the sequence may be empty) and the temperatures;
                               the lifting is dirichlet_lift(n, fixed_nodes, fixed_values) of step 7b
    T0                         (n,): the temperature at the time 0. As in solve_reduced_heat the model sees the lifting
                               plus the projection of T0 - lifting on the span of the basis, for the inner product of M
    fixed_dofs                 the degrees of freedom of the displacements that are held at zero (the supports): at
                               least three, as for the C++ solver
    nu, T_ref                  the Poisson ratio and the reference temperature of the thermal strain
    The attribute num_modes is r, an int. The object keeps its own copy of the arguments: changing them later does not
    change the model. ValueError if basis is not (n, r) with r >= 1, if load or T0 is not (n,), if basis, load, T0 or
    fixed_values has a NaN or an infinity, if the basis does not vanish on the fixed nodes (the largest entry on those
    rows is more than 1e-8 times the largest entry of the whole basis), if the reduced mass matrix B^T M1 B is
    singular, for every error of dirichlet_lift, and for the errors of the C++ solver about the mechanics (nu not
    between -1 and 1/2, fewer than three degrees of freedom, a degree of freedom outside the plate or repeated).
    TypeError for arguments of the wrong type. RuntimeError for a T_ref that is NaN or infinite.

rom.solve(k, rho_c, q, E, alpha, dt, n_steps, theta, save_every=1)
    One run of the reduced model: the theta-scheme of n_steps steps of length dt (theta = 1 backward Euler, 1/2
    Crank-Nicolson, 0 forward Euler), with the heat parameters k, rho_c, q and the mechanical parameters E, alpha.
    Returns a dict with the states at the steps 0, save_every, 2 save_every, ..., n_steps (m = n_steps / save_every + 1
    states), the same keys and shapes as solve_thermomechanical_2d of the C++ solver, all new float64 arrays:

        "time"        (m,)            step * dt
        "T"           (m, n)          the temperatures: the lifting plus basis a, with a the coordinates of the state
        "u"           (m, 2 n)        the displacements, the node I in the places 2 I (u) and 2 I + 1 (v)
        "sigma"       (m, n_tri, 3)   the plane stresses (xx, yy, xy) of every triangle
        "von_mises"   (m, n_tri)      the von Mises stress, from sigma and the mean temperature of every triangle

    solve does not change the object: two calls with the same arguments give the same answer, whatever was solved in
    between. ValueError if k, rho_c, q, E, alpha or dt is not a finite number (NaN and infinity are not), if k, rho_c,
    E or dt is not > 0, if theta is not between 0 and 1 (NaN is not), if n_steps < 0, if save_every < 1 or does not
    divide n_steps. TypeError if n_steps or save_every is not an integer, and for arguments of the wrong type.
"""
import math

import numpy as np
import pytest
import scipy.sparse as sp

import thermomech

# Any warning is an error in these tests.
pytestmark = pytest.mark.filterwarnings("error")

PI = math.pi
NAN = float("nan")
INF = float("inf")


# The classes and the functions are imported inside these wrappers and not at the top: while twin/rom.py does not
# exist, or is not complete, only the tests that need it fail, not the whole session.


def make_rom(*args, **kwargs):
    from twin.rom import ThermoMechanicalROM
    return ThermoMechanicalROM(*args, **kwargs)


def reduced_heat(*args, **kwargs):
    from twin.galerkin import solve_reduced_heat
    return solve_reduced_heat(*args, **kwargs)


def rebuild(*args, **kwargs):
    from twin.galerkin import reconstruct
    return reconstruct(*args, **kwargs)


def modes(*args, **kwargs):
    from twin.mechanics import mechanical_modes
    return mechanical_modes(*args, **kwargs)


def von_mises(*args, **kwargs):
    from twin.mechanics import von_mises_plane_strain
    return von_mises_plane_strain(*args, **kwargs)


def mean_of(*args, **kwargs):
    from twin.mechanics import element_mean
    return element_mean(*args, **kwargs)


def pod(*args, **kwargs):
    from twin.pod import pod_basis
    return pod_basis(*args, **kwargs)


# Friendly numbers for hand calculations: E = 100, nu = 1/4, alpha = 1/100, T_ref = 20. Then a triangle that is heated
# by dT expands by (1 + nu) alpha dT = dT / 80 in x and y, and E alpha = 1.
HAND = dict(E=100.0, nu=0.25, alpha=0.01, T_ref=20.0)
STEEL = dict(E=2.1e5, nu=0.3, alpha=1.2e-5)
KEYS = {"time", "T", "u", "sigma", "von_mises"}


def relative_error(result, expected):
    """The largest difference, relative to the largest absolute value of the expected array."""
    return np.abs(np.asarray(result) - np.asarray(expected)).max() / np.abs(expected).max()


def assert_same_run(result, expected):
    """The two dicts hold the same arrays, bit for bit."""
    assert set(result) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(result[key], expected[key], err_msg=key)


def assert_close_runs(result, expected, temperature_error=1e-9, error=3e-8, stress_error=3e-7):
    """The two dicts hold the same states: temperatures to an absolute error, the rest to a relative one."""
    assert set(result) == set(expected)
    np.testing.assert_allclose(result["time"], expected["time"], rtol=1e-14, atol=0)
    assert np.abs(result["T"] - expected["T"]).max() < temperature_error, "the temperatures"
    assert relative_error(result["u"], expected["u"]) < error, "the displacements"
    assert relative_error(result["sigma"], expected["sigma"]) < stress_error, "the stresses"
    assert relative_error(result["von_mises"], expected["von_mises"]) < stress_error, "the von Mises stress"


def assert_close_cpp(result, expected):
    """As assert_close_runs, for the answers of the C++ solver: its conjugate gradient stops at a relative residual of
    1e-12, and the conditioning of the matrices turns it into errors of 1e-9 in the temperatures and the displacements
    and of 1e-8 and more in the stresses (the thermal strain and the elastic strain nearly cancel)."""
    assert_close_runs(result, expected, temperature_error=2e-8, error=1e-7, stress_error=3e-6)


# ---------------------------------------------------------------- memory layouts and containers


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


def container(data, make):
    """The nested list data as a list, as a tuple (of tuples) or as an array of the type make."""
    if make is list:
        return data
    if make is tuple:
        return tuple(tuple(row) if isinstance(row, list) else row for row in data)
    return np.array(data, dtype=make)


def stored_values(model):
    """Every array and sparse matrix that the object keeps: its attributes and what is inside its lists, tuples and
    dicts."""
    found = {}

    def visit(name, value, depth):
        if isinstance(value, np.ndarray) or sp.issparse(value):
            found[name] = value
        elif depth < 3 and isinstance(value, dict):
            for key, item in value.items():
                visit("%s[%r]" % (name, key), item, depth + 1)
        elif depth < 3 and isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit("%s[%d]" % (name, index), item, depth + 1)

    for name, value in vars(model).items():
        visit(name, value, 0)
    return found


# ---------------------------------------------------------------- the unit square, for the hand calculations


def unit_square():
    """Four nodes (0, 0), (1, 0), (0, 1), (1, 1) and two triangles (0, 1, 3) and (0, 3, 2)."""
    return thermomech.rectangle_mesh(1.0, 1.0, 1, 1)


# u and v of the node 0 and v of the node 1: a plate that is free to expand
PINS = [0, 1, 3]
# the rest of the unit square, as a function of the temperature rise: u = rise (1 + nu) alpha (x, y) for the nodes 0..3
EXPANSION = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def free_plate_arguments(**change):
    """No imposed temperature, one mode (the constant one), the plate free to expand: the heat cannot go anywhere."""
    mesh = unit_square()
    arguments = dict(mesh=mesh, basis=np.ones((4, 1)), load=thermomech.assemble_load_2d(mesh, 1.0), fixed_nodes=[],
                     fixed_values=[], T0=np.full(4, 20.0), fixed_dofs=list(PINS), nu=0.25, T_ref=20.0)
    arguments.update(change)
    return arguments


def free_plate_run(**change):
    """The temperature grows by q / rho_c = 2 per unit of time: 20, 21, 22 at the times 0, 1/2, 1."""
    arguments = dict(k=1.0, rho_c=2.0, q=4.0, E=100.0, alpha=0.01, dt=0.5, n_steps=2, theta=0.5)
    arguments.update(change)
    return arguments


def side_arguments(**change):
    """The left side (nodes 0 and 2) at 0 and one mode: x, that is (0, 1, 0, 1) on the nodes. Then M_r1 = 1/3,
    K_r1 = 1, f_r1 = 1/2 (the load of the unit source: its integral against x) and the lifting term is zero."""
    mesh = unit_square()
    arguments = dict(mesh=mesh, basis=np.array([[0.0], [1.0], [0.0], [1.0]]),
                     load=thermomech.assemble_load_2d(mesh, 1.0), fixed_nodes=[0, 2], fixed_values=[0.0, 0.0],
                     T0=np.zeros(4), fixed_dofs=list(PINS), nu=0.25, T_ref=20.0)
    arguments.update(change)
    return arguments


def side_run(**change):
    """M_r = 1, K_r = 2, f_r = 2 (q = 4): the equation is da/dt + 2 a = 2, with the steady state a = 1."""
    arguments = dict(k=2.0, rho_c=3.0, q=4.0, E=100.0, alpha=0.01, dt=1.0, n_steps=2, theta=1.0)
    arguments.update(change)
    return arguments


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


def dense_response(mesh, T, E, nu, alpha, T_ref, fixed_dofs):
    """u and sigma of the plate at the temperature T: K u = F with dense matrices and the fixed degrees of freedom
    (displacement zero) eliminated, then sigma = D (B u - eps_th) in each triangle."""
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
    is_free = np.ones(2 * n, dtype=bool)
    is_free[np.asarray(fixed_dofs)] = False
    free = np.flatnonzero(is_free)
    u = np.zeros(2 * n)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    sigma = np.array([D @ (B @ u[dofs] - eps_th) for B, dofs, eps_th in elements])
    return u, sigma


# ---------------------------------------------------------------- a plate to run against the C++ solver

NX, NY = 6, 4
BASE_K, BASE_RHO_C = 0.8, 1.5
T_REF = 22.0  # inside the range of the temperatures: the plate is both hotter and colder than the reference


def heat_spot(x, y):
    return 6.0 * math.exp(-((x - 1.2) ** 2 + (y - 0.4) ** 2) / 0.15)


class Plate:
    """A rectangle 2 x 1 of 7 x 5 nodes (30 free nodes), heated by a spot, with the temperature imposed on the left side
    (20 + 5 y) or on the whole border (20 + x)."""

    def __init__(self):
        self.mesh = thermomech.rectangle_mesh(2.0, 1.0, NX, NY)
        self.n = self.mesh.num_nodes
        self.n_tri = self.mesh.num_triangles
        self.x, self.y = np.asarray(self.mesh.x), np.asarray(self.mesh.y)
        self.mass = thermomech.assemble_mass_2d(self.mesh, 1.0)
        self.stiffness = thermomech.assemble_stiffness_2d(self.mesh, 1.0)
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

    def nodal_basis(self, nodes):
        """One mode for every free node: the reduced model is then the full model."""
        free = [node for node in range(self.n) if node not in nodes]
        return np.eye(self.n)[:, free]

    def holds(self, name):
        """The supports: pins (the minimum), rollers (u = 0 on the left side and v = 0 on the bottom side) or the
        clamped left side."""
        if name == "pins":
            return [0, 1, 2 * NX + 1]
        if name == "rollers":
            return [2 * node for node in self.left] + [2 * i + 1 for i in range(NX + 1)]
        return [d for node in self.left for d in (2 * node, 2 * node + 1)]

    def rom(self, basis, side="left", holds="pins", start=None, nu=0.3, T_ref=T_REF, load=None):
        nodes, values, _, default_start = self.side(side)
        return make_rom(self.mesh, basis, self.load if load is None else load, nodes, values,
                        default_start if start is None else start, self.holds(holds), nu, T_ref)

    def full(self, side, holds, k, rho_c, q, E, alpha, dt, n_steps, theta, start=None, nu=0.3, T_ref=T_REF,
             save_every=1):
        """The C++ solver, with the same parameters."""
        nodes, values, _, default_start = self.side(side)
        dofs = self.holds(holds)
        return thermomech.solve_thermomechanical_2d(
            self.mesh, rho_c, k, q * self.load, nodes, values, default_start if start is None else start, dt, n_steps,
            theta, save_every, E, nu, alpha, T_ref, dofs, [0.0] * len(dofs))


@pytest.fixture(scope="module")
def plate():
    return Plate()


CONSTRAINTS = ["pins", "rollers", "clamped"]
SIDES = ["left", "border"]
# forward Euler needs a time step much shorter than rho_c / k: the step is FACTOR rho_c / k
THETAS = [(0.0, 0.002), (0.5, 0.05), (2.0 / 3.0, 0.05), (1.0, 0.05)]
THETA_IDS = ["forward", "crank-nicolson", "two-thirds", "backward"]


def random_parameters(seed):
    """Heat and mechanical parameters in wide ranges: the conductivity and the capacity change by a factor of 10 and
    more, the source can be cold, the material is from soft to stiff."""
    rng = np.random.default_rng(seed)
    return dict(k=float(rng.uniform(0.2, 3.0)), rho_c=float(rng.uniform(0.5, 6.0)), q=float(rng.uniform(-2.0, 4.0)),
                E=float(10.0 ** rng.uniform(3.5, 5.5)), alpha=float(rng.uniform(5e-6, 5e-5)))


# ---------------------------------------------------------------- hand examples

FREE_PLATE_RISE = np.array([0.0, 1.0, 2.0])  # T - T_ref at the times 0, 1/2, 1 of the free plate


def test_rom_hand_a_free_plate_heated_by_a_source_grows_uniformly_and_expands_without_stress():
    # the heat cannot leave: rho_c dT/dt = q, so T = 20 + 2 t. A triangle that is heated by dT expands by dT / 80 in x
    # and y: u = dT / 80 (x, y) and nothing is stressed in the plane. The stress out of the plane is - E alpha dT: the
    # von Mises stress is E alpha dT = dT
    result = make_rom(**free_plate_arguments()).solve(**free_plate_run())
    np.testing.assert_allclose(result["time"], [0.0, 0.5, 1.0], rtol=0, atol=1e-15)
    np.testing.assert_allclose(result["T"], 20.0 + np.outer(FREE_PLATE_RISE, np.ones(4)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["u"], np.outer(FREE_PLATE_RISE / 80.0, EXPANSION), rtol=0, atol=1e-9)
    np.testing.assert_allclose(result["sigma"], 0.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(result["von_mises"], np.outer(FREE_PLATE_RISE, [1.0, 1.0]), rtol=0, atol=1e-9)


@pytest.mark.parametrize("theta", [0.0, 0.5, 2.0 / 3.0, 1.0])
@pytest.mark.parametrize("k", [0.01, 1.0, 250.0])
def test_rom_hand_a_free_plate_does_not_care_about_the_conductivity_nor_the_scheme(k, theta):
    # the constant mode has no gradient: K_r = 0 and every theta-scheme gives the exact T = 20 + 2 t
    result = make_rom(**free_plate_arguments()).solve(**free_plate_run(k=k, theta=theta))
    np.testing.assert_allclose(result["T"], 20.0 + np.outer(FREE_PLATE_RISE, np.ones(4)), rtol=0, atol=1e-11)
    np.testing.assert_allclose(result["von_mises"], np.outer(FREE_PLATE_RISE, [1.0, 1.0]), rtol=0, atol=1e-9)


@pytest.mark.parametrize("q, rho_c", [(8.0, 4.0), (1.0, 0.5), (-4.0, 2.0), (0.0, 2.0), (3.0, 0.25)])
def test_rom_hand_a_free_plate_the_temperature_grows_by_q_over_rho_c_per_unit_of_time(q, rho_c):
    result = make_rom(**free_plate_arguments()).solve(**free_plate_run(q=q, rho_c=rho_c))
    expected = 20.0 + np.outer(np.array([0.0, 0.5, 1.0]) * q / rho_c, np.ones(4))
    np.testing.assert_allclose(result["T"], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("E, alpha", [(100.0, 0.01), (200.0, 0.005), (50.0, 0.02), (2.1e5, 1.2e-5), (1.0, 3.0)])
@pytest.mark.parametrize("nu", [0.25, 0.0, 0.4, -0.3])
def test_rom_hand_a_free_plate_expands_by_one_plus_nu_alpha_and_has_the_von_mises_stress_E_alpha_rise(E, alpha, nu):
    # u = (1 + nu) alpha dT (x, y), no stress in the plane, von Mises stress E alpha |dT| for every material
    result = make_rom(**free_plate_arguments(nu=nu)).solve(**free_plate_run(E=E, alpha=alpha))
    assert relative_error(result["u"], np.outer((1.0 + nu) * alpha * FREE_PLATE_RISE, EXPANSION)) < 1e-8
    assert np.abs(result["sigma"]).max() < 1e-8 * E * alpha
    assert relative_error(result["von_mises"], np.outer(E * alpha * FREE_PLATE_RISE, [1.0, 1.0])) < 1e-8


def test_rom_hand_a_free_plate_that_starts_above_the_reference_temperature():
    # 30 at the start, so 10 above the reference: u = dT / 80 (x, y) with dT = 10, 11, 12, von Mises stress dT
    result = make_rom(**free_plate_arguments(T0=np.full(4, 30.0))).solve(**free_plate_run())
    rise = 10.0 + FREE_PLATE_RISE
    np.testing.assert_allclose(result["T"], 30.0 + np.outer(FREE_PLATE_RISE, np.ones(4)), rtol=0, atol=1e-11)
    np.testing.assert_allclose(result["u"], np.outer(rise / 80.0, EXPANSION), rtol=0, atol=1e-9)
    np.testing.assert_allclose(result["von_mises"], np.outer(rise, [1.0, 1.0]), rtol=0, atol=1e-9)


def test_rom_hand_a_free_plate_that_is_colder_than_the_reference_contracts_and_the_von_mises_stress_is_positive():
    # T_ref = 21: the rise is -1, 0, 1 at the three times; the plate shrinks, rests, then expands; the stress is |rise|
    result = make_rom(**free_plate_arguments(T_ref=21.0)).solve(**free_plate_run())
    rise = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(result["u"], np.outer(rise / 80.0, EXPANSION), rtol=0, atol=1e-9)
    np.testing.assert_allclose(result["von_mises"], np.outer(np.abs(rise), [1.0, 1.0]), rtol=0, atol=1e-9)


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("n_steps", [0, 1, 4])
def test_rom_hand_a_free_plate_every_state_of_a_run_is_the_state_of_the_same_time(n_steps, theta):
    # the steps have the length 1/2 here: the state s is the one at the time s / 2, whatever the number of steps
    result = make_rom(**free_plate_arguments()).solve(**free_plate_run(n_steps=n_steps, theta=theta))
    rise = 0.5 * np.arange(n_steps + 1) * 2.0
    assert result["T"].shape == (n_steps + 1, 4)
    np.testing.assert_allclose(result["T"][:, 0], 20.0 + rise, rtol=0, atol=1e-11)
    np.testing.assert_allclose(result["von_mises"][:, 0], rise, rtol=0, atol=1e-9)


@pytest.mark.parametrize("theta, coordinates", [(1.0, [0.0, 2.0 / 3.0, 8.0 / 9.0]), (0.5, [0.0, 1.0, 1.0]),
                                                (0.0, [0.0, 2.0, 0.0])], ids=["backward", "crank-nicolson", "forward"])
def test_rom_hand_b_one_mode_the_theta_scheme_of_a_scalar_equation(theta, coordinates):
    # (M_r + theta dt K_r) a_new = (M_r - (1 - theta) dt K_r) a + dt f_r with M_r = 1, K_r = 2, f_r = 2 and dt = 1
    #   backward:        3 a_new = a + 2     so a = 0, 2/3, 8/9
    #   Crank-Nicolson:  2 a_new = 2         so a = 0, 1, 1
    #   forward:         a_new = 2 - a       so a = 0, 2, 0
    result = make_rom(**side_arguments()).solve(**side_run(theta=theta))
    np.testing.assert_allclose(result["T"], np.outer(coordinates, [0.0, 1.0, 0.0, 1.0]), rtol=0, atol=1e-12)


def test_rom_hand_b_the_steady_state_is_the_solution_of_K_r_a_equal_f_r():
    # a -> f_r / K_r = 1, as 1 - (1/3)^n: after 40 backward steps the temperature is the mode itself, T = x
    result = make_rom(**side_arguments()).solve(**side_run(n_steps=40, save_every=40))
    np.testing.assert_allclose(result["T"][-1], [0.0, 1.0, 0.0, 1.0], rtol=0, atol=1e-12)


@pytest.mark.parametrize("k, rho_c, q, dt", [(2.0, 3.0, 4.0, 1.0), (1.0, 3.0, 4.0, 1.0), (2.0, 6.0, 4.0, 1.0),
                                             (2.0, 3.0, 8.0, 1.0), (2.0, 3.0, 4.0, 0.5), (0.5, 0.3, -1.0, 2.0)])
def test_rom_hand_b_k_multiplies_the_stiffness_rho_c_the_mass_and_q_the_source(k, rho_c, q, dt):
    # M_r = rho_c / 3, K_r = k, f_r = q / 2: one backward step from zero gives a = dt (q / 2) / (rho_c / 3 + dt k)
    result = make_rom(**side_arguments()).solve(**side_run(k=k, rho_c=rho_c, q=q, dt=dt, n_steps=1))
    a = dt * q / 2.0 / (rho_c / 3.0 + dt * k)
    np.testing.assert_allclose(result["T"][1], [0.0, a, 0.0, a], rtol=1e-12, atol=1e-12)


def hot_side_arguments(**change):
    """As side_arguments, with the left side at 10 and a plate that starts at the lifting: 10 on the side, 0 inside."""
    arguments = dict(fixed_values=[10.0, 10.0], T0=np.array([10.0, 0.0, 10.0, 0.0]))
    arguments.update(change)
    return side_arguments(**arguments)


def test_rom_hand_c_a_hot_side_heats_the_plate_through_the_lifting_term():
    # B^T K1 T_g = -10 for T_g = (10, 0, 10, 0): the side pulls the free nodes up, f_r = q / 2 - k (-10) = 20 for q = 0.
    # Backward Euler with M_r = 1 and K_r = 2: a = 0, 20/3, 80/9. The temperature is T_g + a x = (10, a, 10, a)
    result = make_rom(**hot_side_arguments()).solve(**side_run(q=0.0))
    for state, a in enumerate([0.0, 20.0 / 3.0, 80.0 / 9.0]):
        np.testing.assert_allclose(result["T"][state], [10.0, a, 10.0, a], rtol=0, atol=1e-12)


def test_rom_hand_c_the_heat_of_the_side_and_the_heat_of_the_source_add_up():
    # f_r = q / 2 + 10 k = 2 + 20 = 22: a = 0, 22/3, 88/9
    result = make_rom(**hot_side_arguments()).solve(**side_run())
    for state, a in enumerate([0.0, 22.0 / 3.0, 88.0 / 9.0]):
        np.testing.assert_allclose(result["T"][state], [10.0, a, 10.0, a], rtol=0, atol=1e-12)


@pytest.mark.parametrize("k", [0.5, 2.0, 40.0])
def test_rom_hand_c_the_lifting_term_is_proportional_to_k(k):
    # f_r = 10 k and K_r = k: the steady state of the plate is a = 10 for every k, T = 10 everywhere
    result = make_rom(**hot_side_arguments()).solve(**side_run(k=k, q=0.0, rho_c=3.0, dt=1.0, n_steps=100,
                                                               save_every=100))
    np.testing.assert_allclose(result["T"][-1], 10.0, rtol=0, atol=1e-9)


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
def test_rom_hand_c_a_plate_that_starts_at_the_temperature_of_the_side_stays_there(theta):
    # a0 = 10 and K_r a0 + k (B^T K1 T_g) = 20 - 20 = 0: nothing moves, for every scheme
    result = make_rom(**hot_side_arguments(T0=np.full(4, 10.0))).solve(**side_run(q=0.0, theta=theta, n_steps=3))
    np.testing.assert_allclose(result["T"], 10.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("rho_c", [0.5, 3.0, 20.0])
def test_rom_hand_c_the_model_sees_the_projection_of_the_initial_temperature_whatever_rho_c(rho_c):
    # T0 - T_g = (0, 4, 0, 8) is not a multiple of the mode x. Its projection for the inner product of M1 is
    # a0 = (x . M1 (0, 4, 0, 8)) / (x . M1 x) = (13/6) / (1/3) = 13/2, and rho_c cancels: T[0] = (10, 6.5, 10, 6.5)
    arguments = hot_side_arguments(T0=np.array([10.0, 4.0, 10.0, 8.0]))
    result = make_rom(**arguments).solve(**side_run(rho_c=rho_c, n_steps=0))
    np.testing.assert_allclose(result["T"], [[10.0, 6.5, 10.0, 6.5]], rtol=0, atol=1e-12)


def test_rom_hand_c_the_projection_uses_the_whole_vector_including_the_fixed_nodes():
    # T0 = 0 on the side that is imposed at 10: T0 - T_g = (-10, 0, -10, 0). x . M1 (-10, 0, -10, 0) = -20/12, so
    # a0 = (-5/3) / (1/3) = -5: the free nodes see the cold side and start at -5
    result = make_rom(**hot_side_arguments(T0=np.zeros(4))).solve(**side_run(n_steps=0))
    np.testing.assert_allclose(result["T"], [[10.0, -5.0, 10.0, -5.0]], rtol=0, atol=1e-12)


# ---------------------------------------------------------------- the constructor: the offline phase


def random_basis(plate, r, seed, side="left"):
    """r Gaussian modes that vanish on the nodes with an imposed temperature."""
    basis = np.random.default_rng(seed).standard_normal((plate.n, r))
    basis[plate.side(side)[0]] = 0.0
    return basis


@pytest.mark.parametrize("r", [1, 3, 7, 30])
def test_rom_build_num_modes_is_the_number_of_columns_of_the_basis(plate, r):
    rom = plate.rom(random_basis(plate, r, 5))
    assert rom.num_modes == r
    assert isinstance(rom.num_modes, int)


def test_rom_build_the_arguments_are_not_modified():
    arguments = hot_side_arguments(basis=np.array([[0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [1.0, 0.5]]),
                                   T0=np.array([10.0, 4.0, 10.0, 8.0]))
    saved = {}
    for name, value in arguments.items():
        if isinstance(value, np.ndarray):
            value = value.copy()
        elif isinstance(value, list):
            value = list(value)
        saved[name] = value
    make_rom(**arguments)
    for name, value in saved.items():
        if name != "mesh":
            np.testing.assert_array_equal(arguments[name], value, err_msg=name)
            assert type(arguments[name]) is type(value), name
    assert arguments["mesh"].num_nodes == 4 and arguments["mesh"].num_triangles == 2


def test_rom_build_the_model_keeps_its_own_copy_of_the_arguments():
    expected = make_rom(**hot_side_arguments()).solve(**side_run())
    arguments = hot_side_arguments()
    rom = make_rom(**arguments)
    # spoil everything that the caller still owns
    arguments["basis"][:] = 5.0
    arguments["load"][:] = -3.0
    arguments["T0"][:] = 99.0
    arguments["fixed_nodes"][:] = [1, 3]
    arguments["fixed_values"][:] = [77.0, 88.0]
    arguments["fixed_dofs"][:] = [5, 6, 7]
    assert_same_run(rom.solve(**side_run()), expected)


@pytest.mark.parametrize("make_nodes", [list, tuple, np.int64, np.int32, np.uint8, np.int16])
@pytest.mark.parametrize("make_values", [list, tuple, np.float64, np.float32, np.int64])
def test_rom_build_accepts_the_imposed_temperatures_in_any_container_and_type(make_nodes, make_values):
    expected = make_rom(**hot_side_arguments()).solve(**side_run())
    arguments = hot_side_arguments(fixed_nodes=container([0, 2], make_nodes),
                                   fixed_values=container([10.0, 10.0], make_values))
    assert_same_run(make_rom(**arguments).solve(**side_run()), expected)


@pytest.mark.parametrize("make", [list, tuple, np.int64, np.int32, np.uint16, np.int8])
def test_rom_build_accepts_the_supports_in_any_container_and_type(make):
    expected = make_rom(**side_arguments()).solve(**side_run())
    assert_same_run(make_rom(**side_arguments(fixed_dofs=container([0, 1, 3], make))).solve(**side_run()), expected)


@pytest.mark.parametrize("make_basis", [list, tuple, np.float64, np.float32, np.int64])
@pytest.mark.parametrize("make_load", [list, tuple, np.float64, np.float32])
def test_rom_build_accepts_basis_load_and_T0_of_any_container_and_type(make_basis, make_load):
    # the numbers are exact in every type
    load, T0 = [0.5, 0.25, 0.25, 0.5], [4.0, 2.0, 4.0, 2.0]
    expected = make_rom(**side_arguments(load=np.array(load), T0=np.array(T0))).solve(**side_run())
    arguments = side_arguments(basis=container([[0.0], [1.0], [0.0], [1.0]], make_basis),
                               load=container(load, make_load), T0=container(T0, make_load))
    assert_same_run(make_rom(**arguments).solve(**side_run()), expected)


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_rom_build_arrays_of_any_memory_layout_give_the_same_model(plate, layout):
    basis = random_basis(plate, 4, 8)
    run = dict(k=0.7, rho_c=2.0, q=1.5, E=1.0e5, alpha=1.5e-5, dt=0.05, n_steps=3, theta=0.5)
    start = plate.side("left")[3]
    expected = plate.rom(basis).solve(**run)
    got = plate.rom(layout(basis), start=layout(start), load=layout(plate.load)).solve(**run)
    for key in expected:
        np.testing.assert_allclose(got[key], expected[key], rtol=1e-12, atol=1e-12, err_msg=key)


def test_rom_build_arguments_can_be_given_by_position_in_the_documented_order():
    arguments = hot_side_arguments()
    order = ["mesh", "basis", "load", "fixed_nodes", "fixed_values", "T0", "fixed_dofs", "nu", "T_ref"]
    by_name = make_rom(**hot_side_arguments()).solve(**side_run())
    assert_same_run(make_rom(*[arguments[name] for name in order]).solve(**side_run()), by_name)


def test_rom_build_without_imposed_temperatures_the_basis_does_not_have_to_vanish_anywhere():
    # no fixed nodes: the lifting is zero and the constant mode is a legitimate mode
    rom = make_rom(**free_plate_arguments(fixed_nodes=np.array([], dtype=int), fixed_values=np.array([])))
    assert rom.num_modes == 1
    np.testing.assert_allclose(rom.solve(**free_plate_run())["T"][-1], 22.0, rtol=0, atol=1e-11)


def test_rom_build_a_basis_without_columns_is_refused_with_or_without_imposed_temperatures():
    with pytest.raises(ValueError):
        make_rom(**free_plate_arguments(basis=np.zeros((4, 0))))
    with pytest.raises(ValueError):
        make_rom(**side_arguments(basis=np.zeros((4, 0))))


def test_rom_build_a_basis_that_vanishes_on_the_fixed_nodes_up_to_rounding_is_accepted():
    for leak in (0.0, 1e-12, 1e-9):
        basis = np.array([[leak], [1.0], [-leak], [1.0]])
        assert make_rom(**side_arguments(basis=basis)).num_modes == 1


@pytest.mark.parametrize("leak", [1e-7, 1e-5, 1e-2])
def test_rom_build_a_basis_that_does_not_vanish_on_the_fixed_nodes_is_refused(leak):
    # more than 1e-8 times the largest entry of the basis
    with pytest.raises(ValueError):
        make_rom(**side_arguments(basis=np.array([[0.0], [1.0], [leak], [1.0]])))


def test_rom_build_the_tolerance_on_the_fixed_nodes_is_relative_to_the_largest_entry_of_the_basis():
    # the same shape, a million times larger: 1e-9 of the largest entry is still fine, 1e-7 is not
    assert make_rom(**side_arguments(basis=1e6 * np.array([[1e-9], [1.0], [0.0], [1.0]]))).num_modes == 1
    with pytest.raises(ValueError):
        make_rom(**side_arguments(basis=1e6 * np.array([[1e-7], [1.0], [0.0], [1.0]])))


def test_rom_build_the_tolerance_on_the_fixed_nodes_is_about_the_largest_entry_and_not_about_the_sum_of_the_entries(
        plate):
    # 30 free nodes and 6 modes: the sum of the absolute values is about a hundred times the largest entry
    basis = random_basis(plate, 6, 9)
    largest = np.abs(basis).max()
    node = plate.side("left")[0][1]
    basis[node, 2] = 1e-9 * largest
    assert plate.rom(basis).num_modes == 6
    basis[node, 2] = 1e-7 * largest
    with pytest.raises(ValueError):
        plate.rom(basis)


def test_rom_build_the_initial_temperature_may_differ_from_the_imposed_values_on_the_fixed_nodes():
    # the first step brings the fixed nodes to the imposed values in the full model; here only the projection counts
    expected = make_rom(**hot_side_arguments(T0=np.zeros(4))).solve(**side_run())
    assert expected["T"].shape == (3, 4)
    np.testing.assert_allclose(expected["T"][:, [0, 2]], 10.0, rtol=0, atol=0)


BUILD_VALUE_ERRORS = {
    "basis is a vector": dict(basis=np.array([0.0, 1.0, 0.0, 1.0])),
    "basis is a number": dict(basis=1.0),
    "basis has a row too many": dict(basis=np.ones((5, 1))),
    "basis has a row too few": dict(basis=np.ones((3, 1))),
    "basis has no columns": dict(basis=np.zeros((4, 0))),
    "basis has three axes": dict(basis=np.zeros((4, 1, 1))),
    "load has an entry too many": dict(load=np.zeros(5)),
    "load has an entry too few": dict(load=np.zeros(3)),
    "load is a number": dict(load=1.0),
    "load is a column": dict(load=np.zeros((4, 1))),
    "T0 has an entry too many": dict(T0=np.zeros(5)),
    "T0 has an entry too few": dict(T0=np.zeros(3)),
    "T0 is a number": dict(T0=1.0),
    "T0 has a row per state": dict(T0=np.zeros((2, 4))),
    "fixed_values has an entry too few": dict(fixed_values=[0.0]),
    "fixed_values has an entry too many": dict(fixed_values=[0.0, 0.0, 0.0]),
    "a fixed node outside the plate": dict(fixed_nodes=[0, 4]),
    "a negative fixed node": dict(fixed_nodes=[0, -1]),
    "a fixed node twice": dict(fixed_nodes=[2, 2]),
    "the basis does not vanish on a fixed node": dict(basis=np.array([[1.0], [1.0], [0.0], [1.0]])),
    "the basis does not vanish on the other fixed node": dict(basis=np.array([[0.0], [1.0], [0.5], [1.0]])),
    "the basis is zero": dict(basis=np.zeros((4, 1))),
    "two equal modes": dict(basis=np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])),
    "a mode that is zero next to a good one": dict(basis=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])),
    "nu is one half": dict(nu=0.5),
    "nu is above one half": dict(nu=0.7),
    "nu is minus one": dict(nu=-1.0),
    "nu is NaN": dict(nu=NAN),
    "only two degrees of freedom are fixed": dict(fixed_dofs=[0, 1]),
    "no degree of freedom is fixed": dict(fixed_dofs=[]),
    "a degree of freedom outside the plate": dict(fixed_dofs=[0, 1, 8]),
    "a degree of freedom twice": dict(fixed_dofs=[0, 1, 1]),
}


@pytest.mark.parametrize("change", list(BUILD_VALUE_ERRORS.values()), ids=list(BUILD_VALUE_ERRORS))
def test_rom_build_arguments_that_do_not_fit_are_a_value_error(change):
    with pytest.raises(ValueError):
        make_rom(**side_arguments(**change))


@pytest.mark.parametrize("bad", [NAN, INF])
def test_rom_build_a_nan_or_an_infinity_is_never_accepted_silently(bad):
    for name in ("T_ref", "fixed_values", "load", "T0", "basis"):
        arguments = side_arguments()
        value = arguments[name]
        if isinstance(value, float):
            arguments[name] = bad
        else:
            value = np.array(value, dtype=float)
            value[-1 if name != "basis" else (1, 0)] = bad
            arguments[name] = value
        with pytest.raises((ValueError, RuntimeError)):
            make_rom(**arguments).solve(**side_run())


BUILD_TYPE_ERRORS = {
    "nu is a string": dict(nu="steel"),
    "nu is None": dict(nu=None),
    "T_ref is a string": dict(T_ref="cold"),
    "T_ref is None": dict(T_ref=None),
    "fixed_dofs is None": dict(fixed_dofs=None),
    "a degree of freedom that is not an integer": dict(fixed_dofs=[0.5, 1, 3]),
    "fixed_nodes is None": dict(fixed_nodes=None),
    "a fixed node that is not an integer": dict(fixed_nodes=[0.5, 2]),
}


@pytest.mark.parametrize("change", list(BUILD_TYPE_ERRORS.values()), ids=list(BUILD_TYPE_ERRORS))
def test_rom_build_arguments_of_the_wrong_type_are_a_type_error(change):
    with pytest.raises(TypeError):
        make_rom(**side_arguments(**change))


def test_rom_build_a_negative_degree_of_freedom_is_refused():
    with pytest.raises((TypeError, ValueError)):
        make_rom(**side_arguments(fixed_dofs=[0, 1, -1]))


@pytest.mark.parametrize("nu", [0.0, 0.25, 0.4999, -0.9, np.float32(0.25), np.float64(0.3)])
def test_rom_build_accepts_the_poisson_ratios_of_the_mechanics(nu):
    assert make_rom(**side_arguments(nu=nu)).num_modes == 1


# ---------------------------------------------------------------- solve: the online phase

RUN = dict(k=0.8, rho_c=1.5, q=1.0, E=2.1e5, alpha=1.2e-5, dt=0.05, n_steps=6, theta=0.5)


def run(**change):
    arguments = dict(RUN)
    arguments.update(change)
    return arguments


SOLVE_SHAPES = [(0, 1), (0, 3), (1, 1), (6, 1), (6, 2), (6, 3), (6, 6), (7, 7)]


@pytest.mark.parametrize("n_steps, save_every", SOLVE_SHAPES)
def test_rom_solve_returns_the_states_every_save_every_steps_with_the_keys_and_shapes_of_the_cpp_solver(
        plate, n_steps, save_every):
    result = plate.rom(random_basis(plate, 4, 1)).solve(**run(n_steps=n_steps, save_every=save_every))
    m = n_steps // save_every + 1
    assert set(result) == KEYS
    shapes = dict(time=(m,), T=(m, plate.n), u=(m, 2 * plate.n), sigma=(m, plate.n_tri, 3), von_mises=(m, plate.n_tri))
    for key, shape in shapes.items():
        assert result[key].shape == shape, key
        assert result[key].dtype == np.float64, key
    np.testing.assert_allclose(result["time"], np.arange(0, n_steps + 1, save_every) * 0.05, rtol=1e-14, atol=0)


def test_rom_solve_the_result_is_a_dict_and_save_every_is_one_by_default(plate):
    rom = plate.rom(random_basis(plate, 3, 2))
    result = rom.solve(**run())
    assert isinstance(result, dict)
    assert_same_run(rom.solve(**run(), save_every=1), result)
    assert result["T"].shape[0] == 7


@pytest.mark.parametrize("save_every", [2, 3, 6])
def test_rom_solve_the_saved_states_are_the_rows_of_the_run_that_saves_every_step(plate, save_every):
    rom = plate.rom(random_basis(plate, 5, 2))
    every = rom.solve(**run())
    some = rom.solve(**run(save_every=save_every))
    for key in every:
        np.testing.assert_allclose(some[key], every[key][::save_every], rtol=1e-12, atol=1e-12, err_msg=key)


def test_rom_solve_without_steps_is_the_state_that_the_model_sees_of_the_initial_temperature(plate):
    basis = random_basis(plate, 5, 3)
    nodes, values, lift, start = plate.side("left")
    result = plate.rom(basis).solve(**run(n_steps=0))
    expected = reduced_heat(basis, plate.mass, plate.stiffness, plate.load, nodes, values, start, 0.05, 0, 0.5)
    np.testing.assert_allclose(result["T"], expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result["time"], [0.0], rtol=0, atol=0)
    assert result["T"].shape == (1, plate.n) and result["sigma"].shape == (1, plate.n_tri, 3)


def test_rom_solve_arguments_can_be_given_by_position_in_the_documented_order(plate):
    rom = plate.rom(random_basis(plate, 4, 4))
    arguments = run(save_every=2)
    order = ["k", "rho_c", "q", "E", "alpha", "dt", "n_steps", "theta", "save_every"]
    assert_same_run(rom.solve(*[arguments[name] for name in order]), rom.solve(**arguments))


EXACT = dict(k=2.0, rho_c=4.0, q=8.0, E=64.0, alpha=0.5, dt=1.0, n_steps=2, theta=1.0)


@pytest.mark.parametrize("make", [float, np.float64, np.float32])
def test_rom_solve_accepts_the_parameters_of_any_float_type(make):
    # the numbers are exact in every type
    rom = make_rom(**side_arguments())
    expected = rom.solve(**EXACT)
    arguments = {name: (value if name == "n_steps" else make(value)) for name, value in EXACT.items()}
    assert_same_run(rom.solve(**arguments), expected)


@pytest.mark.parametrize("make", [int, np.int8, np.int32, np.int64, np.uint16])
def test_rom_solve_accepts_the_parameters_of_any_integer_type(make):
    rom = make_rom(**side_arguments())
    integers = dict(k=2, rho_c=4, q=8, E=64, alpha=1, dt=1, n_steps=2, theta=1, save_every=2)
    expected = rom.solve(**{name: (value if name in ("n_steps", "save_every") else float(value))
                            for name, value in integers.items()})
    result = rom.solve(**{name: make(value) for name, value in integers.items()})
    assert_same_run(result, expected)
    assert all(result[key].dtype == np.float64 for key in result)


def test_rom_solve_the_results_are_new_arrays_and_changing_them_does_not_change_the_model(plate):
    rom = plate.rom(random_basis(plate, 4, 4))
    expected = rom.solve(**run())
    spoiled = rom.solve(**run())
    for key in spoiled:
        spoiled[key][...] = -123.0
    assert_same_run(rom.solve(**run()), expected)


def test_rom_solve_two_results_and_the_five_arrays_of_a_result_share_no_memory(plate):
    rom = plate.rom(random_basis(plate, 4, 4))
    a, b = rom.solve(**run()), rom.solve(**run())
    for key in a:
        assert not np.shares_memory(a[key], b[key]), key
        for other in a:
            if other != key:
                assert not np.shares_memory(a[key], a[other]), (key, other)


def test_rom_solve_is_a_pure_function_of_its_arguments(plate):
    # whatever was solved in between, the same arguments give the same answer, bit for bit
    basis = random_basis(plate, 5, 6)
    rom = plate.rom(basis)
    first = rom.solve(**run())
    other = run(k=2.2, rho_c=0.9, q=-0.7, E=5e4, alpha=3e-5, theta=1.0, n_steps=4, save_every=2)
    rom.solve(**other)
    rom.solve(**run(theta=0.0, dt=0.001, E=1.0, alpha=1.0))
    assert_same_run(rom.solve(**run()), first)
    assert_same_run(rom.solve(**other), plate.rom(basis).solve(**other))


def test_rom_solve_does_not_change_the_arrays_of_the_model(plate):
    rom = plate.rom(random_basis(plate, 5, 6))
    before = {name: value.copy() for name, value in stored_values(rom).items()}
    assert before
    rom.solve(**run(k=3.0, rho_c=0.4, q=2.5, E=7e4, alpha=2e-5, theta=0.0, dt=0.001))
    rom.solve(**run())
    after = stored_values(rom)
    assert set(after) == set(before)
    for name, value in before.items():
        np.testing.assert_array_equal(after[name], value, err_msg=name)


def test_rom_solve_two_models_do_not_interfere(plate):
    first, second = random_basis(plate, 3, 10), random_basis(plate, 6, 11)
    a, b = plate.rom(first), plate.rom(second)
    expected_a, expected_b = a.solve(**run()), b.solve(**run(theta=1.0))
    c = plate.rom(random_basis(plate, 2, 12), holds="rollers", nu=0.1, T_ref=30.0)
    c.solve(**run(k=1.7))
    assert a.num_modes == 3 and b.num_modes == 6 and c.num_modes == 2
    assert_same_run(a.solve(**run()), expected_a)
    assert_same_run(b.solve(**run(theta=1.0)), expected_b)


SOLVE_VALUE_ERRORS = {
    "k is zero": dict(k=0.0),
    "k is negative": dict(k=-1.0),
    "k is NaN": dict(k=NAN),
    "k is infinite": dict(k=INF),
    "rho_c is zero": dict(rho_c=0.0),
    "rho_c is negative": dict(rho_c=-3.0),
    "rho_c is NaN": dict(rho_c=NAN),
    "rho_c is infinite": dict(rho_c=INF),
    "E is zero": dict(E=0.0),
    "E is negative": dict(E=-100.0),
    "E is NaN": dict(E=NAN),
    "E is infinite": dict(E=INF),
    "q is NaN": dict(q=NAN),
    "q is infinite": dict(q=INF),
    "q is minus infinity": dict(q=-INF),
    "alpha is NaN": dict(alpha=NAN),
    "alpha is infinite": dict(alpha=INF),
    "alpha is minus infinity": dict(alpha=-INF),
    "dt is zero": dict(dt=0.0),
    "dt is negative": dict(dt=-0.5),
    "dt is NaN": dict(dt=NAN),
    "dt is infinite": dict(dt=INF),
    "theta is below zero": dict(theta=-0.1),
    "theta is above one": dict(theta=1.1),
    "theta is NaN": dict(theta=NAN),
    "n_steps is negative": dict(n_steps=-1),
    "n_steps is negative and a multiple of save_every": dict(n_steps=-4, save_every=2),
    "save_every is zero": dict(save_every=0),
    "save_every is negative": dict(save_every=-1),
    "save_every does not divide n_steps": dict(n_steps=3, save_every=2),
    "save_every is larger than n_steps": dict(n_steps=2, save_every=3),
    "dt is zero without steps": dict(dt=0.0, n_steps=0),
    "k is zero without steps": dict(k=0.0, n_steps=0),
    "theta is above one without steps": dict(theta=2.0, n_steps=0),
}


@pytest.mark.parametrize("change", list(SOLVE_VALUE_ERRORS.values()), ids=list(SOLVE_VALUE_ERRORS))
def test_rom_solve_arguments_that_do_not_fit_are_a_value_error(change):
    with pytest.raises(ValueError):
        make_rom(**side_arguments()).solve(**side_run(**change))


SOLVE_TYPE_ERRORS = {
    "n_steps is a float": dict(n_steps=2.0),
    "n_steps is a string": dict(n_steps="2"),
    "n_steps is None": dict(n_steps=None),
    "save_every is a float": dict(save_every=1.0),
    "save_every is None": dict(save_every=None),
    "k is a string": dict(k="2"),
    "k is None": dict(k=None),
    "rho_c is None": dict(rho_c=None),
    "q is None": dict(q=None),
    "E is None": dict(E=None),
    "alpha is None": dict(alpha=None),
    "alpha is a string": dict(alpha="steel"),
    "dt is None": dict(dt=None),
    "theta is None": dict(theta=None),
    "theta is a string": dict(theta="half"),
}


@pytest.mark.parametrize("change", list(SOLVE_TYPE_ERRORS.values()), ids=list(SOLVE_TYPE_ERRORS))
def test_rom_solve_arguments_of_the_wrong_type_are_a_type_error(change):
    with pytest.raises(TypeError):
        make_rom(**side_arguments()).solve(**side_run(**change))


def test_rom_solve_a_refused_call_leaves_the_model_as_it_was(plate):
    rom = plate.rom(random_basis(plate, 4, 4))
    expected = rom.solve(**run())
    for change in SOLVE_VALUE_ERRORS.values():
        with pytest.raises(ValueError):
            rom.solve(**run(**change))
    assert_same_run(rom.solve(**run()), expected)


@pytest.mark.parametrize("q, alpha", [(-2.0, 1.2e-5), (2.0, -1.2e-5), (-2.0, -1.2e-5), (0.0, 1.2e-5), (2.0, 0.0)])
def test_rom_solve_accepts_a_cold_source_and_a_material_that_contracts_when_heated(plate, q, alpha):
    result = plate.rom(random_basis(plate, 4, 4)).solve(**run(q=q, alpha=alpha))
    for key in KEYS:
        assert np.isfinite(result[key]).all(), key


# ---------------------------------------------------------------- against the reduced model built from scratch

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


def scratch_run(plate, basis, side, holds, nu, T_ref, k, rho_c, q, E, alpha, dt, n_steps, theta, start=None):
    """What the model must give, computed the long way: the matrices of these parameters are assembled in full and
    projected on the basis (step 7b), the mechanical modes are those of this very material (step 7c)."""
    nodes, values, lift, default_start = plate.side(side)
    start = default_start if start is None else start
    dofs = plate.holds(holds)
    mass = thermomech.assemble_mass_2d(plate.mesh, rho_c)
    stiffness = thermomech.assemble_stiffness_2d(plate.mesh, k)
    T = reduced_heat(basis, mass, stiffness, q * plate.load, nodes, values, start, dt, n_steps, theta)
    coordinates = np.linalg.lstsq(basis, (T - lift).T, rcond=None)[0].T  # every state is the lifting + basis a
    u_lift, u_modes, s_lift, s_modes = modes(plate.mesh, basis, lift, E, nu, alpha, T_ref, dofs, [0.0] * len(dofs))
    u = rebuild(u_modes, coordinates, u_lift)
    sigma = rebuild(s_modes, coordinates, s_lift).reshape(n_steps + 1, plate.n_tri, 3)
    rise = mean_of(plate.mesh.triangles, T) - T_ref
    return dict(time=np.arange(n_steps + 1) * dt, T=T, u=u, sigma=sigma, von_mises=von_mises(sigma, rise, E, nu, alpha))


MATERIAL_CASES = [(0.3, 22.0), (0.45, 26.5), (-0.2, 15.0)]
MATERIAL_IDS = ["steel-like", "nearly-incompressible", "auxetic"]


@pytest.mark.parametrize("nu, T_ref", MATERIAL_CASES, ids=MATERIAL_IDS)
@pytest.mark.parametrize("holds", CONSTRAINTS)
@pytest.mark.parametrize("theta, factor", THETAS, ids=THETA_IDS)
def test_rom_scratch_agrees_with_the_reduced_model_built_from_scratch_for_any_parameters(
        plate, theta, factor, holds, nu, T_ref):
    parameters = random_parameters(int(100 * theta) + len(holds) + int(100 * (nu + 1.0)))
    dt = factor * parameters["rho_c"] / parameters["k"]
    basis = random_basis(plate, 6, 21)
    got = plate.rom(basis, holds=holds, nu=nu, T_ref=T_ref).solve(**parameters, dt=dt, n_steps=5, theta=theta)
    expected = scratch_run(plate, basis, "left", holds, nu, T_ref, **parameters, dt=dt, n_steps=5, theta=theta)
    assert_close_runs(got, expected)


@pytest.mark.parametrize("theta, factor", THETAS, ids=THETA_IDS)
def test_rom_scratch_the_same_with_the_temperature_imposed_on_the_whole_border(plate, theta, factor):
    parameters = random_parameters(40 + int(10 * theta))
    dt = factor * parameters["rho_c"] / parameters["k"]
    basis = random_basis(plate, 5, 22, side="border")
    got = plate.rom(basis, side="border", holds="rollers").solve(**parameters, dt=dt, n_steps=4, theta=theta)
    expected = scratch_run(plate, basis, "border", "rollers", 0.3, T_REF, **parameters, dt=dt, n_steps=4, theta=theta)
    assert_close_runs(got, expected)


@pytest.mark.parametrize("r", [1, 2, 5, 12, 30])
def test_rom_scratch_any_number_of_modes(plate, r):
    parameters = random_parameters(60 + r)
    basis = random_basis(plate, r, 23)
    got = plate.rom(basis).solve(**parameters, dt=0.05, n_steps=4, theta=0.5)
    expected = scratch_run(plate, basis, "left", "pins", 0.3, T_REF, **parameters, dt=0.05, n_steps=4, theta=0.5)
    assert_close_runs(got, expected)


@pytest.mark.parametrize("seed", range(4))
def test_rom_scratch_one_model_answers_many_parameters_like_a_model_made_for_each_of_them(plate, seed):
    # the offline phase is done once: the model made for the first parameters gives the answers for all the others
    basis = random_basis(plate, 6, 24)
    rom = plate.rom(basis)
    for index in range(3):
        parameters = random_parameters(100 + 3 * seed + index)
        got = rom.solve(**parameters, dt=0.05, n_steps=3, theta=0.5)
        expected = scratch_run(plate, basis, "left", "pins", 0.3, T_REF, **parameters, dt=0.05, n_steps=3, theta=0.5)
        assert_close_runs(got, expected)


@pytest.mark.parametrize("start_kind", ["lifting", "random"])
def test_rom_scratch_with_other_initial_temperatures(plate, start_kind):
    nodes, values, lift, _ = plate.side("left")
    start = lift.copy() if start_kind == "lifting" else 15.0 + 20.0 * np.random.default_rng(5).random(plate.n)
    start[nodes] = values
    basis = random_basis(plate, 7, 25)
    parameters = random_parameters(7)
    got = plate.rom(basis, start=start).solve(**parameters, dt=0.05, n_steps=4, theta=0.5)
    expected = scratch_run(plate, basis, "left", "pins", 0.3, T_REF, **parameters, dt=0.05, n_steps=4, theta=0.5,
                           start=start)
    assert_close_runs(got, expected)


@pytest.mark.parametrize("holds", CONSTRAINTS)
@pytest.mark.parametrize("nu, T_ref", MATERIAL_CASES[:2], ids=MATERIAL_IDS[:2])
def test_rom_scratch_the_displacements_and_the_stresses_are_the_mechanics_of_the_temperatures(plate, holds, nu, T_ref):
    # every state, with dense matrices that know nothing of the C++ solver, nor of the modes
    parameters = random_parameters(9)
    result = plate.rom(random_basis(plate, 6, 26), holds=holds, nu=nu, T_ref=T_ref).solve(
        **parameters, dt=0.05, n_steps=4, theta=0.5)
    for state in range(5):
        u, sigma = dense_response(plate.mesh, result["T"][state], parameters["E"], nu, parameters["alpha"], T_ref,
                                  plate.holds(holds))
        rise = np.asarray(plate.mesh.triangles)
        rise = result["T"][state][rise].mean(axis=1) - T_ref
        assert relative_error(result["u"][state], u) < 1e-8, state
        assert relative_error(result["sigma"][state], sigma) < 1e-7, state
        vm = von_mises_by_the_deviator(sigma, rise, parameters["E"], nu, parameters["alpha"])
        assert relative_error(result["von_mises"][state], vm) < 1e-7, state


# ---------------------------------------------------------------- against the C++ solver


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("holds", CONSTRAINTS)
@pytest.mark.parametrize("theta, factor", THETAS, ids=THETA_IDS)
def test_rom_full_with_a_mode_for_every_free_node_the_model_is_the_full_model_for_any_parameters(
        plate, theta, factor, holds, seed):
    # no reduction error is possible, so every difference with the C++ solver is a mistake of the parameters
    parameters = random_parameters(seed)
    dt = factor * parameters["rho_c"] / parameters["k"]
    rom = plate.rom(plate.nodal_basis(plate.side("left")[0]), holds=holds)
    got = rom.solve(**parameters, dt=dt, n_steps=4, theta=theta)
    expected = plate.full("left", holds, **parameters, dt=dt, n_steps=4, theta=theta)
    assert_close_cpp(got, expected)


@pytest.mark.parametrize("theta, factor", THETAS, ids=THETA_IDS)
def test_rom_full_the_same_with_the_temperature_imposed_on_the_whole_border(plate, theta, factor):
    parameters = random_parameters(30)
    dt = factor * parameters["rho_c"] / parameters["k"]
    rom = plate.rom(plate.nodal_basis(plate.border), side="border", holds="pins")
    got = rom.solve(**parameters, dt=dt, n_steps=4, theta=theta)
    assert_close_cpp(got, plate.full("border", "pins", **parameters, dt=dt, n_steps=4, theta=theta))


@pytest.mark.parametrize("n_steps, save_every", [(0, 1), (0, 4), (4, 1), (4, 2), (6, 3), (6, 6)])
def test_rom_full_the_saved_states_are_those_of_the_cpp_solver(plate, n_steps, save_every):
    parameters = random_parameters(3)
    rom = plate.rom(plate.nodal_basis(plate.side("left")[0]))
    got = rom.solve(**parameters, dt=0.02, n_steps=n_steps, theta=0.5, save_every=save_every)
    expected = plate.full("left", "pins", **parameters, dt=0.02, n_steps=n_steps, theta=0.5, save_every=save_every)
    assert_close_cpp(got, expected)


@pytest.mark.parametrize("seed", [0, 1])
def test_rom_full_with_any_initial_temperature(plate, seed):
    nodes, values, _, _ = plate.side("left")
    start = 10.0 + 30.0 * np.random.default_rng(seed).random(plate.n)
    start[nodes] = values
    parameters = random_parameters(11 + seed)
    rom = plate.rom(plate.nodal_basis(nodes), start=start)
    got = rom.solve(**parameters, dt=0.05, n_steps=3, theta=0.5)
    assert_close_cpp(got, plate.full("left", "pins", **parameters, dt=0.05, n_steps=3, theta=0.5, start=start))
    np.testing.assert_allclose(got["T"][0], start, rtol=0, atol=1e-9)


@pytest.mark.parametrize("nu, T_ref", MATERIAL_CASES, ids=MATERIAL_IDS)
def test_rom_full_the_poisson_ratio_and_the_reference_temperature_of_the_model_are_those_of_the_solver(
        plate, nu, T_ref):
    parameters = random_parameters(5)
    rom = plate.rom(plate.nodal_basis(plate.side("left")[0]), nu=nu, T_ref=T_ref)
    got = rom.solve(**parameters, dt=0.05, n_steps=3, theta=0.5)
    assert_close_cpp(got, plate.full("left", "pins", **parameters, dt=0.05, n_steps=3, theta=0.5, nu=nu, T_ref=T_ref))


def trajectory_basis(kind, plate, states, lift):
    """Modes that contain the deviations T - lifting of the states: orthonormalized with QR or the POD of step 7a."""
    snapshots = (states - lift).T
    if kind == "qr":
        return np.linalg.qr(snapshots)[0]
    sigma = pod(snapshots, plate.mass, 1)[1]
    return pod(snapshots, plate.mass, int(np.sum(sigma > 1e-6 * sigma[0])))[0]


@pytest.mark.parametrize("kind", ["qr", "pod"])
@pytest.mark.parametrize("theta, dt", [(0.0, 0.01), (0.5, 0.05), (2.0 / 3.0, 0.05), (1.0, 0.05)], ids=THETA_IDS)
@pytest.mark.parametrize("holds", CONSTRAINTS)
@pytest.mark.parametrize("side", SIDES)
def test_rom_full_a_basis_that_contains_the_trajectory_reproduces_the_run_of_the_full_model(
        plate, side, holds, theta, dt, kind):
    # four modes for the four states: the reduced model is exact, in the temperatures and in the mechanics
    lift = plate.side(side)[2]
    base = dict(k=BASE_K, rho_c=BASE_RHO_C, q=1.0, E=STEEL["E"], alpha=STEEL["alpha"])
    full = plate.full(side, holds, **base, dt=dt, n_steps=3, theta=theta, nu=STEEL["nu"])
    basis = trajectory_basis(kind, plate, full["T"], lift)
    assert basis.shape[1] == 4
    got = plate.rom(basis, side=side, holds=holds, nu=STEEL["nu"]).solve(**base, dt=dt, n_steps=3, theta=theta)
    assert_close_cpp(got, full)


@pytest.mark.parametrize("E, alpha", [(100.0, 0.01), (2.1e5, 1.2e-5), (1.0e4, 5.0e-5), (5.0e5, 1.0e-6), (3.0, 2.0),
                                      (8.0e4, -3.0e-5)])
def test_rom_full_one_basis_serves_every_material_because_the_heat_does_not_depend_on_it(plate, E, alpha):
    lift = plate.side("left")[2]
    base = dict(k=BASE_K, rho_c=BASE_RHO_C, q=1.0)
    full = plate.full("left", "pins", **base, E=E, alpha=alpha, dt=0.05, n_steps=3, theta=0.5)
    reference = plate.full("left", "pins", **base, E=STEEL["E"], alpha=STEEL["alpha"], dt=0.05, n_steps=3, theta=0.5)
    basis = trajectory_basis("qr", plate, reference["T"], lift)
    got = plate.rom(basis).solve(**base, E=E, alpha=alpha, dt=0.05, n_steps=3, theta=0.5)
    assert_close_cpp(got, full)


@pytest.mark.parametrize("c", [0.25, 2.0, 7.0])
def test_rom_full_the_three_heat_parameters_multiplied_together_do_not_change_the_states(plate, c):
    # (k, rho_c, q) -> c (k, rho_c, q) multiplies M_r, K_r and f_r by c: the same equations, the same states
    lift = plate.side("left")[2]
    base = dict(k=BASE_K, rho_c=BASE_RHO_C, q=1.0, E=STEEL["E"], alpha=STEEL["alpha"])
    basis = trajectory_basis("qr", plate, plate.full("left", "pins", **base, dt=0.05, n_steps=3, theta=0.5)["T"], lift)
    scaled = dict(base, k=c * BASE_K, rho_c=c * BASE_RHO_C, q=c * 1.0)
    got = plate.rom(basis).solve(**scaled, dt=0.05, n_steps=3, theta=0.5)
    assert_close_cpp(got, plate.full("left", "pins", **scaled, dt=0.05, n_steps=3, theta=0.5))


@pytest.mark.parametrize("c", [0.5, 2.0, 4.0])
def test_rom_full_a_faster_physics_and_a_shorter_step_give_the_same_states_at_shorter_times(plate, c):
    # (k, q) -> c (k, q) and dt -> dt / c: the products dt K_r and dt f_r do not change
    lift = plate.side("left")[2]
    base = dict(k=BASE_K, rho_c=BASE_RHO_C, q=1.0, E=STEEL["E"], alpha=STEEL["alpha"])
    basis = trajectory_basis("qr", plate, plate.full("left", "pins", **base, dt=0.05, n_steps=3, theta=0.5)["T"], lift)
    faster = dict(base, k=c * BASE_K, q=c * 1.0)
    got = plate.rom(basis).solve(**faster, dt=0.05 / c, n_steps=3, theta=0.5)
    assert_close_cpp(got, plate.full("left", "pins", **faster, dt=0.05 / c, n_steps=3, theta=0.5))


@pytest.mark.parametrize("q", [-2.0, 0.0, 0.5, 7.0])
def test_rom_full_the_strength_of_the_source_is_free_when_the_basis_has_two_runs_with_other_strengths(plate, q):
    # the deviation from the lifting is affine in q, so two runs (q = 1 and q = 3) contain the states for every q
    lift = plate.side("left")[2]
    base = dict(k=BASE_K, rho_c=BASE_RHO_C, E=STEEL["E"], alpha=STEEL["alpha"])
    runs = [plate.full("left", "pins", **base, q=strength, dt=0.05, n_steps=3, theta=0.5)["T"]
            for strength in (1.0, 3.0)]
    columns = [runs[0][0] - lift] + [run_states[step] - lift for run_states in runs for step in (1, 2, 3)]
    basis = np.linalg.qr(np.column_stack(columns))[0]
    got = plate.rom(basis).solve(**base, q=q, dt=0.05, n_steps=3, theta=0.5)
    assert_close_cpp(got, plate.full("left", "pins", **base, q=q, dt=0.05, n_steps=3, theta=0.5))


# ---------------------------------------------------------------- the laws that the parameters obey


def law_model(plate, r=6, seed=51):
    return plate.rom(random_basis(plate, r, seed))


def assert_same_states(result, expected, error=1e-11):
    """The same temperatures, displacements and stresses, to rounding; the times are not compared."""
    assert np.abs(result["T"] - expected["T"]).max() < error, "the temperatures"
    for key in ("u", "sigma", "von_mises"):
        assert relative_error(result[key], expected[key]) < error, key


def test_rom_laws_the_temperatures_do_not_depend_on_the_young_modulus_nor_on_the_expansion(plate):
    rom = law_model(plate)
    soft = rom.solve(**run(E=1.0e3, alpha=1.0e-6))
    stiff = rom.solve(**run(E=7.0e5, alpha=9.0e-5))
    np.testing.assert_array_equal(soft["T"], stiff["T"])
    np.testing.assert_array_equal(soft["time"], stiff["time"])


@pytest.mark.parametrize("c", [0.01, 3.0, 250.0])
def test_rom_laws_the_displacements_are_proportional_to_alpha_and_do_not_depend_on_E(plate, c):
    rom = law_model(plate)
    base = rom.solve(**run())
    assert relative_error(rom.solve(**run(alpha=c * RUN["alpha"]))["u"], c * base["u"]) < 1e-11
    assert relative_error(rom.solve(**run(E=c * RUN["E"]))["u"], base["u"]) < 1e-11


@pytest.mark.parametrize("c_E, c_alpha", [(2.0, 1.0), (1.0, 3.0), (0.1, 10.0), (5.0, 0.2)])
def test_rom_laws_the_stresses_and_the_von_mises_stress_are_proportional_to_E_times_alpha(plate, c_E, c_alpha):
    rom = law_model(plate)
    base = rom.solve(**run())
    got = rom.solve(**run(E=c_E * RUN["E"], alpha=c_alpha * RUN["alpha"]))
    assert relative_error(got["sigma"], c_E * c_alpha * base["sigma"]) < 1e-11
    assert relative_error(got["von_mises"], c_E * c_alpha * base["von_mises"]) < 1e-11
    assert relative_error(got["u"], c_alpha * base["u"]) < 1e-11


def test_rom_laws_without_thermal_expansion_there_is_no_displacement_and_no_stress(plate):
    rom = law_model(plate)
    got = rom.solve(**run(alpha=0.0))
    for key in ("u", "sigma", "von_mises"):
        assert not got[key].any(), key
    np.testing.assert_array_equal(got["T"], rom.solve(**run())["T"])


def test_rom_laws_a_negative_expansion_reverses_the_displacements_and_the_stresses_but_not_the_von_mises_stress(plate):
    rom = law_model(plate)
    base = rom.solve(**run())
    got = rom.solve(**run(alpha=-RUN["alpha"]))
    assert relative_error(got["u"], -base["u"]) < 1e-11
    assert relative_error(got["sigma"], -base["sigma"]) < 1e-11
    assert relative_error(got["von_mises"], base["von_mises"]) < 1e-11


@pytest.mark.parametrize("c", [0.25, 3.0, 40.0])
def test_rom_laws_k_rho_c_and_q_multiplied_by_the_same_number_change_nothing(plate, c):
    rom = law_model(plate)
    base = rom.solve(**run())
    got = rom.solve(**run(k=c * RUN["k"], rho_c=c * RUN["rho_c"], q=c * RUN["q"]))
    assert_same_states(got, base)
    np.testing.assert_array_equal(got["time"], base["time"])


@pytest.mark.parametrize("c", [0.5, 2.0, 8.0])
def test_rom_laws_a_physics_c_times_faster_and_a_step_c_times_shorter_give_the_same_states_at_shorter_times(plate, c):
    rom = law_model(plate)
    base = rom.solve(**run())
    got = rom.solve(**run(k=c * RUN["k"], q=c * RUN["q"], dt=RUN["dt"] / c))
    assert_same_states(got, base)
    np.testing.assert_allclose(got["time"], base["time"] / c, rtol=1e-13, atol=0)


def nothing_imposed(plate, r=5, seed=52):
    """Zero on the left side, zero at the start and zero as reference: the only thing that can happen is the source."""
    nodes = plate.left
    return make_rom(plate.mesh, random_basis(plate, r, seed), plate.load, nodes, [0.0] * len(nodes), np.zeros(plate.n),
                    plate.holds("pins"), 0.3, 0.0)


@pytest.mark.parametrize("q", [2.5, -1.5, 1.0e3])
def test_rom_laws_with_nothing_imposed_the_response_is_proportional_to_the_source(plate, q):
    rom = nothing_imposed(plate)
    unit = rom.solve(**run(q=1.0))
    got = rom.solve(**run(q=q))
    for key in ("T", "u", "sigma"):
        assert relative_error(got[key], q * unit[key]) < 1e-11, key
    assert relative_error(got["von_mises"], abs(q) * unit["von_mises"]) < 1e-11


def test_rom_laws_with_nothing_imposed_and_no_source_nothing_happens(plate):
    got = nothing_imposed(plate).solve(**run(q=0.0))
    for key in ("T", "u", "sigma", "von_mises"):
        assert not got[key].any(), key


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("c0", [31.5, 5.0, 20.0])
def test_rom_laws_a_plate_at_a_uniform_temperature_stays_there_and_expands_freely(c0, theta):
    # 20 is the reference temperature. The mode is 1 on the free nodes, the lifting is c0 on the left side: T = c0
    # whatever k, rho_c and the scheme are, because the heat of the side exactly balances the lifting term
    arguments = side_arguments(fixed_values=[c0, c0], T0=np.full(4, c0))
    result = make_rom(**arguments).solve(**side_run(q=0.0, theta=theta))
    rise = c0 - HAND["T_ref"]
    np.testing.assert_allclose(result["T"], c0, rtol=0, atol=1e-12)
    for state in range(3):
        np.testing.assert_allclose(result["u"][state], (1.0 + HAND["nu"]) * HAND["alpha"] * rise * np.array(EXPANSION),
                                   rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["sigma"], 0.0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result["von_mises"], HAND["E"] * HAND["alpha"] * abs(rise), rtol=0, atol=1e-10)


def test_rom_laws_the_von_mises_stress_is_that_of_the_stresses_and_of_the_mean_temperature_of_every_triangle(plate):
    nu, T_ref = 0.28, 24.0
    got = plate.rom(random_basis(plate, 6, 53), nu=nu, T_ref=T_ref).solve(**run())
    rise = mean_of(np.asarray(plate.mesh.triangles), got["T"]) - T_ref
    expected = von_mises(got["sigma"], rise, RUN["E"], nu, RUN["alpha"])
    np.testing.assert_allclose(got["von_mises"], expected, rtol=1e-12, atol=0)


# ---------------------------------------------------------------- offline and online: what is computed where


def forbidden(*args, **kwargs):
    raise AssertionError("the online phase called the C++ core")


@pytest.mark.parametrize("n_steps, save_every", [(0, 1), (6, 1), (6, 3)])
def test_rom_split_the_online_phase_does_not_call_the_cpp_core(plate, monkeypatch, n_steps, save_every):
    rom = plate.rom(random_basis(plate, 5, 61))
    expected = rom.solve(**run(n_steps=n_steps, save_every=save_every))
    names = [name for name in dir(thermomech) if not name.startswith("_") and callable(getattr(thermomech, name))]
    assert "solve_thermomechanical_2d" in names and "assemble_mass_2d" in names
    for name in names:
        monkeypatch.setattr(thermomech, name, forbidden)
    assert_same_run(rom.solve(**run(n_steps=n_steps, save_every=save_every)), expected)


@pytest.mark.parametrize("r", [1, 4, 9])
def test_rom_split_the_model_keeps_no_matrix_of_the_size_of_the_mesh(plate, r):
    stored = stored_values(plate.rom(random_basis(plate, r, 62)))
    assert stored
    for name, value in stored.items():
        assert not sp.issparse(value), name
        assert not (value.ndim >= 2 and value.shape[0] >= plate.n and value.shape[1] >= plate.n), name


@pytest.mark.parametrize("r", [1, 4, 9])
def test_rom_split_the_offline_phase_makes_at_most_one_mechanical_solve_for_the_lifting_and_one_for_every_mode(
        plate, monkeypatch, r):
    # the mechanics of every other material is a multiple of the one for E = 1 and alpha = 1
    calls = []
    real = thermomech.solve_thermomechanical_2d

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(thermomech, "solve_thermomechanical_2d", counting)
    plate.rom(random_basis(plate, r, 63))
    assert len(calls) <= r + 1


# ---------------------------------------------------------------- modes from the C++ solver: new parameters

TRAINING = [(0.5, 1.0, 0.5), (1.2, 1.0, 2.0), (0.5, 3.0, 2.0), (1.2, 3.0, 0.5), (0.8, 1.5, 1.0)]  # (k, rho_c, q)
HELD_OUT = [dict(k=0.65, rho_c=1.8, q=1.3, E=1.9e5, alpha=2.5e-5), dict(k=1.0, rho_c=2.5, q=0.8, E=7.0e4, alpha=2.3e-5),
            dict(k=0.9, rho_c=1.2, q=1.7, E=2.1e5, alpha=1.2e-5)]
POD_STEPS = 40


@pytest.fixture(scope="module")
def trained(plate):
    """Twelve POD modes of the five runs of the C++ solver, and the C++ run of every set of held-out parameters."""
    lift = plate.side("left")[2]
    runs = [plate.full("left", "pins", k=k, rho_c=rho_c, q=q, E=2.1e5, alpha=1.2e-5, dt=0.05, n_steps=POD_STEPS,
                       theta=0.5)["T"] for k, rho_c, q in TRAINING]
    basis = pod(np.hstack([(states - lift).T for states in runs]), plate.mass, 12)[0]
    held = [plate.full("left", "pins", **parameters, dt=0.05, n_steps=POD_STEPS, theta=0.5) for parameters in HELD_OUT]
    return basis, held


def pod_errors(plate, basis, parameters, full):
    got = plate.rom(basis).solve(**parameters, dt=0.05, n_steps=POD_STEPS, theta=0.5)
    return dict(T=np.abs(got["T"] - full["T"]).max() / np.abs(full["T"] - plate.side("left")[2]).max(),
                u=relative_error(got["u"], full["u"]), sigma=relative_error(got["sigma"], full["sigma"]),
                von_mises=relative_error(got["von_mises"], full["von_mises"]))


POD_LIMITS = dict(T=2e-4, u=4e-4, sigma=2e-3, von_mises=1e-3)


@pytest.mark.parametrize("index", range(len(HELD_OUT)))
def test_rom_pod_twelve_modes_of_five_runs_reproduce_a_run_with_other_parameters(plate, trained, index):
    basis, held = trained
    errors = pod_errors(plate, basis, HELD_OUT[index], held[index])
    for key, limit in POD_LIMITS.items():
        assert errors[key] < limit, "%s: %.2e" % (key, errors[key])


def test_rom_pod_every_four_more_modes_make_the_errors_at_least_five_times_smaller(plate, trained):
    basis, held = trained
    errors = {r: [pod_errors(plate, basis[:, :r], parameters, full) for parameters, full in zip(HELD_OUT, held)]
              for r in (4, 8, 12)}
    for key in POD_LIMITS:
        worst = {r: max(errors[r][index][key] for index in range(len(HELD_OUT))) for r in errors}
        assert worst[8] < 0.2 * worst[4], key
        assert worst[12] < 0.2 * worst[8], key
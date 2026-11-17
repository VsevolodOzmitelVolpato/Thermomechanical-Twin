"""Step 8b: from the ROM to the numbers that the Sobol analysis needs, from twin/uq.py.

    from twin.uq import PARAMETER_NAMES, QUANTITY_NAMES, rom_quantities

The question. The Sobol analysis of step 8a asks a model for numbers: it hands the model a table of points of the
parameter space (one point per row) and wants one row of numbers back for every point. The reduced-order model (ROM) of
step 7 answers another question: for ONE set of five numbers it returns whole fields (temperature, displacements and
stresses of the plate, at several times). rom_quantities is the bridge: for every row of a table of parameter sets it
runs the ROM and keeps three numbers that describe the plate at the final time:

    max_T              the largest temperature of the plate
    max_von_mises      the largest von Mises stress of the plate
    max_displacement   the largest displacement of the plate: the largest length sqrt(ux^2 + uy^2) of a node

PARAMETER_NAMES = ("k", "rho_c", "q", "E", "alpha")
    The five parameters of the ROM, in the order of the columns of `points` and of the arguments of rom.solve. A tuple
    of str.

QUANTITY_NAMES = ("max_T", "max_von_mises", "max_displacement")
    The three quantities, in the order of the columns of the answer. A tuple of str.

rom_quantities(rom, points, dt, n_steps, theta, save_every=1)
    rom           the reduced model: an object with the method solve(k, rho_c, q, E, alpha, dt, n_steps, theta,
                  save_every=1) of twin.rom.ThermoMechanicalROM
    points        the parameter sets: a table of numbers with m rows and 5 columns, an array or a list of lists
                  (m = 0 is allowed). The columns are in the order of PARAMETER_NAMES
    dt, n_steps, theta, save_every   the time scheme, handed to the ROM as they are
    Nothing is checked about rom and the time scheme: any object with that method works (the tests use one that is not a
    ThermoMechanicalROM), and the ROM judges the time scheme.

    The ROM is called once for every row, in the order of the rows, as

        rom.solve(k, rho_c, q, E, alpha, dt, n_steps, theta, save_every)

    with the five numbers of the row and the four time arguments unchanged. The ROM is not called in any other way, and
    no row is solved twice: a run of the ROM is the expensive part, and it gives the three quantities together.

    The quantities come from the LAST state that the ROM returns (the last row of its arrays "T", "u" and "von_mises",
    that is the final time) and from that state only:

        max_T             the largest entry of the last row of "T"
        max_von_mises     the largest entry of the last row of "von_mises"
        max_displacement  the last row of "u" holds the displacements of the nodes, x and y alternating: the node I has
                          ux = u[2 I] and uy = u[2 I + 1]. The quantity is the largest, over the nodes, of the length
                          sqrt(ux^2 + uy^2)

    Returns a NEW float64 array of shape (m, 3): the row i holds the three quantities of the point i, in the order of
    QUANTITY_NAMES. Without points (m = 0) the answer has shape (0, 3) and the ROM is not called. The argument points is
    not changed.

    ValueError if points is not a table of numbers with exactly 5 columns (a 1-D list, which could be one point, is
    refused too: one point is a table with one row). The ROM is not called then. The numbers themselves are not judged
    here: whatever the ROM accepts is handed to it, and its errors (for k <= 0, for a NaN, and so on) go through
    unchanged, as the same exception; the rows after the one that failed are not tried.

The tests use two kinds of ROM. A FAKE one, whose fields are made by hand with the largest values in known places, so
that every wrong choice (the wrong state, the wrong column, the wrong formula) gives a different number; and the REAL
one (twin.rom) on a plate of two triangles, where the answers are computed by hand. The names of the tests say what they
check, so that a group can be run alone with  pytest -k <word>  (for example -k constants, -k quantities, -k calls):

    test_uq_constants_*    the two tuples of names and the signature
    test_uq_refuses_*      the arguments that are refused
    test_uq_calls_*        how the ROM is called
    test_uq_output_*       the shape and the type of the answer
    test_uq_points_*       the ways of giving the points (what the ROM receives), and that they are not changed
    test_uq_quantities_*   the three numbers of a point, from the fake ROM
    test_uq_error_*        the errors of the ROM
    test_uq_real_rom_*     the real ROM
    test_uq_with_sobol_*   the function as the model of sobol_indices

The groups constants, refuses, calls, output and points do not look at the numbers of the answer, only at its shape and
at what the ROM is given: they can pass before the three quantities are written.
"""
import inspect
import math

import numpy as np
import pytest

import thermomech

# Any warning is an error in these tests.
pytestmark = pytest.mark.filterwarnings("error")

NAN = float("nan")
INF = float("inf")


# The module under test is imported inside these wrappers and not at the top: while twin/uq.py does not exist, or is not
# complete, only the tests that need it fail, not the whole session.


def uq():
    import twin.uq
    return twin.uq


def quantities(*args, **kwargs):
    from twin.uq import rom_quantities
    return rom_quantities(*args, **kwargs)


# ---------------------------------------------------------------------------------------------------------------------
# A fake ROM, with fields made by hand
# ---------------------------------------------------------------------------------------------------------------------

# The fields have 4 nodes and 3 triangles. In the LAST state the largest temperature is 8 (at the node 1: not the first
# node, not the last one; the last node is at -9, so the largest absolute value is another number), the largest von
# Mises stress is 12 (triangle 1) and the largest displacement is 5: the node 0 moves by (3, 4) and the nodes 1, 2, 3 by
# (-2, 1), (0.5, -4.5) and (4.2, -0.3). The states before the last one are larger in everything (a plate that cools
# down), so that a function that takes the largest value over the time, or a state that is not the last one, finds
# other numbers.
LAST_T = np.array([5.0, 8.0, 3.0, -9.0])
LAST_VM = np.array([4.0, 12.0, 6.0])
LAST_U = np.array([3.0, 4.0, -2.0, 1.0, 0.5, -4.5, 4.2, -0.3])  # ux, uy of the node 0, then of the node 1, and so on
EARLY_T = np.array([[10.0, 90.0, 20.0, 30.0], [40.0, 50.0, 100.0, 60.0]])
EARLY_VM = np.array([[50.0, 20.0, 10.0], [30.0, 80.0, 60.0]])
EARLY_U = np.array([10.0 * LAST_U, 20.0 * LAST_U])


def with_last(early, last, n_states):
    """n_states rows: the last one is `last`, the n_states - 1 before it are the two rows of `early` in turn."""
    return np.array([early[i % 2] for i in range(n_states - 1)] + [last])


def default_maxima(k, rho_c, q, E, alpha):
    """Three different numbers for every set of parameters, each one depending on all the five parameters."""
    return 1.0 + k + 0.01 * rho_c + 0.001 * q, 2.0 + 0.1 * E + 5.0 * alpha, 3.0 + rho_c / (1.0 + k) + alpha * q


class FakeROM:
    """Records every call of solve and returns the fields above, scaled so that the LAST state has the maxima that
    maxima(k, rho_c, q, E, alpha) gives: (max_T, max_von_mises, max_displacement). As the real ROM it returns
    n_steps / save_every + 1 states. failure = (the number of the call, an exception) makes that call raise."""

    def __init__(self, maxima=default_maxima, failure=None):
        self.maxima = maxima
        self.failure = failure
        self.calls = []

    def solve(self, k, rho_c, q, E, alpha, dt, n_steps, theta, save_every=1):
        self.calls.append(dict(k=k, rho_c=rho_c, q=q, E=E, alpha=alpha, dt=dt, n_steps=n_steps, theta=theta,
                               save_every=save_every))
        if self.failure is not None and self.failure[0] == len(self.calls):
            raise self.failure[1]
        t_max, vm_max, disp_max = self.maxima(k, rho_c, q, E, alpha)
        n_states = n_steps // save_every + 1
        return {
            "time": np.arange(n_states) * dt * save_every,
            "T": with_last(EARLY_T, LAST_T, n_states) * (t_max / 8.0),
            "u": with_last(EARLY_U, LAST_U, n_states) * (disp_max / 5.0),
            "sigma": np.full((n_states, 3, 3), 777.0),
            "von_mises": with_last(EARLY_VM, LAST_VM, n_states) * (vm_max / 12.0),
        }


# Four sets of parameters with different numbers everywhere, and what the fake ROM makes of them.
POINTS = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0],
                   [0.5, 1.5, 2.5, 3.5, 4.5],
                   [11.0, 12.0, 13.0, 14.0, 15.0]])
EXPECTED = np.array([default_maxima(*row) for row in POINTS])
NAMES = ("k", "rho_c", "q", "E", "alpha")  # the parameters, written here so that the tests of the calls do not need uq
DT, N_STEPS, THETA, SAVE_EVERY = 0.25, 12, 0.5, 3  # 5 saved states

# (n_steps, save_every) and the number of states that the ROM returns: 1, 2, 3, 6, 3 and 4.
SCHEMES = [pytest.param(0, 1, id="1 state"), pytest.param(1, 1, id="2 states"), pytest.param(2, 1, id="3 states"),
           pytest.param(5, 1, id="6 states"), pytest.param(6, 3, id="3 states, every 3"),
           pytest.param(12, 4, id="4 states, every 4")]


def close(result, expected, err_msg=""):
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0, err_msg=err_msg)


# ---------------------------------------------------------------------------------------------------------------------
# The constants and the signature
# ---------------------------------------------------------------------------------------------------------------------


def test_uq_constants_names():
    module = uq()
    assert module.PARAMETER_NAMES == ("k", "rho_c", "q", "E", "alpha")
    assert module.QUANTITY_NAMES == ("max_T", "max_von_mises", "max_displacement")


def test_uq_constants_are_tuples_of_strings():
    module = uq()
    for names in (module.PARAMETER_NAMES, module.QUANTITY_NAMES):
        assert type(names) is tuple
        assert all(type(name) is str for name in names)


def test_uq_constants_follow_the_arguments_of_solve():
    from twin.rom import ThermoMechanicalROM
    arguments = list(inspect.signature(ThermoMechanicalROM.solve).parameters)  # self, k, rho_c, q, E, alpha, dt, ...
    assert arguments[1:6] == list(uq().PARAMETER_NAMES)


def test_uq_constants_signature():
    parameters = inspect.signature(uq().rom_quantities).parameters
    assert list(parameters) == ["rom", "points", "dt", "n_steps", "theta", "save_every"]
    assert parameters["save_every"].default == 1
    for name in ["rom", "points", "dt", "n_steps", "theta"]:
        assert parameters[name].default is inspect.Parameter.empty, name


# ---------------------------------------------------------------------------------------------------------------------
# The arguments that are refused
# ---------------------------------------------------------------------------------------------------------------------

BAD_POINTS = [
    pytest.param(np.zeros(5), id="one point, 1-D"),
    pytest.param([1.0, 2.0, 3.0, 4.0, 5.0], id="one point, a list"),
    pytest.param(np.zeros((3, 4)), id="4 columns"),
    pytest.param(np.zeros((3, 6)), id="6 columns"),
    pytest.param(np.zeros((5, 3)), id="5 rows and 3 columns"),
    pytest.param(np.zeros((0, 4)), id="no rows, 4 columns"),
    pytest.param(np.zeros((2, 0)), id="no columns"),
    pytest.param(np.zeros((2, 5, 1)), id="3-D"),
    pytest.param([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0]], id="rows of different lengths"),
    pytest.param("abc", id="text"),
    pytest.param([["a", "b", "c", "d", "e"]], id="a table of text"),
]


@pytest.mark.parametrize("points", BAD_POINTS)
def test_uq_refuses_points_that_are_not_a_table_with_5_columns(points):
    rom = FakeROM()
    with pytest.raises(ValueError):
        quantities(rom, points, DT, N_STEPS, THETA, SAVE_EVERY)
    assert rom.calls == [], "the ROM must not be called when the points are refused"


# ---------------------------------------------------------------------------------------------------------------------
# How the ROM is called
# ---------------------------------------------------------------------------------------------------------------------


def test_uq_calls_once_per_row_with_the_numbers_of_the_row():
    rom = FakeROM()
    quantities(rom, POINTS, DT, N_STEPS, THETA, SAVE_EVERY)
    assert len(rom.calls) == len(POINTS), "one call of solve for every row"
    for number, (call, row) in enumerate(zip(rom.calls, POINTS)):
        for name, value in zip(NAMES, row):
            assert np.ndim(call[name]) == 0, "the call %d: %s is not a single number" % (number, name)
            assert call[name] == value, "the call %d: %s should be %r, it is %r" % (number, name, value, call[name])


def test_uq_calls_pass_the_time_arguments_unchanged():
    rom = FakeROM()
    quantities(rom, POINTS, 0.25, 12, 0.5, 3)
    for call in rom.calls:
        assert (call["dt"], call["n_steps"], call["theta"], call["save_every"]) == (0.25, 12, 0.5, 3)
    rom = FakeROM()
    quantities(rom, POINTS, dt=2.0, n_steps=10, theta=1.0, save_every=5)
    for call in rom.calls:
        assert (call["dt"], call["n_steps"], call["theta"], call["save_every"]) == (2.0, 10, 1.0, 5)


def test_uq_calls_save_every_is_1_when_it_is_not_given():
    rom = FakeROM()
    quantities(rom, POINTS, 0.25, 4, 1.0)
    assert [call["save_every"] for call in rom.calls] == [1] * len(POINTS)


def test_uq_calls_the_rom_is_not_called_for_a_row_twice_or_more():
    rom = FakeROM()
    quantities(rom, POINTS[:1], DT, N_STEPS, THETA, SAVE_EVERY)
    assert len(rom.calls) == 1
    rom = FakeROM()
    quantities(rom, POINTS[[2, 2, 2]], DT, N_STEPS, THETA, SAVE_EVERY)
    assert len(rom.calls) == 3, "three equal rows are three runs"


# ---------------------------------------------------------------------------------------------------------------------
# The shape and the type of the answer
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 2, 4])
def test_uq_output_shape_and_type(m):
    result = quantities(FakeROM(), POINTS[:m], DT, N_STEPS, THETA, SAVE_EVERY)
    assert type(result) is np.ndarray
    assert result.shape == (m, 3)
    assert result.dtype == np.float64


def test_uq_output_is_a_new_array_every_time():
    first = quantities(FakeROM(), POINTS, DT, N_STEPS, THETA, SAVE_EVERY)
    second = quantities(FakeROM(), POINTS, DT, N_STEPS, THETA, SAVE_EVERY)
    assert second is not first
    kept = second.copy()
    first[:] = -1.0
    np.testing.assert_array_equal(second, kept, err_msg="changing one answer must not change the other")


def test_uq_output_without_points():
    rom = FakeROM()
    result = quantities(rom, np.zeros((0, 5)), DT, N_STEPS, THETA, SAVE_EVERY)
    assert type(result) is np.ndarray
    assert result.shape == (0, 3)
    assert result.dtype == np.float64
    assert rom.calls == [], "no points, no run"


# ---------------------------------------------------------------------------------------------------------------------
# The ways of giving the points, and that they are not changed
# ---------------------------------------------------------------------------------------------------------------------


def as_list(values):
    return values.tolist()


def as_tuple(values):
    return tuple(tuple(row) for row in values.tolist())


def as_integers(values):
    return values.astype(int)


def as_fortran(values):
    return np.asfortranarray(values)


def as_every_second_column(values):
    wide = np.zeros((values.shape[0], 2 * values.shape[1]))
    wide[:, ::2] = values
    return wide[:, ::2]  # neighbouring numbers of a row are two numbers apart in memory


def as_read_only(values):
    values = values.copy()
    values.setflags(write=False)
    return values


LAYOUTS = [as_list, as_tuple, as_fortran, as_every_second_column, as_read_only]


@pytest.mark.parametrize("layout", LAYOUTS, ids=[layout.__name__ for layout in LAYOUTS])
def test_uq_points_can_be_given_in_other_forms(layout):
    """The ROM receives the same numbers, whatever the form of the table."""
    rom = FakeROM()
    quantities(rom, layout(POINTS), DT, N_STEPS, THETA, SAVE_EVERY)
    assert [[call[name] for name in NAMES] for call in rom.calls] == POINTS.tolist()


def test_uq_points_can_be_integers():
    points = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    for form in (points, points.tolist()):
        rom = FakeROM()
        quantities(rom, form, DT, N_STEPS, THETA, SAVE_EVERY)
        assert [[call[name] for name in NAMES] for call in rom.calls] == points.tolist()


def test_uq_points_are_not_changed():
    points = POINTS.copy()
    quantities(FakeROM(), points, DT, N_STEPS, THETA, SAVE_EVERY)
    np.testing.assert_array_equal(points, POINTS)
    nested = POINTS.tolist()
    quantities(FakeROM(), nested, DT, N_STEPS, THETA, SAVE_EVERY)
    assert nested == POINTS.tolist()


# ---------------------------------------------------------------------------------------------------------------------
# The three numbers of a point, from the fields of the fake ROM
# ---------------------------------------------------------------------------------------------------------------------


def test_uq_quantities_hand_values():
    """One point whose fields are exactly the hand-made ones: 8, 12 and the length 5 of the move (3, 4)."""
    rom = FakeROM(maxima=lambda k, rho_c, q, E, alpha: (8.0, 12.0, 5.0))
    result = quantities(rom, [POINTS[0]], DT, N_STEPS, THETA, SAVE_EVERY)
    close(result, [[8.0, 12.0, 5.0]])


@pytest.mark.parametrize("n_steps, save_every", SCHEMES)
def test_uq_quantities_max_T(n_steps, save_every):
    """Column 0: the largest temperature of the LAST state (the earlier states are hotter)."""
    result = quantities(FakeROM(), POINTS, 1.0, n_steps, 1.0, save_every)
    close(result[:, 0], EXPECTED[:, 0], "max_T")


@pytest.mark.parametrize("n_steps, save_every", SCHEMES)
def test_uq_quantities_max_von_mises(n_steps, save_every):
    """Column 1: the largest von Mises stress of the LAST state (the earlier states are more stressed)."""
    result = quantities(FakeROM(), POINTS, 1.0, n_steps, 1.0, save_every)
    close(result[:, 1], EXPECTED[:, 1], "max_von_mises")


@pytest.mark.parametrize("n_steps, save_every", SCHEMES)
def test_uq_quantities_max_displacement(n_steps, save_every):
    """Column 2: the largest LENGTH of the move of a node, in the LAST state. The x and the y of a node are next to each
    other in the array, and the largest entry of the array (4.2), the largest ux or uy, the sum of the two, the norm of
    the whole array and the lengths of the pairs (u[:n], u[n:]) are all different from the right answer."""
    result = quantities(FakeROM(), POINTS, 1.0, n_steps, 1.0, save_every)
    close(result[:, 2], EXPECTED[:, 2], "max_displacement")


@pytest.mark.parametrize("layout", LAYOUTS, ids=[layout.__name__ for layout in LAYOUTS])
def test_uq_quantities_whatever_the_layout(layout):
    close(quantities(FakeROM(), layout(POINTS), DT, N_STEPS, THETA, SAVE_EVERY), EXPECTED)


def test_uq_quantities_from_integers():
    points = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    expected = [default_maxima(*row) for row in points.astype(float)]
    close(quantities(FakeROM(), points, DT, N_STEPS, THETA, SAVE_EVERY), expected)
    close(quantities(FakeROM(), points.tolist(), DT, N_STEPS, THETA, SAVE_EVERY), expected)


def test_uq_quantities_every_row_is_its_own_point():
    """The rows of the answer follow the rows of points: the answers of two rows are not mixed up or repeated."""
    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2]):
        result = quantities(FakeROM(), POINTS[order], DT, N_STEPS, THETA, SAVE_EVERY)
        close(result, EXPECTED[order])


# ---------------------------------------------------------------------------------------------------------------------
# The errors of the ROM
# ---------------------------------------------------------------------------------------------------------------------


def test_uq_error_of_the_rom_goes_through_unchanged():
    error = RuntimeError("the ROM failed")
    rom = FakeROM(failure=(3, error))
    with pytest.raises(RuntimeError) as caught:
        quantities(rom, POINTS, DT, N_STEPS, THETA, SAVE_EVERY)
    assert caught.value is error, "the exception of the ROM must not be replaced or wrapped"
    assert len(rom.calls) == 3, "the rows after the one that failed are not tried"


def test_uq_error_in_the_first_row():
    error = KeyError("the ROM failed at once")
    rom = FakeROM(failure=(1, error))
    with pytest.raises(KeyError) as caught:
        quantities(rom, POINTS, DT, N_STEPS, THETA, SAVE_EVERY)
    assert caught.value is error
    assert len(rom.calls) == 1


# ---------------------------------------------------------------------------------------------------------------------
# The real ROM
# ---------------------------------------------------------------------------------------------------------------------


def unit_square_rom(basis, fixed_nodes, fixed_values, T0, T_ref):
    """The ROM of the unit square: 4 nodes (0, 0), (1, 0), (0, 1), (1, 1) and 2 triangles, plane strain with nu = 1/4,
    the nodes 0 and 1 and the y of the node 0 supported (the plate can expand freely), one thermal mode."""
    from twin.rom import ThermoMechanicalROM
    mesh = thermomech.rectangle_mesh(1.0, 1.0, 1, 1)
    return ThermoMechanicalROM(mesh=mesh, basis=basis, load=thermomech.assemble_load_2d(mesh, 1.0),
                               fixed_nodes=fixed_nodes, fixed_values=fixed_values, T0=T0, fixed_dofs=[0, 1, 3],
                               nu=0.25, T_ref=T_ref)


def free_plate_rom():
    """No imposed temperature and the constant mode: the temperature is the same everywhere and rises by q / rho_c per
    unit of time from 20, which is also T_ref. By hand, with rise = T - 20 and nu = 1/4: the in-plane stresses are zero,
    so the von Mises stress is E alpha |rise|, and the plate expands freely, u = (1 + nu) alpha rise (x, y): the node
    (1, 1) moves the most, by (1 + nu) alpha |rise| sqrt(2)."""
    return unit_square_rom(np.ones((4, 1)), [], [], np.full(4, 20.0), 20.0)


def cooling_plate_rom():
    """The left side (nodes 0 and 2) held at 0, one mode (the temperature x) and the plate starts from that mode, with
    T_ref = 0 and a very weak source: it cools down, so the temperature, the stress and the displacement are largest
    at the start and smallest at the end."""
    return unit_square_rom(np.array([[0.0], [1.0], [0.0], [1.0]]), [0, 2], [0.0, 0.0], np.array([0.0, 1.0, 0.0, 1.0]),
                           0.0)


def largest_move(u):
    """The largest length of the move of a node: u is one state, with ux and uy of every node next to each other."""
    return np.linalg.norm(u.reshape(-1, 2), axis=1).max()


@pytest.mark.parametrize("save_every", [1, 2])
def test_uq_real_rom_free_plate_by_hand(save_every):
    """Final time 1 (2 steps of 0.5): the temperature rises by q / rho_c, to 22, 23 and 22."""
    points = np.array([[1.0, 2.0, 4.0, 100.0, 0.01],
                       [2.0, 1.0, 3.0, 200.0, 0.02],
                       [0.5, 4.0, 8.0, 50.0, 0.04]])
    k, rho_c, q, E, alpha = points.T
    rise = q / rho_c * 1.0
    expected = np.column_stack([20.0 + rise, E * alpha * rise, 1.25 * alpha * rise * math.sqrt(2.0)])
    result = quantities(free_plate_rom(), points, 0.5, 2, 0.5, save_every)
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=0)


def test_uq_real_rom_cooling_plate_agrees_with_the_states_of_the_rom():
    """The numbers are those of the last state of the dict that rom.solve returns, computed here in another way."""
    points = np.array([[2.0, 3.0, 1e-6, 100.0, 0.01], [1.0, 1.0, 2e-6, 150.0, 0.02], [4.0, 2.0, 1e-6, 80.0, 0.03]])
    rom = cooling_plate_rom()
    expected = []
    for row in points:
        states = rom.solve(*row, 0.25, 8, 0.5, 2)
        T, u, von_mises = states["T"], states["u"], states["von_mises"]
        # the premise of the test: the first state is the largest in everything, the last one is far below it
        assert T[0].max() > 10.0 * T[-1].max()
        assert von_mises[0].max() > 10.0 * von_mises[-1].max()
        assert largest_move(u[0]) > 10.0 * largest_move(u[-1])
        expected.append([T[-1].max(), von_mises[-1].max(), largest_move(u[-1])])
    result = quantities(rom, points, 0.25, 8, 0.5, 2)  # 5 saved states: 0, 2, 4, 6, 8 steps
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)
    result = quantities(rom, points, 0.25, 8, 0.5, 8)  # 2 saved states: the last one is the same
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("column, value", [(0, 0.0), (1, -3.0), (2, INF), (3, 0.0), (4, NAN)],
                         ids=["k = 0", "rho_c < 0", "q = infinity", "E = 0", "alpha = NaN"])
def test_uq_real_rom_a_bad_number_is_refused_by_the_rom(column, value):
    points = np.array([[1.0, 2.0, 4.0, 100.0, 0.01]] * 3)
    points[1, column] = value
    with pytest.raises(ValueError):
        quantities(free_plate_rom(), points, 0.5, 2, 0.5)


# ---------------------------------------------------------------------------------------------------------------------
# The function as the model of sobol_indices
# ---------------------------------------------------------------------------------------------------------------------


def test_uq_with_sobol_indices_fake_rom():
    """max_T depends only on k, rho_c and q; max_von_mises only on E and alpha; max_displacement only on alpha. So the
    indices of the others are exactly 0 (the output of two rows that differ only in a parameter that does not matter
    is the same number), and the other indices are clearly positive."""
    from twin.sobol import sobol_indices
    rom = FakeROM(maxima=lambda k, rho_c, q, E, alpha: (2.0 * k + 3.0 * rho_c + 4.0 * q, E * alpha, alpha))

    def model(points):
        return quantities(rom, points, 0.5, 2, 0.5)

    n = 64
    result = sobol_indices(model, [1.0] * 5, [2.0] * 5, n, seed=5, n_bootstrap=20)
    assert result.first_order.shape == (5, 3)
    assert result.total_order.shape == (5, 3)
    assert result.first_order_ci.shape == (2, 5, 3)
    assert result.mean.shape == (3,)
    assert result.n_evaluations == n * 7
    assert len(rom.calls) == n * 7, "one run of the ROM for every row of every table of points"
    zero = np.array([[False, True, True],     # k
                     [False, True, True],     # rho_c
                     [False, True, True],     # q
                     [True, False, True],     # E
                     [True, False, False]])   # alpha
    assert (result.first_order[zero] == 0.0).all()
    assert (result.total_order[zero] == 0.0).all()
    assert (result.first_order[~zero] > 0.05).all()
    assert (result.total_order[~zero] > 0.05).all()
    assert result.total_order[4, 2] > 0.9, "max_displacement is alpha"


def test_uq_with_sobol_indices_real_rom():
    """The real ROM on the free plate: the temperature does not depend on E and alpha, and the displacements do not
    depend on E (they are proportional to alpha), so those indices are 0 (up to rounding)."""
    from twin.sobol import sobol_indices
    rom = free_plate_rom()

    def model(points):
        return quantities(rom, points, 0.5, 2, 0.5)

    low = [1.0, 1.0, 1.0, 100.0, 0.01]
    high = [2.0, 2.0, 2.0, 200.0, 0.02]
    result = sobol_indices(model, low, high, 32, seed=1, n_bootstrap=20)
    assert result.first_order.shape == (5, 3)
    assert np.all(np.abs(result.first_order[3:, 0]) < 1e-12) and np.all(np.abs(result.total_order[3:, 0]) < 1e-12)
    assert abs(result.first_order[3, 2]) < 1e-12 and abs(result.total_order[3, 2]) < 1e-12
    assert result.total_order[4, 2] > 0.1, "the displacements do depend on alpha"
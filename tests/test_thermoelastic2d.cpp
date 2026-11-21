#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "Assembly2D.h"
#include "Elasticity2D.h"
#include "Mesh2D.h"
#include "Solver2D.h"
#include "Thermoelastic2D.h"

// ---- The complete chain: the heat that changes in time and the stresses that arise from it -------------------------
//
// So far every piece has been tested on its own: the heat in time (solve_transient_2d), the thermal force, the
// displacements, the stresses. solve_thermomechanical_2d chains them together: for every instant to be saved it
// computes the temperature T, then the thermal force F(T), then the displacements u (K u = F with the constraints),
// then the stresses sigma, and returns the list of these states. The coupling is one-way (the heat moves the plate,
// the plate heats nothing) and quasi-static (at every instant the plate is in equilibrium with that instant's temperature).
//
// TWO FACTS THAT THE TESTS USE.
//   1. The elastic part is LINEAR: with zero constraints, u and sigma are linear in the temperature deviation T - T_ref
//      (the thermal force is, K u = F is, sigma = D (B u - eps_th) is). So if the temperature deviation of
//      one state is g times that of another, so are u and sigma.
//   2. The heat has solutions that can be computed by hand. With the plate insulated and a uniform load q, K 1 = 0 (no
//      flux if T is uniform) and the mass carries the load in: T grows by q / rho_c per second at all nodes,
//      for every theta scheme. With a strip of two rows of cells, the faces at T_ref and the middle row hot, the
//      middle row is a single unknown and each step multiplies it by a factor g that is computed by hand (EXAMPLE
//      219).
//
// What the function must do, and it is what the tests check:
//   - the list has n_steps / save_every + 1 states, the first at time 0 (with T = T0), the others at times (step) x dt; every
//     state has the right sizes (EXAMPLES 212 and 213); with zero steps no heat data is needed (EXAMPLE 213);
//   - every state is exactly the chain of the functions that already exist, in the right order and with the right data (EXAMPLE
//     214), and saving every step or every 3 steps gives the same numbers (EXAMPLE 215);
//   - three cases with a hand solution: the free plate that is heated uniformly expands without stresses
//     (EXAMPLE 216), even with an imposed rigid motion (EXAMPLE 217); the clamped plate, heated in the same
//     way, is compressed and the pressure grows linearly in time (EXAMPLE 218);
//   - the roller strip: the heat decays by the hand-computed factor g (EXAMPLE 219) and with it all the displacements and
//     the stresses (EXAMPLE 220); in the middle of a long strip the stresses are the hand-computed ones (EXAMPLE 221); the
//     decay has the time order of the theta scheme (EXAMPLE 222); a steel plate that cools has the
//     textbook stress, -E alpha dT / (1 - nu) (EXAMPLE 223);
//   - scaling laws: E, alpha and the amplitude of the heat (EXAMPLE 224), T_ref not mattering if everything is shifted
//     (EXAMPLE 225);
//   - wrong data: those checked by the function (EXAMPLE 226) and those of the functions it calls (EXAMPLE 227).

namespace {

const double kPi = 3.14159265358979323846;

// Convenient numbers for the hand computations: E = 100, nu = 1/4, alpha = 1/100, T_ref = 20. Then
//     lambda = E nu / ((1 + nu) (1 - 2 nu)) = 100 (1/4) / ((5/4) (1/2)) = 40,        mu = E / (2 (1 + nu)) = 40,
//     D = [[120, 40, 0], [40, 120, 0], [0, 0, 40]],        E alpha = 1,        (1 + nu) alpha = 1/80 = 0.0125.
void use_hand_material(ThermoMechanicalProblem2D& problem) {
    problem.E = 100.0;
    problem.nu = 0.25;
    problem.alpha = 0.01;
    problem.T_ref = 20.0;
}

// The three constraints of a rectangular grid of nx x ... cells that remove the three rigid motions (as for the beam):
// u and v of node 0 (the corner (0, 0)) and v of node nx (the corner (L, 0)), all zero.
void pin_corners(ThermoMechanicalProblem2D& problem, std::size_t nx) {
    problem.fixed_dofs = {0, 1, 2 * nx + 1};
    problem.fixed_displacements = {0.0, 0.0, 0.0};
}

// The roller plate of nx x ny cells, rectangle_mesh(length, height, nx, ny), with no heat load (f = 0):
//   - the two faces, y = 0 (the row of nodes 0, ..., nx) and y = H (the last row), stay at the temperature t_ref;
//   - the initial temperature is t_ref + amplitude sin(pi j / ny) in row j of nodes, the same in all the columns;
//   - the rollers: u = 0 at the nodes of the two sides x = 0 and x = L (columns 0 and nx of every row), and v = 0 at node 0
//     to remove the vertical displacement. No other constraint: the faces expand freely.
// The material, the heat numbers and the time are chosen by the caller.
ThermoMechanicalProblem2D roller_slab_problem(std::size_t nx, std::size_t ny, double t_ref, double amplitude) {
    const std::size_t n = (nx + 1) * (ny + 1);
    ThermoMechanicalProblem2D problem;
    problem.f = DVector(n, 0.0);
    problem.T0 = DVector(n, t_ref);
    for (std::size_t j = 0; j <= ny; ++j) {
        for (std::size_t i = 0; i <= nx; ++i) {
            const double ratio = static_cast<double>(j) / static_cast<double>(ny);
            problem.T0[j * (nx + 1) + i] = t_ref + amplitude * std::sin(kPi * ratio);
        }
    }
    for (std::size_t i = 0; i <= nx; ++i) {
        problem.fixed_nodes.push_back(i);
        problem.fixed_temperatures.push_back(t_ref);
        problem.fixed_nodes.push_back(ny * (nx + 1) + i);
        problem.fixed_temperatures.push_back(t_ref);
    }
    for (std::size_t j = 0; j <= ny; ++j) {
        problem.fixed_dofs.push_back(2 * (j * (nx + 1)));
        problem.fixed_displacements.push_back(0.0);
        problem.fixed_dofs.push_back(2 * (j * (nx + 1) + nx));
        problem.fixed_displacements.push_back(0.0);
    }
    problem.fixed_dofs.push_back(1);
    problem.fixed_displacements.push_back(0.0);
    return problem;
}

// The strip: two rows of square cells (side 1/2, height H = 1, length nx / 2), with the convenient numbers: amplitude
// 60, rho_c = 2, k = 3 (so k / rho_c = 3/2) and the hand material. With dt = 1/36 every step multiplies the middle
// row by a factor g that is computed by hand (EXAMPLE 219).
Mesh2D strip_mesh(std::size_t nx) {
    return rectangle_mesh(0.5 * static_cast<double>(nx), 1.0, nx, 2);
}

ThermoMechanicalProblem2D strip_problem(std::size_t nx, double theta, double dt, std::size_t n_steps,
                                        std::size_t save_every) {
    ThermoMechanicalProblem2D problem = roller_slab_problem(nx, 2, 20.0, 60.0);
    use_hand_material(problem);
    problem.rho_c = 2.0;
    problem.k = 3.0;
    problem.dt = dt;
    problem.theta = theta;
    problem.n_steps = n_steps;
    problem.save_every = save_every;
    return problem;
}

// The "general" problem of the composition and scaling tests: an irregular mesh, and all the numbers different from
// one another (so that a swap between two data changes the result).
Mesh2D general_mesh() {
    return Mesh2D({0.0, 4.0, 3.0, -1.0, 1.0}, {0.0, 0.0, 3.0, 2.0, 0.7},
                  {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}});
}

ThermoMechanicalProblem2D general_problem(const Mesh2D& mesh) {
    ThermoMechanicalProblem2D problem;
    problem.rho_c = 2.5;
    problem.k = 0.7;
    problem.f = assemble_load_2d(mesh, 3.0);
    problem.fixed_nodes = {0, 3};
    problem.fixed_temperatures = {310.0, 280.0};
    problem.T0 = {310.0, 305.0, 295.0, 280.0, 290.0};
    problem.dt = 0.2;
    problem.n_steps = 6;
    problem.theta = 0.6;
    problem.save_every = 2;
    problem.E = 70.0;
    problem.nu = 0.2;
    problem.alpha = 0.003;
    problem.T_ref = 290.0;
    problem.fixed_dofs = {0, 1, 3};
    problem.fixed_displacements = {0.0, 0.0, 0.0};
    return problem;
}

// The largest absolute value of a vector, and of the stresses (all the components of all the triangles).
double max_abs(const std::vector<double>& values) {
    double largest = 0.0;
    for (double value : values) {
        largest = std::max(largest, std::fabs(value));
    }
    return largest;
}

double max_abs(const std::vector<Vector3>& stresses) {
    double largest = 0.0;
    for (const Vector3& s : stresses) {
        for (std::size_t c = 0; c < 3; ++c) {
            largest = std::max(largest, std::fabs(s[c]));
        }
    }
    return largest;
}

// Two vectors equal up to an absolute tolerance.
void expect_vectors_near(const std::vector<double>& got, const std::vector<double>& want, double tolerance,
                         const std::string& label) {
    ASSERT_EQ(got.size(), want.size()) << label;
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tolerance) << label << ", component " << i;
    }
}

// The stresses of two lists of triangles equal up to an absolute tolerance, component by component.
void expect_stresses_near(const std::vector<Vector3>& got, const std::vector<Vector3>& want, double tolerance,
                          const std::string& label) {
    ASSERT_EQ(got.size(), want.size()) << label;
    for (std::size_t e = 0; e < got.size(); ++e) {
        for (std::size_t c = 0; c < 3; ++c) {
            EXPECT_NEAR(got[e][c], want[e][c], tolerance) << label << ", triangle " << e << ", component " << c;
        }
    }
}

// Two equal states (same time, same T, same u and sigma) up to a tolerance RELATIVE to the largest value
// of each field.
void expect_states_near(const ThermoMechanicalState2D& got, const ThermoMechanicalState2D& want, double relative,
                        const std::string& label) {
    EXPECT_DOUBLE_EQ(got.time, want.time) << label;
    expect_vectors_near(got.T, want.T, relative * max_abs(want.T), label + ", T");
    expect_vectors_near(got.u, want.u, relative * max_abs(want.u), label + ", u");
    expect_stresses_near(got.sigma, want.sigma, relative * max_abs(want.sigma), label + ", sigma");
}

}  // namespace

// EXAMPLE 212. How many states, at which times, of what size. The unit square (4 nodes, 2 triangles), the plate
// insulated and with no load, T0 = T_ref = 20 everywhere (nothing happens: only the shape of the result matters) and
// dt = 0.1.
//   - 6 steps, saving every 2: the states are at steps 0, 2, 4, 6, so 6 / 2 + 1 = 4 states, at times 0, 0.2, 0.4 and
//     0.6 (the step times dt);
//   - 10 steps, saving every step: 11 states, the last at time 10 x 0.1 = 1;
//   - in every state T has 4 numbers (one per node), u has 8 (two per node) and sigma has 2 (one per triangle).
TEST(Thermoelastic2d, TheStatesAreOneEverySaveEveryStepsStartingAtTimeZero) {
    const Mesh2D mesh({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});
    ThermoMechanicalProblem2D problem;
    use_hand_material(problem);
    problem.rho_c = 1.0;
    problem.k = 1.0;
    problem.f = DVector(4, 0.0);
    problem.T0 = DVector(4, 20.0);
    problem.dt = 0.1;
    problem.theta = 1.0;
    problem.fixed_dofs = {0, 1, 3};
    problem.fixed_displacements = {0.0, 0.0, 0.0};

    problem.n_steps = 6;
    problem.save_every = 2;
    std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
    ASSERT_EQ(states.size(), 4u);
    const double times[4] = {0.0, 0.2, 0.4, 0.6};
    for (std::size_t m = 0; m < states.size(); ++m) {
        EXPECT_DOUBLE_EQ(states[m].time, times[m]) << "state " << m;
        EXPECT_EQ(states[m].T.size(), 4u) << "state " << m;
        EXPECT_EQ(states[m].u.size(), 8u) << "state " << m;
        EXPECT_EQ(states[m].sigma.size(), 2u) << "state " << m;
    }

    problem.n_steps = 10;
    problem.save_every = 1;
    states = solve_thermomechanical_2d(mesh, problem);
    ASSERT_EQ(states.size(), 11u);
    EXPECT_DOUBLE_EQ(states[0].time, 0.0);
    EXPECT_DOUBLE_EQ(states[10].time, 1.0);
}

// EXAMPLE 213. With zero steps there is only one state, the one at time 0, and the heat data is not needed. The "zero"
// problem (ThermoMechanicalProblem2D problem;) has no steps (n_steps = 0, save_every = 1) and no heat data
// (rho_c = 0, k = 0, dt = 0, no load: with steps to do it would be an error), and the function must not
// complain. The unit square has T0 = 30 everywhere and T_ref = 20, so dT = 10, and the free plate (u and v of
// node 0 and v of node 1 constrained) expands by e = (1 + nu) alpha dT = 0.0125 x 10 = 0.125 in every direction:
//     u = e x,  v = e y:      node 0 = (0, 0): (0, 0),    node 1 = (1, 0): (0.125, 0),
//                             node 2 = (1, 1): (0.125, 0.125),    node 3 = (0, 1): (0, 0.125),
// with no stress. T is T0, identical.
TEST(Thermoelastic2d, WithZeroStepsThereIsOnlyTheInitialStateAndNoHeatDataIsNeeded) {
    const Mesh2D mesh({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});
    ThermoMechanicalProblem2D problem;
    EXPECT_EQ(problem.n_steps, 0u);
    EXPECT_EQ(problem.save_every, 1u);
    EXPECT_EQ(problem.theta, 1.0);
    EXPECT_EQ(problem.rho_c, 0.0);
    EXPECT_EQ(problem.k, 0.0);
    EXPECT_EQ(problem.dt, 0.0);
    EXPECT_TRUE(problem.f.empty());

    use_hand_material(problem);
    problem.T0 = DVector(4, 30.0);
    problem.fixed_dofs = {0, 1, 3};
    problem.fixed_displacements = {0.0, 0.0, 0.0};

    std::vector<ThermoMechanicalState2D> states;
    ASSERT_NO_THROW(states = solve_thermomechanical_2d(mesh, problem));
    ASSERT_EQ(states.size(), 1u);
    EXPECT_EQ(states[0].time, 0.0);
    EXPECT_EQ(states[0].T, problem.T0);
    expect_vectors_near(states[0].u, {0.0, 0.0, 0.125, 0.0, 0.125, 0.125, 0.0, 0.125}, 1e-14, "u");
    expect_stresses_near(states[0].sigma, {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, 1e-12, "sigma");
}

// EXAMPLE 214. Every state is the chain of the functions that already exist, in the right order and with the right data. An
// irregular mesh (the same as EXAMPLE 198), with all the numbers different from one another: rho_c = 2.5, k = 0.7, a
// uniform load q = 3, nodes 0 and 3 at 310 and 280, E = 70, nu = 0.2, alpha = 0.003, T_ref = 290, theta = 0.6, and
// constraints that are NOT zero (u and v of node 0 and v of node 1: 0.01, -0.02 and 0.03). 6 steps of 0.2, saving every
// 2: 4 states. For every state s (at step p = 2 s) the chain is redone by hand, with the earlier functions:
//     T     = T0 for p = 0, otherwise solve_transient_2d after p steps all at once;
//     D     = plane_strain_matrix(E, nu),            F = assemble_thermal_load_2d(mesh, D, nu, alpha, T, T_ref),
//     u     = solve_displacement_2d(mesh, D, F, fixed_dofs, fixed_displacements),
//     sigma = stress_field_2d(mesh, D, nu, alpha, u, T, T_ref).
// The time is p x dt. A swap between two data (rho_c and k, E and nu), a forgotten T_ref, constraints taken as zero, the
// initial temperature used for the stresses in place of the one at that instant: each changes some numbers.
TEST(Thermoelastic2d, EveryStateIsTheChainOfTheExistingFunctions) {
    const Mesh2D mesh = general_mesh();
    ThermoMechanicalProblem2D problem = general_problem(mesh);
    problem.fixed_displacements = {0.01, -0.02, 0.03};

    const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
    ASSERT_EQ(states.size(), 4u);

    const Matrix3 D = plane_strain_matrix(problem.E, problem.nu);
    for (std::size_t s = 0; s < states.size(); ++s) {
        const std::size_t passes = 2 * s;
        ThermoMechanicalState2D want;
        want.time = static_cast<double>(passes) * problem.dt;
        want.T = problem.T0;
        if (passes > 0) {
            want.T = solve_transient_2d(mesh, problem.rho_c, problem.k, problem.f, problem.fixed_nodes,
                                        problem.fixed_temperatures, problem.T0, problem.dt, passes, problem.theta);
        }
        const std::vector<double> F =
            assemble_thermal_load_2d(mesh, D, problem.nu, problem.alpha, want.T, problem.T_ref);
        want.u = solve_displacement_2d(mesh, D, F, problem.fixed_dofs, problem.fixed_displacements);
        want.sigma = stress_field_2d(mesh, D, problem.nu, problem.alpha, want.u, want.T, problem.T_ref);

        expect_states_near(states[s], want, 1e-9, "state " + std::to_string(s));
    }
    EXPECT_EQ(states[0].T, problem.T0);
    EXPECT_GT(max_abs(states[3].sigma), 1e-3) << "the test must have some stresses to compare";
    double moved = 0.0;
    for (std::size_t i = 0; i < problem.T0.size(); ++i) {
        moved = std::max(moved, std::fabs(states[3].T[i] - problem.T0[i]));
    }
    EXPECT_GT(moved, 1.0) << "the test must have heat that changes in time";
}

// EXAMPLE 215. Saving every 1 step or every 3 steps does not change the numbers: the mesh and the problem of EXAMPLE 214,
// but with 6 steps and the constraints set to zero. Saving every step gives 7 states (at steps 0, ..., 6), saving every 3
// gives 3 (at steps 0, 3, 6) and they must equal states 0, 3 and 6 of the long list: restarting every 3 steps from the
// previous result is the same as doing 6 steps in a row.
TEST(Thermoelastic2d, SavingEveryThreeStepsGivesTheSameNumbersAsSavingEveryStep) {
    const Mesh2D mesh = general_mesh();
    ThermoMechanicalProblem2D problem = general_problem(mesh);

    problem.save_every = 1;
    const std::vector<ThermoMechanicalState2D> every_step = solve_thermomechanical_2d(mesh, problem);
    problem.save_every = 3;
    const std::vector<ThermoMechanicalState2D> every_third = solve_thermomechanical_2d(mesh, problem);

    ASSERT_EQ(every_step.size(), 7u);
    ASSERT_EQ(every_third.size(), 3u);
    for (std::size_t s = 0; s < every_third.size(); ++s) {
        expect_states_near(every_third[s], every_step[3 * s], 1e-12, "state " + std::to_string(s));
    }
}

// EXAMPLE 216. The free plate that heats up uniformly. The mesh of 3 x 2 cells of [0, 2] x [0, 1] (12 nodes), the
// plate insulated (no node at imposed temperature), T0 = 25 and T_ref = 20, a uniform load q = 8 with rho_c = 2 and
// steps of 0.25: every step heats by q dt / rho_c = 8 x 0.25 / 2 = 1 degree, so after p steps the temperature is
// 25 + p at every node. For every theta scheme: it is a consequence only of K 1 = 0 (the heat does not flow if T is
// uniform) and of f = q M 1 (the mass carries the load in: M (T_new - T_old) = dt f).
// The plate is free (three constraints at (0, 0) and (L, 0), as for the beam) and expands by e = (1 + nu) alpha dT =
// 0.0125 (5 + p) in every direction, u = e x and v = e y, with no stresses: the strain B u = e (1, 1, 0) is exactly
// the thermal one.
TEST(Thermoelastic2d, AFreePlateThatIsHeatedUniformlyExpandsWithoutStress) {
    const Mesh2D mesh = rectangle_mesh(2.0, 1.0, 3, 2);
    for (double theta : {1.0, 0.5, 0.0, 0.8}) {
        ThermoMechanicalProblem2D problem;
        use_hand_material(problem);
        problem.rho_c = 2.0;
        problem.k = 1.0;
        problem.f = assemble_load_2d(mesh, 8.0);
        problem.T0 = DVector(12, 25.0);
        problem.dt = 0.25;
        problem.n_steps = 4;
        problem.theta = theta;
        pin_corners(problem, 3);

        const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
        ASSERT_EQ(states.size(), 5u);
        for (std::size_t p = 0; p < states.size(); ++p) {
            const std::string label = "theta = " + std::to_string(theta) + ", step " + std::to_string(p);
            const double e = 0.0125 * (5.0 + static_cast<double>(p));
            std::vector<double> u_want;
            for (std::size_t node = 0; node < 12; ++node) {
                u_want.push_back(e * mesh.x(node));
                u_want.push_back(e * mesh.y(node));
            }
            expect_vectors_near(states[p].T, DVector(12, 25.0 + static_cast<double>(p)), 1e-9, label + ", T");
            expect_vectors_near(states[p].u, u_want, 1e-10, label + ", u");
            expect_stresses_near(states[p].sigma, std::vector<Vector3>(mesh.num_triangles(), {0.0, 0.0, 0.0}), 1e-8,
                                 label + ", sigma");
        }
    }
}

// EXAMPLE 217. The constraints matter: the same free plate as EXAMPLE 216 (theta = 1, 2 steps), but with the imposed
// displacements being a rigid motion, a translation (3, -2) and a small rotation of 0.1 radians:
//     u = 3 - 0.1 y + e x,        v = -2 + 0.1 x + e y.
// A rigid rotation does not deform (du/dx = dv/dy = 0 and du/dy + dv/dx = -0.1 + 0.1 = 0), so the stresses stay 0 and
// the values to impose at the three constraints are those of this formula: u(0, 0) = 3, v(0, 0) = -2 and, at (L, 0) = (2, 0),
// v = -2 + 0.1 x 2 = -1.8. Anyone who did not pass the imposed displacements (took them as zero) would find the plate still
// at (0, 0) instead of at (3, -2).
TEST(Thermoelastic2d, TheImposedDisplacementsAreARigidMotionAddedToTheExpansion) {
    const Mesh2D mesh = rectangle_mesh(2.0, 1.0, 3, 2);
    ThermoMechanicalProblem2D problem;
    use_hand_material(problem);
    problem.rho_c = 2.0;
    problem.k = 1.0;
    problem.f = assemble_load_2d(mesh, 8.0);
    problem.T0 = DVector(12, 25.0);
    problem.dt = 0.25;
    problem.n_steps = 2;
    problem.theta = 1.0;
    problem.fixed_dofs = {0, 1, 7};
    problem.fixed_displacements = {3.0, -2.0, -1.8};

    const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
    ASSERT_EQ(states.size(), 3u);
    for (std::size_t p = 0; p < states.size(); ++p) {
        const double e = 0.0125 * (5.0 + static_cast<double>(p));
        std::vector<double> u_want;
        for (std::size_t node = 0; node < 12; ++node) {
            u_want.push_back(3.0 - 0.1 * mesh.y(node) + e * mesh.x(node));
            u_want.push_back(-2.0 + 0.1 * mesh.x(node) + e * mesh.y(node));
        }
        expect_vectors_near(states[p].u, u_want, 1e-10, "step " + std::to_string(p) + ", u");
        expect_stresses_near(states[p].sigma, std::vector<Vector3>(mesh.num_triangles(), {0.0, 0.0, 0.0}), 1e-8,
                             "step " + std::to_string(p) + ", sigma");
    }
}

// EXAMPLE 218. The clamped plate that heats up: the pressure grows linearly in time. The mesh of 4 x 4 cells of
// [0, 2] x [0, 1] (25 nodes), with all the boundary nodes fixed (u = v = 0): the 9 interior nodes are free. The plate
// is insulated, T0 = T_ref = 20, a uniform load q = 8 with rho_c = 2 and steps of 0.25: after p steps dT = p degrees. The
// plate does not move (u = 0: the thermal force of a uniform field is 0 at the interior nodes, and the boundary ones are
// still) and the stress is that of a restrained body, sigma = D (0 - eps_th) = -D eps_th. With
// e = (1 + nu) alpha dT = 0.0125 dT:
//     eps_th = e (1, 1, 0),        D eps_th = e (120 + 40, 40 + 120, 0) = 0.0125 dT (160, 160, 0) = 2 dT (1, 1, 0),
// i.e. sigma = -2 dT (1, 1, 0) = -E alpha dT / (1 - 2 nu) (1, 1, 0) in every triangle, for theta = 1 and theta = 1/2.
TEST(Thermoelastic2d, AClampedPlateIsCompressedMoreAndMoreAsItHeatsUp) {
    const Mesh2D mesh = rectangle_mesh(2.0, 1.0, 4, 4);
    for (double theta : {1.0, 0.5}) {
        ThermoMechanicalProblem2D problem;
        use_hand_material(problem);
        problem.rho_c = 2.0;
        problem.k = 1.0;
        problem.f = assemble_load_2d(mesh, 8.0);
        problem.T0 = DVector(25, 20.0);
        problem.dt = 0.25;
        problem.n_steps = 4;
        problem.theta = theta;
        for (std::size_t node : boundary_nodes(mesh)) {
            problem.fixed_dofs.push_back(2 * node);
            problem.fixed_dofs.push_back(2 * node + 1);
            problem.fixed_displacements.push_back(0.0);
            problem.fixed_displacements.push_back(0.0);
        }
        ASSERT_EQ(problem.fixed_dofs.size(), 32u);

        const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
        ASSERT_EQ(states.size(), 5u);
        for (std::size_t p = 0; p < states.size(); ++p) {
            const std::string label = "theta = " + std::to_string(theta) + ", step " + std::to_string(p);
            const double dT = static_cast<double>(p);
            expect_vectors_near(states[p].u, std::vector<double>(50, 0.0), 1e-10, label + ", u");
            expect_stresses_near(states[p].sigma,
                                 std::vector<Vector3>(mesh.num_triangles(), {-2.0 * dT, -2.0 * dT, 0.0}), 1e-8,
                                 label + ", sigma");
        }
    }
}

// EXAMPLE 219. The strip: the heat decays by a factor g at every step, which is computed by hand. The rows of nodes are
// three, y = 0, 1/2 and 1 (h = 1/2): the two faces stay at T_ref = 20 and the middle row is the unknown (the same in
// all the columns: the strip is uniform in x). T does not depend on x, so the row of a middle node is computed
// as for a 1D rod, with the neighbors (the faces) at zero deviation:
//     mass:       rho_c (2 h / 3) = 2 (1/3) = 2/3,            stiffness:      k (2 / h) = 3 x 4 = 12.
// The theta-scheme step, (M + theta dt K) T_new = (M - (1 - theta) dt K) T_old, written on the deviation from T_ref,
// becomes
//     (2/3 + 12 theta dt) T_new = (2/3 - 12 (1 - theta) dt) T_old,
// and with dt = 1/36 (12 dt = 1/3):   g = (2/3 - (1 - theta) / 3) / (2/3 + theta / 3) = (1 + theta) / (2 + theta).
// For theta = 1 (backward Euler) g = 2/3, for theta = 1/2 (Crank-Nicolson) g = 3/5, for theta = 0 (forward
// Euler) g = 1/2. With the middle row at 20 + 60 at the start, after p steps it is at 20 + 60 g^p, and the faces are at 20
// EXACTLY. It is the test of theta, of rho_c and k (not swapped), of the fixed faces and of the fact that every step
// restarts from the result of the previous one.
TEST(Thermoelastic2d, TheMiddleRowOfTheStripDecaysByTheHandFactorEveryStep) {
    struct Scheme {
        double theta;
        double g;
    };
    const Mesh2D mesh = strip_mesh(8);
    for (const Scheme& scheme : {Scheme{1.0, 2.0 / 3.0}, Scheme{0.5, 3.0 / 5.0}, Scheme{0.0, 1.0 / 2.0}}) {
        const std::vector<ThermoMechanicalState2D> states =
            solve_thermomechanical_2d(mesh, strip_problem(8, scheme.theta, 1.0 / 36.0, 3, 1));
        ASSERT_EQ(states.size(), 4u);
        double factor = 1.0;
        for (std::size_t p = 0; p < states.size(); ++p) {
            const std::string label = "theta = " + std::to_string(scheme.theta) + ", step " + std::to_string(p);
            for (std::size_t i = 0; i <= 8; ++i) {
                EXPECT_NEAR(states[p].T[9 + i], 20.0 + 60.0 * factor, 1e-9) << label << ", middle row, node " << i;
                if (p > 0) {
                    EXPECT_EQ(states[p].T[i], 20.0) << label << ", bottom face, node " << i;
                    EXPECT_EQ(states[p].T[18 + i], 20.0) << label << ", top face, node " << i;
                }
            }
            factor *= scheme.g;
        }
    }
}

// EXAMPLE 220. With the heat, the displacements and the stresses decay too: the elastic part is linear in the temperature
// deviation (and the constraints are zero), so if the deviation of T becomes g times the previous one, u and sigma
// become g times the previous ones, EXACTLY (up to the errors of the conjugate gradient). The strip
// of EXAMPLE 219 with the three schemes and 3 steps: state p has u = g^p u(0) and sigma = g^p sigma(0). It is also checked
// that state 0 is not empty (stresses of tens, displacements of tenths) and that the stress is compressive:
// where the plate is hotter it would like to expand, and the rollers stop it.
TEST(Thermoelastic2d, TheDisplacementsAndStressesOfTheStripDecayLikeTheHeat) {
    struct Scheme {
        double theta;
        double g;
    };
    const Mesh2D mesh = strip_mesh(8);
    for (const Scheme& scheme : {Scheme{1.0, 2.0 / 3.0}, Scheme{0.5, 3.0 / 5.0}, Scheme{0.0, 1.0 / 2.0}}) {
        const std::vector<ThermoMechanicalState2D> states =
            solve_thermomechanical_2d(mesh, strip_problem(8, scheme.theta, 1.0 / 36.0, 3, 1));
        ASSERT_EQ(states.size(), 4u);

        const std::vector<double>& u0 = states[0].u;
        const std::vector<Vector3>& sigma0 = states[0].sigma;
        EXPECT_GT(max_abs(sigma0), 20.0);
        EXPECT_GT(max_abs(u0), 0.1);
        double compressive_sum = 0.0;
        for (const Vector3& s : sigma0) {
            compressive_sum += s[0];
        }
        EXPECT_LT(compressive_sum, 0.0) << "sigma_xx must be compressive (negative)";

        double factor = 1.0;
        for (std::size_t p = 1; p < states.size(); ++p) {
            factor *= scheme.g;
            std::vector<double> u_want = u0;
            for (double& value : u_want) {
                value *= factor;
            }
            std::vector<Vector3> sigma_want = sigma0;
            for (Vector3& s : sigma_want) {
                for (std::size_t c = 0; c < 3; ++c) {
                    s[c] *= factor;
                }
            }
            const std::string label = "theta = " + std::to_string(scheme.theta) + ", step " + std::to_string(p);
            expect_vectors_near(states[p].u, u_want, 1e-9 * max_abs(u0), label + ", u");
            expect_stresses_near(states[p].sigma, sigma_want, 1e-9 * max_abs(sigma0), label + ", sigma");
        }
    }
}

// EXAMPLE 221. In the middle of a LONG strip the stresses are the hand-computed ones. The strip of EXAMPLE 219 with 24
// columns (length 12, square cells of side 1/2) and backward Euler (g = 2/3). Far from the ends (where the
// rollers block u) the strip is uniform in x: u = 0 and v depends only on y. With E = 100, nu = 1/4 (lambda = mu = 40),
// alpha = 1/100, and with dT_X the mean T deviation of the three nodes of a triangle X and e_X = (1 + nu) alpha dT_X:
//   - the "bottom right" triangle of the cell between row 0 and row 1 has two nodes in row 0 (deviation 0) and one
//     in row 1 (deviation 60): dT_X = 20 and e_X = 1/4. The "top left" one has one node at 0 and two at 60: dT_X = 40
//     and e_X = 1/2.
//   - the strain is (0, gamma, 0), with gamma = dv/dy the same in the two triangles (the same difference of v over the
//     same h). At the nodes of the face (free in v) the vertical force is proportional to the sum of the sigma_yy of the
//     two triangles of the cell above, and it must be 0 because there is nothing below: the sum of the sigma_yy of the two
//     triangles of a cell is 0 (and so for the cells of the next row, where the force is the difference between two
//     sums). With sigma_yy = (lambda + 2 mu) gamma - (2 lambda + 2 mu) e_X = 120 gamma - 160 e_X:
//         (120 gamma - 40) + (120 gamma - 80) = 0,        gamma = 1/2.
//   - the elastic part of the strain is (-e_X, gamma - e_X, 0), so
//         sigma_xx = 120 (-e_X) + 40 (gamma - e_X) = 20 - 160 e_X,        sigma_yy = 40 (-e_X) + 120 (gamma - e_X) =
//         60 - 160 e_X.
//     With e_X = 1/4: sigma = (-20, +20). With e_X = 1/2: sigma = (-60, -20).
//   - the cell above (row 1 -> row 2) is the mirror image: the bottom triangle has two nodes at 60 and one at 0 (dT_X = 40,
//     sigma = (-60, -20)) and the top one has one node at 60 and two at 0 (dT_X = 20, sigma = (-20, +20)).
//   - v grows by gamma h = 1/4 between one row and the next: v(row 1) - v(row 0) = v(row 2) - v(row 1) = 1/4.
// These numbers are those of an infinite strip: the effect of the rollers falls by a factor of about 3 per column
// starting from the ends (18 in the first column, 0.6 in the third), and in the three middle columns, at 10-12 columns
// from the ends, it is below 2e-4 on the sigmas and below 1e-7 on v (the test requires 1e-3 for the sigmas and 1e-5 for v).
// After one step everything is multiplied by g = 2/3.
TEST(Thermoelastic2d, InTheMiddleOfALongStripTheStressesAreTheHandNumbers) {
    const std::size_t nx = 24;
    const Mesh2D mesh = strip_mesh(nx);
    const std::vector<ThermoMechanicalState2D> states =
        solve_thermomechanical_2d(mesh, strip_problem(nx, 1.0, 1.0 / 36.0, 1, 1));
    ASSERT_EQ(states.size(), 2u);

    // sigma (xx, yy) of the {bottom, top} triangles of the cells of row 0 and row 1, and the three middle columns
    const double want[2][2][2] = {{{-20.0, 20.0}, {-60.0, -20.0}}, {{-60.0, -20.0}, {-20.0, 20.0}}};
    for (std::size_t p = 0; p < states.size(); ++p) {
        const double factor = (p == 0) ? 1.0 : 2.0 / 3.0;
        for (std::size_t i = nx / 2 - 1; i <= nx / 2 + 1; ++i) {
            for (std::size_t row = 0; row < 2; ++row) {
                for (std::size_t half = 0; half < 2; ++half) {
                    const Vector3& s = states[p].sigma[2 * (row * nx + i) + half];
                    const std::string label = "step " + std::to_string(p) + ", cell (" + std::to_string(i) + ", " +
                                              std::to_string(row) + "), triangle " + std::to_string(half);
                    EXPECT_NEAR(s[0], factor * want[row][half][0], 1e-3) << label << ", sigma_xx";
                    EXPECT_NEAR(s[1], factor * want[row][half][1], 1e-3) << label << ", sigma_yy";
                    EXPECT_NEAR(s[2], 0.0, 1e-3) << label << ", sigma_xy";
                }
            }
        }
        const std::size_t mid = nx / 2;
        const double v0 = states[p].u[2 * mid + 1];
        const double v1 = states[p].u[2 * ((nx + 1) + mid) + 1];
        const double v2 = states[p].u[2 * (2 * (nx + 1) + mid) + 1];
        EXPECT_NEAR(v1 - v0, factor * 0.25, 1e-5) << "step " << p;
        EXPECT_NEAR(v2 - v1, factor * 0.25, 1e-5) << "step " << p;
        EXPECT_NEAR(states[p].u[2 * ((nx + 1) + mid)], 0.0, 1e-9) << "step " << p << ", u in the middle";
    }
}

// EXAMPLE 222. The decay has the time order of the theta scheme, as for the heat alone. In the strip of EXAMPLE 219 the
// middle row is a single mode, and the stiffness-to-mass ratio is 12 / (2/3) = 18: without time-step error the mode would
// decay as e^(-18 t), and at time t = 1/18 it would equal e^(-1). With n steps of 1 / (18 n) the two schemes
// give a factor g^n, with
//     Backward Euler:  g^n = (1 / (1 + 1/n))^n,        Crank-Nicolson:  g^n = ((1 - 1/(2 n)) / (1 + 1/(2 n)))^n,
// and sigma at the end is g^n times sigma at the start. The error with respect to e^(-1) falls by a factor of 2 when n
// doubles for Euler (order 1: the observed orders are 0.96 and 0.98) and by a factor of 4 for Crank-Nicolson (order 2,
// observed 2.00). The test looks at sigma_xx of the bottom triangle of the middle cell of row 0 at n = 8, 16 and 32 steps.
TEST(Thermoelastic2d, TheStressDecayHasTheTimeOrderOfTheScheme) {
    const Mesh2D mesh = strip_mesh(8);
    const std::size_t triangle = 2 * (0 * 8 + 4);
    const double exact = std::exp(-1.0);
    for (double theta : {1.0, 0.5}) {
        std::vector<double> errors;
        for (std::size_t n : {8u, 16u, 32u}) {
            const std::vector<ThermoMechanicalState2D> states =
                solve_thermomechanical_2d(mesh, strip_problem(8, theta, 1.0 / (18.0 * static_cast<double>(n)), n, n));
            ASSERT_EQ(states.size(), 2u);
            ASSERT_GT(std::fabs(states[0].sigma[triangle][0]), 1.0);
            const double ratio = states[1].sigma[triangle][0] / states[0].sigma[triangle][0];
            errors.push_back(std::fabs(ratio - exact));
        }
        const double order_first = std::log2(errors[0] / errors[1]);
        const double order_second = std::log2(errors[1] / errors[2]);
        const double expected = (theta == 1.0) ? 1.0 : 2.0;
        EXPECT_NEAR(order_first, expected, 0.1) << "theta = " << theta << ", n = 8 -> 16";
        EXPECT_NEAR(order_second, expected, 0.1) << "theta = " << theta << ", n = 16 -> 32";
    }
}

// EXAMPLE 223. A steel plate that cools from both faces has the "textbook" stress. The plate is H = 0.1 m thick
// and 0.4 m long, steel with E = 210 GPa, nu = 0.3, alpha = 12e-6, rho_c = 7850 x 470 = 3.6895e6 J/(m^3 K) and
// k = 45 W/(m K) (k / rho_c = 1.22e-5 m^2/s). The faces are held at T_ref = 293.15 K and the plate starts from
// T_ref + 100 sin(pi y / H): it cools, and the mode decays as e^(-t / tau) with tau = H^2 / (pi^2 k / rho_c) = 83 s.
// With the rollers at the ends (u = 0), at mid-length the plate is uniform in x, with u = 0 and v(y): it is the strip
// of EXAMPLE 221 with generic numbers, and the mean of the two sigma_xx of the triangles of a cell is exactly
//     sigma_xx = -E alpha dT / (1 - nu),
// with dT the mean of the T deviation of the two rows of nodes of the cell (it is the formula for a body that cannot
// expand in x, in plane strain: for dT = 100 K it is 360 MPa of compression). With 8 cells in height (32 in
// length), steps of 5 s, Crank-Nicolson and states saved at 0, 50 and 100 s:
//   - the formula with the computed T deviation is verified to within 1e-4 times 360 MPa (the effect of the rollers at
//     mid-length is below 1e-5);
//   - with the "continuous" deviation, 100 sin(pi y / H) e^(-t / tau) taken at mid-height of the cell, the difference is
//     the error of the method: the mean of two nodes instead of the value in the middle (1 - cos(pi/16) = 1.9%) and the decay
//     of the discrete mode, 1.3% too fast with 8 cells. The test requires it to be below 5% of 360 MPa.
TEST(Thermoelastic2d, ASteelSlabThatIsCooledFromBothSidesHasTheTextbookStress) {
    const std::size_t nx = 32;
    const std::size_t ny = 8;
    const double height = 0.1;
    const double length = 0.4;
    const double t_ref = 293.15;
    const double amplitude = 100.0;
    const Mesh2D mesh = rectangle_mesh(length, height, nx, ny);

    ThermoMechanicalProblem2D problem = roller_slab_problem(nx, ny, t_ref, amplitude);
    problem.rho_c = 7850.0 * 470.0;
    problem.k = 45.0;
    problem.dt = 5.0;
    problem.n_steps = 20;
    problem.save_every = 10;
    problem.theta = 0.5;
    problem.E = 210.0e9;
    problem.nu = 0.3;
    problem.alpha = 12.0e-6;
    problem.T_ref = t_ref;

    const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
    ASSERT_EQ(states.size(), 3u);

    const double scale = problem.E * problem.alpha * amplitude / (1.0 - problem.nu);
    const double tau = height * height / (kPi * kPi * (problem.k / problem.rho_c));
    const std::size_t column = nx / 2;
    for (const ThermoMechanicalState2D& state : states) {
        for (std::size_t row = 0; row < ny; ++row) {
            const std::size_t cell = row * nx + column;
            const double average_xx = 0.5 * (state.sigma[2 * cell][0] + state.sigma[2 * cell + 1][0]);
            const double below = state.T[row * (nx + 1) + column] - t_ref;
            const double above = state.T[(row + 1) * (nx + 1) + column] - t_ref;
            const double from_the_formula = -problem.E * problem.alpha * 0.5 * (below + above) / (1.0 - problem.nu);
            EXPECT_NEAR(average_xx, from_the_formula, 1e-4 * scale)
                << "t = " << state.time << ", row of cells " << row << ", formula with the computed T";

            const double y_middle = (static_cast<double>(row) + 0.5) * height / static_cast<double>(ny);
            const double continuum =
                -scale * std::sin(kPi * y_middle / height) * std::exp(-state.time / tau);
            EXPECT_NEAR(average_xx, continuum, 0.05 * scale)
                << "t = " << state.time << ", row of cells " << row << ", continuous solution";
        }
    }

    // the compression is really there: at mid-height, at time 0, it is 346 MPa (96% of 360)
    const std::size_t middle_cell = (ny / 2) * nx + column;
    const double middle_xx = 0.5 * (states[0].sigma[2 * middle_cell][0] + states[0].sigma[2 * middle_cell + 1][0]);
    EXPECT_LT(middle_xx, -0.95 * scale);
    EXPECT_GT(middle_xx, -1.0 * scale);
}

// EXAMPLE 224. Scaling laws: the numbers that depend in a simple way on one datum. The irregular mesh and the data
// of EXAMPLE 214, but with the constraints at zero (with nonzero constraints the scaling laws hold for the elastic
// part, not for the constraints).
//   - E doubled: the displacements stay the same (K and F double) and the stresses double;
//   - alpha doubled: F and eps_th double, so the displacements and the stresses double;
//   - the heat doubled (the deviation from T_ref of the initial temperature, of the imposed temperatures and the load f, all
//     multiplied by 2): the temperature is doubled, and with it the displacements and the stresses;
//   - the heat with the sign changed (factor -1): everything changes sign.
// T does not change with E and with alpha (the heat does not feel the mechanics).
TEST(Thermoelastic2d, DisplacementsAndStressesScaleWithEWithAlphaAndWithTheHeat) {
    const Mesh2D mesh = general_mesh();
    const ThermoMechanicalProblem2D base = general_problem(mesh);
    const std::vector<ThermoMechanicalState2D> base_states = solve_thermomechanical_2d(mesh, base);
    ASSERT_EQ(base_states.size(), 4u);
    ASSERT_GT(max_abs(base_states[3].sigma), 1e-3);

    ThermoMechanicalProblem2D stiff = base;
    stiff.E = 2.0 * base.E;
    const std::vector<ThermoMechanicalState2D> stiff_states = solve_thermomechanical_2d(mesh, stiff);

    ThermoMechanicalProblem2D expanding = base;
    expanding.alpha = 2.0 * base.alpha;
    const std::vector<ThermoMechanicalState2D> expanding_states = solve_thermomechanical_2d(mesh, expanding);

    for (double factor : {2.0, -1.0}) {
        ThermoMechanicalProblem2D hot = base;
        for (double& value : hot.T0) {
            value = base.T_ref + factor * (value - base.T_ref);
        }
        for (double& value : hot.fixed_temperatures) {
            value = base.T_ref + factor * (value - base.T_ref);
        }
        for (double& value : hot.f) {
            value *= factor;
        }
        const std::vector<ThermoMechanicalState2D> hot_states = solve_thermomechanical_2d(mesh, hot);
        ASSERT_EQ(hot_states.size(), 4u);
        for (std::size_t s = 0; s < 4; ++s) {
            const std::string label = "heat x " + std::to_string(factor) + ", state " + std::to_string(s);
            std::vector<double> T_want = base_states[s].T;
            for (double& value : T_want) {
                value = base.T_ref + factor * (value - base.T_ref);
            }
            std::vector<double> u_want = base_states[s].u;
            for (double& value : u_want) {
                value *= factor;
            }
            std::vector<Vector3> sigma_want = base_states[s].sigma;
            for (Vector3& t : sigma_want) {
                for (std::size_t c = 0; c < 3; ++c) {
                    t[c] *= factor;
                }
            }
            expect_vectors_near(hot_states[s].T, T_want, 1e-9 * max_abs(T_want), label + ", T");
            expect_vectors_near(hot_states[s].u, u_want, 1e-9 * max_abs(u_want), label + ", u");
            expect_stresses_near(hot_states[s].sigma, sigma_want, 1e-9 * max_abs(sigma_want), label + ", sigma");
        }
    }

    for (std::size_t s = 0; s < 4; ++s) {
        const std::string label = "state " + std::to_string(s);
        EXPECT_EQ(stiff_states[s].T, base_states[s].T) << label << ", E doubled: T";
        expect_vectors_near(stiff_states[s].u, base_states[s].u, 1e-9 * max_abs(base_states[s].u) + 1e-15,
                            label + ", E doubled: u");
        std::vector<Vector3> sigma_want = base_states[s].sigma;
        std::vector<double> u_want = base_states[s].u;
        for (Vector3& t : sigma_want) {
            for (std::size_t c = 0; c < 3; ++c) {
                t[c] *= 2.0;
            }
        }
        for (double& value : u_want) {
            value *= 2.0;
        }
        expect_stresses_near(stiff_states[s].sigma, sigma_want, 1e-9 * max_abs(sigma_want) + 1e-12,
                             label + ", E doubled: sigma");
        EXPECT_EQ(expanding_states[s].T, base_states[s].T) << label << ", alpha doubled: T";
        expect_vectors_near(expanding_states[s].u, u_want, 1e-9 * max_abs(u_want) + 1e-15,
                            label + ", alpha doubled: u");
        expect_stresses_near(expanding_states[s].sigma, sigma_want, 1e-9 * max_abs(sigma_want) + 1e-12,
                             label + ", alpha doubled: sigma");
    }
}

// EXAMPLE 225. T_ref does not matter if everything is shifted: the irregular mesh and the data of EXAMPLE 214 (constraints
// at zero), with T_ref, the initial temperature and the imposed temperatures increased by 100 degrees. The heat does not
// feel the origin of the temperatures (K and M do not depend on T), so T is the previous one plus 100, and the
// displacements and the stresses, which depend only on T - T_ref, are the same. Anyone who used T_ref = 0 in place of
// T_ref, or forgot it, would find enormous stresses (T is 390 or more against T_ref = 290: 100 degrees more deviation).
TEST(Thermoelastic2d, ShiftingAllTheTemperaturesAndTRefTogetherChangesNothingButT) {
    const Mesh2D mesh = general_mesh();
    const ThermoMechanicalProblem2D base = general_problem(mesh);
    const std::vector<ThermoMechanicalState2D> base_states = solve_thermomechanical_2d(mesh, base);
    ASSERT_EQ(base_states.size(), 4u);

    ThermoMechanicalProblem2D shifted = base;
    shifted.T_ref += 100.0;
    for (double& value : shifted.T0) {
        value += 100.0;
    }
    for (double& value : shifted.fixed_temperatures) {
        value += 100.0;
    }
    const std::vector<ThermoMechanicalState2D> shifted_states = solve_thermomechanical_2d(mesh, shifted);

    ASSERT_EQ(shifted_states.size(), base_states.size());
    for (std::size_t s = 0; s < base_states.size(); ++s) {
        const std::string label = "state " + std::to_string(s);
        std::vector<double> T_want = base_states[s].T;
        for (double& value : T_want) {
            value += 100.0;
        }
        expect_vectors_near(shifted_states[s].T, T_want, 1e-9 * max_abs(T_want), label + ", T");
        expect_vectors_near(shifted_states[s].u, base_states[s].u, 1e-9 * max_abs(base_states[s].u) + 1e-15,
                            label + ", u");
        expect_stresses_near(shifted_states[s].sigma, base_states[s].sigma,
                             1e-9 * max_abs(base_states[s].sigma) + 1e-12, label + ", sigma");
    }
}

// EXAMPLE 226. Wrong data, those checked by the function itself: save_every = 0 (it would never save: it throws
// std::invalid_argument BEFORE dividing by zero, n_steps % 0 would crash the program) and n_steps that is not a
// multiple of save_every (6 steps saving every 4: the last state would not be at the end). With zero steps any
// save_every >= 1 is fine (0 is a multiple of everything), but save_every = 0 is not.
TEST(Thermoelastic2d, SaveEveryMustBeAtLeastOneAndDivideTheSteps) {
    const Mesh2D mesh({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});
    ThermoMechanicalProblem2D problem;
    use_hand_material(problem);
    problem.rho_c = 1.0;
    problem.k = 1.0;
    problem.f = DVector(4, 0.0);
    problem.T0 = DVector(4, 20.0);
    problem.dt = 0.1;
    problem.theta = 1.0;
    problem.fixed_dofs = {0, 1, 3};
    problem.fixed_displacements = {0.0, 0.0, 0.0};

    problem.n_steps = 4;
    problem.save_every = 0;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, problem), std::invalid_argument);
    problem.n_steps = 0;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, problem), std::invalid_argument);

    problem.n_steps = 6;
    problem.save_every = 4;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, problem), std::invalid_argument);
    problem.save_every = 5;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, problem), std::invalid_argument);

    problem.save_every = 3;
    EXPECT_EQ(solve_thermomechanical_2d(mesh, problem).size(), 3u);
    problem.n_steps = 0;
    EXPECT_EQ(solve_thermomechanical_2d(mesh, problem).size(), 1u);
}

// EXAMPLE 227. Wrong data of the functions that are called: the function does not recheck it, it lets the
// std::invalid_argument exception of whoever finds it pass through. The unit square, insulated plate, starts from a
// valid problem (it is checked that it is valid) and changes one datum at a time:
//   - the material: E = 0 or nu = 1/2 (plane_strain_matrix), even with zero steps;
//   - the initial temperature with 3 numbers instead of 4 (assemble_thermal_load_2d);
//   - the constraints: 2 degrees of freedom instead of 3, or a different number of values (solve_displacement_2d);
//   - the heat, with at least one step: k = 0, dt = 0, a load of 3 numbers, theta = 2 (solve_transient_2d).
TEST(Thermoelastic2d, WrongDataIsReportedByTheFunctionThatFindsIt) {
    const Mesh2D mesh({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});
    ThermoMechanicalProblem2D valid;
    use_hand_material(valid);
    valid.rho_c = 1.0;
    valid.k = 1.0;
    valid.f = DVector(4, 0.0);
    valid.T0 = DVector(4, 25.0);
    valid.dt = 0.1;
    valid.n_steps = 2;
    valid.theta = 1.0;
    valid.save_every = 1;
    valid.fixed_dofs = {0, 1, 3};
    valid.fixed_displacements = {0.0, 0.0, 0.0};
    ASSERT_NO_THROW(solve_thermomechanical_2d(mesh, valid));

    ThermoMechanicalProblem2D bad = valid;
    bad.E = 0.0;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);
    bad.n_steps = 0;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);

    bad = valid;
    bad.nu = 0.5;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);

    bad = valid;
    bad.T0 = DVector(3, 25.0);
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);

    bad = valid;
    bad.fixed_dofs = {0, 1};
    bad.fixed_displacements = {0.0, 0.0};
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);

    bad = valid;
    bad.fixed_displacements = {0.0, 0.0};
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);

    bad = valid;
    bad.k = 0.0;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);

    bad = valid;
    bad.dt = 0.0;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);

    bad = valid;
    bad.f = DVector(3, 0.0);
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);

    bad = valid;
    bad.theta = 2.0;
    EXPECT_THROW(solve_thermomechanical_2d(mesh, bad), std::invalid_argument);
}

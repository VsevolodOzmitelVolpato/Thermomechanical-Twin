#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "Assembly2D.h"
#include "Elasticity2D.h"
#include "Mesh2D.h"
#include "Thermoelastic2D.h"

// ---- The von Mises stress: a single number to say how stressed the material is ------------------------------------
//
// The three in-plane stresses (xx, yy, xy) alone do not say whether the material is in danger. A metal compressed from
// all sides by the same value (like a stone at the bottom of the sea) does not yield, even at 1000 MPa; a metal pulled
// in one direction only yields at 300 MPa. What makes the material yield is the part of the stress that changes the
// SHAPE, not the pressure that is equal in all directions. The von Mises equivalent stress is a single number that
// measures exactly that part, and it is what is compared with the yield stress. With the three normal stresses (xx, yy, zz)
// and the shear xy:
//     sigma_vm = sqrt( 1/2 [ (sxx - syy)^2 + (syy - szz)^2 + (szz - sxx)^2 ] + 3 sxy^2 ).
// It is written with the DIFFERENCES between the normal stresses: adding the same pressure to all three changes nothing.
// For simple tension (only sxx = s) it gives |s|: this is how it was calibrated.
//
// WHY sigma_zz IS NEEDED. The plate is in plane strain: it does not elongate in z (epsilon_zz = 0). But "does not elongate"
// does not mean "no stress in z": the body is restrained, and the stress in z is what restrains it. Hooke's
// law in z, with alpha dT the thermal expansion that would like to occur, says (nu = Poisson's ratio)
//     0 = epsilon_zz = [ szz - nu (sxx + syy) ] / E + alpha dT,        i.e.        szz = nu (sxx + syy) - E alpha dT.
// The "thin plate" formula (szz = 0) is WRONG here, and the tests show it with numbers.
//
// THREE FACTS THAT THE TESTS USE (with E = 100, nu = 1/4, alpha = 1/100 one has E alpha = 1).
//   1. A free plate heated uniformly has no in-plane stresses (EXAMPLE 216), but it has szz = -E alpha dT:
//      it would like to expand in z too and cannot. Its von Mises is E alpha dT, not 0.
//   2. A clamped plate heated uniformly has sxx = syy = -2 dT (EXAMPLE 218) and, with the formula for szz,
//      also szz = 0.25 (-4 dT) - dT = -2 dT: three equal stresses, a compression equal in all directions. The
//      von Mises is 0, even though the in-plane stresses are large.
//   3. In the middle of a long roller strip the plate does not deform in x and in z (epsilon_xx = epsilon_zz = 0), and
//      it gives szz = sxx in every triangle: the von Mises is |sxx - syy|, and it is 40 in the hand numbers (EXAMPLE 221),
//      i.e. the textbook formula E alpha dT / (1 - nu) with dT = 30 (EXAMPLE 236). In the steel plate it is the number that
//      the mean of sxx gave in EXAMPLE 223 (EXAMPLE 237).
//
// What the two functions must do, and it is what the tests check:
//   - von_mises_2d: the simple states (EXAMPLE 228), a hand example with all the components (229), the
//     hydrostatic state (230), what does not change (231 and 232);
//   - von_mises_field_2d: every triangle uses its own stresses and its three nodes (233), the free plate (234), the
//     clamped one (235), the strip (236), the steel plate (237), the scaling laws (238) and the wrong data (239);
//   - both: a NaN in the data stays a NaN and does not become a number (240).

namespace {

const double kPi = 3.14159265358979323846;

// Convenient numbers for hand computation: E = 100, nu = 1/4, alpha = 1/100, T_ref = 20 (E alpha = 1, D = [[120, 40, 0],
// [40, 120, 0], [0, 0, 40]]).
void use_hand_material(ThermoMechanicalProblem2D& problem) {
    problem.E = 100.0;
    problem.nu = 0.25;
    problem.alpha = 0.01;
    problem.T_ref = 20.0;
}

// The roller plate of nx x ny cells, rectangle_mesh(length, height, nx, ny), with no heat load (f = 0): the two
// faces (y = 0 and y = H) stay at t_ref, the initial temperature is t_ref + amplitude sin(pi j / ny) in row j of
// nodes, u = 0 at the nodes of the two sides x = 0 and x = L and v = 0 at node 0. The material, the heat numbers and
// the time are chosen by the caller (it is the same plate as in the Thermoelastic2D tests).
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

// The "general" problem: an irregular mesh and all the numbers different from one another (as in EXAMPLES 214 and 224).
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

// The von Mises of all the triangles of a state of the chain, with the material and T_ref of the problem that gave it.
std::vector<double> von_mises_of(const Mesh2D& mesh, const ThermoMechanicalProblem2D& problem,
                                 const ThermoMechanicalState2D& state) {
    return von_mises_field_2d(mesh, state.sigma, problem.E, problem.nu, problem.alpha, state.T, problem.T_ref);
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

// All the numbers of a vector equal to want, up to an absolute tolerance.
void expect_all_near(const std::vector<double>& got, double want, double tolerance, const std::string& label) {
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], want, tolerance) << label << ", triangle " << i;
    }
}

// Two vectors equal up to an absolute tolerance.
void expect_vectors_near(const std::vector<double>& got, const std::vector<double>& want, double tolerance,
                         const std::string& label) {
    ASSERT_EQ(got.size(), want.size()) << label;
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tolerance) << label << ", triangle " << i;
    }
}

// The general problem with all the "heat" (the deviation from T_ref of the initial temperature and of the imposed
// temperatures, and the load) multiplied by factor.
ThermoMechanicalProblem2D with_heat_times(const ThermoMechanicalProblem2D& base, double factor) {
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
    return hot;
}

// The von Mises of all 4 states of the modified general problem, equal to that of the base problem times factor.
void expect_von_mises_scaled(const Mesh2D& mesh, const ThermoMechanicalProblem2D& changed,
                             const std::vector<std::vector<double>>& base_vm, double factor, const std::string& what) {
    const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, changed);
    ASSERT_EQ(states.size(), base_vm.size()) << what;
    for (std::size_t s = 0; s < states.size(); ++s) {
        std::vector<double> want = base_vm[s];
        for (double& value : want) {
            value *= factor;
        }
        expect_vectors_near(von_mises_of(mesh, changed, states[s]), want, 1e-9 * max_abs(base_vm[s]) + 1e-12,
                            what + ", state " + std::to_string(s));
    }
}

}  // namespace

// EXAMPLE 228. The simple states give the number one expects. With sigma = (xx, yy, xy) and the out-of-plane
// stress szz:
//   - tension only, (100, 0, 0) with szz = 0: the three differences are 100, 0 and -100, so 1/2 (100^2 + 0 + 100^2) =
//     100^2 and the von Mises is 100: for simple tension it is the stress itself. The same for (0, -40, 0): 40 (the
//     sign does not matter);
//   - szz = 25 alone (xx = yy = xy = 0): the differences are 0, -25 and 25, so 25;
//   - equal biaxial in the plane, (50, 50, 0) with szz = 0: the differences are 0, 50 and -50, so 1/2 (0 + 2500 +
//     2500) = 2500 and the von Mises is 50;
//   - pure shear, (0, 0, 10): 3 x 10^2 = 300, so 10 sqrt(3) = 17.32. The shear "weighs" sqrt(3) times a tension;
//   - all 0: 0.
TEST(VonMises2d, TheSimpleStatesGiveTheNumberOneExpects) {
    EXPECT_DOUBLE_EQ(von_mises_2d({100.0, 0.0, 0.0}, 0.0), 100.0);
    EXPECT_DOUBLE_EQ(von_mises_2d({0.0, -40.0, 0.0}, 0.0), 40.0);
    EXPECT_DOUBLE_EQ(von_mises_2d({0.0, 0.0, 0.0}, 25.0), 25.0);
    EXPECT_DOUBLE_EQ(von_mises_2d({50.0, 50.0, 0.0}, 0.0), 50.0);
    EXPECT_NEAR(von_mises_2d({0.0, 0.0, 10.0}, 0.0), 10.0 * std::sqrt(3.0), 1e-12);
    EXPECT_EQ(von_mises_2d({0.0, 0.0, 0.0}, 0.0), 0.0);
}

// EXAMPLE 229. A hand example with all the components: sigma = (5, 2, 3) and szz = 2. The differences between the normals
// are 5 - 2 = 3, 2 - 2 = 0 and 2 - 5 = -3, so
//     1/2 (3^2 + 0^2 + (-3)^2) + 3 x 3^2 = 9 + 27 = 36,        sigma_vm = 6.
// Another, (4, 1, 1) with szz = 0: differences 3, 1 and -4, so 1/2 (9 + 1 + 16) + 3 x 1 = 13 + 3 = 16 and the von Mises
// is 4. And the first example's state with szz = 0 instead of 2 (differences 3, 2 and -5: 1/2 (9 + 4 + 25) + 27 = 46)
// gives sqrt(46) = 6.78, another number: szz matters.
TEST(VonMises2d, AHandExampleWithAllTheComponents) {
    EXPECT_DOUBLE_EQ(von_mises_2d({5.0, 2.0, 3.0}, 2.0), 6.0);
    EXPECT_DOUBLE_EQ(von_mises_2d({4.0, 1.0, 1.0}, 0.0), 4.0);
    EXPECT_NEAR(von_mises_2d({5.0, 2.0, 3.0}, 0.0), std::sqrt(46.0), 1e-12);
}

// EXAMPLE 230. A hydrostatic state has no von Mises stress. With the three normals equal, (p, p, 0) and szz = p, the three
// differences are 0 and the von Mises is EXACTLY 0 (not "about": it is a sum of squares of zeros). This is just what
// the expanded form of the formula, sxx^2 + syy^2 + szz^2 - sxx syy - syy szz - szz sxx + 3 sxy^2, does not guarantee:
// for p = 0.7, 1.3, 3.3, 9.1 or -1234.5678 the rounding gives a tiny NEGATIVE number (-1e-16, -4e-16,
// -4e-15, -3e-14, -5e-10) and its square root is NaN. With szz = 0 instead, the same (p, p, 0) is not hydrostatic (the
// differences are 0, p and -p) and it equals |p|.
TEST(VonMises2d, AHydrostaticStateHasNoVonMisesStress) {
    for (double p : {0.7, 1.3, 3.3, 9.1, -1234.5678, -7.0, 5.5}) {
        EXPECT_EQ(von_mises_2d({p, p, 0.0}, p), 0.0) << "p = " << p;
        EXPECT_DOUBLE_EQ(von_mises_2d({p, p, 0.0}, 0.0), std::fabs(p)) << "p = " << p << ", szz = 0";
    }
}

// EXAMPLE 231. What leaves the von Mises unchanged. Start from the hand example, (5, 2, 3) with szz = 2, which gives 6:
//   - the same pressure p added to sxx, syy and szz (the differences stay 3, 0, -3): 6, for p = -10, 0.5, 100, 1e6
//     and 0.1;
//   - the sign changed on all the stresses (the differences change sign, the squares do not): 6;
//   - sxx and syy swapped, or the shear with its sign changed: 6;
//   - all the stresses multiplied by a number f: |f| x 6, i.e. 3, 18 and 12 for f = 0.5, 3 and -2.
TEST(VonMises2d, OnlyTheDistortionCounts) {
    EXPECT_DOUBLE_EQ(von_mises_2d({5.0, 2.0, 3.0}, 2.0), 6.0);

    for (double p : {-10.0, 0.5, 100.0, 1.0e6, 0.1}) {
        EXPECT_NEAR(von_mises_2d({5.0 + p, 2.0 + p, 3.0}, 2.0 + p), 6.0, 1e-9) << "pressure " << p;
    }
    EXPECT_NEAR(von_mises_2d({-5.0, -2.0, -3.0}, -2.0), 6.0, 1e-12);
    EXPECT_NEAR(von_mises_2d({2.0, 5.0, 3.0}, 2.0), 6.0, 1e-12);
    EXPECT_NEAR(von_mises_2d({5.0, 2.0, -3.0}, 2.0), 6.0, 1e-12);
    for (double f : {0.5, 3.0, -2.0}) {
        EXPECT_NEAR(von_mises_2d({5.0 * f, 2.0 * f, 3.0 * f}, 2.0 * f), std::fabs(f) * 6.0, 1e-12) << "f = " << f;
    }
}

// EXAMPLE 232. The von Mises does not depend on the axes chosen in the plane. Rotating x and y by an angle a
// (c = cos a, s = sin a) the in-plane stresses become
//     sxx' = c^2 sxx + s^2 syy + 2 c s sxy,      syy' = s^2 sxx + c^2 syy - 2 c s sxy,
//     sxy' = (syy - sxx) c s + (c^2 - s^2) sxy,
// while szz does not change. The state is the same, seen from different axes, so the von Mises must give the same 6 for
// the hand example (5, 2, 3) with szz = 2. At 90 degrees sxx' = 2, syy' = 5 and sxy' = -3; at 30 degrees the numbers
// are all different (6.85, 0.15 and 0.20) and the von Mises is still 6. This checks the 3 in front of sxy^2: with any
// other coefficient, rotating by 45 degrees (where shear and normal differences mix) the result would change.
TEST(VonMises2d, TheAxesOfThePlaneDoNotMatter) {
    const double sxx = 5.0;
    const double syy = 2.0;
    const double sxy = 3.0;
    const double szz = 2.0;
    for (double degrees : {0.0, 30.0, 45.0, 60.0, 90.0, 123.0}) {
        const double angle = degrees * kPi / 180.0;
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        const Vector3 rotated = {c * c * sxx + s * s * syy + 2.0 * c * s * sxy,
                                 s * s * sxx + c * c * syy - 2.0 * c * s * sxy,
                                 (syy - sxx) * c * s + (c * c - s * s) * sxy};
        EXPECT_NEAR(von_mises_2d(rotated, szz), 6.0, 1e-12) << degrees << " degrees";
        if (degrees == 30.0) {
            EXPECT_GT(std::fabs(rotated[0] - sxx), 1.0) << "at 30 degrees the stresses must have changed";
            EXPECT_GT(std::fabs(rotated[2] - sxy), 1.0) << "at 30 degrees the stresses must have changed";
        }
    }
}

// EXAMPLE 233. Every triangle uses its own stresses and its three nodes. The unit square with the triangles {0, 1, 2} and
// {0, 2, 3}, E = 100, nu = 1/4, alpha = 1/100 (E alpha = 1), T_ref = 20, and all different temperatures at the four nodes,
// T = (19.5, 19.75, 20, 24.25).
//   - Triangle 0, nodes 0, 1, 2: dT = (19.5 + 19.75 + 20) / 3 - 20 = 19.75 - 20 = -1/4. With sigma = (5, 2, 3):
//     szz = nu (5 + 2) - E alpha dT = 7/4 + 1/4 = 2, it is EXAMPLE 229 and the von Mises is 6.
//   - Triangle 1, nodes 0, 2, 3: dT = (19.5 + 20 + 24.25) / 3 - 20 = 21.25 - 20 = 5/4. With sigma = (4, 1, 1):
//     szz = nu (4 + 1) - E alpha dT = 5/4 - 5/4 = 0, and the von Mises is 4.
// Anyone who used the temperature of a single node, the mean of two, the wrong T_ref or the nodes of another triangle
// would find a different szz and another number.
TEST(VonMises2d, TheFieldUsesTheStressAndTheNodesOfEachTriangle) {
    const Mesh2D mesh({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});
    const std::vector<Vector3> sigma = {{5.0, 2.0, 3.0}, {4.0, 1.0, 1.0}};
    const std::vector<double> T = {19.5, 19.75, 20.0, 24.25};

    const std::vector<double> vm = von_mises_field_2d(mesh, sigma, 100.0, 0.25, 0.01, T, 20.0);
    ASSERT_EQ(vm.size(), 2u);
    EXPECT_NEAR(vm[0], 6.0, 1e-12);
    EXPECT_NEAR(vm[1], 4.0, 1e-12);
}

// EXAMPLE 234. The free plate that heats up uniformly: von Mises = E alpha dT. It is the plate of EXAMPLE 216
// (3 x 2 cells of [0, 2] x [0, 1], insulated, T0 = 25, T_ref = 20, one degree more at every step) with theta = 1 and
// 1/2. In the plane there are no stresses (this is checked), so sxx = syy = sxy = 0 and
//     szz = nu x 0 - E alpha dT = -dT,        sigma_vm = sqrt(1/2 (0 + dT^2 + dT^2)) = dT = 5 + p
// after p steps, in every triangle: 5, 6, 7, 8, 9. Anyone who took szz = 0 (the "thin plate" formula) would find 0:
// a plate that heats up by 9 degrees and is not stressed in any way.
TEST(VonMises2d, AFreePlateThatIsHeatedUniformlyIsStressedAcrossThePlane) {
    const Mesh2D mesh = rectangle_mesh(2.0, 1.0, 3, 2);
    for (double theta : {1.0, 0.5}) {
        ThermoMechanicalProblem2D problem;
        use_hand_material(problem);
        problem.rho_c = 2.0;
        problem.k = 1.0;
        problem.f = assemble_load_2d(mesh, 8.0);
        problem.T0 = DVector(12, 25.0);
        problem.dt = 0.25;
        problem.n_steps = 4;
        problem.theta = theta;
        problem.fixed_dofs = {0, 1, 7};
        problem.fixed_displacements = {0.0, 0.0, 0.0};

        const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
        ASSERT_EQ(states.size(), 5u);
        for (std::size_t p = 0; p < states.size(); ++p) {
            const std::string label = "theta = " + std::to_string(theta) + ", step " + std::to_string(p);
            EXPECT_LT(max_abs(states[p].sigma), 1e-8) << label << ": there are no in-plane stresses";
            expect_all_near(von_mises_of(mesh, problem, states[p]), 5.0 + static_cast<double>(p), 1e-7, label);
        }
    }
}

// EXAMPLE 235. The clamped plate that heats up is compressed hydrostatically: von Mises = 0. It is the plate
// of EXAMPLE 218 (4 x 4 cells of [0, 2] x [0, 1], the whole boundary fixed, insulated, T0 = T_ref = 20, one degree more
// at every step) with theta = 1 and 1/2. The in-plane stresses are large, sxx = syy = -2 dT, but szz = nu (sxx + syy) -
// E alpha dT = 0.25 (-4 dT) - dT = -2 dT: the three normal stresses are equal and the shear is 0, the plate is compressed
// in the same way in all directions and the von Mises is 0 (up to the errors of the conjugate gradient). It is
// checked that the in-plane stresses are really there (from step 1) and that the "thin plate" formula (szz = 0)
// would give 2 dT: the plate would seem to be at risk when it is not.
TEST(VonMises2d, AClampedPlateIsPressedFromAllSidesSoVonMisesIsZero) {
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

        const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
        ASSERT_EQ(states.size(), 5u);
        for (std::size_t p = 0; p < states.size(); ++p) {
            const std::string label = "theta = " + std::to_string(theta) + ", step " + std::to_string(p);
            const double dT = static_cast<double>(p);
            expect_all_near(von_mises_of(mesh, problem, states[p]), 0.0, 1e-7, label);
            if (p > 0) {
                EXPECT_GT(max_abs(states[p].sigma), 1.9 * dT) << label << ": in-plane stresses to be found";
            }
            for (const Vector3& s : states[p].sigma) {
                EXPECT_NEAR(von_mises_2d(s, 0.0), 2.0 * dT, 1e-7) << label << ": the formula with szz = 0";
            }
        }
    }
}

// EXAMPLE 236. In the middle of a long strip every triangle has the same von Mises. The strip of EXAMPLE 221 (24
// columns, square cells of side 1/2, backward Euler, one step, E = 100, nu = 1/4, alpha = 1/100, rows of nodes
// at dT = 0, 60, 0). In the three middle columns the four triangles have the sigmas of EXAMPLE 221 and, for those of the
// bottom cell (row 0), with sigma_zz = nu (sxx + syy) - E alpha dT_X:
//   - bottom triangle, sigma = (-20, +20), dT_X = 20: szz = 0.25 x 0 - 20 = -20 (equal to sxx).
//     sigma_vm = sqrt(1/2 [(-20 - 20)^2 + (20 + 20)^2 + (-20 + 20)^2]) = sqrt(1/2 (1600 + 1600)) = 40;
//   - top triangle, sigma = (-60, -20), dT_X = 40: szz = 0.25 x (-80) - 40 = -60 (equal to sxx).
//     sigma_vm = sqrt(1/2 [(-60 + 20)^2 + (-20 + 60)^2 + 0]) = 40.
// Row 1 is the mirror image: 40 in all four. It is the textbook formula, E alpha dT / (1 - nu) = 30 / 0.75 = 40 with
// dT = 30 (the mean of 0 and 60 of the cell). After one step (g = 2/3) everything is multiplied by 2/3. Without szz (with
// szz = 0) one would get sqrt(1200) = 34.6 in the bottom triangles and sqrt(2800) = 52.9 in the top ones.
TEST(VonMises2d, InTheMiddleOfALongStripEveryTriangleHasTheSameVonMises) {
    const std::size_t nx = 24;
    const Mesh2D mesh = rectangle_mesh(0.5 * static_cast<double>(nx), 1.0, nx, 2);
    ThermoMechanicalProblem2D problem = roller_slab_problem(nx, 2, 20.0, 60.0);
    use_hand_material(problem);
    problem.rho_c = 2.0;
    problem.k = 3.0;
    problem.dt = 1.0 / 36.0;
    problem.theta = 1.0;
    problem.n_steps = 1;
    problem.save_every = 1;

    const std::vector<ThermoMechanicalState2D> states = solve_thermomechanical_2d(mesh, problem);
    ASSERT_EQ(states.size(), 2u);
    for (std::size_t p = 0; p < states.size(); ++p) {
        const double factor = (p == 0) ? 1.0 : 2.0 / 3.0;
        const std::vector<double> vm = von_mises_of(mesh, problem, states[p]);
        for (std::size_t i = nx / 2 - 1; i <= nx / 2 + 1; ++i) {
            for (std::size_t row = 0; row < 2; ++row) {
                for (std::size_t half = 0; half < 2; ++half) {
                    EXPECT_NEAR(vm[2 * (row * nx + i) + half], factor * 40.0, 1e-3)
                        << "step " << p << ", cell (" << i << ", " << row << "), triangle " << half;
                }
            }
        }
    }
}

// EXAMPLE 237. A steel plate that cools has the "textbook" von Mises. The plate of EXAMPLE 223 (H = 0.1 m thick
// and 0.4 m long, E = 210 GPa, nu = 0.3, alpha = 12e-6, rho_c = 7850 x 470, k = 45, the faces at T_ref = 293.15 K,
// starting from T_ref + 100 sin(pi y / H), 8 x 32 cells, steps of 5 s with Crank-Nicolson, states at 0, 50 and 100 s).
// At mid-length, with syy averaging 0 (vertical equilibrium) and szz = sxx, the von Mises of every triangle of
// a cell is the |sxx| of the mean in EXAMPLE 223, i.e.
//     sigma_vm = E alpha dT / (1 - nu),        dT = mean of the T deviation of the two rows of nodes of the cell
// (100 K of deviation give 360 MPa). The test checks:
//   - all the numbers of the plate are finite and non-negative (a von Mises is the root of a sum of squares);
//   - the two triangles of every middle cell have the formula with the computed T deviation to within 1e-4 times
//     360 MPa (the effect of the rollers at mid-length is below 1e-5);
//   - with the "continuous" deviation, 100 sin(pi y / H) e^(-t / tau) taken at mid-height of the cell (tau = 83 s), the
//     difference is the error of the method (1.9% the mean of two nodes, 1.3% the decay of the mesh): below 5% of
//     360 MPa;
//   - at mid-height, at time 0, the von Mises is between 95% and 100% of 360 MPa.
TEST(VonMises2d, ASteelSlabHasTheTextbookVonMises) {
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
        const std::vector<double> vm = von_mises_of(mesh, problem, state);
        ASSERT_EQ(vm.size(), mesh.num_triangles());
        for (std::size_t e = 0; e < vm.size(); ++e) {
            EXPECT_TRUE(std::isfinite(vm[e])) << "t = " << state.time << ", triangle " << e;
            EXPECT_GE(vm[e], 0.0) << "t = " << state.time << ", triangle " << e;
        }
        for (std::size_t row = 0; row < ny; ++row) {
            const std::size_t cell = row * nx + column;
            const double below = state.T[row * (nx + 1) + column] - t_ref;
            const double above = state.T[(row + 1) * (nx + 1) + column] - t_ref;
            const double from_the_formula = problem.E * problem.alpha * 0.5 * (below + above) / (1.0 - problem.nu);
            const double y_middle = (static_cast<double>(row) + 0.5) * height / static_cast<double>(ny);
            const double continuum = scale * std::sin(kPi * y_middle / height) * std::exp(-state.time / tau);
            for (std::size_t half = 0; half < 2; ++half) {
                EXPECT_NEAR(vm[2 * cell + half], from_the_formula, 1e-4 * scale)
                    << "t = " << state.time << ", row of cells " << row << ", triangle " << half
                    << ", formula with the computed T";
                EXPECT_NEAR(vm[2 * cell + half], continuum, 0.05 * scale)
                    << "t = " << state.time << ", row of cells " << row << ", triangle " << half
                    << ", continuous solution";
            }
        }
    }

    const std::vector<double> vm_start = von_mises_of(mesh, problem, states[0]);
    const std::size_t middle_cell = (ny / 2) * nx + column;
    EXPECT_GT(vm_start[2 * middle_cell], 0.95 * scale);
    EXPECT_LT(vm_start[2 * middle_cell], 1.0 * scale);
}

// EXAMPLE 238. Scaling laws: the irregular mesh and the data of EXAMPLE 214 with zero constraints. In the state of every
// instant, with respect to the numbers of the base problem:
//   - E doubled (sigma and E alpha double, so szz too): the von Mises is doubled;
//   - alpha doubled (u, sigma and E alpha dT double): doubled;
//   - the heat doubled (initial and imposed T deviation and load times 2): doubled;
//   - the heat with its sign changed (factor -1): the SAME von Mises. All the stresses change sign (szz included) and
//     a von Mises is made of squared differences: a piece that heats up by 10 degrees or cools down by 10 is
//     stressed in the same way. Anyone who put an absolute value in the wrong place, or took the root of a
//     signed number, would find it here;
//   - T and T_ref increased together by 100 degrees: the same von Mises (it depends only on T - T_ref).
TEST(VonMises2d, VonMisesScalesWithEWithAlphaAndWithTheHeat) {
    const Mesh2D mesh = general_mesh();
    const ThermoMechanicalProblem2D base = general_problem(mesh);
    const std::vector<ThermoMechanicalState2D> base_states = solve_thermomechanical_2d(mesh, base);
    ASSERT_EQ(base_states.size(), 4u);
    std::vector<std::vector<double>> base_vm;
    for (const ThermoMechanicalState2D& state : base_states) {
        base_vm.push_back(von_mises_of(mesh, base, state));
    }
    ASSERT_GT(max_abs(base_vm[3]), 1e-3) << "the test must have some stresses to compare";

    ThermoMechanicalProblem2D stiff = base;
    stiff.E = 2.0 * base.E;
    expect_von_mises_scaled(mesh, stiff, base_vm, 2.0, "E doubled");

    ThermoMechanicalProblem2D expanding = base;
    expanding.alpha = 2.0 * base.alpha;
    expect_von_mises_scaled(mesh, expanding, base_vm, 2.0, "alpha doubled");

    expect_von_mises_scaled(mesh, with_heat_times(base, 2.0), base_vm, 2.0, "heat doubled");
    expect_von_mises_scaled(mesh, with_heat_times(base, -1.0), base_vm, 1.0, "heat with the sign changed");

    ThermoMechanicalProblem2D shifted = base;
    shifted.T_ref += 100.0;
    for (double& value : shifted.T0) {
        value += 100.0;
    }
    for (double& value : shifted.fixed_temperatures) {
        value += 100.0;
    }
    expect_von_mises_scaled(mesh, shifted, base_vm, 1.0, "T e T_ref + 100");
}

// EXAMPLE 239. Wrong data: the function throws std::invalid_argument. The unit square with two triangles, two
// stresses, four temperatures, E = 100, nu = 1/4 and alpha = 1/100 (it is checked that the data is valid). One
// datum is changed at a time:
//   - sigma with 1 or with 3 triplets instead of 2 (one per triangle);
//   - T with 3 or with 5 numbers instead of 4 (one per node);
//   - E = 0, E = -5 and E = NaN (E > 0 is needed, and with a NaN the comparison "E <= 0" is not enough: "!(E > 0)" is);
//   - nu = 1/2, -1, 0.7, -1.5 and NaN (-1 < nu < 1/2 is needed, ends excluded, as for plane_strain_matrix).
// Values close to the limits but inside (nu = 0.49 and nu = -0.99) are fine.
TEST(VonMises2d, WrongDataIsReported) {
    const Mesh2D mesh({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});
    const std::vector<Vector3> sigma = {{5.0, 2.0, 3.0}, {4.0, 1.0, 1.0}};
    const std::vector<Vector3> one_sigma(1, Vector3{5.0, 2.0, 3.0});
    const std::vector<Vector3> three_sigma(3, Vector3{5.0, 2.0, 3.0});
    const std::vector<double> T(4, 20.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    ASSERT_NO_THROW(von_mises_field_2d(mesh, sigma, 100.0, 0.25, 0.01, T, 20.0));

    EXPECT_THROW(von_mises_field_2d(mesh, one_sigma, 100.0, 0.25, 0.01, T, 20.0), std::invalid_argument);
    EXPECT_THROW(von_mises_field_2d(mesh, three_sigma, 100.0, 0.25, 0.01, T, 20.0), std::invalid_argument);
    EXPECT_THROW(von_mises_field_2d(mesh, sigma, 100.0, 0.25, 0.01, std::vector<double>(3, 20.0), 20.0),
                 std::invalid_argument);
    EXPECT_THROW(von_mises_field_2d(mesh, sigma, 100.0, 0.25, 0.01, std::vector<double>(5, 20.0), 20.0),
                 std::invalid_argument);
    for (double E : {0.0, -5.0, nan}) {
        EXPECT_THROW(von_mises_field_2d(mesh, sigma, E, 0.25, 0.01, T, 20.0), std::invalid_argument) << "E = " << E;
    }
    for (double nu : {0.5, -1.0, 0.7, -1.5, nan}) {
        EXPECT_THROW(von_mises_field_2d(mesh, sigma, 100.0, nu, 0.01, T, 20.0), std::invalid_argument)
            << "nu = " << nu;
    }
    EXPECT_NO_THROW(von_mises_field_2d(mesh, sigma, 100.0, 0.49, 0.01, T, 20.0));
    EXPECT_NO_THROW(von_mises_field_2d(mesh, sigma, 100.0, -0.99, 0.01, T, 20.0));
}

// EXAMPLE 240. A NaN in the data does not become a number. If a stress is NaN (for example because an earlier
// computation went wrong) the von Mises must be NaN: a 0, or any other number, would look like "no stress" and
// would hide the fault. It holds for each of the four stresses. (One way of hiding it: protecting the root with
// std::max(0.0, ...) or with "if > 0": with a NaN the comparison gives false and the result becomes 0.) In the field, a
// NaN temperature at a node gives NaN in the triangles that have that node and only in those: in the unit square with the
// triangles {0, 1, 2} and {0, 2, 3} and node 3 at NaN, triangle 0 has its von Mises (6, as in EXAMPLE 233 with
// T = (19.5, 19.75, 20)) and triangle 1 is NaN.
TEST(VonMises2d, ANanIsNeverHiddenAsANumber) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(von_mises_2d({nan, 2.0, 3.0}, 2.0)));
    EXPECT_TRUE(std::isnan(von_mises_2d({5.0, nan, 3.0}, 2.0)));
    EXPECT_TRUE(std::isnan(von_mises_2d({5.0, 2.0, nan}, 2.0)));
    EXPECT_TRUE(std::isnan(von_mises_2d({5.0, 2.0, 3.0}, nan)));

    const Mesh2D mesh({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});
    const std::vector<Vector3> sigma = {{5.0, 2.0, 3.0}, {4.0, 1.0, 1.0}};
    const std::vector<double> T = {19.5, 19.75, 20.0, nan};
    const std::vector<double> vm = von_mises_field_2d(mesh, sigma, 100.0, 0.25, 0.01, T, 20.0);
    ASSERT_EQ(vm.size(), 2u);
    EXPECT_NEAR(vm[0], 6.0, 1e-12);
    EXPECT_TRUE(std::isnan(vm[1]));
}

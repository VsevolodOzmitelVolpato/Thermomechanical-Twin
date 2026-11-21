#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Elasticity2D.h"
#include "Fem1D.h"
#include "Mesh2D.h"
#include "Verification2D.h"

// ---- The bending beam: an exact solution in plane strain -----------------------------------------------------------
//
// So far every elasticity function has been tested with hand-made numbers or limit cases. Here the whole chain is tested
// together (mesh, matrix, thermal force, solver, stresses) on a problem whose EXACT solution is known,
// as in 2D for heat with the manufactured sine (EXAMPLES 93 and 94): the error of the method is computed and we look
// at whether it goes down as it should when the mesh is refined.
//
// THE PROBLEM. A free beam, the rectangle [0, L] x [0, H], in plane strain, with the temperature increasing
// linearly with the height (G is the gradient: kelvin per unit of y):
//     T(x, y) = T_ref + G y~,        y~ = y - H / 2.
// Above the center it is hotter, below colder. The thermal strain of a point is
//     eps_th = (1 + nu) alpha G y~ (1, 1, 0) = kappa y~ (1, 1, 0),        kappa = (1 + nu) alpha G.
//
// THE EXACT SOLUTION. We look for a displacement (u, v) whose strain is exactly eps_th: then B u - eps_th = 0
// and the stress sigma = D (B u - eps_th) is zero everywhere. The conditions are
//     du/dx = kappa y~,        dv/dy = kappa y~,        du/dy + dv/dx = 0.
// The first says u = kappa x y~ (plus a function of y), the second v = kappa y~^2 / 2 (plus a function of x). With
// u = kappa x y~ we have du/dy = kappa x, so the third asks dv/dx = -kappa x: the function of x in v must be
// -kappa x^2 / 2. We get
//     u(x, y) = kappa x y~,        v(x, y) = kappa (y~^2 - x^2) / 2.
// The beam has no stress at all: it just bends. The mid line (y~ = 0) becomes u = 0, v = -kappa x^2 / 2, a
// parabola, which for small kappa is an arc of a circle of radius 1 / kappa: the hot side gets longer, the cold one
// gets shorter. (In plane stress it would be kappa = alpha G; in plane strain there is the extra factor (1 + nu).)
//
// THE THREE CONDITIONS. The beam is free, so moving or rotating it changes nothing: three conditions are needed that
// remove the three rigid motions (two translations and a rotation). They are taken from the exact solution, at the corner (0, 0)
// and at the one (L, 0):
//     u(0, 0) = 0,        v(0, 0) = kappa H^2 / 8,        v(L, 0) = kappa (H^2 / 4 - L^2) / 2.
// With these the problem has a single solution and, if the method is right, it is the exact one.
//
// THE ERROR OF THE METHOD. The triangles are linear: the displacement is linear in each triangle, the exact one is
// quadratic. And the temperature of a triangle is the mean of its three nodes, so the thermal strain is
// constant in each triangle, whereas the exact one grows with y. The method is wrong, and the theory says how the error
// goes down with the size h of the cells: the displacement like h^2, the stress like h (it is a derivative, and here it should
// be ZERO). The errors are two numbers:
//     displacement = the maximum, over the nodes, of the distance between the computed node and the exact one,
//                    sqrt((u_FEM - u_exact)^2 + (v_FEM - v_exact)^2);
//     stress       = the maximum, over the triangles, of the computed magnitude sqrt(sigma_xx^2 + sigma_yy^2 + sigma_xy^2).
// With few cells in height the linear triangles are too stiff in bending (with a single cell they hardly
// bend): the theoretical behavior appears from the levels with 4 cells in height onwards.
//
// There are two functions, in Verification2D like the ones for the sine.
//   - bending_beam_exact_displacement_2d(mesh, beam) gives the exact displacements at the nodes of the mesh (2 numbers per
//     node);
//   - bending_beam_errors_2d(beam, nx, ny) solves the beam with nx x ny cells (rectangle_mesh) and gives the two errors.
// What they must do, and this is what the tests check:
//   - the exact displacements are computed by hand, on the square (EXAMPLE 197) and on an irregular mesh with H = 3
//     (EXAMPLE 198); the values of the three constraints and the mid line of the steel are checked (EXAMPLE 199); their
//     strain is the thermal one, tested with finite differences (EXAMPLE 200); they depend only on height, nu,
//     alpha and gradient, and are linear in alpha G (EXAMPLE 201);
//   - without a temperature gradient the beam does not deform and has no stress (EXAMPLE 202);
//   - the errors have the numbers of the README table (EXAMPLE 203) and go down with order 2 for the displacement and 1
//     for the stress (EXAMPLES 204 and 205);
//   - scaling laws that the method must respect: the errors are linear in G and in alpha (EXAMPLE 206), only the
//     stress depends on E (EXAMPLE 207), T_ref does not matter (EXAMPLE 208), and if the beam shrinks by a
//     factor s the displacement scales like s^2 and the stress like s (EXAMPLE 210);
//   - other Poisson ratios (EXAMPLE 209) and wrong data (EXAMPLE 211).

namespace {

// A beam with convenient numbers for hand computations: E = 1, nu = 1/4, alpha = 1/100 and gradient = 160, so
// kappa = (1 + 1/4) (1/100) 160 = 2. Length and height are chosen case by case.
BendingBeam hand_beam(double length, double height) {
    BendingBeam beam;
    beam.length = length;
    beam.height = height;
    beam.E = 1.0;
    beam.nu = 0.25;
    beam.alpha = 0.01;
    beam.gradient = 160.0;
    beam.T_ref = 0.0;
    return beam;
}

// The steel beam: 4 m by 1 m, E = 210 GPa, nu = 0.3, alpha = 12e-6 per kelvin, the temperature rises by 100 K for
// every meter in y (50 K more on the upper side, 50 K less on the lower one) and the reference is 20 degrees Celsius.
// kappa = (1 + 0.3) 12e-6 100 = 1.56e-3 per meter: the radius of curvature is 1 / kappa = 641 m.
BendingBeam steel_beam() {
    BendingBeam beam;
    beam.length = 4.0;
    beam.height = 1.0;
    beam.E = 210.0e9;
    beam.nu = 0.3;
    beam.alpha = 12.0e-6;
    beam.gradient = 100.0;
    beam.T_ref = 293.15;
    return beam;
}

// The errors of the steel beam with 4 k x k cells (square cells of side H / k). Each level is computed only once:
// the tests that use more than one do not redo the computations.
const BendingBeamErrors& steel_errors(std::size_t k) {
    static std::map<std::size_t, BendingBeamErrors> computed;
    auto found = computed.find(k);
    if (found == computed.end()) {
        found = computed.emplace(k, bending_beam_errors_2d(steel_beam(), 4 * k, k)).first;
    }
    return found->second;
}

// A pair of errors, assigned by name (the order of the structure members does not matter).
BendingBeamErrors errors_of(double displacement, double stress) {
    BendingBeamErrors errors;
    errors.displacement = displacement;
    errors.stress = stress;
    return errors;
}

// Two vectors equal up to an absolute tolerance.
void expect_vectors_near(const std::vector<double>& got, const std::vector<double>& want, double tolerance) {
    ASSERT_EQ(got.size(), want.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tolerance) << "component " << i;
    }
}

// Two errors equal up to a relative tolerance: displacement with displacement, stress with stress.
void expect_errors_near(const BendingBeamErrors& got, const BendingBeamErrors& want, double relative,
                        const std::string& label) {
    EXPECT_NEAR(got.displacement, want.displacement, relative * std::fabs(want.displacement))
        << label << ", displacement";
    EXPECT_NEAR(got.stress, want.stress, relative * std::fabs(want.stress)) << label << ", stress";
}

}  // namespace

// EXAMPLE 197. The unit square, with kappa = 2 and H = 1 (so y~ = y - 1/2): u = 2 x y~ and v = (y~^2 - x^2).
//     node 0 = (0, 0): y~ = -1/2:  u = 0,               v = 1/4 - 0 = 1/4
//     node 1 = (1, 0): y~ = -1/2:  u = 2 (1) (-1/2) = -1,   v = 1/4 - 1 = -3/4
//     node 2 = (1, 1): y~ = +1/2:  u = 2 (1) (1/2) = 1,     v = 1/4 - 1 = -3/4
//     node 3 = (0, 1): y~ = +1/2:  u = 0,               v = 1/4
// The vector is (u0, v0, u1, v1, u2, v2, u3, v3).
TEST(ElasticityBeam2d, ExactDisplacementOfTheUnitSquareByHand) {
    const Mesh2D mesh({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});

    const std::vector<double> got = bending_beam_exact_displacement_2d(mesh, hand_beam(1.0, 1.0));
    const std::vector<double> want = {0.0, 0.25, -1.0, -0.75, 1.0, -0.75, 0.0, 0.25};
    expect_vectors_near(got, want, 1e-14);
}

// EXAMPLE 198. An arbitrary mesh, not a grid, and a height different from 1: the five nodes of the irregular mesh
// (0, 0), (4, 0), (3, 3), (-1, 2), (1, 0.7) with kappa = 2 and H = 3, so y~ = y - 3/2. The function must use the
// node coordinates and the TRUE height (H / 2 = 3/2, not 1/2).
//     node 0 (0, 0):     y~ = -3/2:  u = 2 (0) (-3/2) = 0,       v = (9/4 - 0) = 9/4
//     node 1 (4, 0):     y~ = -3/2:  u = 2 (4) (-3/2) = -12,     v = 9/4 - 16 = -55/4
//     node 2 (3, 3):     y~ = +3/2:  u = 2 (3) (3/2) = 9,        v = 9/4 - 9 = -27/4
//     node 3 (-1, 2):    y~ = +1/2:  u = 2 (-1) (1/2) = -1,      v = 1/4 - 1 = -3/4
//     node 4 (1, 0.7):   y~ = -4/5:  u = 2 (1) (-4/5) = -1.6,    v = 16/25 - 1 = -0.36
TEST(ElasticityBeam2d, ExactDisplacementOnAnIrregularMeshByHand) {
    const Mesh2D mesh({0.0, 4.0, 3.0, -1.0, 1.0}, {0.0, 0.0, 3.0, 2.0, 0.7},
                      {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}});

    const std::vector<double> got = bending_beam_exact_displacement_2d(mesh, hand_beam(4.0, 3.0));
    const std::vector<double> want = {0.0, 2.25, -12.0, -13.75, 9.0, -6.75, -1.0, -0.75, -1.6, -0.36};
    expect_vectors_near(got, want, 1e-13);
}

// EXAMPLE 199. The steel beam, kappa = 1.56e-3, on the 8 x 2 mesh (9 nodes per row: row y = 0 has nodes 0..8,
// row y = 0.5 nodes 9..17, row y = 1 nodes 18..26; the nodes of a row are at x = 0, 0.5, ..., 4).
//   - The three constraints (the values that bending_beam_errors_2d imposes at node 0 = (0, 0) and at node 8 = (4, 0)):
//         u(0, 0) = 0,    v(0, 0) = kappa H^2 / 8 = 1.56e-3 / 8 = 1.95e-4,
//         v(L, 0) = kappa (H^2 / 4 - L^2) / 2 = 1.56e-3 (0.25 - 16) / 2 = -0.012285.
//   - The mid line y = H / 2 = 0.5 (nodes 9..17): y~ = 0, so u = 0 and v = -kappa x^2 / 2. At the tip (x = 4) the
//     deflection is kappa L^2 / 2 = 1.56e-3 (16) / 2 = 0.01248, i.e. 12.48 mm downwards: a steel beam of
//     four meters, 100 K between top and bottom over a meter of height.
TEST(ElasticityBeam2d, TheSupportValuesAndTheCenterLineOfTheSteelBeam) {
    const double kappa = 1.56e-3;
    const Mesh2D mesh = rectangle_mesh(4.0, 1.0, 8, 2);

    const std::vector<double> u = bending_beam_exact_displacement_2d(mesh, steel_beam());
    ASSERT_EQ(u.size(), 54u);

    EXPECT_NEAR(u[0], 0.0, 1e-18);
    EXPECT_NEAR(u[1], 1.95e-4, 1e-15);
    EXPECT_NEAR(u[2 * 8 + 1], -0.012285, 1e-14);

    for (std::size_t i = 0; i <= 8; ++i) {
        const double x = 0.5 * static_cast<double>(i);
        EXPECT_NEAR(u[2 * (9 + i)], 0.0, 1e-18) << "x = " << x;
        EXPECT_NEAR(u[2 * (9 + i) + 1], -kappa * x * x / 2.0, 1e-15) << "x = " << x;
    }
    EXPECT_NEAR(u[2 * 17 + 1], -0.01248, 1e-14);
}

// EXAMPLE 200. The exact solution has exactly the thermal strain as its strain. This is checked with finite
// differences: at a point (x0, y0) four nodes are taken at distance d (right, above, left, below) and
//     eps_xx = du/dx = (u_right - u_left) / (2 d),        eps_yy = dv/dy = (v_above - v_below) / (2 d),
//     gamma_xy = du/dy + dv/dx = (u_above - u_below) / (2 d) + (v_right - v_left) / (2 d).
// The displacements are of second degree (u is linear in x and in y, v is of second degree in x and in y): the central
// differences are EXACT, so the three numbers must be (up to rounding) those of the thermal
// strain of plane_strain_thermal_strain with dT = G y~ = G (y0 - H / 2), i.e. kappa y~ (1, 1, 0), and the shear 0. This
// is tried with three beams (kappa = 2 and H = 1, kappa = 2 and H = 3, the steel) and four points, also outside the beam.
TEST(ElasticityBeam2d, TheExactDisplacementCancelsTheThermalStrain) {
    const double d = 0.25;
    const std::vector<std::pair<double, double>> points = {{0.3, 0.2}, {0.9, 0.55}, {3.5, 0.95}, {-1.0, 2.0}};

    for (const BendingBeam& beam : {hand_beam(1.0, 1.0), hand_beam(4.0, 3.0), steel_beam()}) {
        const double kappa = (1.0 + beam.nu) * beam.alpha * beam.gradient;
        for (const auto& point : points) {
            const double x0 = point.first;
            const double y0 = point.second;
            // nodes: 0 = center, 1 = right, 2 = above, 3 = left, 4 = below; a single triangle (0, 1, 2)
            const Mesh2D mesh({x0, x0 + d, x0, x0 - d, x0}, {y0, y0, y0 + d, y0, y0 - d}, {{0, 1, 2}});

            const std::vector<double> u = bending_beam_exact_displacement_2d(mesh, beam);
            ASSERT_EQ(u.size(), 10u);
            const double exx = (u[2 * 1] - u[2 * 3]) / (2.0 * d);
            const double eyy = (u[2 * 2 + 1] - u[2 * 4 + 1]) / (2.0 * d);
            const double gxy = (u[2 * 2] - u[2 * 4]) / (2.0 * d) + (u[2 * 1 + 1] - u[2 * 3 + 1]) / (2.0 * d);

            const Vector3 eps_th =
                plane_strain_thermal_strain(beam.nu, beam.alpha, beam.gradient * (y0 - beam.height / 2.0));
            const double tolerance = 1e-12 * std::fabs(kappa);
            EXPECT_NEAR(exx, eps_th[0], tolerance) << "H = " << beam.height << ", (" << x0 << ", " << y0 << ")";
            EXPECT_NEAR(eyy, eps_th[1], tolerance) << "H = " << beam.height << ", (" << x0 << ", " << y0 << ")";
            EXPECT_NEAR(gxy, 0.0, tolerance) << "H = " << beam.height << ", (" << x0 << ", " << y0 << ")";
            EXPECT_GT(std::fabs(eps_th[0]), 1e-4 * std::fabs(kappa)) << "the point must be off the mid line";
        }
    }
}

// EXAMPLE 201. The exact displacements depend only on height, nu, alpha and gradient, and only through
// kappa = (1 + nu) alpha G, linearly. On the 8 x 2 mesh of the steel beam:
//   - a specific node: node 26 = (4, 1) has y~ = 1/2, so u = kappa (4) (1/2) = 2 kappa = 3.12e-3 and
//     v = kappa (1/4 - 16) / 2 = -7.875 kappa = -0.012285;
//   - length, E and T_ref do not matter: changing them leaves the vector identical;
//   - doubling G or doubling alpha doubles every number, changing the sign of G changes all of them in sign, G = 0 zeroes them.
TEST(ElasticityBeam2d, TheExactDisplacementUsesOnlyHeightPoissonRatioAlphaAndGradient) {
    const Mesh2D mesh = rectangle_mesh(4.0, 1.0, 8, 2);
    const BendingBeam beam = steel_beam();
    const std::vector<double> base = bending_beam_exact_displacement_2d(mesh, beam);
    ASSERT_EQ(base.size(), 54u);

    EXPECT_NEAR(base[2 * 26], 3.12e-3, 1e-15);
    EXPECT_NEAR(base[2 * 26 + 1], -0.012285, 1e-14);

    BendingBeam other = beam;
    other.length = 17.0;
    other.E = 1.0;
    other.T_ref = -40.0;
    EXPECT_EQ(bending_beam_exact_displacement_2d(mesh, other), base);

    other = beam;
    other.gradient = 2.0 * beam.gradient;
    std::vector<double> got = bending_beam_exact_displacement_2d(mesh, other);
    for (std::size_t i = 0; i < base.size(); ++i) {
        EXPECT_DOUBLE_EQ(got[i], 2.0 * base[i]) << "doubled gradient, component " << i;
    }

    other = beam;
    other.alpha = 2.0 * beam.alpha;
    got = bending_beam_exact_displacement_2d(mesh, other);
    for (std::size_t i = 0; i < base.size(); ++i) {
        EXPECT_DOUBLE_EQ(got[i], 2.0 * base[i]) << "doubled alpha, component " << i;
    }

    other = beam;
    other.gradient = -beam.gradient;
    got = bending_beam_exact_displacement_2d(mesh, other);
    for (std::size_t i = 0; i < base.size(); ++i) {
        EXPECT_DOUBLE_EQ(got[i], -base[i]) << "opposite gradient, component " << i;
    }

    other = beam;
    other.gradient = 0.0;
    for (double value : bending_beam_exact_displacement_2d(mesh, other)) {
        EXPECT_EQ(value, 0.0);
    }
}

// EXAMPLE 202. Without a temperature gradient (G = 0) the temperature is T_ref everywhere, the beam does not expand and
// does not bend: the thermal force is 0, the exact displacement is 0, the computed one is 0 and the stresses are 0. So
// both errors are 0, on every mesh (four are tried).
TEST(ElasticityBeam2d, ABeamWithoutTemperatureGradientIsNeitherBentNorStressed) {
    BendingBeam beam = steel_beam();
    beam.gradient = 0.0;

    const std::vector<std::pair<std::size_t, std::size_t>> cells = {{1, 1}, {4, 1}, {8, 2}, {3, 5}};
    for (const auto& c : cells) {
        const BendingBeamErrors errors = bending_beam_errors_2d(beam, c.first, c.second);
        EXPECT_NEAR(errors.displacement, 0.0, 1e-15) << c.first << " x " << c.second << " cells";
        EXPECT_NEAR(errors.stress, 0.0, 1e-3) << c.first << " x " << c.second << " cells";
    }
}

// EXAMPLE 203. The numbers of the README table, for the steel beam (4 m by 1 m, G = 100 K/m) with 4 k x k cells,
// measured with this method (and confirmed by a second independent program, with a direct solver):
//      k     displacement      stress (Pa)
//      1     2.2730600e-03     8.1688961e+07
//      2     1.3743698e-03     5.9064736e+07
//      4     5.5030371e-04     4.3477999e+07
//      8     1.6650762e-04     2.1585121e+07
//     16     4.5533882e-05     1.0131672e+07
// The test pins them: if the method changes (the load rule, the constraints, the error norm) the numbers change, and
// the test says that the README table must be redone. With one cell in height the computed stress is 82 MPa, where
// the exact one is 0: for comparison the typical thermal stress of this beam is E alpha dT = 210e9 x 12e-6 x 50 = 126
// MPa. The mesh cannot bend and the heat becomes stress.
TEST(ElasticityBeam2d, ErrorsMatchTheTableInTheReadme) {
    struct Row {
        std::size_t k;
        double displacement;
        double stress;
    };
    const std::vector<Row> table = {{1, 2.2730600e-03, 8.1688961e+07},
                                    {2, 1.3743698e-03, 5.9064736e+07},
                                    {4, 5.5030371e-04, 4.3477999e+07},
                                    {8, 1.6650762e-04, 2.1585121e+07},
                                    {16, 4.5533882e-05, 1.0131672e+07}};

    for (const Row& row : table) {
        const BendingBeamErrors& errors = steel_errors(row.k);
        EXPECT_NEAR(errors.displacement, row.displacement, 1e-6 * row.displacement) << "k = " << row.k;
        EXPECT_NEAR(errors.stress, row.stress, 1e-6 * row.stress) << "k = " << row.k;
    }
}

// EXAMPLE 204. With linear elements the displacement goes like h^2: when the cells are halved the error is divided by
// 4, i.e. the observed order (observed_order) is 2. Measured: 0.73 from 1 to 2 cells in height, 1.32 from 2 to 4, 1.72 from
// 4 to 8, 1.87 from 8 to 16 (and 1.90 from 16 to 32): the order rises towards 2 as the mesh becomes fine enough to see
// the curvature. That is why we look from 4 upwards: between 1.6 and 2.1. At all levels the error must go down anyway.
TEST(ElasticityBeam2d, DisplacementErrorGoesDownWithObservedOrderTwo) {
    double previous = steel_errors(1).displacement;
    for (std::size_t k : {2u, 4u, 8u, 16u}) {
        const double current = steel_errors(k).displacement;
        EXPECT_LT(current, previous) << "k = " << k / 2 << " -> " << k;
        if (k >= 8) {
            const double order = observed_order(previous, current, 2.0);
            EXPECT_GT(order, 1.6) << "k = " << k / 2 << " -> " << k;
            EXPECT_LT(order, 2.1) << "k = " << k / 2 << " -> " << k;
        }
        previous = current;
    }
}

// EXAMPLE 205. The stress is a derivative of the displacement: in linear triangles it is constant, and the error goes like
// h, i.e. the observed order is 1 (here the error is all of the computed stress, because the exact one is 0).
// Measured: 0.47 from 1 to 2 cells in height, 0.44 from 2 to 4, 1.01 from 4 to 8, 1.09 from 8 to 16, 1.01 from 16 to 32. We look
// from 4 upwards: between 0.9 and 1.25. Here too the error must go down at every level.
TEST(ElasticityBeam2d, StressErrorGoesDownWithObservedOrderOne) {
    double previous = steel_errors(1).stress;
    for (std::size_t k : {2u, 4u, 8u, 16u}) {
        const double current = steel_errors(k).stress;
        EXPECT_LT(current, previous) << "k = " << k / 2 << " -> " << k;
        if (k >= 8) {
            const double order = observed_order(previous, current, 2.0);
            EXPECT_GT(order, 0.9) << "k = " << k / 2 << " -> " << k;
            EXPECT_LT(order, 1.25) << "k = " << k / 2 << " -> " << k;
        }
        previous = current;
    }
}

// EXAMPLE 206. The problem is linear: if the gradient G (or alpha) doubles, the exact solution doubles and so does the one
// of the method, so the two errors double; if G changes sign the beam bends the other way and the
// errors are the same. In general the errors are proportional to |G| and to alpha. On the 8 x 2 mesh (k = 2) with
// G = 250, G = -100, G = 2.5 and alpha = 6e-6 and 24e-6 (instead of 12e-6): the factors are 2.5, 1, 0.025, 0.5 and 2.
TEST(ElasticityBeam2d, ErrorsAreLinearInTheGradientAndInTheExpansionCoefficient) {
    const BendingBeamErrors base = steel_errors(2);
    EXPECT_GT(base.displacement, 1e-4);
    EXPECT_GT(base.stress, 1e6);

    for (double gradient : {250.0, -100.0, 2.5}) {
        BendingBeam beam = steel_beam();
        beam.gradient = gradient;
        const double factor = std::fabs(gradient) / 100.0;
        const BendingBeamErrors got = bending_beam_errors_2d(beam, 8, 2);
        const BendingBeamErrors want = errors_of(factor * base.displacement, factor * base.stress);
        expect_errors_near(got, want, 1e-9, "G = " + std::to_string(gradient));
    }
    for (double alpha : {6.0e-6, 24.0e-6}) {
        BendingBeam beam = steel_beam();
        beam.alpha = alpha;
        const double factor = alpha / 12.0e-6;
        const BendingBeamErrors got = bending_beam_errors_2d(beam, 8, 2);
        const BendingBeamErrors want = errors_of(factor * base.displacement, factor * base.stress);
        expect_errors_near(got, want, 1e-9, "alpha = " + std::to_string(alpha));
    }
}

// EXAMPLE 207. Young's modulus E multiplies the whole matrix D, so it does not change the displacements
// (K u = F: K and F grow together) but multiplies the stresses (sigma = D (...)). With E = 1, 2.1e9 and 4.2e11
// (steel is 2.1e11, i.e. 210 GPa) the displacement error stays the same and the stress error is proportional
// to E: the factors are 1 / 2.1e11, 0.01 and 2.
TEST(ElasticityBeam2d, OnlyTheStressDependsOnYoungModulus) {
    const BendingBeamErrors base = steel_errors(2);
    EXPECT_GT(base.displacement, 1e-4);
    EXPECT_GT(base.stress, 1e6);

    for (double young : {1.0, 2.1e9, 4.2e11}) {
        BendingBeam beam = steel_beam();
        beam.E = young;
        const double factor = young / 210.0e9;
        const BendingBeamErrors got = bending_beam_errors_2d(beam, 8, 2);
        const BendingBeamErrors want = errors_of(base.displacement, factor * base.stress);
        expect_errors_near(got, want, 1e-9, "E = " + std::to_string(young));
    }
}

// EXAMPLE 208. The thermal strain depends on T - T_ref, not on T: if T_ref is changed the temperature T = T_ref + G y~
// of the beam changes too, and the difference stays G y~. With T_ref = 0, -500 and 10000 the errors are the same
// (compared with T_ref = 293.15). Whoever forgets T_ref in one of the two functions that use it (the thermal load and the
// stresses) sees a beam heated by hundreds of degrees.
TEST(ElasticityBeam2d, TheReferenceTemperatureDoesNotMatter) {
    const BendingBeamErrors base = steel_errors(2);
    EXPECT_GT(base.displacement, 1e-4);
    EXPECT_GT(base.stress, 1e6);

    for (double reference : {0.0, -500.0, 10000.0}) {
        BendingBeam beam = steel_beam();
        beam.T_ref = reference;
        const BendingBeamErrors got = bending_beam_errors_2d(beam, 8, 2);
        expect_errors_near(got, base, 1e-9, "T_ref = " + std::to_string(reference));
    }
}

// EXAMPLE 209. Other Poisson ratios, with the mesh k = 4 (16 x 4 cells). If nu changes, kappa =
// (1 + nu) alpha G, the matrix D and the thermal strain change: if nu is used differently in a single place (for
// example a 0.3 written in the code) the method no longer converges to the exact solution and the errors are much
// larger. Measured (and confirmed by the independent program):
//       nu       displacement      stress   (Pa)
//     -0.50      3.2753386e-04     3.2214892e+07     (auxetic material)
//      0.00      4.9988890e-04     3.6344696e+07
//      0.45      7.2274450e-04     6.9570010e+07
// With nu close to 1/2 (almost incompressible material) the linear triangles become stiffer (the "volumetric
// locking") and for the same mesh the error is larger.
TEST(ElasticityBeam2d, ErrorsForOtherPoissonRatiosMatchTheTable) {
    struct Row {
        double nu;
        double displacement;
        double stress;
    };
    const std::vector<Row> table = {{-0.5, 3.2753386e-04, 3.2214892e+07},
                                    {0.0, 4.9988890e-04, 3.6344696e+07},
                                    {0.45, 7.2274450e-04, 6.9570010e+07}};

    for (const Row& row : table) {
        BendingBeam beam = steel_beam();
        beam.nu = row.nu;
        const BendingBeamErrors errors = bending_beam_errors_2d(beam, 16, 4);
        EXPECT_NEAR(errors.displacement, row.displacement, 1e-6 * row.displacement) << "nu = " << row.nu;
        EXPECT_NEAR(errors.stress, row.stress, 1e-6 * row.stress) << "nu = " << row.nu;
    }
}

// EXAMPLE 210. If the beam shrinks (or grows) by a factor s, with the same mesh
// (same number of cells, 8 x 2) and the same gradient G (kelvin per meter), all the lengths are s times the previous ones;
// the temperature difference between top and bottom is s times, so the thermal strain is s
// times too. The solution is linear: the displacements (strain times length) are s^2 times, the stresses
// (E times strain) s times. The errors follow: displacement s^2, stress s. With s = 0.05
// (a beam of 20 cm by 5 cm), 3 and 20. Whoever writes 0.5 in place of H / 2 does not notice it with H = 1, but here they do.
TEST(ElasticityBeam2d, ScalingTheBeamScalesTheErrors) {
    const BendingBeamErrors base = steel_errors(2);
    EXPECT_GT(base.displacement, 1e-4);
    EXPECT_GT(base.stress, 1e6);

    for (double s : {0.05, 3.0, 20.0}) {
        BendingBeam beam = steel_beam();
        beam.length = 4.0 * s;
        beam.height = 1.0 * s;
        const BendingBeamErrors got = bending_beam_errors_2d(beam, 8, 2);
        const BendingBeamErrors want = errors_of(s * s * base.displacement, s * base.stress);
        expect_errors_near(got, want, 1e-9, "s = " + std::to_string(s));
    }
}

// EXAMPLE 211. Invalid data: std::invalid_argument for nx or ny equal to 0; for length or height zero, negative or
// NaN; for E zero, negative or NaN; for nu outside (-1, 1/2) (the ends excluded) or NaN. These are the checks of the
// functions that bending_beam_errors_2d calls (rectangle_mesh, plane_strain_matrix). A valid beam does not throw.
TEST(ElasticityBeam2d, BadInputsThrowInvalidArgument) {
    const double not_a_number = std::numeric_limits<double>::quiet_NaN();

    EXPECT_NO_THROW(bending_beam_errors_2d(steel_beam(), 4, 1));
    EXPECT_THROW(bending_beam_errors_2d(steel_beam(), 0, 1), std::invalid_argument);
    EXPECT_THROW(bending_beam_errors_2d(steel_beam(), 4, 0), std::invalid_argument);
    EXPECT_THROW(bending_beam_errors_2d(steel_beam(), 0, 0), std::invalid_argument);

    for (double bad : {0.0, -1.0, not_a_number}) {
        BendingBeam beam = steel_beam();
        beam.length = bad;
        EXPECT_THROW(bending_beam_errors_2d(beam, 4, 1), std::invalid_argument) << "length = " << bad;

        beam = steel_beam();
        beam.height = bad;
        EXPECT_THROW(bending_beam_errors_2d(beam, 4, 1), std::invalid_argument) << "height = " << bad;

        beam = steel_beam();
        beam.E = bad;
        EXPECT_THROW(bending_beam_errors_2d(beam, 4, 1), std::invalid_argument) << "E = " << bad;
    }
    for (double bad : {-1.0, -2.0, 0.5, 0.75, not_a_number}) {
        BendingBeam beam = steel_beam();
        beam.nu = bad;
        EXPECT_THROW(bending_beam_errors_2d(beam, 4, 1), std::invalid_argument) << "nu = " << bad;
    }
}

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "CSR_and_Vectors.h"
#include "Fem1D.h"
#include "Mesh2D.h"
#include "Verification2D.h"

// ---- Verification in 2D: the L2 error and the manufactured solution ---------------------------------------
//
// In 1D we measured whether the code converges as it should by comparing it with an exact solution: l2_error,
// observed_order, decaying_sine_error. Here we do the same for the 2D plate in steady state.
//
// THE L2 ERROR. The method gives one temperature per node, T_num; the exact solution, taken at the nodes, is another one,
// T_ref. Between the nodes both are piecewise linear. With e = T_num - T_ref, in a triangle
//
//     e(x, y) = e0 N0 + e1 N1 + e2 N2         (e_i = value of e at local node i)
//
// so the integral of e^2 over the triangle is the sum over i, j of e_i e_j * (integral of N_i N_j) = e^T M e, with M
// the mass matrix with rho_c = 1: A / 6 on the diagonal, A / 12 off it. Summing over the triangles (as
// in the assembly)
//
//     || e ||_L2 = sqrt( integral of e^2 dA ) = sqrt( e^T M e )        with M the mass matrix of the whole mesh.
//
// On a single triangle this becomes   integral of e^2 dA = A / 12 * ( e0^2 + e1^2 + e2^2 + (e0 + e1 + e2)^2 ).
//
// THE MANUFACTURED SOLUTION. On the square [0, 1] x [0, 1] we choose the temperature T(x, y) = sin(pi x) sin(pi y),
// which is 0 on the whole boundary. The second derivatives are T_xx = T_yy = -pi^2 T, so the source that produces it is
//
//     q(x, y) = -k (T_xx + T_yy) = 2 pi^2 k sin(pi x) sin(pi y)
//
// and T does not depend on k. The problem is solved with this source and with T = 0 on the boundary, and compared with the chosen
// T: if the code is right the error goes down as it should when the mesh is refined.

namespace {

// A "random" number in [0, 1): a small congruential generator, so the tests give the same numbers with
// any compiler. `state` is the generator state, it changes at every call.
double next_random(unsigned long long& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) / 9007199254740992.0;
}

// A mesh like rectangle_mesh but with the INTERIOR nodes moved at random, in x and in y, by at most 20% of the grid
// step: triangles of different shapes and areas, with obtuse angles. The boundary does not move: it is still the
// rectangle.
Mesh2D perturbed_rectangle(double lx, double ly, std::size_t nx, std::size_t ny, unsigned long long seed) {
    const Mesh2D base = rectangle_mesh(lx, ly, nx, ny);
    const std::size_t n = base.num_nodes();
    std::vector<bool> on_boundary(n, false);
    for (std::size_t d : boundary_nodes(base)) {
        on_boundary[d] = true;
    }
    const double hx = lx / static_cast<double>(nx);
    const double hy = ly / static_cast<double>(ny);
    unsigned long long state = seed;
    std::vector<double> xs(n);
    std::vector<double> ys(n);
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = base.x(i);
        ys[i] = base.y(i);
        if (!on_boundary[i]) {
            xs[i] += (2.0 * next_random(state) - 1.0) * 0.2 * hx;
            ys[i] += (2.0 * next_random(state) - 1.0) * 0.2 * hy;
        }
    }
    std::vector<Triangle> triangles;
    for (std::size_t e = 0; e < base.num_triangles(); ++e) {
        triangles.push_back(base.triangle(e));
    }
    return Mesh2D(xs, ys, triangles);
}

// The triangle (0, 0), (1, 0), (0, 1), of area 1/2.
Mesh2D right_triangle() {
    return Mesh2D({0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {{0, 1, 2}});
}

// The large triangle with a node inside: (0, 0), (3, 0), (0, 3) and node 3 = (1, 1). It is not a rectangle.
Mesh2D triangle_with_interior_node() {
    return Mesh2D({0.0, 3.0, 0.0, 1.0}, {0.0, 0.0, 3.0, 1.0}, {{0, 1, 3}, {1, 2, 3}, {2, 0, 3}});
}

// The irregular quadrilateral with a node inside: four triangles of different area.
Mesh2D irregular_quadrilateral() {
    return Mesh2D({0.0, 4.0, 3.0, -1.0, 1.0}, {0.0, 0.0, 3.0, 2.0, 0.7},
                  {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}});
}

// A mesh in which the nodes 0 = (5, 5) and 5 = (7, 7) belong to no triangle.
Mesh2D mesh_with_nodes_outside_every_triangle() {
    return Mesh2D({5.0, 0.0, 1.0, 1.0, 0.0, 7.0}, {5.0, 0.0, 0.0, 1.0, 1.0, 7.0}, {{1, 2, 3}, {1, 3, 4}});
}

// Meshes of all shapes: a rectangle, the triangle with a node inside, the irregular quadrilateral, two skewed
// meshes, the square of two triangles and the mesh with two nodes outside every triangle.
std::vector<Mesh2D> varied_meshes() {
    std::vector<Mesh2D> meshes;
    meshes.push_back(rectangle_mesh(3.0, 1.5, 4, 3));
    meshes.push_back(triangle_with_interior_node());
    meshes.push_back(irregular_quadrilateral());
    meshes.push_back(perturbed_rectangle(2.0, 1.0, 8, 6, 5));
    meshes.push_back(perturbed_rectangle(1.0, 1.0, 12, 12, 31));
    meshes.push_back(rectangle_mesh(1.0, 1.0, 1, 1));
    meshes.push_back(mesh_with_nodes_outside_every_triangle());
    return meshes;
}

}  // namespace

// EXAMPLE 88. The triangle (0, 0), (1, 0), (0, 1): A = 1/2, so M has 1/12 on the diagonal and 1/24 off it. For an
// error e = (e0, e1, e2) at the three nodes
//     || e ||^2 = 1/12 (e0^2 + e1^2 + e2^2) + 1/12 (e0 e1 + e1 e2 + e0 e2) = 1/24 ( sum e_i^2 + (sum e_i)^2 )
//   e = (1, 1, 1):    1/24 (3 + 9)  = 1/2      ->  || e || = sqrt(1/2)    (e constant: the integral is the area, 1/2)
//   e = (1, 0, 0):    1/24 (1 + 1)  = 1/12     ->  sqrt(1/12)             (e = N_0: the integral of N_0^2 is A / 6)
//   e = (1, -1, 0):   1/24 (2 + 0)  = 1/12     ->  sqrt(1/12)
//   e = (1, 2, 3):    1/24 (14 + 36) = 25/12   ->  5 / sqrt(12)
//   e = (2, 0, -1):   1/24 (5 + 1)  = 1/4      ->  1/2
// Only the differences count: T_num and T_ref are chosen different from e and from 0, and swapping them e changes sign but the
// norm does not.
TEST(Verification2D, L2ErrorOfTheRightTriangleMatchesHandComputation) {
    const Mesh2D m = right_triangle();

    EXPECT_NEAR(l2_error_2d(m, {4.0, 3.0, 7.0}, {3.0, 2.0, 6.0}), std::sqrt(0.5), 1e-14) << "e = (1, 1, 1)";
    EXPECT_NEAR(l2_error_2d(m, {3.0, 2.0, 6.0}, {4.0, 3.0, 7.0}), std::sqrt(0.5), 1e-14) << "e = (-1, -1, -1)";
    EXPECT_NEAR(l2_error_2d(m, {5.0, 1.0, 2.0}, {4.0, 1.0, 2.0}), std::sqrt(1.0 / 12.0), 1e-14) << "e = (1, 0, 0)";
    EXPECT_NEAR(l2_error_2d(m, {2.0, 0.0, 1.0}, {1.0, 1.0, 1.0}), std::sqrt(1.0 / 12.0), 1e-14) << "e = (1, -1, 0)";
    EXPECT_NEAR(l2_error_2d(m, {1.0, 2.0, 3.0}, {0.0, 0.0, 0.0}), 5.0 / std::sqrt(12.0), 1e-14) << "e = (1, 2, 3)";
    EXPECT_NEAR(l2_error_2d(m, {2.0, 0.0, -1.0}, {0.0, 0.0, 0.0}), 0.5, 1e-14) << "e = (2, 0, -1)";
    EXPECT_NEAR(l2_error_2d(m, {7.0, -1.5, 3.0}, {7.0, -1.5, 3.0}), 0.0, 1e-14) << "e = 0";
}

// EXAMPLE 89. If e is a linear function, e(x, y) = c0 + cx x + cy y, its values at the nodes describe it
// EXACTLY, even on triangles of any shape, and the norm is computed by hand on the whole plate [0, a] x [0, b]
// with the monomial formulas (integral of x^i y^j = a^(i+1) b^(j+1) / ((i + 1) (j + 1))):
//     integral of (c0 + cx x + cy y)^2 = c0^2 a b + cx^2 a^3 b / 3 + cy^2 a b^3 / 3
//                                         + c0 cx a^2 b + c0 cy a b^2 + cx cy a^2 b^2 / 2
// We take T_num = 3/2 - 2 x + 3/4 y and T_ref = 1/2 + x - y / 4, i.e. e = 1 - 3 x + y (c0 = 1, cx = -3, cy = 1).
// For example on the unit square: 1 + 9/3 + 1/3 - 3 + 1 - 3/2 = 5/6. Also on the meshes with the interior nodes
// moved: the boundary does not change, so the plate is always the same.
TEST(Verification2D, L2ErrorOfALinearFieldIsTheExactIntegral) {
    const double c0 = 1.0;
    const double cx = -3.0;
    const double cy = 1.0;

    struct Shape {
        double a;
        double b;
        std::size_t nx;
        std::size_t ny;
        unsigned long long seed;   // 0: regular rectangle
    };
    const std::vector<Shape> shapes = {
        {3.0, 1.5, 4, 3, 0}, {2.0, 1.0, 8, 6, 5}, {1.0, 1.0, 12, 12, 31}, {0.5, 4.0, 3, 9, 7}};
    for (const Shape& s : shapes) {
        SCOPED_TRACE("rectangle " + std::to_string(s.a) + " x " + std::to_string(s.b) + ", seed " +
                     std::to_string(s.seed));
        const Mesh2D m = s.seed == 0 ? rectangle_mesh(s.a, s.b, s.nx, s.ny)
                                     : perturbed_rectangle(s.a, s.b, s.nx, s.ny, s.seed);
        DVector t_num(m.num_nodes());
        DVector t_ref(m.num_nodes());
        for (std::size_t i = 0; i < m.num_nodes(); ++i) {
            t_num[i] = 1.5 - 2.0 * m.x(i) + 0.75 * m.y(i);
            t_ref[i] = 0.5 + m.x(i) - 0.25 * m.y(i);
        }
        const double exact = c0 * c0 * s.a * s.b + cx * cx * s.a * s.a * s.a * s.b / 3.0 +
                             cy * cy * s.a * s.b * s.b * s.b / 3.0 + c0 * cx * s.a * s.a * s.b +
                             c0 * cy * s.a * s.b * s.b + cx * cy * s.a * s.a * s.b * s.b / 2.0;
        EXPECT_NEAR(l2_error_2d(m, t_num, t_ref), std::sqrt(exact), 1e-12 * (1.0 + std::sqrt(exact)));
    }
}

// EXAMPLE 90. A second route to the same number, which assembles no matrix: the formula on a single
// triangle, A / 12 * ( e0^2 + e1^2 + e2^2 + (e0 + e1 + e2)^2 ), summed over the triangles. It is compared on seven meshes
// of different shapes (rectangles, obtuse triangles, and one with two nodes that belong to no triangle: their
// value must not count) with "random" but always identical vectors, with values between -4 and 6.
TEST(Verification2D, L2ErrorAgreesWithTheTriangleByTriangleFormula) {
    unsigned long long state = 12345;
    std::size_t number = 0;
    for (const Mesh2D& m : varied_meshes()) {
        SCOPED_TRACE("mesh number " + std::to_string(number++));
        DVector t_num(m.num_nodes());
        DVector t_ref(m.num_nodes());
        for (std::size_t i = 0; i < m.num_nodes(); ++i) {
            t_num[i] = 10.0 * next_random(state) - 4.0;
            t_ref[i] = 10.0 * next_random(state) - 4.0;
        }
        double sum = 0.0;
        for (std::size_t e = 0; e < m.num_triangles(); ++e) {
            const Triangle& t = m.triangle(e);
            double s1 = 0.0;   // e0 + e1 + e2
            double s2 = 0.0;   // e0^2 + e1^2 + e2^2
            for (std::size_t i = 0; i < 3; ++i) {
                const double d = t_num[t[i]] - t_ref[t[i]];
                s1 += d;
                s2 += d * d;
            }
            sum += m.area(e) / 12.0 * (s2 + s1 * s1);
        }
        EXPECT_NEAR(l2_error_2d(m, t_num, t_ref), std::sqrt(sum), 1e-12 * (1.0 + std::sqrt(sum)));
    }
}

// EXAMPLE 91. T_num and T_ref have one value per node: otherwise std::invalid_argument, even if only one of the two is
// wrong or if both are (same length, but not the number of nodes).
TEST(Verification2D, L2ErrorRejectsVectorsOfTheWrongSize) {
    const Mesh2D m = right_triangle();   // 3 nodes
    const DVector ok(3, 1.0);
    const DVector too_short(2, 1.0);
    const DVector too_long(4, 1.0);
    const DVector empty;

    EXPECT_THROW(l2_error_2d(m, too_short, ok), std::invalid_argument);
    EXPECT_THROW(l2_error_2d(m, ok, too_short), std::invalid_argument);
    EXPECT_THROW(l2_error_2d(m, too_long, ok), std::invalid_argument);
    EXPECT_THROW(l2_error_2d(m, ok, too_long), std::invalid_argument);
    EXPECT_THROW(l2_error_2d(m, too_short, too_short), std::invalid_argument);
    EXPECT_THROW(l2_error_2d(m, too_long, too_long), std::invalid_argument);
    EXPECT_THROW(l2_error_2d(m, empty, empty), std::invalid_argument);
    EXPECT_NO_THROW(l2_error_2d(m, ok, ok));
}

// EXAMPLE 92. n = 2: the 2 x 2 mesh has 9 nodes, 8 on the boundary (T = 0) and only one at the center,
// c = (1/2, 1/2). There is a single unknown, T_c, and it is computed by hand.
//   - The center belongs to 6 triangles, all isosceles right triangles with legs of length h = 1/2: area A = h^2 / 2 = 1/8.
//   - Row of the center of K: in a triangle K[i][i] = k (b_i^2 + c_i^2) / (4 A). At the right angle (the opposite side
//     is the hypotenuse: b^2 + c^2 = 2 h^2) it is k; at the other two angles (the opposite side is a leg: b^2 + c^2 = h^2)
//     it is k / 2. In two cells the center is the right angle of one triangle (k each), in the other two it is an
//     acute angle of both triangles (k/2 + k/2 = k each): K[c][c] = 4 k. The other nodes have T = 0, so the
//     equation is  4 k T_c = f_c.
//   - The load: f_c = sum over the 6 triangles of A / 6 * (q(M) + q(M')), with M and M' the midpoints of the two sides that
//     leave the center (EXAMPLE 80). Each of the 6 sides that leave the center belongs to two triangles, so
//         f_c = 2 * 1/48 * (sum of q at the 6 midpoints).
//     The sides go to (1/2, 0), (1, 1/2), (1/2, 1), (0, 1/2): midpoints (1/2, 1/4), (3/4, 1/2), (1/2, 3/4), (1/4, 1/2),
//     where sin(pi x) sin(pi y) is sqrt(2) / 2; and to (1, 1) and (0, 0): midpoints (3/4, 3/4) and (1/4, 1/4), where it is 1/2.
//     Sum: 4 sqrt(2) / 2 + 1/2 + 1/2 = 2 sqrt(2) + 1, so f_c = 2 pi^2 k (2 sqrt(2) + 1) / 24.
//   - T_c = f_c / (4 k) = pi^2 (1 + 2 sqrt(2)) / 48 = 0.78719..., whereas the exact solution at the center is 1. k does not
//     appear: the result is the same for every k.
//   - The error is e_c = T_c - 1 at the center and (almost) 0 at the other nodes; M[c][c] = 6 * A / 6 = 1/8, so
//         || e ||_L2 = |e_c| / sqrt(8) = (1 - T_c) / (2 sqrt(2)) = 0.0752...
// With n = 1 the four nodes are all on the boundary: T = 0 everywhere, like the exact solution, and the error is 0.
TEST(Verification2D, SingleUnknownCaseMatchesHandComputation) {
    const double pi = std::acos(-1.0);
    const double sqrt2 = std::sqrt(2.0);
    const double center = pi * pi * (1.0 + 2.0 * sqrt2) / 48.0;
    const double want = (1.0 - center) / (2.0 * sqrt2);

    for (double k : {1.0, 3.5, 0.25}) {
        EXPECT_NEAR(manufactured_sine_error_2d(2, k), want, 1e-12) << "k = " << k;
    }
    EXPECT_NEAR(manufactured_sine_error_2d(1, 1.0), 0.0, 1e-15);
}

// EXAMPLE 93. The numbers of the README table, for the manufactured solution with k = 1, measured with this method:
//     n = 4:  2.342328e-02      n = 8:  6.338199e-03      n = 16:  1.620920e-03
// The test pins them: if the method changes (the load rule, the solver, the norm) the numbers change, and the test
// says that the README table must be redone. With k different from 1 the error is the same (T does not depend on k).
TEST(Verification2D, ErrorsMatchTheTableInTheReadme) {
    const double want_4 = 2.342328e-02;
    const double want_8 = 6.338199e-03;
    const double want_16 = 1.620920e-03;

    EXPECT_NEAR(manufactured_sine_error_2d(4, 1.0), want_4, 1e-6 * want_4);
    EXPECT_NEAR(manufactured_sine_error_2d(8, 1.0), want_8, 1e-6 * want_8);
    EXPECT_NEAR(manufactured_sine_error_2d(16, 1.0), want_16, 1e-6 * want_16);

    EXPECT_NEAR(manufactured_sine_error_2d(8, 7.5), want_8, 1e-6 * want_8) << "k = 7.5";
    EXPECT_NEAR(manufactured_sine_error_2d(8, 0.02), want_8, 1e-6 * want_8) << "k = 0.02";
}

// EXAMPLE 94. With linear elements the theory says that the L2 error goes like h^2: when n doubles (h halves)
// the error is divided by 4, i.e. the observed order (observed_order, the 1D one) is 2. Measured: 1.97 from 8 to 16,
// 1.99 from 16 to 32, 2.00 from 32 to 64. From 4 to 8 it is 1.89: with such a coarse mesh we are not yet in the true
// asymptotic regime, which is why we start from 8.
TEST(Verification2D, ErrorGoesDownWithObservedOrderTwo) {
    double previous = manufactured_sine_error_2d(8, 1.0);
    for (std::size_t n : {16, 32, 64}) {
        const double current = manufactured_sine_error_2d(n, 1.0);
        EXPECT_LT(current, previous) << "n = " << n / 2 << " -> " << n;
        EXPECT_NEAR(observed_order(previous, current, 2.0), 2.0, 0.05) << "n = " << n / 2 << " -> " << n;
        previous = current;
    }
}

// EXAMPLE 95. Invalid data: no cells (n = 0), conductivity zero, negative or NaN: std::invalid_argument.
TEST(Verification2D, ManufacturedErrorRejectsInvalidArguments) {
    const double not_a_number = std::numeric_limits<double>::quiet_NaN();

    EXPECT_THROW(manufactured_sine_error_2d(0, 1.0), std::invalid_argument);
    EXPECT_THROW(manufactured_sine_error_2d(4, 0.0), std::invalid_argument);
    EXPECT_THROW(manufactured_sine_error_2d(4, -1.0), std::invalid_argument);
    EXPECT_THROW(manufactured_sine_error_2d(4, not_a_number), std::invalid_argument);
    EXPECT_NO_THROW(manufactured_sine_error_2d(4, 1.0));
}

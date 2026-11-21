#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "Assembly2D.h"
#include "CSR_and_Vectors.h"
#include "Fem2D.h"
#include "Mesh2D.h"

// ---- The load of a source that changes from point to point ------------------------------------------------
//
// So far the heat source was a number q, the same at every point of the plate. Now it is a function q(x, y):
// the power per unit volume at the point (x, y). The load of triangle e is still
//
//     f[i] = integral over the triangle of  q(x, y) N_i  dA        (i = 0, 1, 2, in the order of mesh.triangle(e))
//
// but now the integral is no longer q A / 3: it depends on how q changes inside the triangle, and in general it cannot
// be computed by hand. A formula is used: q is computed at a few points and the values are added with weights. Here the
// EDGE MIDPOINT RULE: for any function g,
//
//     integral of g dA  =  A / 3 * ( g(M01) + g(M12) + g(M20) )      (approximately)
//
// where M01 is the midpoint of the side between local nodes 0 and 1, M12 that between 1 and 2, M20 that between 2 and 0. The
// formula is EXACT for every polynomial of degree 0, 1 and 2. In the triangle (0, 0), (1, 0), (0, 1), for example, with
// g = x^2 the exact integral is 1/12 and the formula gives 1/6 * (1/4 + 1/4 + 0) = 1/12.
//
// To get f[i] we take g = q N_i. Along a side that starts from node i, N_i goes down from 1 to 0 linearly,
// so at the midpoint it is 1/2; along the side opposite i it is 0 everywhere. Therefore (A / 3 times 1/2 = A / 6)
//
//     f[0] = A / 6 * (q(M01) + q(M20))
//     f[1] = A / 6 * (q(M01) + q(M12))
//     f[2] = A / 6 * (q(M12) + q(M20))
//
// i.e. each node receives A / 6 times the sum of q at the midpoints of the TWO sides that touch it.
//   - If q is constant it is q A / 3 for every node: it is the previous function (element_load with q a number).
//   - If q is linear, q N_i has degree 2 and the three numbers are the exact integrals.
//   - If q has degree 2, q N_i has degree 3 and the three numbers are no longer exact one by one. But N_0 + N_1 + N_2 = 1,
//     so their SUM is the integral of q, which has degree 2: that one is still exact.
//
// assemble_load_2d(mesh, q) adds up, at the nodes, the loads of the triangles, like the previous function.

namespace {

using Source = std::function<double(double, double)>;

// Compare two vectors of three numbers, element by element.
void expect_vector3_near(const Vector3& got, const Vector3& want, double tol, const std::string& what = "") {
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(got[i], want[i], tol) << what << " element " << i;
    }
}

// Compare two vectors element by element. The tolerance is relative: rel_tol * (1 + |expected value|).
void expect_vector_near(const std::vector<double>& got, const std::vector<double>& want, double rel_tol,
                        const std::string& what = "") {
    ASSERT_EQ(got.size(), want.size()) << what << ": size";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], rel_tol * (1.0 + std::fabs(want[i]))) << what << ": element " << i;
    }
}

// The sum of all the elements.
double total_of(const std::vector<double>& v) {
    double s = 0.0;
    for (double value : v) {
        s += value;
    }
    return s;
}

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

// EXAMPLE 80. The triangle (0, 0), (1, 0), (0, 1): A = 1/2, so A / 6 = 1/12. The midpoints of the sides are
// M01 = (1/2, 0), M12 = (1/2, 1/2) and M20 = (0, 1/2). With the linear source q = x:
//     q(M01) = 1/2,  q(M12) = 1/2,  q(M20) = 0
//     f[0] = 1/12 * (1/2 + 0) = 1/24      f[1] = 1/12 * (1/2 + 1/2) = 1/12      f[2] = 1/12 * (1/2 + 0) = 1/24
// These are the exact integrals: the integral of x N_1 = x^2 is 1/12, those of x N_0 and of x N_2 = x y are 1/24,
// and the sum 1/6 is the integral of x, i.e. A times the abscissa of the centroid (1/2 * 1/3).
// With q = y nodes 1 and 2 are swapped: (1/24, 1/24, 1/12). With q = 3 + x - y the two things are added to 3 times the
// constant load (1/6 per node): f[0] = 1/2 + 1/24 - 1/24 = 1/2, f[1] = 1/2 + 1/12 - 1/24 = 13/24 and
// f[2] = 1/2 + 1/24 - 1/12 = 11/24.
TEST(Load2d, LinearSourceOnTheRightTriangleIsIntegratedExactly) {
    const Mesh2D m = right_triangle();
    const Source qx = [](double x, double) { return x; };
    const Source qy = [](double, double y) { return y; };
    const Source mixed = [](double x, double y) { return 3.0 + x - y; };

    expect_vector3_near(element_load(m, 0, qx), {{1.0 / 24.0, 1.0 / 12.0, 1.0 / 24.0}}, 1e-15, "q = x:");
    expect_vector3_near(element_load(m, 0, qy), {{1.0 / 24.0, 1.0 / 24.0, 1.0 / 12.0}}, 1e-15, "q = y:");
    expect_vector3_near(element_load(m, 0, mixed), {{1.0 / 2.0, 13.0 / 24.0, 11.0 / 24.0}}, 1e-15, "q = 3 + x - y:");
}

// EXAMPLE 81. The same triangle, with sources of degree 2: the midpoint rule is not exact for every node,
// but it is for the sum.
//     q = x^2:  q(M01) = 1/4, q(M12) = 1/4, q(M20) = 0    ->  f = (1/48, 1/24, 1/48)
//     q = x y:  q(M01) = 0,   q(M12) = 1/4, q(M20) = 0    ->  f = (0, 1/48, 1/48)
// The exact integrals of x^2 N_i would be (1/60, 1/20, 1/60): they are different, and that is normal. The sums, instead, are
// the integral of q: 1/12 for x^2 and 1/24 for x y. In the second case node 0, which is the origin, receives 0: x y is 0
// at the midpoints of its two sides.
// This test pins the RULE: a more accurate formula would give (1/60, 1/20, 1/60) and would be rejected. This is intended:
// the next step measures errors that depend on this rule, and with the rule written here anyone can redo
// the numbers.
TEST(Load2d, QuadraticSourceFollowsTheMidpointRuleAndKeepsTheTotal) {
    const Mesh2D m = right_triangle();
    const Source qxx = [](double x, double) { return x * x; };
    const Source qxy = [](double x, double y) { return x * y; };

    const Vector3 a = element_load(m, 0, qxx);
    expect_vector3_near(a, {{1.0 / 48.0, 1.0 / 24.0, 1.0 / 48.0}}, 1e-15, "q = x^2:");
    EXPECT_NEAR(a[0] + a[1] + a[2], 1.0 / 12.0, 1e-15) << "q = x^2: sum";

    const Vector3 b = element_load(m, 0, qxy);
    expect_vector3_near(b, {{0.0, 1.0 / 48.0, 1.0 / 48.0}}, 1e-15, "q = x y:");
    EXPECT_NEAR(b[0] + b[1] + b[2], 1.0 / 24.0, 1e-15) << "q = x y: sum";
}

// EXAMPLE 82. Arbitrary triangles, with the linear source q = 2x + 3y - 1. Four nodes: 0 = (1, 1), 1 = (5, 1),
// 2 = (2, 4), 3 = (6, 5); T0 = (0, 1, 2) and T1 = (1, 3, 2).
//   T0: A = 1/2 * (4 * 3 - 1 * 0) = 6, so A / 6 = 1. The midpoints are M01 = (3, 1), M12 = (7/2, 5/2) and
//       M20 = (3/2, 5/2), where q is 8, 27/2 and 19/2:
//           f = (8 + 19/2, 8 + 27/2, 27/2 + 19/2) = (35/2, 43/2, 23)
//   T1: the local nodes are 1 = (5, 1), 3 = (6, 5), 2 = (2, 4). A = 1/2 * (1 * 3 - (-3) * 4) = 15/2, so
//       A / 6 = 5/4. M01 = (11/2, 3), M12 = (4, 9/2), M20 = (7/2, 5/2), where q is 19, 41/2 and 27/2:
//           f = 5/4 * (19 + 27/2, 19 + 41/2, 41/2 + 27/2) = (325/8, 395/8, 85/2)
// Here we need the coordinates of the REAL nodes of the triangle (not those of the reference triangle), its area and
// the local order of the nodes: T1 lists nodes 1, 3, 2, not 0, 1, 2. With q = 2x + 3y - 1, which is linear, these are
// the exact integrals: for example the sum of T0, 62, is A times the value of q at the centroid (8/3, 2): 6 * 31/3.
TEST(Load2d, GeneralTrianglesUseTheirOwnNodesAndArea) {
    const Mesh2D m({1.0, 5.0, 2.0, 6.0}, {1.0, 1.0, 4.0, 5.0}, {{0, 1, 2}, {1, 3, 2}});
    const Source q = [](double x, double y) { return 2.0 * x + 3.0 * y - 1.0; };

    expect_vector3_near(element_load(m, 0, q), {{35.0 / 2.0, 43.0 / 2.0, 23.0}}, 1e-13, "triangle 0:");
    expect_vector3_near(element_load(m, 1, q), {{325.0 / 8.0, 395.0 / 8.0, 85.0 / 2.0}}, 1e-13, "triangle 1:");
}

// EXAMPLE 83. The usual two square cells, rectangle_mesh(2, 1, 2, 1): nodes 0, 1, 2 are at the bottom (x = 0, 1,
// 2) and 3, 4, 5 at the top. The triangles are T0 = (0, 1, 4), T1 = (0, 4, 3), T2 = (1, 2, 5) and T3 = (1, 5, 4), all of
// area 1/2, so A / 6 = 1/12. Source q = x (linear). In a triangle each node receives 1/12 times the sum of the
// abscissas of the midpoints of its two sides. The abscissas of the midpoints of sides 0-1, 1-2, 2-0 of each triangle:
//     T0: 1/2, 1, 1/2        T1: 1/2, 1/2, 0        T2: 3/2, 2, 3/2        T3: 3/2, 3/2, 1
// So, in twelfths (sum of the two abscissas):
//     node 0:  T0 (1/2 + 1/2) = 1,   T1 (1/2 + 0) = 1/2                             total  3/2  ->  1/8
//     node 1:  T0 (1/2 + 1) = 3/2,   T2 (3/2 + 3/2) = 3,   T3 (3/2 + 1) = 5/2       total  7    ->  7/12
//     node 2:  T2 (3/2 + 2) = 7/2                                                    total  7/2  ->  7/24
//     node 3:  T1 (1/2 + 0) = 1/2                                                    total  1/2  ->  1/24
//     node 4:  T0 (1 + 1/2) = 3/2,   T1 (1/2 + 1/2) = 1,   T3 (3/2 + 1) = 5/2       total  5    ->  5/12
//     node 5:  T2 (2 + 3/2) = 7/2,   T3 (3/2 + 3/2) = 3                             total  13/2 ->  13/24
// The total load is 2, the integral of x over the 2 x 1 rectangle: the area 2 times the abscissa of the centroid, 1.
TEST(Load2d, LoadOfTwoSquareCellsWithLinearSourceMatchesHandComputation) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const Source q = [](double x, double) { return x; };

    expect_vector_near(assemble_load_2d(m, q),
                       {1.0 / 8.0, 7.0 / 12.0, 7.0 / 24.0, 1.0 / 24.0, 5.0 / 12.0, 13.0 / 24.0}, 1e-14, "load");
}

// EXAMPLE 84. A constant source is a special case: the new function with q(x, y) = c must give the same
// result as the previous function with the number c, on every mesh and for every triangle. Also with negative c (a
// heat sink) and with c = 0.
TEST(Load2d, ConstantSourceGivesTheUniformLoad) {
    std::size_t number = 0;
    for (const Mesh2D& m : varied_meshes()) {
        SCOPED_TRACE("mesh number " + std::to_string(number++));
        for (double c : {7.0, -2.5, 0.0}) {
            const Source constant = [c](double, double) { return c; };
            expect_vector_near(assemble_load_2d(m, constant), assemble_load_2d(m, c), 1e-13,
                               "c = " + std::to_string(c) + ", whole mesh");
            for (std::size_t e = 0; e < m.num_triangles(); ++e) {
                const double tol = 1e-13 * (1.0 + std::fabs(c) * m.area(e));
                expect_vector3_near(element_load(m, e, constant), element_load(m, e, c), tol,
                                    "c = " + std::to_string(c) + ", triangle " + std::to_string(e) + ":");
            }
        }
    }
}

// EXAMPLE 85. If q is linear, the computation can be done along another route, which does not use the midpoints. Let q_i be the
// value of q at node i. On each triangle the linear function q is q_0 N_0 + q_1 N_1 + q_2 N_2, so
//     f[i] = integral of q N_i = sum over the nodes j of the triangle of  q_j * (integral of N_i N_j)
//          = sum over the nodes j of  M[i][j] q_j
// with M the usual mass matrix (rho_c = 1). Summing the triangles: the load of the whole mesh is M q_nodes, the
// product of the mass matrix by the vector of the values of q at the nodes. The test checks it on seven meshes of
// different shapes (rectangles, obtuse triangles, two nodes outside every triangle, which must have load 0) with
// q = 3 - 2x + y / 2. The load has one number per node, also for the nodes that belong to no triangle.
TEST(Load2d, LinearSourceGivesTheMassMatrixTimesTheNodalValues) {
    const Source q = [](double x, double y) { return 3.0 - 2.0 * x + 0.5 * y; };

    std::size_t number = 0;
    for (const Mesh2D& m : varied_meshes()) {
        SCOPED_TRACE("mesh number " + std::to_string(number++));
        std::vector<double> nodal(m.num_nodes());
        for (std::size_t i = 0; i < m.num_nodes(); ++i) {
            nodal[i] = q(m.x(i), m.y(i));
        }
        const std::vector<double> want = assemble_mass_2d(m, 1.0) * nodal;
        expect_vector_near(assemble_load_2d(m, q), want, 1e-12, "load");
    }
}

// EXAMPLE 86. The total heat: the sum of the loads of all the nodes is the integral of q over the whole plate. For
// each triangle the sum of the three numbers is the exact integral if q has degree <= 2 (EXAMPLE 81), so it is also
// over the whole mesh, for any shape of the triangles. We take
//     q = 3/2 - 2 x + 3/4 y + 3 x^2 - 4 x y + 5/2 y^2         (with positive and negative numbers)
// and the exact integral, computed with the monomial formulas:
//   - on the rectangle [0, a] x [0, b]:  integral of x^i y^j = a^(i+1) b^(j+1) / ((i + 1) (j + 1));
//   - on the triangle with vertices (0, 0), (3, 0), (0, 3): integral of x^i y^j = 3^(i+j+2) i! j! / (i + j + 2)!, i.e.
//     9/2 for 1, 9/2 for x and for y (A times the centroid (1, 1)), 27/4 for x^2 and y^2, 27/8 for x y.
// On the rectangles also with the interior nodes moved: the boundary does not change, so the plate is always the same.
TEST(Load2d, QuadraticSourceAddsUpToTheExactTotalHeat) {
    const double c0 = 1.5;
    const double cx = -2.0;
    const double cy = 0.75;
    const double cxx = 3.0;
    const double cxy = -4.0;
    const double cyy = 2.5;
    const Source q = [=](double x, double y) {
        return c0 + cx * x + cy * y + cxx * x * x + cxy * x * y + cyy * y * y;
    };

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
        const double exact = c0 * s.a * s.b + cx * s.a * s.a * s.b / 2.0 + cy * s.a * s.b * s.b / 2.0 +
                             cxx * s.a * s.a * s.a * s.b / 3.0 + cxy * s.a * s.a * s.b * s.b / 4.0 +
                             cyy * s.a * s.b * s.b * s.b / 3.0;
        EXPECT_NEAR(total_of(assemble_load_2d(m, q)), exact, 1e-12 * (1.0 + std::fabs(exact)));
    }

    const double exact_triangle = c0 * 4.5 + cx * 4.5 + cy * 4.5 + cxx * 6.75 + cxy * 3.375 + cyy * 6.75;
    EXPECT_NEAR(total_of(assemble_load_2d(triangle_with_interior_node(), q)), exact_triangle,
                1e-12 * (1.0 + std::fabs(exact_triangle)))
        << "triangle with a node inside";
}

// EXAMPLE 87. A triangle that does not exist: std::out_of_range, as for the previous function.
TEST(Load2d, ElementLoadOfAnUnknownTriangleThrows) {
    const Mesh2D m = right_triangle();   // has only triangle 0
    const Source q = [](double x, double y) { return x + y; };

    EXPECT_NO_THROW(element_load(m, 0, q));
    EXPECT_THROW(element_load(m, 1, q), std::out_of_range);
    EXPECT_THROW(element_load(m, 1000, q), std::out_of_range);
}

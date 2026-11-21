#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "Fem2D.h"
#include "Mesh2D.h"

// ---- Stiffness matrix of a triangle: element_stiffness ------------------------------------------
//
// element_stiffness(mesh, e, k) is the 3 x 3 matrix of triangle e, for a material of
// conductivity k (plate of thickness 1). K[i][j], with i and j = 0, 1, 2 the three nodes of the triangle
// in the order of mesh.triangle(e), is
//
//     K[i][j] = integral over the triangle of  k grad(N_i) . grad(N_j)  dA
//
// where N_i is the linear function that is 1 at node i and 0 at the other two. A linear function has
// a constant gradient, so the integral is simply  k * (grad N_i . grad N_j) * A.
//
// The gradient: N_i is 0 on the whole side opposite node i, so grad(N_i) is perpendicular to
// that side, and equals 1 / (height from i). In coordinates, with p = (i+1) % 3 and q = (i+2) % 3 (the two nodes
// that follow i, cyclically):
//
//     b_i = y_p - y_q        c_i = x_q - x_p        grad(N_i) = (b_i, c_i) / (2 A)
//     K[i][j] = k / (4 A) * (b_i b_j + c_i c_j)

namespace {

// The vertices of a triangle, counter-clockwise.
struct Tri {
    double x0, y0, x1, y1, x2, y2;
};

// A mesh with that single triangle (nodes 0, 1, 2 in this order).
Mesh2D mesh_of(const Tri& t) {
    return Mesh2D({t.x0, t.x1, t.x2}, {t.y0, t.y1, t.y2}, {{0, 1, 2}});
}

// The triangle t scaled by s, rotated by theta (radians) about the origin and shifted by (dx, dy).
Tri moved(const Tri& t, double s, double theta, double dx, double dy) {
    const double c = std::cos(theta);
    const double n = std::sin(theta);
    Tri r;
    r.x0 = s * (c * t.x0 - n * t.y0) + dx;
    r.y0 = s * (n * t.x0 + c * t.y0) + dy;
    r.x1 = s * (c * t.x1 - n * t.y1) + dx;
    r.y1 = s * (n * t.x1 + c * t.y1) + dy;
    r.x2 = s * (c * t.x2 - n * t.y2) + dx;
    r.y2 = s * (n * t.x2 + c * t.y2) + dy;
    return r;
}

// Six counter-clockwise triangles, of very different shapes and sizes.
std::vector<Tri> sample_triangles() {
    return {{0.0, 0.0, 1.0, 0.0, 0.0, 1.0},         // right angle (EXAMPLE 11)
            {0.0, 0.0, 4.0, 0.0, 1.0, 2.0},         // scalene (EXAMPLE 12)
            {1.0, 1.0, 5.0, 2.0, 2.0, 4.0},         // scalene, far from the origin
            {-1.5, 2.0, 0.5, -1.0, 3.0, 1.5},       // with negative coordinates
            {0.0, 0.0, 10.0, 0.0, 5.0, 0.1},        // very flat
            {0.0, 0.0, 0.004, 0.0, 0.001, 0.002}};  // small (millimeters)
}

// The largest of the elements of K in absolute value (to set the tolerances).
double largest(const Matrix3& K) {
    double m = 0.0;
    for (const auto& row : K) {
        for (double v : row) {
            m = std::max(m, std::fabs(v));
        }
    }
    return m;
}

// Compares two matrices element by element.
void expect_matrix_near(const Matrix3& got, const Matrix3& want, double tol, const std::string& what = "") {
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(got[i][j], want[i][j], tol) << what << " element (" << i << ", " << j << ")";
        }
    }
}

// The matrix of EXAMPLE 12, computed by hand.
Matrix3 scalene_stiffness() {
    return {{{13.0 / 16, -1.0 / 16, -12.0 / 16},
             {-1.0 / 16, 5.0 / 16, -4.0 / 16},
             {-12.0 / 16, -4.0 / 16, 16.0 / 16}}};
}

}  // namespace

// EXAMPLE 11. The triangle with nodes 0 = (0, 0), 1 = (1, 0), 2 = (0, 1), by hand. Here the N_i can be seen
// by eye: N_1 = x, N_2 = y and N_0 = 1 - x - y (each is 1 at its own node and 0 at the other two).
// So, with area A = 1/2,
//     grad N_0 = (-1, -1)     grad N_1 = (1, 0)     grad N_2 = (0, 1)
// and K[i][j] = k A (grad N_i . grad N_j):
//     K[0][0] = k/2 * 2 = k      K[0][1] = k/2 * (-1) = -k/2      K[0][2] = -k/2
//     K[1][1] = k/2 * 1 = k/2    K[1][2] = k/2 * 0 = 0            K[2][2] = k/2
// (the other three by symmetry). With the coefficient formulas one finds the same: b = (-1, 1, 0) and
// c = (-1, 0, 1), so grad N_i = (b_i, c_i) / (2 A) = (b_i, c_i).
TEST(Fem2d, StiffnessOfTheRightTriangleMatchesHandComputation) {
    const Mesh2D m = mesh_of({0.0, 0.0, 1.0, 0.0, 0.0, 1.0});

    const Matrix3 k1 = {{{1.0, -0.5, -0.5}, {-0.5, 0.5, 0.0}, {-0.5, 0.0, 0.5}}};
    expect_matrix_near(element_stiffness(m, 0, 1.0), k1, 1e-14, "k = 1:");

    const Matrix3 k3 = {{{3.0, -1.5, -1.5}, {-1.5, 1.5, 0.0}, {-1.5, 0.0, 1.5}}};
    expect_matrix_near(element_stiffness(m, 0, 3.0), k3, 1e-14, "k = 3:");
}

// EXAMPLE 12. Any triangle: 0 = (0, 0), 1 = (4, 0), 2 = (1, 2). Area A = 1/2 [4 * 2 - 1 * 0] = 4.
// For each node i, p and q are the two nodes that follow (cyclically):
//     node 0:  b = y_1 - y_2 = 0 - 2 = -2     c = x_2 - x_1 = 1 - 4 = -3
//     node 1:  b = y_2 - y_0 = 2 - 0 =  2     c = x_0 - x_2 = 0 - 1 = -1
//     node 2:  b = y_0 - y_1 = 0 - 0 =  0     c = x_1 - x_0 = 4 - 0 =  4
// (the b add up to 0 and so do the c: a free check). With 4 A = 16, K[i][j] = (b_i b_j + c_i c_j) / 16:
//     K[0][0] = (4 + 9) / 16 = 13/16     K[0][1] = (-4 + 3) / 16 = -1/16     K[0][2] = (0 - 12) / 16 = -12/16
//     K[1][1] = (4 + 1) / 16 = 5/16      K[1][2] = (0 - 4) / 16 = -4/16
//     K[2][2] = (0 + 16) / 16 = 1
// Every row adds up to zero: 13 - 1 - 12 = 0, -1 + 5 - 4 = 0, -12 - 4 + 16 = 0.
TEST(Fem2d, StiffnessOfAScaleneTriangleMatchesHandComputation) {
    const Mesh2D m = mesh_of({0.0, 0.0, 4.0, 0.0, 1.0, 2.0});
    expect_matrix_near(element_stiffness(m, 0, 1.0), scalene_stiffness(), 1e-14);
}

// EXAMPLE 13. Two different triangles that share a side: it serves to check that the nodes and
// the area of the requested triangle are used (and not those of the first). Nodes: 0 = (0, 0), 1 = (4, 0), 2 = (1, 2),
// 3 = (6, 4). Triangles: T0 = (0, 1, 2), the one of EXAMPLE 12 (A = 4), and T1 = (1, 3, 2) (A = 8).
// For T1 the local nodes 0, 1, 2 are the mesh nodes 1, 3, 2, i.e. (4, 0), (6, 4), (1, 2):
//     node 0:  b = 4 - 2 =  2     c = 1 - 6 = -5
//     node 1:  b = 2 - 0 =  2     c = 4 - 1 =  3
//     node 2:  b = 0 - 4 = -4     c = 6 - 4 =  2
// With 4 A = 32:
//     K[0][0] = (4 + 25) / 32 = 29/32     K[0][1] = (4 - 15) / 32 = -11/32     K[0][2] = (-8 - 10) / 32 = -18/32
//     K[1][1] = (4 + 9) / 32 = 13/32      K[1][2] = (-8 + 6) / 32 = -2/32
//     K[2][2] = (16 + 4) / 32 = 20/32
TEST(Fem2d, StiffnessUsesTheNodesAndTheAreaOfTheRequestedTriangle) {
    const Mesh2D m({0.0, 4.0, 1.0, 6.0}, {0.0, 0.0, 2.0, 4.0}, {{0, 1, 2}, {1, 3, 2}});
    EXPECT_NEAR(m.area(0), 4.0, 1e-14);   // the mesh is the right one
    EXPECT_NEAR(m.area(1), 8.0, 1e-14);

    expect_matrix_near(element_stiffness(m, 0, 1.0), scalene_stiffness(), 1e-14, "triangle 0:");

    const Matrix3 want1 = {{{29.0 / 32, -11.0 / 32, -18.0 / 32},
                            {-11.0 / 32, 13.0 / 32, -2.0 / 32},
                            {-18.0 / 32, -2.0 / 32, 20.0 / 32}}};
    expect_matrix_near(element_stiffness(m, 1, 1.0), want1, 1e-14, "triangle 1:");
}

// EXAMPLE 14. A cell 2 wide and 1 high, divided into two triangles by rectangle_mesh(2, 1, 1, 1): nodes
// 0 = (0, 0), 1 = (2, 0), 2 = (0, 1), 3 = (2, 1); T0 = (0, 1, 3) and T1 = (0, 3, 2), both of area 1.
// Same computation as before, with 4 A = 4:
//     T0:  b = (-1, 1, 0)  c = (0, -2, 2)         T1:  b = (0, 1, -1)  c = (-2, 0, 2)
//
//          T0:   1/4  -1/4    0                        T1:    1     0    -1
//               -1/4   5/4   -1                               0    1/4  -1/4
//                 0    -1     1                              -1   -1/4   5/4
//
// The coupling between two nodes is strong along the short side (-1) and weak along the long one (-1/4).
TEST(Fem2d, StiffnessOfTheTwoTrianglesOfAWideCell) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 1, 1);

    const Matrix3 t0 = {{{0.25, -0.25, 0.0}, {-0.25, 1.25, -1.0}, {0.0, -1.0, 1.0}}};
    const Matrix3 t1 = {{{1.0, 0.0, -1.0}, {0.0, 0.25, -0.25}, {-1.0, -0.25, 1.25}}};
    expect_matrix_near(element_stiffness(m, 0, 1.0), t0, 1e-14, "T0:");
    expect_matrix_near(element_stiffness(m, 1, 1.0), t1, 1e-14, "T1:");
}

// EXAMPLE 15. Two things that hold for ANY triangle, and have a physical reason.
//   - K is symmetric: K[i][j] = K[j][i] (the coupling between two nodes is the same in both directions).
//   - Every row adds up to zero: if the temperature is the same at all the nodes no heat flows. In formulas:
//     the three N_i add up to 1 everywhere, so their three gradients add up to 0.
TEST(Fem2d, StiffnessIsSymmetricAndEachRowSumsToZero) {
    std::size_t n = 0;
    for (const Tri& t : sample_triangles()) {
        SCOPED_TRACE("triangle number " + std::to_string(n++));
        const Matrix3 K = element_stiffness(mesh_of(t), 0, 2.5);
        const double tol = 1e-12 * largest(K);

        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                EXPECT_NEAR(K[i][j], K[j][i], tol) << "element (" << i << ", " << j << ")";
            }
            EXPECT_NEAR(K[i][0] + K[i][1] + K[i][2], 0.0, tol) << "row " << i;
        }
    }
}

// EXAMPLE 16. The energy of a LINEAR temperature field T(x, y) = t0 + gx x + gy y, whose gradient
// is constant (gx, gy), is the integral of k |grad T|^2 over the triangle, i.e. k A (gx^2 + gy^2). With the
// temperatures T_i of the three nodes, the matrix computes it as  T^T K T = sum of T_i K[i][j] T_j.
// It is a check of the whole of K by another route, without using the formulas of the coefficients b and c.
TEST(Fem2d, StiffnessGivesTheEnergyOfALinearField) {
    const double k = 2.5;
    // t0, gx, gy
    const double fields[][3] = {{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {2.0, 1.0, -3.0}, {-1.0, 0.5, 0.25}};

    std::size_t n = 0;
    for (const Tri& t : sample_triangles()) {
        SCOPED_TRACE("triangle number " + std::to_string(n++));
        const Mesh2D m = mesh_of(t);
        const Matrix3 K = element_stiffness(m, 0, k);

        for (const auto& f : fields) {
            const double T[3] = {f[0] + f[1] * t.x0 + f[2] * t.y0, f[0] + f[1] * t.x1 + f[2] * t.y1,
                                 f[0] + f[1] * t.x2 + f[2] * t.y2};
            double energy = 0.0;
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    energy += T[i] * K[i][j] * T[j];
                }
            }
            const double expected = k * m.area(0) * (f[1] * f[1] + f[2] * f[2]);
            EXPECT_NEAR(energy, expected, 1e-8 * expected) << "field with t0 = " << f[0] << ", gx = " << f[1]
                                                          << ", gy = " << f[2];
        }
    }
}

// EXAMPLE 17. K does not change if the triangle is shifted, rotated or scaled up. Shifting and rotating is
// obvious (the material is isotropic). Scaling by a factor s multiplies b and c by s and the area by
// s^2, so b^2 / A stays the same: in 2D the conductance of a plate does not depend on its
// size (a square of side 1 mm and one of side 1 m conduct equally, at equal thickness).
TEST(Fem2d, StiffnessDoesNotChangeWithPositionSizeOrRotation) {
    // s (scale factor), theta (rotation in radians), dx and dy (shift)
    const double moves[][4] = {{1.0, 0.0, 3.7, -2.2},                // only shifted
                               {250.0, 0.0, 0.0, 0.0},               // 250 times larger
                               {0.01, 0.0, 0.0, 0.0},                // 100 times smaller
                               {1.0, 1.5707963267948966, 0.0, 0.0},  // rotated by 90 degrees
                               {2.5, 0.5235987755982988, -3.0, 4.0}};  // 30 degrees, scaled up, shifted

    std::size_t n = 0;
    for (const Tri& t : sample_triangles()) {
        const Matrix3 base = element_stiffness(mesh_of(t), 0, 1.0);
        for (const auto& mv : moves) {
            SCOPED_TRACE("triangle number " + std::to_string(n) + ", s = " + std::to_string(mv[0]) +
                         ", theta = " + std::to_string(mv[1]));
            const Mesh2D other = mesh_of(moved(t, mv[0], mv[1], mv[2], mv[3]));
            expect_matrix_near(element_stiffness(other, 0, 1.0), base, 1e-9 * largest(base));
        }
        ++n;
    }
}

// EXAMPLE 18. The three nodes of a triangle can be listed starting from any vertex, provided
// one stays counter-clockwise: (0, 1, 2), (1, 2, 0) and (2, 0, 1) are the same triangle. The matrix
// must only change the order of rows and columns: in the mesh with (1, 2, 0) the local node i is the one
// that in the first was the local node (i + 1) % 3, so  K'[i][j] = K[(i+1)%3][(j+1)%3].
TEST(Fem2d, StiffnessFollowsTheNumberingOfTheNodes) {
    const std::vector<double> xs = {0.0, 4.0, 1.0};   // the triangle of EXAMPLE 12
    const std::vector<double> ys = {0.0, 0.0, 2.0};
    const Matrix3 a = element_stiffness(Mesh2D(xs, ys, {{0, 1, 2}}), 0, 1.0);
    const Matrix3 b = element_stiffness(Mesh2D(xs, ys, {{1, 2, 0}}), 0, 1.0);
    const Matrix3 c = element_stiffness(Mesh2D(xs, ys, {{2, 0, 1}}), 0, 1.0);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(b[i][j], a[(i + 1) % 3][(j + 1) % 3], 1e-14) << "(1, 2, 0), element (" << i << ", " << j << ")";
            EXPECT_NEAR(c[i][j], a[(i + 2) % 3][(j + 2) % 3], 1e-14) << "(2, 0, 1), element (" << i << ", " << j << ")";
        }
    }
}

// EXAMPLE 19. The result does not depend on where the triangle is in the mesh nor on the numbers of its nodes:
// for each triangle of a large mesh, the matrix is equal to that of the same triangle taken on
// its own (same vertices, nodes 0, 1, 2).
TEST(Fem2d, StiffnessDoesNotDependOnWhereTheTriangleSitsInTheMesh) {
    const Mesh2D big = rectangle_mesh(2.5, 0.5, 5, 3);   // 30 triangles

    for (std::size_t e = 0; e < big.num_triangles(); ++e) {
        const Triangle& t = big.triangle(e);
        const Tri alone = {big.x(t[0]), big.y(t[0]), big.x(t[1]), big.y(t[1]), big.x(t[2]), big.y(t[2])};
        const Matrix3 want = element_stiffness(mesh_of(alone), 0, 1.0);
        expect_matrix_near(element_stiffness(big, e, 1.0), want, 1e-13 * largest(want),
                           "triangle " + std::to_string(e) + ":");
    }
}

// EXAMPLE 20. A triangle that does not exist: std::out_of_range, as for Mesh2D::triangle.
TEST(Fem2d, StiffnessOfAnUnknownTriangleThrows) {
    const Mesh2D m = mesh_of({0.0, 0.0, 1.0, 0.0, 0.0, 1.0});   // it has only triangle 0
    EXPECT_NO_THROW(element_stiffness(m, 0, 1.0));
    EXPECT_THROW(element_stiffness(m, 1, 1.0), std::out_of_range);
    EXPECT_THROW(element_stiffness(m, 1000, 1.0), std::out_of_range);
}

// ---- Mass and load of a triangle: element_mass, element_load ------------------------------------
//
// element_mass(mesh, e, rho_c) is the 3 x 3 matrix of triangle e, for a material of volumetric heat
// capacity rho_c (density * specific heat), plate of thickness 1:
//
//     M[i][j] = integral over the triangle of  rho_c N_i N_j  dA
//
// element_load(mesh, e, q) is the vector of the three right-hand-side terms, for a heat source q
// (power per unit volume) uniform in the triangle:
//
//     f[i] = integral over the triangle of  q N_i  dA
//
// How they are computed by hand, with A = area of the triangle:
//   - The graph of N_i above the triangle is a pyramid: the triangle as base, height 1 above
//     node i. The volume of a pyramid is base * height / 3, so  integral of N_i dA = A / 3  and
//     f[i] = q A / 3: each node receives a third of the total heat q A.
//   - The three N_i add up to 1, so every row of M adds up to  rho_c * integral of N_i = rho_c A / 3.
//   - The three nodes are equivalent: every triangle is obtained from any other with an
//     affine transformation that sends N_i to N_i and multiplies the areas by a fixed factor. So M[i][i] = a rho_c A
//     and M[i][j] = b rho_c A (i different from j), with the same a and b for all the triangles, and a + 2 b = 1/3.
//     One integral is enough to find them. In the triangle (0, 0), (1, 0), (0, 1), where N_1 = x and A = 1/2:
//         a A = integral of x^2 dA = integral over x from 0 to 1 of  x^2 (1 - x) dx = 1/3 - 1/4 = 1/12
//     (for each x, y goes from 0 to 1 - x), i.e. a = 1/6, and then b = (1/3 - 1/6) / 2 = 1/12. So
//
//         M = rho_c A / 12 * [[2, 1, 1],           f = q A / 3 * [1, 1, 1]
//                             [1, 2, 1],
//                             [1, 1, 2]]
//
// In 1D it was the same: rho_c h / 6 * [[2, 1], [1, 2]] for the mass and q h / 2 per node for the load.

namespace {

// Compares two vectors element by element.
void expect_vector_near(const Vector3& got, const Vector3& want, double tol, const std::string& what = "") {
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(got[i], want[i], tol) << what << " element " << i;
    }
}

// The sum of all nine elements of K.
double sum_of_entries(const Matrix3& K) {
    double s = 0.0;
    for (const auto& row : K) {
        for (double v : row) {
            s += v;
        }
    }
    return s;
}

}  // namespace

// EXAMPLE 21. The triangle with nodes (0, 0), (1, 0), (0, 1), A = 1/2, where N_1 = x, N_2 = y and
// N_0 = 1 - x - y. With rho_c = 1 each element is computed directly:
//     M[1][1] = integral of x^2 dA = integral over x from 0 to 1 of  x^2 (1 - x) dx = 1/3 - 1/4 = 1/12
//     M[1][2] = integral of x y dA = integral over x from 0 to 1 of  x (1 - x)^2 / 2 dx = 1/24
// (for each x, y goes from 0 to 1 - x). The formula rho_c A / 12 = 1/24 gives the same: diagonal
// 2/24 = 1/12 and off the diagonal 1/24. With rho_c = 6 everything is multiplied by 6: 1/2 on the diagonal and 1/4 off it.
TEST(Fem2d, MassOfTheRightTriangleMatchesHandComputation) {
    const Mesh2D m = mesh_of({0.0, 0.0, 1.0, 0.0, 0.0, 1.0});

    const Matrix3 want1 = {{{1.0 / 12, 1.0 / 24, 1.0 / 24},
                            {1.0 / 24, 1.0 / 12, 1.0 / 24},
                            {1.0 / 24, 1.0 / 24, 1.0 / 12}}};
    expect_matrix_near(element_mass(m, 0, 1.0), want1, 1e-14, "rho_c = 1:");

    const Matrix3 want6 = {{{0.5, 0.25, 0.25}, {0.25, 0.5, 0.25}, {0.25, 0.25, 0.5}}};
    expect_matrix_near(element_mass(m, 0, 6.0), want6, 1e-14, "rho_c = 6:");
}

// EXAMPLE 22. The mass depends only on the AREA (and on rho_c), not on the shape. Here the two triangles
// of EXAMPLE 13 (nodes 0 = (0, 0), 1 = (4, 0), 2 = (1, 2), 3 = (6, 4); T0 = (0, 1, 2) with A = 4 and
// T1 = (1, 3, 2) with A = 8), with rho_c = 1.5:
//     T0:  rho_c A / 12 = 1.5 * 4 / 12 = 1/2    ->  M = 1/2 * [[2, 1, 1], [1, 2, 1], [1, 1, 2]]
//     T1:  rho_c A / 12 = 1.5 * 8 / 12 = 1      ->  M =       [[2, 1, 1], [1, 2, 1], [1, 1, 2]]
// It serves to check that the area of the requested triangle is used (and not that of the first).
TEST(Fem2d, MassUsesTheAreaOfTheRequestedTriangle) {
    const Mesh2D m({0.0, 4.0, 1.0, 6.0}, {0.0, 0.0, 2.0, 4.0}, {{0, 1, 2}, {1, 3, 2}});

    const Matrix3 want0 = {{{1.0, 0.5, 0.5}, {0.5, 1.0, 0.5}, {0.5, 0.5, 1.0}}};
    const Matrix3 want1 = {{{2.0, 1.0, 1.0}, {1.0, 2.0, 1.0}, {1.0, 1.0, 2.0}}};
    expect_matrix_near(element_mass(m, 0, 1.5), want0, 1e-14, "triangle 0:");
    expect_matrix_near(element_mass(m, 1, 1.5), want1, 1e-14, "triangle 1:");
}

// EXAMPLE 23. Things that hold for ANY triangle (here rho_c = 2.5):
//   - M is symmetric and all its elements are positive (N_i N_j > 0 inside the triangle);
//   - every row adds up to rho_c A / 3, because the three N_i add up to 1;
//   - all nine elements together add up to rho_c A, the heat capacity of the triangle (it is the integral
//     of rho_c (N_0 + N_1 + N_2)^2 = rho_c);
//   - off the diagonal there is half of the diagonal, always: the shape of the triangle does not matter.
TEST(Fem2d, MassIsSymmetricPositiveAndSumsToTheHeatCapacity) {
    const double rho_c = 2.5;

    std::size_t n = 0;
    for (const Tri& t : sample_triangles()) {
        SCOPED_TRACE("triangle number " + std::to_string(n++));
        const Mesh2D m = mesh_of(t);
        const double area = m.area(0);
        const Matrix3 M = element_mass(m, 0, rho_c);
        const double tol = 1e-12 * rho_c * area;

        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_NEAR(M[i][0] + M[i][1] + M[i][2], rho_c * area / 3.0, tol) << "row " << i;
            for (std::size_t j = 0; j < 3; ++j) {
                EXPECT_GT(M[i][j], 0.0) << "element (" << i << ", " << j << ")";
                EXPECT_NEAR(M[i][j], M[j][i], tol) << "element (" << i << ", " << j << ")";
                EXPECT_NEAR(M[i][j], (i == j ? 1.0 : 0.5) * M[0][0], tol) << "element (" << i << ", " << j << ")";
            }
        }
        EXPECT_NEAR(sum_of_entries(M), rho_c * area, tol);
    }
}

// EXAMPLE 24. A check of the whole of M by another route. Three values T_0, T_1, T_2 at the nodes
// define a LINEAR field T(x, y) in the triangle, and  T^T M T = sum of T_i M[i][j] T_j  is
// the integral of rho_c T^2. T^2 is a polynomial of degree 2, and for polynomials of degree 2 the integral over
// a triangle is EXACTLY area / 3 times the sum of the values at the three MIDPOINTS of the sides. At a
// midpoint a linear field equals the mean of the values at the two ends of the side. So
//     T^T M T = rho_c A / 3 * [ ((T_0 + T_1) / 2)^2 + ((T_1 + T_2) / 2)^2 + ((T_2 + T_0) / 2)^2 ]
TEST(Fem2d, MassGivesTheIntegralOfTheSquareOfALinearField) {
    const double rho_c = 2.5;
    const double fields[][3] = {{1.0, 0.0, 0.0},  {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
                                {1.0, 1.0, 1.0},  {1.0, -1.0, 0.0}, {2.0, -3.0, 5.0}};

    std::size_t n = 0;
    for (const Tri& t : sample_triangles()) {
        SCOPED_TRACE("triangle number " + std::to_string(n++));
        const Mesh2D m = mesh_of(t);
        const Matrix3 M = element_mass(m, 0, rho_c);

        for (const auto& T : fields) {
            double energy = 0.0;
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    energy += T[i] * M[i][j] * T[j];
                }
            }
            const double mid01 = 0.5 * (T[0] + T[1]);
            const double mid12 = 0.5 * (T[1] + T[2]);
            const double mid20 = 0.5 * (T[2] + T[0]);
            const double expected = rho_c * m.area(0) / 3.0 * (mid01 * mid01 + mid12 * mid12 + mid20 * mid20);
            EXPECT_NEAR(energy, expected, 1e-12 * expected)
                << "campo (" << T[0] << ", " << T[1] << ", " << T[2] << ")";
        }
    }
}

// EXAMPLE 25. The triangle (0, 0), (1, 0), (0, 1) has A = 1/2, so q A / 3 = q / 6 for each node:
//     q = 6: 1 each (total heat 6 * 1/2 = 3, a third per node);     q = 3: 1/2;
//     q = -6: -1 each (a heat sink: the sign is preserved);        q = 0: nothing.
TEST(Fem2d, LoadOfTheRightTriangleMatchesHandComputation) {
    const Mesh2D m = mesh_of({0.0, 0.0, 1.0, 0.0, 0.0, 1.0});

    expect_vector_near(element_load(m, 0, 6.0), {{1.0, 1.0, 1.0}}, 1e-14, "q = 6:");
    expect_vector_near(element_load(m, 0, 3.0), {{0.5, 0.5, 0.5}}, 1e-14, "q = 3:");
    expect_vector_near(element_load(m, 0, -6.0), {{-1.0, -1.0, -1.0}}, 1e-14, "q = -6:");
    expect_vector_near(element_load(m, 0, 0.0), {{0.0, 0.0, 0.0}}, 1e-14, "q = 0:");
}

// EXAMPLE 26. Like EXAMPLE 22, with the two triangles of EXAMPLE 13 and q = 1.5:
//     T0 (A = 4):  q A / 3 = 1.5 * 4 / 3 = 2 per node        T1 (A = 8):  q A / 3 = 1.5 * 8 / 3 = 4 per node
TEST(Fem2d, LoadUsesTheAreaOfTheRequestedTriangle) {
    const Mesh2D m({0.0, 4.0, 1.0, 6.0}, {0.0, 0.0, 2.0, 4.0}, {{0, 1, 2}, {1, 3, 2}});

    expect_vector_near(element_load(m, 0, 1.5), {{2.0, 2.0, 2.0}}, 1e-14, "triangle 0:");
    expect_vector_near(element_load(m, 1, 1.5), {{4.0, 4.0, 4.0}}, 1e-14, "triangle 1:");
}

// EXAMPLE 27. The load and the mass are linked. The three N_i add up to 1, so
//     f[i] = integral of q N_i = (q / rho_c) * integral of rho_c N_i (N_0 + N_1 + N_2)
//          = (q / rho_c) * (M[i][0] + M[i][1] + M[i][2])
// i.e. the load of node i is the sum of row i of the mass, times q / rho_c. Besides, the three nodes
// receive the same heat, and the total is q A, for any shape of the triangle.
TEST(Fem2d, LoadIsTheRowSumOfTheMassAndAddsUpToTheTotalHeat) {
    const double rho_c = 2.5;
    const double q = 7.0;

    std::size_t n = 0;
    for (const Tri& t : sample_triangles()) {
        SCOPED_TRACE("triangle number " + std::to_string(n++));
        const Mesh2D m = mesh_of(t);
        const Matrix3 M = element_mass(m, 0, rho_c);
        const Vector3 f = element_load(m, 0, q);
        const double tol = 1e-12 * q * m.area(0);

        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_NEAR(f[i], q / rho_c * (M[i][0] + M[i][1] + M[i][2]), tol) << "node " << i;
        }
        EXPECT_NEAR(f[0], f[1], tol);
        EXPECT_NEAR(f[1], f[2], tol);
        EXPECT_NEAR(f[0] + f[1] + f[2], q * m.area(0), tol);
    }
}

// EXAMPLE 28. A triangle that does not exist: std::out_of_range, for both functions.
TEST(Fem2d, MassAndLoadOfAnUnknownTriangleThrow) {
    const Mesh2D m = mesh_of({0.0, 0.0, 1.0, 0.0, 0.0, 1.0});   // it has only triangle 0
    EXPECT_NO_THROW(element_mass(m, 0, 1.0));
    EXPECT_NO_THROW(element_load(m, 0, 1.0));
    EXPECT_THROW(element_mass(m, 1, 1.0), std::out_of_range);
    EXPECT_THROW(element_mass(m, 1000, 1.0), std::out_of_range);
    EXPECT_THROW(element_load(m, 1, 1.0), std::out_of_range);
    EXPECT_THROW(element_load(m, 1000, 1.0), std::out_of_range);
}

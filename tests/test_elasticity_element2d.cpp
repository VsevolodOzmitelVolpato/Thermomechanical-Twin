#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "Elasticity2D.h"
#include "Mesh2D.h"

// ---- The deforming triangle: the matrix B -------------------------------------------------------------------------
//
// In the heat problem the unknown of a triangle was 3 temperatures, one per node, and inside the triangle the temperature was the
// linear function  T = N_0 T_0 + N_1 T_1 + N_2 T_2.  Now the unknowns are the DISPLACEMENTS (u, v): each node has
// two, so a triangle has 6 numbers, in the order
//
//     u_e = (u_0, v_0, u_1, v_1, u_2, v_2)        (for each node first x and then y)
//
// and inside the triangle the displacements are linear, with the same functions N_i as in the heat problem:
//
//     u(x, y) = N_0 u_0 + N_1 u_1 + N_2 u_2,        v(x, y) = N_0 v_0 + N_1 v_1 + N_2 v_2.
//
// We already know the derivatives (see element_stiffness): grad(N_i) = (b_i, c_i) / (2 A), with p = (i+1) % 3,
// q = (i+2) % 3, b_i = y_p - y_q and c_i = x_q - x_p. So, for the three strains of the plane,
//
//     eps_xx   = du/dx           = sum_i  b_i u_i / (2 A)
//     eps_yy   = dv/dy           = sum_i  c_i v_i / (2 A)
//     gamma_xy = du/dy + dv/dx   = sum_i (c_i u_i + b_i v_i) / (2 A)
//
// These are three linear combinations of the 6 numbers: eps = B u_e, with the 3 x 6 matrix
//
//               [ b_0   0    b_1   0    b_2   0  ]
//     1 / (2 A) [  0   c_0    0   c_1    0   c_2 ]       (rows: eps_xx, eps_yy, gamma_xy;
//               [ c_0  b_0   c_1  b_1   c_2  b_2 ]        columns: u_0, v_0, u_1, v_1, u_2, v_2).
//
// The gradients are constant, so the strain is the SAME in the whole triangle ("constant-strain
// triangle"). A linear function is reproduced exactly: it is the strongest check (EXAMPLE 126).

// B has 3 rows (the strains) and 6 columns (the degrees of freedom), u_e has 6 numbers: if in the types the numbers are
// swapped or wrong, the test does not compile and says which one.
static_assert(std::tuple_size<Matrix3x6>::value == 3, "Matrix3x6 must have 3 rows");
static_assert(std::tuple_size<Matrix3x6::value_type>::value == 6, "Matrix3x6 must have 6 columns");
static_assert(std::tuple_size<Vector6>::value == 6, "Vector6 must have 6 numbers");

namespace {

// The vertices of a triangle, counter-clockwise.
struct Tri {
    double x0, y0, x1, y1, x2, y2;
};

// A mesh with that single triangle (nodes 0, 1, 2 in this order).
Mesh2D mesh_of(const Tri& t) {
    return Mesh2D({t.x0, t.x1, t.x2}, {t.y0, t.y1, t.y2}, {{0, 1, 2}});
}

// The triangle t scaled by s (about the origin) and shifted by (dx, dy).
Tri scaled_and_shifted(const Tri& t, double s, double dx, double dy) {
    return {s * t.x0 + dx, s * t.y0 + dy, s * t.x1 + dx, s * t.y1 + dy, s * t.x2 + dx, s * t.y2 + dy};
}

// Six counter-clockwise triangles, of very different shapes and sizes (the same as in the element_stiffness tests).
std::vector<Tri> sample_triangles() {
    return {{0.0, 0.0, 1.0, 0.0, 0.0, 1.0},         // with a right angle
            {0.0, 0.0, 4.0, 0.0, 1.0, 2.0},         // scalene
            {1.0, 1.0, 5.0, 2.0, 2.0, 4.0},         // scalene, far from the origin
            {-1.5, 2.0, 0.5, -1.0, 3.0, 1.5},       // with negative coordinates
            {0.0, 0.0, 10.0, 0.0, 5.0, 0.1},        // very flat
            {0.0, 0.0, 0.004, 0.0, 0.001, 0.002}};  // small (millimeters)
}

// Three counter-clockwise triangles with shared nodes and different areas (2, 1.75 and 1.5): they check
// that B uses the nodes and the area of the right triangle of the mesh.
Mesh2D irregular_mesh() {
    return Mesh2D({0.0, 2.0, 3.0, 1.0, -1.0}, {0.0, 0.0, 1.5, 2.0, 1.0}, {{0, 1, 3}, {1, 2, 3}, {0, 3, 4}});
}

// A linear displacement field:  u = a0 + a1 x + a2 y,  v = b0 + b1 x + b2 y.
// At every point it has the same strain: (eps_xx, eps_yy, gamma_xy) = (a1, b2, a2 + b1).
struct LinearField {
    double a0, a1, a2, b0, b1, b2;
};

// The six numbers (u_0, v_0, u_1, v_1, u_2, v_2) of the field f at the three nodes of triangle e of the mesh.
Vector6 nodal_values(const Mesh2D& mesh, std::size_t e, const LinearField& f) {
    const Triangle& t = mesh.triangle(e);
    Vector6 u{};
    for (std::size_t i = 0; i < 3; ++i) {
        const double x = mesh.x(t[i]);
        const double y = mesh.y(t[i]);
        u[2 * i] = f.a0 + f.a1 * x + f.a2 * y;
        u[2 * i + 1] = f.b0 + f.b1 * x + f.b2 * y;
    }
    return u;
}

// The strain eps = B u.
Vector3 strain_of(const Matrix3x6& B, const Vector6& u) {
    Vector3 eps{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            eps[i] += B[i][j] * u[j];
        }
    }
    return eps;
}

// The largest entry of B in absolute value (to set the tolerances).
double largest(const Matrix3x6& B) {
    double m = 0.0;
    for (const auto& row : B) {
        for (double v : row) {
            m = std::max(m, std::fabs(v));
        }
    }
    return m;
}

// Check that the B of triangle e gives the exact strain (a1, b2, a2 + b1) of the field f.
void expect_exact_strain(const Mesh2D& mesh, std::size_t e, const LinearField& f, double tol, const std::string& what) {
    const Vector3 got = strain_of(strain_displacement_matrix(mesh, e), nodal_values(mesh, e, f));
    const Vector3 want = {f.a1, f.b2, f.a2 + f.b1};
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(got[i], want[i], tol) << what << ", component " << i;
    }
}

}  // namespace

// EXAMPLE 125. By hand. The triangle (0,0), (1,0), (0,1) has A = 1/2, so 2 A = 1:
//     i = 0 (p = 1, q = 2):  b_0 = y_1 - y_2 = 0 - 1 = -1,   c_0 = x_2 - x_1 = 0 - 1 = -1
//     i = 1 (p = 2, q = 0):  b_1 = y_2 - y_0 = 1 - 0 =  1,   c_1 = x_0 - x_2 = 0 - 0 =  0
//     i = 2 (p = 0, q = 1):  b_2 = y_0 - y_1 = 0 - 0 =  0,   c_2 = x_1 - x_0 = 1 - 0 =  1
// and therefore
//     B = [[-1,  0, 1, 0, 0, 0],
//          [ 0, -1, 0, 0, 0, 1],
//          [-1, -1, 0, 1, 1, 0]].
// It reads as follows: moving only node 1 by 1 in x (u_1 = 1) stretches the triangle in x by 1 (eps_xx = 1) and that is all;
// v_2 = 1 stretches it in y; u_2 = 1 and v_1 = 1 each give a shear strain gamma_xy = 1.
// The triangle (0,0), (2,0), (0,1) has A = 1, so 2 A = 2, with b = (-1, 1, 0) and c = (-2, 0, 2):
//     B = [[-0.5,  0,   0.5, 0,   0, 0],
//          [ 0,   -1,   0,   0,   0, 1],
//          [-1,   -0.5, 0,   0.5, 1, 0]].
TEST(StrainDisplacementMatrix, MatchesHandComputation) {
    const double want_unit[3][6] = {{-1.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                                    {0.0, -1.0, 0.0, 0.0, 0.0, 1.0},
                                    {-1.0, -1.0, 0.0, 1.0, 1.0, 0.0}};
    const Matrix3x6 B1 = strain_displacement_matrix(mesh_of({0.0, 0.0, 1.0, 0.0, 0.0, 1.0}), 0);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            EXPECT_NEAR(B1[i][j], want_unit[i][j], 1e-14) << "triangle (0,0) (1,0) (0,1), B[" << i << "][" << j << "]";
        }
    }

    const double want_wide[3][6] = {{-0.5, 0.0, 0.5, 0.0, 0.0, 0.0},
                                    {0.0, -1.0, 0.0, 0.0, 0.0, 1.0},
                                    {-1.0, -0.5, 0.0, 0.5, 1.0, 0.0}};
    const Matrix3x6 B2 = strain_displacement_matrix(mesh_of({0.0, 0.0, 2.0, 0.0, 0.0, 1.0}), 0);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            EXPECT_NEAR(B2[i][j], want_wide[i][j], 1e-14) << "triangle (0,0) (2,0) (0,1), B[" << i << "][" << j << "]";
        }
    }
}

// EXAMPLE 126. B must give the EXACT strain of every linear displacement field, because the displacements of the
// triangle are linear. Take  u = a0 + a1 x + a2 y,  v = b0 + b1 x + b2 y, compute the 6 numbers at the nodes and
// check that B u_e gives (eps_xx, eps_yy, gamma_xy) = (a1, b2, a2 + b1). Six "pure" fields: the two translations
// (strain 0), u = x (eps_xx = 1), v = y (eps_yy = 1), u = y and v = x (gamma_xy = 1 each); plus an
// arbitrary field with all the numbers different from 0. Tried on six triangles of very different shapes.
// By hand, on the triangle (0,0), (2,0), (0,1) with u = 0.1 + 0.01 x + 0.02 y,  v = -0.2 + 0.03 x + 0.04 y:
//     u_e = (0.1, -0.2, 0.12, -0.14, 0.12, -0.16)
//     eps_xx   = -0.5 * 0.1 + 0.5 * 0.12                                          = 0.01  (= a1)
//     eps_yy   = -1 * (-0.2) + 1 * (-0.16)                                        = 0.04  (= b2)
//     gamma_xy = -1 * 0.1 - 0.5 * (-0.2) + 0.5 * (-0.14) + 1 * 0.12               = 0.05  (= a2 + b1 = 0.02 + 0.03)
TEST(StrainDisplacementMatrix, ReproducesTheStrainOfLinearFields) {
    const Mesh2D hand = mesh_of({0.0, 0.0, 2.0, 0.0, 0.0, 1.0});
    const Vector6 u_hand = {0.1, -0.2, 0.12, -0.14, 0.12, -0.16};
    const Vector3 eps_hand = strain_of(strain_displacement_matrix(hand, 0), u_hand);
    EXPECT_NEAR(eps_hand[0], 0.01, 1e-14);
    EXPECT_NEAR(eps_hand[1], 0.04, 1e-14);
    EXPECT_NEAR(eps_hand[2], 0.05, 1e-14);

    const LinearField fields[] = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},                    // translation in x
                                  {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},                    // translation in y
                                  {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},                    // u = x
                                  {0.0, 0.0, 0.0, 0.0, 0.0, 1.0},                    // v = y
                                  {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},                    // u = y
                                  {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},                    // v = x
                                  {0.3, 1.2e-3, -0.7e-3, -0.2, 0.5e-3, 2.0e-3}};     // an arbitrary field
    const std::vector<Tri> triangles = sample_triangles();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        for (std::size_t m = 0; m < 7; ++m) {
            const std::string what = "triangle " + std::to_string(k) + ", field " + std::to_string(m);
            expect_exact_strain(mesh, 0, fields[m], 1e-11, what);
        }
    }
}

// EXAMPLE 127. A rigid motion does not deform. A translation (tx, ty) and a small rotation by an angle theta
// about the point (xc, yc) give the displacements
//     u = tx - theta (y - yc),        v = ty + theta (x - xc),
// which are linear: du/dx = 0, dv/dy = 0 and gamma_xy = du/dy + dv/dx = -theta + theta = 0. So B u_e must be
// (0, 0, 0), whatever the triangle. This holds only with the shear strain gamma_xy = du/dy + dv/dx: with only one of the two
// terms a rotation would look like a deformation. (A B made of all zeros would pass this check without proving
// anything: that is why we first check that B is not zero.)
TEST(StrainDisplacementMatrix, RigidMotionsDoNotDeform) {
    const double centers[2][2] = {{0.0, 0.0}, {0.7, -0.3}};
    const double angles[4] = {0.0, 0.01, 0.3, -0.5};
    const std::vector<Tri> triangles = sample_triangles();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        const Matrix3x6 B = strain_displacement_matrix(mesh, 0);
        ASSERT_GT(largest(B), 0.0) << "B is all zeros (triangle " << k << "): a zero B never deforms";
        for (const auto& center : centers) {
            for (double theta : angles) {
                const double tx = 0.2;
                const double ty = -0.4;
                const LinearField rigid = {tx + theta * center[1], 0.0, -theta, ty - theta * center[0], theta, 0.0};
                const Vector3 eps = strain_of(B, nodal_values(mesh, 0, rigid));
                for (std::size_t i = 0; i < 3; ++i) {
                    EXPECT_NEAR(eps[i], 0.0, 1e-11)
                        << "triangle " << k << ", theta = " << theta << ", component " << i;
                }
            }
        }
    }
}

// EXAMPLE 128. How B changes with the triangle. If the triangle is scaled by s (and possibly shifted), the sides b_i and
// c_i grow by s and 2 A by s^2: B becomes B / s. A strain is a displacement divided by a length, so with
// the same displacements at the nodes a larger triangle deforms less. Shifting the triangle without scaling it
// (s = 1) does not change B.
TEST(StrainDisplacementMatrix, ScalesWithTheInverseOfTheSize) {
    const double scales[4] = {1.0, 2.0, 0.25, 10.0};
    const double shifts[2][2] = {{0.0, 0.0}, {3.0, -5.0}};
    const std::vector<Tri> triangles = sample_triangles();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Matrix3x6 base = strain_displacement_matrix(mesh_of(triangles[k]), 0);
        ASSERT_GT(largest(base), 0.0) << "B is all zeros (triangle " << k << ")";
        for (double s : scales) {
            for (const auto& shift : shifts) {
                const Tri moved = scaled_and_shifted(triangles[k], s, shift[0], shift[1]);
                const Matrix3x6 B = strain_displacement_matrix(mesh_of(moved), 0);
                const double tol = 1e-10 * largest(base) / s;
                for (std::size_t i = 0; i < 3; ++i) {
                    for (std::size_t j = 0; j < 6; ++j) {
                        EXPECT_NEAR(B[i][j], base[i][j] / s, tol)
                            << "triangle " << k << ", s = " << s << ", B[" << i << "][" << j << "]";
                    }
                }
            }
        }
    }
}

// EXAMPLE 129. B of a triangle of a real mesh, where the nodes of the triangle are not 0, 1, 2 and the triangles have
// different areas. Each triangle, taken with its nodes (mesh.triangle(e)) and its area (mesh.area(e)), must reproduce the
// same constant strain of a linear field given at the mesh nodes. Tried on a hand-made mesh of 3 irregular triangles
// and on the rectangular mesh of 4 x 3 cells (24 triangles).
TEST(StrainDisplacementMatrix, WorksOnEveryTriangleOfAMesh) {
    const LinearField field = {0.3, 1.2e-3, -0.7e-3, -0.2, 0.5e-3, 2.0e-3};

    const Mesh2D irregular = irregular_mesh();
    for (std::size_t e = 0; e < irregular.num_triangles(); ++e) {
        expect_exact_strain(irregular, e, field, 1e-12, "irregular mesh, triangle " + std::to_string(e));
    }

    const Mesh2D rectangle = rectangle_mesh(2.0, 1.5, 4, 3);
    ASSERT_EQ(rectangle.num_triangles(), 24u);
    for (std::size_t e = 0; e < rectangle.num_triangles(); ++e) {
        expect_exact_strain(rectangle, e, field, 1e-12, "rectangular mesh, triangle " + std::to_string(e));
    }
}

// EXAMPLE 130. A triangle that does not exist: std::out_of_range, as for the other functions of the element.
TEST(StrainDisplacementMatrix, RejectsMissingTriangle) {
    const Mesh2D mesh = irregular_mesh();
    EXPECT_NO_THROW(strain_displacement_matrix(mesh, 0));
    EXPECT_NO_THROW(strain_displacement_matrix(mesh, mesh.num_triangles() - 1));
    EXPECT_THROW(strain_displacement_matrix(mesh, mesh.num_triangles()), std::out_of_range);
    EXPECT_THROW(strain_displacement_matrix(mesh, 1000), std::out_of_range);
}

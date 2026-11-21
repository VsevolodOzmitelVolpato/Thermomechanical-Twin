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

// ---- The stiffness of the triangle: Ke = A B^T D B ------------------------------------------------------------------
//
// From the matrix B we get to the stiffness through ENERGY. A deformed triangle stores energy: per unit area
// it is (1/2) eps^T D eps (like a spring, (1/2) k x^2) and, since eps is constant in the triangle, it is enough to multiply by
// the area A. With eps = B u_e:
//
//     U = (1/2) A (B u_e)^T D (B u_e) = (1/2) u_e^T (A B^T D B) u_e = (1/2) u_e^T Ke u_e.
//
// As for the spring, where U = (1/2) k x^2 gives the force k x, the 6 forces on the degrees of freedom of the triangle are the
// derivative of U with respect to the 6 displacements: f_e = Ke u_e. So Ke (6 x 6, thickness 1) is the "spring constant" of the
// triangle, and Ke[i][j] is the force on degree of freedom i when only degree of freedom j is moved by 1.
//
// What a real stiffness must have, and this is what the tests check:
//   - it is symmetric (the force on i when moving j by 1 is the one on j when moving i by 1);
//   - a rigid motion (translation, rotation) does not deform, so there is no energy and no forces: Ke u_rigid = 0;
//   - a linear field has constant strain: its forces are those of the stress sigma = D eps and its energy
//     is A eps^T D eps;
//   - it does not depend on the size of the triangle: A grows like s^2 and B^T B decreases like 1/s^2. (To move by d a
//     side of length L a force = stress x length = E (d / L) L = E d is needed, which does not depend on L);
//   - an isotropic material has no preferred direction: rotating the triangle and the displacements, the forces rotate
//     in the same way, K' = Q K Q^T.

// Ke is a 6 x 6 table: if the type has other dimensions the test does not compile and says which one.
static_assert(std::tuple_size<Matrix6>::value == 6, "Matrix6 must have 6 rows");
static_assert(std::tuple_size<Matrix6::value_type>::value == 6, "Matrix6 must have 6 columns");

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

// The triangle t rotated by phi (in radians) about the origin: (x, y) becomes (c x - s y, s x + c y).
Tri rotated(const Tri& t, double phi) {
    const double c = std::cos(phi);
    const double s = std::sin(phi);
    return {c * t.x0 - s * t.y0, s * t.x0 + c * t.y0, c * t.x1 - s * t.y1,
            s * t.x1 + c * t.y1, c * t.x2 - s * t.y2, s * t.x2 + c * t.y2};
}

// Six counter-clockwise triangles, of very different shapes and sizes (the same as in the B tests).
std::vector<Tri> sample_triangles() {
    return {{0.0, 0.0, 1.0, 0.0, 0.0, 1.0},         // with a right angle
            {0.0, 0.0, 4.0, 0.0, 1.0, 2.0},         // scalene
            {1.0, 1.0, 5.0, 2.0, 2.0, 4.0},         // scalene, far from the origin
            {-1.5, 2.0, 0.5, -1.0, 3.0, 1.5},       // with negative coordinates
            {0.0, 0.0, 10.0, 0.0, 5.0, 0.1},        // very flat
            {0.0, 0.0, 0.004, 0.0, 0.001, 0.002}};  // small (millimeters)
}

// Three counter-clockwise triangles with shared nodes and different areas (2, 1.75 and 1.5).
Mesh2D irregular_mesh() {
    return Mesh2D({0.0, 2.0, 3.0, 1.0, -1.0}, {0.0, 0.0, 1.5, 2.0, 1.0}, {{0, 1, 3}, {1, 2, 3}, {0, 3, 4}});
}

// The area of triangle e from its three nodes (determinant formula), without using mesh.area.
double triangle_area(const Mesh2D& mesh, std::size_t e) {
    const Triangle& t = mesh.triangle(e);
    const double x0 = mesh.x(t[0]);
    const double y0 = mesh.y(t[0]);
    return 0.5 * ((mesh.x(t[1]) - x0) * (mesh.y(t[2]) - y0) - (mesh.x(t[2]) - x0) * (mesh.y(t[1]) - y0));
}

// Hand-made materials (3 x 3, indices xx, yy, xy), symmetric and positive definite.
Matrix3 hooke_quarter() {  // E = 1, nu = 1/4: lambda = mu = 0.4
    return {{{1.2, 0.4, 0.0}, {0.4, 1.2, 0.0}, {0.0, 0.0, 0.4}}};
}
Matrix3 stretched() {  // xx and yy different: if the indices of D are swapped it shows
    return {{{2.0, 1.0, 0.0}, {1.0, 3.0, 0.0}, {0.0, 0.0, 1.0}}};
}
Matrix3 general() {  // all the numbers different, also the coupling with shear
    return {{{2.0, 0.5, 0.1}, {0.5, 3.0, -0.2}, {0.1, -0.2, 1.0}}};
}

// Real materials (from plane_strain_matrix), among them one as stiff as steel, one with nu < 0 and one almost
// incompressible.
std::vector<Matrix3> sample_materials() {
    return {general(), plane_strain_matrix(1.0, 0.25), plane_strain_matrix(210e9, 0.3), plane_strain_matrix(1.0, -0.5),
            plane_strain_matrix(1.0, 0.49)};
}

// A linear displacement field:  u = a0 + a1 x + a2 y,  v = b0 + b1 x + b2 y.
// At every point it has the same strain (eps_xx, eps_yy, gamma_xy) = (a1, b2, a2 + b1).
struct LinearField {
    double a0, a1, a2, b0, b1, b2;
};

// The six numbers (u_0, v_0, u_1, v_1, u_2, v_2) of the field f at the three nodes of triangle e.
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

// The six "pure" fields, one for each number a0, b0, a1, b2, a2, b1 (the two translations, u = x, v = y, u = y, v = x),
// plus an arbitrary field with all the numbers different from 0. The first six are a basis of all the linear fields.
std::vector<LinearField> sample_fields() {
    return {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
            {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
            {0.003, 1.2e-3, -0.7e-3, -0.002, 0.5e-3, 2.0e-3}};
}

// The strain and the stress of a linear field, and the number eps^T D eps.
Vector3 strain_of(const LinearField& f) {
    return {f.a1, f.b2, f.a2 + f.b1};
}
Vector3 stress_of(const Matrix3& D, const Vector3& eps) {
    Vector3 sigma{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            sigma[i] += D[i][j] * eps[j];
        }
    }
    return sigma;
}
double strain_times_stress(const Matrix3& D, const Vector3& eps) {
    const Vector3 sigma = stress_of(D, eps);
    return eps[0] * sigma[0] + eps[1] * sigma[1] + eps[2] * sigma[2];
}

// The forces on the 6 degrees of freedom produced by a constant stress sigma: A B^T sigma. For node i, with
// p = (i+1) % 3, q = (i+2) % 3, b_i = y_p - y_q and c_i = x_q - x_p (as in the matrix B), they are
//     f_x = (b_i sigma_xx + c_i sigma_xy) / 2,        f_y = (c_i sigma_yy + b_i sigma_xy) / 2.
Vector6 nodal_forces(const Mesh2D& mesh, std::size_t e, const Vector3& sigma) {
    const Triangle& t = mesh.triangle(e);
    Vector6 f{};
    for (std::size_t i = 0; i < 3; ++i) {
        const std::size_t p = (i + 1) % 3;
        const std::size_t q = (i + 2) % 3;
        const double b = mesh.y(t[p]) - mesh.y(t[q]);
        const double c = mesh.x(t[q]) - mesh.x(t[p]);
        f[2 * i] = (b * sigma[0] + c * sigma[2]) / 2.0;
        f[2 * i + 1] = (c * sigma[1] + b * sigma[2]) / 2.0;
    }
    return f;
}

// The largest of the numbers of Ke (or of u) in absolute value, to set the tolerances.
double largest(const Matrix6& K) {
    double m = 0.0;
    for (const auto& row : K) {
        for (double v : row) {
            m = std::max(m, std::fabs(v));
        }
    }
    return m;
}
double largest(const Vector6& u) {
    double m = 0.0;
    for (double v : u) {
        m = std::max(m, std::fabs(v));
    }
    return m;
}

// The product Ke u and the number u^T Ke u.
Vector6 times(const Matrix6& K, const Vector6& u) {
    Vector6 f{};
    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            f[i] += K[i][j] * u[j];
        }
    }
    return f;
}
double energy_of(const Matrix6& K, const Vector6& u) {
    const Vector6 f = times(K, u);
    double s = 0.0;
    for (std::size_t i = 0; i < 6; ++i) {
        s += u[i] * f[i];
    }
    return s;
}

// Q K Q^T, where Q has on the diagonal three 2 x 2 rotation blocks R(phi) = [[c, -s], [s, c]], one per node:
// it is the stiffness matrix of a triangle rotated by phi, if the material is isotropic.
Matrix6 rotated_matrix(const Matrix6& K, double phi) {
    const double c = std::cos(phi);
    const double s = std::sin(phi);
    Matrix6 Q{};
    for (std::size_t n = 0; n < 3; ++n) {
        Q[2 * n][2 * n] = c;
        Q[2 * n][2 * n + 1] = -s;
        Q[2 * n + 1][2 * n] = s;
        Q[2 * n + 1][2 * n + 1] = c;
    }
    Matrix6 R{};
    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            for (std::size_t k = 0; k < 6; ++k) {
                for (std::size_t l = 0; l < 6; ++l) {
                    R[i][j] += Q[i][k] * K[k][l] * Q[j][l];
                }
            }
        }
    }
    return R;
}

}  // namespace

// EXAMPLE 131. By hand. The triangle (0,0), (1,0), (0,1) has A = 1/2 and (see the matrix B)
//     B = [[-1,  0, 1, 0, 0, 0],
//          [ 0, -1, 0, 0, 0, 1],
//          [-1, -1, 0, 1, 1, 0]].
// With E = 1 and nu = 1/4 (lambda = mu = 0.4) D = [[1.2, 0.4, 0], [0.4, 1.2, 0], [0, 0, 0.4]]. First D B is computed
// (3 x 6) and then Ke[i][j] = A (column i of B) . (column j of D B):
//     D B = [[-1.2, -0.4, 1.2, 0,   0,   0.4],
//            [-0.4, -1.2, 0.4, 0,   0,   1.2],
//            [-0.4, -0.4, 0,   0.4, 0.4, 0  ]]
//     Ke[0][0] = 1/2 ((-1)(-1.2) + 0 (-0.4) + (-1)(-0.4)) = 0.8,     Ke[0][2] = 1/2 ((-1)(1.2) + 0 + (-1) 0) = -0.6,
// and so on for all 36 entries:
//     Ke = [[ 0.8,  0.4, -0.6, -0.2, -0.2, -0.2],
//           [ 0.4,  0.8, -0.2, -0.2, -0.2, -0.6],
//           [-0.6, -0.2,  0.6,  0,    0,    0.2],
//           [-0.2, -0.2,  0,    0.2,  0.2,  0  ],
//           [-0.2, -0.2,  0,    0.2,  0.2,  0  ],
//           [-0.2, -0.6,  0.2,  0,    0,    0.6]].
// Column 0 is the force when only u_0 = 1 (node 0 moves by 1 in x): (0.8, 0.4, -0.6, -0.2, -0.2, -0.2). The
// forces in x add up to 0.8 - 0.6 - 0.2 = 0 and those in y to 0.4 - 0.2 - 0.2 = 0: no net force on the triangle.
// The second triangle, (0,0), (2,0), (0,1) with A = 1, has a material with different xx and yy, D = [[2, 1, 0], [1, 3, 0],
// [0, 0, 1]], to check whether each index of D is in its place:
//     D B = [[-1,   -1,   1,   0,   0,   1],
//            [-0.5, -3,   0.5, 0,   0,   3],
//            [-1,   -0.5, 0,   0.5, 1,   0]]
//     Ke = [[ 1.5,  1,    -0.5, -0.5,  -1,   -0.5],
//           [ 1,    3.25, -0.5, -0.25, -0.5, -3  ],
//           [-0.5, -0.5,   0.5,  0,     0,    0.5],
//           [-0.5, -0.25,  0,    0.25,  0.5,  0  ],
//           [-1,   -0.5,   0,    0.5,   1,    0  ],
//           [-0.5, -3,     0.5,  0,     0,    3  ]].
TEST(ElementElasticityStiffness, MatchesHandComputation) {
    const double want_unit[6][6] = {{0.8, 0.4, -0.6, -0.2, -0.2, -0.2},
                                    {0.4, 0.8, -0.2, -0.2, -0.2, -0.6},
                                    {-0.6, -0.2, 0.6, 0.0, 0.0, 0.2},
                                    {-0.2, -0.2, 0.0, 0.2, 0.2, 0.0},
                                    {-0.2, -0.2, 0.0, 0.2, 0.2, 0.0},
                                    {-0.2, -0.6, 0.2, 0.0, 0.0, 0.6}};
    const Matrix6 K1 = element_elasticity_stiffness(mesh_of({0.0, 0.0, 1.0, 0.0, 0.0, 1.0}), 0, hooke_quarter());
    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            EXPECT_NEAR(K1[i][j], want_unit[i][j], 1e-14)
                << "triangle (0,0) (1,0) (0,1), Ke[" << i << "][" << j << "]";
        }
    }

    const double want_wide[6][6] = {{1.5, 1.0, -0.5, -0.5, -1.0, -0.5},
                                    {1.0, 3.25, -0.5, -0.25, -0.5, -3.0},
                                    {-0.5, -0.5, 0.5, 0.0, 0.0, 0.5},
                                    {-0.5, -0.25, 0.0, 0.25, 0.5, 0.0},
                                    {-1.0, -0.5, 0.0, 0.5, 1.0, 0.0},
                                    {-0.5, -3.0, 0.5, 0.0, 0.0, 3.0}};
    const Matrix6 K2 = element_elasticity_stiffness(mesh_of({0.0, 0.0, 2.0, 0.0, 0.0, 1.0}), 0, stretched());
    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            EXPECT_NEAR(K2[i][j], want_wide[i][j], 1e-14)
                << "triangle (0,0) (2,0) (0,1), Ke[" << i << "][" << j << "]";
        }
    }
}

// EXAMPLE 132. Ke is symmetric: Ke[i][j] = Ke[j][i]. A B^T D B always is if D is symmetric. Tried on six
// triangles and five materials (also stiff ones like steel, with nu < 0 and almost incompressible). An all-zero Ke
// would be symmetric without proving anything: that is why we first check that it is not zero.
TEST(ElementElasticityStiffness, IsSymmetric) {
    const std::vector<Tri> triangles = sample_triangles();
    const std::vector<Matrix3> materials = sample_materials();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        for (std::size_t m = 0; m < materials.size(); ++m) {
            const Matrix6 K = element_elasticity_stiffness(mesh, 0, materials[m]);
            const double big = largest(K);
            ASSERT_GT(big, 0.0) << "Ke is all zeros (triangle " << k << ", material " << m << ")";
            for (std::size_t i = 0; i < 6; ++i) {
                for (std::size_t j = i + 1; j < 6; ++j) {
                    EXPECT_NEAR(K[i][j], K[j][i], 1e-13 * big)
                        << "triangle " << k << ", material " << m << ", Ke[" << i << "][" << j << "]";
                }
            }
        }
    }
}

// EXAMPLE 133. A rigid motion does not deform, so it gives no forces: Ke u_rigid = 0. Four rigid motions: translation in
// x, translation in y, rotation by 1 about the origin (u = -y, v = x) and rotation by 0.5 about the point (0.7,
// -0.3), i.e. u = -theta (y - yc), v = theta (x - xc). Six triangles and three materials. They are three independent
// rigid motions, so Ke alone cannot be inverted: to solve a problem the constraints will be needed.
TEST(ElementElasticityStiffness, RigidMotionsProduceNoForce) {
    const LinearField rigid[4] = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                  {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
                                  {0.0, 0.0, -1.0, 0.0, 1.0, 0.0},
                                  {-0.15, 0.0, -0.5, -0.35, 0.5, 0.0}};
    const std::vector<Tri> triangles = sample_triangles();
    const std::vector<Matrix3> materials = sample_materials();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        for (std::size_t m = 0; m < 3; ++m) {
            const Matrix6 K = element_elasticity_stiffness(mesh, 0, materials[m]);
            ASSERT_GT(largest(K), 0.0) << "Ke is all zeros (triangle " << k << ", material " << m << ")";
            for (std::size_t r = 0; r < 4; ++r) {
                const Vector6 u = nodal_values(mesh, 0, rigid[r]);
                const Vector6 f = times(K, u);
                const double tol = 1e-13 * largest(K) * largest(u);
                for (std::size_t i = 0; i < 6; ++i) {
                    EXPECT_NEAR(f[i], 0.0, tol)
                        << "triangle " << k << ", material " << m << ", motion " << r << ", force " << i;
                }
            }
        }
    }
}

// EXAMPLE 134. The forces of a linear field. A linear field has strain eps = (a1, b2, a2 + b1) and stress
// sigma = D eps, constant. The forces that arise from it on the six degrees of freedom are A B^T sigma, i.e. for node i
//     f_x = (b_i sigma_xx + c_i sigma_xy) / 2,        f_y = (c_i sigma_yy + b_i sigma_xy) / 2,
// with b_i and c_i from the node coordinates. Ke u_e must give these forces. The first six fields are a basis: if Ke
// gives the right forces for all of them, it is right in each of the 36 entries. Tried on six triangles and two materials.
TEST(ElementElasticityStiffness, GivesTheForcesOfLinearFields) {
    const std::vector<Tri> triangles = sample_triangles();
    const std::vector<LinearField> fields = sample_fields();
    const Matrix3 materials[2] = {general(), plane_strain_matrix(1.0, 0.25)};
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        for (std::size_t m = 0; m < 2; ++m) {
            const Matrix6 K = element_elasticity_stiffness(mesh, 0, materials[m]);
            for (std::size_t r = 0; r < fields.size(); ++r) {
                const Vector6 u = nodal_values(mesh, 0, fields[r]);
                const Vector6 got = times(K, u);
                const Vector6 want = nodal_forces(mesh, 0, stress_of(materials[m], strain_of(fields[r])));
                const double tol = 1e-13 * largest(K) * largest(u);
                for (std::size_t i = 0; i < 6; ++i) {
                    EXPECT_NEAR(got[i], want[i], tol)
                        << "triangle " << k << ", material " << m << ", field " << r << ", force " << i;
                }
            }
        }
    }
}

// EXAMPLE 135. The energy of a linear field. With eps = (a1, b2, a2 + b1) constant in the whole triangle, the energy
// u_e^T Ke u_e is A eps^T D eps (the area A from the node coordinates). It does not use B or the formulas for b_i and c_i:
// it is the definition of Ke as a spring. It is 0 for a rigid motion and must be positive for every true deformation.
TEST(ElementElasticityStiffness, StrainEnergyOfLinearFieldsIsExact) {
    const std::vector<Tri> triangles = sample_triangles();
    const std::vector<LinearField> fields = sample_fields();
    const Matrix3 materials[2] = {general(), plane_strain_matrix(1.0, 0.25)};
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        for (std::size_t m = 0; m < 2; ++m) {
            const Matrix6 K = element_elasticity_stiffness(mesh, 0, materials[m]);
            for (std::size_t r = 0; r < fields.size(); ++r) {
                const Vector6 u = nodal_values(mesh, 0, fields[r]);
                const double want = triangle_area(mesh, 0) * strain_times_stress(materials[m], strain_of(fields[r]));
                const double tol = 1e-13 * largest(K) * largest(u) * largest(u);
                EXPECT_NEAR(energy_of(K, u), want, tol)
                    << "triangle " << k << ", material " << m << ", field " << r;
            }
        }
    }
}

// EXAMPLE 136. The stiffness does not depend on the size. If the triangle is scaled by s (and possibly shifted), A
// grows by s^2 and B^T B decreases by 1/s^2: Ke is the same. (In 2D and with thickness 1 a large plate is no stiffer
// than a small one: the force to move by d a side of length L is E (d / L) L = E d.) Tried with s = 1, 2, 0.25 and 10.
TEST(ElementElasticityStiffness, DoesNotDependOnTheSize) {
    const double scales[4] = {1.0, 2.0, 0.25, 10.0};
    const double shifts[2][2] = {{0.0, 0.0}, {3.0, -5.0}};
    const Matrix3 materials[2] = {general(), plane_strain_matrix(1.0, 0.3)};
    const std::vector<Tri> triangles = sample_triangles();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        for (std::size_t m = 0; m < 2; ++m) {
            const Matrix6 base = element_elasticity_stiffness(mesh_of(triangles[k]), 0, materials[m]);
            ASSERT_GT(largest(base), 0.0) << "Ke is all zeros (triangle " << k << ", material " << m << ")";
            for (double s : scales) {
                for (const auto& shift : shifts) {
                    const Tri moved = scaled_and_shifted(triangles[k], s, shift[0], shift[1]);
                    const Matrix6 K = element_elasticity_stiffness(mesh_of(moved), 0, materials[m]);
                    for (std::size_t i = 0; i < 6; ++i) {
                        for (std::size_t j = 0; j < 6; ++j) {
                            EXPECT_NEAR(K[i][j], base[i][j], 1e-10 * largest(base))
                                << "triangle " << k << ", material " << m << ", s = " << s << ", Ke[" << i << "]["
                                << j << "]";
                        }
                    }
                }
            }
        }
    }
}

// EXAMPLE 137. An isotropic material has no preferred direction. If the triangle is rotated by phi about
// the origin and the displacements of the nodes are rotated too (u' = R u at each node, i.e. u_e' = Q u_e), the forces
// rotate in the same way: f' = Q f. Since f = K u, this means K' = Q K Q^T. The true D of
// plane_strain_matrix is needed: the shear mu and the shear strain gamma = du/dy + dv/dx go together only with the right
// conventions (D[2][2] = mu), otherwise Ke would depend on the direction. Tried with 4 angles, six triangles and two nu.
TEST(ElementElasticityStiffness, IsIndifferentToRotation) {
    const double angles[4] = {0.3, 1.5707963267948966, -1.0, 3.0};
    const Matrix3 materials[2] = {plane_strain_matrix(1.0, 0.3), plane_strain_matrix(1.0, 0.25)};
    const std::vector<Tri> triangles = sample_triangles();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        for (std::size_t m = 0; m < 2; ++m) {
            const Matrix6 K = element_elasticity_stiffness(mesh_of(triangles[k]), 0, materials[m]);
            ASSERT_GT(largest(K), 0.0) << "Ke is all zeros (triangle " << k << ", material " << m << ")";
            for (double phi : angles) {
                const Matrix6 got = element_elasticity_stiffness(mesh_of(rotated(triangles[k], phi)), 0, materials[m]);
                const Matrix6 want = rotated_matrix(K, phi);
                for (std::size_t i = 0; i < 6; ++i) {
                    for (std::size_t j = 0; j < 6; ++j) {
                        EXPECT_NEAR(got[i][j], want[i][j], 1e-12 * largest(K))
                            << "triangle " << k << ", material " << m << ", phi = " << phi << ", Ke[" << i
                            << "][" << j << "]";
                    }
                }
            }
        }
    }
}

// EXAMPLE 138. Ke of a triangle of a real mesh, where the nodes are not 0, 1, 2 and the areas are different. In every
// triangle e, for a linear field with the values at the mesh nodes, the energy u_e^T Ke u_e must be
// area(e) eps^T D eps. On the rectangular mesh of 4 x 3 cells (24 triangles) the sum is the energy of the whole plate,
// i.e. (area of the plate) eps^T D eps = 2 x 1.5 x eps^T D eps. Also tried on 3 irregular hand-made triangles.
TEST(ElementElasticityStiffness, WorksOnEveryTriangleOfAMesh) {
    const LinearField field = {0.003, 1.2e-3, -0.7e-3, -0.002, 0.5e-3, 2.0e-3};
    const Matrix3 D = hooke_quarter();
    const double psi = strain_times_stress(D, strain_of(field));

    const Mesh2D irregular = irregular_mesh();
    for (std::size_t e = 0; e < irregular.num_triangles(); ++e) {
        const Matrix6 K = element_elasticity_stiffness(irregular, e, D);
        const double want = triangle_area(irregular, e) * psi;
        EXPECT_NEAR(energy_of(K, nodal_values(irregular, e, field)), want, 1e-12 * want)
            << "irregular mesh, triangle " << e;
    }

    const Mesh2D rectangle = rectangle_mesh(2.0, 1.5, 4, 3);
    ASSERT_EQ(rectangle.num_triangles(), 24u);
    double total = 0.0;
    for (std::size_t e = 0; e < rectangle.num_triangles(); ++e) {
        const Matrix6 K = element_elasticity_stiffness(rectangle, e, D);
        const double energy = energy_of(K, nodal_values(rectangle, e, field));
        EXPECT_NEAR(energy, triangle_area(rectangle, e) * psi, 1e-12 * triangle_area(rectangle, e) * psi)
            << "rectangular mesh, triangle " << e;
        total += energy;
    }
    EXPECT_NEAR(total, 2.0 * 1.5 * psi, 1e-12 * 2.0 * 1.5 * psi);
}

// EXAMPLE 139. A triangle that does not exist: std::out_of_range, as for B and for the other functions of the element.
TEST(ElementElasticityStiffness, RejectsMissingTriangle) {
    const Mesh2D mesh = irregular_mesh();
    const Matrix3 D = hooke_quarter();
    EXPECT_NO_THROW(element_elasticity_stiffness(mesh, 0, D));
    EXPECT_NO_THROW(element_elasticity_stiffness(mesh, mesh.num_triangles() - 1, D));
    EXPECT_THROW(element_elasticity_stiffness(mesh, mesh.num_triangles(), D), std::out_of_range);
    EXPECT_THROW(element_elasticity_stiffness(mesh, 1000, D), std::out_of_range);
}

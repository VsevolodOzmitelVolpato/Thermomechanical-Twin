#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "Elasticity2D.h"
#include "Mesh2D.h"

// ---- The thermal force of the triangle: f_th = A B^T D eps_th ------------------------------------------------------
//
// A heated spring would like to lengthen by x_th. If it is free it does so and no force arises; if it is brought to x the
// force is k (x - x_th) = k x - k x_th. The piece k x_th is the "thermal force": it is the force that would be needed to
// stretch the cold spring by x_th.
// For the triangle it is the same. The strain is eps = B u_e, the one that heat would like is eps_th, and the stress
// arises from the difference, sigma = D (eps - eps_th). The energy, for the area A of the triangle, is
//     U = (1/2) A (B u_e - eps_th)^T D (B u_e - eps_th)
// and its derivative with respect to the 6 displacements, i.e. the 6 forces on the degrees of freedom, is
//     A B^T D (B u_e - eps_th) = Ke u_e - f_th,        with        f_th = A B^T D eps_th.
// So f_th is the formula A B^T sigma (the forces of a constant stress) with the stress D eps_th, the one that
// heat would like to produce. A heated, free triangle with no external forces moves so that Ke u_e = f_th.
// The f_th push every node outwards (the triangle wants to expand) but they are in equilibrium with each other.
//
// What a real thermal force must have, and this is what the tests check:
//   - it is the A B^T sigma of sigma = D eps_th, and it is linear in eps_th: no heat, no force;
//   - a constant thermal strain does not push the triangle and does not make it rotate: the sum of the forces and their
//     moment are 0 (a rigid motion does no work);
//   - the work of the f_th on a linear field u is A eps(u)^T D eps_th;
//   - if the triangle expands freely, with eps = eps_th and therefore sigma = 0, the elastic forces Ke u_e are
//     exactly the f_th (which is the same as saying that Ke u_th = f_th, as for the spring);
//   - it is "stress times length": enlarging the triangle by s multiplies the forces by s (the stiffness Ke, instead, does not
//     change);
//   - an isotropic material has no preferred direction: rotating triangle and strain, the forces rotate;
//   - on a whole mesh, a uniform heating does not load the interior nodes (the forces of the surrounding triangles
//     cancel): only the force on the boundary remains, sigma times the outward normal times the length.

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

// Six counter-clockwise triangles, of very different shapes and sizes (the same as in the B and Ke tests).
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

// Test thermal strains (eps_xx, eps_yy, gamma_xy): a unit one equal in x and y, a real one, (1 + nu) alpha dT
// with nu = 0.3, alpha = 12e-6 and dT = 50, a cooling and two general ones with shear. (The heat of an isotropic
// material gives no shear, but the function holds for any eps_th and the shear shows whether the indices of D and B
// are right.)
std::vector<Vector3> sample_thermal_strains() {
    return {{1.0, 1.0, 0.0},
            {7.8e-4, 7.8e-4, 0.0},
            {-3.0e-3, -3.0e-3, 0.0},
            {2.0e-3, -1.0e-3, 1.5e-3},
            {1.0, 2.0, 1.0}};
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
// plus an arbitrary field with all the numbers different from 0.
std::vector<LinearField> sample_fields() {
    return {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
            {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
            {0.003, 1.2e-3, -0.7e-3, -0.002, 0.5e-3, 2.0e-3}};
}

// The field of a free expansion: it starts from the origin and has constant strain eps = (eps_xx, eps_yy, gamma_xy),
//     u = eps_xx x + (gamma / 2) y,        v = eps_yy y + (gamma / 2) x.
// With eps = (e, e, 0) it is u = e x, v = e y.
LinearField free_expansion(const Vector3& eps) {
    return {0.0, eps[0], 0.5 * eps[2], 0.0, 0.5 * eps[2], eps[1]};
}

// The strain of a linear field, the stress D eps and the scalar product of two triples.
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
double product(const Vector3& a, const Vector3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
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

// The largest of the numbers of a table or of a vector in absolute value, to set the tolerances.
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
double largest(const Vector3& u) {
    return std::max(std::max(std::fabs(u[0]), std::fabs(u[1])), std::fabs(u[2]));
}

// The product Ke u and the work u . f of the forces f on the displacements u.
Vector6 times(const Matrix6& K, const Vector6& u) {
    Vector6 f{};
    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            f[i] += K[i][j] * u[j];
        }
    }
    return f;
}
double work_of(const Vector6& u, const Vector6& f) {
    double s = 0.0;
    for (std::size_t i = 0; i < 6; ++i) {
        s += u[i] * f[i];
    }
    return s;
}

// The forces f of a triangle rotated by phi: at each node (f_x, f_y) becomes (c f_x - s f_y, s f_x + c f_y).
Vector6 rotated_forces(const Vector6& f, double phi) {
    const double c = std::cos(phi);
    const double s = std::sin(phi);
    Vector6 r{};
    for (std::size_t n = 0; n < 3; ++n) {
        r[2 * n] = c * f[2 * n] - s * f[2 * n + 1];
        r[2 * n + 1] = s * f[2 * n] + c * f[2 * n + 1];
    }
    return r;
}

// The strain (eps_xx, eps_yy, gamma_xy) seen from axes rotated by phi, i.e. R eps R^T with eps the tensor
// [[eps_xx, g], [g, eps_yy]] and g = gamma / 2 (the "engineering" shear strain is twice the tensor). With
// c = cos(phi) and s = sin(phi):
//     eps_xx' = c^2 eps_xx + s^2 eps_yy - 2 s c g,        eps_yy' = s^2 eps_xx + c^2 eps_yy + 2 s c g,
//     gamma'  = 2 (s c (eps_xx - eps_yy) + (c^2 - s^2) g).
Vector3 rotated_strain(const Vector3& eps, double phi) {
    const double c = std::cos(phi);
    const double s = std::sin(phi);
    const double g = 0.5 * eps[2];
    return {c * c * eps[0] + s * s * eps[1] - 2.0 * s * c * g, s * s * eps[0] + c * c * eps[1] + 2.0 * s * c * g,
            2.0 * (s * c * (eps[0] - eps[1]) + (c * c - s * s) * g)};
}

}  // namespace

// EXAMPLE 140. By hand. The triangle (0,0), (1,0), (0,1) has A = 1/2 and (see the matrix B)
//     B = [[-1,  0, 1, 0, 0, 0],
//          [ 0, -1, 0, 0, 0, 1],
//          [-1, -1, 0, 1, 1, 0]].
// With E = 1 and nu = 1/4 (lambda = mu = 0.4) D = [[1.2, 0.4, 0], [0.4, 1.2, 0], [0, 0, 0.4]]. For eps_th = (1, 1, 0)
// the stress is D eps_th = (1.2 + 0.4, 0.4 + 1.2, 0) = (1.6, 1.6, 0), and the forces are f_th = A B^T (1.6, 1.6, 0): the
// degree of freedom j has A (B[0][j] 1.6 + B[1][j] 1.6 + B[2][j] 0), i.e.
//     f_th = 1/2 (-1.6, -1.6, 1.6, 0, 0, 1.6) = (-0.8, -0.8, 0.8, 0, 0, 0.8).
// The signs make sense: node 0, at (0,0), is pushed towards (-, -), node 1, at (1,0), towards +x and node 2, at
// (0,1), towards +y: every node is pushed outwards, as if the triangle wanted to expand.
// The second triangle, (0,0), (2,0), (0,1) with A = 1, has a material with different xx and yy, D = [[2, 1, 0], [1, 3, 0],
// [0, 0, 1]], and a general strain with shear, eps_th = (1, 2, 1): D eps_th = (4, 7, 1) and, with
//     B = [[-0.5, 0, 0.5, 0, 0, 0], [0, -1, 0, 0, 0, 1], [-1, -0.5, 0, 0.5, 1, 0]],
//     f_th = A B^T (4, 7, 1) = (-3, -7.5, 2, 0.5, 1, 7).
TEST(ElementThermalLoad, MatchesHandComputation) {
    const double want_unit[6] = {-0.8, -0.8, 0.8, 0.0, 0.0, 0.8};
    const Vector6 f1 =
        element_thermal_load(mesh_of({0.0, 0.0, 1.0, 0.0, 0.0, 1.0}), 0, hooke_quarter(), {1.0, 1.0, 0.0});
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(f1[i], want_unit[i], 1e-14) << "triangle (0,0) (1,0) (0,1), f[" << i << "]";
    }

    const double want_wide[6] = {-3.0, -7.5, 2.0, 0.5, 1.0, 7.0};
    const Vector6 f2 =
        element_thermal_load(mesh_of({0.0, 0.0, 2.0, 0.0, 0.0, 1.0}), 0, stretched(), {1.0, 2.0, 1.0});
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(f2[i], want_wide[i], 1e-14) << "triangle (0,0) (2,0) (0,1), f[" << i << "]";
    }
}

// EXAMPLE 141. The forces of a constant stress. The stress that heat would like to produce is sigma = D eps_th,
// constant in the triangle, and the forces are A B^T sigma, i.e. for node i
//     f_x = (b_i sigma_xx + c_i sigma_xy) / 2,        f_y = (c_i sigma_yy + b_i sigma_xy) / 2,
// with b_i and c_i from the node coordinates (here they are written out, without B). Tried on six triangles, five
// materials and five thermal strains, and on the three triangles of an irregular mesh (with shared nodes and different
// areas, 2, 1.75 and 1.5), where triangle e is not always number 0. Without heat (eps_th = 0) the forces are 0,
// exactly.
TEST(ElementThermalLoad, GivesTheForcesOfAConstantStress) {
    const std::vector<Tri> triangles = sample_triangles();
    const std::vector<Matrix3> materials = sample_materials();
    const std::vector<Vector3> strains = sample_thermal_strains();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        for (std::size_t m = 0; m < materials.size(); ++m) {
            for (std::size_t s = 0; s < strains.size(); ++s) {
                const Vector6 got = element_thermal_load(mesh, 0, materials[m], strains[s]);
                const Vector6 want = nodal_forces(mesh, 0, stress_of(materials[m], strains[s]));
                ASSERT_GT(largest(want), 0.0) << "triangle " << k << ", material " << m << ", eps_th " << s;
                for (std::size_t i = 0; i < 6; ++i) {
                    EXPECT_NEAR(got[i], want[i], 1e-13 * largest(want))
                        << "triangle " << k << ", material " << m << ", eps_th " << s << ", force " << i;
                }
            }
        }
        for (std::size_t m = 0; m < materials.size(); ++m) {
            const Vector6 none = element_thermal_load(mesh, 0, materials[m], {0.0, 0.0, 0.0});
            for (std::size_t i = 0; i < 6; ++i) {
                EXPECT_EQ(none[i], 0.0) << "no heat, triangle " << k << ", material " << m << ", force " << i;
            }
        }
    }

    const Mesh2D irregular = irregular_mesh();
    for (std::size_t e = 0; e < irregular.num_triangles(); ++e) {
        for (std::size_t s = 0; s < strains.size(); ++s) {
            const Vector6 got = element_thermal_load(irregular, e, materials[0], strains[s]);
            const Vector6 want = nodal_forces(irregular, e, stress_of(materials[0], strains[s]));
            for (std::size_t i = 0; i < 6; ++i) {
                EXPECT_NEAR(got[i], want[i], 1e-13 * largest(want))
                    << "irregular mesh, triangle " << e << ", eps_th " << s << ", force " << i;
            }
        }
    }
}

// EXAMPLE 142. The work on a linear field. A linear field u has strain eps = (a1, b2, a2 + b1), constant, and
// the forces f_th do on it the work u_e . f_th = A eps^T D eps_th (the area A from the node coordinates). It does not use
// B or the formulas for b_i and c_i. For a rigid motion (eps = 0) the work is 0. The first six fields are a basis: if
// the work is right for all of them, the forces are right in each of the 6 numbers. Tried on six triangles, two materials and
// five thermal strains.
TEST(ElementThermalLoad, WorkOfLinearFieldsIsExact) {
    const std::vector<Tri> triangles = sample_triangles();
    const std::vector<LinearField> fields = sample_fields();
    const std::vector<Vector3> strains = sample_thermal_strains();
    const Matrix3 materials[2] = {general(), plane_strain_matrix(1.0, 0.25)};
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        for (std::size_t m = 0; m < 2; ++m) {
            for (std::size_t s = 0; s < strains.size(); ++s) {
                const Vector6 f = element_thermal_load(mesh, 0, materials[m], strains[s]);
                ASSERT_GT(largest(f), 0.0) << "triangle " << k << ", material " << m << ", eps_th " << s;
                for (std::size_t r = 0; r < fields.size(); ++r) {
                    const Vector6 u = nodal_values(mesh, 0, fields[r]);
                    const double want =
                        triangle_area(mesh, 0) * product(strain_of(fields[r]), stress_of(materials[m], strains[s]));
                    EXPECT_NEAR(work_of(u, f), want, 1e-13 * largest(f) * largest(u))
                        << "triangle " << k << ", material " << m << ", eps_th " << s << ", field " << r;
                }
            }
        }
    }
}

// EXAMPLE 143. Equilibrium. A constant thermal strain does not push the triangle and does not make it rotate: the three forces
// sum to zero in x and in y, and their moment, the sum of x_i f_y - y_i f_x, is 0 (which is the same as saying that a rigid
// motion does no work). It depends only on sigma being constant and symmetric: summing b_i and c_i over the three nodes
// gives 0. An all-zero force would be in equilibrium without proving anything: that is why we first check that it is not
// zero. Tried on six triangles, five materials and five thermal strains.
TEST(ElementThermalLoad, IsInEquilibrium) {
    const std::vector<Tri> triangles = sample_triangles();
    const std::vector<Matrix3> materials = sample_materials();
    const std::vector<Vector3> strains = sample_thermal_strains();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Tri& t = triangles[k];
        const Mesh2D mesh = mesh_of(t);
        double size = 0.0;
        for (double v : {t.x0, t.y0, t.x1, t.y1, t.x2, t.y2}) {
            size = std::max(size, std::fabs(v));
        }
        for (std::size_t m = 0; m < materials.size(); ++m) {
            for (std::size_t s = 0; s < strains.size(); ++s) {
                const Vector6 f = element_thermal_load(mesh, 0, materials[m], strains[s]);
                ASSERT_GT(largest(f), 0.0) << "triangle " << k << ", material " << m << ", eps_th " << s;
                const double tol = 1e-13 * largest(f);
                EXPECT_NEAR(f[0] + f[2] + f[4], 0.0, tol)
                    << "sum in x, triangle " << k << ", material " << m << ", eps_th " << s;
                EXPECT_NEAR(f[1] + f[3] + f[5], 0.0, tol)
                    << "sum in y, triangle " << k << ", material " << m << ", eps_th " << s;
                const double moment = t.x0 * f[1] - t.y0 * f[0] + t.x1 * f[3] - t.y1 * f[2] + t.x2 * f[5] - t.y2 * f[4];
                EXPECT_NEAR(moment, 0.0, tol * size)
                    << "moment, triangle " << k << ", material " << m << ", eps_th " << s;
            }
        }
    }
}

// EXAMPLE 144. Free expansion. If the triangle expands freely, u = e x, v = e y (in general the field with
// strain eps = eps_th), the stress D (B u - eps_th) is 0 and there are no elastic forces: Ke u_e = f_th, i.e. the
// f_th are the forces that, on a cold triangle, would give the same displacement (like k x_th for the spring). It puts
// Ke and f_th together: if f_th has the wrong sign or factor (or if the material D is not used in the same way),
// the free expansion is no longer in equilibrium. Tried on six triangles, three materials and five thermal
// strains.
TEST(ElementThermalLoad, FreeExpansionBalancesTheStiffness) {
    const std::vector<Tri> triangles = sample_triangles();
    const std::vector<Vector3> strains = sample_thermal_strains();
    const Matrix3 materials[3] = {general(), plane_strain_matrix(1.0, 0.25), plane_strain_matrix(210e9, 0.3)};
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        const Mesh2D mesh = mesh_of(triangles[k]);
        for (std::size_t m = 0; m < 3; ++m) {
            const Matrix6 K = element_elasticity_stiffness(mesh, 0, materials[m]);
            for (std::size_t s = 0; s < strains.size(); ++s) {
                const Vector6 f = element_thermal_load(mesh, 0, materials[m], strains[s]);
                ASSERT_GT(largest(f), 0.0) << "triangle " << k << ", material " << m << ", eps_th " << s;
                const Vector6 u = nodal_values(mesh, 0, free_expansion(strains[s]));
                const Vector6 got = times(K, u);
                for (std::size_t i = 0; i < 6; ++i) {
                    EXPECT_NEAR(got[i], f[i], 1e-13 * largest(K) * largest(u))
                        << "triangle " << k << ", material " << m << ", eps_th " << s << ", force " << i;
                }
            }
        }
    }
}

// EXAMPLE 145. The thermal forces are "stress times length". If the triangle is scaled by s (and possibly
// shifted), A grows by s^2 and B decreases by 1/s: f_th = A B^T D eps_th grows by s. (In 2D and with thickness 1 the force of a
// stress sigma on a side of length L is sigma L.) It is the opposite of Ke, which does not depend on the size. Tried with
// s = 1, 2, 0.25 and 10, two displacements, six triangles and two materials.
TEST(ElementThermalLoad, ScalesWithTheSizeOfTheTriangle) {
    const double scales[4] = {1.0, 2.0, 0.25, 10.0};
    const double shifts[2][2] = {{0.0, 0.0}, {3.0, -5.0}};
    const Matrix3 materials[2] = {general(), plane_strain_matrix(1.0, 0.3)};
    const Vector3 eps = {2.0e-3, -1.0e-3, 1.5e-3};
    const std::vector<Tri> triangles = sample_triangles();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        for (std::size_t m = 0; m < 2; ++m) {
            const Vector6 base = element_thermal_load(mesh_of(triangles[k]), 0, materials[m], eps);
            ASSERT_GT(largest(base), 0.0) << "triangle " << k << ", material " << m;
            for (double s : scales) {
                for (const auto& shift : shifts) {
                    const Tri moved = scaled_and_shifted(triangles[k], s, shift[0], shift[1]);
                    const Vector6 f = element_thermal_load(mesh_of(moved), 0, materials[m], eps);
                    for (std::size_t i = 0; i < 6; ++i) {
                        EXPECT_NEAR(f[i], s * base[i], 1e-10 * s * largest(base))
                            << "triangle " << k << ", material " << m << ", s = " << s << ", force " << i;
                    }
                }
            }
        }
    }
}

// EXAMPLE 146. An isotropic material has no preferred direction. If the triangle is rotated by phi about
// the origin and the thermal strain is rotated too (eps' = R eps R^T, with the shear gamma = 2 eps_xy), the
// forces rotate in the same way: f' = Q f, at each node (f_x, f_y) -> R (f_x, f_y). The true D of
// plane_strain_matrix is needed: D[2][2] = mu links shear stress and shear strain only with the right conventions, otherwise the
// stress D eps_th, and therefore the forces, would depend on the direction. An expansion equal in x and y is invariant under
// rotation; the second strain has shear and changes. Tried with 4 angles, six triangles and two nu.
TEST(ElementThermalLoad, IsIndifferentToRotation) {
    const double angles[4] = {0.3, 1.5707963267948966, -1.0, 3.0};
    const Matrix3 materials[2] = {plane_strain_matrix(1.0, 0.3), plane_strain_matrix(1.0, 0.25)};
    const Vector3 strains[2] = {{2.0e-3, 2.0e-3, 0.0}, {2.0e-3, -1.0e-3, 1.5e-3}};
    const std::vector<Tri> triangles = sample_triangles();
    for (std::size_t k = 0; k < triangles.size(); ++k) {
        for (std::size_t m = 0; m < 2; ++m) {
            for (std::size_t s = 0; s < 2; ++s) {
                const Vector6 base = element_thermal_load(mesh_of(triangles[k]), 0, materials[m], strains[s]);
                ASSERT_GT(largest(base), 0.0) << "triangle " << k << ", material " << m << ", eps_th " << s;
                for (double phi : angles) {
                    const Vector6 got = element_thermal_load(mesh_of(rotated(triangles[k], phi)), 0, materials[m],
                                                             rotated_strain(strains[s], phi));
                    const Vector6 want = rotated_forces(base, phi);
                    for (std::size_t i = 0; i < 6; ++i) {
                        EXPECT_NEAR(got[i], want[i], 1e-12 * largest(base))
                            << "triangle " << k << ", material " << m << ", eps_th " << s << ", phi = " << phi
                            << ", force " << i;
                    }
                }
            }
        }
    }
}

// EXAMPLE 147. A whole mesh. The forces of the triangles are summed at each node (the x degree of freedom of node n is
// 2 n, the y one is 2 n + 1) for a uniform heating, with the same D and the same eps_th in all the triangles. By
// the divergence theorem the sum at node n is sigma g_n, with sigma = D eps_th and
//     g_n = sum, over the boundary sides that touch n, of (outward normal) x (half the side length):
// the interior nodes have no boundary side and receive no forces (the forces of the surrounding triangles cancel);
// a node on the right boundary, far from the corners, has g = (hy, 0) (two sides of length hy, each counts for half); a
// corner has one side per direction and g = (+-hy/2, +-hx/2). On the 2 x 1.5 rectangular mesh with 4 x 3 cells (24 triangles,
// hx = 0.5 and hy = 0.5).
TEST(ElementThermalLoad, UniformHeatingLoadsOnlyTheBoundaryOfAMesh) {
    const double lx = 2.0;
    const double ly = 1.5;
    const std::size_t nx = 4;
    const std::size_t ny = 3;
    const Mesh2D mesh = rectangle_mesh(lx, ly, nx, ny);
    const Matrix3 D = general();
    const Vector3 eps = {2.0e-3, -1.0e-3, 1.5e-3};
    const Vector3 sigma = stress_of(D, eps);

    std::vector<double> F(2 * mesh.num_nodes(), 0.0);
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Vector6 f = element_thermal_load(mesh, e, D, eps);
        const Triangle& t = mesh.triangle(e);
        for (std::size_t i = 0; i < 3; ++i) {
            F[2 * t[i]] += f[2 * i];
            F[2 * t[i] + 1] += f[2 * i + 1];
        }
    }

    const double hx = lx / static_cast<double>(nx);
    const double hy = ly / static_cast<double>(ny);
    const double edge = 1e-9 * lx;
    const double tol = 1e-13 * largest(sigma) * std::max(hx, hy);
    double sum_x = 0.0;
    double sum_y = 0.0;
    for (std::size_t n = 0; n < mesh.num_nodes(); ++n) {
        const bool left = std::fabs(mesh.x(n)) < edge;
        const bool right = std::fabs(mesh.x(n) - lx) < edge;
        const bool bottom = std::fabs(mesh.y(n)) < edge;
        const bool top = std::fabs(mesh.y(n) - ly) < edge;
        const double share = ((left || right) && (bottom || top)) ? 0.5 : 1.0;  // a corner has only one side per direction
        const double gx = ((right ? 1.0 : 0.0) - (left ? 1.0 : 0.0)) * hy * share;
        const double gy = ((top ? 1.0 : 0.0) - (bottom ? 1.0 : 0.0)) * hx * share;
        EXPECT_NEAR(F[2 * n], sigma[0] * gx + sigma[2] * gy, tol)
            << "node " << n << " (" << mesh.x(n) << ", " << mesh.y(n) << "), force in x";
        EXPECT_NEAR(F[2 * n + 1], sigma[2] * gx + sigma[1] * gy, tol)
            << "node " << n << " (" << mesh.x(n) << ", " << mesh.y(n) << "), force in y";
        sum_x += F[2 * n];
        sum_y += F[2 * n + 1];
    }
    EXPECT_NEAR(sum_x, 0.0, tol * static_cast<double>(mesh.num_nodes()));
    EXPECT_NEAR(sum_y, 0.0, tol * static_cast<double>(mesh.num_nodes()));
}

// EXAMPLE 148. A triangle that does not exist: std::out_of_range, as for B and for Ke.
TEST(ElementThermalLoad, RejectsMissingTriangle) {
    const Mesh2D mesh = irregular_mesh();
    const Matrix3 D = hooke_quarter();
    const Vector3 eps = {1.0, 1.0, 0.0};
    EXPECT_NO_THROW(element_thermal_load(mesh, 0, D, eps));
    EXPECT_NO_THROW(element_thermal_load(mesh, mesh.num_triangles() - 1, D, eps));
    EXPECT_THROW(element_thermal_load(mesh, mesh.num_triangles(), D, eps), std::out_of_range);
    EXPECT_THROW(element_thermal_load(mesh, 1000, D, eps), std::out_of_range);
}

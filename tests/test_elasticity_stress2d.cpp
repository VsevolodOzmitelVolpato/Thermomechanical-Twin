#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "CSR_and_Vectors.h"
#include "Elasticity2D.h"
#include "Mesh2D.h"

// ---- From displacements to stresses: sigma = D (B u - eps_th) ------------------------------------------------------
//
// With the displacements u of the plate (solve_displacement_2d) the stress is computed in every triangle. Inside a
// triangle the displacement is linear, so the strain B u_e is constant, and so is the stress: three numbers
// (sigma_xx, sigma_yy, sigma_xy) per triangle. As in 1D (sigma = E (u' - alpha dT)) the stress arises only from the
// part of the strain that is NOT due to heat:
//     sigma = D (B u_e - eps_th),        eps_th = (1 + nu) alpha dT (1, 1, 0).
// There are two functions.
//   - element_stress_2d(mesh, e, D, u_e, eps_th) is the stress of ONE triangle: it needs its six displacements
//     (u0, v0, u1, v1, u2, v2) and its thermal strain.
//   - stress_field_2d(mesh, D, nu, alpha, u, T, T_ref) is that of the WHOLE mesh: for each triangle it takes from
//     the 2 n numbers of u its six displacements (the local i is the global 2 t[i / 2] + i % 2), computes dT as the
//     mean of the temperatures of its three nodes minus T_ref (as assemble_thermal_load_2d) and calls the triangle function.
// What they must do, and this is what the tests check:
//   - the stress of a triangle is computed by hand (EXAMPLE 181), and for every linear field, every shape of triangle and
//     every D, even a non-symmetric one (EXAMPLE 182); a nonexistent triangle throws std::out_of_range (EXAMPLE 183);
//   - on the unit square the stresses of the problems of the previous steps are computed by hand (EXAMPLE 184);
//   - a linear field has the same stress in all the triangles, on every mesh (EXAMPLE 185); the heated plate that is
//     clamped has stress -D eps_th, the free one none (EXAMPLE 186);
//   - each triangle uses the mean of the temperatures of its nodes (EXAMPLE 187) and its own nodes (EXAMPLE 188);
//   - the forces on the nodes from the stresses are K u - F (EXAMPLE 189), and in a free heated body they are all 0
//     (EXAMPLE 190): the stresses are in equilibrium;
//   - the result is linear in displacements, temperature and material (EXAMPLE 191) and does not read the nodes
//     outside every triangle (EXAMPLE 192); wrong data throw std::invalid_argument (EXAMPLE 193);
//   - together with solve_displacement_2d: the stretching of a rectangle (EXAMPLE 194), the plate heated between two walls,
//     with the textbook stress sigma_xx = -E alpha dT / (1 - nu) (EXAMPLE 195), and the same plate in steel with 40 x
//     40 cells (EXAMPLE 196).

namespace {

// A rectangle lx x ly divided into nx x ny cells, as in rectangle_mesh.
struct Grid {
    double lx, ly;
    std::size_t nx, ny;
};

// Six rectangles of very different shapes (the same as in the displacement tests).
std::vector<Grid> sample_grids() {
    return {{1.0, 1.0, 1, 1},    // the unit square: two triangles
            {2.0, 1.0, 2, 1},    // two cells
            {3.0, 1.5, 4, 3},    // 0.75 x 0.5 cells
            {1.0, 1.0, 6, 6},    // 72 triangles
            {0.2, 0.05, 5, 2},   // small (20 cm by 5 cm, in meters)
            {10.0, 0.1, 1, 7}};  // very flat: 10 x 0.0143 cells
}

// The unit square divided along the diagonal from (1, 0) to (0, 1): nodes 0 = (0, 0), 1 = (1, 0), 2 = (0, 1), 3 =
// (1, 1) and triangles (0, 1, 2) and (1, 3, 2), both with area 1/2.
Mesh2D unit_square() {
    return Mesh2D({0.0, 1.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {1, 3, 2}});
}

// The large triangle with node 3 inside (area 9/2), which is not a rectangle.
Mesh2D triangle_with_interior_node() {
    return Mesh2D({0.0, 3.0, 0.0, 1.0}, {0.0, 0.0, 3.0, 1.0}, {{0, 1, 3}, {1, 2, 3}, {2, 0, 3}});
}

// An irregular quadrilateral with a node inside: 0 = (0, 0), 1 = (4, 0), 2 = (3, 3), 3 = (-1, 2) and, inside,
// 4 = (1, 0.7). Four triangles around node 4, all of different shape and area.
Mesh2D irregular_quadrilateral() {
    return Mesh2D({0.0, 4.0, 3.0, -1.0, 1.0}, {0.0, 0.0, 3.0, 2.0, 0.7},
                  {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}});
}

// Two triangles of DIFFERENT area: 0 = (0, 0), 1 = (4, 0), 2 = (1, 2), 3 = (6, 4); (0, 1, 2) has A = 4 and (1, 3, 2) has
// A = 8.
Mesh2D two_triangles_of_different_area() {
    return Mesh2D({0.0, 4.0, 1.0, 6.0}, {0.0, 0.0, 2.0, 4.0}, {{0, 1, 2}, {1, 3, 2}});
}

// Six nodes and only two triangles, (1, 2, 3) and (1, 3, 4): nodes 0 = (5, 5) and 5 = (7, 7) are in no
// triangle. The other four are the unit square: 1 = (0, 0), 2 = (1, 0), 3 = (1, 1), 4 = (0, 1).
Mesh2D mesh_with_nodes_outside_every_triangle() {
    return Mesh2D({5.0, 0.0, 1.0, 1.0, 0.0, 7.0}, {5.0, 0.0, 0.0, 1.0, 1.0, 7.0}, {{1, 2, 3}, {1, 3, 4}});
}

// The meshes in which every node is in at least one triangle: six rectangles and three meshes that are not rectangles.
std::vector<Mesh2D> sample_meshes() {
    std::vector<Mesh2D> meshes;
    for (const Grid& g : sample_grids()) {
        meshes.push_back(rectangle_mesh(g.lx, g.ly, g.nx, g.ny));
    }
    meshes.push_back(triangle_with_interior_node());
    meshes.push_back(irregular_quadrilateral());
    meshes.push_back(two_triangles_of_different_area());
    return meshes;
}

// Hand-made materials (3 x 3, indices xx, yy, xy), symmetric and positive definite.
Matrix3 hooke_quarter() {  // E = 1, nu = 1/4: lambda = mu = 0.4
    return {{{1.2, 0.4, 0.0}, {0.4, 1.2, 0.0}, {0.0, 0.0, 0.4}}};
}
Matrix3 general() {  // all the numbers different, also the coupling with shear
    return {{{2.0, 0.5, 0.1}, {0.5, 3.0, -0.2}, {0.1, -0.2, 1.0}}};
}
Matrix3 steel() {  // a real, stiff material: numbers of order 1e11
    return plane_strain_matrix(210e9, 0.3);
}
// A D that is NOT symmetric (no real material is, but the function uses D as it receives it): the rows matter, not
// the columns.
Matrix3 nonsymmetric() {
    return {{{2.0, 0.5, 0.1}, {0.7, 3.0, -0.2}, {0.3, -0.4, 1.0}}};
}

// A material for the thermal force: the matrix D, the Poisson ratio nu and the expansion coefficient alpha
// (D and nu reach the functions separately, so non-physical materials are tried too: negative nu and alpha).
struct Material {
    Matrix3 D;
    double nu;
    double alpha;
};

std::vector<Material> sample_materials() {
    return {{hooke_quarter(), 0.25, 1.0e-3},  // E = 1, nu = 1/4
            {general(), -0.3, -2.0e-3},       // any D, negative nu and negative alpha
            {steel(), 0.3, 12.0e-6}};         // steel: E = 210 GPa, nu = 0.3, alpha = 12e-6 per kelvin
}

// The reference temperature, the one with no thermal expansion (20 degrees Celsius in kelvin).
const double reference_temperature = 293.15;

// The largest absolute value of a vector, of three numbers and of a 3 x 3 table.
double largest_entry(const std::vector<double>& v) {
    double m = 0.0;
    for (double a : v) {
        m = std::max(m, std::fabs(a));
    }
    return m;
}
double largest_entry(const Vector3& v) {
    return std::max(std::fabs(v[0]), std::max(std::fabs(v[1]), std::fabs(v[2])));
}
double largest_entry(const Matrix3& D) {
    return std::max(largest_entry(D[0]), std::max(largest_entry(D[1]), largest_entry(D[2])));
}
// The largest absolute value of all the stresses.
double largest_entry(const std::vector<Vector3>& sigma) {
    double m = 0.0;
    for (const Vector3& s : sigma) {
        m = std::max(m, largest_entry(s));
    }
    return m;
}

// The larger dimension of the mesh (the larger of the width and the height).
double mesh_size(const Mesh2D& mesh) {
    double x_min = mesh.x(0);
    double x_max = mesh.x(0);
    double y_min = mesh.y(0);
    double y_max = mesh.y(0);
    for (std::size_t I = 1; I < mesh.num_nodes(); ++I) {
        x_min = std::min(x_min, mesh.x(I));
        x_max = std::max(x_max, mesh.x(I));
        y_min = std::min(y_min, mesh.y(I));
        y_max = std::max(y_max, mesh.y(I));
    }
    return std::max(x_max - x_min, y_max - y_min);
}

// The size of the smallest triangle: the square root of the area.
double smallest_size(const Mesh2D& mesh) {
    double s = std::sqrt(mesh.area(0));
    for (std::size_t e = 1; e < mesh.num_triangles(); ++e) {
        s = std::min(s, std::sqrt(mesh.area(e)));
    }
    return s;
}

// The node farthest from node p.
std::size_t farthest_node(const Mesh2D& mesh, std::size_t p) {
    std::size_t q = p;
    double best = -1.0;
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        const double d = std::hypot(mesh.x(I) - mesh.x(p), mesh.y(I) - mesh.y(p));
        if (d > best) {
            best = d;
            q = I;
        }
    }
    return q;
}

// Three degrees of freedom that block the three rigid motions, and nothing more: u and v of node p (the two
// translations) and one component of the node q farthest from p (the rotation). A rotation by theta about p moves q by
// theta (-(y_q - y_p), x_q - x_p): v is fixed if x_q - x_p is the larger difference in absolute value (so that the
// rotation would have to move v by a number different from 0), otherwise u.
std::vector<std::size_t> isostatic_dofs(const Mesh2D& mesh, std::size_t p) {
    const std::size_t q = farthest_node(mesh, p);
    const double dx = mesh.x(q) - mesh.x(p);
    const double dy = mesh.y(q) - mesh.y(p);
    return {2 * p, 2 * p + 1, std::fabs(dx) >= std::fabs(dy) ? 2 * q + 1 : 2 * q};
}

// A linear displacement field:  u = a0 + a1 x + a2 y,  v = b0 + b1 x + b2 y.
// At every point it has the same strain (eps_xx, eps_yy, gamma_xy) = (a1, b2, a2 + b1).
struct LinearField {
    double a0, a1, a2, b0, b1, b2;
};

// The strain (eps_xx, eps_yy, gamma_xy) of the field f, the same at every point.
Vector3 strain_of(const LinearField& f) {
    return {f.a1, f.b2, f.a2 + f.b1};
}

// The 2 n numbers (u_0, v_0, u_1, v_1, ...) of the field f at the nodes of the mesh: u of node I at number 2 I, v at 2 I + 1.
std::vector<double> nodal_displacements(const Mesh2D& mesh, const LinearField& f) {
    std::vector<double> u(2 * mesh.num_nodes());
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        const double x = mesh.x(I);
        const double y = mesh.y(I);
        u[2 * I] = f.a0 + f.a1 * x + f.a2 * y;
        u[2 * I + 1] = f.b0 + f.b1 * x + f.b2 * y;
    }
    return u;
}

// The six displacements (u0, v0, u1, v1, u2, v2) of the field f at the three nodes of the triangle e.
Vector6 local_values(const Mesh2D& mesh, std::size_t e, const LinearField& f) {
    const Triangle& t = mesh.triangle(e);
    Vector6 u_e{};
    for (std::size_t i = 0; i < 3; ++i) {
        const double x = mesh.x(t[i]);
        const double y = mesh.y(t[i]);
        u_e[2 * i] = f.a0 + f.a1 * x + f.a2 * y;
        u_e[2 * i + 1] = f.b0 + f.b1 * x + f.b2 * y;
    }
    return u_e;
}

// The six displacements of the triangle e taken from the vector u of the whole mesh: the local degree of freedom
// i, from 0 to 5, is the global one 2 t[i / 2] + i % 2.
Vector6 gather(const Mesh2D& mesh, std::size_t e, const std::vector<double>& u) {
    const Triangle& t = mesh.triangle(e);
    Vector6 u_e{};
    for (std::size_t i = 0; i < 6; ++i) {
        u_e[i] = u[2 * t[i / 2] + i % 2];
    }
    return u_e;
}

// The mean of the temperatures of the three nodes of the triangle e, minus T_ref.
double mean_excess(const Mesh2D& mesh, std::size_t e, const std::vector<double>& T, double T_ref) {
    const Triangle& t = mesh.triangle(e);
    return (T[t[0]] + T[t[1]] + T[t[2]]) / 3.0 - T_ref;
}

// The rigid motions: two translations, a rotation about the origin (u = -y, v = x) and a combination.
std::vector<LinearField> rigid_motions() {
    return {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},     // translation in x
            {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},     // translation in y
            {0.0, 0.0, -1.0, 0.0, 1.0, 0.0},    // rotation
            {0.3, 0.0, -0.7, -1.2, 0.7, 0.0}};  // translation and rotation together
}

// The six "pure" fields (u = x, v = y, u = y, v = x and the two translations) plus one with all the numbers
// different from 0 and small strains, like the real ones (1e-3).
std::vector<LinearField> strain_fields() {
    return {{0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
            {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
            {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
            {0.3, 1.0e-3, -2.0e-3, -0.2, 3.0e-3, 5.0e-4}};
}

// All the linear fields: the rigid motions and those with strain.
std::vector<LinearField> all_fields() {
    std::vector<LinearField> fields = rigid_motions();
    for (const LinearField& f : strain_fields()) {
        fields.push_back(f);
    }
    return fields;
}

// Numbers that change from one degree of freedom to the next with no simple rule, of amplitude amplitude.
std::vector<double> wavy_vector(std::size_t size, double amplitude) {
    std::vector<double> v(size);
    for (std::size_t i = 0; i < size; ++i) {
        const double t = static_cast<double>(i);
        v[i] = amplitude * (std::sin(1.3 * t + 0.4) + 0.3 * std::cos(2.9 * t));
    }
    return v;
}

// The same numbers shifted by shift positions (the last goes in place of the first): a different vector, of the same
// amplitude.
std::vector<double> shifted(const std::vector<double>& v, std::size_t shift) {
    std::vector<double> w(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        w[i] = v[(i + shift) % v.size()];
    }
    return w;
}

// D v, row by column: (D v)[i] is the sum over k of D[i][k] v[k].
Vector3 times(const Matrix3& D, const Vector3& v) {
    Vector3 r{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t k = 0; k < 3; ++k) {
            r[i] += D[i][k] * v[k];
        }
    }
    return r;
}

Vector3 minus(const Vector3& a, const Vector3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

// A tolerance for the stresses: 1e-9 times the largest number of D times the sum of the strains involved:
// the mechanical one (strain), the thermal one (thermal) and the displacement divided by the size of the triangles (B u_e
// subtracts nearby numbers and loses digits in proportion).
double stress_tolerance(const Matrix3& D, double strain, double thermal, double displacement, double size) {
    return 1e-9 * largest_entry(D) * (strain + thermal + displacement / size);
}

// Compares two stresses (three numbers) with an absolute tolerance.
void expect_stress_near(const Vector3& got, const Vector3& want, double tolerance) {
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(got[i], want[i], tolerance) << "component " << i;
    }
}

// Compares two vectors element by element, with an absolute tolerance.
void expect_vectors_near(const std::vector<double>& got, const std::vector<double>& want, double tolerance) {
    ASSERT_EQ(got.size(), want.size()) << "size";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tolerance) << "degree of freedom " << i;
    }
}

// The forces on the nodes from the stresses: for each triangle the six numbers A B^T sigma_e (like the thermal
// forces A B^T D eps_th), added into the global degrees of freedom 2 t[i / 2] + i % 2.
std::vector<double> nodal_forces(const Mesh2D& mesh, const std::vector<Vector3>& sigma) {
    std::vector<double> f(2 * mesh.num_nodes(), 0.0);
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Matrix3x6 B = strain_displacement_matrix(mesh, e);
        const Triangle& t = mesh.triangle(e);
        for (std::size_t i = 0; i < 6; ++i) {
            for (std::size_t r = 0; r < 3; ++r) {
                f[2 * t[i / 2] + i % 2] += mesh.area(e) * B[r][i] * sigma[e][r];
            }
        }
    }
    return f;
}

// A problem with a known exact solution: the forces, the fixed degrees of freedom with their imposed displacements and the
// exact solution itself (2 n numbers).
struct Problem {
    std::vector<double> F;
    std::vector<std::size_t> dofs;
    std::vector<double> values;
    std::vector<double> exact;
};

// The uniform stretching of a rectangle (sides lx x ly, E, nu, plane_strain_matrix): the left side (x = 0) cannot
// move in x (u = 0 at all its nodes) and the bottom-left node cannot move in y (v = 0, so that the
// plate does not slide in y); on the right side the stress sigma acts, i.e. on each node the force sigma times the
// length of boundary that belongs to it (half of the distance from the node below plus half of that from the node above).
// With sigma_xx = sigma and sigma_yy = 0, in plane strain (epsilon_zz = 0, hence sigma_zz = nu sigma_xx):
//     epsilon_xx = (1 - nu^2) sigma / E,        epsilon_yy = -nu (1 + nu) sigma / E,
// and the exact solution is u = epsilon_xx x, v = epsilon_yy y: a linear field, which the elements reproduce.
Problem stretched_rectangle(const Mesh2D& mesh, double lx, double E, double nu, double sigma) {
    std::vector<std::pair<double, std::size_t>> left;   // (y, node)
    std::vector<std::pair<double, std::size_t>> right;  // (y, node)
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        if (mesh.x(I) <= 1e-9 * lx) {
            left.push_back({mesh.y(I), I});
        } else if (mesh.x(I) >= (1.0 - 1e-9) * lx) {
            right.push_back({mesh.y(I), I});
        }
    }
    std::sort(left.begin(), left.end());
    std::sort(right.begin(), right.end());

    Problem pr;
    pr.F.assign(2 * mesh.num_nodes(), 0.0);
    for (std::size_t j = 0; j < right.size(); ++j) {
        const double below = (j == 0) ? right[j].first : right[j - 1].first;
        const double above = (j + 1 == right.size()) ? right[j].first : right[j + 1].first;
        pr.F[2 * right[j].second] = sigma * 0.5 * (above - below);
    }
    for (const auto& node : left) {
        pr.dofs.push_back(2 * node.second);
    }
    pr.dofs.push_back(2 * left.front().second + 1);
    pr.values.assign(pr.dofs.size(), 0.0);

    const double exx = (1.0 - nu * nu) * sigma / E;
    const double eyy = -nu * (1.0 + nu) * sigma / E;
    pr.exact = nodal_displacements(mesh, LinearField{0.0, exx, 0.0, 0.0, 0.0, eyy});
    return pr;
}

// A plate heated uniformly by dT (T = T_ref + dT at all the nodes) and held between two walls: the nodes of the
// left side (x = 0) and of the right one (x = lx) do not move in x (u = 0), and the bottom-left node does not
// move in y (v = 0, so that the plate does not slide in y); above and below the plate is free. In plane strain
// (epsilon_zz = 0, hence sigma_zz = nu (sigma_xx + sigma_yy) - E alpha dT) with epsilon_xx = 0 and sigma_yy = 0 the
// Hooke equations with heat give
//     sigma_xx = -E alpha dT / (1 - nu),        epsilon_yy = (1 + nu) alpha dT / (1 - nu),
// and the exact solution is u = 0, v = epsilon_yy y: a linear field, which the elements reproduce. The plate cannot
// lengthen in x, and the heat that cannot expand it becomes compression.
Problem walls_problem(const Mesh2D& mesh, double lx, double E, double nu, double alpha, double dT) {
    Problem pr;
    pr.F = assemble_thermal_load_2d(mesh, plane_strain_matrix(E, nu), nu, alpha,
                                    std::vector<double>(mesh.num_nodes(), reference_temperature + dT),
                                    reference_temperature);
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        if (mesh.x(I) <= 1e-9 * lx || mesh.x(I) >= (1.0 - 1e-9) * lx) {
            pr.dofs.push_back(2 * I);
        }
    }
    pr.dofs.push_back(1);  // v of node 0, which is at (0, 0)
    pr.values.assign(pr.dofs.size(), 0.0);

    const double eyy = (1.0 + nu) * alpha * dT / (1.0 - nu);
    pr.exact = nodal_displacements(mesh, LinearField{0.0, 0.0, 0.0, 0.0, 0.0, eyy});
    return pr;
}

}  // namespace

// EXAMPLE 181. By hand, one triangle. Triangle 0 of the unit square has nodes (0, 0), (1, 0), (0, 1) and area
// 1/2; with E = 1 and nu = 1/4 it is D = [[1.2, 0.4, 0], [0.4, 1.2, 0], [0, 0, 0.4]]. The six displacements are
// (u0, v0, u1, v1, u2, v2) and, with 2 A = 1, the strain B u_e is eps_xx = u1 - u0, eps_yy = v2 - v0, gamma_xy =
// (u2 - u0) + (v1 - v0).
//   (a) Tension. u_e = (0, 0, 15/16, 0, 0, -5/16) has eps = (15/16, -5/16, 0) and sigma = D eps =
//       (1.2 15/16 - 0.4 5/16, 0.4 15/16 - 1.2 5/16, 0) = (1, 0, 0): it is the field of EXAMPLE 168 (a), stress 1 in x.
//   (b) Free expansion. With eps_th = (2.5, 2.5, 0) the triangle moves by u_e = (0, 0, 2.5, 0, 0, 2.5), i.e. by
//       B u_e = (2.5, 2.5, 0) = eps_th: all the strain is due to heat and the stress is 0.
//   (c) Held. u_e = 0 with the same eps_th: sigma = D (0 - eps_th) = -(1.2 2.5 + 0.4 2.5, 0.4 2.5 + 1.2 2.5, 0) =
//       (-4, -4, 0): compression.
//   (d) Shear. u_e = (0, 0, 0, 0, 1, 0) (u = y) has gamma_xy = 1 and sigma = (0, 0, 0.4 1) = (0, 0, 0.4).
//   (e) A bit of both. u_e = (0, 0, 0.5, 0, 0, 0.5) (eps = (0.5, 0.5, 0)) with eps_th = (2.5, 2.5, 0): what counts is the
//       DIFFERENCE, eps - eps_th = (-2, -2, 0), and sigma = (1.2 (-2) + 0.4 (-2), 0.4 (-2) + 1.2 (-2), 0) =
//       (-3.2, -3.2, 0).
//   (f) Triangle 1. It has nodes (1, 0), (1, 1), (0, 1), with b = (0, 1, -1) and c = (-1, 1, 0) (2 A = 1). If only its
//       node 1 moves in x by 0.2 (u_e = (0, 0, 0.2, 0, 0, 0)), eps = (b_1 0.2, 0, c_1 0.2) = (0.2, 0, 0.2) and
//       sigma = (1.2 0.2, 0.4 0.2, 0.4 0.2) = (0.24, 0.08, 0.08).
TEST(ElasticityStress2d, ElementStressOfASingleTriangleByHand) {
    const Mesh2D mesh = unit_square();
    const Matrix3 D = hooke_quarter();
    const Vector3 no_heat = {0.0, 0.0, 0.0};
    const Vector3 heat = {2.5, 2.5, 0.0};

    expect_stress_near(element_stress_2d(mesh, 0, D, {0.0, 0.0, 0.9375, 0.0, 0.0, -0.3125}, no_heat), {1.0, 0.0, 0.0},
                       1e-12);
    expect_stress_near(element_stress_2d(mesh, 0, D, {0.0, 0.0, 2.5, 0.0, 0.0, 2.5}, heat), {0.0, 0.0, 0.0}, 1e-12);
    expect_stress_near(element_stress_2d(mesh, 0, D, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, heat), {-4.0, -4.0, 0.0}, 1e-12);
    expect_stress_near(element_stress_2d(mesh, 0, D, {0.0, 0.0, 0.0, 0.0, 1.0, 0.0}, no_heat), {0.0, 0.0, 0.4}, 1e-12);
    expect_stress_near(element_stress_2d(mesh, 0, D, {0.0, 0.0, 0.5, 0.0, 0.0, 0.5}, heat), {-3.2, -3.2, 0.0}, 1e-12);
    expect_stress_near(element_stress_2d(mesh, 1, D, {0.0, 0.0, 0.2, 0.0, 0.0, 0.0}, no_heat), {0.24, 0.08, 0.08},
                       1e-12);
}

// EXAMPLE 182. The stress of any triangle, with any linear field. A field u = a0 + a1 x + a2 y,
// v = b0 + b1 x + b2 y has at every point the strain (a1, b2, a2 + b1), and it is computed without B: the stress is D
// ((a1, b2, a2 + b1) - eps_th), with D used row by column (sigma[i] is the sum over k of D[i][k] (...)[k]). It is
// tried on every triangle of nine meshes (six rectangles, the triangle with a node inside, the irregular quadrilateral,
// two triangles of different area), with four D (the Hooke matrix with E = 1, an arbitrary one, steel and a NON
// symmetric one, which tells the rows from the columns), three thermal strains (also with shear, which real thermal
// expansion does not have but the function must subtract) and eleven fields: the rigid motions, which have stress 0
// whatever D is, the "pure" strains and a field with strains of 1e-3 and a translation of 0.3.
TEST(ElasticityStress2d, ElementStressOfAnyLinearFieldOnAnyTriangle) {
    const std::vector<Matrix3> materials = {hooke_quarter(), general(), steel(), nonsymmetric()};
    const std::vector<Vector3> heats = {{0.0, 0.0, 0.0}, {2.5e-3, 2.5e-3, 0.0}, {3.0e-3, -1.0e-3, 2.0e-3}};
    const std::vector<LinearField> fields = all_fields();

    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t k = 0; k < meshes.size(); ++k) {
        const Mesh2D& mesh = meshes[k];
        const double size = smallest_size(mesh);
        for (std::size_t m = 0; m < materials.size(); ++m) {
            const Matrix3& D = materials[m];
            SCOPED_TRACE("mesh " + std::to_string(k) + ", material " + std::to_string(m));
            for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
                for (const LinearField& f : fields) {
                    const Vector6 u_e = local_values(mesh, e, f);
                    const Vector3 strain = strain_of(f);
                    double u_max = 0.0;
                    for (double a : u_e) {
                        u_max = std::max(u_max, std::fabs(a));
                    }
                    for (const Vector3& eps_th : heats) {
                        SCOPED_TRACE("triangle " + std::to_string(e));
                        const Vector3 want = times(D, minus(strain, eps_th));
                        const double tolerance = stress_tolerance(D, largest_entry(strain), largest_entry(eps_th),
                                                                  u_max, size);
                        expect_stress_near(element_stress_2d(mesh, e, D, u_e, eps_th), want, tolerance);
                    }
                }
            }
            if (HasFailure()) {
                return;
            }
        }
    }
}

// EXAMPLE 183. A nonexistent triangle throws std::out_of_range (like strain_displacement_matrix, which it relies on):
// the unit square has triangles 0 and 1, the mesh of 3 x 3 cells has triangles from 0 to 17.
TEST(ElasticityStress2d, AMissingTriangleThrowsOutOfRange) {
    const Matrix3 D = hooke_quarter();
    const Vector6 u_e = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    const Vector3 eps_th = {0.5, 0.5, 0.0};

    const Mesh2D square = unit_square();
    EXPECT_NO_THROW(element_stress_2d(square, 0, D, u_e, eps_th));
    EXPECT_NO_THROW(element_stress_2d(square, 1, D, u_e, eps_th));
    EXPECT_THROW(element_stress_2d(square, 2, D, u_e, eps_th), std::out_of_range);
    EXPECT_THROW(element_stress_2d(square, 1000000000, D, u_e, eps_th), std::out_of_range);
    EXPECT_THROW(element_stress_2d(square, std::numeric_limits<std::size_t>::max(), D, u_e, eps_th),
                 std::out_of_range);

    const Mesh2D grid = rectangle_mesh(1.0, 1.0, 3, 3);
    ASSERT_EQ(grid.num_triangles(), 18u);
    EXPECT_NO_THROW(element_stress_2d(grid, 17, D, u_e, eps_th));
    EXPECT_THROW(element_stress_2d(grid, 18, D, u_e, eps_th), std::out_of_range);
}

// EXAMPLE 184. By hand, the whole mesh. The unit square has 4 nodes, 0 = (0, 0), 1 = (1, 0), 2 = (0, 1), 3 = (1, 1), and
// two triangles, (0, 1, 2) and (1, 3, 2). The material has E = 1 and nu = 1/4 (as in EXAMPLE 168) and alpha = 1, with
// T_ref = 5. The displacements are (u0, v0, u1, v1, u2, v2, u3, v3) and the temperatures (T0, T1, T2, T3). Each problem
// has two stresses, one per triangle.
//   (a) The tension of EXAMPLE 168 (a): u = (0, 0, 15/16, 0, 0, -5/16, 15/16, -5/16), T = 5 everywhere (dT = 0).
//       Both triangles have stress (1, 0, 0).
//   (b) The free heated body (b): u = (0, 0, 2.5, 0, 0, 2.5, 2.5, 2.5), T = 7 (dT = 2, eps_th = 1.25 2 (1, 1, 0) =
//       2.5 (1, 1, 0)): B u = eps_th and both stresses are 0.
//   (c) The imposed displacement (c): u = (0, 0, 0.1, 0, 0, -1/30, 0.1, -1/30), T = 5: eps = (0.1, -1/30, 0) and sigma =
//       (1.2 0.1 - 0.4 / 30, 0.4 0.1 - 1.2 / 30, 0) = (8/75, 0, 0), i.e. (0.10666..., 0, 0) in both.
//   (d) Heated and clamped: u = 0, T = 7: sigma = -D eps_th = (-4, -4, 0) in both.
//   (e) Only one node moves: u = 0 except u3 = 0.2 (number 6), T = 5. Triangle 0, which has nodes 0, 1, 2,
//       does not feel node 3: stress (0, 0, 0). Triangle 1, which has nodes 1, 3, 2, has the displacement
//       (0, 0, 0.2, 0, 0, 0) of EXAMPLE 181 (f): stress (0.24, 0.08, 0.08).
//   (f) The temperature changes from node to node: u = 0, T = (5, 5, 5, 11). Triangle 0 has nodes 0, 1, 2 and
//       dT = (5 + 5 + 5) / 3 - 5 = 0: stress 0. Triangle 1 has nodes 1, 3, 2 and dT = (5 + 11 + 5) / 3 - 5 = 2:
//       stress -D eps_th = (-4, -4, 0).
TEST(ElasticityStress2d, StressesOfTheUnitSquareByHand) {
    const Mesh2D mesh = unit_square();
    const Matrix3 D = hooke_quarter();
    const double nu = 0.25;
    const double alpha = 1.0;
    const double T_ref = 5.0;
    const std::vector<double> rest(8, 0.0);

    const auto solve = [&](const std::vector<double>& u, const std::vector<double>& T) {
        return stress_field_2d(mesh, D, nu, alpha, u, T, T_ref);
    };

    std::vector<Vector3> sigma =
        solve({0.0, 0.0, 15.0 / 16.0, 0.0, 0.0, -5.0 / 16.0, 15.0 / 16.0, -5.0 / 16.0}, {5.0, 5.0, 5.0, 5.0});
    ASSERT_EQ(sigma.size(), 2u);
    expect_stress_near(sigma[0], {1.0, 0.0, 0.0}, 1e-12);
    expect_stress_near(sigma[1], {1.0, 0.0, 0.0}, 1e-12);

    sigma = solve({0.0, 0.0, 2.5, 0.0, 0.0, 2.5, 2.5, 2.5}, {7.0, 7.0, 7.0, 7.0});
    ASSERT_EQ(sigma.size(), 2u);
    expect_stress_near(sigma[0], {0.0, 0.0, 0.0}, 1e-12);
    expect_stress_near(sigma[1], {0.0, 0.0, 0.0}, 1e-12);

    sigma = solve({0.0, 0.0, 0.1, 0.0, 0.0, -1.0 / 30.0, 0.1, -1.0 / 30.0}, {5.0, 5.0, 5.0, 5.0});
    ASSERT_EQ(sigma.size(), 2u);
    expect_stress_near(sigma[0], {8.0 / 75.0, 0.0, 0.0}, 1e-12);
    expect_stress_near(sigma[1], {8.0 / 75.0, 0.0, 0.0}, 1e-12);

    sigma = solve(rest, {7.0, 7.0, 7.0, 7.0});
    ASSERT_EQ(sigma.size(), 2u);
    expect_stress_near(sigma[0], {-4.0, -4.0, 0.0}, 1e-12);
    expect_stress_near(sigma[1], {-4.0, -4.0, 0.0}, 1e-12);

    sigma = solve({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0}, {5.0, 5.0, 5.0, 5.0});
    ASSERT_EQ(sigma.size(), 2u);
    expect_stress_near(sigma[0], {0.0, 0.0, 0.0}, 1e-12);
    expect_stress_near(sigma[1], {0.24, 0.08, 0.08}, 1e-12);

    sigma = solve(rest, {5.0, 5.0, 5.0, 11.0});
    ASSERT_EQ(sigma.size(), 2u);
    expect_stress_near(sigma[0], {0.0, 0.0, 0.0}, 1e-12);
    expect_stress_near(sigma[1], {-4.0, -4.0, 0.0}, 1e-12);
}

// EXAMPLE 185. The patch test for the stresses. With a LINEAR displacement field u = a0 + a1 x + a2 y, v =
// b0 + b1 x + b2 y the strain is the same in every triangle, (a1, b2, a2 + b1), for any shape of the
// triangles; with a uniform temperature T = T_ref + dT so is eps_th = c (1, 1, 0), with c = (1 + nu) alpha dT.
// The stress D ((a1, b2, a2 + b1) - eps_th) is then the same in all the triangles. It is tried on nine meshes, three
// materials (also non-physical ones, and steel), three temperatures (dT = 0, +100 and -60 kelvin) and eleven fields: the
// rigid motions (which have stress -D eps_th, whatever the displacement is), the "pure" strains and a field with
// strains of 1e-3 and a translation of 0.3.
TEST(ElasticityStress2d, ALinearDisplacementFieldGivesTheSameStressInEveryTriangle) {
    const std::vector<LinearField> fields = all_fields();
    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t k = 0; k < meshes.size(); ++k) {
        const Mesh2D& mesh = meshes[k];
        const double size = smallest_size(mesh);
        for (const Material& m : sample_materials()) {
            for (double dT : {0.0, 100.0, -60.0}) {
                const std::vector<double> T(mesh.num_nodes(), reference_temperature + dT);
                const double c = (1.0 + m.nu) * m.alpha * dT;
                for (const LinearField& f : fields) {
                    SCOPED_TRACE("mesh " + std::to_string(k) + ", dT = " + std::to_string(dT));
                    const std::vector<double> u = nodal_displacements(mesh, f);
                    const Vector3 strain = strain_of(f);
                    const Vector3 want = times(m.D, minus(strain, {c, c, 0.0}));
                    const double tolerance =
                        stress_tolerance(m.D, largest_entry(strain), std::fabs(c), largest_entry(u), size);

                    const std::vector<Vector3> sigma =
                        stress_field_2d(mesh, m.D, m.nu, m.alpha, u, T, reference_temperature);
                    ASSERT_EQ(sigma.size(), mesh.num_triangles());
                    for (std::size_t e = 0; e < sigma.size(); ++e) {
                        expect_stress_near(sigma[e], want, tolerance);
                    }
                }
            }
        }
        if (HasFailure()) {
            return;
        }
    }
}

// EXAMPLE 186. The two extremes of the plate heated uniformly (T = T_ref + dT at all the nodes, so
// eps_th = c (1, 1, 0) with c = (1 + nu) alpha dT).
//   - CLAMPED (u = 0 at all the nodes): the heat would like to expand it and cannot. The stress is -D eps_th in
//     every triangle, a compression if it is heated: it is the stress that in the previous step arose together with
//     the displacement 0 (EXAMPLE 173). All the heat becomes stress.
//   - FREE (u = (a0 + c x, b0 + c y), with a translation (a0, b0) = (0.3, -0.2) that changes nothing): the body
//     expands by c in every direction, B u = eps_th and the stress is 0. All the heat becomes displacement.
// Nine meshes, three materials and two heatings (+100 and -60 kelvin).
TEST(ElasticityStress2d, AHeatedPlateIsStressedWhenClampedAndStressFreeWhenFree) {
    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t k = 0; k < meshes.size(); ++k) {
        const Mesh2D& mesh = meshes[k];
        const double size = smallest_size(mesh);
        for (const Material& m : sample_materials()) {
            for (double dT : {100.0, -60.0}) {
                SCOPED_TRACE("mesh " + std::to_string(k) + ", dT = " + std::to_string(dT));
                const std::vector<double> T(mesh.num_nodes(), reference_temperature + dT);
                const double c = (1.0 + m.nu) * m.alpha * dT;

                const std::vector<Vector3> clamped = stress_field_2d(
                    mesh, m.D, m.nu, m.alpha, std::vector<double>(2 * mesh.num_nodes(), 0.0), T, reference_temperature);
                ASSERT_EQ(clamped.size(), mesh.num_triangles());
                const Vector3 compression = times(m.D, {-c, -c, 0.0});
                for (const Vector3& s : clamped) {
                    expect_stress_near(s, compression, stress_tolerance(m.D, 0.0, std::fabs(c), 0.0, size));
                }

                const std::vector<double> u = nodal_displacements(mesh, LinearField{0.3, c, 0.0, -0.2, 0.0, c});
                const std::vector<Vector3> free_body =
                    stress_field_2d(mesh, m.D, m.nu, m.alpha, u, T, reference_temperature);
                ASSERT_EQ(free_body.size(), mesh.num_triangles());
                for (const Vector3& s : free_body) {
                    expect_stress_near(s, {0.0, 0.0, 0.0},
                                       stress_tolerance(m.D, std::fabs(c), std::fabs(c), largest_entry(u), size));
                }
            }
        }
        if (HasFailure()) {
            return;
        }
    }
}

// EXAMPLE 187. The temperature of a triangle is the MEAN of the temperatures of its three nodes, as for the thermal
// forces (assemble_thermal_load_2d): dT_e = (T[t[0]] + T[t[1]] + T[t[2]]) / 3 - T_ref, and eps_th is (1 + nu) alpha
// dT_e (1, 1, 0). With u = 0 the stress is -D eps_th, different from one triangle to another
// (L = the larger dimension of the mesh).
//   - With a LINEAR temperature, T = T_ref + g0 + gx x + gy y (g0 = 20 K, gx = 80 K / L, gy = -50 K / L), the mean
//     of the nodes is the value at the centroid: dT_e = g0 + gx cx + gy cy, with (cx, cy) the centroid of the triangle.
//   - With a temperature that is NOT linear, T = T_ref + 30 (x / L)^2 - 20 (y / L)^2 + 10 (x / L) (y / L), the mean
//     of the nodes is not the value at the centroid, and the function must use the mean: dT_e = (f0 + f1 + f2) / 3, with f_i
//     the value of 30 (x / L)^2 - ... at node i.
// Nine meshes, three materials.
TEST(ElasticityStress2d, TheTemperatureOfATriangleIsTheMeanOfItsThreeNodes) {
    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t k = 0; k < meshes.size(); ++k) {
        const Mesh2D& mesh = meshes[k];
        const double L = mesh_size(mesh);
        const double g0 = 20.0;
        const double gx = 80.0 / L;
        const double gy = -50.0 / L;
        const auto bump = [&](std::size_t I) {
            const double X = mesh.x(I) / L;
            const double Y = mesh.y(I) / L;
            return 30.0 * X * X - 20.0 * Y * Y + 10.0 * X * Y;
        };
        std::vector<double> linear(mesh.num_nodes());
        std::vector<double> curved(mesh.num_nodes());
        for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
            linear[I] = reference_temperature + g0 + gx * mesh.x(I) + gy * mesh.y(I);
            curved[I] = reference_temperature + bump(I);
        }
        const std::vector<double> rest(2 * mesh.num_nodes(), 0.0);

        for (const Material& m : sample_materials()) {
            SCOPED_TRACE("mesh " + std::to_string(k));
            const std::vector<Vector3> from_linear =
                stress_field_2d(mesh, m.D, m.nu, m.alpha, rest, linear, reference_temperature);
            const std::vector<Vector3> from_curved =
                stress_field_2d(mesh, m.D, m.nu, m.alpha, rest, curved, reference_temperature);
            ASSERT_EQ(from_linear.size(), mesh.num_triangles());
            ASSERT_EQ(from_curved.size(), mesh.num_triangles());

            // The temperature deviations are at most 150 kelvin (20 + 80 + 50 in the linear case, 60 in the curved one).
            const double thermal = (1.0 + std::fabs(m.nu)) * std::fabs(m.alpha) * 150.0;
            const double tolerance = stress_tolerance(m.D, 0.0, thermal, 0.0, 1.0);
            for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
                SCOPED_TRACE("triangle " + std::to_string(e));
                const Triangle& t = mesh.triangle(e);
                const double cx = (mesh.x(t[0]) + mesh.x(t[1]) + mesh.x(t[2])) / 3.0;
                const double cy = (mesh.y(t[0]) + mesh.y(t[1]) + mesh.y(t[2])) / 3.0;

                const double dT_linear = g0 + gx * cx + gy * cy;
                const double c_linear = (1.0 + m.nu) * m.alpha * dT_linear;
                expect_stress_near(from_linear[e], times(m.D, {-c_linear, -c_linear, 0.0}), tolerance);

                const double dT_curved = (bump(t[0]) + bump(t[1]) + bump(t[2])) / 3.0;
                const double c_curved = (1.0 + m.nu) * m.alpha * dT_curved;
                expect_stress_near(from_curved[e], times(m.D, {-c_curved, -c_curved, 0.0}), tolerance);
            }
        }
        if (HasFailure()) {
            return;
        }
    }
}

// EXAMPLE 188. Each triangle uses ITS OWN nodes. The fields of the previous tests are linear or uniform, and give the same
// stress in all the triangles: a swap of nodes goes unseen. Here displacements and temperatures change from one node
// to the next with no simple rule (wavy_vector), so each triangle has a stress of its own, and that of the triangle e
// must be that of element_stress_2d with its displacements u[2 t[i / 2] + i % 2], the mean of the temperatures
// of its nodes minus T_ref and its thermal strain plane_strain_thermal_strain(nu, alpha, dT_e). Nine meshes,
// four materials (also with the non-symmetric D). It is also checked that the stresses are not all equal.
TEST(ElasticityStress2d, EveryTriangleUsesItsOwnNodes) {
    std::vector<Material> materials = sample_materials();
    materials.push_back({nonsymmetric(), 0.2, 5.0e-4});

    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t k = 0; k < meshes.size(); ++k) {
        const Mesh2D& mesh = meshes[k];
        const double size = smallest_size(mesh);
        const std::vector<double> u = wavy_vector(2 * mesh.num_nodes(), 0.01 * mesh_size(mesh));
        std::vector<double> T = wavy_vector(mesh.num_nodes(), 40.0);
        for (double& a : T) {
            a += reference_temperature;
        }
        for (const Material& m : materials) {
            SCOPED_TRACE("mesh " + std::to_string(k));
            const std::vector<Vector3> sigma = stress_field_2d(mesh, m.D, m.nu, m.alpha, u, T, reference_temperature);
            ASSERT_EQ(sigma.size(), mesh.num_triangles());

            const double thermal = (1.0 + std::fabs(m.nu)) * std::fabs(m.alpha) * 80.0;
            const double tolerance = stress_tolerance(m.D, 0.02, thermal, largest_entry(u), size);
            double spread = 0.0;
            for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
                SCOPED_TRACE("triangle " + std::to_string(e));
                const Vector3 eps_th =
                    plane_strain_thermal_strain(m.nu, m.alpha, mean_excess(mesh, e, T, reference_temperature));
                expect_stress_near(sigma[e], element_stress_2d(mesh, e, m.D, gather(mesh, e, u), eps_th), tolerance);
                for (std::size_t i = 0; i < 3; ++i) {
                    spread = std::max(spread, std::fabs(sigma[e][i] - sigma[0][i]));
                }
            }
            EXPECT_GT(spread, 1e-3 * largest_entry(sigma)) << "the stresses of the triangles are all equal";
        }
        if (HasFailure()) {
            return;
        }
    }
}

// EXAMPLE 189. The forces on the nodes from the stresses are K u - F. For each triangle A B^T sigma_e are six forces on its
// nodes (like the thermal forces A B^T D eps_th), and summed as in K they give, degree of freedom by degree of freedom,
//     sum over the triangles of A B^T D (B u_e - eps_th) = K u - F,
// with K = assemble_elasticity_stiffness_2d and F = assemble_thermal_load_2d: it is the equilibrium equation, written with
// the stresses. It holds for ANY u and T (not only for the solution): the left side is computed with the stress
// of the function, the right one with K and F. It is tried with arbitrary displacements and temperatures
// (wavy_vector), on nine meshes and four materials (also with the non-symmetric D: K and F use it as it is given).
TEST(ElasticityStress2d, TheNodalForcesOfTheStressesAreKuMinusF) {
    std::vector<Material> materials = sample_materials();
    materials.push_back({nonsymmetric(), 0.2, 5.0e-4});

    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t k = 0; k < meshes.size(); ++k) {
        const Mesh2D& mesh = meshes[k];
        const std::vector<double> u = wavy_vector(2 * mesh.num_nodes(), 0.01 * mesh_size(mesh));
        std::vector<double> T = wavy_vector(mesh.num_nodes(), 40.0);
        for (double& a : T) {
            a += reference_temperature;
        }
        for (const Material& m : materials) {
            SCOPED_TRACE("mesh " + std::to_string(k));
            const std::vector<double> F =
                assemble_thermal_load_2d(mesh, m.D, m.nu, m.alpha, T, reference_temperature);
            const std::vector<double> Ku = assemble_elasticity_stiffness_2d(mesh, m.D) * u;
            const std::vector<double> f =
                nodal_forces(mesh, stress_field_2d(mesh, m.D, m.nu, m.alpha, u, T, reference_temperature));
            ASSERT_EQ(f.size(), F.size());
            ASSERT_EQ(Ku.size(), F.size());

            const double scale = largest_entry(Ku) + largest_entry(F);
            for (std::size_t i = 0; i < f.size(); ++i) {
                EXPECT_NEAR(f[i], Ku[i] - F[i], 1e-10 * scale) << "degree of freedom " << i;
            }
        }
        if (HasFailure()) {
            return;
        }
    }
}

// EXAMPLE 190. In a free heated body the stresses are in equilibrium by themselves. With an arbitrary temperature
// (wavy_vector, of amplitude 40 kelvin) and only the three isostatic constraints (no external force, hence no
// reaction) u is computed with solve_displacement_2d: the body deforms irregularly and has stresses different from 0
// (at least a tenth of the thermal stress D eps_th it would have if clamped: without this check a vector of stresses
// that are all 0 would have forces 0 and would pass), and the forces on the nodes A B^T sigma added up are 0 at ALL
// the degrees of freedom (K u = F at the free ones, reaction 0 at the three fixed ones). Mesh of 4 x 3 cells, three materials.
TEST(ElasticityStress2d, TheStressesOfAFreeHeatedBodyAreInEquilibrium) {
    const Mesh2D mesh = rectangle_mesh(3.0, 1.5, 4, 3);
    std::vector<double> T = wavy_vector(mesh.num_nodes(), 40.0);
    for (double& a : T) {
        a += reference_temperature;
    }
    const std::vector<std::size_t> dofs = isostatic_dofs(mesh, 0);

    for (const Material& m : sample_materials()) {
        const std::vector<double> F = assemble_thermal_load_2d(mesh, m.D, m.nu, m.alpha, T, reference_temperature);
        const std::vector<double> u = solve_displacement_2d(mesh, m.D, F, dofs, {0.0, 0.0, 0.0});
        const std::vector<Vector3> sigma = stress_field_2d(mesh, m.D, m.nu, m.alpha, u, T, reference_temperature);
        ASSERT_EQ(sigma.size(), mesh.num_triangles());

        const double thermal_stress = largest_entry(m.D) * (1.0 + std::fabs(m.nu)) * std::fabs(m.alpha) * 40.0;
        EXPECT_GT(largest_entry(sigma), 0.1 * thermal_stress) << "the stresses are too small to prove anything";

        const std::vector<double> f = nodal_forces(mesh, sigma);
        const double scale = largest_entry(F);
        for (std::size_t i = 0; i < f.size(); ++i) {
            EXPECT_NEAR(f[i], 0.0, 1e-8 * scale) << "degree of freedom " << i;
        }
    }
}

// EXAMPLE 191. The stresses are LINEAR in displacements, temperature and material. With the same material and two pairs
// (u1, T1) and (u2, T2), with dT = T - T_ref:
//   - the sum: sigma(u1 + u2, T_ref + dT1 + dT2) = sigma(u1, T1) + sigma(u2, T2);
//   - a factor s = -2.5 (even negative) on both: sigma(s u1, T_ref + s dT1) = s sigma(u1, T1);
//   - the material: with 3 D in place of D (and the same nu and alpha) all the stresses are 3 times those of D.
// And two things about heat: with alpha = 0 the temperature does not matter (the stresses are those of B u_e
// alone, with any T or with T = T_ref), and with T = T_ref alpha does not matter. Nine meshes, three materials.
TEST(ElasticityStress2d, TheStressIsLinearInDisplacementTemperatureAndMaterial) {
    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t k = 0; k < meshes.size(); ++k) {
        const Mesh2D& mesh = meshes[k];
        const std::size_t N = 2 * mesh.num_nodes();
        const double size = smallest_size(mesh);
        const std::vector<double> u1 = wavy_vector(N, 0.01 * mesh_size(mesh));
        const std::vector<double> u2 = shifted(u1, 7);
        const std::vector<double> dT1 = wavy_vector(mesh.num_nodes(), 40.0);
        const std::vector<double> dT2 = shifted(dT1, 3);

        const auto temperature = [&](double s1, double s2) {
            std::vector<double> T(mesh.num_nodes());
            for (std::size_t I = 0; I < T.size(); ++I) {
                T[I] = reference_temperature + s1 * dT1[I] + s2 * dT2[I];
            }
            return T;
        };
        const auto combine = [&](const std::vector<double>& a, double sa, const std::vector<double>& b, double sb) {
            std::vector<double> w(a.size());
            for (std::size_t i = 0; i < w.size(); ++i) {
                w[i] = sa * a[i] + sb * b[i];
            }
            return w;
        };

        for (const Material& m : sample_materials()) {
            SCOPED_TRACE("mesh " + std::to_string(k));
            const double thermal = (1.0 + std::fabs(m.nu)) * std::fabs(m.alpha) * 250.0;
            const double tolerance = stress_tolerance(m.D, 0.05, thermal, 2.5 * largest_entry(u1), size);
            const auto stress = [&](const Matrix3& D, double alpha, const std::vector<double>& u,
                                    const std::vector<double>& T) {
                return stress_field_2d(mesh, D, m.nu, alpha, u, T, reference_temperature);
            };

            const std::vector<Vector3> s1 = stress(m.D, m.alpha, u1, temperature(1.0, 0.0));
            const std::vector<Vector3> s2 = stress(m.D, m.alpha, u2, temperature(0.0, 1.0));
            const std::vector<Vector3> sum = stress(m.D, m.alpha, combine(u1, 1.0, u2, 1.0), temperature(1.0, 1.0));
            const std::vector<Vector3> scaled =
                stress(m.D, m.alpha, combine(u1, -2.5, u2, 0.0), temperature(-2.5, 0.0));
            Matrix3 triple = m.D;
            for (auto& row : triple) {
                for (double& a : row) {
                    a *= 3.0;
                }
            }
            const std::vector<Vector3> tripled = stress(triple, m.alpha, u1, temperature(1.0, 0.0));
            const std::vector<Vector3> no_alpha_hot = stress(m.D, 0.0, u1, temperature(1.0, 0.0));
            const std::vector<Vector3> no_alpha_cold = stress(m.D, 0.0, u1, temperature(0.0, 0.0));
            const std::vector<Vector3> cold = stress(m.D, m.alpha, u1, temperature(0.0, 0.0));
            ASSERT_EQ(s1.size(), mesh.num_triangles());

            for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
                SCOPED_TRACE("triangle " + std::to_string(e));
                Vector3 want_sum{};
                Vector3 want_scaled{};
                Vector3 want_tripled{};
                for (std::size_t i = 0; i < 3; ++i) {
                    want_sum[i] = s1[e][i] + s2[e][i];
                    want_scaled[i] = -2.5 * s1[e][i];
                    want_tripled[i] = 3.0 * s1[e][i];
                }
                expect_stress_near(sum[e], want_sum, tolerance);
                expect_stress_near(scaled[e], want_scaled, tolerance);
                expect_stress_near(tripled[e], want_tripled, 3.0 * tolerance);
                expect_stress_near(no_alpha_hot[e], no_alpha_cold[e], tolerance);
                expect_stress_near(cold[e], no_alpha_cold[e], tolerance);
            }
        }
        if (HasFailure()) {
            return;
        }
    }
}

// EXAMPLE 192. The nodes that are in no triangle (the mesh has nodes 0 = (5, 5) and 5 = (7, 7) outside the two
// triangles of the square 1, 2, 3, 4) are not read: neither their displacement (numbers 0, 1, 10 and 11 of u) nor their
// temperature. With NaNs in their place the stresses of the two triangles are IDENTICAL to those computed with
// arbitrary numbers, and without NaN.
TEST(ElasticityStress2d, NodesOutsideEveryTriangleAreNotRead) {
    const Mesh2D mesh = mesh_with_nodes_outside_every_triangle();
    const Matrix3 D = hooke_quarter();
    const double not_a_number = std::numeric_limits<double>::quiet_NaN();

    std::vector<double> u = wavy_vector(12, 0.1);
    std::vector<double> T = wavy_vector(6, 30.0);
    for (double& a : T) {
        a += reference_temperature;
    }
    std::vector<double> u_nan = u;
    std::vector<double> T_nan = T;
    for (std::size_t d : {0u, 1u, 10u, 11u}) {
        u[d] = 123.0;  // arbitrary numbers
        u_nan[d] = not_a_number;
    }
    T[0] = -5000.0;
    T[5] = 1.0e9;
    T_nan[0] = not_a_number;
    T_nan[5] = not_a_number;

    const std::vector<Vector3> clean = stress_field_2d(mesh, D, 0.25, 1.0e-3, u, T, reference_temperature);
    const std::vector<Vector3> with_nan = stress_field_2d(mesh, D, 0.25, 1.0e-3, u_nan, T_nan, reference_temperature);
    ASSERT_EQ(clean.size(), 2u);
    ASSERT_EQ(with_nan.size(), 2u);
    for (std::size_t e = 0; e < 2; ++e) {
        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_FALSE(std::isnan(with_nan[e][i])) << "triangle " << e << ", component " << i;
            EXPECT_EQ(with_nan[e][i], clean[e][i]) << "triangle " << e << ", component " << i;
        }
    }
}

// EXAMPLE 193. Wrong data: the function throws std::invalid_argument (and does not read outside a vector). With the
// mesh of 2 x 1 cells (6 nodes, 12 displacements) the following are wrong:
//   - a u that does not have 12 numbers (11, 13, 6, 24, 0);
//   - a T that does not have 6 numbers (5, 7, 12, 0);
//   - a Poisson ratio that is not between -1 and 1/2, ends excluded (-1, 0.5, -2, 1, +-infinity, NaN).
// These are fine instead: nu = -0.99 and 0.49 (almost the ends), alpha = 0 and negative, and any T_ref (even below absolute
// zero): D, alpha, u, T and T_ref are not checked.
TEST(ElasticityStress2d, BadInputsThrowInvalidArgument) {
    const Mesh2D mesh = rectangle_mesh(2.0, 1.0, 2, 1);
    ASSERT_EQ(mesh.num_nodes(), 6u);
    const Matrix3 D = hooke_quarter();
    const std::vector<double> u(12, 0.01);
    const std::vector<double> T(6, 300.0);
    const double not_a_number = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();

    EXPECT_NO_THROW(stress_field_2d(mesh, D, 0.25, 1.0e-3, u, T, 293.15));

    for (std::size_t size : {11u, 13u, 6u, 24u, 0u}) {
        EXPECT_THROW(stress_field_2d(mesh, D, 0.25, 1.0e-3, std::vector<double>(size, 0.01), T, 293.15),
                     std::invalid_argument)
            << "u of " << size << " numbers";
    }
    for (std::size_t size : {5u, 7u, 12u, 0u}) {
        EXPECT_THROW(stress_field_2d(mesh, D, 0.25, 1.0e-3, u, std::vector<double>(size, 300.0), 293.15),
                     std::invalid_argument)
            << "T of " << size << " numbers";
    }
    for (double nu : {-1.0, 0.5, -2.0, 1.0, -infinity, infinity, not_a_number}) {
        EXPECT_THROW(stress_field_2d(mesh, D, nu, 1.0e-3, u, T, 293.15), std::invalid_argument) << "nu = " << nu;
    }

    EXPECT_NO_THROW(stress_field_2d(mesh, D, -0.99, 1.0e-3, u, T, 293.15));
    EXPECT_NO_THROW(stress_field_2d(mesh, D, 0.49, 1.0e-3, u, T, 293.15));
    EXPECT_NO_THROW(stress_field_2d(mesh, D, 0.25, 0.0, u, T, 293.15));
    EXPECT_NO_THROW(stress_field_2d(mesh, D, 0.25, -1.0e-3, u, T, 293.15));
    EXPECT_NO_THROW(stress_field_2d(mesh, D, 0.25, 1.0e-3, u, T, -300.0));
}

// EXAMPLE 194. Together with solve_displacement_2d: the stretching of a rectangle. As in EXAMPLE 169 (left side with
// u = 0, one node with v = 0, on the right side the stress sigma = 1e-3 E) the displacements are computed, and from them the
// stresses: in every triangle sigma_xx = sigma, sigma_yy = 0 and sigma_xy = 0, and T = T_ref heats nothing. It is
// tried on six rectangles of different shapes (even very flat ones), with E = 1 and with E = 210e9 and with
// nu = 0, 1/4, 0.3, 0.4: the errors of the displacements (1e-12 and less) are multiplied by 1 / h in the strain, and
// the stresses coincide with sigma within 1e-6 sigma.
TEST(ElasticityStress2d, TensionOfARectangleThroughTheSolver) {
    for (const Grid& g : sample_grids()) {
        const Mesh2D mesh = rectangle_mesh(g.lx, g.ly, g.nx, g.ny);
        for (double E : {1.0, 210e9}) {
            for (double nu : {0.0, 0.25, 0.3, 0.4}) {
                SCOPED_TRACE("grid " + std::to_string(g.nx) + " x " + std::to_string(g.ny) + ", E = " +
                             std::to_string(E) + ", nu = " + std::to_string(nu));
                const double sigma = 1e-3 * E;
                const Problem pr = stretched_rectangle(mesh, g.lx, E, nu, sigma);
                const Matrix3 D = plane_strain_matrix(E, nu);
                const std::vector<double> u = solve_displacement_2d(mesh, D, pr.F, pr.dofs, pr.values);

                const std::vector<Vector3> stress = stress_field_2d(
                    mesh, D, nu, 1.0e-5, u, std::vector<double>(mesh.num_nodes(), reference_temperature),
                    reference_temperature);
                ASSERT_EQ(stress.size(), mesh.num_triangles());
                for (const Vector3& s : stress) {
                    expect_stress_near(s, {sigma, 0.0, 0.0}, 1e-6 * sigma);
                }
            }
        }
    }
}

// EXAMPLE 195. The plate heated between two walls, with the textbook formula (walls_problem). A plate that cannot
// lengthen in x (u = 0 on the left and right sides) and is free in y is heated uniformly by dT. The heat
// that cannot expand it in x becomes compression:
//     sigma_xx = -E alpha dT / (1 - nu),        sigma_yy = 0,        sigma_xy = 0,
// and in y the plate lengthens by epsilon_yy = (1 + nu) alpha dT / (1 - nu) (displacements u = 0, v = epsilon_yy y).
// With E = 1 and nu = 1/4 and dT = 2 and alpha = 1 these are sigma_xx = -(4/3) 2 = -8/3 and epsilon_yy = 1.25 2 / 0.75 =
// 10/3. It is tried on six rectangles, with four materials (nu = 0 where sigma_xx = -E alpha dT, steel and aluminum)
// and two heatings (+50 and -30 kelvin, hence tension if it is cooled): the displacements coincide with the textbook
// ones within 1e-8 times the largest (the CG stops with a residual of 1e-12, and on the very flat cells, 10 x 0.014,
// the error on the displacements reaches 1e-10), the stresses within 1e-6 times sigma_xx.
TEST(ElasticityStress2d, AHeatedPlateBetweenTwoWallsMatchesTheTextbook) {
    struct Plate {
        double E, nu, alpha;
    };
    const std::vector<Plate> plates = {{1.0, 0.25, 1.0e-3},     // E = 1, nu = 1/4
                                       {1.0, 0.0, 2.0e-3},      // nu = 0: sigma_xx = -E alpha dT
                                       {210e9, 0.3, 12.0e-6},   // steel
                                       {70e9, 0.33, 23.0e-6}};  // aluminum

    for (const Grid& g : sample_grids()) {
        const Mesh2D mesh = rectangle_mesh(g.lx, g.ly, g.nx, g.ny);
        for (const Plate& p : plates) {
            for (double dT : {50.0, -30.0}) {
                SCOPED_TRACE("grid " + std::to_string(g.nx) + " x " + std::to_string(g.ny) + ", E = " +
                             std::to_string(p.E) + ", nu = " + std::to_string(p.nu) + ", dT = " + std::to_string(dT));
                const Problem pr = walls_problem(mesh, g.lx, p.E, p.nu, p.alpha, dT);
                const Matrix3 D = plane_strain_matrix(p.E, p.nu);
                const std::vector<double> u = solve_displacement_2d(mesh, D, pr.F, pr.dofs, pr.values);
                expect_vectors_near(u, pr.exact, 1e-8 * largest_entry(pr.exact));

                const double sigma_xx = -p.E * p.alpha * dT / (1.0 - p.nu);
                const std::vector<Vector3> stress =
                    stress_field_2d(mesh, D, p.nu, p.alpha, u,
                                    std::vector<double>(mesh.num_nodes(), reference_temperature + dT),
                                    reference_temperature);
                ASSERT_EQ(stress.size(), mesh.num_triangles());
                for (const Vector3& s : stress) {
                    expect_stress_near(s, {sigma_xx, 0.0, 0.0}, 1e-6 * std::fabs(sigma_xx));
                }
            }
        }
    }
}

// EXAMPLE 196. The same plate, large: a steel square of 1 m side with 40 x 40 cells
// (1681 nodes, 3362 degrees of freedom), E = 210 GPa, nu = 0.3, alpha = 12e-6 per kelvin, heated by 100 kelvin and
// held between two walls. The textbook stress is sigma_xx = -E alpha dT / (1 - nu) = -360 MPa (-3.6e8 Pa) in
// all 3200 triangles, with sigma_yy = sigma_xy = 0: the function must find it again within 1e-6 times its value (the
// CG takes about 540 iterations, with numbers of K of order 1e11). The time is that of solve_displacement_2d: a
// fraction of a second.
TEST(ElasticityStress2d, ALargeSteelPlateBetweenTwoWalls) {
    const double E = 210e9;
    const double nu = 0.3;
    const double alpha = 12.0e-6;
    const double dT = 100.0;
    const Mesh2D mesh = rectangle_mesh(1.0, 1.0, 40, 40);
    ASSERT_EQ(mesh.num_triangles(), 3200u);

    const Problem pr = walls_problem(mesh, 1.0, E, nu, alpha, dT);
    const Matrix3 D = plane_strain_matrix(E, nu);
    const std::vector<double> u = solve_displacement_2d(mesh, D, pr.F, pr.dofs, pr.values);

    const double sigma_xx = -E * alpha * dT / (1.0 - nu);
    EXPECT_NEAR(sigma_xx, -3.6e8, 1.0);
    const std::vector<double> T(mesh.num_nodes(), reference_temperature + dT);
    const std::vector<Vector3> stress = stress_field_2d(mesh, D, nu, alpha, u, T, reference_temperature);
    ASSERT_EQ(stress.size(), 3200u);
    for (std::size_t e = 0; e < stress.size(); ++e) {
        expect_stress_near(stress[e], {sigma_xx, 0.0, 0.0}, 1e-6 * std::fabs(sigma_xx));
        if (HasFailure()) {
            return;
        }
    }
}

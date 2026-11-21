#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "CSR_and_Vectors.h"
#include "Elasticity2D.h"
#include "Mesh2D.h"

// ---- From the thermal forces of the triangles to the thermal forces of the whole mesh ------------------------------
//
// element_thermal_load(mesh, e, D, eps_th) gives the 6 forces of ONE heated triangle, on its LOCAL degrees of freedom
// (u0, v0, u1, v1, u2, v2). The vector of the whole mesh, F, has 2 n numbers (n = number of nodes) and its indices are the
// GLOBAL degrees of freedom of K: node I has its x force at number 2 I and its y one at number 2 I + 1. As for the
// heat load (assemble_load_2d) it is the sum of the contributions of the triangles, with the correspondence of K:
//
//     F[P] = sum, over the triangles e that contain P, of f_e[i]
//            where i is the local degree of freedom of P, i.e. with t = mesh.triangle(e):  P = 2 t[i / 2] + i % 2.
//
// What is new with respect to heat is that every triangle has ITS OWN heating. The temperature T is given at the nodes and is
// linear inside each triangle, so the thermal strain eps_th = (1 + nu) alpha (T - T_ref) (1, 1, 0) is
// linear. The force A B^T D eps_th (B is constant in the triangle) only needs the MEAN VALUE of eps_th, which for
// a linear function is the one at the centroid, i.e. the mean of the values at the three nodes (the centroid has
// barycentric coordinates 1/3, 1/3, 1/3): it is not an approximation, it is exact. Therefore for each triangle
//
//     dT_e = (T[t[0]] + T[t[1]] + T[t[2]]) / 3 - T_ref,
//     eps_e = plane_strain_thermal_strain(nu, alpha, dT_e),
//     f_e = element_thermal_load(mesh, e, D, eps_e).
//
// assemble_thermal_load_2d(mesh, D, nu, alpha, T, T_ref) returns F. It throws std::invalid_argument if T does not have a
// number per node or if nu is not between -1 and 1/2 (checked by plane_strain_thermal_strain).
//
// What a real thermal force must have, and this is what the tests check:
//   - on the unit square it is computed by hand (EXAMPLE 157);
//   - with a uniform temperature the thermal stress D eps_th is constant: the forces cancel at the interior nodes and
//     remain on the boundary, where each side pushes its two nodes outwards (EXAMPLE 158);
//   - rigid motions do no work: the resultant and the moment of the forces are 0, whatever T is (EXAMPLE 159);
//   - for a linear displacement the work u^T F is known in advance, with the integral of T - T_ref (EXAMPLE 160);
//   - a free body heated uniformly expands without stresses: K u = F (EXAMPLE 161);
//   - F is linear in T - T_ref and in alpha, and with no temperature change there is no force (EXAMPLE 162);
//   - the node from which one starts writing a triangle, and the order of the triangles, do not change F (EXAMPLE 163);
//   - nodes outside every triangle have zero forces and their temperature does not matter (EXAMPLE 164);
//   - it is the sum of the contributions of the triangles, on every kind of mesh (EXAMPLE 165);
//   - wrong data throw std::invalid_argument (EXAMPLE 166).

namespace {

// A rectangle lx x ly divided into nx x ny cells, as in rectangle_mesh.
struct Grid {
    double lx, ly;
    std::size_t nx, ny;
};

// Six rectangles of very different shapes (the same as in the K assembly tests).
std::vector<Grid> sample_grids() {
    return {{1.0, 1.0, 1, 1},    // the unit square: two triangles
            {2.0, 1.0, 2, 1},    // two cells
            {3.0, 1.5, 4, 3},    // 0.75 x 0.5 cells
            {1.0, 1.0, 6, 6},    // 72 triangles
            {0.2, 0.05, 5, 2},   // small (20 cm by 5 cm, in meters)
            {10.0, 0.1, 1, 7}};  // very flat: 10 x 0.0143 cells
}

// The unit square divided along the diagonal from (1, 0) to (0, 1): nodes 0 = (0, 0), 1 = (1, 0), 2 = (0, 1),
// 3 = (1, 1) and triangles (0, 1, 2) and (1, 3, 2), both with area 1/2.
Mesh2D unit_square() {
    return Mesh2D({0.0, 1.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {1, 3, 2}});
}

// The large triangle with node 3 inside (area 9/2), which is not a rectangle.
Mesh2D triangle_with_interior_node() {
    return Mesh2D({0.0, 3.0, 0.0, 1.0}, {0.0, 0.0, 3.0, 1.0}, {{0, 1, 3}, {1, 2, 3}, {2, 0, 3}});
}

// An irregular quadrilateral with a node inside: 0 = (0, 0), 1 = (4, 0), 2 = (3, 3), 3 = (-1, 2) and, inside,
// 4 = (1, 0.7). Four triangles around node 4, all of different shape and area (10.5 in total).
Mesh2D irregular_quadrilateral() {
    return Mesh2D({0.0, 4.0, 3.0, -1.0, 1.0}, {0.0, 0.0, 3.0, 2.0, 0.7},
                  {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}});
}

// Two triangles of DIFFERENT area: 0 = (0, 0), 1 = (4, 0), 2 = (1, 2), 3 = (6, 4); (0, 1, 2) has A = 4 and (1, 3, 2) has
// A = 8.
Mesh2D two_triangles_of_different_area() {
    return Mesh2D({0.0, 4.0, 1.0, 6.0}, {0.0, 0.0, 2.0, 4.0}, {{0, 1, 2}, {1, 3, 2}});
}

// Six nodes and only two triangles, (1, 2, 3) and (1, 3, 4): nodes 0 and 5 belong to no triangle.
Mesh2D mesh_with_nodes_outside_every_triangle() {
    return Mesh2D({5.0, 0.0, 1.0, 1.0, 0.0, 7.0}, {5.0, 0.0, 0.0, 1.0, 1.0, 7.0}, {{1, 2, 3}, {1, 3, 4}});
}

// The meshes in which every node belongs to at least one triangle: six rectangles and three meshes that are not rectangles.
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

// All the meshes of sample_meshes() plus the one with two nodes outside every triangle.
std::vector<Mesh2D> sample_meshes_with_isolated_nodes() {
    std::vector<Mesh2D> meshes = sample_meshes();
    meshes.push_back(mesh_with_nodes_outside_every_triangle());
    return meshes;
}

// Hand-made materials (3 x 3, indices xx, yy, xy), symmetric: the thermal force does not need them to be physical.
Matrix3 hooke_quarter() {  // E = 1, nu = 1/4: lambda = mu = 0.4
    return {{{1.2, 0.4, 0.0}, {0.4, 1.2, 0.0}, {0.0, 0.0, 0.4}}};
}
Matrix3 general() {  // all the numbers different, also the coupling with shear
    return {{{2.0, 0.5, 0.1}, {0.5, 3.0, -0.2}, {0.1, -0.2, 1.0}}};
}
Matrix3 steel() {  // a real, stiff material: numbers of order 1e11
    return plane_strain_matrix(210e9, 0.3);
}

// A material for the thermal force: the matrix D, the Poisson ratio nu and the expansion coefficient alpha
// (D and nu reach the function separately, so non-physical materials are tried too: negative nu and alpha).
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

// The largest absolute value of a vector.
double largest_entry(const std::vector<double>& v) {
    double m = 0.0;
    for (double a : v) {
        m = std::max(m, std::fabs(a));
    }
    return m;
}

// A temperature that changes from node to node with no simple rule (around 300, with deviations of tens of kelvin).
std::vector<double> wavy_temperature(const Mesh2D& mesh) {
    std::vector<double> T(mesh.num_nodes());
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        const double x = mesh.x(I);
        const double y = mesh.y(I);
        T[I] = 300.0 + 40.0 * std::sin(1.7 * x + 0.4 * y) + 25.0 * std::cos(0.9 * y) - 3.0 * x * y +
               0.5 * static_cast<double>(I % 7);
    }
    return T;
}

// Another temperature, different from the previous one.
std::vector<double> other_temperature(const Mesh2D& mesh) {
    std::vector<double> T(mesh.num_nodes());
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        T[I] = 280.0 + 10.0 * std::cos(2.0 * mesh.x(I)) - 4.0 * mesh.y(I) + 1.5 * static_cast<double>(I % 3);
    }
    return T;
}

// F for a material m.
std::vector<double> thermal_load(const Mesh2D& mesh, const Material& m, const std::vector<double>& T, double T_ref) {
    return assemble_thermal_load_2d(mesh, m.D, m.nu, m.alpha, T, T_ref);
}

// A linear displacement field:  u = a0 + a1 x + a2 y,  v = b0 + b1 x + b2 y.
// At every point it has the same strain (eps_xx, eps_yy, gamma_xy) = (a1, b2, a2 + b1).
struct LinearField {
    double a0, a1, a2, b0, b1, b2;
};

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

// The rigid motions: two translations, a rotation about the origin (u = -y, v = x) and a combination.
std::vector<LinearField> rigid_motions() {
    return {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},     // translation in x
            {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},     // translation in y
            {0.0, 0.0, -1.0, 0.0, 1.0, 0.0},    // rotation
            {0.3, 0.0, -0.7, -1.2, 0.7, 0.0}};  // translation and rotation together
}

// The six "pure" fields (u = x, v = y, u = y, v = x and the two translations) plus one with all the numbers different from 0 and
// small strains, like the real ones (1e-3).
std::vector<LinearField> strain_fields() {
    return {{0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
            {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
            {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
            {0.3, 1.0e-3, -2.0e-3, -0.2, 3.0e-3, 5.0e-4}};
}

// The work u^T F and the sum of the absolute values of its terms (the "scale" of a work that can come out as 0 by
// cancellation).
struct Work {
    double value;
    double scale;
};

Work work_of(const std::vector<double>& u, const std::vector<double>& F) {
    Work w{0.0, 0.0};
    for (std::size_t i = 0; i < u.size() && i < F.size(); ++i) {
        w.value += u[i] * F[i];
        w.scale += std::fabs(u[i] * F[i]);
    }
    return w;
}

// The integral of T - T_ref over the plate, for a temperature that is linear in each triangle: for each triangle it is the area
// times the mean of the three values at the nodes (exact).
double integral_of_temperature_change(const Mesh2D& mesh, const std::vector<double>& T, double T_ref) {
    double integral = 0.0;
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Triangle& t = mesh.triangle(e);
        integral += mesh.area(e) * ((T[t[0]] + T[t[1]] + T[t[2]]) / 3.0 - T_ref);
    }
    return integral;
}

// The key of a side, without direction: its two nodes in increasing order.
std::pair<std::size_t, std::size_t> edge_key(std::size_t p, std::size_t q) {
    return {std::min(p, q), std::max(p, q)};
}

// The forces a CONSTANT stress sigma = (sxx, syy, sxy) produces at the nodes, computed from the boundary, not the triangles.
// Every side in a single triangle is a boundary side: if it goes from node p to node q (counter-clockwise in its
// triangle) and (dx, dy) = q - p, its outward normal times the length is (dy, -dx) and it pushes each of its two
// nodes with the force  (1/2) sigma (dy, -dx) = (1/2) (sxx dy - sxy dx, sxy dy - syy dx). The interior sides, which are in
// two triangles, cancel; the interior nodes and those outside every triangle stay at 0.
std::vector<double> boundary_forces(const Mesh2D& mesh, const Vector3& sigma) {
    std::map<std::pair<std::size_t, std::size_t>, int> uses;
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Triangle& t = mesh.triangle(e);
        for (std::size_t a = 0; a < 3; ++a) {
            ++uses[edge_key(t[a], t[(a + 1) % 3])];
        }
    }
    std::vector<double> F(2 * mesh.num_nodes(), 0.0);
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Triangle& t = mesh.triangle(e);
        for (std::size_t a = 0; a < 3; ++a) {
            const std::size_t p = t[a];
            const std::size_t q = t[(a + 1) % 3];
            if (uses.at(edge_key(p, q)) != 1) {
                continue;
            }
            const double dx = mesh.x(q) - mesh.x(p);
            const double dy = mesh.y(q) - mesh.y(p);
            const double fx = 0.5 * (sigma[0] * dy - sigma[2] * dx);
            const double fy = 0.5 * (sigma[2] * dy - sigma[1] * dx);
            F[2 * p] += fx;
            F[2 * p + 1] += fy;
            F[2 * q] += fx;
            F[2 * q + 1] += fy;
        }
    }
    return F;
}

// The same mesh with the three nodes of each triangle written starting from another one: (a, b, c) becomes (b, c, a) if shift
// is 1 and (c, a, b) if shift is 2. They are the same triangles, still counter-clockwise.
Mesh2D with_rotated_triangles(const Mesh2D& mesh, std::size_t shift) {
    std::vector<double> xs;
    std::vector<double> ys;
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        xs.push_back(mesh.x(I));
        ys.push_back(mesh.y(I));
    }
    std::vector<Triangle> triangles;
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Triangle& t = mesh.triangle(e);
        triangles.push_back({t[shift % 3], t[(shift + 1) % 3], t[(shift + 2) % 3]});
    }
    return Mesh2D(xs, ys, triangles);
}

// The same mesh with the triangles listed in reverse order.
Mesh2D with_reversed_triangle_order(const Mesh2D& mesh) {
    std::vector<double> xs;
    std::vector<double> ys;
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        xs.push_back(mesh.x(I));
        ys.push_back(mesh.y(I));
    }
    std::vector<Triangle> triangles;
    for (std::size_t e = mesh.num_triangles(); e > 0; --e) {
        triangles.push_back(mesh.triangle(e - 1));
    }
    return Mesh2D(xs, ys, triangles);
}

// The "slow computation": first the force of each triangle (with the mean of the three nodes written backwards), then for each
// global degree of freedom P we look, in all the triangles and in all their six local degrees of freedom, for those
// with global number P and add up their forces. It is only used for comparison, and only for small meshes.
std::vector<double> slow_thermal_load(const Mesh2D& mesh, const Matrix3& D, double nu, double alpha,
                                      const std::vector<double>& T, double T_ref) {
    std::vector<Vector6> forces;
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Triangle& t = mesh.triangle(e);
        const double mean = T[t[2]] / 3.0 + T[t[1]] / 3.0 + T[t[0]] / 3.0;
        forces.push_back(element_thermal_load(mesh, e, D, plane_strain_thermal_strain(nu, alpha, mean - T_ref)));
    }
    std::vector<double> F(2 * mesh.num_nodes(), 0.0);
    for (std::size_t P = 0; P < F.size(); ++P) {
        for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
            const Triangle& t = mesh.triangle(e);
            for (std::size_t a = 0; a < 3; ++a) {
                for (std::size_t c = 0; c < 2; ++c) {
                    if (2 * t[a] + c == P) {
                        F[P] += forces[e][2 * a + c];
                    }
                }
            }
        }
    }
    return F;
}

// Compare two vectors element by element, with an absolute tolerance.
void expect_vectors_near(const std::vector<double>& got, const std::vector<double>& want, double tolerance) {
    ASSERT_EQ(got.size(), want.size()) << "size";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tolerance) << "degree of freedom " << i;
    }
}

}  // namespace

// EXAMPLE 157. By hand. The unit square has 4 nodes, 0 = (0, 0), 1 = (1, 0), 2 = (0, 1), 3 = (1, 1), and two triangles,
// T0 = (0, 1, 2) and T1 = (1, 3, 2), both of area 1/2. The material has E = 1 and nu = 1/4 (lambda = mu = 0.4, so
// D = [[1.2, 0.4, 0], [0.4, 1.2, 0], [0, 0, 0.4]]) and alpha = 1 (a convenient number, not a real material); the
// reference temperature is T_ref = 5 and that of the nodes T = (5, 6, 7, 8).
// T0 has nodes (0, 1, 2): the mean is (5 + 6 + 7) / 3 = 6, so dT = 1. T1 has nodes (1, 3, 2): (6 + 8 + 7) / 3 = 7,
// so dT = 2. The thermal strain is (1 + nu) alpha dT (1, 1, 0) = 1.25 dT (1, 1, 0) and the stress is
// D eps_th = 1.25 dT (1.6, 1.6, 0) = 2 dT (1, 1, 0): sigma = (2, 2, 0) in T0 and (4, 4, 0) in T1.
// For a stress (s, s, 0) the forces of the triangle, A B^T sigma, are (s / 2) (b_0, c_0, b_1, c_1, b_2, c_2), with
// b_i = y_p - y_q and c_i = x_q - x_p (p and q are the two nodes after node i). In T0 the (b_i, c_i) are (-1, -1), (1, 0),
// (0, 1); in T1, with nodes (1, 3, 2), they are (0, -1), (1, 1), (-1, 0). So
//     f_0 = (-1, -1, 1, 0, 0, 1)        (s = 2),        f_1 = (0, -2, 2, 2, -2, 0)        (s = 4).
// The local degrees of freedom of T0 are the global ones (0, 1, 2, 3, 4, 5), those of T1 the global ones (2, 3, 6, 7, 4, 5). Each
// number is added into its place, for example  F[2] = f_0[2] + f_1[0] = 1 + 0 = 1  and  F[3] = f_0[3] + f_1[1] = 0 - 2:
//     F = (-1, -1, 1, -2, -2, 1, 2, 2).
// The forces on the x components add up to -1 + 1 - 2 + 2 = 0 and those on the y ones to 0: the free plate has no
// external forces. With the uniform temperature T = (7, 7, 7, 7) (dT = 2 in both triangles) the four corners are
// pushed outwards along the diagonals, F = (-2, -2, 2, -2, -2, 2, 2, 2).
TEST(ElasticityLoad2d, ThermalLoadOfTwoRightTrianglesMatchesHandComputation) {
    const Mesh2D mesh = unit_square();
    const Matrix3 D = hooke_quarter();

    const std::vector<double> got = assemble_thermal_load_2d(mesh, D, 0.25, 1.0, {5.0, 6.0, 7.0, 8.0}, 5.0);
    const std::vector<double> want = {-1.0, -1.0, 1.0, -2.0, -2.0, 1.0, 2.0, 2.0};
    ASSERT_EQ(got.size(), want.size()) << "size";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], 1e-14) << "degree of freedom " << i;
    }

    const std::vector<double> got_uniform = assemble_thermal_load_2d(mesh, D, 0.25, 1.0, {7.0, 7.0, 7.0, 7.0}, 5.0);
    const std::vector<double> want_uniform = {-2.0, -2.0, 2.0, -2.0, -2.0, 2.0, 2.0, 2.0};
    ASSERT_EQ(got_uniform.size(), want_uniform.size()) << "size";
    for (std::size_t i = 0; i < want_uniform.size(); ++i) {
        EXPECT_NEAR(got_uniform[i], want_uniform[i], 1e-14) << "degree of freedom " << i << " (uniform temperature)";
    }
}

// EXAMPLE 158. Uniform heating. If T is the same at all nodes, every triangle has the same dT, the same
// thermal strain eps_th = (1 + nu) alpha dT (1, 1, 0) and therefore the same stress sigma = D eps_th. The forces of
// a constant stress on a triangle are A B^T sigma: node i receives (1/2) sigma (b_i, c_i), and (b_i, c_i) is the
// sum of the outward normals (multiplied by the length) of the two sides that reach the node. Summing the triangles
// the interior sides appear twice, with opposite normals, and cancel: only the BOUNDARY SIDES remain. A boundary
// side, from node p to node q counter-clockwise, with (dx, dy) = q - p, has outward normal times length (dy, -dx) and
// pushes each of its two nodes with the force (1/2) sigma (dy, -dx). For example, with sigma = (s, s, 0), the right
// vertical side of a plate, of length l, pushes its two nodes with (s l / 2, 0) each. The interior nodes (and those
// outside every triangle) have force 0. The comparison uses the boundary formula, which does not go through the
// functions under test. Tried on ten meshes and three materials, with T = 350 and T_ref = 293.15.
TEST(ElasticityLoad2d, UniformHeatingPushesTheBoundaryOutwardAndLeavesTheInteriorBalanced) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    const std::vector<Material> materials = sample_materials();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        for (std::size_t m = 0; m < materials.size(); ++m) {
            SCOPED_TRACE("mesh number " + std::to_string(a) + ", material number " + std::to_string(m));
            const Mesh2D& mesh = meshes[a];
            const Material& mat = materials[m];

            const double e = (1.0 + mat.nu) * mat.alpha * (350.0 - reference_temperature);
            const Vector3 sigma = {e * (mat.D[0][0] + mat.D[0][1]), e * (mat.D[1][0] + mat.D[1][1]),
                                   e * (mat.D[2][0] + mat.D[2][1])};
            const std::vector<double> want = boundary_forces(mesh, sigma);
            ASSERT_GT(largest_entry(want), 0.0);

            const std::vector<double> got = thermal_load(mesh, mat, std::vector<double>(mesh.num_nodes(), 350.0),
                                                         reference_temperature);
            expect_vectors_near(got, want, 1e-12 * largest_entry(want));
        }
    }
}

// EXAMPLE 159. Equilibrium. A rigid motion, a translation or a rotation, deforms no triangle and therefore does no
// work against the thermal forces: u^T F = 0 for u = (1, 0, 1, 0, ...), (0, 1, 0, 1, ...), (-y_I, x_I, ...)
// and their combinations, whatever the temperature. With the three choices this means that the sum of the forces in x is
// 0, that of the forces in y is 0 and so is the moment, the sum of x F_y - y F_x: a free body that heats up
// has no external forces, does not move and does not start to rotate. This is what breaks if u and v are swapped or if a
// degree of freedom ends up at the wrong node. Tried on ten meshes, three materials and the "wavy" temperature.
TEST(ElasticityLoad2d, ThermalForcesAreInEquilibriumAndRigidMotionsDoNoWork) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    const std::vector<Material> materials = sample_materials();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        for (std::size_t m = 0; m < materials.size(); ++m) {
            SCOPED_TRACE("mesh number " + std::to_string(a) + ", material number " + std::to_string(m));
            const Mesh2D& mesh = meshes[a];
            const std::vector<double> F =
                thermal_load(mesh, materials[m], wavy_temperature(mesh), reference_temperature);
            ASSERT_EQ(F.size(), 2 * mesh.num_nodes());
            ASSERT_GT(largest_entry(F), 0.0);

            const std::vector<LinearField> motions = rigid_motions();
            for (std::size_t r = 0; r < motions.size(); ++r) {
                const Work w = work_of(nodal_displacements(mesh, motions[r]), F);
                EXPECT_NEAR(w.value, 0.0, 1e-12 * w.scale) << "rigid motion number " << r;
            }
        }
    }
}

// EXAMPLE 160. The work of a linear field. For a linear displacement u = a0 + a1 x + a2 y, v = b0 + b1 x + b2 y
// the strain is the same everywhere, eps = (a1, b2, a2 + b1), and the work of the thermal forces is the sum, over the
// triangles, of A eps^T D eps_th,e. With eps_th,e = (1 + nu) alpha dT_e (1, 1, 0) we get
//     u^T F = (1 + nu) alpha (eps^T D (1, 1, 0)) * sum_e A_e dT_e = (1 + nu) alpha (eps^T D (1, 1, 0)) * I,
// with I the integral of T - T_ref over the plate (A_e dT_e, with dT_e the mean of the three nodes, is the exact integral
// over the triangle of a linear temperature). The work depends on the temperature ONLY through I, and not on the translation
// of the field (a0, b0). The comparison does not go through any tested function and checks together the mean of the nodes,
// T_ref, the factor (1 + nu) alpha and the place of the forces. Tried with seven fields on ten meshes, three materials and the
// "wavy" temperature.
TEST(ElasticityLoad2d, ThermalForcesDoTheWorkOfALinearDisplacementField) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    const std::vector<Material> materials = sample_materials();
    const std::vector<LinearField> fields = strain_fields();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        const Mesh2D& mesh = meshes[a];
        const std::vector<double> T = wavy_temperature(mesh);
        const double integral = integral_of_temperature_change(mesh, T, reference_temperature);
        for (std::size_t m = 0; m < materials.size(); ++m) {
            const Material& mat = materials[m];
            const std::vector<double> F = thermal_load(mesh, mat, T, reference_temperature);
            for (std::size_t f = 0; f < fields.size(); ++f) {
                SCOPED_TRACE("mesh number " + std::to_string(a) + ", material number " + std::to_string(m) +
                             ", field number " + std::to_string(f));
                const LinearField& field = fields[f];
                const double eps[3] = {field.a1, field.b2, field.a2 + field.b1};
                double eps_D_ones = 0.0;
                for (std::size_t i = 0; i < 3; ++i) {
                    eps_D_ones += eps[i] * (mat.D[i][0] + mat.D[i][1]);
                }
                const double want = (1.0 + mat.nu) * mat.alpha * eps_D_ones * integral;

                const Work w = work_of(nodal_displacements(mesh, field), F);
                EXPECT_NEAR(w.value, want, 1e-12 * w.scale);
            }
        }
    }
}

// EXAMPLE 161. A free body, heated uniformly, expands without any stress. The displacement is that
// of the thermal expansion, u = e x, v = e y, with e = (1 + nu) alpha dT: it is linear, so its strain is
// exactly eps_th = e (1, 1, 0) in every triangle and the stress D (eps - eps_th) is 0 everywhere. The deformation forces
// are K u (the stiffness times the displacement) and those of heat are F: if there is no stress they balance,
//     K u = F        (for any D, even a non-physical one).
// It links the two functions, K of the previous step and F of this one, which must use the same degrees of freedom. Tried on
// ten meshes and three materials, with T = 350 and T_ref = 293.15.
TEST(ElasticityLoad2d, AUniformlyHeatedFreeBodyExpandsWithoutStress) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    const std::vector<Material> materials = sample_materials();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        for (std::size_t m = 0; m < materials.size(); ++m) {
            SCOPED_TRACE("mesh number " + std::to_string(a) + ", material number " + std::to_string(m));
            const Mesh2D& mesh = meshes[a];
            const Material& mat = materials[m];

            const std::vector<double> F =
                thermal_load(mesh, mat, std::vector<double>(mesh.num_nodes(), 350.0), reference_temperature);
            ASSERT_GT(largest_entry(F), 0.0);

            const double e = (1.0 + mat.nu) * mat.alpha * (350.0 - reference_temperature);
            const std::vector<double> u = nodal_displacements(mesh, {0.0, e, 0.0, 0.0, 0.0, e});
            const std::vector<double> force = assemble_elasticity_stiffness_2d(mesh, mat.D) * u;
            expect_vectors_near(force, F, 1e-9 * largest_entry(F));
        }
    }
}

// EXAMPLE 162. No variation, no force; and F is linear. The temperature enters only as T - T_ref, and F is
// proportional to (T - T_ref) and to alpha:
//   - if T = T_ref at every node (whatever T_ref is: 0, 0.1, -5, 293.15, 1e6) there is no expansion and F = 0; and also
//     with alpha = 0;
//   - F(T, T_ref) = F(T - T_ref, 0): adding the same number to T and to T_ref changes nothing;
//   - F(T_ref + c (T - T_ref), T_ref) = c F(T, T_ref), for c = 2.5 and for c = -1 (the force changes sign);
//   - F with 2 alpha is twice F;
//   - the sum: if T1 and T2 are two temperatures, F(T1 + T2 - T_ref) = F(T1) + F(T2) (the two fields of
//     T - T_ref add up).
// Tried on ten meshes and three materials.
TEST(ElasticityLoad2d, NoTemperatureChangeMeansNoForceAndTheForceIsLinear) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    const std::vector<Material> materials = sample_materials();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        for (std::size_t m = 0; m < materials.size(); ++m) {
            SCOPED_TRACE("mesh number " + std::to_string(a) + ", material number " + std::to_string(m));
            const Mesh2D& mesh = meshes[a];
            const Material& mat = materials[m];
            const std::size_t n = mesh.num_nodes();
            const double ref = reference_temperature;

            const std::vector<double> T1 = wavy_temperature(mesh);
            const std::vector<double> F = thermal_load(mesh, mat, T1, ref);
            const double scale = largest_entry(F);
            ASSERT_GT(scale, 0.0);

            for (double value : {0.0, 0.1, -5.0, 293.15, 1.0e6}) {
                const std::vector<double> flat = thermal_load(mesh, mat, std::vector<double>(n, value), value);
                EXPECT_LE(largest_entry(flat), 1e-9 * scale) << "T = T_ref = " << value;
            }
            const std::vector<double> no_expansion = assemble_thermal_load_2d(mesh, mat.D, mat.nu, 0.0, T1, ref);
            EXPECT_EQ(largest_entry(no_expansion), 0.0) << "alpha = 0";

            std::vector<double> shifted(n);
            for (std::size_t I = 0; I < n; ++I) {
                shifted[I] = T1[I] - ref;
            }
            expect_vectors_near(thermal_load(mesh, mat, shifted, 0.0), F, 1e-12 * scale);

            for (double c : {2.5, -1.0}) {
                std::vector<double> scaled(n);
                std::vector<double> want(F.size());
                for (std::size_t I = 0; I < n; ++I) {
                    scaled[I] = ref + c * (T1[I] - ref);
                }
                for (std::size_t P = 0; P < F.size(); ++P) {
                    want[P] = c * F[P];
                }
                expect_vectors_near(thermal_load(mesh, mat, scaled, ref), want, 1e-12 * scale * std::fabs(c));
            }

            std::vector<double> doubled(F.size());
            const std::vector<double> twice_alpha =
                assemble_thermal_load_2d(mesh, mat.D, mat.nu, 2.0 * mat.alpha, T1, ref);
            for (std::size_t P = 0; P < F.size(); ++P) {
                doubled[P] = 2.0 * F[P];
            }
            expect_vectors_near(twice_alpha, doubled, 1e-12 * scale);

            const std::vector<double> T2 = other_temperature(mesh);
            std::vector<double> sum_field(n);
            for (std::size_t I = 0; I < n; ++I) {
                sum_field[I] = T1[I] + T2[I] - ref;
            }
            const std::vector<double> F2 = thermal_load(mesh, mat, T2, ref);
            std::vector<double> sum_forces(F.size());
            for (std::size_t P = 0; P < F.size(); ++P) {
                sum_forces[P] = F[P] + F2[P];
            }
            expect_vectors_near(thermal_load(mesh, mat, sum_field, ref), sum_forces,
                                1e-12 * (scale + largest_entry(F2)));
        }
    }
}

// EXAMPLE 163. Writing a triangle starting from another of its nodes, (a, b, c) instead of (b, c, a), or listing the
// triangles in a different order, does not change the mesh and therefore does not change F: the mean of the three nodes
// and the forces of the six degrees of freedom move together. F of the mesh is compared with F of the same mesh with the
// triangles "rotated" by one place and by two places and with the triangles in reverse order, on ten meshes and three
// materials. Only the order of the sums changes, so the numbers are the same up to a rounding error.
TEST(ElasticityLoad2d, TheOrderOfTheTrianglesAndOfTheirNodesDoesNotMatter) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    const std::vector<Material> materials = sample_materials();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        const Mesh2D& mesh = meshes[a];
        const std::vector<double> T = wavy_temperature(mesh);
        for (std::size_t m = 0; m < materials.size(); ++m) {
            SCOPED_TRACE("mesh number " + std::to_string(a) + ", material number " + std::to_string(m));
            const std::vector<double> F = thermal_load(mesh, materials[m], T, reference_temperature);
            const double scale = largest_entry(F);
            ASSERT_GT(scale, 0.0);

            for (std::size_t shift : {1u, 2u}) {
                const Mesh2D rotated = with_rotated_triangles(mesh, shift);
                expect_vectors_near(thermal_load(rotated, materials[m], T, reference_temperature), F,
                                    1e-12 * scale);
            }
            const Mesh2D reversed = with_reversed_triangle_order(mesh);
            expect_vectors_near(thermal_load(reversed, materials[m], T, reference_temperature), F, 1e-12 * scale);
        }
    }
}

// EXAMPLE 164. Nodes outside every triangle. The mesh has 6 nodes, but only (1, 2, 3) and (1, 3, 4) are in a triangle: nodes
// 0 and 5 have nothing to assemble. F still has 12 numbers, two per node, and the forces of nodes 0 and 5, i.e.
// F[0], F[1], F[10], F[11], are 0. Their temperature is not even read: if any number is put at those nodes,
// even NaN, infinity or 1e300, F does not change by even a bit.
TEST(ElasticityLoad2d, NodesOutsideEveryTriangleGetNoForceAndTheirTemperatureIsNotRead) {
    const Mesh2D mesh = mesh_with_nodes_outside_every_triangle();
    const std::vector<double> clean = {0.0, 350.0, 330.0, 360.0, 320.0, 0.0};
    const std::vector<double> want =
        assemble_thermal_load_2d(mesh, hooke_quarter(), 0.25, 1.0e-3, clean, reference_temperature);
    ASSERT_EQ(want.size(), 12u);
    ASSERT_GT(largest_entry(want), 0.0);
    for (std::size_t P : {0u, 1u, 10u, 11u}) {
        EXPECT_EQ(want[P], 0.0) << "degree of freedom " << P;
    }

    const double infinity = std::numeric_limits<double>::infinity();
    for (double garbage : {std::numeric_limits<double>::quiet_NaN(), infinity, -infinity, 1.0e300, -7.0}) {
        std::vector<double> T = clean;
        T[0] = garbage;
        T[5] = -garbage;
        const std::vector<double> got =
            assemble_thermal_load_2d(mesh, hooke_quarter(), 0.25, 1.0e-3, T, reference_temperature);
        ASSERT_EQ(got.size(), want.size());
        for (std::size_t P = 0; P < want.size(); ++P) {
            EXPECT_EQ(got[P], want[P]) << "degree of freedom " << P << ", temperature " << garbage;
        }
    }
}

// EXAMPLE 165. The "slow computation": first the force of each triangle, with the mean of the three nodes written
// backwards, then, for each global degree of freedom P, we look among all the triangles and all their six local degrees
// of freedom for those that have global number P (2 t[a] + c == P, with a the local node and c the component) and add
// up their forces. It is the same computation as the global degrees of freedom above, written with three loops instead
// of one. The assembled forces must be the same. The meshes: six rectangles, the triangle with a node inside, the
// irregular quadrilateral, two triangles of different area and the mesh with two nodes outside every triangle; the
// materials are three, the temperature is the "wavy" one and the "different" one.
TEST(ElasticityLoad2d, ThermalLoadMatchesTheSlowAssemblyOnEveryKindOfMesh) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    const std::vector<Material> materials = sample_materials();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        for (std::size_t m = 0; m < materials.size(); ++m) {
            SCOPED_TRACE("mesh number " + std::to_string(a) + ", material number " + std::to_string(m));
            const Mesh2D& mesh = meshes[a];
            const Material& mat = materials[m];
            for (const std::vector<double>& T : {wavy_temperature(mesh), other_temperature(mesh)}) {
                const std::vector<double> slow = slow_thermal_load(mesh, mat.D, mat.nu, mat.alpha, T,
                                                                   reference_temperature);
                ASSERT_GT(largest_entry(slow), 0.0);
                expect_vectors_near(thermal_load(mesh, mat, T, reference_temperature), slow,
                                    1e-12 * largest_entry(slow));
            }
        }
    }
}

// EXAMPLE 166. Wrong data. T must have one number per node (the unit square has 4 nodes: 0, 1, 3, 5 or 8 numbers
// are wrong) and nu must be between -1 and 1/2, ends excluded (NaN included): in all these cases the function
// throws std::invalid_argument and not another kind of error. With nu just inside (-0.99, 0.49) all is well. The
// other data are not checked: alpha can be 0 or negative, the temperature can be negative and T_ref
// anything.
TEST(ElasticityLoad2d, BadInputsThrowInvalidArgument) {
    const Mesh2D mesh = unit_square();
    const Matrix3 D = hooke_quarter();
    const std::vector<double> T = {5.0, 6.0, 7.0, 8.0};

    EXPECT_NO_THROW(assemble_thermal_load_2d(mesh, D, 0.25, 1.0, T, 5.0));

    for (std::size_t size : {0u, 1u, 3u, 5u, 8u}) {
        EXPECT_THROW(assemble_thermal_load_2d(mesh, D, 0.25, 1.0, std::vector<double>(size, 5.0), 5.0),
                     std::invalid_argument)
            << size << " temperature";
    }

    const double not_a_number = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    for (double nu : {-1.0, -2.0, 0.5, 0.75, 1.0, not_a_number, infinity, -infinity}) {
        EXPECT_THROW(assemble_thermal_load_2d(mesh, D, nu, 1.0, T, 5.0), std::invalid_argument) << "nu = " << nu;
    }
    EXPECT_NO_THROW(assemble_thermal_load_2d(mesh, D, -0.99, 1.0, T, 5.0));
    EXPECT_NO_THROW(assemble_thermal_load_2d(mesh, D, 0.49, 1.0, T, 5.0));

    EXPECT_NO_THROW(assemble_thermal_load_2d(mesh, D, 0.25, 0.0, T, 5.0));
    EXPECT_NO_THROW(assemble_thermal_load_2d(mesh, D, 0.25, -1.0, T, 5.0));
    EXPECT_NO_THROW(assemble_thermal_load_2d(mesh, D, 0.25, 1.0, {-40.0, 6.0, -7.0, 8.0}, -300.0));
}

// EXAMPLE 167. A large mesh. A 1 x 1 square of 100 x 100 cells has 101 x 101 = 10201 nodes and 20000 triangles: F
// has 20402 numbers. The temperature is linear, T = T_ref + 10 + 100 x, so the integral of T - T_ref is 10 + 100 / 2 =
// 60 (area 1) for any fineness. The work is that of EXAMPLE 160, (1 + nu) alpha (eps^T D (1, 1, 0)) 60: for
// u = x (eps = (1, 0, 0)) it is (1 + nu) alpha (D00 + D01) 60, for v = y (eps = (0, 1, 0)) it is (1 + nu) alpha (D10 + D11)
// 60, for u = y (eps = (0, 0, 1)) it is (1 + nu) alpha (D20 + D21) 60, which for steel gives 0. Rigid motions do no
// work, so the forces in x and in y add up to 0 and so does the moment. Steel.
TEST(ElasticityLoad2d, AssemblyOfALargeMeshIsRight) {
    const Mesh2D mesh = rectangle_mesh(1.0, 1.0, 100, 100);
    ASSERT_EQ(mesh.num_nodes(), 10201u);
    const Matrix3 D = steel();
    const double nu = 0.3;
    const double alpha = 12.0e-6;

    std::vector<double> T(mesh.num_nodes());
    for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
        T[I] = reference_temperature + 10.0 + 100.0 * mesh.x(I);
    }
    const std::vector<double> F = assemble_thermal_load_2d(mesh, D, nu, alpha, T, reference_temperature);
    ASSERT_EQ(F.size(), 20402u);
    ASSERT_GT(largest_entry(F), 0.0);

    const double common = (1.0 + nu) * alpha * 60.0;
    const Work along_x = work_of(nodal_displacements(mesh, {0.0, 1.0, 0.0, 0.0, 0.0, 0.0}), F);
    EXPECT_NEAR(along_x.value, common * (D[0][0] + D[0][1]), 1e-11 * along_x.scale);
    const Work along_y = work_of(nodal_displacements(mesh, {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}), F);
    EXPECT_NEAR(along_y.value, common * (D[1][0] + D[1][1]), 1e-11 * along_y.scale);
    const Work shear = work_of(nodal_displacements(mesh, {0.0, 0.0, 1.0, 0.0, 0.0, 0.0}), F);
    EXPECT_NEAR(shear.value, common * (D[2][0] + D[2][1]), 1e-11 * shear.scale);

    const std::vector<LinearField> motions = rigid_motions();
    for (std::size_t r = 0; r < motions.size(); ++r) {
        const Work w = work_of(nodal_displacements(mesh, motions[r]), F);
        EXPECT_NEAR(w.value, 0.0, 1e-11 * w.scale) << "rigid motion number " << r;
    }
}

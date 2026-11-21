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

// ---- From forces to displacements: solving K u = F -----------------------------------------------------------------
//
// With the matrix K (assemble_elasticity_stiffness_2d) and the forces F (for example the thermal ones,
// assemble_thermal_load_2d) the displacements of the plate, u, are the solution of K u = F. For heat it is the same
// step as solve_steady_2d, with one difference that changes everything: K alone CANNOT be solved. A free body
// moves and rotates without deforming, and the rigid motions (in 2D there are 3: two translations and one
// rotation) have K u = 0: if u solves K u = F, so does u + (any rigid motion). To have ONE solution constraints are
// needed, i.e. IMPOSED displacements on some degrees of freedom: at least 3, placed so as to block all 3 motions.
//
// solve_displacement_2d(mesh, D, F, fixed_dofs, fixed_values) returns u (2 n numbers: u of node I at number 2 I and
// v at 2 I + 1). fixed_dofs are numbers of degrees of freedom, not of nodes: to block node I completely, 2 I
// and 2 I + 1 are needed. As for heat the system is A u = b, with A from dirichlet_matrix_triplets_2d and b from
// dirichlet_rhs_2d (the same two functions, with degrees of freedom in place of nodes), solved with the conjugate
// gradient with the Jacobi preconditioner (tolerance 1e-12, at most 10 times as many iterations as degrees of
// freedom), starting from the imposed displacements: this way the fixed degrees of freedom come out with the imposed
// value EXACTLY. There are two novelties with respect to heat.
//   - K and F are DIVIDED by the largest absolute value of the triplets of K, before the CG (u does not
//     change). Without it, a stiff material (steel, E = 210e9: K has numbers of order 1e11) makes the CG reject K,
//     since it checks the symmetry with an absolute tolerance of 1e-12 and finds the rounding errors (1e-16 relative,
//     so 1e-5 absolute); and a very soft one (E = 1e-12) makes the CG stop at once and return a wrong solution
//     (EXAMPLE 176).
//   - At the end the true residual b - A u is recomputed and std::runtime_error is thrown if norm(b - A u) >
//     1e-6 norm(b): the CG updates its residual with a formula, and on an impossible system (constraints that leave
//     a rotation free and a load that makes it turn) it brings it to 0 while u becomes huge (EXAMPLE 179).
//
// What it must do, and this is what the tests check:
//   - on the unit square three problems are solved by hand (EXAMPLE 168);
//   - the stretching of a rectangle has a textbook solution, for every E and nu (EXAMPLE 169);
//   - the imposed displacements come out identical, in any order (EXAMPLE 170);
//   - a linear field imposed on the boundary is reproduced inside too, on every kind of mesh (EXAMPLE 171);
//   - a free body heated uniformly expands as expected (EXAMPLE 172), and if it is clamped along the whole
//     boundary it does not move (EXAMPLE 173);
//   - u really satisfies K u = F, for any load and imposed displacement (EXAMPLE 174), and it is linear in both
//     (EXAMPLE 175);
//   - the result does not depend on the units of Young's modulus (EXAMPLE 176);
//   - the nodes outside every triangle must be fixed (EXAMPLE 177);
//   - wrong data throw std::invalid_argument (EXAMPLE 178) and systems with no solution std::runtime_error
//     (EXAMPLE 179);
//   - on a large mesh the CG converges and the result is exact (EXAMPLE 180).

namespace {

// A rectangle lx x ly divided into nx x ny cells, as in rectangle_mesh.
struct Grid {
    double lx, ly;
    std::size_t nx, ny;
};

// Six rectangles of very different shapes (the same as in the assembly tests).
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

// Two SEPARATE triangles, with no node in common: (0, 1, 2) and (3, 4, 5). Two bodies, each with its 3 rigid motions.
Mesh2D two_separate_triangles() {
    return Mesh2D({0.0, 1.0, 0.0, 3.0, 4.0, 3.0}, {0.0, 0.0, 1.0, 0.0, 0.0, 1.0}, {{0, 1, 2}, {3, 4, 5}});
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

// The largest absolute value of a vector.
double largest_entry(const std::vector<double>& v) {
    double m = 0.0;
    for (double a : v) {
        m = std::max(m, std::fabs(a));
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

// Four degrees of freedom: u and v of p and u and v of the node farthest from p (two nodes blocked completely).
std::vector<std::size_t> two_node_clamp(const Mesh2D& mesh, std::size_t p) {
    const std::size_t q = farthest_node(mesh, p);
    return {2 * p, 2 * p + 1, 2 * q, 2 * q + 1};
}

// The degrees of freedom (u and v) of all the boundary nodes.
std::vector<std::size_t> boundary_dofs(const Mesh2D& mesh) {
    std::vector<std::size_t> dofs;
    for (std::size_t I : boundary_nodes(mesh)) {
        dofs.push_back(2 * I);
        dofs.push_back(2 * I + 1);
    }
    return dofs;
}

// The values of u at the degrees of freedom dofs, in the same order.
std::vector<double> pick(const std::vector<double>& u, const std::vector<std::size_t>& dofs) {
    std::vector<double> values;
    for (std::size_t d : dofs) {
        values.push_back(u[d]);
    }
    return values;
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

// Forces (or any numbers) that change from one degree of freedom to the next with no simple rule, of amplitude
// amplitude.
std::vector<double> wavy_vector(std::size_t size, double amplitude) {
    std::vector<double> v(size);
    for (std::size_t i = 0; i < size; ++i) {
        const double t = static_cast<double>(i);
        v[i] = amplitude * (std::sin(1.3 * t + 0.4) + 0.3 * std::cos(2.9 * t));
    }
    return v;
}

// A problem with a known exact solution: the forces, the fixed degrees of freedom with their imposed displacements
// (taken from the exact solution) and the exact solution itself (2 n numbers).
struct Problem {
    std::vector<double> F;
    std::vector<std::size_t> dofs;
    std::vector<double> values;
    std::vector<double> exact;
};

// A FREE body heated uniformly by dT (T = T_ref + dT at all the nodes) and blocked only in the three rigid
// motions (isostatic_dofs about node p). There are no stresses: the body expands by epsilon_th = c (1, 1, 0) with
// c = (1 + nu) alpha dT, and the exact solution is u = (a0 + c x, b0 + c y) where (a0, b0) is the displacement of a
// translation that the constraints impose (with (a0, b0) = (0, 0) and p = 0 the imposed values are all 0). With this
// displacement B u = epsilon_th in every triangle and therefore K u = F for any D.
Problem heated_free_body(const Mesh2D& mesh, const Material& m, double dT, std::size_t p, double a0, double b0) {
    const double c = (1.0 + m.nu) * m.alpha * dT;
    Problem pr;
    pr.F = assemble_thermal_load_2d(mesh, m.D, m.nu, m.alpha,
                                    std::vector<double>(mesh.num_nodes(), reference_temperature + dT),
                                    reference_temperature);
    pr.exact = nodal_displacements(mesh, LinearField{a0, c, 0.0, b0, 0.0, c});
    pr.dofs = isostatic_dofs(mesh, p);
    pr.values = pick(pr.exact, pr.dofs);
    return pr;
}

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

// Compares two vectors element by element, with an absolute tolerance.
void expect_vectors_near(const std::vector<double>& got, const std::vector<double>& want, double tolerance) {
    ASSERT_EQ(got.size(), want.size()) << "size";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tolerance) << "degree of freedom " << i;
    }
}

}  // namespace

// EXAMPLE 168. By hand. The unit square has 4 nodes, 0 = (0, 0), 1 = (1, 0), 2 = (0, 1), 3 = (1, 1), and two triangles,
// (0, 1, 2) and (1, 3, 2). The material has E = 1 and nu = 1/4 (lambda = mu = 0.4, so
// D = [[1.2, 0.4, 0], [0.4, 1.2, 0], [0, 0, 0.4]]). The degrees of freedom are (u0, v0, u1, v1, u2, v2, u3, v3), i.e.
// the numbers 0, ..., 7. The displacement field of each of the three problems is LINEAR, and a linear field the elements
// reproduce exactly: the solution is that of the continuous body.
//   (a) Tension. The left side does not move in x (u0 = u2 = 0, degrees of freedom 0 and 4) and node 0 does not
//       move in y (v0 = 0, number 1: without it, the plate would slide in y). On the two nodes of the right side, 1 and 3,
//       a force of 0.5 in x pulls (numbers 2 and 6): 1 in total, i.e. a stress sigma_xx = 1 on a side of length 1.
//       The continuous body has stress (1, 0, 0), and the strain follows from sigma = D epsilon: with sigma_yy = 0,
//       0.4 eps_xx + 1.2 eps_yy = 0, i.e. eps_yy = -eps_xx / 3, and sigma_xx = 1.2 eps_xx + 0.4 eps_yy = (1.2 - 0.4 /
//       3) eps_xx = (16 / 15) eps_xx = 1: eps_xx = 15/16 and eps_yy = -5/16. The displacements are u = 15/16 x and v =
//       -5/16 y:  u = (u0, v0, ..., v3) = (0, 0, 15/16, 0, 0, -5/16, 15/16, -5/16). The plate lengthens by 15/16 and,
//       because of the Poisson ratio (nu / (1 - nu) = 1/3 in plane strain), it shrinks by a third of
//       this.
//   (b) Thermal expansion. With T = (7, 7, 7, 7), T_ref = 5 and alpha = 1 the thermal strain is epsilon_th =
//       (1 + nu) alpha dT (1, 1, 0) = 1.25 * 2 (1, 1, 0) = 2.5 (1, 1, 0) and the forces (EXAMPLE 157 of the previous
//       step) are F = (-2, -2, 2, -2, -2, 2, 2, 2). The body is free: only the three constraints u0 = v0 = v1 = 0 are
//       needed (numbers 0, 1 and 3; v1 and not u1, because nodes 0 and 1 have the same y and u1 would not prevent the
//       rotation about node 0). The body expands without stresses by 2.5 x and 2.5 y: u = (0, 0, 2.5, 0, 0, 2.5, 2.5, 2.5).
//   (c) Imposed displacement instead of force. Like (a), but with no forces (F = 0) and with the two nodes of the right side
//       displaced by 0.1 in x (u1 = u3 = 0.1, numbers 2 and 6: there are 5 fixed degrees of freedom, 0, 1, 4, 2 and 6).
//       v1, v2 and v3 remain free (numbers 3, 5 and 7). The field u = 0.1 x, v = -(0.1 / 3) y has stress
//       sigma_yy = 0.4 * 0.1 + 1.2 * (-1/30) = 0: no vertical force, so the free vertical degrees of freedom
//       are in equilibrium, and the solution is u = (0, 0, 0.1, 0, 0, -1/30, 0.1, -1/30).
TEST(ElasticitySolve2d, ThreeHandSolvedProblemsOnTheUnitSquare) {
    const Mesh2D mesh = unit_square();
    const Matrix3 D = hooke_quarter();

    const std::vector<double> tension = {0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0};
    expect_vectors_near(solve_displacement_2d(mesh, D, tension, {0, 1, 4}, {0.0, 0.0, 0.0}),
                        {0.0, 0.0, 0.9375, 0.0, 0.0, -0.3125, 0.9375, -0.3125}, 1e-12);

    const std::vector<double> heating = {-2.0, -2.0, 2.0, -2.0, -2.0, 2.0, 2.0, 2.0};
    expect_vectors_near(solve_displacement_2d(mesh, D, heating, {0, 1, 3}, {0.0, 0.0, 0.0}),
                        {0.0, 0.0, 2.5, 0.0, 0.0, 2.5, 2.5, 2.5}, 1e-12);

    const std::vector<double> no_forces(8, 0.0);
    expect_vectors_near(solve_displacement_2d(mesh, D, no_forces, {0, 1, 4, 2, 6}, {0.0, 0.0, 0.0, 0.1, 0.1}),
                        {0.0, 0.0, 0.1, 0.0, 0.0, -1.0 / 30.0, 0.1, -1.0 / 30.0}, 1e-12);
}

// EXAMPLE 169. The stretching of a rectangle with the textbook solution (stretched_rectangle): on the right side the
// stress sigma = 1e-3 E (a strain of 1e-3, like the real ones), on the left u = 0, and at the bottom-left node
// v = 0. In plane strain eps_xx = (1 - nu^2) sigma / E,        eps_yy = -nu (1 + nu) sigma / E, and the
// displacements are u = eps_xx x, v = eps_yy y. It is tried on six rectangles of different shapes (even very flat ones),
// with E = 1 and with E = 210e9 and with nu = 0, 1/4, 0.3, 0.4: all the nodes, not only the extreme ones. With nu = 0
// the plate does not shrink (eps_yy = 0). With E = 1, nu = 1/4 and sigma = 1 on the unit square these are the numbers
// of EXAMPLE 168 (a).
TEST(ElasticitySolve2d, UniaxialTensionOfARectangleMatchesTheTextbookFormulas) {
    for (const Grid& g : sample_grids()) {
        const Mesh2D mesh = rectangle_mesh(g.lx, g.ly, g.nx, g.ny);
        for (double E : {1.0, 210e9}) {
            for (double nu : {0.0, 0.25, 0.3, 0.4}) {
                SCOPED_TRACE("grid " + std::to_string(g.nx) + " x " + std::to_string(g.ny) + ", E = " +
                             std::to_string(E) + ", nu = " + std::to_string(nu));
                const Problem pr = stretched_rectangle(mesh, g.lx, E, nu, 1e-3 * E);
                const std::vector<double> got =
                    solve_displacement_2d(mesh, plane_strain_matrix(E, nu), pr.F, pr.dofs, pr.values);
                expect_vectors_near(got, pr.exact, 1e-9 * largest_entry(pr.exact));
            }
        }
    }
}

// EXAMPLE 170. The imposed displacements come out EXACTLY as they were imposed (equal bit for bit, not "almost"),
// whatever the order in which they are listed and even when all of them are imposed. On a mesh of 4 x 3 cells (20
// nodes, 40 degrees of freedom) eight scattered degrees of freedom are fixed, in no particular order, with arbitrary
// numbers (large, small, negative), among them 0, 1 and 3, which block the rigid motions, and with arbitrary forces on
// the others; then the same thing with the lists in reverse order (same degree of freedom - value pairs): the result is
// the same. Finally all 40 degrees of freedom are fixed: the result is the list of values, and the system has nothing
// to solve.
TEST(ElasticitySolve2d, ImposedDisplacementsComeBackExactlyInAnyOrder) {
    const Mesh2D mesh = rectangle_mesh(3.0, 1.5, 4, 3);
    const Matrix3 D = hooke_quarter();
    const std::size_t N = 2 * mesh.num_nodes();
    ASSERT_EQ(N, 40u);
    const std::vector<double> F = wavy_vector(N, 0.01);

    const std::vector<std::size_t> dofs = {17, 3, 0, 1, 38, 22, 9, 30};
    const std::vector<double> values = {0.1, -7.3, 2.5e-9, 123.456, -0.5, 1.0e-3, 4.0, -1.0e-7};
    const std::vector<double> u = solve_displacement_2d(mesh, D, F, dofs, values);
    ASSERT_EQ(u.size(), N);
    for (std::size_t q = 0; q < dofs.size(); ++q) {
        EXPECT_EQ(u[dofs[q]], values[q]) << "degree of freedom " << dofs[q];
    }

    const std::vector<std::size_t> reversed_dofs(dofs.rbegin(), dofs.rend());
    const std::vector<double> reversed_values(values.rbegin(), values.rend());
    const std::vector<double> v = solve_displacement_2d(mesh, D, F, reversed_dofs, reversed_values);
    expect_vectors_near(v, u, 1e-12 * largest_entry(u));
    for (std::size_t q = 0; q < reversed_dofs.size(); ++q) {
        EXPECT_EQ(v[reversed_dofs[q]], reversed_values[q]) << "degree of freedom " << reversed_dofs[q];
    }

    std::vector<std::size_t> all_dofs;
    for (std::size_t d = 0; d < N; ++d) {
        all_dofs.push_back((d * 7) % N);  // all the numbers from 0 to 39 (7 and 40 are coprime), in shuffled order
    }
    const std::vector<double> all_values = wavy_vector(N, 3.0);
    const std::vector<double> w = solve_displacement_2d(mesh, D, F, all_dofs, all_values);
    ASSERT_EQ(w.size(), N);
    for (std::size_t q = 0; q < N; ++q) {
        EXPECT_EQ(w[all_dofs[q]], all_values[q]) << "degree of freedom " << all_dofs[q];
    }
}

// EXAMPLE 171. The patch test, on every kind of mesh. With a LINEAR displacement field u =
// a0 + a1 x + a2 y, v = b0 + b1 x + b2 y the strain is the same at every point, so the stress too is the
// same everywhere, and inside the body (with no forces) it is in equilibrium. So by imposing the field on
// the degrees of freedom of all the boundary nodes (boundary_nodes), with F = 0, the solution must be the field
// itself at the interior nodes too, for any material D and any shape of the triangles. It is tried on nine meshes
// (six rectangles, the triangle with a node inside, the irregular quadrilateral, two triangles of different area) with
// three materials (a non-physical one, and steel) and with eleven fields: the rigid motions, the six "pure" strains and
// a field with strains of 1e-3 and a translation of 0.3. With no interior nodes everything is imposed: nothing to solve.
TEST(ElasticitySolve2d, ALinearDisplacementFieldIsReproducedOnEveryKindOfMesh) {
    std::vector<LinearField> fields = rigid_motions();
    for (const LinearField& f : strain_fields()) {
        fields.push_back(f);
    }
    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t mi = 0; mi < meshes.size(); ++mi) {
        const Mesh2D& mesh = meshes[mi];
        const std::vector<std::size_t> dofs = boundary_dofs(mesh);
        const std::vector<double> no_forces(2 * mesh.num_nodes(), 0.0);
        for (const Material& m : sample_materials()) {
            for (std::size_t fi = 0; fi < fields.size(); ++fi) {
                SCOPED_TRACE("mesh " + std::to_string(mi) + ", material with nu = " + std::to_string(m.nu) +
                             ", field " + std::to_string(fi));
                const std::vector<double> want = nodal_displacements(mesh, fields[fi]);
                const std::vector<double> got = solve_displacement_2d(mesh, m.D, no_forces, dofs, pick(want, dofs));
                expect_vectors_near(got, want, 1e-9 * largest_entry(want));
            }
        }
    }
}

// EXAMPLE 172. A free body heated uniformly expands without stresses (heated_free_body). With T = T_ref +
// dT at all the nodes the thermal strain is epsilon_th = c (1, 1, 0), c = (1 + nu) alpha dT, and the field u = (a0 +
// c x, b0 + c y) has B u = epsilon_th in every triangle: K u = F for any D and any mesh (a0 and b0 are a
// translation, imposed by the constraints). The constraints are the three of isostatic_dofs and take the values of
// the exact solution: u and v of a node p and one component of another. It is tried on nine meshes, three materials
// (also non-physical: nu and alpha negative) and two heatings (+100 and -60 kelvin), with p = node 0 and p = last node,
// without translation and with the translation (a0, b0) = (0.3, -0.2): 216 problems. With steel the numbers of K are
// of order 1e11 (EXAMPLE 176).
TEST(ElasticitySolve2d, AUniformlyHeatedFreeBodyExpandsLikeTheExactField) {
    const std::vector<Mesh2D> meshes = sample_meshes();
    const std::vector<std::pair<double, double>> shifts = {{0.0, 0.0}, {0.3, -0.2}};
    for (std::size_t mi = 0; mi < meshes.size(); ++mi) {
        const Mesh2D& mesh = meshes[mi];
        for (const Material& m : sample_materials()) {
            for (double dT : {100.0, -60.0}) {
                for (std::size_t p : {static_cast<std::size_t>(0), mesh.num_nodes() - 1}) {
                    for (const auto& shift : shifts) {
                        SCOPED_TRACE("mesh " + std::to_string(mi) + ", nu = " + std::to_string(m.nu) +
                                     ", dT = " + std::to_string(dT) + ", node " + std::to_string(p) +
                                     ", shift " + std::to_string(shift.first));
                        const Problem pr = heated_free_body(mesh, m, dT, p, shift.first, shift.second);
                        const std::vector<double> got =
                            solve_displacement_2d(mesh, m.D, pr.F, pr.dofs, pr.values);
                        expect_vectors_near(got, pr.exact, 1e-9 * largest_entry(pr.exact));
                    }
                }
            }
        }
    }
}

// EXAMPLE 173. The opposite case: a plate heated uniformly and CLAMPED along the whole boundary (all the degrees of
// freedom of the boundary nodes at 0) does not move: the heat would like to expand it, but it cannot. The thermal
// forces (the same as for the free body) cancel at the interior nodes, which are left with no forces, and those on the
// boundary are taken by the constraints. The displacements are 0 at all the nodes (within 1e-9 times the free expansion
// c L, where c = (1 + nu) alpha dT and L is the size of the plate). It is the case from which the stress will be computed
// in the next step: displacement 0 but stress -D epsilon_th. Nine meshes, three materials, two heatings.
TEST(ElasticitySolve2d, AFullyClampedHeatedPlateStaysAtRest) {
    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t mi = 0; mi < meshes.size(); ++mi) {
        const Mesh2D& mesh = meshes[mi];
        const std::vector<std::size_t> dofs = boundary_dofs(mesh);
        const std::vector<double> zeros(dofs.size(), 0.0);
        for (const Material& m : sample_materials()) {
            for (double dT : {100.0, -60.0}) {
                SCOPED_TRACE("mesh " + std::to_string(mi) + ", nu = " + std::to_string(m.nu) +
                             ", dT = " + std::to_string(dT));
                const std::vector<double> F =
                    assemble_thermal_load_2d(mesh, m.D, m.nu, m.alpha,
                                             std::vector<double>(mesh.num_nodes(), reference_temperature + dT),
                                             reference_temperature);
                const double free_expansion = std::fabs((1.0 + m.nu) * m.alpha * dT) * mesh_size(mesh);
                const std::vector<double> got = solve_displacement_2d(mesh, m.D, F, dofs, zeros);
                expect_vectors_near(got, std::vector<double>(2 * mesh.num_nodes(), 0.0), 1e-9 * free_expansion);
            }
        }
    }
}

// EXAMPLE 174. u really satisfies the equations K u = F, for any load and any imposed displacement: not
// only the problems with a solution known in advance. On every mesh and material the four degrees of freedom of
// two nodes are fixed (two_node_clamp: node 0 and the one farthest from it), with imposed displacements different
// from 0 and any vector F (wavy_vector, with an amplitude that moves the nodes by about 1e-2 times the dimensions).
// The residual K u - F must be 0 at the FREE degrees of freedom (within 1e-9 times the scale of the numbers involved,
// i.e. the largest number of K times the largest displacement, plus the largest F); at the fixed ones it is the
// reaction of the constraint, and there is nothing to check. The product K u is that of the K of
// assemble_elasticity_stiffness_2d, with no scaling.
TEST(ElasticitySolve2d, TheSolutionSatisfiesTheEquationsForAnyLoadAndAnyImposedValues) {
    const std::vector<Mesh2D> meshes = sample_meshes();
    for (std::size_t mi = 0; mi < meshes.size(); ++mi) {
        const Mesh2D& mesh = meshes[mi];
        const std::vector<std::size_t> dofs = two_node_clamp(mesh, 0);
        for (const Material& m : sample_materials()) {
            SCOPED_TRACE("mesh " + std::to_string(mi) + ", nu = " + std::to_string(m.nu));
            const CSR K = assemble_elasticity_stiffness_2d(mesh, m.D);
            const double k_scale = largest_entry(K.diag());
            const double size = mesh_size(mesh);
            const std::vector<double> F = wavy_vector(2 * mesh.num_nodes(), 1e-2 * k_scale * size);
            const std::vector<double> values = {0.003 * size, -0.002 * size, 0.001 * size, 0.004 * size};

            const std::vector<double> u = solve_displacement_2d(mesh, m.D, F, dofs, values);
            ASSERT_EQ(u.size(), F.size());
            const std::vector<double> Ku = K * u;
            std::vector<bool> is_fixed(u.size(), false);
            for (std::size_t d : dofs) {
                is_fixed[d] = true;
            }
            const double scale = k_scale * largest_entry(u) + largest_entry(F);
            for (std::size_t i = 0; i < u.size(); ++i) {
                if (!is_fixed[i]) {
                    EXPECT_NEAR(Ku[i], F[i], 1e-9 * scale) << "degree of freedom " << i;
                }
            }
        }
    }
}

// EXAMPLE 175. The solution is LINEAR in the load and in the imposed displacements: with the same constraints, if
// u1 solves the problem (F1, g1) and u2 the problem (F2, g2) (g = the imposed displacements), then u1 + u2 solves
// (F1 + F2, g1 + g2) and s u1 solves (s F1, s g1), for any number s (even negative). It is tried on five meshes (two
// rectangles, the three that are not) with two materials (E = 1 and steel) and with s = -2.5, with three isostatic
// constraints and imposed displacements different from 0.
TEST(ElasticitySolve2d, TheSolutionIsLinearInTheLoadAndInTheImposedValues) {
    const std::vector<Mesh2D> all = sample_meshes();
    const std::vector<std::size_t> chosen = {2, 3, 6, 7, 8};
    for (std::size_t mi : chosen) {
        const Mesh2D& mesh = all[mi];
        const std::vector<std::size_t> dofs = isostatic_dofs(mesh, 0);
        const double size = mesh_size(mesh);
        for (const Matrix3& D : {hooke_quarter(), steel()}) {
            SCOPED_TRACE("mesh " + std::to_string(mi) + ", D00 = " + std::to_string(D[0][0]));
            const double k_scale = D[0][0];
            const std::size_t N = 2 * mesh.num_nodes();
            const std::vector<double> F1 = wavy_vector(N, 1e-2 * k_scale * size);
            std::vector<double> F2(N);
            for (std::size_t i = 0; i < N; ++i) {
                F2[i] = 1e-2 * k_scale * size * std::cos(0.7 * static_cast<double>(i) + 1.0);
            }
            const std::vector<double> g1 = {0.002 * size, -0.001 * size, 0.003 * size};
            const std::vector<double> g2 = {-0.004 * size, 0.002 * size, 0.001 * size};
            const double s = -2.5;

            const std::vector<double> u1 = solve_displacement_2d(mesh, D, F1, dofs, g1);
            const std::vector<double> u2 = solve_displacement_2d(mesh, D, F2, dofs, g2);
            std::vector<double> F12(N);
            std::vector<double> sF1(N);
            for (std::size_t i = 0; i < N; ++i) {
                F12[i] = F1[i] + F2[i];
                sF1[i] = s * F1[i];
            }
            const std::vector<double> g12 = {g1[0] + g2[0], g1[1] + g2[1], g1[2] + g2[2]};
            const std::vector<double> sg1 = {s * g1[0], s * g1[1], s * g1[2]};

            std::vector<double> sum(N);
            std::vector<double> scaled(N);
            for (std::size_t i = 0; i < N; ++i) {
                sum[i] = u1[i] + u2[i];
                scaled[i] = s * u1[i];
            }
            const double tolerance = 1e-9 * (largest_entry(u1) + largest_entry(u2));
            expect_vectors_near(solve_displacement_2d(mesh, D, F12, dofs, g12), sum, tolerance);
            expect_vectors_near(solve_displacement_2d(mesh, D, sF1, dofs, sg1), scaled, tolerance);
        }
    }
}

// EXAMPLE 176. The result does not depend on the units of Young's modulus. The same plates with E = 1 are solved
// with E = s, for s from 1e-12 (a material with almost no stiffness) to 1e15; K grows with s and the CG would see
// numbers of every order. Two checks, with nu = 0.3 and alpha = 12e-6, on a rectangle of 4 x 3 cells and one of 6 x 6
// (K has rounding asymmetries of 1e-16 times its numbers: above 1e-12 in absolute value with large E, and the CG,
// which checks the symmetry with an absolute tolerance of 1e-12, would throw std::invalid_argument):
//   - heated by 100 kelvin and free, the plate expands by c = (1 + nu) alpha dT = 1.56e-3, for any E (the thermal
//     forces grow with E and the displacements stay the same). The constraints also impose a translation of (0.3, -0.2):
//     with a very small E the fixed degrees of freedom, which in the system are worth 1, weigh more than all the
//     rest in the tolerance relative to norm(b), and without scaling the CG stops at once, leaving all the free
//     degrees of freedom at 0, and says it has converged;
//   - pulled with the same force (the stretching of EXAMPLE 169, stress sigma = 1 on the right side), the plate
//     lengthens as 1 / E: the displacements with E = s are those with E = 1 divided by s.
TEST(ElasticitySolve2d, TheSolutionDoesNotDependOnTheUnitsOfTheModulus) {
    const double nu = 0.3;
    const double alpha = 12.0e-6;
    const double dT = 100.0;
    for (const Grid& g : std::vector<Grid>{{3.0, 1.5, 4, 3}, {1.0, 1.0, 6, 6}}) {
        const Mesh2D mesh = rectangle_mesh(g.lx, g.ly, g.nx, g.ny);
        const Problem unit_pull = stretched_rectangle(mesh, g.lx, 1.0, nu, 1.0);
        for (double s : {1.0e-12, 1.0e-9, 1.0e-6, 1.0, 1.0e6, 210.0e9, 1.0e15}) {
            SCOPED_TRACE("grid " + std::to_string(g.nx) + " x " + std::to_string(g.ny) + ", E = " + std::to_string(s));
            const Matrix3 D = plane_strain_matrix(s, nu);

            const Problem heated = heated_free_body(mesh, Material{D, nu, alpha}, dT, 0, 0.3, -0.2);
            expect_vectors_near(solve_displacement_2d(mesh, D, heated.F, heated.dofs, heated.values), heated.exact,
                                1e-9 * largest_entry(heated.exact));

            std::vector<double> want = unit_pull.exact;
            for (double& w : want) {
                w /= s;
            }
            expect_vectors_near(solve_displacement_2d(mesh, D, unit_pull.F, unit_pull.dofs, unit_pull.values), want,
                                1e-9 * largest_entry(want));
        }
    }
}

// EXAMPLE 177. The nodes that are in no triangle (the mesh has nodes 0 = (5, 5) and 5 = (7, 7) outside the two
// triangles of the square 1, 2, 3, 4) have no stiffness: their rows of K are all 0 and there is no "right"
// displacement. They must be fixed, both degrees of freedom: then they come out with the imposed value, exactly, and the
// rest of the plate is solved as if they were not there (here the free heated square, which expands by
// c = (1 + nu) alpha dT in x and in y). If a degree of freedom of a node outside every triangle is NOT fixed (even
// just one of its two) the function throws std::invalid_argument, and does not return an invented number.
TEST(ElasticitySolve2d, NodesOutsideEveryTriangleMustBeFixed) {
    const Mesh2D mesh = mesh_with_nodes_outside_every_triangle();
    const Material m = {steel(), 0.3, 12.0e-6};
    const double dT = 100.0;
    const double c = (1.0 + m.nu) * m.alpha * dT;
    // the temperatures of the nodes outside the triangles (1e9 and -1e9) are not read
    const double hot = reference_temperature + dT;
    const std::vector<double> F = assemble_thermal_load_2d(mesh, m.D, m.nu, m.alpha,
                                                           {1.0e9, hot, hot, hot, hot, -1.0e9}, reference_temperature);
    ASSERT_EQ(F.size(), 12u);

    // the square: node 1 = (0, 0) with u and v (numbers 2 and 3) and v of node 2 = (1, 0) (number 5), all at 0
    const std::vector<std::size_t> square = {2, 3, 5};
    const std::vector<double> square_values = {0.0, 0.0, 0.0};
    const std::vector<std::size_t> isolated_dofs = {0, 1, 10, 11};

    // the four numbers of nodes 0 and 5 (0, 1, 10, 11) are fixed too: they come out as they are, and the square expands
    const std::vector<std::size_t> dofs = {0, 1, 2, 3, 5, 10, 11};
    const std::vector<double> values = {5.0, 6.0, 0.0, 0.0, 0.0, -7.0, 8.0};
    const std::vector<double> got = solve_displacement_2d(mesh, m.D, F, dofs, values);
    ASSERT_EQ(got.size(), 12u);
    EXPECT_EQ(got[0], 5.0);
    EXPECT_EQ(got[1], 6.0);
    EXPECT_EQ(got[10], -7.0);
    EXPECT_EQ(got[11], 8.0);
    // nodes 1, 2, 3, 4 are (0, 0), (1, 0), (1, 1), (0, 1) and the square expands by c in x and in y
    const std::vector<double> want_square = {0.0, 0.0, c, 0.0, c, c, 0.0, c};
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(got[2 + i], want_square[i], 1e-9 * c) << "degree of freedom " << 2 + i;
    }

    // it is enough for one degree of freedom of a node outside every triangle to stay free ("missing": the unfixed ones)
    const std::vector<std::vector<std::size_t>> missing_sets = {{0, 1, 10, 11}, {0, 1}, {10, 11}, {0}, {1},
                                                                {10},           {11},   {0, 1, 10}, {0, 10, 11}};
    for (const std::vector<std::size_t>& missing : missing_sets) {
        std::vector<std::size_t> some_dofs = square;
        std::vector<double> some_values = square_values;
        for (std::size_t d : isolated_dofs) {
            if (std::find(missing.begin(), missing.end(), d) == missing.end()) {
                some_dofs.push_back(d);
                some_values.push_back(1.0);
            }
        }
        EXPECT_THROW(solve_displacement_2d(mesh, m.D, F, some_dofs, some_values), std::invalid_argument)
            << missing.size() << " degrees of freedom of the nodes outside the triangles not fixed";
    }
}

// EXAMPLE 178. Wrong data: the function throws std::invalid_argument (and does not write outside a vector or
// return an invented result). With the mesh of 2 x 1 cells (6 nodes, 12 degrees of freedom: 0, ..., 11) the following are
// wrong:
//   - an F that does not have 12 numbers (11, 13, 6, 0);
//   - fewer than 3 fixed degrees of freedom (0, 1 or 2): K has 3 rigid motions and is singular without 3 constraints;
//   - a different number of imposed displacements and of fixed degrees of freedom (in both directions);
//   - a fixed degree of freedom that does not exist (12 or more, even a billion: the index must be checked BEFORE
//     using it to write into the starting point of the CG, otherwise the program crashes instead of throwing
//     the exception);
//   - a degree of freedom fixed twice (with the same value or with different values).
// These are fine instead: exactly 3 constraints (even in shuffled order), all 12, 4 constraints. The cases of a wrong index
// have at least 3 elements, so as not to confuse the index check with that of the minimum number.
TEST(ElasticitySolve2d, BadInputsThrowInvalidArgument) {
    const Mesh2D mesh = rectangle_mesh(2.0, 1.0, 2, 1);
    const Matrix3 D = hooke_quarter();
    const std::vector<double> F(12, 0.0);

    EXPECT_NO_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 3}, {0.0, 0.0, 0.0}));
    EXPECT_NO_THROW(solve_displacement_2d(mesh, D, F, {3, 0, 1}, {0.1, 0.0, 0.2}));
    EXPECT_NO_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 3, 11}, {0.0, 0.0, 0.0, 0.5}));
    EXPECT_NO_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                                          {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}));

    EXPECT_THROW(solve_displacement_2d(mesh, D, std::vector<double>(11, 0.0), {0, 1, 3}, {0.0, 0.0, 0.0}),
                 std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, std::vector<double>(13, 0.0), {0, 1, 3}, {0.0, 0.0, 0.0}),
                 std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, std::vector<double>(6, 0.0), {0, 1, 3}, {0.0, 0.0, 0.0}),
                 std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, std::vector<double>(), {0, 1, 3}, {0.0, 0.0, 0.0}),
                 std::invalid_argument);

    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {}, {}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0}, {0.0}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1}, {0.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {1, 0}, {0.0, 0.0}), std::invalid_argument);

    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 3}, {0.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 3}, {0.0, 0.0, 0.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 3}, {}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1}, {0.0, 0.0, 0.0}), std::invalid_argument);

    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 12}, {0.0, 0.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 3, 100}, {0.0, 0.0, 0.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 1000000000}, {0.0, 0.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {1000000000, 0, 1, 3}, {0.0, 0.0, 0.0, 0.0}),
                 std::invalid_argument);

    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 3, 1}, {0.0, 0.0, 0.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {0, 1, 3, 3}, {0.0, 0.0, 0.0, 0.5}), std::invalid_argument);
    EXPECT_THROW(solve_displacement_2d(mesh, D, F, {2, 2, 2}, {0.0, 0.0, 0.0}), std::invalid_argument);
}

// EXAMPLE 179. Systems with no solution: the function throws std::runtime_error, and does not return a wrong
// vector, full of NaN or of huge numbers.
//   - A NaN in the forces, or in an imposed displacement: comparisons with a NaN are all false and the CG stops
//     at once, not converged.
//   - Three constraints that leave the rotation free (u and v of node 0 and u of node 1 with the SAME y: the rotation
//     about node 0 moves node 1 only in y) and a load that sets it in motion (a vertical force on node 3,
//     which has a moment different from 0 about node 0): the body rotates without stopping, there is no state of
//     equilibrium. The system is singular and has no solution, and the CG does not really converge: on the unit square,
//     however, its residual (which it updates with a formula, never recomputing b - A u) reaches 0 in 9 iterations
//     while u has numbers of order 1e15, and it says it has converged; only the residual recomputed from u (about 0.36
//     times norm(b)) exposes it. With 4 x 3 cells and with 6 x 6 the CG instead exceeds the maximum number of iterations.
//   - Two separate triangles (two bodies), with the three constraints on only one and a force on the other: the
//     other is free and the force sets it in motion.
// With a load IN EQUILIBRIUM (the thermal forces) the same constraints leave a rotation free but the system has
// infinitely many solutions, and one is found: no exception. In the same way the constraints on both bodies solve
// the problem of the two triangles.
TEST(ElasticitySolve2d, ThrowsRuntimeErrorWhenTheSystemCannotBeSolved) {
    const double not_a_number = std::numeric_limits<double>::quiet_NaN();
    const Matrix3 D = hooke_quarter();

    const Mesh2D square = unit_square();
    const std::vector<double> heating = {-2.0, -2.0, 2.0, -2.0, -2.0, 2.0, 2.0, 2.0};
    std::vector<double> with_nan = heating;
    with_nan[5] = not_a_number;
    EXPECT_NO_THROW(solve_displacement_2d(square, D, heating, {0, 1, 3}, {0.0, 0.0, 0.0}));
    EXPECT_THROW(solve_displacement_2d(square, D, with_nan, {0, 1, 3}, {0.0, 0.0, 0.0}), std::runtime_error);
    EXPECT_THROW(solve_displacement_2d(square, D, heating, {0, 1, 3}, {0.0, not_a_number, 0.0}), std::runtime_error);

    // the rotation about node 0 is free: u0 = v0 = u1 = 0 (nodes 0 and 1 with y = 0)
    std::vector<double> push_up(8, 0.0);
    push_up[7] = 1.0;  // vertical force on node 3 = (1, 1)
    std::vector<double> push_right(8, 0.0);
    push_right[6] = 1.0;  // horizontal force on node 3
    EXPECT_THROW(solve_displacement_2d(square, D, push_up, {0, 1, 2}, {0.0, 0.0, 0.0}), std::runtime_error);
    EXPECT_THROW(solve_displacement_2d(square, D, push_right, {0, 1, 2}, {0.0, 0.0, 0.0}), std::runtime_error);
    EXPECT_NO_THROW(solve_displacement_2d(square, D, heating, {0, 1, 2}, {0.0, 0.0, 0.0}));
    EXPECT_NO_THROW(solve_displacement_2d(square, D, push_up, {0, 1, 3}, {0.0, 0.0, 0.0}));

    for (const Grid& g : std::vector<Grid>{{3.0, 1.5, 4, 3}, {1.0, 1.0, 6, 6}}) {
        const Mesh2D mesh = rectangle_mesh(g.lx, g.ly, g.nx, g.ny);
        std::vector<double> load(2 * mesh.num_nodes(), 0.0);
        load[2 * (mesh.num_nodes() - 1) + 1] = -1000.0;  // downward force on the last node
        const std::size_t bottom_right = g.nx;  // the node (lx, 0): it has the same y as node 0
        EXPECT_THROW(solve_displacement_2d(mesh, steel(), load, {0, 1, 2 * bottom_right}, {0.0, 0.0, 0.0}),
                     std::runtime_error)
            << "rectangle " << g.nx << " x " << g.ny;
        EXPECT_NO_THROW(solve_displacement_2d(mesh, steel(), load, {0, 1, 2 * bottom_right + 1}, {0.0, 0.0, 0.0}))
            << "rectangle " << g.nx << " x " << g.ny;
    }

    const Mesh2D two_pieces = two_separate_triangles();
    std::vector<double> pull(12, 0.0);
    pull[6] = 1.0;  // force in x on node 3, of the second triangle
    EXPECT_THROW(solve_displacement_2d(two_pieces, D, pull, {0, 1, 3}, {0.0, 0.0, 0.0}), std::runtime_error);
    EXPECT_NO_THROW(solve_displacement_2d(two_pieces, D, pull, {0, 1, 3, 6, 7, 10}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
}

// EXAMPLE 180. A large plate: a steel square of 1 m side with 40 x 40 cells (1681 nodes, 3362 degrees of
// freedom), heated by 100 kelvin and free (three isostatic constraints). The exact solution is the expansion
// u = c x, v = c y with c = (1 + nu) alpha dT = 1.56e-3: the CG takes about 700 iterations (never more than 100
// on the meshes of the other tests) with the numbers of K of order 1e11, enough to see whether the scaling, the
// tolerance and the maximum number of iterations are right on a problem of real size. The displacement of each node must
// match the formula within 1e-9 times the expansion c (true error about 3e-13); the function takes about half a second.
TEST(ElasticitySolve2d, ALargeSteelPlateIsSolvedAccurately) {
    const Mesh2D mesh = rectangle_mesh(1.0, 1.0, 40, 40);
    ASSERT_EQ(mesh.num_nodes(), 1681u);
    const Material m = {steel(), 0.3, 12.0e-6};

    const Problem pr = heated_free_body(mesh, m, 100.0, 0, 0.0, 0.0);
    const std::vector<double> u = solve_displacement_2d(mesh, m.D, pr.F, pr.dofs, pr.values);
    ASSERT_EQ(u.size(), 3362u);
    double worst = 0.0;
    for (std::size_t i = 0; i < u.size(); ++i) {
        worst = std::max(worst, std::fabs(u[i] - pr.exact[i]));
    }
    const double c = (1.0 + m.nu) * m.alpha * 100.0;
    EXPECT_LT(worst, 1e-9 * c);
}

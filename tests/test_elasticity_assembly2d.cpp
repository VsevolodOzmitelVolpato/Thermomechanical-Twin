#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "Assembly2D.h"
#include "CSR_and_Vectors.h"
#include "Elasticity2D.h"
#include "Mesh2D.h"
#include "Triplets.h"

// ---- From the triangle matrices to the matrix of the whole mesh, with two displacements per node -------------------
//
// element_elasticity_stiffness(mesh, e, D) gives the 6 x 6 matrix of ONE triangle, with the LOCAL degrees of freedom
// (u0, v0, u1, v1, u2, v2). The matrix of the whole mesh, K, is 2 n x 2 n (n = number of nodes) and its indices are the
// GLOBAL degrees of freedom: node I has its x displacement at number 2 I and its y one at number 2 I + 1.
// K[P][Q] is the force on degree of freedom P when only degree of freedom Q is moved by 1. As for heat, it is
// the sum of the contributions of all the triangles that contain both P and Q:
//
//     K[P][Q] = sum, over the triangles e that contain P and Q, of Ke[i][j]
//               where i and j are the local degrees of freedom of P and Q, i.e. with t = mesh.triangle(e):
//               P = 2 t[i / 2] + i % 2   and   Q = 2 t[j / 2] + j % 2
//
// (the local degree of freedom i is component i % 2 of local node i / 2). Each triangle produces 36 triplets
// (row, column, value) = (2 t[i / 2] + i % 2, 2 t[j / 2] + j % 2, Ke[i][j]) with i and j from 0 to 5; the triplets of all the
// triangles go to csr_from_triplets, which sums those with the same position. Two nodes that are never in the
// same triangle do not influence each other: the 2 x 2 block between them is 0 and takes no memory.
//
// elasticity_stiffness_triplets_2d(mesh, D) returns the triplets and assemble_elasticity_stiffness_2d(mesh, D) the CSR
// (2 n x 2 n). There are no errors to report: the mesh has already been checked by the Mesh2D constructor and D is not
// checked (as for element_elasticity_stiffness).
//
// What a real stiffness matrix must have, and this is what the tests check:
//   - it is symmetric;
//   - a rigid motion (a translation, a rotation) deforms nothing, so it produces no force: K u = 0;
//   - for a linear displacement (constant strain eps) the energy is  u^T K u = eps^T D eps * (total area),
//     and the forces of a constant stress cancel at every interior node (they remain only on the boundary);
//   - it is the sum of the contributions of the triangles, also for meshes with node numbers out of order and with nodes that
//     belong to no triangle (two empty rows for each);
//   - it occupies 4 slots for each slot of the heat matrix (a 2 x 2 block for each pair of nodes in the same
//     triangle) and stays sparse even for large meshes.

namespace {

using Dense = std::vector<std::vector<double>>;

// A rectangle lx x ly divided into nx x ny cells, as in rectangle_mesh.
struct Grid {
    double lx, ly;
    std::size_t nx, ny;
};

// Six rectangles of very different shapes (the same as in the heat assembly tests).
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

// Hand-made materials (3 x 3, indices xx, yy, xy), symmetric: the stiffness matrix does not need them to be physical.
Matrix3 hooke_quarter() {  // E = 1, nu = 1/4: lambda = mu = 0.4
    return {{{1.2, 0.4, 0.0}, {0.4, 1.2, 0.0}, {0.0, 0.0, 0.4}}};
}
Matrix3 general() {  // all the numbers different, also the coupling with shear
    return {{{2.0, 0.5, 0.1}, {0.5, 3.0, -0.2}, {0.1, -0.2, 1.0}}};
}
Matrix3 steel() {  // a real, stiff material: numbers of order 1e11
    return plane_strain_matrix(210e9, 0.3);
}

std::vector<Matrix3> sample_materials() { return {hooke_quarter(), general(), steel()}; }

// The largest absolute value on the diagonal of A: the "scale" of the matrix, to set the tolerances.
double largest_diagonal(const CSR& A) {
    double m = 0.0;
    for (double d : A.diag()) {
        m = std::max(m, std::fabs(d));
    }
    return m;
}

// The largest absolute value of a vector.
double largest_entry(const std::vector<double>& v) {
    double m = 0.0;
    for (double a : v) {
        m = std::max(m, std::fabs(a));
    }
    return m;
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
    return {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},      // translation in x
            {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},      // translation in y
            {0.0, 0.0, -1.0, 0.0, 1.0, 0.0},     // rotation
            {0.3, 0.0, -0.7, -1.2, 0.7, 0.0}};   // translation and rotation together
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

// The total area: the sum of the areas of all the triangles.
double total_area(const Mesh2D& mesh) {
    double a = 0.0;
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        a += mesh.area(e);
    }
    return a;
}

// eps^T D eps, with eps = (a1, b2, a2 + b1) the strain of the field f.
double strain_energy_density_times_two(const Matrix3& D, const LinearField& f) {
    const double eps[3] = {f.a1, f.b2, f.a2 + f.b1};
    double s = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            s += eps[i] * D[i][j] * eps[j];
        }
    }
    return s;
}

// The "slow computation": K in a full 2 n x 2 n table, in which each triangle adds by hand its 36 numbers into the cells
// (2 t[a] + c, 2 t[b] + d), with a and b the local nodes and c and d the components (the local degree of freedom is 2 a + c).
// It is only used for comparison, and only for small meshes.
Dense slow_stiffness(const Mesh2D& mesh, const Matrix3& D) {
    const std::size_t n = 2 * mesh.num_nodes();
    Dense K(n, std::vector<double>(n, 0.0));
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Triangle& t = mesh.triangle(e);
        const Matrix6 Ke = element_elasticity_stiffness(mesh, e, D);
        for (std::size_t a = 0; a < 3; ++a) {
            for (std::size_t c = 0; c < 2; ++c) {
                for (std::size_t b = 0; b < 3; ++b) {
                    for (std::size_t d = 0; d < 2; ++d) {
                        K[2 * t[a] + c][2 * t[b] + d] += Ke[2 * a + c][2 * b + d];
                    }
                }
            }
        }
    }
    return K;
}

// Compare A with a full table, cell by cell (with a tolerance relative to the largest cell).
void expect_same_as_dense(const CSR& A, const Dense& table) {
    ASSERT_EQ(A.size(), table.size());
    double scale = 0.0;
    for (const auto& row : table) {
        for (double v : row) {
            scale = std::max(scale, std::fabs(v));
        }
    }
    for (std::size_t i = 0; i < table.size(); ++i) {
        for (std::size_t j = 0; j < table.size(); ++j) {
            EXPECT_NEAR(A.get(i, j), table[i][j], 1e-13 * scale) << "element (" << i << ", " << j << ")";
        }
    }
}

// Check that A is the matrix `want`, element by element, and that it occupies exactly `places` slots.
void expect_matrix(const CSR& A, const Dense& want, std::size_t places, double tolerance) {
    ASSERT_EQ(A.size(), want.size()) << "size";
    EXPECT_EQ(A.get_val_size(), places) << "occupied slots";
    for (std::size_t i = 0; i < want.size(); ++i) {
        for (std::size_t j = 0; j < want.size(); ++j) {
            EXPECT_NEAR(A.get(i, j), want[i][j], tolerance) << "element (" << i << ", " << j << ")";
        }
    }
}

// The values of the triplets that are at position (row, col), in increasing order.
std::vector<double> values_at(const std::vector<Triplet>& triplets, std::size_t row, std::size_t col) {
    std::vector<double> out;
    for (const Triplet& t : triplets) {
        if (t.row == row && t.col == col) {
            out.push_back(t.val);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Compare the values of the triplets at position (row, col) with `want` (in increasing order).
void expect_values_at(const std::vector<Triplet>& triplets, std::size_t row, std::size_t col,
                      const std::vector<double>& want) {
    const std::vector<double> got = values_at(triplets, row, col);
    ASSERT_EQ(got.size(), want.size()) << "position (" << row << ", " << col << ")";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], 1e-14) << "position (" << row << ", " << col << "), value number " << i;
    }
}

// How many DIFFERENT positions (row, column) appear in the list, in an n x n matrix.
std::size_t distinct_places(const std::vector<Triplet>& triplets, std::size_t n) {
    std::vector<std::size_t> places;
    for (const Triplet& t : triplets) {
        places.push_back(t.row * n + t.col);
    }
    std::sort(places.begin(), places.end());
    places.erase(std::unique(places.begin(), places.end()), places.end());
    return places.size();
}

}  // namespace

// EXAMPLE 149. By hand. The unit square has 4 nodes, 0 = (0, 0), 1 = (1, 0), 2 = (0, 1), 3 = (1, 1), and two
// triangles, T0 = (0, 1, 2) and T1 = (1, 3, 2), both of area 1/2. With E = 1 and nu = 1/4 (lambda = mu = 0.4) the
// material is D = [[1.2, 0.4, 0], [0.4, 1.2, 0], [0, 0, 0.4]] and the global degrees of freedom are 8: node I has u at
// number 2 I and v at 2 I + 1. The triangle matrices (A B^T D B, the Ke step), in units of 0.1:
//     Ke0 = [[ 8,  4, -6, -2, -2, -2],        Ke1 = [[ 2,  0, -2, -2,  0,  2],
//            [ 4,  8, -2, -2, -2, -6],               [ 0,  6, -2, -6,  2,  0],
//            [-6, -2,  6,  0,  0,  2],               [-2, -2,  8,  4, -6, -2],
//            [-2, -2,  0,  2,  2,  0],               [-2, -6,  4,  8, -2, -2],
//            [-2, -2,  0,  2,  2,  0],               [ 0,  2, -6, -2,  6,  0],
//            [-2, -6,  2,  0,  0,  6]]               [ 2,  0, -2, -2,  0,  2]]
// T0 has nodes (0, 1, 2), so its 6 local degrees of freedom are the global ones (0, 1, 2, 3, 4, 5). T1 has nodes
// (1, 3, 2): its local degrees of freedom (0, 1, 2, 3, 4, 5) are the global ones (2, 3, 6, 7, 4, 5). Each number
// of Ke0 and of Ke1 is added into the cell (global of the row, global of the column). For example
//     K[2][2] = Ke0[2][2] + Ke1[0][0] = 0.6 + 0.2 = 0.8      (node 1 is in both triangles)
//     K[2][5] = Ke0[2][5] + Ke1[0][5] = 0.2 + 0.2 = 0.4
//     K[6][4] = Ke1[2][4] = -0.6                              (node 3 is only in T1)
//     K[0][6] = 0                                             (nodes 0 and 3 are never in the same triangle)
// and the whole matrix is (in units of 0.1) the one written below. It occupies 56 slots: 14 blocks 2 x 2 (the 4 nodes with
// themselves and the 5 sides, in both directions), and 8 of the slots are 0 (like K[2][3]: the contribution exists, and is 0).
TEST(ElasticityAssembly2d, StiffnessOfTwoRightTrianglesMatchesHandComputation) {
    const Dense want = {{0.8, 0.4, -0.6, -0.2, -0.2, -0.2, 0.0, 0.0},
                        {0.4, 0.8, -0.2, -0.2, -0.2, -0.6, 0.0, 0.0},
                        {-0.6, -0.2, 0.8, 0.0, 0.0, 0.4, -0.2, -0.2},
                        {-0.2, -0.2, 0.0, 0.8, 0.4, 0.0, -0.2, -0.6},
                        {-0.2, -0.2, 0.0, 0.4, 0.8, 0.0, -0.6, -0.2},
                        {-0.2, -0.6, 0.4, 0.0, 0.0, 0.8, -0.2, -0.2},
                        {0.0, 0.0, -0.2, -0.2, -0.6, -0.2, 0.8, 0.4},
                        {0.0, 0.0, -0.2, -0.6, -0.2, -0.2, 0.4, 0.8}};
    expect_matrix(assemble_elasticity_stiffness_2d(unit_square(), hooke_quarter()), want, 56, 1e-14);
}

// EXAMPLE 150. Symmetry and rigid motions. K is symmetric (the reaction forces are equal and opposite: the force on P
// for a displacement of Q is the one on Q for a displacement of P) and, in meshes in which every node is in a
// triangle, its diagonal is positive (moving a single degree of freedom deforms some triangle). A rigid
// motion, a translation or a rotation, deforms no triangle, so the forces are all 0:
//     K u = 0   per   u = (1, 0, 1, 0, ...),   (0, 1, 0, 1, ...),   (-y_I, x_I, ...)   and their combinations
// (the numbers of u are in the order of the global degrees of freedom: u of node I at 2 I, v at 2 I + 1, so a
// translation in x has 1 at the even places and 0 at the odd ones, and a wrong numbering of the degrees of freedom shows).
// Tried on nine meshes and on three materials.
TEST(ElasticityAssembly2d, StiffnessIsSymmetricAndRigidMotionsProduceNoForce) {
    const std::vector<Mesh2D> meshes = sample_meshes();
    const std::vector<Matrix3> materials = sample_materials();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        for (std::size_t m = 0; m < materials.size(); ++m) {
            SCOPED_TRACE("mesh number " + std::to_string(a) + ", material number " + std::to_string(m));
            const Mesh2D& mesh = meshes[a];
            const CSR K = assemble_elasticity_stiffness_2d(mesh, materials[m]);
            ASSERT_EQ(K.size(), 2 * mesh.num_nodes());
            const double scale = largest_diagonal(K);
            ASSERT_GT(scale, 0.0);

            EXPECT_TRUE(K.is_symmetric(1e-12 * scale));
            const std::vector<double> diagonal = K.diag();
            for (std::size_t i = 0; i < diagonal.size(); ++i) {
                EXPECT_GT(diagonal[i], 0.0) << "diagonal " << i;
            }

            const std::vector<LinearField> motions = rigid_motions();
            for (std::size_t r = 0; r < motions.size(); ++r) {
                const std::vector<double> u = nodal_displacements(mesh, motions[r]);
                const std::vector<double> force = K * u;
                ASSERT_EQ(force.size(), u.size());
                const double tolerance = 1e-12 * scale * largest_entry(u);
                for (std::size_t i = 0; i < force.size(); ++i) {
                    ASSERT_NEAR(force[i], 0.0, tolerance) << "rigid motion number " << r << ", force " << i;
                }
            }
        }
    }
}

// EXAMPLE 151. Energy and interior nodes. If the displacement is linear, u = a0 + a1 x + a2 y, v = b0 + b1 x + b2 y, the
// strain is the same in the whole mesh, eps = (a1, b2, a2 + b1), and the strain energy is (1/2) eps^T D eps
// times the total area, so
//     u^T K u = eps^T D eps * (total area).
// For u = x (eps = (1, 0, 0)) it is D[0][0] times the area, for u = y (eps = (0, 0, 1)) it is D[2][2] times the area, and so on.
// In addition the stress is constant, D eps, and the forces of a constant stress on an interior node (surrounded by
// triangles) cancel: (K u)[2 I] = (K u)[2 I + 1] = 0 for every node I that is not on the boundary. The forces remain
// only on the boundary, where the stress "pushes" the contour.
// Tried with the two translations, the four pure fields and a general one, on nine meshes and three materials.
TEST(ElasticityAssembly2d, StiffnessGivesTheEnergyOfLinearFieldsAndBalancesTheInteriorNodes) {
    const std::vector<Mesh2D> meshes = sample_meshes();
    const std::vector<Matrix3> materials = sample_materials();
    const std::vector<LinearField> fields = strain_fields();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        const Mesh2D& mesh = meshes[a];
        std::vector<bool> on_boundary(mesh.num_nodes(), false);
        for (std::size_t I : boundary_nodes(mesh)) {
            on_boundary[I] = true;
        }
        for (std::size_t m = 0; m < materials.size(); ++m) {
            const CSR K = assemble_elasticity_stiffness_2d(mesh, materials[m]);
            const double scale = largest_diagonal(K);
            for (std::size_t f = 0; f < fields.size(); ++f) {
                SCOPED_TRACE("mesh number " + std::to_string(a) + ", material number " + std::to_string(m) +
                             ", field number " + std::to_string(f));
                const std::vector<double> u = nodal_displacements(mesh, fields[f]);
                const std::vector<double> force = K * u;
                const double umax = largest_entry(u);

                double energy = 0.0;
                for (std::size_t i = 0; i < u.size(); ++i) {
                    energy += u[i] * force[i];
                }
                const double want = strain_energy_density_times_two(materials[m], fields[f]) * total_area(mesh);
                EXPECT_NEAR(energy, want, 1e-12 * scale * umax * umax);

                for (std::size_t I = 0; I < mesh.num_nodes(); ++I) {
                    if (!on_boundary[I]) {
                        EXPECT_NEAR(force[2 * I], 0.0, 1e-12 * scale * umax) << "interior node " << I << ", x";
                        EXPECT_NEAR(force[2 * I + 1], 0.0, 1e-12 * scale * umax) << "interior node " << I << ", y";
                    }
                }
            }
        }
    }
}

// EXAMPLE 152. The "slow computation": the full 2 n x 2 n table in which each triangle adds by hand its 36 numbers into the
// cells (2 t[a] + c, 2 t[b] + d), with a and b the local nodes and c and d the components (the local degree of freedom is 2 a +
// c: it is the same computation as the global degrees of freedom above, written with four loops instead of two). The
// assembled K and the triplets summed with csr_from_triplets must give the same cells. The triplets are 36 per triangle and
// all their indices are less than 2 n (otherwise csr_from_triplets throws std::invalid_argument).
// The meshes: six rectangles, the triangle with a node inside, the irregular quadrilateral, two triangles of different area
// and the mesh with two nodes outside every triangle; the materials are three.
TEST(ElasticityAssembly2d, StiffnessMatchesTheSlowDenseAssemblyOnEveryKindOfMesh) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    const std::vector<Matrix3> materials = sample_materials();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        for (std::size_t m = 0; m < materials.size(); ++m) {
            SCOPED_TRACE("mesh number " + std::to_string(a) + ", material number " + std::to_string(m));
            const Mesh2D& mesh = meshes[a];
            const Dense table = slow_stiffness(mesh, materials[m]);

            expect_same_as_dense(assemble_elasticity_stiffness_2d(mesh, materials[m]), table);

            const std::vector<Triplet> triplets = elasticity_stiffness_triplets_2d(mesh, materials[m]);
            ASSERT_EQ(triplets.size(), 36 * mesh.num_triangles());
            expect_same_as_dense(csr_from_triplets(2 * mesh.num_nodes(), triplets), table);
        }
    }
}

// EXAMPLE 153. The occupied slots. Two nodes that are in the same triangle (also a node with itself) have a
// 2 x 2 block of slots in K: four slots, even if some are 0. In the heat matrix (assemble_stiffness_2d)
// the same pair has a single slot, so K occupies exactly 4 times the slots of the heat matrix, on every
// mesh. For a rectangle of nx x ny cells the nodes are (nx + 1) (ny + 1), the sides are
// nx (ny + 1) + (nx + 1) ny + nx ny (horizontal, vertical, diagonal) and the slots of the heat matrix are
// nodes + 2 sides, so K occupies
//     4 (nodes + 2 sides).
// For the unit square of EXAMPLE 149 it is 4 (4 + 2 * 5) = 56.
TEST(ElasticityAssembly2d, StiffnessOccupiesFourPlacesForEveryPlaceOfTheHeatMatrix) {
    const std::vector<Mesh2D> meshes = sample_meshes_with_isolated_nodes();
    for (std::size_t a = 0; a < meshes.size(); ++a) {
        SCOPED_TRACE("mesh number " + std::to_string(a));
        const CSR K = assemble_elasticity_stiffness_2d(meshes[a], hooke_quarter());
        const CSR heat = assemble_stiffness_2d(meshes[a], 1.0);
        EXPECT_EQ(K.get_val_size(), 4 * heat.get_val_size());
    }

    for (const Grid& g : sample_grids()) {
        SCOPED_TRACE("rectangle " + std::to_string(g.lx) + " x " + std::to_string(g.ly) + " with " +
                     std::to_string(g.nx) + " x " + std::to_string(g.ny) + " cells");
        const std::size_t nodes = (g.nx + 1) * (g.ny + 1);
        const std::size_t edges = g.nx * (g.ny + 1) + (g.nx + 1) * g.ny + g.nx * g.ny;
        const CSR K = assemble_elasticity_stiffness_2d(rectangle_mesh(g.lx, g.ly, g.nx, g.ny), steel());
        EXPECT_EQ(K.size(), 2 * nodes);
        EXPECT_EQ(K.get_val_size(), 4 * (nodes + 2 * edges));
    }
}

// EXAMPLE 154. Nodes outside every triangle. The mesh has 6 nodes, but only (1, 2, 3) and (1, 3, 4) are in a triangle:
// nodes 0 and 5 have nothing to assemble. K still has 12 rows, two per node, and the rows and columns 0, 1 (node
// 0) and 10, 11 (node 5) are empty: they are 0 and take no memory. The slots are those of the 4 connected nodes: 4 nodes and
// 5 sides (the square divided along a diagonal), i.e. 4 (4 + 2 * 5) = 56.
TEST(ElasticityAssembly2d, NodesOutsideEveryTriangleGetEmptyRowsAndColumns) {
    const Mesh2D mesh = mesh_with_nodes_outside_every_triangle();
    const CSR K = assemble_elasticity_stiffness_2d(mesh, hooke_quarter());
    ASSERT_EQ(K.size(), 12u);
    EXPECT_EQ(K.get_val_size(), 56u);

    const std::size_t empty[4] = {0, 1, 10, 11};
    for (std::size_t p : empty) {
        for (std::size_t q = 0; q < 12; ++q) {
            EXPECT_EQ(K.get(p, q), 0.0) << "row " << p << ", column " << q;
            EXPECT_EQ(K.get(q, p), 0.0) << "row " << q << ", column " << p;
        }
    }
    for (std::size_t p = 2; p < 10; ++p) {
        EXPECT_GT(K.get(p, p), 0.0) << "diagonal " << p;
    }
}

// EXAMPLE 155. The triplets before summing them (the same as in EXAMPLE 149). The two triangles give 36 each, 72 in
// total, but the distinct positions are only 56 (the 56 slots of K): the other 16 triplets fall on slots already taken
// by the other triangle. With the matrices Ke0 and Ke1 of EXAMPLE 149 (in units of 0.1 they are the ones written there):
//     (2, 2): T0 gives 0.6 and T1 gives 0.2                0.6 + 0.2 = 0.8 = K[2][2]
//     (2, 5): T0 gives 0.2 and T1 gives 0.2                0.4 = K[2][5], and the same at (5, 2)
//     (2, 3): T0 and T1 both give 0            the slot exists all the same and the triplets must be kept
//     (6, 4): only T1, which gives -0.6                  (node 3 is only in T1)
//     (0, 0): only T0, which gives 0.8
//     (0, 6): none                                (nodes 0 and 3 are never in the same triangle)
// Summed with csr_from_triplets, the 72 triplets give the K of EXAMPLE 149.
TEST(ElasticityAssembly2d, StiffnessTripletsAreTheContributionsOfEveryTriangle) {
    const Mesh2D mesh = unit_square();
    const std::vector<Triplet> triplets = elasticity_stiffness_triplets_2d(mesh, hooke_quarter());

    ASSERT_EQ(triplets.size(), 72u);
    EXPECT_EQ(distinct_places(triplets, 8), 56u);
    expect_values_at(triplets, 2, 2, {0.2, 0.6});
    expect_values_at(triplets, 2, 5, {0.2, 0.2});
    expect_values_at(triplets, 5, 2, {0.2, 0.2});
    expect_values_at(triplets, 2, 3, {0.0, 0.0});
    expect_values_at(triplets, 6, 4, {-0.6});
    expect_values_at(triplets, 4, 6, {-0.6});
    expect_values_at(triplets, 0, 0, {0.8});
    expect_values_at(triplets, 0, 6, {});

    const CSR K = csr_from_triplets(mesh.num_nodes() * 2, triplets);
    const CSR want = assemble_elasticity_stiffness_2d(mesh, hooke_quarter());
    ASSERT_EQ(K.size(), want.size());
    EXPECT_EQ(K.get_val_size(), want.get_val_size());
    for (std::size_t i = 0; i < K.size(); ++i) {
        for (std::size_t j = 0; j < K.size(); ++j) {
            EXPECT_DOUBLE_EQ(K.get(i, j), want.get(i, j)) << "element (" << i << ", " << j << ")";
        }
    }
}

// EXAMPLE 156. A large mesh stays sparse. The unit square with 100 x 100 cells has 10201 nodes and 20402 degrees of
// freedom: the full matrix would have 20402 x 20402 numbers (over 3 GB), but K occupies 4 (nodes + 2 sides) slots, i.e.
// 282404 instead of 416 million. As for heat, a translation gives no forces and the energy of u = x is
// u^T K u = D[0][0] * (area) = 1.2 (with the material of EXAMPLE 149).
TEST(ElasticityAssembly2d, AssemblyOfALargeMeshStaysSparse) {
    const std::size_t n = 100;
    const Mesh2D mesh = rectangle_mesh(1.0, 1.0, n, n);
    const CSR K = assemble_elasticity_stiffness_2d(mesh, hooke_quarter());

    const std::size_t nodes = (n + 1) * (n + 1);
    const std::size_t edges = n * (n + 1) + (n + 1) * n + n * n;
    ASSERT_EQ(K.size(), 2 * nodes);
    EXPECT_EQ(K.get_val_size(), 4 * (nodes + 2 * edges));

    const std::vector<double> shift = nodal_displacements(mesh, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    const std::vector<double> force = K * shift;
    for (std::size_t i = 0; i < force.size(); ++i) {
        ASSERT_NEAR(force[i], 0.0, 1e-12) << "row " << i;
    }

    const std::vector<double> stretch = nodal_displacements(mesh, {0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
    const std::vector<double> Ku = K * stretch;
    double energy = 0.0;
    for (std::size_t i = 0; i < stretch.size(); ++i) {
        energy += stretch[i] * Ku[i];
    }
    EXPECT_NEAR(energy, 1.2, 1e-9);
}

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "Assembly2D.h"
#include "CSR_and_Vectors.h"
#include "Fem2D.h"
#include "Mesh2D.h"

// ---- From the triangle matrices to the matrix of the whole mesh: the assembly --------------------
//
// element_stiffness(mesh, e, k) gives the 3 x 3 matrix of ONE triangle, with the LOCAL indices 0, 1, 2.
// The matrix of the whole mesh, K, is n x n, with n = number of nodes, and its indices are the NODES of the
// mesh: K[I][J] is the coupling between node I and node J. It is obtained by adding up the contributions of all the
// triangles that have both I and J among their nodes:
//
//     K[I][J] = sum, over the triangles e with I and J among their nodes, of Ke[i][j]
//               where i and j are the positions of I and J in the three nodes t = mesh.triangle(e)   (t[i] = I, t[j] = J)
//
// So each triangle produces 9 triplets (row, column, value) = (t[i], t[j], Ke[i][j]): row and column
// are the nodes of the mesh, not the local indices. The triplets of all the triangles go to csr_from_triplets,
// which adds up those with the same position. Two nodes that are never in the same triangle do not
// influence each other: K[I][J] = 0 and it takes no memory.
//
// assemble_stiffness_2d(mesh, k) returns K in CSR format (n x n, n = mesh.num_nodes()). There are no
// errors to report: the mesh has already been checked by the constructor of Mesh2D.

namespace {

using Dense = std::vector<std::vector<double>>;

// A rectangle lx x ly divided into nx x ny cells, as in rectangle_mesh.
struct Grid {
    double lx, ly;
    std::size_t nx, ny;
};

// Six rectangles of very different shapes.
std::vector<Grid> sample_grids() {
    return {{1.0, 1.0, 1, 1},     // the unit square: two triangles
            {2.0, 1.0, 2, 1},     // the two cells of EXAMPLE 37
            {3.0, 1.5, 4, 3},     // 0.75 x 0.5 cells
            {1.0, 1.0, 6, 6},     // 72 triangles
            {0.2, 0.05, 5, 2},    // small (20 cm by 5 cm, in meters)
            {10.0, 0.1, 1, 7}};   // very flat: 10 x 0.0143 cells
}

std::size_t nodes_of(const Grid& g) { return (g.nx + 1) * (g.ny + 1); }

std::string describe(const Grid& g) {
    return "rectangle " + std::to_string(g.lx) + " x " + std::to_string(g.ly) + " with " + std::to_string(g.nx) +
           " x " + std::to_string(g.ny) + " cells";
}

// The large triangle of EXAMPLE 8 of the meshes, with node 3 inside (area 9/2). It is not a rectangle.
Mesh2D triangle_with_interior_node() {
    return Mesh2D({0.0, 3.0, 0.0, 1.0}, {0.0, 0.0, 3.0, 1.0}, {{0, 1, 3}, {1, 2, 3}, {2, 0, 3}});
}

// An irregular quadrilateral with a node inside: 0 = (0, 0), 1 = (4, 0), 2 = (3, 3), 3 = (-1, 2) and, inside,
// 4 = (1, 0.7). Four triangles around node 4, all of different shape and area: 1.4, 4.15, 3.6 and 1.35 (10.5
// in total).
Mesh2D irregular_quadrilateral() {
    return Mesh2D({0.0, 4.0, 3.0, -1.0, 1.0}, {0.0, 0.0, 3.0, 2.0, 0.7},
                  {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}});
}

// Checks that A is the matrix `want`, element by element, and that it occupies exactly nnz slots.
void expect_matrix(const CSR& A, const Dense& want, std::size_t nnz, const std::string& what = "") {
    ASSERT_EQ(A.size(), want.size()) << what << " size";
    EXPECT_EQ(A.get_val_size(), nnz) << what << " occupied slots";
    for (std::size_t i = 0; i < want.size(); ++i) {
        for (std::size_t j = 0; j < want.size(); ++j) {
            EXPECT_DOUBLE_EQ(A.get(i, j), want[i][j]) << what << " element (" << i << ", " << j << ")";
        }
    }
}

// The largest absolute value on the diagonal of A: the "scale" of the matrix, used to set the tolerances.
double largest_diagonal(const CSR& A) {
    double m = 0.0;
    for (double d : A.diag()) {
        m = std::max(m, std::fabs(d));
    }
    return m;
}

// The linear temperature field T(x, y) = t0 + gx x + gy y, evaluated at the nodes of the mesh.
std::vector<double> linear_field(const Mesh2D& mesh, double t0, double gx, double gy) {
    std::vector<double> T;
    for (std::size_t i = 0; i < mesh.num_nodes(); ++i) {
        T.push_back(t0 + gx * mesh.x(i) + gy * mesh.y(i));
    }
    return T;
}

// The sum of the areas of all the triangles.
double total_area(const Mesh2D& mesh) {
    double a = 0.0;
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        a += mesh.area(e);
    }
    return a;
}

// The "slow computation": K in a full n x n table, in which each triangle adds by hand its 9 numbers into the
// cells (t[i], t[j]). It is only used for comparison, and only for small meshes.
Dense slow_stiffness(const Mesh2D& mesh, double k) {
    Dense D(mesh.num_nodes(), std::vector<double>(mesh.num_nodes(), 0.0));
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Triangle& t = mesh.triangle(e);
        const Matrix3 Ke = element_stiffness(mesh, e, k);
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                D[t[i]][t[j]] += Ke[i][j];
            }
        }
    }
    return D;
}

// Compares A with a full table, cell by cell.
void expect_same_as_dense(const CSR& A, const Dense& D) {
    ASSERT_EQ(A.size(), D.size());
    double scale = 0.0;
    for (const auto& row : D) {
        for (double v : row) {
            scale = std::max(scale, std::fabs(v));
        }
    }
    for (std::size_t i = 0; i < D.size(); ++i) {
        for (std::size_t j = 0; j < D.size(); ++j) {
            EXPECT_NEAR(A.get(i, j), D[i][j], 1e-13 * scale) << "element (" << i << ", " << j << ")";
        }
    }
}

// The energy of a linear field and the equilibrium at the interior nodes (EXAMPLE 39). Adds to `interior` the
// number of interior nodes checked.
void check_linear_field(const Mesh2D& m, double k, std::size_t& interior) {
    const double t0 = 1.5;
    const double gx = 2.0;
    const double gy = -3.0;

    const CSR K = assemble_stiffness_2d(m, k);
    const std::vector<double> T = linear_field(m, t0, gx, gy);
    ASSERT_EQ(K.size(), T.size());
    const std::vector<double> KT = K * T;
    ASSERT_EQ(KT.size(), T.size());

    // T^T K T = k |grad T|^2 * area
    double energy = 0.0;
    double tmax = 0.0;
    for (std::size_t i = 0; i < T.size(); ++i) {
        energy += T[i] * KT[i];
        tmax = std::max(tmax, std::fabs(T[i]));
    }
    const double expected = k * (gx * gx + gy * gy) * total_area(m);
    EXPECT_NEAR(energy, expected, 1e-9 * expected);

    // (K T)[I] = 0 at the interior nodes
    std::vector<bool> on_boundary(m.num_nodes(), false);
    for (std::size_t node : boundary_nodes(m)) {
        on_boundary[node] = true;
    }
    const double tol = 1e-11 * largest_diagonal(K) * tmax;
    for (std::size_t i = 0; i < KT.size(); ++i) {
        if (!on_boundary[i]) {
            EXPECT_NEAR(KT[i], 0.0, tol) << "interior node " << i;
            ++interior;
        }
    }
}

}  // namespace

// EXAMPLE 37. Two 1 x 1 square cells side by side, rectangle_mesh(2, 1, 2, 1), with k = 2. The nodes, row by
// row, and the four triangles (the diagonal of each cell goes from the bottom-left node to the top-right
// one):
//
//         3 ----- 4 ----- 5            T0 = (0, 1, 4)   T1 = (0, 4, 3)      <- left cell
//         |     / |     / |            T2 = (1, 2, 5)   T3 = (1, 5, 4)      <- right cell
//         |   /   |   /   |
//         | /     | /     |
//         0 ----- 1 ----- 2
//
// Every triangle has area 1/2, so k / (4 A) = 1 and K[i][j] = b_i b_j + c_i c_j (the formula of EXAMPLE 11).
// The triangles are of two kinds: (a, b, c) = (bottom-left, bottom-right, top-right), for T0 and T2, and
// (a, c, d) = (bottom-left, top-right, top-left), for T1 and T3. With the vertices (0,0), (1,0), (1,1) and
// (0,0), (1,1), (0,1):
//
//     (a, b, c):  b = (-1, 1, 0)  c = (0, -1, 1)          (a, c, d):  b = (0, 1, -1)  c = (-1, 0, 1)
//
//                 [ 1 -1  0 ]                                         [ 1  0 -1 ]
//                 [-1  2 -1 ]                                         [ 0  1 -1 ]
//                 [ 0 -1  1 ]                                         [-1 -1  2 ]
//
// Now we add up, node by node. Node 1 is in T0 (as b: diagonal 2), in T2 (as a: 1) and in T3
// (as a: 1), so K[1][1] = 2 + 1 + 1 = 4. The pair (1, 4) is in T0 (as b-c: -1) and in T3 (as a-d: -1),
// so K[1][4] = -2. With all the nodes:
//
//         [ 2 -1  0 -1  0  0 ]
//         [-1  4 -1  0 -2  0 ]
//     K = [ 0 -1  2  0  0 -1 ]
//         [-1  0  0  2 -1  0 ]
//         [ 0 -2  0 -1  4 -1 ]
//         [ 0  0 -1  0 -1  2 ]
//
// Every row adds up to 0. Each pair of nodes that are in the same triangle takes one slot (including the
// pair of a node with itself), even if the sum is 0: the diagonals of the cells, 0-4 and 1-5, are each in two
// triangles, with coupling 0 in both (right angle), and they take a slot all the same (as in
// EXAMPLE 30). The pairs that are in no triangle (1-3, 2-4, 0-5...) are 0 and take up
// nothing. In total 6 + 2 * 9 = 24 slots: 6 on the diagonal and 9 off-diagonal pairs
// (0-1, 0-3, 0-4, 1-2, 1-4, 1-5, 2-5, 3-4, 4-5), each in two slots.
TEST(Assembly2d, StiffnessOfTwoSquareCellsMatchesHandComputation) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);

    const Dense want = {{2, -1, 0, -1, 0, 0},
                        {-1, 4, -1, 0, -2, 0},
                        {0, -1, 2, 0, 0, -1},
                        {-1, 0, 0, 2, -1, 0},
                        {0, -2, 0, -1, 4, -1},
                        {0, 0, -1, 0, -1, 2}};
    expect_matrix(assemble_stiffness_2d(m, 2.0), want, 24);
}

// EXAMPLE 38. Two things that hold for ANY mesh, as for the single triangle (EXAMPLE 15):
//   - K is symmetric;
//   - every row adds up to zero: if the temperature is the same at all the nodes no heat flows. In formulas:
//     the row I of K times the vector (1, ..., 1) is the sum of the contributions of the triangles around I, and every
//     row of every Ke adds up to zero.
// Here on the six rectangles of sample_grids: square, wide, narrow, small and very flat cells.
TEST(Assembly2d, StiffnessIsSymmetricAndAConstantTemperatureHasNoFlux) {
    for (const Grid& g : sample_grids()) {
        SCOPED_TRACE(describe(g));
        const CSR K = assemble_stiffness_2d(rectangle_mesh(g.lx, g.ly, g.nx, g.ny), 2.5);
        ASSERT_EQ(K.size(), nodes_of(g));
        const double tol = 1e-12 * largest_diagonal(K);

        EXPECT_TRUE(K.is_symmetric(tol));

        const std::vector<double> flux = K * std::vector<double>(K.size(), 1.0);
        ASSERT_EQ(flux.size(), K.size());
        for (std::size_t i = 0; i < flux.size(); ++i) {
            EXPECT_NEAR(flux[i], 0.0, tol) << "row " << i;
        }
    }
}

// EXAMPLE 39. The energy of a LINEAR temperature field T(x, y) = t0 + gx x + gy y over the whole mesh: the
// gradient is constant (gx, gy) everywhere, so
//     T^T K T = integral of k |grad T|^2 = k (gx^2 + gy^2) * (area of the mesh),
// the sum of the areas of the triangles. It is the same check as EXAMPLE 16, now on the assembled matrix
// (T_I is the temperature of node I).
//
// There is more. The row I of K times T is
//     (K T)[I] = integral of k grad T . grad N_I,    over the triangles that contain I,
// where N_I is the hat function of node I: it is 1 at I and drops to 0 on the boundary of the piece of mesh around I.
// With T linear, grad T = (gx, gy) is constant and comes out of the integral:
//     (K T)[I] = k (gx, gy) . integral of grad N_I.
// If I is INTERIOR (not on the boundary, according to boundary_nodes) that integral is 0: N_I is 0 on the whole
// boundary of the piece of mesh, and by the divergence theorem the integral of the gradient is the integral over
// the boundary of N_I times the normal, i.e. 0. So at every interior node the heat balance comes back to 0, for
// every linear field and on ANY mesh, even an irregular one. Not at the boundary nodes: there heat flows in and out.
// This is the "patch test", the classic check of finite elements, and it uses no value of K
// computed by hand. Here on the six rectangles, on the triangle with a node inside (EXAMPLE 8 of the meshes, area 9/2,
// so the energy is 2.5 * 13 * 9/2 = 146.25) and on an irregular quadrilateral with a node inside, made of
// four triangles that are all different (area 10.5).
TEST(Assembly2d, StiffnessGivesTheEnergyOfALinearFieldAndBalancesTheInteriorNodes) {
    const double k = 2.5;
    std::size_t interior = 0;

    for (const Grid& g : sample_grids()) {
        SCOPED_TRACE(describe(g));
        check_linear_field(rectangle_mesh(g.lx, g.ly, g.nx, g.ny), k, interior);
    }
    {
        SCOPED_TRACE("triangle with a node inside");
        check_linear_field(triangle_with_interior_node(), k, interior);
    }
    {
        SCOPED_TRACE("irregular quadrilateral with a node inside");
        check_linear_field(irregular_quadrilateral(), k, interior);
    }
    EXPECT_GT(interior, 30u) << "the check on the interior nodes must not be empty";
}

// EXAMPLE 40. The same K computed another way, slow and simple: a full n x n table, in which
// each triangle adds by hand its 9 numbers into the cells (t[i], t[j]). For small meshes this can be done;
// for large ones the full table does not fit in memory (EXAMPLE 43), and that is the reason for the CSR format. The
// two methods must give the same matrix in every cell, with non-trivial coefficients (k = 3.5, 0.7 and 1.3)
// and on three different meshes, the last one made of triangles that are all different.
TEST(Assembly2d, StiffnessMatchesTheSlowDenseAssembly) {
    {
        SCOPED_TRACE("rectangle 3 x 1.5 with 4 x 3 cells, k = 3.5");
        const Mesh2D m = rectangle_mesh(3.0, 1.5, 4, 3);
        expect_same_as_dense(assemble_stiffness_2d(m, 3.5), slow_stiffness(m, 3.5));
    }
    {
        SCOPED_TRACE("triangle with a node inside, k = 0.7");
        const Mesh2D m = triangle_with_interior_node();
        expect_same_as_dense(assemble_stiffness_2d(m, 0.7), slow_stiffness(m, 0.7));
    }
    {
        SCOPED_TRACE("irregular quadrilateral with a node inside, k = 1.3");
        const Mesh2D m = irregular_quadrilateral();
        expect_same_as_dense(assemble_stiffness_2d(m, 1.3), slow_stiffness(m, 1.3));
    }
}

// EXAMPLE 41. How many slots K occupies. The count is on the mesh, not on the numbers: one slot for each pair of
// nodes that are in the same triangle, and two nodes are in the same triangle if and only if a mesh
// EDGE joins them. So
//     slots = (one slot per node, on the diagonal) + 2 * (edges)       because every edge occupies (I, J) and (J, I).
// In a rectangle of nx x ny cells the nodes are (nx + 1)(ny + 1) and the edges are
//     horizontal nx (ny + 1)  +  vertical (nx + 1) ny  +  diagonal nx ny.
// For nx = 2, ny = 1: 6 nodes, 4 + 3 + 2 = 9 edges, 6 + 18 = 24 slots, as in EXAMPLE 37. It does not depend on k,
// nor on how many elements turn out to be 0.
TEST(Assembly2d, StiffnessOccupiesOnePlacePerNodeAndTwoPerEdge) {
    const std::size_t sizes[][2] = {{1, 1}, {2, 1}, {1, 2}, {3, 2}, {5, 4}, {1, 9}, {7, 1}};

    for (const auto& s : sizes) {
        const std::size_t nx = s[0];
        const std::size_t ny = s[1];
        const std::size_t nodes = (nx + 1) * (ny + 1);
        const std::size_t edges = nx * (ny + 1) + (nx + 1) * ny + nx * ny;

        for (double k : {1.0, 7.5}) {
            const CSR K = assemble_stiffness_2d(rectangle_mesh(1.0, 1.0, nx, ny), k);
            EXPECT_EQ(K.get_val_size(), nodes + 2 * edges) << nx << " x " << ny << " cells, k = " << k;
        }
    }
}

// EXAMPLE 42. A mesh in which two nodes belong to no triangle: 0 = (5, 5) and 5 = (7, 7). Nodes
// 1..4 are the unit square with the two triangles (1, 2, 3) and (1, 3, 4). The matrix still has a row and
// a column for EVERY node of the mesh (6 x 6), not only for those used by the triangles: rows 0 and 5 are
// empty. The size is mesh.num_nodes(), it cannot be derived from the triangles (the largest node they use is
// 4, and 4 + 1 would make 5). With k = 2 the block of nodes 1..4 is the unit square, 14 slots in total:
//
//         [ 0  0  0  0  0  0 ]
//         [ 0  2 -1  0 -1  0 ]
//         [ 0 -1  2 -1  0  0 ]
//     K = [ 0  0 -1  2 -1  0 ]
//         [ 0 -1  0 -1  2  0 ]
//         [ 0  0  0  0  0  0 ]
//
// (the slots 1-3 and 3-1 are the diagonal of the cell: occupied, with value 0).
TEST(Assembly2d, NodesOutsideEveryTriangleGetEmptyRowsAndColumns) {
    const Mesh2D m({5.0, 0.0, 1.0, 1.0, 0.0, 7.0}, {5.0, 0.0, 0.0, 1.0, 1.0, 7.0}, {{1, 2, 3}, {1, 3, 4}});
    ASSERT_EQ(m.num_nodes(), 6u);

    const Dense want = {{0, 0, 0, 0, 0, 0},
                        {0, 2, -1, 0, -1, 0},
                        {0, -1, 2, -1, 0, 0},
                        {0, 0, -1, 2, -1, 0},
                        {0, -1, 0, -1, 2, 0},
                        {0, 0, 0, 0, 0, 0}};
    expect_matrix(assemble_stiffness_2d(m, 2.0), want, 14);
}

// EXAMPLE 43. A mesh of 150 x 150 cells on the unit square: 22801 nodes and 45000 triangles. A full
// table would have 22801^2 = 520 million numbers (about 4 GB); in CSR format 158401 are enough (22801 on the
// diagonal and 2 * 67800 edges). If the assembly built a full table, or were quadratic in the number
// of nodes, this test would never finish. Checks: the slots (the formula of EXAMPLE 41), a constant field (every
// row adds up to 0) and the field T = x, whose energy is k * 1 * 1 = 1 whatever the fineness of the mesh.
TEST(Assembly2d, AssemblyOfALargeMeshStaysSparse) {
    const std::size_t n = 150;
    const Mesh2D m = rectangle_mesh(1.0, 1.0, n, n);
    const CSR K = assemble_stiffness_2d(m, 1.0);

    const std::size_t nodes = (n + 1) * (n + 1);
    const std::size_t edges = n * (n + 1) + (n + 1) * n + n * n;
    ASSERT_EQ(K.size(), nodes);
    EXPECT_EQ(K.get_val_size(), nodes + 2 * edges);

    const std::vector<double> flux = K * std::vector<double>(nodes, 1.0);
    for (std::size_t i = 0; i < nodes; ++i) {
        ASSERT_NEAR(flux[i], 0.0, 1e-12) << "row " << i;
    }

    std::vector<double> T;
    for (std::size_t i = 0; i < nodes; ++i) {
        T.push_back(m.x(i));
    }
    const std::vector<double> KT = K * T;
    double energy = 0.0;
    for (std::size_t i = 0; i < nodes; ++i) {
        energy += T[i] * KT[i];
    }
    EXPECT_NEAR(energy, 1.0, 1e-9);
}

// ---- Mass and load of the whole mesh ------------------------------------------------------------
//
// Same construction as K, with the contributions of other elements:
//   - assemble_mass_2d(mesh, rho_c) is the mass matrix M (n x n, CSR), with the contributions of element_mass.
//     It occupies the same slots as K: they only depend on the mesh.
//   - assemble_load_2d(mesh, q) is the load vector f (n numbers), with the contributions of element_load.
//     Here the triplets are not needed: f is a vector, and the number fe[i] of the triangle e is simply added
//     to the cell t[i]. A node that is in no triangle receives 0.

namespace {

// Like slow_stiffness, for the mass.
Dense slow_mass(const Mesh2D& mesh, double rho_c) {
    Dense D(mesh.num_nodes(), std::vector<double>(mesh.num_nodes(), 0.0));
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Triangle& t = mesh.triangle(e);
        const Matrix3 Me = element_mass(mesh, e, rho_c);
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                D[t[i]][t[j]] += Me[i][j];
            }
        }
    }
    return D;
}

// The x (or y) coordinates of the nodes, as a vector.
std::vector<double> coordinate(const Mesh2D& mesh, bool x) {
    std::vector<double> c;
    for (std::size_t i = 0; i < mesh.num_nodes(); ++i) {
        c.push_back(x ? mesh.x(i) : mesh.y(i));
    }
    return c;
}

// u^T M v.
double bilinear(const CSR& M, const std::vector<double>& u, const std::vector<double>& v) {
    const std::vector<double> Mv = M * v;
    double s = 0.0;
    for (std::size_t i = 0; i < u.size(); ++i) {
        s += u[i] * Mv[i];
    }
    return s;
}

// The mesh of EXAMPLE 13: two triangles of DIFFERENT area. 0 = (0, 0), 1 = (4, 0), 2 = (1, 2), 3 = (6, 4);
// T0 = (0, 1, 2) with A = 4 and T1 = (1, 3, 2) with A = 8.
Mesh2D two_triangles_of_different_area() {
    return Mesh2D({0.0, 4.0, 1.0, 6.0}, {0.0, 0.0, 2.0, 4.0}, {{0, 1, 2}, {1, 3, 2}});
}

}  // namespace

// EXAMPLE 44. The same two cells as EXAMPLE 37, with rho_c = 12. Every triangle has area 1/2, so
// rho_c A / 6 = 1 and rho_c A / 12 = 1/2, and the matrix of EVERY triangle is
//         [  1  1/2 1/2 ]
//         [ 1/2  1  1/2 ]
//         [ 1/2 1/2  1  ]
// The sum is easier than that of K, because the triangles are all equal: M[I][I] is 1 for every triangle
// that contains I, and M[I][J] is 1/2 for every triangle that contains both I and J.
//     triangles of each node:  0: T0 T1 (2)   1: T0 T2 T3 (3)   2: T2 (1)   3: T1 (1)   4: T0 T1 T3 (3)   5: T2 T3 (2)
//     pairs in TWO triangles: 0-4 (T0 T1), 1-4 (T0 T3), 1-5 (T2 T3), so 1;
//     the other six pairs (0-1, 0-3, 1-2, 2-5, 3-4, 4-5) are in a single triangle, so 1/2.
//
//         [  2  1/2  0  1/2  1   0  ]
//         [ 1/2  3  1/2  0   1   1  ]
//     M = [  0  1/2  1   0   0  1/2 ]          24 slots, the same as K
//         [ 1/2  0   0   1  1/2  0  ]
//         [  1   1   0  1/2  3  1/2 ]
//         [  0   1  1/2  0  1/2  2  ]
//
// The rows add up to 4, 6, 2, 2, 6, 4: for each node, rho_c * (areas of its triangles) / 3 = 12 * (n/2) / 3 = 2 n,
// with n the number of its triangles. In total it makes 24 = rho_c * area of the mesh = 12 * 2.
TEST(Assembly2d, MassOfTwoSquareCellsMatchesHandComputation) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);

    const Dense want = {{2, 0.5, 0, 0.5, 1, 0},
                        {0.5, 3, 0.5, 0, 1, 1},
                        {0, 0.5, 1, 0, 0, 0.5},
                        {0.5, 0, 0, 1, 0.5, 0},
                        {1, 1, 0, 0.5, 3, 0.5},
                        {0, 1, 0.5, 0, 0.5, 2}};
    expect_matrix(assemble_mass_2d(m, 12.0), want, 24);
}

// EXAMPLE 45. Two triangles of DIFFERENT AREA, the mesh of EXAMPLE 13: nodes 0 = (0, 0), 1 = (4, 0), 2 = (1, 2),
// 3 = (6, 4); T0 = (0, 1, 2) with A = 4 and T1 = (1, 3, 2) with A = 8. In the rectangle meshes all the triangles
// are equal and a mix-up between one triangle and another goes unseen: here it does not (the mass and the load of a
// triangle only depend on its area).
//
// With rho_c = 6:    T0: rho_c A / 6 = 4 and rho_c A / 12 = 2        T1: rho_c A / 6 = 8 and rho_c A / 12 = 4
// Node 0 is only in T0, node 3 only in T1, nodes 1 and 2 in both (diagonal 4 + 8, off-diagonal
// 2 + 4). Nodes 0 and 3 are never together in a triangle: 0 and no slot.
//
//         [ 4   2   2   0 ]
//     M = [ 2  12   6   4 ]            14 slots
//         [ 2   6  12   4 ]
//         [ 0   4   4   8 ]
//
// With q = 3 every triangle gives q A / 3 to each of its nodes: T0 gives 3 * 4 / 3 = 4, T1 gives 3 * 8 / 3 = 8. So
//     f = [4, 4 + 8, 4 + 8, 8] = [4, 12, 12, 8]       (in total 36 = q * (4 + 8))
TEST(Assembly2d, MassAndLoadOfTrianglesOfDifferentAreaMatchHandComputation) {
    const Mesh2D m = two_triangles_of_different_area();

    const Dense want = {{4, 2, 2, 0}, {2, 12, 6, 4}, {2, 6, 12, 4}, {0, 4, 4, 8}};
    expect_matrix(assemble_mass_2d(m, 6.0), want, 14);

    const std::vector<double> f = assemble_load_2d(m, 3.0);
    const std::vector<double> want_f = {4.0, 12.0, 12.0, 8.0};
    ASSERT_EQ(f.size(), want_f.size());
    for (std::size_t i = 0; i < want_f.size(); ++i) {
        EXPECT_DOUBLE_EQ(f[i], want_f[i]) << "node " << i;
    }
}

// EXAMPLE 46. Things that hold for ANY rectangle (here rho_c = 2.5, on the six of sample_grids):
//   - M is symmetric, and every row adds up to a positive number (the "mass" concentrated at the node);
//   - all the elements together add up to rho_c * (area of the mesh), the total heat capacity
//     (1^T M 1 = integral of rho_c);
//   - for two LINEAR fields a(x, y) and b(x, y), with their values at the nodes, a^T M b is the EXACT integral of
//     rho_c a b: the product has degree 2, and linear elements integrate it exactly (EXAMPLE 24). On a
//     rectangle [0, Lx] x [0, Ly]:
//         integral of x = Lx^2 Ly / 2     integral of x^2 = Lx^3 Ly / 3     integral of x y = Lx^2 Ly^2 / 4
//     So  x^T M 1,  x^T M x  and  x^T M y  must give these numbers, times rho_c, with ANY number of
//     cells: they are not approximations that improve with the mesh, they are exact.
TEST(Assembly2d, MassIsSymmetricPositiveAndIntegratesLinearFieldsExactly) {
    const double rho_c = 2.5;

    for (const Grid& g : sample_grids()) {
        SCOPED_TRACE(describe(g));
        const Mesh2D m = rectangle_mesh(g.lx, g.ly, g.nx, g.ny);
        const CSR M = assemble_mass_2d(m, rho_c);
        ASSERT_EQ(M.size(), nodes_of(g));

        const std::vector<double> one(M.size(), 1.0);
        const std::vector<double> x = coordinate(m, true);
        const std::vector<double> y = coordinate(m, false);
        const double area = g.lx * g.ly;
        const double tol = 1e-12 * rho_c * area;

        EXPECT_TRUE(M.is_symmetric(1e-12 * largest_diagonal(M)));
        const std::vector<double> lumped = M * one;
        for (std::size_t i = 0; i < lumped.size(); ++i) {
            EXPECT_GT(lumped[i], 0.0) << "row " << i;
        }

        EXPECT_NEAR(bilinear(M, one, one), rho_c * area, tol);
        EXPECT_NEAR(bilinear(M, x, one), rho_c * g.lx * g.lx * g.ly / 2.0, tol * g.lx);
        EXPECT_NEAR(bilinear(M, x, x), rho_c * g.lx * g.lx * g.lx * g.ly / 3.0, tol * g.lx * g.lx);
        EXPECT_NEAR(bilinear(M, x, y), rho_c * g.lx * g.lx * g.ly * g.ly / 4.0, tol * g.lx * g.ly);
    }
}

// EXAMPLE 47. Like EXAMPLE 40, for the mass: the same M with the slow computation (full table), on three meshes
// (rho_c = 3.5, 0.7 and 1.3), the last made of triangles that all have different areas. Also, the mass occupies the same
// slots as the stiffness: they depend on the mesh, not on the element.
TEST(Assembly2d, MassMatchesTheSlowDenseAssemblyAndHasTheSamePlacesAsStiffness) {
    {
        SCOPED_TRACE("rectangle 3 x 1.5 with 4 x 3 cells, rho_c = 3.5");
        const Mesh2D m = rectangle_mesh(3.0, 1.5, 4, 3);
        const CSR M = assemble_mass_2d(m, 3.5);
        expect_same_as_dense(M, slow_mass(m, 3.5));
        EXPECT_EQ(M.get_val_size(), assemble_stiffness_2d(m, 1.0).get_val_size());
    }
    {
        SCOPED_TRACE("triangle with a node inside, rho_c = 0.7");
        const Mesh2D m = triangle_with_interior_node();
        const CSR M = assemble_mass_2d(m, 0.7);
        expect_same_as_dense(M, slow_mass(m, 0.7));
        EXPECT_EQ(M.get_val_size(), assemble_stiffness_2d(m, 1.0).get_val_size());
    }
    {
        SCOPED_TRACE("irregular quadrilateral with a node inside, rho_c = 1.3");
        const Mesh2D m = irregular_quadrilateral();
        const CSR M = assemble_mass_2d(m, 1.3);
        expect_same_as_dense(M, slow_mass(m, 1.3));
        EXPECT_EQ(M.get_val_size(), assemble_stiffness_2d(m, 1.0).get_val_size());
    }
}

// EXAMPLE 48. The load of the same two cells as EXAMPLE 37, with q = 6. Every triangle has area 1/2 and gives
// q A / 3 = 1 to each of its three nodes. So f[I] is the number of triangles of node I (EXAMPLE 44):
//     f = [2, 3, 1, 1, 3, 2]
// and in total it makes 12 = q * area of the mesh = 6 * 2: the heat produced by the source in the whole plate.
TEST(Assembly2d, LoadOfTwoSquareCellsMatchesHandComputation) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);

    const std::vector<double> f = assemble_load_2d(m, 6.0);
    const std::vector<double> want = {2.0, 3.0, 1.0, 1.0, 3.0, 2.0};
    ASSERT_EQ(f.size(), want.size());
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_DOUBLE_EQ(f[i], want[i]) << "node " << i;
    }
}

// EXAMPLE 49. Two things that hold for ANY mesh (here q = 7, rho_c = 2.5):
//   - the total heat: the sum of f is q * (area of the mesh);
//   - the load and the mass are linked, as for the triangle (EXAMPLE 27): f[I] = q / rho_c * (row I of M)
//     summed, i.e. f = (q / rho_c) * (M times the vector of all 1).
// With negative q (a heat sink) everything changes sign. On the six rectangles, on the triangle with a node inside,
// on the irregular quadrilateral and on the two triangles of different area.
TEST(Assembly2d, LoadAddsUpToTheTotalHeatAndIsTheRowSumOfTheMass) {
    const double rho_c = 2.5;

    std::vector<Mesh2D> meshes;
    for (const Grid& g : sample_grids()) {
        meshes.push_back(rectangle_mesh(g.lx, g.ly, g.nx, g.ny));
    }
    meshes.push_back(triangle_with_interior_node());
    meshes.push_back(irregular_quadrilateral());
    meshes.push_back(two_triangles_of_different_area());

    for (std::size_t index = 0; index < meshes.size(); ++index) {
        SCOPED_TRACE("mesh number " + std::to_string(index));
        const Mesh2D& m = meshes[index];
        const double area = total_area(m);
        const std::vector<double> lumped = assemble_mass_2d(m, rho_c) * std::vector<double>(m.num_nodes(), 1.0);

        for (double q : {7.0, -7.0}) {
            SCOPED_TRACE("q = " + std::to_string(q));
            const std::vector<double> f = assemble_load_2d(m, q);
            ASSERT_EQ(f.size(), m.num_nodes());
            const double tol = 1e-12 * std::fabs(q) * area;

            double total = 0.0;
            for (std::size_t i = 0; i < f.size(); ++i) {
                total += f[i];
                EXPECT_NEAR(f[i], q / rho_c * lumped[i], tol) << "node " << i;
            }
            EXPECT_NEAR(total, q * area, tol);
        }
    }
}

// EXAMPLE 50. The mesh of EXAMPLE 42, with nodes 0 and 5 outside every triangle. With rho_c = 12 and q = 6, the
// two triangles (1, 2, 3) and (1, 3, 4) have area 1/2, so the matrix of each triangle is the one of EXAMPLE
// 44 and the load per node is 1:
//         [ 0   0   0   0   0  0 ]
//         [ 0   2  1/2  1  1/2 0 ]
//         [ 0  1/2  1  1/2  0  0 ]                       f = [0, 2, 1, 2, 1, 0]
//     M = [ 0   1  1/2  2  1/2 0 ]       14 slots
//         [ 0  1/2  0  1/2  1  0 ]
//         [ 0   0   0   0   0  0 ]
// The sizes are those of the mesh (6), the isolated nodes have an empty row in M and load 0.
TEST(Assembly2d, MassAndLoadKeepOneRowPerNodeEvenForNodesOutsideEveryTriangle) {
    const Mesh2D m({5.0, 0.0, 1.0, 1.0, 0.0, 7.0}, {5.0, 0.0, 0.0, 1.0, 1.0, 7.0}, {{1, 2, 3}, {1, 3, 4}});
    ASSERT_EQ(m.num_nodes(), 6u);

    const Dense want = {{0, 0, 0, 0, 0, 0},
                        {0, 2, 0.5, 1, 0.5, 0},
                        {0, 0.5, 1, 0.5, 0, 0},
                        {0, 1, 0.5, 2, 0.5, 0},
                        {0, 0.5, 0, 0.5, 1, 0},
                        {0, 0, 0, 0, 0, 0}};
    expect_matrix(assemble_mass_2d(m, 12.0), want, 14);

    const std::vector<double> f = assemble_load_2d(m, 6.0);
    const std::vector<double> want_f = {0.0, 2.0, 1.0, 2.0, 1.0, 0.0};
    ASSERT_EQ(f.size(), want_f.size());
    for (std::size_t i = 0; i < want_f.size(); ++i) {
        EXPECT_DOUBLE_EQ(f[i], want_f[i]) << "node " << i;
    }
}

// ---- The triplets before adding them up -----------------------------------------------------------
//
// assemble_stiffness_2d and assemble_mass_2d return a CSR, and a CSR cannot be modified: it offers no way
// to change a row. The Dirichlet conditions (next step) change rows and columns of K, so they have to be
// applied BEFORE, on the list of triplets (row, column, value) from which the CSR is built. For this reason the
// two lists become public:
//   - stiffness_triplets_2d(mesh, k): the triplets of K;
//   - mass_triplets_2d(mesh, rho_c): the triplets of M.
// They are the contributions of the triangles, still to be added up: 9 triplets per triangle, (t[i], t[j], Ke[i][j])
// with i and j from 0 to 2, in any order. Triplets with the same position are NOT added up (csr_from_triplets does
// that) and those with value 0 are NOT removed. For whoever uses them, assemble_stiffness_2d and assemble_mass_2d do
// not change: they are the CSR of these lists.

namespace {

// The values of the triplets at position (row, col), in increasing order.
std::vector<double> values_at(const std::vector<Triplet>& triplets, std::size_t row, std::size_t col) {
    std::vector<double> out;
    for (const Triplet& t : triplets) {
        const auto& [r, c, v] = t;
        if (r == row && c == col) {
            out.push_back(v);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Compares the values of the triplets at position (row, col) with `want` (in increasing order).
void expect_values_at(const std::vector<Triplet>& triplets, std::size_t row, std::size_t col,
                      const std::vector<double>& want) {
    const std::vector<double> got = values_at(triplets, row, col);
    ASSERT_EQ(got.size(), want.size()) << "position (" << row << ", " << col << ")";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_DOUBLE_EQ(got[i], want[i]) << "position (" << row << ", " << col << "), value number " << i;
    }
}

// How many DIFFERENT positions (row, column) appear in the list, in an n x n matrix.
std::size_t distinct_places(const std::vector<Triplet>& triplets, std::size_t n) {
    std::vector<std::size_t> places;
    for (const Triplet& t : triplets) {
        const auto& [r, c, v] = t;
        places.push_back(r * n + c);
    }
    std::sort(places.begin(), places.end());
    places.erase(std::unique(places.begin(), places.end()), places.end());
    return places.size();
}

// The mesh of EXAMPLE 42: nodes 0 and 5 are in no triangle.
Mesh2D mesh_with_nodes_outside_every_triangle() {
    return Mesh2D({5.0, 0.0, 1.0, 1.0, 0.0, 7.0}, {5.0, 0.0, 0.0, 1.0, 1.0, 7.0}, {{1, 2, 3}, {1, 3, 4}});
}

}  // namespace

// EXAMPLE 51. The triplets of K for the two cells of EXAMPLE 37 (k = 2). Each triangle gives 9 of them and there are
// 4 triangles, so 36 triplets, but the different positions are only 24 (the 24 slots of K): the other 12 triplets fall on
// slots already taken by another triangle. With the triangle matrices of EXAMPLE 37:
//     (1, 1): T0 gives 2, T2 gives 1 and T3 gives 1       2 + 1 + 1 = 4 = K[1][1]
//     (1, 4): T0 gives -1 and T3 gives -1                 -2 = K[1][4], and the same in (4, 1)
//     (0, 4): T0 and T1 both give 0                       the slot exists all the same and the triplets must be kept
//     (2, 2): only T2, which gives 2
//     (1, 3): none (1 and 3 are never in the same triangle)
// Added up with csr_from_triplets, the 36 triplets give the K of EXAMPLE 37.
TEST(Assembly2d, StiffnessTripletsAreTheContributionsOfEveryTriangle) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const std::vector<Triplet> trip = stiffness_triplets_2d(m, 2.0);

    ASSERT_EQ(trip.size(), 36u);
    EXPECT_EQ(distinct_places(trip, 6), 24u);
    expect_values_at(trip, 1, 1, {1, 1, 2});
    expect_values_at(trip, 1, 4, {-1, -1});
    expect_values_at(trip, 4, 1, {-1, -1});
    expect_values_at(trip, 0, 4, {0, 0});
    expect_values_at(trip, 2, 2, {2});
    expect_values_at(trip, 1, 3, {});

    const Dense want = {{2, -1, 0, -1, 0, 0},
                        {-1, 4, -1, 0, -2, 0},
                        {0, -1, 2, 0, 0, -1},
                        {-1, 0, 0, 2, -1, 0},
                        {0, -2, 0, -1, 4, -1},
                        {0, 0, -1, 0, -1, 2}};
    expect_matrix(csr_from_triplets(m.num_nodes(), trip), want, 24);
}

// EXAMPLE 52. The same for the mass, with rho_c = 12 (EXAMPLE 44): 36 triplets in 24 different positions. Each
// triangle has the matrix [1 1/2 1/2; 1/2 1 1/2; 1/2 1/2 1], so
//     (1, 1): T0, T2 and T3 give 1 each         1 + 1 + 1 = 3 = M[1][1]
//     (0, 4): T0 and T1 give 1/2 each           1 = M[0][4]
//     (2, 5): only T2, which gives 1/2
//     (1, 3): none
// and added up they give the M of EXAMPLE 44.
TEST(Assembly2d, MassTripletsAreTheContributionsOfEveryTriangle) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const std::vector<Triplet> trip = mass_triplets_2d(m, 12.0);

    ASSERT_EQ(trip.size(), 36u);
    EXPECT_EQ(distinct_places(trip, 6), 24u);
    expect_values_at(trip, 1, 1, {1, 1, 1});
    expect_values_at(trip, 0, 4, {0.5, 0.5});
    expect_values_at(trip, 4, 0, {0.5, 0.5});
    expect_values_at(trip, 2, 5, {0.5});
    expect_values_at(trip, 1, 3, {});

    const Dense want = {{2, 0.5, 0, 0.5, 1, 0},
                        {0.5, 3, 0.5, 0, 1, 1},
                        {0, 0.5, 1, 0, 0, 0.5},
                        {0.5, 0, 0, 1, 0.5, 0},
                        {1, 1, 0, 0.5, 3, 0.5},
                        {0, 1, 0.5, 0, 0.5, 2}};
    expect_matrix(csr_from_triplets(m.num_nodes(), trip), want, 24);
}

// EXAMPLE 53. The two lists on meshes of different shapes, compared with the slow computation of EXAMPLE 40 (the full
// table in which each triangle adds by hand its 9 numbers):
//   - 9 triplets per triangle;
//   - all the indices less than the number of nodes (otherwise csr_from_triplets throws std::invalid_argument);
//   - added up, they give the same K (and the same M) as the slow computation.
// The meshes: a rectangle, the triangle with a node inside, the irregular quadrilateral, two triangles of
// different area and the mesh with two nodes outside every triangle.
TEST(Assembly2d, TripletsSumToTheSlowDenseAssemblyOnEveryKindOfMesh) {
    std::vector<Mesh2D> meshes;
    meshes.push_back(rectangle_mesh(3.0, 1.5, 4, 3));
    meshes.push_back(triangle_with_interior_node());
    meshes.push_back(irregular_quadrilateral());
    meshes.push_back(two_triangles_of_different_area());
    meshes.push_back(mesh_with_nodes_outside_every_triangle());

    for (std::size_t index = 0; index < meshes.size(); ++index) {
        SCOPED_TRACE("mesh number " + std::to_string(index));
        const Mesh2D& m = meshes[index];

        const std::vector<Triplet> stiff = stiffness_triplets_2d(m, 3.5);
        ASSERT_EQ(stiff.size(), 9 * m.num_triangles());
        expect_same_as_dense(csr_from_triplets(m.num_nodes(), stiff), slow_stiffness(m, 3.5));

        const std::vector<Triplet> mass = mass_triplets_2d(m, 0.7);
        ASSERT_EQ(mass.size(), 9 * m.num_triangles());
        expect_same_as_dense(csr_from_triplets(m.num_nodes(), mass), slow_mass(m, 0.7));
    }
}

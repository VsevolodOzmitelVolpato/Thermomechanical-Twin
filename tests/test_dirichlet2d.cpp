#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "Assembly2D.h"
#include "CSR_and_Vectors.h"
#include "Dirichlet2D.h"
#include "Mesh2D.h"
#include "Triplets.h"

// ---- Imposed temperatures (Dirichlet): the matrix ---------------------------------------------------------
//
// The system K T = f has one row per node. If the temperature of some nodes is imposed, T[d] = g[d] for the nodes d
// "fixed", for those nodes the unknown is already known. Call D the set of fixed nodes and F the others, the
// free ones. The rows of the free nodes say
//     K_FF T_F + K_FD g = f_F        i.e.        K_FF T_F = f_F - K_FD g
// (K_FF is K restricted to the free nodes, K_FD links the free nodes to the fixed ones), while the rows of the fixed nodes
// only say T_D = g. To still have a square n x n and SYMMETRIC system (CG requires it) we keep
// all n rows: those of the fixed nodes become rows of the identity and their columns are zeroed (the
// piece K_FD g goes to the right-hand side: that is the next step). The new matrix is
//
//     A = [ K_FF  0 ]        free nodes first, then the fixed ones
//         [  0    I ]
//
// and it only depends on WHICH nodes are fixed, not on the values g. It is built on the list of triplets of K:
//   - the triplets (r, c, v) with the row r or the column c fixed are thrown away;
//   - for each fixed node d the triplet (d, d, 1) is added.
// dirichlet_matrix_triplets_2d(n, triplets, fixed_nodes) returns the new list; the one received does not change.
// The triplets that remain are those from before, NOT added up, and those with value 0 are NOT removed. It throws
// std::invalid_argument if a fixed node is >= n, if a node appears twice in fixed_nodes (two triplets
// (d, d, 1) would add up to 2) or if a triplet received has its row or column >= n.

namespace {

using Dense = std::vector<std::vector<double>>;
using Nodes = std::vector<std::size_t>;

// The CSR written as a full n x n table, read with get(i, j).
Dense to_dense(const CSR& A) {
    const std::size_t n = A.size();
    Dense out(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            out[i][j] = A.get(i, j);
        }
    }
    return out;
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

// The "slow computation" of the matrix: the table K with the rows and columns of the fixed nodes zeroed and 1 on
// their diagonal.
Dense slow_dirichlet(Dense K, const Nodes& fixed) {
    for (std::size_t d : fixed) {
        for (std::size_t i = 0; i < K.size(); ++i) {
            K[d][i] = 0.0;
            K[i][d] = 0.0;
        }
        K[d][d] = 1.0;
    }
    return K;
}

// How many slots the matrix must occupy: one for each fixed node (its diagonal) and one for each DIFFERENT
// position (row, column) of the triplets with row and column both free.
std::size_t expected_places(const std::vector<Triplet>& trip, std::size_t n, const Nodes& fixed) {
    std::vector<bool> is_fixed(n, false);
    for (std::size_t d : fixed) {
        is_fixed[d] = true;
    }
    std::vector<Triplet> both_free;
    for (const Triplet& t : trip) {
        const auto& [r, c, v] = t;
        if (!is_fixed[r] && !is_fixed[c]) {
            both_free.push_back(t);
        }
    }
    return fixed.size() + distinct_places(both_free, n);
}

// The large triangle with a node inside (EXAMPLE 8 of the meshes): it is not a rectangle.
Mesh2D triangle_with_interior_node() {
    return Mesh2D({0.0, 3.0, 0.0, 1.0}, {0.0, 0.0, 3.0, 1.0}, {{0, 1, 3}, {1, 2, 3}, {2, 0, 3}});
}

// The irregular quadrilateral with a node inside (EXAMPLE 39): four triangles of different area.
Mesh2D irregular_quadrilateral() {
    return Mesh2D({0.0, 4.0, 3.0, -1.0, 1.0}, {0.0, 0.0, 3.0, 2.0, 0.7},
                  {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}});
}

// The mesh of EXAMPLE 42: nodes 0 and 5 are in no triangle.
Mesh2D mesh_with_nodes_outside_every_triangle() {
    return Mesh2D({5.0, 0.0, 1.0, 1.0, 0.0, 7.0}, {5.0, 0.0, 0.0, 1.0, 1.0, 7.0}, {{1, 2, 3}, {1, 3, 4}});
}

}  // namespace

// EXAMPLE 54. The two square cells of EXAMPLE 37 (k = 2), with the temperature imposed on the left side (nodes
// 0 and 3) and on the right side (nodes 2 and 5). The free nodes are 1 and 4. Of the 36 triplets of K only those with
// row AND column among the free nodes remain:
//     (1, 1): 3 triplets      (1, 4): 2 triplets      (4, 1): 2 triplets      (4, 4): 3 triplets
// i.e. 10 triplets, plus 4 triplets (d, d, 1) for the fixed nodes: 14 in total. Added up they give
//
//         [ 1  0  0  0  0  0 ]
//         [ 0  4  0  0 -2  0 ]
//         [ 0  0  1  0  0  0 ]
//     A = [ 0  0  0  1  0  0 ]
//         [ 0 -2  0  0  4  0 ]
//         [ 0  0  0  0  0  1 ]
//
// which occupies 8 slots: the 4 of the diagonal of the fixed nodes and the 4 of the free 2 x 2 block.
TEST(Dirichlet2d, TwoSquareCellsWithTheLeftAndRightSidesFixedMatchHandComputation) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const std::vector<Triplet> trip = stiffness_triplets_2d(m, 2.0);
    const std::vector<Triplet> out = dirichlet_matrix_triplets_2d(m.num_nodes(), trip, {0, 2, 3, 5});

    ASSERT_EQ(out.size(), 14u);
    expect_values_at(out, 0, 0, {1});
    expect_values_at(out, 2, 2, {1});
    expect_values_at(out, 3, 3, {1});
    expect_values_at(out, 5, 5, {1});
    expect_values_at(out, 1, 1, {1, 1, 2});
    expect_values_at(out, 4, 4, {1, 1, 2});
    expect_values_at(out, 1, 4, {-1, -1});
    expect_values_at(out, 4, 1, {-1, -1});
    expect_values_at(out, 0, 1, {});
    expect_values_at(out, 1, 0, {});
    expect_values_at(out, 1, 2, {});
    expect_values_at(out, 2, 1, {});
    expect_values_at(out, 3, 4, {});
    expect_values_at(out, 4, 3, {});

    const Dense want = {{1, 0, 0, 0, 0, 0},
                        {0, 4, 0, 0, -2, 0},
                        {0, 0, 1, 0, 0, 0},
                        {0, 0, 0, 1, 0, 0},
                        {0, -2, 0, 0, 4, 0},
                        {0, 0, 0, 0, 0, 1}};
    expect_matrix(csr_from_triplets(m.num_nodes(), out), want, 8);
}

// EXAMPLE 55. A hand-made list, with no mesh, to see what happens to EVERY triplet. n = 4 and the fixed nodes
// are 1 and 3. The 11 triplets received (row, column, value):
//     (0, 0, 1)  (0, 0, 2)  (0, 1, 5)  (1, 0, 5)  (1, 1, 3)  (1, 2, -1)  (2, 1, -1)  (2, 2, 4)  (2, 2, 6)
//     (0, 2, 0)  (3, 3, 7)
// The 6 with the row or the column fixed are thrown away: (0, 1), (1, 0), (1, 1), (1, 2), (2, 1) and (3, 3).
// 5 triplets remain, including those with the same position ((0, 0) twice and (2, 2) twice: they are NOT added up)
// and the one with value 0, plus (1, 1, 1) and (3, 3, 1): 7 in total. Added up they give a matrix that occupies 5
// slots, one of which is 0:
//         [ 3  0  0  0 ]
//     A = [ 0  1  0  0 ]
//         [ 0  0 10  0 ]
//         [ 0  0  0  1 ]
TEST(Dirichlet2d, OnlyTheTripletsOfFixedNodesAreDroppedAndNothingIsSummed) {
    const std::vector<Triplet> trip = {{0, 0, 1.0}, {0, 0, 2.0}, {0, 1, 5.0}, {1, 0, 5.0}, {1, 1, 3.0}, {1, 2, -1.0},
                                       {2, 1, -1.0}, {2, 2, 4.0}, {2, 2, 6.0}, {0, 2, 0.0}, {3, 3, 7.0}};
    const std::vector<Triplet> out = dirichlet_matrix_triplets_2d(4, trip, {1, 3});

    ASSERT_EQ(out.size(), 7u);
    expect_values_at(out, 0, 0, {1, 2});
    expect_values_at(out, 2, 2, {4, 6});
    expect_values_at(out, 0, 2, {0});
    expect_values_at(out, 1, 1, {1});
    expect_values_at(out, 3, 3, {1});
    expect_values_at(out, 0, 1, {});
    expect_values_at(out, 1, 0, {});
    expect_values_at(out, 1, 2, {});
    expect_values_at(out, 2, 1, {});

    const Dense want = {{3, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 10, 0}, {0, 0, 0, 1}};
    expect_matrix(csr_from_triplets(4, out), want, 5);
}

// EXAMPLE 56. The two extreme cases on the two cells of EXAMPLE 37.
//   - No fixed node: the list stays as it was, all 36 triplets, and the matrix is K.
//   - All the nodes fixed: only the 6 triplets (d, d, 1) remain and the matrix is the 6 x 6 identity, with 6 slots.
TEST(Dirichlet2d, NoFixedNodesChangesNothingAndAllFixedLeavesTheIdentity) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const std::vector<Triplet> trip = stiffness_triplets_2d(m, 2.0);

    const std::vector<Triplet> same = dirichlet_matrix_triplets_2d(6, trip, {});
    ASSERT_EQ(same.size(), 36u);
    EXPECT_EQ(distinct_places(same, 6), 24u);
    const Dense K = {{2, -1, 0, -1, 0, 0},
                     {-1, 4, -1, 0, -2, 0},
                     {0, -1, 2, 0, 0, -1},
                     {-1, 0, 0, 2, -1, 0},
                     {0, -2, 0, -1, 4, -1},
                     {0, 0, -1, 0, -1, 2}};
    expect_matrix(csr_from_triplets(6, same), K, 24);

    const std::vector<Triplet> all = dirichlet_matrix_triplets_2d(6, trip, {0, 1, 2, 3, 4, 5});
    ASSERT_EQ(all.size(), 6u);
    for (std::size_t d = 0; d < 6; ++d) {
        expect_values_at(all, d, d, {1});
    }
    const Dense identity = {{1, 0, 0, 0, 0, 0},
                            {0, 1, 0, 0, 0, 0},
                            {0, 0, 1, 0, 0, 0},
                            {0, 0, 0, 1, 0, 0},
                            {0, 0, 0, 0, 1, 0},
                            {0, 0, 0, 0, 0, 1}};
    expect_matrix(csr_from_triplets(6, all), identity, 6);
}

// EXAMPLE 57. The result on meshes of different shapes and for different sets of nodes, compared with the "slow
// computation": the full table of K in which the rows and columns of the fixed nodes are zeroed by hand and 1 is put
// on their diagonal. The meshes: a rectangle, the triangle with a node inside, the irregular quadrilateral, the
// mesh with two nodes outside every triangle and the square of two triangles. The sets: node 0 alone, the last
// node alone, the whole boundary (boundary_nodes), every other node, all the nodes.
// The occupied slots are: one slot for each fixed node, plus the different positions of the triplets with row and
// column both free (no zero is left in the rows and columns of the fixed nodes).
TEST(Dirichlet2d, MatchesTheSlowDenseEliminationOnEveryKindOfMesh) {
    std::vector<Mesh2D> meshes;
    meshes.push_back(rectangle_mesh(3.0, 1.5, 4, 3));
    meshes.push_back(triangle_with_interior_node());
    meshes.push_back(irregular_quadrilateral());
    meshes.push_back(mesh_with_nodes_outside_every_triangle());
    meshes.push_back(rectangle_mesh(1.0, 1.0, 1, 1));

    for (std::size_t index = 0; index < meshes.size(); ++index) {
        const Mesh2D& m = meshes[index];
        const std::size_t n = m.num_nodes();
        const std::vector<Triplet> trip = stiffness_triplets_2d(m, 3.5);
        const Dense K = to_dense(csr_from_triplets(n, trip));

        Nodes first;
        first.push_back(0);
        Nodes last;
        last.push_back(n - 1);
        Nodes every_second;
        for (std::size_t i = 0; i < n; i += 2) {
            every_second.push_back(i);
        }
        Nodes everything;
        for (std::size_t i = 0; i < n; ++i) {
            everything.push_back(i);
        }
        std::vector<Nodes> sets;
        sets.push_back(first);
        sets.push_back(last);
        sets.push_back(boundary_nodes(m));
        sets.push_back(every_second);
        sets.push_back(everything);

        for (std::size_t s = 0; s < sets.size(); ++s) {
            SCOPED_TRACE("mesh number " + std::to_string(index) + ", node set number " + std::to_string(s));
            const std::vector<Triplet> out = dirichlet_matrix_triplets_2d(n, trip, sets[s]);
            expect_matrix(csr_from_triplets(n, out), slow_dirichlet(K, sets[s]), expected_places(trip, n, sets[s]));
        }
    }
}

// EXAMPLE 58. The whole boundary of a rectangle fixed: the interior nodes remain. The occupied slots are
//     one slot per node (the identity on the boundary nodes, the diagonal of the free block on the interior ones)
//     + 2 for each EDGE between two interior nodes (each edge occupies (I, J) and (J, I)).
// In a rectangle of nx x ny cells (nx, ny >= 2) the interior nodes are (nx - 1)(ny - 1) and the edges between two
// interior nodes are
//     horizontal (nx - 2)(ny - 1)  +  vertical (nx - 1)(ny - 2)  +  diagonal (nx - 2)(ny - 2).
// For nx = 4, ny = 3: 20 nodes, 6 interior, interior edges 2 * 2 + 3 * 1 + 2 * 1 = 9, slots 20 + 2 * 9 = 38.
// Also, the rows and columns of the boundary nodes are those of the identity.
TEST(Dirichlet2d, FixingTheWholeBoundaryLeavesOnePlacePerNodeAndTwoPerInnerEdge) {
    const std::size_t sizes[][2] = {{2, 2}, {3, 2}, {2, 4}, {4, 3}, {6, 6}, {10, 7}};

    for (const auto& s : sizes) {
        const std::size_t nx = s[0];
        const std::size_t ny = s[1];
        SCOPED_TRACE(std::to_string(nx) + " x " + std::to_string(ny) + " cells");
        const Mesh2D m = rectangle_mesh(1.0, 1.0, nx, ny);
        const std::size_t nodes = (nx + 1) * (ny + 1);
        const std::size_t inner_edges = (nx - 2) * (ny - 1) + (nx - 1) * (ny - 2) + (nx - 2) * (ny - 2);

        const Nodes boundary = boundary_nodes(m);
        const std::vector<Triplet> trip = stiffness_triplets_2d(m, 1.0);
        const CSR A = csr_from_triplets(nodes, dirichlet_matrix_triplets_2d(nodes, trip, boundary));
        EXPECT_EQ(A.get_val_size(), nodes + 2 * inner_edges);

        for (std::size_t d : boundary) {
            for (std::size_t c = 0; c < nodes; ++c) {
                ASSERT_DOUBLE_EQ(A.get(d, c), d == c ? 1.0 : 0.0) << "row " << d << ", column " << c;
                ASSERT_DOUBLE_EQ(A.get(c, d), d == c ? 1.0 : 0.0) << "row " << c << ", column " << d;
            }
        }
    }
}

// EXAMPLE 59. The errors. With n = 3 the nodes are 0, 1, 2:
//   - a fixed node equal to n (it does not exist: the last one is n - 1) or larger, even huge (a billion);
//   - a repeated node, even if it is not next to its twin;
//   - a triplet with the row or the column equal to n, or huge.
// throw std::invalid_argument, even if the list of fixed nodes is empty. Node n - 1, the empty list and
// all the nodes in shuffled order are fine instead.
// The huge index is there for a precise check: if the function uses the index to read the table of fixed
// nodes BEFORE comparing it with n, it reads a place billions of bits away from the table and the
// program crashes instead of throwing the exception.
TEST(Dirichlet2d, BadNodesAndBadTripletsThrowInvalidArgument) {
    const std::vector<Triplet> trip = {{0, 0, 2.0}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 2.0}, {2, 2, 5.0}};

    EXPECT_NO_THROW(dirichlet_matrix_triplets_2d(3, trip, {}));
    EXPECT_NO_THROW(dirichlet_matrix_triplets_2d(3, trip, {2}));
    EXPECT_NO_THROW(dirichlet_matrix_triplets_2d(3, trip, {2, 0, 1}));

    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, trip, {3}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, trip, {1000}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, trip, {1000000000}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, trip, {0, 7}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, trip, {1, 1}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, trip, {0, 2, 1, 2}), std::invalid_argument);

    const std::vector<Triplet> bad_row = {{0, 0, 1.0}, {3, 0, 1.0}};
    const std::vector<Triplet> bad_col = {{0, 0, 1.0}, {0, 3, 1.0}};
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, bad_row, {}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, bad_col, {}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, bad_row, {1}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, bad_col, {1}), std::invalid_argument);

    const std::vector<Triplet> huge_row = {{0, 0, 1.0}, {1000000000, 0, 1.0}};
    const std::vector<Triplet> huge_col = {{0, 0, 1.0}, {0, 1000000000, 1.0}};
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, huge_row, {}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, huge_col, {}), std::invalid_argument);
}

// EXAMPLE 60. Three properties that the solver needs.
//   - A is symmetric (is_symmetric(1e-12)): CG checks it and rejects the others.
//   - The diagonal of A is positive: the Jacobi preconditioner divides by it and rejects the others.
//   - The order of the fixed nodes does not change the matrix: {0, 2, 3, 5} and {5, 0, 3, 2} give the same A.
TEST(Dirichlet2d, TheResultIsSymmetricWithPositiveDiagonalAndDoesNotDependOnTheOrderOfTheNodes) {
    const Mesh2D m = rectangle_mesh(3.0, 1.5, 4, 3);
    const std::size_t n = m.num_nodes();
    const std::vector<Triplet> trip = stiffness_triplets_2d(m, 3.5);

    std::vector<Nodes> sets;
    sets.push_back(boundary_nodes(m));
    Nodes only_first;
    only_first.push_back(0);
    sets.push_back(only_first);
    Nodes scattered;
    scattered.push_back(n - 1);
    scattered.push_back(0);
    scattered.push_back(7);
    sets.push_back(scattered);

    for (std::size_t s = 0; s < sets.size(); ++s) {
        SCOPED_TRACE("node set number " + std::to_string(s));
        const CSR A = csr_from_triplets(n, dirichlet_matrix_triplets_2d(n, trip, sets[s]));
        EXPECT_TRUE(A.is_symmetric(1e-12));
        const std::vector<double> diagonal = A.diag();
        ASSERT_EQ(diagonal.size(), n);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_GT(diagonal[i], 0.0) << "node " << i;
        }
    }

    const Mesh2D two = rectangle_mesh(2.0, 1.0, 2, 1);
    const std::vector<Triplet> t2 = stiffness_triplets_2d(two, 2.0);
    const CSR in_order = csr_from_triplets(6, dirichlet_matrix_triplets_2d(6, t2, {0, 2, 3, 5}));
    const CSR shuffled = csr_from_triplets(6, dirichlet_matrix_triplets_2d(6, t2, {5, 0, 3, 2}));
    expect_matrix(shuffled, to_dense(in_order), in_order.get_val_size());
}

// EXAMPLE 61. A mesh of 150 x 150 cells (22801 nodes, 405000 triplets) with one node in three fixed (7601 nodes).
// A search in the list of fixed nodes for each triplet (405000 x 7601 comparisons) gives the right result but
// takes tens of seconds; with a table with a label per node ("fixed yes/no"), read in a single pass,
// a few hundredths of a second are enough. The test measures the time of the function alone and demands less than 2
// seconds: a hundred times the right time, and much less than the search for each triplet. Then it checks the result
// with the matrix-vector product alone: for every vector u, A u is
//   - on the rows of the free nodes: (K u0)[i], where u0 is u with the elements of the fixed nodes set to 0
//     (the row of A has zero on the fixed columns);
//   - on the rows of the fixed nodes: u[d] (row of the identity).
TEST(Dirichlet2d, LargeMeshWithManyFixedNodes) {
    const std::size_t cells = 150;
    const Mesh2D m = rectangle_mesh(1.0, 1.0, cells, cells);
    const std::size_t n = m.num_nodes();
    const std::vector<Triplet> trip = stiffness_triplets_2d(m, 1.0);
    ASSERT_EQ(n, 22801u);
    ASSERT_EQ(trip.size(), 405000u);

    Nodes fixed;
    std::vector<bool> is_fixed(n, false);
    for (std::size_t i = 0; i < n; i += 3) {
        fixed.push_back(i);
        is_fixed[i] = true;
    }
    ASSERT_EQ(fixed.size(), 7601u);

    const auto start = std::chrono::steady_clock::now();
    const std::vector<Triplet> out = dirichlet_matrix_triplets_2d(n, trip, fixed);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    EXPECT_LT(seconds, 2.0) << "dirichlet_matrix_triplets_2d took " << seconds
                            << " s: it must read each triplet only once, without searching the list of fixed nodes";
    const CSR A = csr_from_triplets(n, out);
    const CSR K = csr_from_triplets(n, trip);

    std::vector<double> u(n);
    std::vector<double> u0(n);
    for (std::size_t i = 0; i < n; ++i) {
        u[i] = 1.0 + 0.001 * static_cast<double>(i % 97);
        u0[i] = is_fixed[i] ? 0.0 : u[i];
    }
    const std::vector<double> got = A * u;
    const std::vector<double> Ku0 = K * u0;
    ASSERT_EQ(got.size(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(got[i], is_fixed[i] ? u[i] : Ku0[i], 1e-9) << "row " << i;
    }
}

// ---- Imposed temperatures (Dirichlet): the right-hand side ------------------------------------------------
//
// The system with the imposed temperatures is A T = b. The matrix A is that of the previous step; the right-hand side b has
// one row per node:
//   - fixed node d: b[d] = g[d], the imposed value (the row of A is that of the identity: T[d] = g[d]);
//   - free node i:  b[i] = f[i] - (K_FD g)[i] = f[i] - sum, over the fixed nodes d, of K(i, d) * g[d]
//     (the contribution of the fixed nodes, which is known, goes to the right).
// A double loop (for each free node, for each fixed node, K.get(i, d)) costs "free nodes x fixed nodes"
// reads and each one scans a row: too much. ONE matrix-vector product is enough. Let g^ be the vector of n numbers that
// is g[d] on the fixed nodes and 0 on the others. For EVERY row i
//     (K g^)[i] = sum over all the nodes j of K(i, j) g^[j] = sum over the fixed nodes d of K(i, d) g[d]
// and on the rows of the free nodes it is exactly (K_FD g)[i]. So
//     b = f - K g^        and then        b[d] = g[d]  for every fixed node d
// (on the rows of the fixed nodes the product is not needed: the second step rewrites them). K is the ORIGINAL matrix, not
// the A of the previous step: A has already zeroed the piece K_FD.
// dirichlet_rhs_2d(K, f, fixed_nodes, fixed_values) returns b; fixed_values[q] is the value imposed on node
// fixed_nodes[q]. It throws std::invalid_argument if K.size() != f.size(), if fixed_values has a number of elements
// different from fixed_nodes, if a fixed node is >= n or if a node appears twice in fixed_nodes (even with the
// same value).

namespace {

// Compares two vectors element by element, with the absolute tolerance tol.
void expect_vector(const DVector& got, const DVector& want, double tol, const std::string& what = "") {
    ASSERT_EQ(got.size(), want.size()) << what << ": size";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tol) << what << ": element " << i;
    }
}

// The "slow computation" of the right-hand side, written as the definition says: for each free node i, f[i] minus the sum
// over ALL the fixed nodes of the products K(i, d) * g[d]; for each fixed node, its value.
DVector slow_rhs(const Dense& K, const DVector& f, const Nodes& fixed, const DVector& values) {
    std::vector<bool> is_fixed(f.size(), false);
    for (std::size_t d : fixed) {
        is_fixed[d] = true;
    }
    DVector b = f;
    for (std::size_t i = 0; i < f.size(); ++i) {
        if (!is_fixed[i]) {
            for (std::size_t q = 0; q < fixed.size(); ++q) {
                b[i] -= K[i][fixed[q]] * values[q];
            }
        }
    }
    for (std::size_t q = 0; q < fixed.size(); ++q) {
        b[fixed[q]] = values[q];
    }
    return b;
}

// The temperature T (one per node) is the solution of the problem -k Laplacian(T) = q on the mesh m, with the
// temperature imposed on the nodes `fixed` (equal to T at those nodes). If so, A T - b = 0 in every row: in the
// rows of the free nodes it is the equation K T = f, in those of the fixed ones it is T[d] = g[d].
void expect_solution_solves_the_system(const Mesh2D& m, double k, double q, const Nodes& fixed, const DVector& T) {
    const std::size_t n = m.num_nodes();
    DVector values;
    for (std::size_t d : fixed) {
        values.push_back(T[d]);
    }
    const DVector b = dirichlet_rhs_2d(assemble_stiffness_2d(m, k), assemble_load_2d(m, q), fixed, values);
    const CSR A = csr_from_triplets(n, dirichlet_matrix_triplets_2d(n, stiffness_triplets_2d(m, k), fixed));
    const DVector AT = A * T;
    ASSERT_EQ(b.size(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(AT[i], b[i], 1e-9) << "row " << i;
    }
}

}  // namespace

// EXAMPLE 63. Two examples by hand.
//
// (a) A bar of 4 nodes, three equal elements with conductance 1 (the matrix with no constraints):
//         [  1 -1  0  0 ]
//     K = [ -1  2 -1  0 ]       f = (7, 1, 1, 9)       node 0 fixed at 10, node 3 fixed at 40
//         [  0 -1  2 -1 ]
//         [  0  0 -1  1 ]
//   g^ = (10, 0, 0, 40) and K g^ = (10, -10, -40, 40). At the free nodes b = f - K g^: b1 = 1 + 10 = 11 and
//   b2 = 1 + 40 = 41. At the fixed nodes b is the imposed value (the loads 7 and 9 do not matter):
//     b = (10, 11, 41, 40).
//
// (b) The two square cells of the previous examples (k = 2) with the source q = 6, which gives the load
//     f = (2, 3, 1, 1, 3, 2), the left side (nodes 0 and 3) at 10 and the right side (nodes 2 and 5) at 30. The free nodes
//     are 1 and 4. Row 1 of K is (-1, 4, -1, 0, -2, 0): node 1 is connected to the fixed nodes 0 and 2, with -1 and -1, and
//     not to the fixed nodes 3 and 5. Row 4 is (0, -2, 0, -1, 4, -1): it is connected to the fixed nodes 3 and 5. With
//     g^ = (10, 0, 30, 10, 0, 30):
//       (K g^)[1] = -1 * 10 + -1 * 30 = -40         b1 = 3 - (-40) = 43
//       (K g^)[4] = -1 * 10 + -1 * 30 = -40         b4 = 3 - (-40) = 43
//     and b = (10, 43, 30, 10, 43, 30). With the A of the previous step the system has the solution
//     T = (10, 21.5, 30, 10, 21.5, 30): at node 1, 4 * 21.5 - 2 * 21.5 = 43.
TEST(Dirichlet2d, RightHandSideMatchesTwoHandComputedExamples) {
    const CSR bar(Dense{{1, -1, 0, 0}, {-1, 2, -1, 0}, {0, -1, 2, -1}, {0, 0, -1, 1}});
    const DVector load = {7.0, 1.0, 1.0, 9.0};
    expect_vector(dirichlet_rhs_2d(bar, load, {0, 3}, {10.0, 40.0}), {10.0, 11.0, 41.0, 40.0}, 1e-12, "bar");

    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const CSR K = assemble_stiffness_2d(m, 2.0);
    const DVector f = assemble_load_2d(m, 6.0);
    expect_vector(f, {2.0, 3.0, 1.0, 1.0, 3.0, 2.0}, 1e-12, "load");

    const Nodes fixed = {0, 2, 3, 5};
    const DVector b = dirichlet_rhs_2d(K, f, fixed, {10.0, 30.0, 10.0, 30.0});
    expect_vector(b, {10.0, 43.0, 30.0, 10.0, 43.0, 30.0}, 1e-12, "two cells");

    const CSR A = csr_from_triplets(6, dirichlet_matrix_triplets_2d(6, stiffness_triplets_2d(m, 2.0), fixed));
    expect_vector(A * DVector{10.0, 21.5, 30.0, 10.0, 21.5, 30.0}, b, 1e-12, "A times the solution");
}

// EXAMPLE 64. The extreme cases on the two cells of EXAMPLE 63 (f = (2, 3, 1, 1, 3, 2)).
//   - No fixed node: b = f.
//   - All the nodes fixed, listed in shuffled order: b[node] = the value of that node (the value is at the SAME
//     position of the list as the node, not at the position the node has in the mesh).
//   - Only one node fixed, with the value 0: g^ = 0, so b = f except at the fixed node, where it is 0.
//   - The load of the fixed nodes does not matter: two vectors f that differ only at the fixed nodes give the same b.
TEST(Dirichlet2d, RightHandSideWithNoFixedNodesIsTheLoadAndWithAllFixedIsTheValues) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const CSR K = assemble_stiffness_2d(m, 2.0);
    const DVector f = assemble_load_2d(m, 6.0);

    expect_vector(dirichlet_rhs_2d(K, f, {}, {}), f, 1e-12, "no fixed node");

    const DVector all = dirichlet_rhs_2d(K, f, {4, 0, 5, 2, 1, 3}, {40.0, 0.5, 50.0, 20.0, 10.0, 30.0});
    expect_vector(all, {0.5, 10.0, 20.0, 30.0, 40.0, 50.0}, 1e-12, "all the nodes fixed");

    expect_vector(dirichlet_rhs_2d(K, f, {1}, {0.0}), {2.0, 0.0, 1.0, 1.0, 3.0, 2.0}, 1e-12, "node 1 at zero");

    DVector other_load = f;
    other_load[0] += 1000.0;
    other_load[5] -= 77.0;
    const Nodes fixed = {0, 2, 3, 5};
    const DVector values = {10.0, 30.0, 10.0, 30.0};
    expect_vector(dirichlet_rhs_2d(K, other_load, fixed, values), dirichlet_rhs_2d(K, f, fixed, values), 1e-12,
                  "different load at the fixed nodes");
}

// EXAMPLE 65. The result on meshes of different shapes and for different sets of nodes (the same as in EXAMPLE 57:
// node 0 alone, the last node alone, the whole boundary, every other node, all the nodes), in the order of the mesh
// and also in reverse order, with a load and imposed values that are all different from one another. The comparison
// is with the "slow computation": the sum written as the definition says, one free node at a time.
// At the end, a NON-symmetric matrix (it is not a K, but the formula does not require it): b uses K(i, d), with the row i
// free and the column d fixed, not K(d, i). With
//         [  4  1 -2  0  3 ]
//         [  0  5  1 -1  2 ]
//     G = [  2 -3  6  1  0 ]     f = (1, 2, 3, 4, 5)     node 3 at 10 and node 0 at 20
//         [  1  0 -1  7  2 ]
//         [ -2  4  0  1  8 ]
// g^ = (20, 0, 0, 10, 0), G g^ = (80, -10, 50, 90, -30), so b = (20, 2 + 10, 3 - 50, 10, 5 + 30) =
// (20, 12, -47, 10, 35). With the transposed matrix node 1 would give 2 - 20 = -18.
TEST(Dirichlet2d, RightHandSideMatchesTheSlowSumOnEveryKindOfMesh) {
    std::vector<Mesh2D> meshes;
    meshes.push_back(rectangle_mesh(3.0, 1.5, 4, 3));
    meshes.push_back(triangle_with_interior_node());
    meshes.push_back(irregular_quadrilateral());
    meshes.push_back(mesh_with_nodes_outside_every_triangle());
    meshes.push_back(rectangle_mesh(1.0, 1.0, 1, 1));

    for (std::size_t index = 0; index < meshes.size(); ++index) {
        const Mesh2D& m = meshes[index];
        const std::size_t n = m.num_nodes();
        const CSR K = assemble_stiffness_2d(m, 3.5);
        const Dense dense = to_dense(K);
        DVector f(n);
        for (std::size_t i = 0; i < n; ++i) {
            f[i] = 2.0 + 0.3 * static_cast<double>(i % 7) + 0.01 * static_cast<double>(i);
        }

        Nodes first;
        first.push_back(0);
        Nodes last;
        last.push_back(n - 1);
        Nodes every_second;
        for (std::size_t i = 0; i < n; i += 2) {
            every_second.push_back(i);
        }
        Nodes everything;
        for (std::size_t i = 0; i < n; ++i) {
            everything.push_back(i);
        }
        std::vector<Nodes> sets;
        sets.push_back(first);
        sets.push_back(last);
        sets.push_back(boundary_nodes(m));
        sets.push_back(every_second);
        sets.push_back(everything);

        for (std::size_t s = 0; s < sets.size(); ++s) {
            for (int reversed = 0; reversed < 2; ++reversed) {
                SCOPED_TRACE("mesh number " + std::to_string(index) + ", node set number " +
                             std::to_string(s) + (reversed ? ", in reverse order" : ", in order"));
                Nodes nodes = sets[s];
                if (reversed) {
                    std::reverse(nodes.begin(), nodes.end());
                }
                DVector values;
                for (std::size_t q = 0; q < nodes.size(); ++q) {
                    values.push_back(100.0 + 7.0 * static_cast<double>(q) - 0.37 * static_cast<double>(nodes[q]));
                }
                expect_vector(dirichlet_rhs_2d(K, f, nodes, values), slow_rhs(dense, f, nodes, values), 1e-9);
            }
        }
    }

    const Dense general = {{4, 1, -2, 0, 3}, {0, 5, 1, -1, 2}, {2, -3, 6, 1, 0}, {1, 0, -1, 7, 2}, {-2, 4, 0, 1, 8}};
    const DVector load = {1.0, 2.0, 3.0, 4.0, 5.0};
    expect_vector(dirichlet_rhs_2d(CSR(general), load, {3, 0}, {10.0, 20.0}), {20.0, 12.0, -47.0, 10.0, 35.0}, 1e-12,
                  "non-symmetric matrix");
}

// EXAMPLE 66. Two problems whose exact solution is known, on rectangles of different shapes (including non-square
// cells). k = 2.5 and the source is q = 120.
//   (a) A plate of length L between x = 0 and x = L, with the temperature imposed on the left side (20) and the right side
//       (80); the bottom side and the top one are free (no flux). The problem -k T'' = q has the solution
//           T(x) = T_left + (T_right - T_left) x / L + q x (L - x) / (2 k).
//   (b) The whole boundary fixed at the temperature of
//           T(x, y) = 50 - q (x^2 + y^2) / (4 k),
//       for which -k (T_xx + T_yy) = -k (-q / (2k) - q / (2k)) = q.
// On these meshes K coincides with the five-point finite-difference scheme (the two triplets of K on the
// oblique side of each triangle are 0), which is exact for polynomials of second degree: the exact solution,
// computed at the nodes, solves the discrete system, and therefore A T = b in every row (with T equal to the exact
// solution). A wrong right-hand side, at a single node, breaks the equality.
TEST(Dirichlet2d, TheExactSolutionOfTwoSimpleProblemsSolvesTheSystemWithImposedTemperatures) {
    struct Grid {
        double lx;
        double ly;
        std::size_t nx;
        std::size_t ny;
    };
    const Grid grids[] = {{1.0, 1.0, 2, 2}, {3.0, 1.5, 4, 3}, {2.0, 0.5, 8, 2}, {1.0, 2.0, 5, 7}, {4.0, 1.0, 6, 1}};
    const double k = 2.5;
    const double q = 120.0;

    for (const Grid& g : grids) {
        SCOPED_TRACE(std::to_string(g.nx) + " x " + std::to_string(g.ny) + " cells, " + std::to_string(g.lx) + " x " +
                     std::to_string(g.ly));
        const Mesh2D m = rectangle_mesh(g.lx, g.ly, g.nx, g.ny);
        const std::size_t n = m.num_nodes();

        Nodes sides;
        DVector slab(n);
        DVector paraboloid(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double x = m.x(i);
            const double y = m.y(i);
            slab[i] = 20.0 + (80.0 - 20.0) * x / g.lx + q * x * (g.lx - x) / (2.0 * k);
            paraboloid[i] = 50.0 - q * (x * x + y * y) / (4.0 * k);
            if (x < 1e-12 || x > g.lx - 1e-12) {
                sides.push_back(i);
            }
        }
        {
            SCOPED_TRACE("plate");
            expect_solution_solves_the_system(m, k, q, sides, slab);
        }
        {
            SCOPED_TRACE("the whole boundary");
            expect_solution_solves_the_system(m, k, q, boundary_nodes(m), paraboloid);
        }
    }
}

// EXAMPLE 62. There can be few triplets, even none, and far fewer than the nodes (an n x n matrix that is almost
// empty). The table of fixed nodes has one label per NODE: it has n elements, not as many as the triplets.
// With n = 5:
//   - no triplets and no fixed node: the list stays empty;
//   - no triplets and node 1 fixed: only (1, 1, 1) is left;
//   - the two triplets (0, 0, 2) and (1, 1, 3) with nodes 1 and 4 fixed: (1, 1, 3) is dropped and (0, 0, 2), (1, 1, 1)
//     and (4, 4, 1) are left;
//   - no triplets and every node fixed: the 5 triplets (d, d, 1) are left.
// A table with fewer than n elements (for example as many as the triplets) does not exist at all when the list is
// empty, and reading it crashes the program.
TEST(Dirichlet2d, FewOrNoTripletsAreFine) {
    const std::vector<Triplet> none;
    const std::vector<Triplet> out0 = dirichlet_matrix_triplets_2d(5, none, {});
    EXPECT_EQ(out0.size(), 0u);

    const std::vector<Triplet> out1 = dirichlet_matrix_triplets_2d(5, none, {1});
    ASSERT_EQ(out1.size(), 1u);
    expect_values_at(out1, 1, 1, {1});

    const std::vector<Triplet> two = {{0, 0, 2.0}, {1, 1, 3.0}};
    const std::vector<Triplet> out2 = dirichlet_matrix_triplets_2d(5, two, {1, 4});
    ASSERT_EQ(out2.size(), 3u);
    expect_values_at(out2, 0, 0, {2});
    expect_values_at(out2, 1, 1, {1});
    expect_values_at(out2, 4, 4, {1});

    const std::vector<Triplet> out3 = dirichlet_matrix_triplets_2d(5, none, {0, 1, 2, 3, 4});
    ASSERT_EQ(out3.size(), 5u);
    for (std::size_t d = 0; d < 5; ++d) {
        expect_values_at(out3, d, d, {1});
    }
}

// EXAMPLE 67. The errors. With K of 3 x 3 and f of 3 numbers:
//   - fixed_values with a number of elements different from fixed_nodes, in both directions (also with an
//     empty list);
//   - K and f of different sizes (f of 2 or of 4 numbers);
//   - a fixed node equal to n = 3 (the last one is 2) or larger, even huge (a billion);
//   - a repeated node, with the same value or with different values, next to its twin or far from it;
// throw std::invalid_argument. The empty list, node n - 1 and all the nodes in shuffled order are fine instead.
// As for the matrix, the check of an index must come BEFORE using it to read or write: an index of
// a billion used first makes the program crash instead of throwing the exception.
TEST(Dirichlet2d, RightHandSideThrowsInvalidArgumentOnBadInput) {
    const CSR K(Dense{{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}});
    const DVector f = {1.0, 2.0, 3.0};

    EXPECT_NO_THROW(dirichlet_rhs_2d(K, f, {}, {}));
    EXPECT_NO_THROW(dirichlet_rhs_2d(K, f, {2}, {7.0}));
    EXPECT_NO_THROW(dirichlet_rhs_2d(K, f, {2, 0, 1}, {1.0, 2.0, 3.0}));

    EXPECT_THROW(dirichlet_rhs_2d(K, f, {0, 1}, {1.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {0}, {1.0, 2.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {}, {1.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {0}, {}), std::invalid_argument);

    EXPECT_THROW(dirichlet_rhs_2d(K, DVector{1.0, 2.0}, {}, {}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, DVector{1.0, 2.0, 3.0, 4.0}, {}, {}), std::invalid_argument);

    EXPECT_THROW(dirichlet_rhs_2d(K, f, {3}, {1.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {1000}, {1.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {1000000000}, {1.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {0, 7}, {1.0, 2.0}), std::invalid_argument);

    EXPECT_THROW(dirichlet_rhs_2d(K, f, {1, 1}, {5.0, 5.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {1, 1}, {5.0, 6.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {0, 2, 1, 2}, {1.0, 2.0, 3.0, 4.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {2, 0, 1, 2}, {9.0, 2.0, 3.0, 9.0}), std::invalid_argument);
}

// EXAMPLE 68. A mesh of 200 x 200 cells (40401 nodes) with one node in two fixed (20201 nodes).
// A double loop with K.get(i, d) for each free node i and each fixed node d (20200 x 20201 reads, each with
// the scan of a row) gives the right result but takes tens of seconds, and so does a matrix-vector
// product for each fixed node; with a single matrix-vector product a few thousandths of a second are enough.
// The test measures the time of the function alone and demands less than 1 second.
// Then it checks the result without redoing the computation: for every vector T that has the imposed value
// at the fixed nodes,
//   - on the rows of the free nodes:  (A T - b)[i] = (K T - f)[i]    (K T = K_FF T_F + K_FD g, A T = K_FF T_F);
//   - on the rows of the fixed nodes: (A T - b)[d] = T[d] - g[d] = 0.
TEST(Dirichlet2d, RightHandSideOnALargeMeshWithManyFixedNodes) {
    const std::size_t cells = 200;
    const Mesh2D m = rectangle_mesh(1.0, 1.0, cells, cells);
    const std::size_t n = m.num_nodes();
    ASSERT_EQ(n, 40401u);
    const CSR K = assemble_stiffness_2d(m, 1.0);
    const DVector f = assemble_load_2d(m, 1.0);

    Nodes fixed;
    DVector values;
    std::vector<bool> is_fixed(n, false);
    for (std::size_t i = 0; i < n; i += 2) {
        fixed.push_back(i);
        values.push_back(1.0 + 0.01 * static_cast<double>(i % 89));
        is_fixed[i] = true;
    }
    ASSERT_EQ(fixed.size(), 20201u);

    const auto start = std::chrono::steady_clock::now();
    const DVector b = dirichlet_rhs_2d(K, f, fixed, values);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    EXPECT_LT(seconds, 1.0) << "dirichlet_rhs_2d took " << seconds
                            << " s: it must do a single matrix-vector product, without reading K one element at a time";
    ASSERT_EQ(b.size(), n);

    const CSR A = csr_from_triplets(n, dirichlet_matrix_triplets_2d(n, stiffness_triplets_2d(m, 1.0), fixed));
    DVector T(n);
    for (std::size_t i = 0; i < n; ++i) {
        T[i] = 1.0 + 0.001 * static_cast<double>(i % 97);
    }
    for (std::size_t q = 0; q < fixed.size(); ++q) {
        T[fixed[q]] = values[q];
    }
    const DVector AT = A * T;
    const DVector KT = K * T;
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(AT[i] - b[i], is_fixed[i] ? 0.0 : KT[i] - f[i], 1e-9) << "row " << i;
    }
}

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "CSR_and_Vectors.h"
#include "Triplets.h"

// ---- From triplets (row, column, value) to the CSR matrix: csr_from_triplets ---------------------
//
// The assembly of a finite-element matrix produces a list of CONTRIBUTIONS: each element
// (each triangle) says "add this number at position (row, column)". Two elements that
// share nodes contribute to the same positions, so in the list the same (row, column) pair
// appears several times, and the values must be SUMMED. The list is in arbitrary order.
//
// csr_from_triplets(n, list) builds the n x n matrix that has at (r, c) the sum of the values of
// all the triplets in the list with that row and that column:
//   - triplets with the same position are summed, whatever the order of the list;
//   - a position that appears in the list is stored even if the sum is 0 (the sparsity pattern
//     depends only on WHERE the contributions are, not on their values);
//   - positions that do not appear are 0 and take no memory;
//   - a triplet outside the matrix (row or column >= n) and a matrix with no rows (n = 0) are
//     errors: std::invalid_argument, as in the other constructors of the project.
// The CSR format is the one already used by the project: for each row, in order, its values and their
// columns; row r occupies the slots from row_ptr[r] to row_ptr[r+1] - 1.

namespace {

using Dense = std::vector<std::vector<double>>;

// Check that A is the matrix `want`, element by element, and that it occupies exactly nnz slots.
void expect_matrix(const CSR& A, const Dense& want, std::size_t nnz, const std::string& what = "") {
    ASSERT_EQ(A.size(), want.size()) << what << " size";
    EXPECT_EQ(A.get_val_size(), nnz) << what << " occupied slots";
    for (std::size_t i = 0; i < want.size(); ++i) {
        for (std::size_t j = 0; j < want.size(); ++j) {
            EXPECT_DOUBLE_EQ(A.get(i, j), want[i][j]) << what << " element (" << i << ", " << j << ")";
        }
    }
}

}  // namespace

// EXAMPLE 29. A 3 x 3 matrix by hand, with the list out of order and one duplicate:
//     (2, 2, 1)   (0, 0, 2)   (0, 2, 3)   (0, 0, 5)   (1, 1, 4)   (2, 0, 6)
// The position (0, 0) appears twice: 2 + 5 = 7. The matrix is
//         [ 7  0  3 ]
//         [ 0  4  0 ]          5 occupied slots
//         [ 6  0  1 ]
// In CSR format, row by row with the columns in order:
//     values = [7, 3, 4, 6, 1]     col_indx = [0, 2, 1, 0, 2]     row_ptr = [0, 2, 3, 5]
// (two slots in row 0, one in row 1, two in row 2: row_ptr is obtained by adding up these counts
// step by step, starting from 0).
TEST(Triplets, HandExampleWithUnsortedInputAndADuplicate) {
    const std::vector<Triplet> list = {{2, 2, 1.0}, {0, 0, 2.0}, {0, 2, 3.0},
                                       {0, 0, 5.0}, {1, 1, 4.0}, {2, 0, 6.0}};
    const CSR A = csr_from_triplets(3, list);

    expect_matrix(A, {{7.0, 0.0, 3.0}, {0.0, 4.0, 0.0}, {6.0, 0.0, 1.0}}, 5);

    // The product with x = (1, 2, 3):  (7 + 9, 8, 6 + 3) = (16, 8, 9).
    const std::vector<double> y = A * std::vector<double>{1.0, 2.0, 3.0};
    ASSERT_EQ(y.size(), 3u);
    EXPECT_DOUBLE_EQ(y[0], 16.0);
    EXPECT_DOUBLE_EQ(y[1], 8.0);
    EXPECT_DOUBLE_EQ(y[2], 9.0);

    // The diagonal is (7, 4, 1).
    const std::vector<double> d = A.diag();
    ASSERT_EQ(d.size(), 3u);
    EXPECT_DOUBLE_EQ(d[0], 7.0);
    EXPECT_DOUBLE_EQ(d[1], 4.0);
    EXPECT_DOUBLE_EQ(d[2], 1.0);
}

// EXAMPLE 30. Four contributions to the same position (1, 2), mixed with others: 4 + 1 + 8 + 2 = 15.
// Then two opposite contributions, +2 and -2, at position (0, 1): the sum is 0 but the position stays
// stored. Finally a contribution that is already 0 by itself, at (2, 0): it also stays stored
// (the matrix of a right triangle has terms that are exactly 0, and their
// position counts all the same). In total 3 occupied slots, two of which contain 0.
TEST(Triplets, DuplicatesAreSummedWhateverTheirOrderAndZerosAreKept) {
    const CSR A = csr_from_triplets(
        3, {{1, 2, 4.0}, {0, 1, 2.0}, {1, 2, 1.0}, {2, 0, 0.0}, {1, 2, 8.0}, {0, 1, -2.0}, {1, 2, 2.0}});
    expect_matrix(A, {{0.0, 0.0, 0.0}, {0.0, 0.0, 15.0}, {0.0, 0.0, 0.0}}, 3);
}

// EXAMPLE 31. Same column but different rows: they are different positions and are NOT summed, even if in the
// sorted list (by row, then by column) they sit next to each other. A 3 x 3 matrix with
// contributions only in column 1, and with non-integer values (so it is clear that the numbers stay double):
//     (2, 1, -2.5)   (0, 1, 5.5)   (1, 1, 7.25)   (1, 1, 3.125)
// The position (1, 1) receives two contributions: 7.25 + 3.125 = 10.375. The matrix is
//         [ 0   5.5     0 ]
//         [ 0   10.375  0 ]          3 occupied slots
//         [ 0  -2.5     0 ]
// (these are fractions with denominator 2, 4 or 8: exact in binary, so the comparison needs no tolerance).
TEST(Triplets, SameColumnInDifferentRowsIsNotMerged) {
    const CSR A = csr_from_triplets(3, {{2, 1, -2.5}, {0, 1, 5.5}, {1, 1, 7.25}, {1, 1, 3.125}});
    expect_matrix(A, {{0.0, 5.5, 0.0}, {0.0, 10.375, 0.0}, {0.0, -2.5, 0.0}}, 3);

    // The product with x = (1, 2, 3) uses only column 1:  2 * (5.5, 10.375, -2.5) = (11, 20.75, -5).
    const std::vector<double> y = A * std::vector<double>{1.0, 2.0, 3.0};
    ASSERT_EQ(y.size(), 3u);
    EXPECT_DOUBLE_EQ(y[0], 11.0);
    EXPECT_DOUBLE_EQ(y[1], 20.75);
    EXPECT_DOUBLE_EQ(y[2], -5.0);
}

// EXAMPLE 32. Rows without contributions. A 4 x 4 matrix with contributions only in rows 0 and 3: rows
// 1 and 2 are empty, in the middle (here row_ptr = [0, 2, 2, 2, 3]: an empty row occupies no slots, so
// row_ptr does not change). One with a single contribution in row 0: the last three rows are empty, at
// the end (row_ptr = [0, 1, 1, 1, 1]). And one with no contribution at all: the zero matrix.
TEST(Triplets, EmptyRowsAndEmptyList) {
    const Dense zeros4(4, std::vector<double>(4, 0.0));

    const CSR A = csr_from_triplets(4, {{3, 3, 1.0}, {0, 1, 2.0}, {0, 0, 3.0}});
    Dense wantA = zeros4;
    wantA[0][0] = 3.0;
    wantA[0][1] = 2.0;
    wantA[3][3] = 1.0;
    expect_matrix(A, wantA, 3, "empty rows in the middle:");

    const CSR B = csr_from_triplets(4, {{0, 3, 5.0}});
    Dense wantB = zeros4;
    wantB[0][3] = 5.0;
    expect_matrix(B, wantB, 1, "empty rows at the end:");

    const CSR Z = csr_from_triplets(3, {});
    expect_matrix(Z, Dense(3, std::vector<double>(3, 0.0)), 0, "empty list:");
    const std::vector<double> y = Z * std::vector<double>{1.0, 2.0, 3.0};
    ASSERT_EQ(y.size(), 3u);
    EXPECT_DOUBLE_EQ(y[0], 0.0);
    EXPECT_DOUBLE_EQ(y[1], 0.0);
    EXPECT_DOUBLE_EQ(y[2], 0.0);
}

// EXAMPLE 33. The order of the list does not matter. A list of 12 triplets on a 4 x 4 matrix, with 5
// doubled positions:
//     (0,0): 1 + 4 = 5     (0,1): -2 + 2 = 0     (1,0): 3 - 3 = 0     (1,1): 6
//     (2,3): 5 + 2 = 7     (3,2): 7              (3,3): -1 + 1 = 0
// i.e. the matrix [[5,0,0,0], [0,6,0,0], [0,0,0,7], [0,0,7,0]] with 7 occupied slots (three are stored
// zeros). The same list read backwards, and shuffled 20 times (always with the same seed,
// so the test is repeatable), always gives the same matrix. The values are integers, so every summation
// order gives the same exact number.
TEST(Triplets, InputOrderDoesNotMatter) {
    std::vector<Triplet> list = {{0, 0, 1.0}, {0, 1, -2.0}, {1, 0, 3.0}, {0, 0, 4.0}, {2, 3, 5.0}, {3, 3, -1.0},
                                 {1, 0, -3.0}, {2, 3, 2.0}, {3, 2, 7.0}, {0, 1, 2.0}, {1, 1, 6.0}, {3, 3, 1.0}};
    const Dense want = {{5.0, 0.0, 0.0, 0.0}, {0.0, 6.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 7.0}, {0.0, 0.0, 7.0, 0.0}};

    expect_matrix(csr_from_triplets(4, list), want, 7, "given order:");

    std::reverse(list.begin(), list.end());
    expect_matrix(csr_from_triplets(4, list), want, 7, "reversed:");

    std::mt19937 gen(2024);
    for (int k = 0; k < 20; ++k) {
        std::shuffle(list.begin(), list.end(), gen);
        expect_matrix(csr_from_triplets(4, list), want, 7, "shuffled " + std::to_string(k) + ":");
    }
}

// EXAMPLE 34. A chain of 3 bars on 4 nodes (0-1, 1-2, 2-3): the familiar 1D assembly.
// Each bar a-b contributes the matrix [[1, -1], [-1, 1]], i.e. 4 triplets:
// (a,a,1) (a,b,-1) (b,a,-1) (b,b,1). Nodes 1 and 2 belong to two bars, so their
// diagonals receive two contributions: 1 + 1 = 2. In total 3 * 4 = 12 triplets, 2 of them duplicated, i.e.
// 10 occupied slots:
//         [  1  -1   0   0 ]
//         [ -1   2  -1   0 ]
//         [  0  -1   2  -1 ]
//         [  0   0  -1   1 ]
// The matrix is symmetric and every row sums to 0 (a uniform temperature lets no heat through).
TEST(Triplets, AssemblingAChainOfBarsGivesTheTridiagonalMatrix) {
    std::vector<Triplet> list;
    for (std::size_t a = 0; a < 3; ++a) {
        const std::size_t b = a + 1;
        list.push_back({a, a, 1.0});
        list.push_back({a, b, -1.0});
        list.push_back({b, a, -1.0});
        list.push_back({b, b, 1.0});
    }
    ASSERT_EQ(list.size(), 12u);

    const CSR A = csr_from_triplets(4, list);
    expect_matrix(A,
                  {{1.0, -1.0, 0.0, 0.0}, {-1.0, 2.0, -1.0, 0.0}, {0.0, -1.0, 2.0, -1.0}, {0.0, 0.0, -1.0, 1.0}},
                  10);
    EXPECT_TRUE(A.is_symmetric(1e-14));

    const std::vector<double> y = A * std::vector<double>(4, 1.0);
    ASSERT_EQ(y.size(), 4u);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(y[i], 0.0) << "row " << i;
    }
}

// EXAMPLE 35. A "random" test against the direct computation. 1500 contributions on a 20 x 20 matrix (400
// positions, so almost every position receives several contributions, and with integer values from -3 to 3 many
// sums are 0). For each position, the expected value is the direct sum of the contributions; the occupied
// slots are as many as the distinct positions that appear in the list. The seed is fixed.
TEST(Triplets, MatchesDirectAccumulationOnRandomTriplets) {
    const std::size_t n = 20;
    std::mt19937 gen(12345);
    std::uniform_int_distribution<std::size_t> index(0, n - 1);
    std::uniform_int_distribution<int> value(-3, 3);

    std::vector<Triplet> list;
    Dense want(n, std::vector<double>(n, 0.0));
    std::set<std::pair<std::size_t, std::size_t>> used;
    for (int k = 0; k < 1500; ++k) {
        const std::size_t r = index(gen);
        const std::size_t c = index(gen);
        const double v = value(gen);
        list.push_back({r, c, v});
        want[r][c] += v;
        used.insert({r, c});
    }

    const CSR A = csr_from_triplets(n, list);
    expect_matrix(A, want, used.size());

    // Also the product with a vector (integers: the sum is exact in any order).
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) - 7.0;
    }
    const std::vector<double> y = A * x;
    ASSERT_EQ(y.size(), n);
    for (std::size_t i = 0; i < n; ++i) {
        double expected = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            expected += want[i][j] * x[j];
        }
        EXPECT_DOUBLE_EQ(y[i], expected) << "row " << i;
    }
}

// EXAMPLE 36. What is not allowed: a matrix with no rows (n = 0) and a triplet with row or column >= n,
// alone or among other valid ones (also as the first or as the last of the list). The valid positions
// go from 0 to n - 1, the last one included.
TEST(Triplets, RejectsNonsense) {
    EXPECT_THROW(csr_from_triplets(0, {}), std::invalid_argument);
    EXPECT_THROW(csr_from_triplets(3, {{3, 0, 1.0}}), std::invalid_argument);                  // row = n
    EXPECT_THROW(csr_from_triplets(3, {{0, 3, 1.0}}), std::invalid_argument);                  // column = n
    EXPECT_THROW(csr_from_triplets(3, {{0, 0, 1.0}, {1000, 1, 1.0}}), std::invalid_argument);  // very large row, last
    EXPECT_THROW(csr_from_triplets(3, {{1, 1, 1.0}, {2, 1000, 1.0}}), std::invalid_argument);  // very large column, last
    EXPECT_THROW(csr_from_triplets(3, {{1000, 1, 1.0}, {0, 0, 1.0}}), std::invalid_argument);  // very large row, first
    EXPECT_THROW(csr_from_triplets(3, {{2, 1000, 1.0}, {1, 1, 1.0}}), std::invalid_argument);  // very large column, first
    EXPECT_THROW(csr_from_triplets(3, {{0, 0, 1.0}, {1, 3, 1.0}, {2, 2, 1.0}}), std::invalid_argument);  // in mezzo
    EXPECT_NO_THROW(csr_from_triplets(3, {{2, 2, 1.0}}));                                      // the last valid position
}

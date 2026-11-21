#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "Assembly2D.h"
#include "CSR_and_Vectors.h"
#include "Mesh2D.h"
#include "Solver2D.h"

// ---- Steady-state temperature with imposed temperatures: all together -----------------------------------------
//
// solve_steady_2d(mesh, k, f, fixed_nodes, fixed_values) puts together the pieces of the previous steps:
//   1. K = the stiffness matrix of the mesh, with the conductivity k;
//   2. A = K with the rows and columns of the fixed nodes replaced by those of the identity;
//   3. b = f - K g^, with the imposed values on the rows of the fixed nodes;
//   4. T = the solution of A T = b, with the conjugate gradient and the Jacobi preconditioner (tolerance 1e-12,
//      at most 10 n iterations, as in the 1D solvers).
// f is the load vector (n numbers, one per node): for a uniform source q it is assemble_load_2d(mesh, q).
// fixed_values[q] is the temperature imposed on the node fixed_nodes[q].
//
// The computation starts from T0 = g^ (the imposed values on the fixed nodes and 0 on the others). The row of a
// fixed node d of A T = b is T[d] = g[d] and T0 already satisfies it: its residual is 0 and the CG never touches it
// (the matrix A is block-structured, the Jacobi preconditioner does not mix the rows), so the fixed nodes come out
// with the imposed temperature EXACTLY. Starting from T0 = 0 they would come out with an error of about 1e-13.
//
// At least one fixed node is needed: without it, K is singular (its rows add up to 0, so if T solves K T = f,
// T + c solves it too, for any c). It throws std::invalid_argument if k is not > 0 (NaN included), if f does not have
// one number per node, if fixed_nodes is empty, if fixed_values has a number of elements different from fixed_nodes,
// if a fixed node is >= n or appears twice, or if a node that is NOT fixed is in no triangle (its row
// of K is all 0). It throws std::runtime_error if the CG does not converge.

namespace {

using Dense = std::vector<std::vector<double>>;
using Nodes = std::vector<std::size_t>;

// Compares two vectors element by element, with the absolute tolerance tol.
void expect_vector(const DVector& got, const DVector& want, double tol, const std::string& what = "") {
    ASSERT_EQ(got.size(), want.size()) << what << ": size";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tol) << what << ": element " << i;
    }
}

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

// Solves A x = b (full table) with Gauss elimination and row swapping. It is slow (n^3) but uses
// neither the CG nor any function of this step or of the two before: it serves as an "oracle".
DVector gauss_solve(Dense A, DVector b) {
    const std::size_t n = b.size();
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t pivot = c;
        for (std::size_t r = c + 1; r < n; ++r) {
            if (std::fabs(A[r][c]) > std::fabs(A[pivot][c])) {
                pivot = r;
            }
        }
        std::swap(A[c], A[pivot]);
        std::swap(b[c], b[pivot]);
        for (std::size_t r = c + 1; r < n; ++r) {
            const double ratio = A[r][c] / A[c][c];
            for (std::size_t j = c; j < n; ++j) {
                A[r][j] -= ratio * A[c][j];
            }
            b[r] -= ratio * b[c];
        }
    }
    DVector x(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double sum = b[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }
    return x;
}

// The temperature computed "by hand": the fixed nodes equal the imposed value and for each free node i
//     sum over the free nodes j of K(i, j) T[j] = f[i] - sum over the fixed nodes d of K(i, d) g[d]
// is a system in the free nodes only, solved with Gauss elimination.
DVector dense_solution(const Mesh2D& m, double k, const DVector& f, const Nodes& fixed, const DVector& values) {
    const std::size_t n = m.num_nodes();
    const Dense K = to_dense(assemble_stiffness_2d(m, k));
    std::vector<bool> is_fixed(n, false);
    DVector T(n, 0.0);
    for (std::size_t q = 0; q < fixed.size(); ++q) {
        is_fixed[fixed[q]] = true;
        T[fixed[q]] = values[q];
    }
    Nodes free_nodes;
    for (std::size_t i = 0; i < n; ++i) {
        if (!is_fixed[i]) {
            free_nodes.push_back(i);
        }
    }
    const std::size_t nf = free_nodes.size();
    Dense A(nf, DVector(nf, 0.0));
    DVector b(nf, 0.0);
    for (std::size_t a = 0; a < nf; ++a) {
        b[a] = f[free_nodes[a]];
        for (std::size_t c = 0; c < nf; ++c) {
            A[a][c] = K[free_nodes[a]][free_nodes[c]];
        }
        for (std::size_t q = 0; q < fixed.size(); ++q) {
            b[a] -= K[free_nodes[a]][fixed[q]] * values[q];
        }
    }
    if (nf > 0) {
        const DVector x = gauss_solve(A, b);
        for (std::size_t a = 0; a < nf; ++a) {
            T[free_nodes[a]] = x[a];
        }
    }
    return T;
}

// A "random" number in [0, 1): a small congruential generator, so that the tests give the same numbers with
// any compiler. `state` is the state of the generator, it changes at every call.
double next_random(unsigned long long& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) / 9007199254740992.0;
}

// A mesh like rectangle_mesh but with the INTERIOR nodes moved at random, in x and in y, by at most 20% of the grid
// step. With 20% every triangle stays counter-clockwise (the Mesh2D constructor checks it and would throw
// an exception); the mesh is skewed instead: triangles of various shapes and areas, with obtuse angles (over 120 degrees).
// The boundary does not move: it is still the rectangle.
Mesh2D perturbed_rectangle(double lx, double ly, std::size_t nx, std::size_t ny, unsigned long long seed) {
    const Mesh2D base = rectangle_mesh(lx, ly, nx, ny);
    const std::size_t n = base.num_nodes();
    std::vector<bool> on_boundary(n, false);
    for (std::size_t d : boundary_nodes(base)) {
        on_boundary[d] = true;
    }
    const double hx = lx / static_cast<double>(nx);
    const double hy = ly / static_cast<double>(ny);
    unsigned long long state = seed;
    std::vector<double> xs(n);
    std::vector<double> ys(n);
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = base.x(i);
        ys[i] = base.y(i);
        if (!on_boundary[i]) {
            xs[i] += (2.0 * next_random(state) - 1.0) * 0.2 * hx;
            ys[i] += (2.0 * next_random(state) - 1.0) * 0.2 * hy;
        }
    }
    std::vector<Triangle> triangles;
    for (std::size_t e = 0; e < base.num_triangles(); ++e) {
        triangles.push_back(base.triangle(e));
    }
    return Mesh2D(xs, ys, triangles);
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

// Meshes of all shapes: a rectangle, the triangle with a node inside, the irregular quadrilateral, two skewed meshes
// and the square of two triangles.
std::vector<Mesh2D> varied_meshes() {
    std::vector<Mesh2D> meshes;
    meshes.push_back(rectangle_mesh(3.0, 1.5, 4, 3));
    meshes.push_back(triangle_with_interior_node());
    meshes.push_back(irregular_quadrilateral());
    meshes.push_back(perturbed_rectangle(2.0, 1.0, 8, 6, 5));
    meshes.push_back(perturbed_rectangle(1.0, 1.0, 12, 12, 31));
    meshes.push_back(rectangle_mesh(1.0, 1.0, 1, 1));
    return meshes;
}

// Four sets of fixed nodes, valid for every connected mesh: node 0 alone, the last node alone, the whole
// boundary, every other node.
std::vector<Nodes> some_fixed_sets(const Mesh2D& m) {
    const std::size_t n = m.num_nodes();
    Nodes first;
    first.push_back(0);
    Nodes last;
    last.push_back(n - 1);
    Nodes every_second;
    for (std::size_t i = 0; i < n; i += 2) {
        every_second.push_back(i);
    }
    std::vector<Nodes> sets;
    sets.push_back(first);
    sets.push_back(last);
    sets.push_back(boundary_nodes(m));
    sets.push_back(every_second);
    return sets;
}

// A non-uniform load, with positive and negative numbers (nodes that heat and nodes that cool).
DVector mixed_load(std::size_t n) {
    DVector f(n);
    for (std::size_t i = 0; i < n; ++i) {
        f[i] = 4.0 * std::sin(1.3 * static_cast<double>(i)) + static_cast<double>(i % 3);
    }
    return f;
}

// Imposed values all different from one another, one per node of the list.
DVector varied_values(const Nodes& nodes) {
    DVector values;
    for (std::size_t q = 0; q < nodes.size(); ++q) {
        values.push_back(10.0 + 7.0 * std::cos(0.7 * static_cast<double>(q)) + 0.5 * static_cast<double>(nodes[q]));
    }
    return values;
}

}  // namespace

// EXAMPLE 69. The usual two square cells (rectangle_mesh(2, 1, 2, 1), k = 2) with the source q = 6, i.e. with
// the load f = (2, 3, 1, 1, 3, 2). Nodes 0, 1, 2 are at the bottom and 3, 4, 5 at the top; nodes 1 and 4 are in the
// middle. Row 1 of K is (-1, 4, -1, 0, -2, 0) and row 4 is (0, -2, 0, -1, 4, -1).
//   (a) Left side (nodes 0 and 3) at 10 and right side (nodes 2 and 5) at 30. The free nodes are 1 and 4 and their two
//       rows say (the fixed ones go to the right: 3 of load plus 10 + 30)
//           4 T1 - 2 T4 = 43        and        -2 T1 + 4 T4 = 43
//       i.e. T1 = T4 = 21.5. It is also the exact formula of the plate, T(x) = 10 + 20 x / 2 + 6 x (2 - x) / (2 * 2),
//       at x = 1.
//   (b) All fixed except node 1, with T0 = 10, T2 = 30, T3 = 10, T4 = 18, T5 = 30. Row 1 says
//           4 T1 = f1 + 1 * T0 + 1 * T2 + 2 * T4 = 3 + 10 + 30 + 36 = 79       i.e. T1 = 19.75
//       (the temperature of a node is a weighted mean of its neighbors, plus a quarter of its load).
//   (c) All fixed except node 4, with T0 = 10, T1 = 20, T2 = 30, T3 = 10, T5 = 30. Row 4 says
//           4 T4 = f4 + 2 * T1 + 1 * T3 + 1 * T5 = 3 + 40 + 10 + 30 = 83       i.e. T4 = 20.75
TEST(Solver2d, TwoSquareCellsMatchHandComputedTemperatures) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const DVector f = assemble_load_2d(m, 6.0);
    expect_vector(f, {2.0, 3.0, 1.0, 1.0, 3.0, 2.0}, 1e-12, "load");

    expect_vector(solve_steady_2d(m, 2.0, f, {0, 2, 3, 5}, {10.0, 30.0, 10.0, 30.0}),
                  {10.0, 21.5, 30.0, 10.0, 21.5, 30.0}, 1e-9, "case (a)");
    expect_vector(solve_steady_2d(m, 2.0, f, {0, 2, 3, 4, 5}, {10.0, 30.0, 10.0, 18.0, 30.0}),
                  {10.0, 19.75, 30.0, 10.0, 18.0, 30.0}, 1e-9, "case (b)");
    expect_vector(solve_steady_2d(m, 2.0, f, {0, 1, 2, 3, 5}, {10.0, 20.0, 30.0, 10.0, 30.0}),
                  {10.0, 20.0, 30.0, 10.0, 20.75, 30.0}, 1e-9, "case (c)");
}

// EXAMPLE 70. The mesh of EXAMPLE 42: two triangles (the unit square cut along the diagonal 1-3) and two
// far-away nodes, 0 and 5, that are in no triangle. With k = 2 and q = 12 the load is f = (0, 4, 2, 4, 2, 0).
// K has 2 on the diagonal at nodes 1, 2, 3, 4, -1 between neighboring nodes (1-2, 2-3, 3-4, 4-1) and 0 between 1-3 and 2-4.
// With node 1 fixed at 10, the rows of nodes 2, 3, 4 say
//         2 T2 - T3 = 2 + 10        2 T3 - T2 - T4 = 4        2 T4 - T3 = 2 + 10.
// By symmetry T2 = T4 = s, so T3 = 2 s - 12 and 2 (2 s - 12) - 2 s = 4, i.e. s = 14 and T3 = 16.
// Nodes 0 and 5 have a row of K that is all 0: no triangle connects them to anything. If they are fixed they
// equal the imposed value and that is all (here 99 and -5). If they are NOT, the problem is singular (their
// temperature can be anything) and the function must throw std::invalid_argument: the diagonal of A would be 0 and
// the Jacobi preconditioner rejects it, without the function having to check anything.
TEST(Solver2d, NodesOutsideEveryTriangleMustBeFixed) {
    const Mesh2D m = mesh_with_nodes_outside_every_triangle();
    const DVector f = assemble_load_2d(m, 12.0);
    expect_vector(f, {0.0, 4.0, 2.0, 4.0, 2.0, 0.0}, 1e-12, "load");

    expect_vector(solve_steady_2d(m, 2.0, f, {0, 1, 5}, {99.0, 10.0, -5.0}), {99.0, 10.0, 14.0, 16.0, 14.0, -5.0},
                  1e-9, "all the isolated nodes fixed");

    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {1}, {10.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {0, 1}, {99.0, 10.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {1, 5}, {10.0, -5.0}), std::invalid_argument);
}

// EXAMPLE 71. Two problems whose exact solution is known, on rectangles of different shapes (including non-square
// cells): the same as in EXAMPLE 66, but now the problem is really solved and T is compared with the formula, node by
// node. k = 2.5 and the source is q = 120.
//   (a) A plate of length L between x = 0 and x = L, with the temperature imposed on the left side (20) and the right side
//       (80); the bottom side and the top one are free (no flux). The problem -k T'' = q has the solution
//           T(x) = T_left + (T_right - T_left) x / L + q x (L - x) / (2 k).
//   (b) The whole boundary fixed at the temperature of
//           T(x, y) = 50 - q (x^2 + y^2) / (4 k),
//       for which -k (T_xx + T_yy) = -k (-q / (2k) - q / (2k)) = q.
// On these meshes K coincides with the five-point finite-difference scheme, which is exact for polynomials of
// second degree: the exact solution, computed at the nodes, solves the discrete system. So T must come out equal
// to the formula (up to the CG error), not "close": it is a check on the number, not on the order of convergence.
TEST(Solver2d, ExactSolutionsOfTwoSimpleProblemsAreReproduced) {
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
        DVector side_values;
        DVector slab(n);
        DVector paraboloid(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double x = m.x(i);
            const double y = m.y(i);
            slab[i] = 20.0 + (80.0 - 20.0) * x / g.lx + q * x * (g.lx - x) / (2.0 * k);
            paraboloid[i] = 50.0 - q * (x * x + y * y) / (4.0 * k);
            if (x < 1e-12 || x > g.lx - 1e-12) {
                sides.push_back(i);
                side_values.push_back(slab[i]);
            }
        }
        const DVector f = assemble_load_2d(m, q);
        expect_vector(solve_steady_2d(m, k, f, sides, side_values), slab, 1e-9, "plate");

        const Nodes boundary = boundary_nodes(m);
        DVector boundary_values;
        for (std::size_t d : boundary) {
            boundary_values.push_back(paraboloid[d]);
        }
        expect_vector(solve_steady_2d(m, k, f, boundary, boundary_values), paraboloid, 1e-9, "paraboloid");
    }
}

// EXAMPLE 72. The "patch test", the classic check of a finite-element code: a LINEAR field
//     T(x, y) = a + b x + c y
// with zero source (f = 0) must be reproduced exactly on ANY mesh of triangles, even a skewed one.
// The value of the field is imposed on the whole boundary and the interior nodes must come out equal to the field.
// (A linear function is among those that linear elements represent without error, and it solves -k Laplacian(T) = 0.)
// On the meshes of EXAMPLE 71 the exact solution is a polynomial of second degree and the mesh is regular; here the
// mesh is skewed (triangles of different shapes and areas, obtuse angles) and k changes between fields (even 0.37, 250).
TEST(Solver2d, ALinearFieldIsReproducedOnIrregularMeshes) {
    const std::vector<Mesh2D> meshes = varied_meshes();
    const double ks[] = {0.37, 250.0, 1.0, 3.5, 12.0, 0.8};
    const double as[] = {20.0, -5.0, 0.0, 100.0, 7.0, 1.0};
    const double bs[] = {3.0, 40.0, 1.0, -8.0, 0.5, 2.0};
    const double cs[] = {-2.0, 11.0, 1.0, 5.0, 9.0, -3.0};

    for (std::size_t index = 0; index < meshes.size(); ++index) {
        SCOPED_TRACE("mesh number " + std::to_string(index));
        const Mesh2D& m = meshes[index];
        const std::size_t n = m.num_nodes();

        DVector field(n);
        for (std::size_t i = 0; i < n; ++i) {
            field[i] = as[index] + bs[index] * m.x(i) + cs[index] * m.y(i);
        }
        const Nodes boundary = boundary_nodes(m);
        DVector values;
        for (std::size_t d : boundary) {
            values.push_back(field[d]);
        }
        expect_vector(solve_steady_2d(m, ks[index], DVector(n, 0.0), boundary, values), field, 1e-9);
    }
}

// EXAMPLE 73. The comparison with an "oracle": the same temperature computed another way, slow and simple,
// that uses neither the CG nor A nor b. K is written as a full table, the equations of the free nodes only are
// written by hand
//     sum over the free nodes j of K(i, j) T[j] = f[i] - sum over the fixed nodes d of K(i, d) g[d]
// and solved with Gauss elimination.
// The meshes are the usual ones, including the skewed ones; the fixed nodes are four different sets (node 0 alone, the
// last node alone, the whole boundary, every other node); the load is not uniform (positive and negative) and the imposed
// values are all different from one another. With a single fixed node the CG works hard (almost 90 iterations on 169 nodes).
// The tolerance is 1e-9. The CG stops when the residual is small, not when the error is: the error on the
// temperature is the residual divided by a small number, so much larger than the CG tolerance. Measured on
// these cases, the maximum difference from the oracle is
//     CG tolerance:            1e-8      1e-10     1e-11     1e-12     1e-14
//     maximum difference:      5e-7      5e-9      1.3e-9    7e-11     4e-12
// With the tolerance 1e-12 (that of the 1D solvers) the test passes with a margin of 14 times; with 1e-11 or wider it does not.
TEST(Solver2d, MatchesTheSlowDenseSolutionOnEveryKindOfMesh) {
    const std::vector<Mesh2D> meshes = varied_meshes();

    for (std::size_t index = 0; index < meshes.size(); ++index) {
        const Mesh2D& m = meshes[index];
        const DVector f = mixed_load(m.num_nodes());
        const std::vector<Nodes> sets = some_fixed_sets(m);

        for (std::size_t s = 0; s < sets.size(); ++s) {
            SCOPED_TRACE("mesh number " + std::to_string(index) + ", node set number " + std::to_string(s));
            const DVector values = varied_values(sets[s]);
            expect_vector(solve_steady_2d(m, 2.5, f, sets[s], values), dense_solution(m, 2.5, f, sets[s], values),
                          1e-9);
        }
    }
}

// EXAMPLE 74. The fixed nodes come out with the imposed temperature EXACTLY, equal bit for bit (ASSERT_EQ, not
// ASSERT_NEAR), and the order in which the nodes are listed does not matter.
//   - On a skewed mesh of 80 nodes, with the whole boundary fixed (32 nodes) at "awkward" values like 37.25 + 0.1 q +
//     sin(q), every fixed node must equal its value. A CG that starts from zero only gets there at the end, with an
//     error of about 1e-13 (37.25 comes out as 37.24999999999973); one that starts from g^ never touches those nodes.
//   - The same nodes with the same values, listed in reverse order, give the same temperature: each value
//     is at the POSITION of the list where its node is.
//   - All the nodes fixed, in shuffled order: the temperature is the list of values, node by node.
TEST(Solver2d, ImposedTemperaturesComeBackExactlyInAnyOrder) {
    const Mesh2D m = perturbed_rectangle(2.0, 1.0, 9, 7, 3);
    const std::size_t n = m.num_nodes();
    ASSERT_EQ(n, 80u);
    const DVector f = assemble_load_2d(m, 5.0);

    const Nodes fixed = boundary_nodes(m);
    DVector values;
    for (std::size_t q = 0; q < fixed.size(); ++q) {
        values.push_back(37.25 + 0.1 * static_cast<double>(q) + std::sin(static_cast<double>(q)));
    }
    const DVector T = solve_steady_2d(m, 1.7, f, fixed, values);
    ASSERT_EQ(T.size(), n);
    for (std::size_t q = 0; q < fixed.size(); ++q) {
        ASSERT_EQ(T[fixed[q]], values[q]) << std::setprecision(17) << "fixed node " << fixed[q] << ": came out "
                                          << T[fixed[q]] << " instead of " << values[q];
    }

    const Nodes reversed_nodes(fixed.rbegin(), fixed.rend());
    const DVector reversed_values(values.rbegin(), values.rend());
    expect_vector(solve_steady_2d(m, 1.7, f, reversed_nodes, reversed_values), T, 1e-12, "reverse order");

    const Mesh2D two_cells = rectangle_mesh(2.0, 1.0, 2, 1);
    const DVector all = solve_steady_2d(two_cells, 2.0, assemble_load_2d(two_cells, 6.0), {4, 0, 5, 2, 1, 3},
                                        {40.0, 0.5, 50.0, 20.0, 10.0, 30.0});
    ASSERT_EQ(all.size(), 6u);
    const DVector want = {0.5, 10.0, 20.0, 30.0, 40.0, 50.0};
    for (std::size_t i = 0; i < 6; ++i) {
        ASSERT_EQ(all[i], want[i]) << "all fixed, node " << i;
    }
}

// EXAMPLE 75. The heat balance. The rows of K add up to 0 (a constant temperature lets no heat through) and K
// is symmetric, so the sum of all the n rows of K T is 0. At the free nodes K T = f; at the fixed nodes d, the
// quantity
//     f[d] - (K T)[d] = (heat produced at d) - (heat that flows from d to the neighbors)
// is the heat that the fixed node sends to the outside. The sum of all the rows gives
//     sum over the fixed nodes d of (f[d] - (K T)[d]) = sum over ALL the nodes of f
// i.e. all the heat produced in the plate leaves through the fixed nodes: nothing is lost and nothing is created. The test
// checks it on every mesh, with four sets of fixed nodes and a load with parts that heat and parts that
// cool (the sum of the loads can even be negative or close to 0).
TEST(Solver2d, TheFixedNodesRemoveExactlyTheHeatThatIsProduced) {
    const std::vector<Mesh2D> meshes = varied_meshes();

    for (std::size_t index = 0; index < meshes.size(); ++index) {
        const Mesh2D& m = meshes[index];
        const std::size_t n = m.num_nodes();
        const CSR K = assemble_stiffness_2d(m, 2.5);
        const DVector f = mixed_load(n);
        double produced = 0.0;
        double size = 1.0;
        for (double x : f) {
            produced += x;
            size += std::fabs(x);
        }
        const std::vector<Nodes> sets = some_fixed_sets(m);

        for (std::size_t s = 0; s < sets.size(); ++s) {
            SCOPED_TRACE("mesh number " + std::to_string(index) + ", node set number " + std::to_string(s));
            const DVector T = solve_steady_2d(m, 2.5, f, sets[s], varied_values(sets[s]));
            const DVector KT = K * T;
            double removed = 0.0;
            for (std::size_t d : sets[s]) {
                removed += f[d] - KT[d];
            }
            EXPECT_NEAR(removed, produced, 1e-9 * size);
        }
    }
}

// EXAMPLE 76. The maximum principle, another law of physics: with no source (f = 0) the temperature of every
// node lies between the minimum and the maximum imposed temperature (heat goes from hot to cold: no node becomes
// hotter than all its neighbors); with a heating source (f >= 0) no node falls below the minimum imposed, and with
// a cooling one (f <= 0) none rises above the maximum.
// In the discrete problem it holds when the off-diagonal numbers of K are all <= 0 (each node is "pulled" toward
// its neighbors). On the rectangles of rectangle_mesh this happens (even with non-square cells: the triangles
// have a right angle and two acute ones), but NOT on every mesh: K(i, j) becomes positive if the two angles opposite
// the side i-j add up to more than 180 degrees, and on the skewed meshes of EXAMPLE 72 this happens (with obtuse
// angles of 120-140 degrees). For this reason the test is only on the rectangles.
TEST(Solver2d, TemperaturesStayBetweenTheImposedOnesOnRectangles) {
    struct Grid {
        double lx;
        double ly;
        std::size_t nx;
        std::size_t ny;
    };
    const Grid grids[] = {{1.0, 1.0, 10, 10}, {3.0, 1.0, 12, 4}, {1.0, 2.0, 3, 9}};
    unsigned long long state = 2024;

    for (const Grid& g : grids) {
        SCOPED_TRACE(std::to_string(g.nx) + " x " + std::to_string(g.ny) + " cells");
        const Mesh2D m = rectangle_mesh(g.lx, g.ly, g.nx, g.ny);
        const std::size_t n = m.num_nodes();
        const Nodes boundary = boundary_nodes(m);
        DVector values;
        for (std::size_t q = 0; q < boundary.size(); ++q) {
            values.push_back(20.0 + 80.0 * next_random(state));
        }
        const double lowest = *std::min_element(values.begin(), values.end());
        const double highest = *std::max_element(values.begin(), values.end());
        const double slack = 1e-9;

        const DVector none = solve_steady_2d(m, 1.5, DVector(n, 0.0), boundary, values);
        const DVector heating = solve_steady_2d(m, 1.5, assemble_load_2d(m, 5.0), boundary, values);
        const DVector cooling = solve_steady_2d(m, 1.5, assemble_load_2d(m, -5.0), boundary, values);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_GE(none[i], lowest - slack) << "no source, node " << i;
            EXPECT_LE(none[i], highest + slack) << "no source, node " << i;
            EXPECT_GE(heating[i], lowest - slack) << "heating source, node " << i;
            EXPECT_LE(cooling[i], highest + slack) << "cooling source, node " << i;
        }
    }
}

// EXAMPLE 77. The errors that are the caller's fault. On the two cells of EXAMPLE 69 (6 nodes) the following throw
// std::invalid_argument:
//   - k = 0, negative or NaN: even when ALL the nodes are fixed (then A is the identity and no other
//     check would notice);
//   - f with a number of elements different from 6 (5, 7, none);
//   - no fixed node, both with a nonzero load and with f = 0 (with f = 0 the CG would start from T = 0, with
//     residual 0, and "converge" at once, returning a meaningless temperature without complaining);
//   - fixed_values with a number of elements different from fixed_nodes, in both directions;
//   - a fixed node equal to 6 (the last one is 5) or larger, even huge (a billion);
//   - a repeated node, with the same value or with different values.
// These are fine instead: a single fixed node (any), the fixed nodes in shuffled order and all the nodes fixed.
// The check of an index must come BEFORE using it to read or write (for example to build the starting point of
// the CG): an index of a billion used first makes the program crash instead of throwing
// the exception.
TEST(Solver2d, ThrowsInvalidArgumentOnBadInput) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const DVector f = assemble_load_2d(m, 6.0);
    const DVector zero(6, 0.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();

    EXPECT_NO_THROW(solve_steady_2d(m, 2.0, f, {3}, {10.0}));
    EXPECT_NO_THROW(solve_steady_2d(m, 2.0, f, {5}, {10.0}));
    EXPECT_NO_THROW(solve_steady_2d(m, 2.0, f, {4, 0, 1}, {1.0, 2.0, 3.0}));
    EXPECT_NO_THROW(solve_steady_2d(m, 2.0, f, {0, 1, 2, 3, 4, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));

    EXPECT_THROW(solve_steady_2d(m, 0.0, f, {0, 2}, {10.0, 30.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, -2.0, f, {0, 2}, {10.0, 30.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, nan, f, {0, 2}, {10.0, 30.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 0.0, f, {0, 1, 2, 3, 4, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
                 std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, -2.0, f, {0, 1, 2, 3, 4, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
                 std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, nan, f, {0, 1, 2, 3, 4, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
                 std::invalid_argument);

    EXPECT_THROW(solve_steady_2d(m, 2.0, DVector(5, 1.0), {0, 2}, {10.0, 30.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, DVector(7, 1.0), {0, 2}, {10.0, 30.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, DVector(), {0, 2}, {10.0, 30.0}), std::invalid_argument);

    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {}, {}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, zero, {}, {}), std::invalid_argument);

    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {0, 2}, {10.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {0}, {10.0, 30.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {}, {10.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {0}, {}), std::invalid_argument);

    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {6}, {10.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {0, 100}, {10.0, 30.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {1000000000}, {10.0}), std::invalid_argument);

    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {2, 2}, {10.0, 10.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {0, 2, 3, 2}, {10.0, 30.0, 10.0, 31.0}), std::invalid_argument);
}

// EXAMPLE 78. Problems that have no solution, and that the CG cannot solve: the function throws
// std::runtime_error (and does not return a wrong vector or one full of NaN).
//   - A NaN in the load, or in an imposed value: comparisons with a NaN are all false and the CG stops at once,
//     not converged.
//   - A mesh of TWO separate pieces (two triangles with no nodes in common) with a fixed node in only one, and a
//     heat source: the piece with no fixed node heats up forever, a steady state does not
//     exist (mathematically, K of that piece is singular and the system has no solution). The CG never converges.
//     With a fixed node in EVERY piece instead the problem is solvable: no exception.
// (Without a source that piece would have solutions galore and the CG would return one, T = 0: the check "at least one
// fixed node per piece" is not made, it is the caller's job.)
TEST(Solver2d, ThrowsRuntimeErrorWhenTheSystemCannotBeSolved) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    DVector f = assemble_load_2d(m, 6.0);

    DVector with_nan = f;
    with_nan[4] = nan;
    EXPECT_THROW(solve_steady_2d(m, 2.0, with_nan, {0, 2, 3, 5}, {10.0, 30.0, 10.0, 30.0}), std::runtime_error);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {0, 2, 3, 5}, {10.0, nan, 10.0, 30.0}), std::runtime_error);

    const Mesh2D two_pieces({0.0, 1.0, 0.0, 3.0, 4.0, 3.0}, {0.0, 0.0, 1.0, 0.0, 0.0, 1.0}, {{0, 1, 2}, {3, 4, 5}});
    const DVector heat = assemble_load_2d(two_pieces, 1.0);
    EXPECT_THROW(solve_steady_2d(two_pieces, 1.0, heat, {0}, {5.0}), std::runtime_error);
    EXPECT_THROW(solve_steady_2d(two_pieces, 1.0, heat, {4}, {5.0}), std::runtime_error);
    EXPECT_NO_THROW(solve_steady_2d(two_pieces, 1.0, heat, {0, 4}, {5.0, 7.0}));
}

// EXAMPLE 79. A mesh of 60 x 60 cells (3721 nodes), the paraboloid of EXAMPLE 71 (the whole boundary fixed,
// k = 1, q = 10, T = -q (x^2 + y^2) / 4). The CG takes about 200 iterations (never more than 100 on the
// meshes of the other tests): enough to see whether the tolerance and the maximum number of iterations are right on a
// problem of real size. The result must match the exact formula within 1e-8 (the true error is about 5e-12); the
// function takes about two tenths of a second (and if the CG does not converge it runs for up to 10 n iterations
// before throwing the exception: about twenty seconds).
TEST(Solver2d, ALargeMeshIsSolvedAccurately) {
    const std::size_t cells = 60;
    const double k = 1.0;
    const double q = 10.0;
    const Mesh2D m = rectangle_mesh(1.0, 1.0, cells, cells);
    const std::size_t n = m.num_nodes();
    ASSERT_EQ(n, 3721u);

    DVector exact(n);
    for (std::size_t i = 0; i < n; ++i) {
        exact[i] = -q * (m.x(i) * m.x(i) + m.y(i) * m.y(i)) / (4.0 * k);
    }
    const Nodes boundary = boundary_nodes(m);
    ASSERT_EQ(boundary.size(), 240u);
    DVector values;
    for (std::size_t d : boundary) {
        values.push_back(exact[d]);
    }

    const DVector T = solve_steady_2d(m, k, assemble_load_2d(m, q), boundary, values);
    ASSERT_EQ(T.size(), n);
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        worst = std::max(worst, std::fabs(T[i] - exact[i]));
    }
    EXPECT_LT(worst, 1e-8) << "the maximum error is " << worst;
}

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "Assembly2D.h"
#include "CSR_and_Vectors.h"
#include "Mesh2D.h"
#include "Solver2D.h"

// ---- The plate that changes in time: the theta scheme in 2D ----------------------------------------------------
//
// In 1D we followed a bar in time with the theta scheme (solve_transient). Here we do the same with the
// plate: solve_transient_2d(mesh, rho_c, k, f, fixed_nodes, fixed_values, T0, dt, n_steps, theta). For the
// temperatures at the nodes the heat law is
//
//     M dT/dt + K T = f
//
// with M the mass matrix (assemble_mass_2d(mesh, rho_c)), K the stiffness matrix (assemble_stiffness_2d(mesh, k))
// and f the load vector (n numbers, it does not change in time). From T_old (at time t) to T_new (at time t + dt) the
// theta scheme computes K T as an average of the two instants:
//
//     M (T_new - T_old) / dt + K (theta T_new + (1 - theta) T_old) = f
//
// and, multiplying by dt and moving T_new to the left, every step is a linear system
//
//     (M + theta dt K) T_new = (M - (1 - theta) dt K) T_old + dt f
//
// theta = 1 is backward Euler (stable for every dt), theta = 1/2 is Crank-Nicolson (stable for every dt and of
// order 2 in time), theta = 0 is forward Euler (stable only if dt is small). It is the same scheme as in 1D:
// only the matrices change.
//
// The fixed nodes have the imposed temperature, the same at every step. They are removed from the system as in the steady
// state: A = M + theta dt K becomes the matrix with the rows and columns of the fixed nodes replaced by those of
// the identity (dirichlet_matrix_triplets_2d: it does not change from one step to the next, it is built only once) and the
// right-hand side (M - (1 - theta) dt K) T_old + dt f becomes b with dirichlet_rhs_2d, taken with the whole A (the one
// before the substitution). It is solved with the conjugate gradient and the first step starts from T0, taken as is,
// also on the fixed nodes.
//
// As "oracles" (right answers computed another way) we use: a mesh with a single free node, where every
// step is one equation with one unknown; the solution with full matrices and Gauss elimination; three laws of
// physics (the steady state is a fixed point, the heat ends up in the steady state, in an insulated plate the
// heat is conserved); the stability of the three schemes; and the fact that 7 steps are 3 steps followed by another 4.

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
// neither the CG nor other functions of the project: it serves as an "oracle".
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

// The theta scheme "by hand", with full matrices. At every step, for every free node i,
//     sum over the free nodes j of (M + theta dt K)(i, j) T_new[j]
//         = sum over all the nodes j of (M - (1 - theta) dt K)(i, j) T_old[j] + dt f[i]
//           - sum over the fixed nodes d of (M + theta dt K)(i, d) g[d]
// is a system in the free nodes only (Gauss); the fixed nodes equal the imposed value. T0 is taken as it is.
DVector dense_transient(const Mesh2D& m, double rho_c, double k, const DVector& f, const Nodes& fixed,
                        const DVector& values, DVector T, double dt, std::size_t n_steps, double theta) {
    const std::size_t n = m.num_nodes();
    const Dense M = to_dense(assemble_mass_2d(m, rho_c));
    const Dense K = to_dense(assemble_stiffness_2d(m, k));
    std::vector<bool> is_fixed(n, false);
    for (std::size_t d : fixed) {
        is_fixed[d] = true;
    }
    Nodes free_nodes;
    for (std::size_t i = 0; i < n; ++i) {
        if (!is_fixed[i]) {
            free_nodes.push_back(i);
        }
    }
    const std::size_t nf = free_nodes.size();
    Dense A(nf, DVector(nf, 0.0));
    for (std::size_t a = 0; a < nf; ++a) {
        for (std::size_t c = 0; c < nf; ++c) {
            A[a][c] = M[free_nodes[a]][free_nodes[c]] + theta * dt * K[free_nodes[a]][free_nodes[c]];
        }
    }
    for (std::size_t step = 0; step < n_steps; ++step) {
        DVector b(nf, 0.0);
        for (std::size_t a = 0; a < nf; ++a) {
            const std::size_t i = free_nodes[a];
            double rhs = dt * f[i];
            for (std::size_t j = 0; j < n; ++j) {
                rhs += (M[i][j] - (1.0 - theta) * dt * K[i][j]) * T[j];
            }
            for (std::size_t q = 0; q < fixed.size(); ++q) {
                rhs -= (M[i][fixed[q]] + theta * dt * K[i][fixed[q]]) * values[q];
            }
            b[a] = rhs;
        }
        if (nf > 0) {
            const DVector x = gauss_solve(A, b);
            for (std::size_t a = 0; a < nf; ++a) {
                T[free_nodes[a]] = x[a];
            }
        }
        for (std::size_t q = 0; q < fixed.size(); ++q) {
            T[fixed[q]] = values[q];
        }
    }
    return T;
}

// The total heat of the plate, the integral of rho_c T: the sum of all the numbers of M T (i.e. 1^T M T).
double total_heat(const Mesh2D& m, double rho_c, const DVector& T) {
    const DVector MT = assemble_mass_2d(m, rho_c) * T;
    double sum = 0.0;
    for (double v : MT) {
        sum += v;
    }
    return sum;
}

// A "random" number in [0, 1): a small congruential generator, so that the tests give the same numbers with
// any compiler. `state` is the state of the generator, it changes at every call.
double next_random(unsigned long long& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) / 9007199254740992.0;
}

// A mesh like rectangle_mesh but with the INTERIOR nodes moved at random, in x and in y, by at most 20% of the grid
// step: triangles of various shapes and areas, with obtuse angles. The boundary does not move: it is still the
// rectangle.
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

// A mesh in which nodes 0 = (5, 5) and 5 = (7, 7) belong to no triangle.
Mesh2D mesh_with_nodes_outside_every_triangle() {
    return Mesh2D({5.0, 0.0, 1.0, 1.0, 0.0, 7.0}, {5.0, 0.0, 0.0, 1.0, 1.0, 7.0}, {{1, 2, 3}, {1, 3, 4}});
}

// A non-uniform load, with positive and negative numbers (nodes that heat and nodes that cool).
DVector mixed_load(std::size_t n) {
    DVector f(n);
    for (std::size_t i = 0; i < n; ++i) {
        f[i] = 4.0 * std::sin(1.3 * static_cast<double>(i)) + static_cast<double>(i % 3);
    }
    return f;
}

// The data of a VALID call to solve_transient_2d, on the 2 x 2 mesh (9 nodes, 8 fixed on the boundary and the center
// free): the error tests change only one of them at a time.
struct Call {
    Mesh2D mesh = rectangle_mesh(1.0, 1.0, 2, 2);
    double rho_c = 2.0;
    double k = 3.0;
    DVector f = DVector(9, 1.0);
    Nodes fixed = {0, 1, 2, 3, 5, 6, 7, 8};
    DVector values = DVector(8, 5.0);
    DVector T0 = DVector(9, 20.0);
    double dt = 0.01;
    std::size_t n_steps = 3;
    double theta = 1.0;

    DVector run() const {
        return solve_transient_2d(mesh, rho_c, k, f, fixed, values, T0, dt, n_steps, theta);
    }
};

// The call c throws std::invalid_argument, with 0 steps and with 3 steps: the errors are checked right away, before
// taking any step.
void expect_invalid(const Call& c, const std::string& what) {
    for (std::size_t steps : {0, 3}) {
        Call copy = c;
        copy.n_steps = steps;
        EXPECT_THROW(copy.run(), std::invalid_argument) << what << " (n_steps = " << steps << ")";
    }
}

}  // namespace

// EXAMPLE 96. The 2 x 2 mesh (rectangle_mesh(1, 1, 2, 2)): 9 nodes, 8 on the boundary (fixed) and only one free, the
// center c = 4. With a single free node every step is ONE equation. The center belongs to 6 triangles of area 1/8:
// M(c, c) = rho_c / 8 and K(c, c) = 4 k; the four nodes on the axes (1, 3, 5, 7) have K(c, j) = -k, the other neighbors
// (0 and 8, on the diagonal of the squares) K(c, j) = 0; and f(c) = q / 4 with the uniform source q. If, at the fixed nodes,
// T0 equals the imposed value g_j, the mass terms of the fixed nodes are equal on both sides and cancel; what remains:
//
//     (rho_c / 8 + 4 theta k dt) T_new = (rho_c / 8 - 4 (1 - theta) k dt) T_old + dt (q / 4 + k S)
//
// with S = g1 + g3 + g5 + g7, i.e. T_n = T_inf + gamma^n (T_0 - T_inf), with
//
//     gamma = (rho_c - 32 (1 - theta) k dt) / (rho_c + 32 theta k dt)        T_inf = (q / 4 + k S) / (4 k)
//
// Numbers: rho_c = 2, k = 3, q = 7, dt = 0.01, g_j = 3 + 2 j (so S = 5 + 9 + 13 + 17 = 44), T_0 = 20 at the center.
// T_inf = 133.75 / 12 = 11.1458. With theta = 1: gamma = 2 / 2.96 = 0.67568 and after one step T = 11.1458 + 0.67568 (20 -
// 11.1458) = 17.128.
TEST(Transient2D, SingleUnknownFollowsTheScalarRecursion) {
    const Mesh2D mesh = rectangle_mesh(1.0, 1.0, 2, 2);
    const double rho_c = 2.0;
    const double k = 3.0;
    const double q = 7.0;
    const double dt = 0.01;
    const Nodes fixed = {0, 1, 2, 3, 5, 6, 7, 8};
    DVector values;
    DVector T0(9, 0.0);
    for (std::size_t d : fixed) {
        values.push_back(3.0 + 2.0 * static_cast<double>(d));
        T0[d] = values.back();
    }
    T0[4] = 20.0;
    const DVector f = assemble_load_2d(mesh, q);

    double S = 0.0;
    for (std::size_t d : {1, 3, 5, 7}) {
        S += 3.0 + 2.0 * static_cast<double>(d);
    }
    const double t_inf = (q / 4.0 + k * S) / (4.0 * k);

    for (double theta : {1.0, 0.5, 0.3, 0.0}) {
        const double gamma = (rho_c - 32.0 * (1.0 - theta) * k * dt) / (rho_c + 32.0 * theta * k * dt);
        for (std::size_t steps : {1, 6}) {
            const DVector T = solve_transient_2d(mesh, rho_c, k, f, fixed, values, T0, dt, steps, theta);
            ASSERT_EQ(T.size(), 9u);
            EXPECT_NEAR(T[4], t_inf + std::pow(gamma, steps) * (20.0 - t_inf), 1e-9)
                << "theta = " << theta << ", steps = " << steps;
            for (std::size_t q2 = 0; q2 < fixed.size(); ++q2) {
                EXPECT_EQ(T[fixed[q2]], values[q2]) << "fixed node " << fixed[q2];
            }
        }
    }
}

// EXAMPLE 97. The comparison with an oracle: the same scheme computed with full matrices and Gauss elimination
// (dense_transient), on a skewed mesh of 30 nodes, with a source that heats at some nodes and cools at others,
// with the whole boundary and one interior node fixed and with a random T0 (also on the fixed nodes, where it does NOT
// equal the imposed value: the first step uses T0 as it is and the result at the fixed nodes is nevertheless the imposed
// value, EXACTLY, even after a single step). Four values of theta, with 1 and with 10 steps; dt = 0.002 is small enough
// even for theta = 0.
TEST(Transient2D, MatchesADenseSolutionOnAnIrregularMesh) {
    const Mesh2D mesh = perturbed_rectangle(2.0, 1.0, 5, 4, 11);
    const std::size_t n = mesh.num_nodes();
    Nodes fixed = boundary_nodes(mesh);
    fixed.push_back(15);  // an interior node (column 3, row 2 of the 6 x 5 grid)
    DVector values;
    for (std::size_t q = 0; q < fixed.size(); ++q) {
        values.push_back(30.0 + 20.0 * std::cos(0.7 * static_cast<double>(q)));
    }
    const DVector f = mixed_load(n);
    DVector T0(n);
    unsigned long long state = 123;
    for (std::size_t i = 0; i < n; ++i) {
        T0[i] = 100.0 * next_random(state);
    }
    const double rho_c = 1.7;
    const double k = 2.3;
    const double dt = 0.002;
    for (double theta : {1.0, 0.5, 0.25, 0.0}) {
        for (std::size_t steps : {1, 10}) {
            const DVector got = solve_transient_2d(mesh, rho_c, k, f, fixed, values, T0, dt, steps, theta);
            const DVector want = dense_transient(mesh, rho_c, k, f, fixed, values, T0, dt, steps, theta);
            const std::string what = "theta = " + std::to_string(theta) + ", steps = " + std::to_string(steps);
            expect_vector(got, want, 1e-6, what);
            for (std::size_t q = 0; q < fixed.size(); ++q) {
                EXPECT_EQ(got[fixed[q]], values[q]) << what << ", fixed node " << fixed[q];
            }
        }
    }
}

// EXAMPLE 98. The steady state is a fixed point. Let T_s be any temperature and f = K T_s: then T_s is the
// steady state of the problem with the imposed temperatures equal to T_s at the fixed nodes (K T_s = f). If the plate
// starts from T_s, one step of any scheme (any theta and any dt) leaves it where it is:
//     (M + theta dt K) T_s = (M - (1 - theta) dt K) T_s + dt K T_s     because  K T_s = f
// and so after 5 steps T_s has not changed. If the right-hand side had a dt too few, or too many, or a wrong
// sign, the plate would move. With theta < 1/2 the dt must be small (stability), otherwise the rounding
// error would grow at every step.
TEST(Transient2D, TheSteadyStateIsAFixedPoint) {
    const Mesh2D mesh = perturbed_rectangle(1.5, 1.0, 6, 4, 7);
    const std::size_t n = mesh.num_nodes();
    const double rho_c = 2.2;
    const double k = 1.9;
    DVector Ts(n);
    for (std::size_t i = 0; i < n; ++i) {
        Ts[i] = 40.0 + 15.0 * std::sin(0.9 * static_cast<double>(i)) + 0.3 * static_cast<double>(i);
    }
    Nodes fixed = boundary_nodes(mesh);
    fixed.push_back(17);  // an interior node (column 3, row 2 of the 7 x 5 grid)
    DVector values;
    for (std::size_t d : fixed) {
        values.push_back(Ts[d]);
    }
    const DVector f = assemble_stiffness_2d(mesh, k) * Ts;

    const double cases[6][2] = {{1.0, 0.5}, {1.0, 20.0}, {0.5, 0.5}, {0.5, 20.0}, {0.3, 0.002}, {0.0, 0.001}};
    for (const auto& c : cases) {
        const DVector T = solve_transient_2d(mesh, rho_c, k, f, fixed, values, Ts, c[1], 5, c[0]);
        expect_vector(T, Ts, 1e-6, "theta = " + std::to_string(c[0]) + ", dt = " + std::to_string(c[1]));
    }
}

// EXAMPLE 99. The heat reaches the steady state. A plate at 20 degrees, with the boundary at different temperatures and
// a source that depends on the point: with backward Euler (stable for every dt) and long steps (dt = 0.5, 40
// steps, i.e. 20 seconds, many relaxation times) the temperature becomes the one that solve_steady_2d gives
// directly, with the same k, the same f and the same imposed temperatures. (rho_c does not matter in the steady
// state: M disappears when T no longer changes.) If f entered the step without dt, or with a wrong weight, the
// final state would be different.
TEST(Transient2D, BackwardEulerRelaxesToTheSteadyState) {
    const Mesh2D mesh = perturbed_rectangle(2.0, 1.0, 6, 6, 21);
    const double rho_c = 1.5;
    const double k = 0.7;
    const DVector f = assemble_load_2d(mesh, [](double x, double y) { return 6.0 + 2.0 * x - 3.0 * y; });
    const Nodes fixed = boundary_nodes(mesh);
    DVector values;
    for (std::size_t q = 0; q < fixed.size(); ++q) {
        values.push_back(20.0 + 10.0 * std::sin(0.5 * static_cast<double>(q)));
    }
    const DVector steady = solve_steady_2d(mesh, k, f, fixed, values);
    const DVector T0(mesh.num_nodes(), 20.0);
    const DVector T = solve_transient_2d(mesh, rho_c, k, f, fixed, values, T0, 0.5, 40, 1.0);
    expect_vector(T, steady, 1e-6, "after 40 steps");
}

// EXAMPLE 100. An insulated plate: no fixed node (empty fixed_nodes), the heat does not leave. The columns of K
// add up to 0 (a constant temperature lets no heat through), so summing the rows of the system 1^T K disappears and
// the total heat H = 1^T M T changes only because of the source:
//     H_new = H_old + dt (sum of the numbers of f)        and the sum of f is (source q) x (area of the plate).
// Without a source the total heat stays the same, for every theta. With the uniform source q it grows by dt q area at
// every step. Finally with backward Euler and long steps the plate goes to the same temperature everywhere,
// the one that has the same heat: H / (rho_c area), because 1^T M 1 = rho_c area.
TEST(Transient2D, InsulatedPlateConservesTheHeat) {
    const Mesh2D mesh = perturbed_rectangle(2.0, 1.0, 6, 4, 3);
    const std::size_t n = mesh.num_nodes();
    const double rho_c = 1.4;
    const double k = 0.8;
    double area = 0.0;
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        area += mesh.area(e);
    }
    DVector T0(n);
    unsigned long long state = 77;
    for (std::size_t i = 0; i < n; ++i) {
        T0[i] = 100.0 * next_random(state);
    }
    const double heat0 = total_heat(mesh, rho_c, T0);
    const Nodes none;
    const DVector no_values;

    for (double theta : {1.0, 0.5, 0.25, 0.0}) {
        const double dt = 0.003;
        const DVector cold = solve_transient_2d(mesh, rho_c, k, DVector(n, 0.0), none, no_values, T0, dt, 8, theta);
        EXPECT_NEAR(total_heat(mesh, rho_c, cold), heat0, 1e-9 * heat0) << "no source, theta = " << theta;

        const double q = 3.0;
        const DVector f = assemble_load_2d(mesh, q);
        const DVector hot = solve_transient_2d(mesh, rho_c, k, f, none, no_values, T0, dt, 8, theta);
        EXPECT_NEAR(total_heat(mesh, rho_c, hot), heat0 + 8.0 * dt * q * area, 1e-9 * heat0)
            << "with the source, theta = " << theta;
    }

    const DVector settled = solve_transient_2d(mesh, rho_c, k, DVector(n, 0.0), none, no_values, T0, 5.0, 40, 1.0);
    const double mean = heat0 / (rho_c * area);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(settled[i], mean, 1e-6) << "node " << i;
    }
}

// EXAMPLE 101. Zero steps: the result is T0, EQUAL (EXPECT_EQ on the vector, bit for bit), even at the fixed nodes
// where T0 does not equal the imposed value: no time has passed and there is nothing to impose.
TEST(Transient2D, ZeroStepsReturnTheInitialTemperature) {
    const Mesh2D mesh = rectangle_mesh(1.0, 1.0, 3, 3);
    const std::size_t n = mesh.num_nodes();
    DVector T0(n);
    for (std::size_t i = 0; i < n; ++i) {
        T0[i] = 1.0 + 0.37 * static_cast<double>(i);
    }
    const Nodes fixed = boundary_nodes(mesh);
    const DVector values(fixed.size(), 100.0);
    const DVector f = assemble_load_2d(mesh, 2.0);
    for (double theta : {0.0, 0.5, 1.0}) {
        EXPECT_EQ(solve_transient_2d(mesh, 1.5, 0.9, f, fixed, values, T0, 0.1, 0, theta), T0)
            << "with fixed nodes, theta = " << theta;
        EXPECT_EQ(solve_transient_2d(mesh, 1.5, 0.9, f, Nodes(), DVector(), T0, 0.1, 0, theta), T0)
            << "without fixed nodes, theta = " << theta;
    }
}

// EXAMPLE 102. After every step the only "state" of the plate is its temperature T: the data (f, the imposed
// temperatures, dt) do not change in time. So 7 steps are 3 steps followed by another 4 steps that start from the result
// of the first 3 (at the fixed nodes the imposed value already holds): the same vector, up to the CG error. If the
// right-hand side used T0 instead of the T of the previous step, or if the function remembered something from one call
// to the next, the two results would be different.
TEST(Transient2D, SevenStepsAreThreeStepsFollowedByFour) {
    const Mesh2D mesh = perturbed_rectangle(1.5, 1.0, 4, 3, 5);
    const std::size_t n = mesh.num_nodes();
    Nodes fixed = boundary_nodes(mesh);
    DVector values;
    for (std::size_t q = 0; q < fixed.size(); ++q) {
        values.push_back(10.0 + 3.0 * static_cast<double>(q % 4));
    }
    const DVector f = mixed_load(n);
    DVector T0(n);
    unsigned long long state = 9;
    for (std::size_t i = 0; i < n; ++i) {
        T0[i] = 50.0 * next_random(state);
    }
    for (double theta : {1.0, 0.5, 0.0}) {
        const DVector all = solve_transient_2d(mesh, 1.2, 0.9, f, fixed, values, T0, 0.002, 7, theta);
        const DVector first = solve_transient_2d(mesh, 1.2, 0.9, f, fixed, values, T0, 0.002, 3, theta);
        const DVector second = solve_transient_2d(mesh, 1.2, 0.9, f, fixed, values, first, 0.002, 4, theta);
        expect_vector(second, all, 1e-6, "theta = " + std::to_string(theta));
    }
}

// EXAMPLE 103. Stability. With no source and with temperature 0 imposed on the whole boundary the plate cools down and
// its energy E = sqrt(T^T M T) cannot grow. Every eigenmode of the plate cools with a rate
// lambda >= 0 and one step multiplies its amplitude by
//     1 / (1 + dt lambda)                              backward Euler
//     (1 - dt lambda / 2) / (1 + dt lambda / 2)        Crank-Nicolson
//     1 - dt lambda                                    forward Euler
// The first two are ALWAYS between -1 and 1, whatever dt is: the energy never grows, not even with a huge step
// (dt = 100 with a plate of side 1 and k = rho_c = 1). The third is between -1 and 1 only if dt lambda <= 2: with
// dt = 0.003 the plate of 4 x 4 cells cools down, with dt = 0.1 the fastest mode is multiplied each time by a number
// of modulus larger than 1, the plate "blows up" and the energy grows at every step. The test checks both things.
TEST(Transient2D, ImplicitSchemesNeverGainEnergyAndExplicitOneBlowsUp) {
    const Mesh2D mesh = rectangle_mesh(1.0, 1.0, 4, 4);
    const std::size_t n = mesh.num_nodes();
    const Nodes fixed = boundary_nodes(mesh);
    const DVector values(fixed.size(), 0.0);
    const DVector f(n, 0.0);
    std::vector<bool> on_boundary(n, false);
    for (std::size_t d : fixed) {
        on_boundary[d] = true;
    }
    DVector T0(n, 0.0);
    unsigned long long state = 5;
    for (std::size_t i = 0; i < n; ++i) {
        if (!on_boundary[i]) {
            T0[i] = 100.0 * next_random(state);
        }
    }
    const CSR M = assemble_mass_2d(mesh, 1.0);
    const double energy0 = std::sqrt(dot(T0, M * T0));

    const double cases[7][2] = {{1.0, 0.01}, {1.0, 1.0},   {1.0, 100.0}, {0.5, 0.01},
                                {0.5, 1.0},  {0.5, 100.0}, {0.0, 0.003}};
    for (const auto& c : cases) {
        DVector T = T0;
        double energy = energy0;
        for (int step = 1; step <= 10; ++step) {
            T = solve_transient_2d(mesh, 1.0, 1.0, f, fixed, values, T, c[1], 1, c[0]);
            const double now = std::sqrt(dot(T, M * T));
            EXPECT_LE(now, energy * (1.0 + 1e-12)) << "theta = " << c[0] << ", dt = " << c[1] << ", step " << step;
            energy = now;
        }
    }

    const DVector blown = solve_transient_2d(mesh, 1.0, 1.0, f, fixed, values, T0, 0.1, 3, 0.0);
    EXPECT_GT(std::sqrt(dot(blown, M * blown)), 100.0 * energy0) << "Forward Euler with too large a dt";
}

// EXAMPLE 104. The errors that are the caller's fault throw std::invalid_argument, and right away, even with zero
// steps (not after taking some steps): rho_c and k not > 0, dt not > 0, theta outside [0, 1] (NaN included: with
// a NaN the comparisons are all false, so one checks "if it is NOT inside"), T0 or f with a number of elements different
// from the number of nodes, fixed_values with a number different from fixed_nodes. The limit values (theta = 0,
// theta = 1) and a plate with no fixed nodes are instead valid.
TEST(Transient2D, RejectsInvalidScalarsAndSizes) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    {
        Call ok;
        EXPECT_NO_THROW(ok.run()) << "valid call";
        ok.theta = 0.0;
        EXPECT_NO_THROW(ok.run()) << "theta = 0";
        ok.theta = 1.0;
        EXPECT_NO_THROW(ok.run()) << "theta = 1";
        ok.fixed.clear();
        ok.values.clear();
        EXPECT_NO_THROW(ok.run()) << "no fixed node";
    }
    Call c;
    c.rho_c = 0.0;
    expect_invalid(c, "rho_c = 0");
    c = Call();
    c.rho_c = -1.0;
    expect_invalid(c, "rho_c < 0");
    c = Call();
    c.rho_c = nan;
    expect_invalid(c, "rho_c = NaN");
    c = Call();
    c.k = 0.0;
    expect_invalid(c, "k = 0");
    c = Call();
    c.k = -3.0;
    expect_invalid(c, "k < 0");
    c = Call();
    c.k = nan;
    expect_invalid(c, "k = NaN");
    c = Call();
    c.dt = 0.0;
    expect_invalid(c, "dt = 0");
    c = Call();
    c.dt = -0.01;
    expect_invalid(c, "dt < 0");
    c = Call();
    c.dt = nan;
    expect_invalid(c, "dt = NaN");
    c = Call();
    c.theta = -0.1;
    expect_invalid(c, "theta < 0");
    c = Call();
    c.theta = 1.1;
    expect_invalid(c, "theta > 1");
    c = Call();
    c.theta = nan;
    expect_invalid(c, "theta = NaN");
    c = Call();
    c.T0.pop_back();
    expect_invalid(c, "T0 with one number fewer");
    c = Call();
    c.T0.push_back(20.0);
    expect_invalid(c, "T0 with one number more");
    c = Call();
    c.f.pop_back();
    expect_invalid(c, "f with one number fewer");
    c = Call();
    c.f.push_back(1.0);
    expect_invalid(c, "f with one number more");
    c = Call();
    c.values.pop_back();
    expect_invalid(c, "one imposed value fewer");
    c = Call();
    c.values.push_back(5.0);
    expect_invalid(c, "one imposed value more");
}

// EXAMPLE 105. The wrong nodes. A fixed node that does not exist (9 with 9 nodes, or a huge number) or that appears twice
// (even with the same value) throws std::invalid_argument, even with zero steps: the matrices are built
// before the steps. And a node that is NOT fixed and is in no triangle (mesh with nodes 0 and 5 outside every
// triangle) has the row of M and of K all 0 and cannot be solved: invalid_argument, from the first step. If it is fixed
// the plate is computed and coincides with the oracle.
TEST(Transient2D, RejectsInvalidFixedNodesAndNodesOutsideEveryTriangle) {
    Call c;
    c.fixed = {0, 1, 2, 3, 5, 6, 7, 9};
    expect_invalid(c, "fixed node 9 out of 9 nodes");
    c = Call();
    c.fixed = {0, 1, 2, 3, 5, 6, 7, std::numeric_limits<std::size_t>::max()};
    expect_invalid(c, "huge fixed node");
    c = Call();
    c.fixed = {0, 1, 2, 3, 5, 6, 7, 7};
    expect_invalid(c, "fixed node twice, same value");
    c = Call();
    c.fixed = {0, 1, 2, 3, 5, 6, 7, 0};
    c.values = {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 8.0};
    expect_invalid(c, "fixed node twice, different values");

    Call d;
    d.mesh = mesh_with_nodes_outside_every_triangle();
    d.f = DVector(6, 1.0);
    d.T0 = DVector(6, 20.0);
    d.fixed = {1, 2, 3, 4};
    d.values = DVector(4, 5.0);
    d.n_steps = 3;
    EXPECT_THROW(d.run(), std::invalid_argument) << "nodes 0 and 5 free and outside every triangle";

    d.fixed = {0, 1, 5};
    d.values = {1.0, 2.0, 3.0};
    const DVector want =
        dense_transient(d.mesh, d.rho_c, d.k, d.f, d.fixed, d.values, d.T0, d.dt, d.n_steps, d.theta);
    expect_vector(d.run(), want, 1e-9, "nodes 0 and 5 fixed");
}

// EXAMPLE 106. A problem that the conjugate gradient does not solve: NaNs in the data (in the source, in T0 or in an
// imposed temperature) make the right-hand side a NaN and the CG does not converge. The function does not return NaNs
// as if nothing had happened: it throws std::runtime_error.
TEST(Transient2D, ThrowsWhenTheLinearSolverCannotConverge) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Call c;
    c.f[4] = nan;
    EXPECT_THROW(c.run(), std::runtime_error) << "NaN in the source";
    c = Call();
    c.T0[4] = nan;
    EXPECT_THROW(c.run(), std::runtime_error) << "NaN in T0";
    c = Call();
    c.values[0] = nan;
    EXPECT_THROW(c.run(), std::runtime_error) << "NaN in an imposed temperature";
}

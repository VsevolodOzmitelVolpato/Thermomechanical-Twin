#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>

#include "CGSolver.h"

// tridiag(-1, 2, -1) of size n, built from a dense matrix.
static CSR poisson1d(size_t n) {
    std::vector<std::vector<double>> M(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        M[i][i] = 2.0;
        if (i > 0)     M[i][i - 1] = -1.0;
        if (i + 1 < n) M[i][i + 1] = -1.0;
    }
    return CSR(M);
}

// -u'' = 1 on (0,1), u(0) = u(1) = 0 has the exact solution u = x(1-x)/2,
// a quadratic: the 3-point scheme is exact on quadratics, so the discrete
// solution equals u at the nodes. Any gap is a bug in the solver, not in the
// discretisation.
TEST(Cg, Poisson1dIsExactAtTheNodes) {
    const size_t n = 100;
    const double h = 1.0 / (n + 1);
    CSR A = poisson1d(n);
    DVector b(n, h * h), x0(n, 0.0);

    CGSolver solver(A, b, x0, 1e-12, static_cast<unsigned>(10 * n));
    solver.solve(false);
    DVector x = solver.get_solution();

    double max_error = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double xi = (i + 1) * h;
        max_error = std::max(max_error, std::fabs(x[i] - xi * (1.0 - xi) / 2.0));
    }
    EXPECT_LT(max_error, 1e-10);

    // A constant right-hand side only touches half of the eigenvectors,
    // so CG finishes in about n/2 iterations.
    EXPECT_GT(solver.get_iterations(), 0.4 * n);
    EXPECT_LT(solver.get_iterations(), 0.6 * n);
}

TEST(Cg, JacobiPcgGivesTheSameAsCgOnPoisson1d) {
    // [1] the problem
    const size_t n = 100;
    const double h = 1.0 / (n + 1);
    CSR A = poisson1d(n);
    DVector b(n, h * h), x0(n, 0.0);
    const unsigned max_it = static_cast<unsigned>(10 * n);

    // [2] two solvers with the SAME data
    CGSolver cg(A, b, x0, 1e-12, max_it);
    CGSolver pcg(A, b, x0, 1e-12, max_it);
    cg.solve(false);   // plain CG
    pcg.solve(true);   // CG with Jacobi

    // [3] same number of iterations
    EXPECT_EQ(cg.get_iterations(), pcg.get_iterations());

    // [4] same solution, node by node
    DVector x_cg = cg.get_solution();
    DVector x_pcg = pcg.get_solution();
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x_cg[i], x_pcg[i], 1e-12) << "node " << i;
    }
}


static double max_distance_from_ones(const DVector& x) {
    double e = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        e = std::max(e, std::fabs(x[i] - 1.0));
    }
    return e;
}

TEST(Cg, JacobiPcgNeedsFarFewerIterationsOnBadlyScaledMatrix) {
    // [1] the scaling: one number s[i] per point, from 1 to about 31.6
    const size_t n = 400;
    std::vector<double> s(n);
    for (size_t i = 0; i < n; ++i) {
        s[i] = std::pow(10.0, 1.5 * static_cast<double>(i) / static_cast<double>(n - 1));
    }

    // [2] the matrix B = S * A * S  (A = Poisson), filled in by hand
    std::vector<std::vector<double>> M(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        M[i][i] = s[i] * 2.0 * s[i];
        if (i > 0)     M[i][i - 1] = s[i] * (-1.0) * s[i - 1];
        if (i + 1 < n) M[i][i + 1] = s[i] * (-1.0) * s[i + 1];
    }
    CSR B(M);

    // [3] start from the ANSWER (all ones) and compute the right-hand side b = B * 1
    DVector ones(n, 1.0), x0(n, 0.0);
    DVector b = B * ones;
    const unsigned max_it = static_cast<unsigned>(10 * n);

    // [4] two solvers, one without and one with Jacobi
    CGSolver cg(B, b, x0, 1e-12, max_it);
    CGSolver pcg(B, b, x0, 1e-12, max_it);
    cg.solve(false);
    pcg.solve(true);
    std::cout << "CG: " << cg.get_iterations() << " iterations, PCG: "
              << pcg.get_iterations() << " iterations" << std::endl;

    // [5] checks
    EXPECT_LT(4 * pcg.get_iterations(), cg.get_iterations());
    EXPECT_LT(max_distance_from_ones(cg.get_solution()), 1e-8);
    EXPECT_LT(max_distance_from_ones(pcg.get_solution()), 1e-8);
}

TEST(Cg, WrongInputsThrow) {
    CSR A = poisson1d(10);   // 10x10 matrix, symmetric: fine

    // [1] b too short: 9 numbers instead of 10
    EXPECT_THROW(CGSolver(A, DVector(9, 1.0), DVector(10, 0.0), 1e-10, 100), std::invalid_argument);

    // [2] x0 too short: 9 numbers instead of 10
    EXPECT_THROW(CGSolver(A, DVector(10, 1.0), DVector(9, 0.0), 1e-10, 100), std::invalid_argument);

    // [3] NON-symmetric matrix: CG cannot work
    std::vector<std::vector<double>> N = {{2, -1, 0}, {-3, 2, 0}, {0, 0, 1}};
    CSR not_symmetric(N);
    EXPECT_THROW(CGSolver(not_symmetric, DVector(3, 1.0), DVector(3, 0.0), 1e-10, 100), std::invalid_argument);
}

// ---- The solver reports whether it has converged -------------------------------------

// A problem CG solves without any trouble: 1D Poisson with 100 unknowns and 10*n iterations available
// (the same as in the first test). Both plain CG and CG with Jacobi must report "convergence".
TEST(Cg, ReportsConvergence) {
    const size_t n = 100;
    const double h = 1.0 / (n + 1);
    CSR A = poisson1d(n);
    DVector b(n, h * h), x0(n, 0.0);
    const unsigned max_it = static_cast<unsigned>(10 * n);

    CGSolver cg(A, b, x0, 1e-12, max_it);
    CGSolver pcg(A, b, x0, 1e-12, max_it);
    cg.solve(false);
    pcg.solve(true);

    EXPECT_TRUE(cg.has_converged());
    EXPECT_TRUE(pcg.has_converged());
}

// The same problem with only 5 iterations: with 100 unknowns CG cannot get to 1e-12.
// The solver stops because the iterations run out (it does exactly 5) and it must say so.
TEST(Cg, ReportsFailureWhenTheIterationsRunOut) {
    const size_t n = 100;
    const double h = 1.0 / (n + 1);
    CSR A = poisson1d(n);
    DVector b(n, h * h), x0(n, 0.0);

    CGSolver cg(A, b, x0, 1e-12, 5);
    CGSolver pcg(A, b, x0, 1e-12, 5);
    cg.solve(false);
    pcg.solve(true);

    EXPECT_EQ(cg.get_iterations(), 5u);
    EXPECT_EQ(pcg.get_iterations(), 5u);
    EXPECT_FALSE(cg.has_converged());
    EXPECT_FALSE(pcg.has_converged());
}

// The answer depends on the RESIDUAL at the end, not on the number of iterations used.
// First find out how many iterations K are really needed (reference computation). Then solve the
// same problem with max_it = K: the iterations run out exactly when the residual drops below the
// threshold, so this is convergence. With max_it = K - 1 the last iteration is missing: it is not.
TEST(Cg, ConvergenceDependsOnTheResidualNotOnTheIterationCount) {
    const size_t n = 100;
    const double h = 1.0 / (n + 1);
    CSR A = poisson1d(n);
    DVector b(n, h * h), x0(n, 0.0);

    CGSolver reference(A, b, x0, 1e-12, static_cast<unsigned>(10 * n));
    reference.solve(false);
    ASSERT_TRUE(reference.has_converged());
    const unsigned K = reference.get_iterations();
    ASSERT_GT(K, 1u);

    CGSolver just_in_time(A, b, x0, 1e-12, K);
    just_in_time.solve(false);
    EXPECT_EQ(just_in_time.get_iterations(), K);
    EXPECT_TRUE(just_in_time.has_converged());

    CGSolver one_short(A, b, x0, 1e-12, K - 1);
    one_short.solve(false);
    EXPECT_EQ(one_short.get_iterations(), K - 1);
    EXPECT_FALSE(one_short.has_converged());
}

// A NaN in the right-hand side: the residual is NaN, and a NaN is neither > nor <= any number.
// The CG loop does not even start (0 iterations) and the solution is still the initial guess.
// The solver must NOT report "convergence": a meaningless result is not a result.
TEST(Cg, NaNInTheRightHandSideIsNotConvergence) {
    CSR A = poisson1d(10);
    DVector b(10, 1.0), x0(10, 0.0);
    b[3] = std::nan("");

    CGSolver cg(A, b, x0, 1e-10, 100);
    CGSolver pcg(A, b, x0, 1e-10, 100);
    cg.solve(false);
    pcg.solve(true);

    EXPECT_FALSE(cg.has_converged());
    EXPECT_FALSE(pcg.has_converged());
}

// b = 0 and starting point x0 = 0: the solution is exactly x = 0 and the residual is 0 from the start.
// The criterion |r| <= tol |b| becomes 0 <= 0, which is true: convergence at 0 iterations.
// (With "<" instead of "<=" the solver would say it has not converged, on an already solved problem.)
TEST(Cg, ZeroRightHandSideWithZeroGuessConvergesAtOnce) {
    CSR A = poisson1d(10);
    DVector b(10, 0.0), x0(10, 0.0);

    CGSolver cg(A, b, x0, 1e-12, 100);
    cg.solve(false);

    EXPECT_EQ(cg.get_iterations(), 0u);
    EXPECT_TRUE(cg.has_converged());
    for (double xi : cg.get_solution()) {
        EXPECT_EQ(xi, 0.0);
    }
}

// The tolerance is RELATIVE to |b|. Here b is 1e-14 times that of the first test, so |b| is about
// 1e-17: the initial residual is already smaller than 1e-12 in absolute value, but relative to |b|
// it equals 1, i.e. it has not decreased at all. With 5 iterations the solver cannot say "convergence";
// with 10*n iterations it does get there (CG behaves the same way at any scale).
TEST(Cg, ConvergenceIsRelativeToTheSizeOfB) {
    const size_t n = 100;
    const double h = 1.0 / (n + 1);
    CSR A = poisson1d(n);
    DVector b(n, 1e-14 * h * h), x0(n, 0.0);

    CGSolver too_few(A, b, x0, 1e-12, 5);
    too_few.solve(false);
    EXPECT_FALSE(too_few.has_converged());

    CGSolver enough(A, b, x0, 1e-12, static_cast<unsigned>(10 * n));
    enough.solve(false);
    EXPECT_TRUE(enough.has_converged());
}

TEST(Jacobi, ZeroDiagonalAndWrongSizeThrow) {
    // [1] a 0 on the diagonal: Jacobi would have to divide by 0
    std::vector<std::vector<double>> Z = {{0, 1}, {1, 2}};
    CSR zero_diag(Z);
    EXPECT_THROW(JacobiPreconditioner jac(zero_diag), std::invalid_argument);

    // [2] apply() with a vector of the wrong length (3 instead of 4)
    JacobiPreconditioner J(poisson1d(4));
    EXPECT_THROW(J.apply(DVector(3, 1.0)), std::invalid_argument);
}

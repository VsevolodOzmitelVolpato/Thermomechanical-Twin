#include <gtest/gtest.h>

#include <vector>
#include <cmath>
#include "Fem1D.h"


TEST(Fem1d, DirichletWrongInputsThrow) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DMatrix K = assemble_stiffness(x, 2.0);
    DVector F = assemble_load(x, 4.0);
    DVector F_short = {1.0, 2.0};    // 2 numbers instead of 3

    EXPECT_THROW(apply_dirichlet(K, F_short, 0, 10.0), std::invalid_argument);
    EXPECT_THROW(apply_dirichlet(K, F, 3, 10.0), std::invalid_argument);   // the nodes are 0, 1, 2
}

TEST(Fem1d, UniformMeshHasTheRightNodes) {
    std::vector<double> x = uniform_mesh(2.0, 4);   // 4 elements between 0 and 2
    ASSERT_EQ(x.size(), 5u);                         // so 5 nodes
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(x[i], 0.5 * i, 1e-14) << "node " << i;
    }
}

// Two elements of length 1 and 2: nodes 0, 1, 3. Conductivity k = 2.
TEST(Fem1d, StiffnessMatchesHandComputation) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DMatrix K = assemble_stiffness(x, 2.0);

    DMatrix expected = {{ 2.0, -2.0,  0.0},
                        {-2.0,  3.0, -1.0},
                        { 0.0, -1.0,  1.0}};
    ASSERT_EQ(K.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(K[i].size(), 3u);
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(K[i][j], expected[i][j], 1e-14) << "K[" << i << "][" << j << "]";
        }
    }
}

// Same nodes, source q = 4.
TEST(Fem1d, LoadMatchesHandComputation) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DVector F = assemble_load(x, 4.0);

    DVector expected = {2.0, 6.0, 4.0};
    ASSERT_EQ(F.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(F[i], expected[i], 1e-14) << "F[" << i << "]";
    }
}

static void expect_matrix_near(const DMatrix& A, const DMatrix& B, double tol) {
    ASSERT_EQ(A.size(), B.size());
    for (size_t i = 0; i < A.size(); ++i) {
        ASSERT_EQ(A[i].size(), B[i].size());
        for (size_t j = 0; j < A[i].size(); ++j) {
            EXPECT_NEAR(A[i][j], B[i][j], tol) << "K[" << i << "][" << j << "]";
        }
    }
}

static void expect_vector_near(const DVector& a, const DVector& b, double tol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << "F[" << i << "]";
    }
}

// Same example as before: x = {0, 1, 3}, k = 2, q = 4. Node 0 is fixed at 10.
TEST(Fem1d, DirichletOnFirstNode) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DMatrix K = assemble_stiffness(x, 2.0);
    DVector F = assemble_load(x, 4.0);

    apply_dirichlet(K, F, 0, 10.0);

    expect_matrix_near(K, {{1.0,  0.0,  0.0},
                           {0.0,  3.0, -1.0},
                           {0.0, -1.0,  1.0}}, 1e-14);
    expect_vector_near(F, {10.0, 26.0, 4.0}, 1e-14);
}

// Same K and F, but the middle node (index 1) is fixed at 5.
TEST(Fem1d, DirichletOnAMiddleNode) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DMatrix K = assemble_stiffness(x, 2.0);
    DVector F = assemble_load(x, 4.0);

    apply_dirichlet(K, F, 1, 5.0);

    expect_matrix_near(K, {{2.0, 0.0, 0.0},
                           {0.0, 1.0, 0.0},
                           {0.0, 0.0, 1.0}}, 1e-14);
    expect_vector_near(F, {12.0, 5.0, 9.0}, 1e-14);
}

// Both ends are fixed: T(0) = 10 and T(3) = 40. K becomes diagonal and the
// middle node is solved by hand. The exact solution of -k T'' = q is
// T(x) = -x^2 + 13 x + 10, so T(1) = 22.
TEST(Fem1d, DirichletOnBothEndsGivesTheExactMiddleTemperature) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DMatrix K = assemble_stiffness(x, 2.0);
    DVector F = assemble_load(x, 4.0);

    apply_dirichlet(K, F, 0, 10.0);
    apply_dirichlet(K, F, 2, 40.0);

    expect_matrix_near(K, {{1.0, 0.0, 0.0},
                           {0.0, 3.0, 0.0},
                           {0.0, 0.0, 1.0}}, 1e-14);
    expect_vector_near(F, {10.0, 66.0, 40.0}, 1e-14);
    EXPECT_NEAR(F[1] / K[1][1], 22.0, 1e-14);
}

static double exact_steady(double x, double L, double k, double q, double a, double b) {
    return a + (b - a) * x / L + q * x * (L - x) / (2.0 * k);
}

// Solves on the given nodes and compares with the exact solution at every node.
// (In 1D, with constant k and q, linear elements give the EXACT values at the nodes, on any
// mesh: that is why the tolerance is so tight and a red test means an error in the code.)
// If something does not match it prints a single message: the maximum error and where it is.
static void check_against_exact(const std::vector<double>& nodes, double k, double q,
                                double a, double b, double tol) {
    DVector T = solve_steady(nodes, k, q, a, b);
    ASSERT_EQ(T.size(), nodes.size());
    const double L = nodes.back() - nodes.front();
    double max_err = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const double err = std::fabs(T[i] - exact_steady(nodes[i], L, k, q, a, b));
        if (err > max_err) {
            max_err = err;
            worst = i;
        }
    }
    EXPECT_LT(max_err, tol) << "maximum error at node " << worst << " (x = " << nodes[worst]
                            << "): T = " << T[worst] << ", exact = "
                            << exact_steady(nodes[worst], L, k, q, a, b);
}

// q = 0: no source, the temperature is a straight line between the two ends (10 and 40).
TEST(Fem1d, SteadyLineBetweenTwoTemperatures) {
    check_against_exact(uniform_mesh(2.0, 20), 3.0, 0.0, 10.0, 40.0, 1e-9);
}

// Ends at 0 and uniform source: parabola. Uniform mesh.
TEST(Fem1d, SteadyParabolaOnUniformMesh) {
    check_against_exact(uniform_mesh(2.0, 50), 3.0, 5.0, 0.0, 0.0, 1e-9);
}

// Everything together (source + different ends) on a NON-uniform mesh: x_i = L (i/N)^1.5.
// The nodes are denser near x = 0.
TEST(Fem1d, SteadySourceAndEndsOnGradedMesh) {
    const size_t N = 40;
    const double L = 2.0;
    std::vector<double> x(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = L * std::pow(static_cast<double>(i) / N, 1.5);
    }
    check_against_exact(x, 3.0, 5.0, 10.0, 40.0, 1e-9);
}

TEST(Fem1d, MassMatchesHandComputation) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DMatrix M = assemble_mass(x, 6.0);

    DMatrix expected = {{2.0, 1.0, 0.0},
                        {1.0, 6.0, 2.0},
                        {0.0, 2.0, 4.0}};
    ASSERT_EQ(M.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(M[i].size(), 3u);
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(M[i][j], expected[i][j], 1e-14) << "M[" << i << "][" << j << "]";
        }
    }
}

// On a NON-uniform mesh: M is symmetric and the sum of all its numbers is the
// total thermal capacity of the bar, rho_c * L.
TEST(Fem1d, MassIsSymmetricAndSumsToTotalHeatCapacity) {
    const size_t N = 40;
    const double L = 1.7;
    const double rho_c = 2.5;
    std::vector<double> x(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = L * std::pow(static_cast<double>(i) / N, 1.5);
    }

    DMatrix M = assemble_mass(x, rho_c);

    ASSERT_EQ(M.size(), N + 1);
    double sum = 0.0;
    for (size_t i = 0; i < M.size(); ++i) {
        ASSERT_EQ(M[i].size(), N + 1);
        for (size_t j = 0; j < M.size(); ++j) {
            sum += M[i][j];
            EXPECT_NEAR(M[i][j], M[j][i], 1e-14) << "M[" << i << "][" << j << "]";
        }
    }
    EXPECT_NEAR(sum, rho_c * L, 1e-12);
}

// ---- Evolution in time (theta scheme) ------------------------------------------------

// Amplification factor of the theta scheme on the mode sin(pi x / L) of a uniform mesh.
// With T(0) = T(L) = 0 and q = 0, that vector is an eigenvector of both K and M:
//   lambda_K = (k/h) (2 - 2 cos(pi h / L)),   lambda_M = (rho_c h / 6) (4 + 2 cos(pi h / L)).
// So at every step the temperature is multiplied by
//   g = (lambda_M - (1 - theta) dt lambda_K) / (lambda_M + theta dt lambda_K).
static double amplification_factor(double h, double L, double k, double rho_c, double dt, double theta) {
    const double pi = std::acos(-1.0);
    const double c = std::cos(pi * h / L);
    const double lambda_K = (k / h) * (2.0 - 2.0 * c);
    const double lambda_M = (rho_c * h / 6.0) * (4.0 + 2.0 * c);
    return (lambda_M - (1.0 - theta) * dt * lambda_K) / (lambda_M + theta * dt * lambda_K);
}

// The result must coincide with g^n * sin(pi x / L) at every node. There is no discretisation
// error to justify: it is the EXACT solution of the discrete scheme, so 1e-9 holds.
// Three values of theta are tried, also one different from 0.5 and from 1 (so theta cannot be swapped
// with 1 - theta without the test noticing).
TEST(Fem1d, TransientMatchesTheAmplificationFactor) {
    const double pi = std::acos(-1.0);
    const size_t N = 20;
    const double L = 1.0, k = 2.0, rho_c = 3.0, dt = 0.01;
    const size_t n_steps = 10;
    std::vector<double> x = uniform_mesh(L, N);
    DVector T0(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        T0[i] = std::sin(pi * x[i] / L);
    }

    for (double theta : {1.0, 0.5, 0.3}) {
        DVector T = solve_transient(x, rho_c, k, 0.0, 0.0, 0.0, T0, dt, n_steps, theta);
        ASSERT_EQ(T.size(), N + 1);
        const double g = amplification_factor(L / N, L, k, rho_c, dt, theta);
        double max_err = 0.0;
        for (size_t i = 0; i <= N; ++i) {
            const double err = std::fabs(T[i] - std::pow(g, n_steps) * std::sin(pi * x[i] / L));
            if (err > max_err) {
                max_err = err;
            }
        }
        EXPECT_LT(max_err, 1e-9) << "theta = " << theta << " (g = " << g << ")";
    }
}

// The steady state is a fixed point of the scheme: if you start from a T with K T = F (ends
// included), after one step, of any size and with any theta, you stay where you are.
// Non-uniform mesh, different source and ends: checks F, the dt*F term and the ends.
TEST(Fem1d, TransientKeepsTheSteadyState) {
    const size_t N = 40;
    const double L = 2.0, k = 3.0, rho_c = 1.5, q = 5.0, T_left = 10.0, T_right = 40.0;
    std::vector<double> x(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = L * std::pow(static_cast<double>(i) / N, 1.5);
    }
    DVector T_steady = solve_steady(x, k, q, T_left, T_right);

    for (double theta : {0.5, 1.0}) {
        DVector T = solve_transient(x, rho_c, k, q, T_left, T_right, T_steady, 0.37, 1, theta);
        ASSERT_EQ(T.size(), N + 1);
        double max_diff = 0.0;
        for (size_t i = 0; i <= N; ++i) {
            const double diff = std::fabs(T[i] - T_steady[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
        EXPECT_LT(max_diff, 1e-9) << "theta = " << theta;
    }
}

// Start from a uniform temperature (20) different from the ends (10 and 40): over time the
// steady state is reached, and the ends stay fixed at every step.
// theta = 1 (backward Euler): it damps even the fast modes, so 100 steps are enough.
TEST(Fem1d, TransientRelaxesToTheSteadyState) {
    const size_t N = 40;
    const double L = 2.0, k = 3.0, rho_c = 1.5, q = 5.0, T_left = 10.0, T_right = 40.0;
    std::vector<double> x = uniform_mesh(L, N);
    DVector T0(N + 1, 20.0);

    DVector T = solve_transient(x, rho_c, k, q, T_left, T_right, T0, 0.1, 100, 1.0);   // t = 10
    DVector T_steady = solve_steady(x, k, q, T_left, T_right);

    ASSERT_EQ(T.size(), N + 1);
    EXPECT_NEAR(T.front(), T_left, 1e-9);
    EXPECT_NEAR(T.back(), T_right, 1e-9);
    double max_diff = 0.0;
    for (size_t i = 0; i <= N; ++i) {
        const double diff = std::fabs(T[i] - T_steady[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    EXPECT_LT(max_diff, 1e-7);
}

// Wrong inputs: they must throw an exception.
TEST(Fem1d, TransientWrongInputsThrow) {
    std::vector<double> x = uniform_mesh(1.0, 4);    // 5 nodes
    DVector T0(5, 0.0);
    DVector T0_short(4, 0.0);                        // 4 values instead of 5

    EXPECT_THROW(solve_transient(x, 1.0, 1.0, 0.0, 0.0, 0.0, T0_short, 0.1, 1, 0.5), std::invalid_argument);
    EXPECT_THROW(solve_transient(x, 1.0, 1.0, 0.0, 0.0, 0.0, T0, 0.0, 1, 0.5), std::invalid_argument);   // dt = 0
    EXPECT_THROW(solve_transient(x, 1.0, 1.0, 0.0, 0.0, 0.0, T0, 0.1, 1, 1.5), std::invalid_argument);   // theta > 1
}

// ---- Verification: L2 error and order of convergence --------------------------------

// The error is the piecewise linear function that at the nodes is e_i = T_num[i] - T_ref[i].
// Its L2 norm is sqrt(integral of e^2) = sqrt(e^T M e), with M the mass matrix (rho_c = 1).

// Two hand cases.
//  1) "Hat": e = (0, 1, 0) on 2 elements of length 1. The function goes up from 0 to 1 and back down,
//     so integral of e^2 = 1/3 + 1/3 = 2/3.
//  2) e = (0, 1, 1, 0) on 3 elements of length 1: it goes up, stays at 1, goes back down.
//     integral of e^2 = 1/3 + 1 + 1/3 = 5/3. (Here the off-diagonal terms of M count too.)
TEST(Fem1d, L2ErrorMatchesHandComputation) {
    {
        std::vector<double> x = uniform_mesh(2.0, 2);
        DVector T_num = {0.0, 1.0, 0.0};
        DVector T_ref = {0.0, 0.0, 0.0};
        EXPECT_NEAR(l2_error(x, T_num, T_ref), std::sqrt(2.0 / 3.0), 1e-12);
    }
    {
        std::vector<double> x = uniform_mesh(3.0, 3);
        DVector T_num = {0.0, 1.0, 1.0, 0.0};
        DVector T_ref = {0.0, 0.0, 0.0, 0.0};
        EXPECT_NEAR(l2_error(x, T_num, T_ref), std::sqrt(5.0 / 3.0), 1e-12);
    }
}

// On a NON-uniform mesh, of length 2:
//  - constant error 0.5: integral = 0.5^2 * 2, so the norm is 0.5 * sqrt(2);
//  - error e(x) = x (the values at the nodes are the x_i): the piecewise linear function is exactly the
//    line e(x) = x, so integral of x^2 over [0, 2] = 8/3, whatever the mesh.
TEST(Fem1d, L2ErrorOfConstantAndLinearFieldsOnAGradedMesh) {
    const size_t N = 10;
    std::vector<double> x(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = 2.0 * std::pow(static_cast<double>(i) / N, 1.5);
    }
    DVector zero(N + 1, 0.0);
    DVector constant(N + 1, 0.5);

    EXPECT_NEAR(l2_error(x, constant, zero), 0.5 * std::sqrt(2.0), 1e-12);
    EXPECT_NEAR(l2_error(x, x, zero), std::sqrt(8.0 / 3.0), 1e-12);
}

// Two arbitrary fields (neither of them zero) on a non-uniform mesh. It is compared with
// the integral computed element by element: on an element of length h, with errors ea (on the left)
// and eb (on the right), the integral of e^2 is h (ea^2 + ea eb + eb^2) / 3.
// In addition: equal fields give error 0, and swapping the two fields does not change the result.
TEST(Fem1d, L2ErrorMatchesTheElementByElementIntegral) {
    const size_t N = 8;
    std::vector<double> x(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = 3.0 * std::pow(static_cast<double>(i) / N, 2.0);
    }
    DVector a = {1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0, 6.0, 9.0};
    DVector b = {3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0};

    double integral = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const double h = x[i + 1] - x[i];
        const double ea = a[i] - b[i];
        const double eb = a[i + 1] - b[i + 1];
        integral += h * (ea * ea + ea * eb + eb * eb) / 3.0;
    }

    EXPECT_NEAR(l2_error(x, a, b), std::sqrt(integral), 1e-12);
    EXPECT_NEAR(l2_error(x, b, a), std::sqrt(integral), 1e-12);
    EXPECT_NEAR(l2_error(x, a, a), 0.0, 1e-15);
}

// A field with a number of values different from the number of nodes: exception.
TEST(Fem1d, L2ErrorWrongInputsThrow) {
    std::vector<double> x = uniform_mesh(1.0, 4);   // 5 nodes
    DVector ok(5, 0.0);
    DVector too_short(4, 0.0);

    EXPECT_THROW(l2_error(x, too_short, ok), std::invalid_argument);
    EXPECT_THROW(l2_error(x, ok, too_short), std::invalid_argument);
    EXPECT_THROW(l2_error(x, too_short, too_short), std::invalid_argument);   // both wrong
}

// If the error goes like C h^p, when h is divided by "ratio" the error is divided by ratio^p.
// Hand examples:
//   0.04 -> 0.01 with ratio 2:  4 = 2^p, so p = 2
//   0.08 -> 0.04 with ratio 2:  2 = 2^p, so p = 1
//   0.09 -> 0.01 with ratio 3:  9 = 3^p, so p = 2
//   error that does not change:     p = 0 (refining is useless)
//   error that GROWS (0.01 -> 0.04): p = -2 (the fine computation is worse)
TEST(Fem1d, ObservedOrderMatchesHandComputation) {
    EXPECT_NEAR(observed_order(0.04, 0.01, 2.0), 2.0, 1e-12);
    EXPECT_NEAR(observed_order(0.08, 0.04, 2.0), 1.0, 1e-12);
    EXPECT_NEAR(observed_order(0.09, 0.01, 3.0), 2.0, 1e-12);
    EXPECT_NEAR(observed_order(0.05, 0.05, 2.0), 0.0, 1e-12);
    EXPECT_NEAR(observed_order(0.01, 0.04, 2.0), -2.0, 1e-12);
}

// The logarithm of a zero or negative error does not exist, and "ratio" must be > 1
// (it is the ratio between the coarse step and the fine one).
TEST(Fem1d, ObservedOrderWrongInputsThrow) {
    EXPECT_THROW(observed_order(0.0, 0.01, 2.0), std::invalid_argument);     // zero coarse error
    EXPECT_THROW(observed_order(0.04, 0.0, 2.0), std::invalid_argument);     // zero fine error
    EXPECT_THROW(observed_order(0.04, -0.01, 2.0), std::invalid_argument);   // negative error
    EXPECT_THROW(observed_order(0.04, 0.01, 1.0), std::invalid_argument);    // ratio = 1
    EXPECT_THROW(observed_order(0.04, 0.01, 0.5), std::invalid_argument);    // ratio < 1
}

// ---- Verification problem: the decaying sine ----------------------------------------

// The result of decaying_sine_error must coincide with the L2 error between
//  - the EXACT SOLUTION OF THE SCHEME, g^n sin(pi x_i) (see amplification_factor), and
//  - the exact solution of the problem, exp(-pi^2 t) sin(pi x_i).
// This way T0, k, rho_c, L, the final time t = n_steps * dt, theta and the exponent are checked together.
TEST(Fem1d, DecayingSineErrorMatchesTheDiscreteSolution) {
    const double pi = std::acos(-1.0);
    const size_t N = 20, n_steps = 10;
    const double dt = 0.01, t = n_steps * dt;
    std::vector<double> x = uniform_mesh(1.0, N);

    for (double theta : {1.0, 0.5}) {
        const double g = amplification_factor(1.0 / N, 1.0, 1.0, 1.0, dt, theta);
        DVector T_scheme(N + 1), T_exact(N + 1);
        for (size_t i = 0; i <= N; ++i) {
            T_scheme[i] = std::pow(g, n_steps) * std::sin(pi * x[i]);
            T_exact[i] = std::exp(-pi * pi * t) * std::sin(pi * x[i]);
        }
        EXPECT_NEAR(decaying_sine_error(N, dt, n_steps, theta), l2_error(x, T_scheme, T_exact), 1e-9)
            << "theta = " << theta;
    }
}

// At time 0 the initial temperature is already the exact solution: zero error.
TEST(Fem1d, DecayingSineErrorIsZeroAtTimeZero) {
    EXPECT_NEAR(decaying_sine_error(10, 0.01, 0, 1.0), 0.0, 1e-14);
}

// No element: exception. The other two cases are already checked by solve_transient.
TEST(Fem1d, DecayingSineErrorWrongInputsThrow) {
    EXPECT_THROW(decaying_sine_error(0, 0.01, 1, 1.0), std::invalid_argument);    // 0 elements
    EXPECT_THROW(decaying_sine_error(10, 0.0, 1, 1.0), std::invalid_argument);    // dt = 0
    EXPECT_THROW(decaying_sine_error(10, 0.01, 1, 2.0), std::invalid_argument);   // theta > 1
}

// ---- Orders of convergence -----------------------------------------------------------
// These three tests do not look for new errors (the previous tests have already caught those):
// they document that the solver has the order of accuracy predicted by the theory.

// SPACE. Crank-Nicolson with a very small time step (the error in time is negligible),
// up to t = 0.1; the mesh goes from 10 to 160 elements, halving h each time. Expected: order 2.
TEST(Fem1d, SpatialConvergenceIsSecondOrder) {
    const double dt = 1e-4;
    const size_t n_steps = 1000;   // t = 0.1
    double previous = decaying_sine_error(10, dt, n_steps, 0.5);
    for (size_t N : {20, 40, 80, 160}) {
        const double current = decaying_sine_error(N, dt, n_steps, 0.5);
        EXPECT_NEAR(observed_order(previous, current, 2.0), 2.0, 0.05) << "N = " << N / 2 << " -> " << N;
        previous = current;
    }
}

// TIME. Very fine mesh, N = 400: the mesh error (about 1e-6) is much smaller than the temporal one
// (at least 5e-5), so only the error in time is measured. Up to t = 0.1 with 5, 10, 20 steps
// (dt = 0.02, 0.01, 0.005), halving dt each time.
// Backward Euler (theta = 1): order 1.
TEST(Fem1d, TemporalConvergenceOfBackwardEulerIsFirstOrder) {
    const size_t N = 400;
    double previous = decaying_sine_error(N, 0.1 / 5, 5, 1.0);
    for (size_t steps : {10, 20}) {
        const double current = decaying_sine_error(N, 0.1 / steps, steps, 1.0);
        EXPECT_NEAR(observed_order(previous, current, 2.0), 1.0, 0.1) << steps / 2 << " -> " << steps << " steps";
        previous = current;
    }
}

// Crank-Nicolson (theta = 0.5): order 2.
TEST(Fem1d, TemporalConvergenceOfCrankNicolsonIsSecondOrder) {
    const size_t N = 400;
    double previous = decaying_sine_error(N, 0.1 / 5, 5, 0.5);
    for (size_t steps : {10, 20}) {
        const double current = decaying_sine_error(N, 0.1 / steps, steps, 0.5);
        EXPECT_NEAR(observed_order(previous, current, 2.0), 2.0, 0.1) << steps / 2 << " -> " << steps << " steps";
        previous = current;
    }
}

// ---- If CG does not converge, the solvers say so -------------------------------------

// A NaN in the data (source or imposed temperature) reaches the right-hand side of the system, and CG
// cannot converge. Before, these cases returned an arbitrary temperature (zeros) with no
// error. Now they must throw std::runtime_error: the arguments were acceptable, it is the computation
// that failed (invalid_argument is for wrong arguments, which are seen before computing).
TEST(Fem1d, SteadyThrowsWhenTheLinearSolverCannotConverge) {
    const double not_a_number = std::nan("");
    std::vector<double> x = uniform_mesh(1.0, 10);

    EXPECT_THROW(solve_steady(x, 1.0, not_a_number, 0.0, 0.0), std::runtime_error);   // NaN source
    EXPECT_THROW(solve_steady(x, 1.0, 5.0, not_a_number, 0.0), std::runtime_error);   // NaN left end
    EXPECT_THROW(solve_steady(x, 1.0, 5.0, 0.0, not_a_number), std::runtime_error);   // NaN right end
}

// The same for the evolution in time. ONE step with a NaN, in any of the data, is enough, and it must
// throw: with n_steps = 1 the test also checks that the first step is verified.
TEST(Fem1d, TransientThrowsWhenTheLinearSolverCannotConverge) {
    const double not_a_number = std::nan("");
    std::vector<double> x = uniform_mesh(1.0, 10);
    DVector T0(11, 1.0);
    DVector T0_with_nan = T0;
    T0_with_nan[4] = not_a_number;

    // arguments: x, rho_c, k, q, T_left, T_right, T0, dt, n_steps, theta
    EXPECT_THROW(solve_transient(x, 1.0, 1.0, not_a_number, 0.0, 0.0, T0, 0.01, 1, 1.0), std::runtime_error);
    EXPECT_THROW(solve_transient(x, 1.0, 1.0, 0.0, not_a_number, 0.0, T0, 0.01, 1, 1.0), std::runtime_error);
    EXPECT_THROW(solve_transient(x, 1.0, 1.0, 0.0, 0.0, not_a_number, T0, 0.01, 1, 1.0), std::runtime_error);
    EXPECT_THROW(solve_transient(x, 1.0, 1.0, 0.0, 0.0, 0.0, T0_with_nan, 0.01, 1, 1.0), std::runtime_error);
}

// The realistic case: an unstable scheme. With theta < 0.5 and dt above the stability limit the
// temperatures grow at every step, until they leave the scale of doubles. Before, in several of these
// computations, the function returned a vector of NaN and inf as if nothing had happened. Now, for every
// combination, one of two things happens: it throws std::runtime_error, or it returns only
// finite numbers. The test does NOT say that the result is right (an unstable computation can give huge
// but finite numbers): it says that NaN and inf never pass silently.
TEST(Fem1d, TransientNeverReturnsNonFiniteNumbersInSilence) {
    const double pi = std::acos(-1.0);
    for (double theta : {0.0, 0.3}) {
        for (size_t N : {10u, 20u}) {
            for (double dt : {0.1, 1.0, 5.0}) {
                std::vector<double> x = uniform_mesh(1.0, N);
                DVector T0(N + 1);
                for (size_t i = 0; i <= N; ++i) {
                    T0[i] = std::sin(pi * x[i]);
                }
                try {
                    DVector T = solve_transient(x, 1.0, 1.0, 0.0, 0.0, 0.0, T0, dt, 400, theta);
                    bool all_finite = true;
                    for (double v : T) {
                        all_finite = all_finite && std::isfinite(v);
                    }
                    EXPECT_TRUE(all_finite) << "theta = " << theta << ", N = " << N << ", dt = " << dt;
                } catch (const std::runtime_error&) {
                    // fine: the computation failed and says so
                }
            }
        }
    }
}

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "Fem1D.h"
#include "Verification2D.h"

// ---- The decaying sine in 2D: verification of the theta scheme ------------------------------------------------
//
// In 1D we checked the theta scheme with a problem whose exact solution is known, the decaying sine
// (decaying_sine_error). Here we do the same for the plate: decaying_sine_error_2d(n, dt, n_steps, theta).
//
// THE PROBLEM. The square [0, 1] x [0, 1], rho_c = 1, k = 1, no source, T = 0 on the whole boundary and
// initial temperature T0(x, y) = sin(pi x) sin(pi y). The heat equation is  T_t = T_xx + T_yy  and for this T0
//
//     T_xx + T_yy = -2 pi^2 T0
//
// so the shape stays that of the sine and only the amplitude decays:
//
//     T(x, y, t) = exp(-2 pi^2 t) sin(pi x) sin(pi y)                              (2 pi^2 = 19.74...).
//
// The function solves with the mesh of n x n cells and n_steps steps of size dt, and returns the L2 error
// (l2_error_2d) with respect to this solution, taken at the nodes, at the final time t = n_steps * dt.
//
// THE ORACLES. In 1D the sine was an eigenvector of the matrices, and by hand one found the EXACT SOLUTION OF THE SCHEME
// (g^n sin(pi x), amplification_factor). On the triangle mesh the mass matrix links each node to two diagonal neighbors
// (and not to the other two), so the sine is no longer an eigenvector and for general n there is no
// closed formula. The right answers come from:
//   - the 2 x 2 mesh, which has a single unknown and therefore a closed formula (EXAMPLES 109 and 110);
//   - numbers computed along another route, with full matrices and without conjugate gradient (EXAMPLE 111);
//   - the convergence orders, in space and in time (EXAMPLES 112, 113 and 114).
//
// TWO ERRORS THAT ADD UP. With respect to the exact solution the method is wrong for two reasons: the mesh (an error E_h
// that goes like h^2 and remains even with dt -> 0) and the time step (an error E_dt that goes like dt^p, with p = 1 for
// backward Euler and p = 2 for Crank-Nicolson). With a single decaying sine the two add up:
//
//     error = E_h + E_dt,        E_h does not depend on dt.
//
// For Crank-Nicolson E_h and E_dt have the same sign (the computed temperature is below the exact one: it decays
// too fast). For backward Euler E_dt has the opposite sign (it is above: it decays too slowly).
//
// HOW THE ORDER IN TIME IS MEASURED. In 1D a very fine mesh was enough (N = 400), so E_h was negligible. In 2D such a
// fine mesh costs too much (at n = 64, without optimizations, one step takes about 0.1 s) and the ratio
// e(dt) / e(dt / 2) would be spoiled by E_h. Instead three runs are made, with dt, dt / 2 and dt / 4, and one looks at the
// DIFFERENCES between successive runs, in which E_h cancels:
//
//     d1 = e(dt) - e(dt / 2) = E_dt(dt) - E_dt(dt / 2) = C dt^p (1 - 2^-p)
//     d2 = e(dt / 2) - e(dt / 4)                       = C (dt / 2)^p (1 - 2^-p)            =>   d1 / d2 = 2^p.
//
// So  p = observed_order(d1, d2, 2)  (the function for the order in space, applied to the differences).
// Warning: e is a norm, always positive, and it does not remember the sign of E_h + E_dt. The subtraction holds only if that
// sign is the same in the three runs: with backward Euler a dt large enough to make E_dt dominate is needed.

namespace {

// The expected error on the 2 x 2 mesh (EXAMPLE 110): |gamma^n_steps - exp(-2 pi^2 n_steps dt)| / sqrt(8), with
// gamma = (1 - 32 (1 - theta) dt) / (1 + 32 theta dt).
double two_by_two_error(double dt, std::size_t n_steps, double theta) {
    const double pi = std::acos(-1.0);
    const double steps = static_cast<double>(n_steps);
    const double gamma = (1.0 - 32.0 * (1.0 - theta) * dt) / (1.0 + 32.0 * theta * dt);
    return std::fabs(std::pow(gamma, steps) - std::exp(-2.0 * pi * pi * dt * steps)) / std::sqrt(8.0);
}

}  // namespace

// EXAMPLE 107. With 0 steps the final temperature is the initial one, and T0 is already the exact solution at time 0:
// the error is 0 (up to rounding), whatever n, dt and theta.
TEST(DecayingSine2D, ZeroStepsGiveZeroError) {
    EXPECT_NEAR(decaying_sine_error_2d(4, 0.01, 0, 0.5), 0.0, 1e-14);
    EXPECT_NEAR(decaying_sine_error_2d(9, 0.3, 0, 1.0), 0.0, 1e-14);
    EXPECT_NEAR(decaying_sine_error_2d(2, 1.0, 0, 0.0), 0.0, 1e-14);
}

// EXAMPLE 108. n = 1: a single cell, 4 nodes all on the boundary, no unknowns. The temperature is 0 at all nodes at
// every step, and the exact solution at the 4 nodes is 0 (sin(pi) in a computer is 1.2e-16, a rounding): error 0
// whatever dt, steps and theta. It is the smallest accepted case: n = 1 is valid, n = 0 is not (EXAMPLE 115).
TEST(DecayingSine2D, SingleCellMeshHasNothingToSolve) {
    EXPECT_NEAR(decaying_sine_error_2d(1, 0.01, 3, 0.5), 0.0, 1e-12);
    EXPECT_NEAR(decaying_sine_error_2d(1, 0.5, 1, 1.0), 0.0, 1e-12);
    EXPECT_NEAR(decaying_sine_error_2d(1, 0.001, 10, 0.0), 0.0, 1e-12);
}

// EXAMPLE 109. n = 2, backward Euler (theta = 1), ONE step of dt = 0.01, by hand. As in EXAMPLE 92 the nodes are
// 9: 8 on the boundary (T = 0) and only one at the center c = (1/2, 1/2), so there is a single unknown, T_c.
//   - From EXAMPLE 92: K[c][c] = 4 k = 4. The center belongs to 6 triangles of area 1/8 and in each of them the mass M[c][c] is
//     rho_c A / 6: M[c][c] = 6 * (1/8) / 6 = 1/8.
//   - The other nodes always have T = 0, and there is no source: the row of the center of (M + dt K) T_new = M T_old is
//         (1/8 + 4 dt) T_c,new = 1/8 T_c,old.
//   - At the center T0 = sin(pi/2) sin(pi/2) = 1, so T_c,new = (1/8) / (1/8 + 0.04) = 0.125 / 0.165 = 0.75757...
//   - The exact solution at the center: exp(-2 pi^2 0.01) = exp(-0.19739...) = 0.820868...
//   - The error at the center is e_c = 0.757575... - 0.820868... = -0.063292...; at the other 8 nodes it is (almost) 0: T = 0 and
//     the exact solution is at most 1e-16. Therefore
//         || e ||_L2 = sqrt(e^T M e) = |e_c| sqrt(M[c][c]) = 0.063292... / sqrt(8) = 0.022377...
TEST(DecayingSine2D, TwoByTwoBackwardEulerMatchesHandComputation) {
    EXPECT_NEAR(decaying_sine_error_2d(2, 0.01, 1, 1.0), 2.237744055204e-02, 1e-11);
}

// EXAMPLE 110. Still n = 2, but for more steps and for every theta. One step of the theta scheme at the center is
//     (1/8 + theta dt 4) T_new = (1/8 - (1 - theta) dt 4) T_old,
// i.e. T_new = gamma T_old with
//     gamma = (1 - 32 (1 - theta) dt) / (1 + 32 theta dt)        (multiply numerator and denominator by 8).
// After n_steps steps T_c = gamma^n_steps (T_c = 1 at the start) and the error is
//     | gamma^n_steps - exp(-2 pi^2 n_steps dt) | / sqrt(8).
// The cases (the expected errors, with the formula, are written next to them):
//   dt = 0.01,  2 steps, theta = 1/2:   gamma =  0.7241   error  5.2838e-02
//   dt = 0.005, 4 steps, theta = 1:     gamma =  0.8621   error  4.2969e-02
//   dt = 0.002, 5 steps, theta = 0:     gamma =  0.9360   error  3.6221e-02   (forward Euler, stable: dt < 1/16)
//   dt = 0.01,  3 steps, theta = 0.3:   gamma =  0.7080   error  7.0068e-02
//   dt = 0.2,   3 steps, theta = 1/2:   gamma = -0.5238   error  5.0816e-02
// The last one is a very large step: Crank-Nicolson is stable (|gamma| < 1) but gamma is NEGATIVE, the computed temperature
// changes sign at every step while the true one decays without oscillating. The formula reproduces this, and the code
// must do the same.
TEST(DecayingSine2D, TwoByTwoMatchesClosedFormForEveryScheme) {
    struct Case {
        double dt;
        std::size_t n_steps;
        double theta;
    };
    const Case cases[] = {{0.01, 2, 0.5}, {0.005, 4, 1.0}, {0.002, 5, 0.0}, {0.01, 3, 0.3}, {0.2, 3, 0.5}};

    for (const Case& c : cases) {
        EXPECT_NEAR(decaying_sine_error_2d(2, c.dt, c.n_steps, c.theta), two_by_two_error(c.dt, c.n_steps, c.theta),
                    1e-12)
            << "dt = " << c.dt << ", n_steps = " << c.n_steps << ", theta = " << c.theta;
    }
}

// EXAMPLE 111. Numbers computed along another route: the mesh, the matrices M and K, the theta scheme and the L2 error were
// rewritten from scratch in Python with full matrices and Gaussian elimination (no conjugate gradient).
//     n = 4, dt = 5e-4,  100 steps, theta = 1/2:   2.452250395721e-02
//     n = 8, dt = 0.01,    5 steps, theta = 1:     1.018063914618e-02
//     n = 8, dt = 0.001,  10 steps, theta = 0:     4.073834024549e-03
//     n = 6, dt = 0.002,  25 steps, theta = 0.3:   1.338984429158e-02
// C++ and Python agree on these digits to about 1e-10 (in relative terms): the test asks for 1e-7, because the
// conjugate gradient stops at a residual of 1e-12 and does not give the same noise as Gauss. If some significant
// digits change, the problem or the scheme has changed, not the rounding.
TEST(DecayingSine2D, MatchesNumbersFromADenseReference) {
    struct Case {
        std::size_t n;
        double dt;
        std::size_t n_steps;
        double theta;
        double want;
    };
    const Case cases[] = {{4, 5e-4, 100, 0.5, 2.452250395721e-02},
                          {8, 0.01, 5, 1.0, 1.018063914618e-02},
                          {8, 0.001, 10, 0.0, 4.073834024549e-03},
                          {6, 0.002, 25, 0.3, 1.338984429158e-02}};

    for (const Case& c : cases) {
        EXPECT_NEAR(decaying_sine_error_2d(c.n, c.dt, c.n_steps, c.theta), c.want, 1e-7 * c.want)
            << "n = " << c.n << ", dt = " << c.dt << ", n_steps = " << c.n_steps << ", theta = " << c.theta;
    }
}

// EXAMPLE 112. SPACE. With linear elements the L2 error goes like h^2: when n doubles the error is divided by 4, i.e.
// the observed order is 2. Crank-Nicolson with dt = 1e-3 and 50 steps (t = 0.05): the error in time, about 6e-6, is 1%
// of the mesh error at n = 32 (4.5e-4) and is neglected. Measured (errors 6.90e-3, 1.78e-3, 4.53e-4): order 1.95 from 8 to
// 16 and 1.97 from 16 to 32. We start from n = 8 because with a coarser mesh (from 4 to 8 the order is 1.83) we are not
// yet in the true asymptotic regime.
TEST(DecayingSine2D, SpatialConvergenceIsSecondOrder) {
    const double dt = 1e-3;
    const std::size_t n_steps = 50;
    double previous = decaying_sine_error_2d(8, dt, n_steps, 0.5);
    for (std::size_t n : {16u, 32u}) {
        const double current = decaying_sine_error_2d(n, dt, n_steps, 0.5);
        EXPECT_LT(current, previous) << "n = " << n / 2 << " -> " << n;
        EXPECT_NEAR(observed_order(previous, current, 2.0), 2.0, 0.1) << "n = " << n / 2 << " -> " << n;
        previous = current;
    }
}

// EXAMPLE 113. TIME, Crank-Nicolson (theta = 1/2): order 2. 8 x 8 mesh, up to t = 0.05 with 4, 8 and 16 steps. The
// errors are 7.876e-3, 7.139e-3, 6.956e-3: almost equal, because E_h (about 6.9e-3) does not go down. The direct ratio
// would give an order of 0.14 and 0.04. The differences are d1 = 7.38e-4 and d2 = 1.83e-4: ratio 4.03, order 2.01.
TEST(DecayingSine2D, TemporalConvergenceOfCrankNicolsonIsSecondOrder) {
    const double t = 0.05;
    const double e1 = decaying_sine_error_2d(8, t / 4, 4, 0.5);
    const double e2 = decaying_sine_error_2d(8, t / 8, 8, 0.5);
    const double e3 = decaying_sine_error_2d(8, t / 16, 16, 0.5);
    const double d1 = e1 - e2;
    const double d2 = e2 - e3;

    ASSERT_GT(d1, 0.0) << "e = " << e1 << ", " << e2 << ", " << e3;
    ASSERT_GT(d2, 0.0) << "e = " << e1 << ", " << e2 << ", " << e3;
    EXPECT_NEAR(observed_order(d1, d2, 2.0), 2.0, 0.1) << "e = " << e1 << ", " << e2 << ", " << e3;
}

// EXAMPLE 114. TIME, backward Euler (theta = 1): order 1. 16 x 16 mesh, up to t = 0.05 with 8, 16 and 32 steps.
// The errors are 9.07e-3, 3.80e-3, 1.09e-3. The direct ratio would give 1.26 and 1.81 (wrong: E_h = 1.8e-3 has the
// opposite sign to E_dt and the norm forgets it). The differences are d1 = 5.27e-3 and d2 = 2.71e-3: ratio 1.94, order
// 0.96; for backward Euler the order approaches 1 from below as dt decreases (0.90 with 4, 8, 16 steps). With 64
// steps E_dt (about 1.4e-3) would drop below E_h and the error would change sign: the subtraction would no longer hold.
TEST(DecayingSine2D, TemporalConvergenceOfBackwardEulerIsFirstOrder) {
    const double t = 0.05;
    const double e1 = decaying_sine_error_2d(16, t / 8, 8, 1.0);
    const double e2 = decaying_sine_error_2d(16, t / 16, 16, 1.0);
    const double e3 = decaying_sine_error_2d(16, t / 32, 32, 1.0);
    const double d1 = e1 - e2;
    const double d2 = e2 - e3;

    ASSERT_GT(d1, 0.0) << "e = " << e1 << ", " << e2 << ", " << e3;
    ASSERT_GT(d2, 0.0) << "e = " << e1 << ", " << e2 << ", " << e3;
    EXPECT_NEAR(observed_order(d1, d2, 2.0), 1.0, 0.1) << "e = " << e1 << ", " << e2 << ", " << e3;
}

// EXAMPLE 115. Invalid data: no cells (n = 0): std::invalid_argument. Time step dt zero, negative or NaN, theta
// outside [0, 1] or NaN: solve_transient_2d notices (EXAMPLE 104), and the exception must propagate all the way to the
// caller. The limiting cases, theta = 0 and theta = 1, are valid.
TEST(DecayingSine2D, RejectsInvalidArguments) {
    const double not_a_number = std::numeric_limits<double>::quiet_NaN();

    EXPECT_THROW(decaying_sine_error_2d(0, 0.01, 1, 0.5), std::invalid_argument);
    EXPECT_THROW(decaying_sine_error_2d(4, 0.0, 1, 0.5), std::invalid_argument);
    EXPECT_THROW(decaying_sine_error_2d(4, -0.01, 1, 0.5), std::invalid_argument);
    EXPECT_THROW(decaying_sine_error_2d(4, not_a_number, 1, 0.5), std::invalid_argument);
    EXPECT_THROW(decaying_sine_error_2d(4, 0.01, 1, -0.1), std::invalid_argument);
    EXPECT_THROW(decaying_sine_error_2d(4, 0.01, 1, 1.1), std::invalid_argument);
    EXPECT_THROW(decaying_sine_error_2d(4, 0.01, 1, not_a_number), std::invalid_argument);
    EXPECT_NO_THROW(decaying_sine_error_2d(4, 0.01, 1, 0.0));
    EXPECT_NO_THROW(decaying_sine_error_2d(4, 0.01, 1, 1.0));
}

// EXAMPLE 116. The numbers of the three tables of the decaying sine in 2D, the ones that apps/convergence_study prints for the
// README: SPACE with Crank-Nicolson (dt = 2.5e-4, 200 steps) and TIME with backward Euler and with Crank-Nicolson
// (32 x 32 mesh, dt = 0.05 / steps). All of them reach t = 0.05. Measured with this method and re-checked in Python
// (sparse matrices and a direct solver, no conjugate gradient): they agree to the 7 digits written here.
//
//     SPACE (Crank-Nicolson, dt = 2.5e-4, 200 steps)
//         n = 4:  2.452116e-02      n = 8:  6.894963e-03      n = 16:  1.776746e-03
//     TIME  (mesh 32 x 32)                       4 steps        8 steps        16 steps
//         Backward Euler      (theta = 1):       2.016970e-02   1.035828e-02   5.092920e-03
//         Crank-Nicolson (theta = 1/2):          1.386333e-03   6.800985e-04   5.054105e-04
//
// The test pins the inexpensive rows: the others (n = 32 and n = 64 in space, 32 and 64 steps in time) cost
// more without optimizations. If the method changes (the theta scheme, the solver, the norm) the numbers change, and the test
// says that the README tables must be redone.
TEST(DecayingSine2D, ErrorsMatchTheTablesInTheReadme) {
    struct Case {
        std::size_t n;
        double dt;
        std::size_t n_steps;
        double theta;
        double want;
    };
    const Case cases[] = {{4, 2.5e-4, 200, 0.5, 2.452116e-02},  {8, 2.5e-4, 200, 0.5, 6.894963e-03},
                          {16, 2.5e-4, 200, 0.5, 1.776746e-03}, {32, 0.05 / 4, 4, 1.0, 2.016970e-02},
                          {32, 0.05 / 8, 8, 1.0, 1.035828e-02}, {32, 0.05 / 16, 16, 1.0, 5.092920e-03},
                          {32, 0.05 / 4, 4, 0.5, 1.386333e-03}, {32, 0.05 / 8, 8, 0.5, 6.800985e-04},
                          {32, 0.05 / 16, 16, 0.5, 5.054105e-04}};

    for (const Case& c : cases) {
        EXPECT_NEAR(decaying_sine_error_2d(c.n, c.dt, c.n_steps, c.theta), c.want, 1e-6 * c.want)
            << "n = " << c.n << ", dt = " << c.dt << ", n_steps = " << c.n_steps << ", theta = " << c.theta;
    }
}

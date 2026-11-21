#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "Thermoelastic1D.h"

// Helper function: compares two vectors number by number and, if something does not match,
// says which element is wrong.
static void expect_vector_near(const DVector& a, const DVector& b, double tol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << "index " << i;
    }
}

// ---- Nodal forces equivalent to thermal expansion -------------------------------------

// A single element, of length 2. E = 3, alpha = 5. The temperature is 1 degree above T_ref = 0 at
// both nodes, so dT = 1 and the force is E alpha dT = 15.
// The heat pushes the left node to the left (-15) and the right one to the right (+15).
TEST(Thermoelastic1d, ThermalLoadOfOneElement) {
    std::vector<double> x = {0.0, 2.0};
    DVector f = assemble_thermal_load(x, 3.0, 5.0, {1.0, 1.0}, 0.0);

    expect_vector_near(f, {-15.0, 15.0}, 1e-12);
}

// Two elements of length 1 and 2 (nodes at 0, 1, 3). E = 2 and alpha = 0.5, so E alpha = 1.
// T_ref = 10 and T = {10, 20, 40}: the deviations from T_ref are {0, 10, 30}.
//   element 0: mean deviation (0 + 10) / 2 = 5   -> force 5:  node 0 receives -5,  node 1 receives +5
//   element 1: mean deviation (10 + 30) / 2 = 20 -> force 20: node 1 receives -20, node 2 receives +20
//   node 1 is shared: +5 - 20 = -15
// The elements have different lengths but no length appears in the formula.
TEST(Thermoelastic1d, ThermalLoadMatchesHandComputation) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DVector f = assemble_thermal_load(x, 2.0, 0.5, {10.0, 20.0, 40.0}, 10.0);

    expect_vector_near(f, {-5.0, -15.0, 20.0}, 1e-12);
}

// If the temperature equals T_ref everywhere the bar is at rest: no force.
// (T_ref is deliberately not 0: anyone who forgot to subtract it would find some forces.)
TEST(Thermoelastic1d, NoTemperatureChangeMeansNoForce) {
    std::vector<double> x = {0.0, 1.0, 3.0, 4.0};
    DVector f = assemble_thermal_load(x, 200.0, 12e-6, DVector(4, 300.0), 300.0);

    expect_vector_near(f, DVector(4, 0.0), 1e-12);
}

// The heat does not push the bar from outside: the forces are all internal and their sum is 0.
// Try it with an arbitrary temperature on an irregular mesh (and also check that the
// forces exist: a sum of zeros would prove nothing).
TEST(Thermoelastic1d, ThermalLoadsSumToZero) {
    std::vector<double> x = {0.0, 0.1, 0.5, 0.6, 1.5, 2.0};
    DVector T = {20.0, 35.0, 18.0, 90.0, 60.0, 25.0};
    DVector f = assemble_thermal_load(x, 70.0, 2.3e-5, T, 20.0);

    ASSERT_EQ(f.size(), 6u);
    double sum = 0.0;
    for (double fi : f) {
        sum += fi;
    }
    EXPECT_NEAR(sum, 0.0, 1e-12);
    EXPECT_GT(std::fabs(f[0]), 1e-3);
    EXPECT_GT(std::fabs(f[5]), 1e-3);
}

// Wrong inputs: T with a number of values different from the number of nodes.
TEST(Thermoelastic1d, ThermalLoadWrongInputsThrow) {
    std::vector<double> x = {0.0, 1.0, 3.0};   // 3 nodes

    EXPECT_THROW(assemble_thermal_load(x, 2.0, 0.5, {10.0, 20.0}, 10.0), std::invalid_argument);               // 2 values
    EXPECT_THROW(assemble_thermal_load(x, 2.0, 0.5, {10.0, 20.0, 40.0, 50.0}, 10.0), std::invalid_argument);   // 4 values
}

// ---- Displacement of the bar ---------------------------------------------------------

// The same hand example as before: x = {0, 1, 3}, E = 2, alpha = 0.5, T_ref = 10, T = {10, 20, 40}.
// The forces are f = {-5, -15, 20} and the stiffness matrix (E/h for each element) is
//     K = [[ 2, -2,  0],
//          [-2,  3, -1],
//          [ 0, -1,  1]]
// The left end is clamped (u0 = 0). Two equations remain, rows 1 and 2 of K u = f:
//      3 u1 - u2 = -15
//       -u1 + u2 =  20
// Adding them: 2 u1 = 5, so u1 = 2.5 and u2 = 22.5.
// Check: each element elongates by alpha dT_mean h, i.e. 0.5*5*1 = 2.5 and 0.5*20*2 = 20.
// If the right end is clamped too (u2 = 0) only one equation remains: 3 u1 = -15, u1 = -5.
TEST(Thermoelastic1d, ThermalExpansionMatchesHandComputation) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DVector T = {10.0, 20.0, 40.0};

    DVector u_free = solve_thermal_expansion(x, 2.0, 0.5, T, 10.0, RightEnd::Free);
    expect_vector_near(u_free, {0.0, 2.5, 22.5}, 1e-9);

    DVector u_fixed = solve_thermal_expansion(x, 2.0, 0.5, T, 10.0, RightEnd::Fixed);
    expect_vector_near(u_fixed, {0.0, -5.0, 0.0}, 1e-9);
}

// Bar of length 1 m (10 elements), heated uniformly from 20 to 120 degrees, right end free.
// Free to move, the bar elongates by alpha dT L = 12e-6 * 100 * 1 = 1.2e-3 m (1.2 mm),
// and the node at distance x has moved by alpha dT x. The result does not depend on E (nothing
// stops the bar, so there are no forces): it is tested with two different moduli, steel and aluminum.
TEST(Thermoelastic1d, FreeBarElongatesByAlphaDeltaTTimesLength) {
    const double alpha = 12e-6, T_ref = 20.0, T_hot = 120.0;
    std::vector<double> x = uniform_mesh(1.0, 10);
    DVector T(x.size(), T_hot);

    for (double E : {200e9, 70e9}) {
        DVector u = solve_thermal_expansion(x, E, alpha, T, T_ref, RightEnd::Free);
        ASSERT_EQ(u.size(), x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            EXPECT_NEAR(u[i], alpha * (T_hot - T_ref) * x[i], 1e-12) << "E = " << E << ", node " << i;
        }
        EXPECT_NEAR(u.back(), 1.2e-3, 1e-12) << "E = " << E;
    }
}

// The same bar, but clamped on the right too: the walls hold it still, so u = 0 at
// every node. (The stress, instead, is not 0: it is computed in step 4c.)
TEST(Thermoelastic1d, ClampedBarWithUniformHeatingDoesNotMove) {
    std::vector<double> x = uniform_mesh(1.0, 10);
    DVector T(x.size(), 120.0);

    DVector u = solve_thermal_expansion(x, 200e9, 12e-6, T, 20.0, RightEnd::Fixed);

    expect_vector_near(u, DVector(x.size(), 0.0), 1e-12);
}

// Linear temperature T(x) = 30 + 15 x on a bar of length 2, with a NON-uniform mesh
// (x_i = L (i/N)^1.5). T_ref = 10, so dT(x) = 20 + 15 x. E = 3, alpha = 0.02.
//  - Right end free: u(x) = alpha * integral of dT = 0.02 (20 x + 7.5 x^2) = 0.4 x + 0.15 x^2.
//  - Right end clamped: the stress is the same everywhere and u(L) = 0, which gives
//    u(x) = 0.15 x (x - 2).
// With linear T the trapezoidal rule on each element is exact, so the values at the nodes are
// exact on any mesh: the tolerance is tight and a red test means an error in the code.
TEST(Thermoelastic1d, LinearTemperatureOnAGradedMeshIsExactAtTheNodes) {
    const size_t N = 30;
    const double L = 2.0;
    std::vector<double> x(N + 1);
    DVector T(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = L * std::pow(static_cast<double>(i) / N, 1.5);
        T[i] = 30.0 + 15.0 * x[i];
    }

    DVector u_free = solve_thermal_expansion(x, 3.0, 0.02, T, 10.0, RightEnd::Free);
    DVector u_fixed = solve_thermal_expansion(x, 3.0, 0.02, T, 10.0, RightEnd::Fixed);

    ASSERT_EQ(u_free.size(), N + 1);
    ASSERT_EQ(u_fixed.size(), N + 1);
    for (size_t i = 0; i <= N; ++i) {
        EXPECT_NEAR(u_free[i], 0.4 * x[i] + 0.15 * x[i] * x[i], 1e-9) << "free, node " << i;
        EXPECT_NEAR(u_fixed[i], 0.15 * x[i] * (x[i] - 2.0), 1e-9) << "clamped, node " << i;
    }
}

// Free bar, very irregular temperature, 200 elements of different length.
// With the right end free nothing stops the bar and every element expands on its own:
//     u[i] - u[i-1] = alpha * dT_mean * h      (for ANY temperature and ANY mesh).
// So u is rebuilt by adding up the elongations element by element, without solving anything.
// The test also serves to check that CG was run all the way to the end: with such an
// irregular temperature the residual falls slowly, and stopping too early
// (tolerance too loose) gives a large error.
TEST(Thermoelastic1d, FreeBarOnARoughTemperatureExpandsElementByElement) {
    const size_t N = 200;
    const double L = 2.0, E = 7.0e10, alpha = 1.5e-5, T_ref = 300.0;
    std::vector<double> x(N + 1);
    DVector T(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = L * std::pow(static_cast<double>(i) / N, 1.3);
        T[i] = 300.0 + 80.0 * std::sin(1.7 * i) + 30.0 * std::cos(0.37 * i * i);
    }

    DVector u = solve_thermal_expansion(x, E, alpha, T, T_ref, RightEnd::Free);

    ASSERT_EQ(u.size(), N + 1);
    EXPECT_NEAR(u[0], 0.0, 1e-15);
    double expected = 0.0;   // displacement of node i = sum of the elongations of the first i elements
    for (size_t i = 1; i <= N; ++i) {
        expected += alpha * (x[i] - x[i - 1]) * ((T[i - 1] - T_ref) + (T[i] - T_ref)) / 2.0;
        EXPECT_NEAR(u[i], expected, 1e-12) << "node " << i;
    }
}

// Maximum error at the nodes for T(x) = sin(pi x) on a bar of length 1, with T_ref = 0, E = 3,
// alpha = 2 and the right end free. Exact solution: u(x) = alpha (1 - cos(pi x)) / pi.
static double sine_expansion_error(size_t N) {
    const double pi = std::acos(-1.0);
    std::vector<double> x = uniform_mesh(1.0, N);
    DVector T(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        T[i] = std::sin(pi * x[i]);
    }

    DVector u = solve_thermal_expansion(x, 3.0, 2.0, T, 0.0, RightEnd::Free);

    double max_err = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double err = std::fabs(u[i] - 2.0 * (1.0 - std::cos(pi * x[i])) / pi);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

// Here T is not linear, so the values at the nodes are no longer exact. Linear elements are
// second order: halving h the error must become about 4 times smaller.
TEST(Thermoelastic1d, ThermalExpansionIsSecondOrderForASineTemperature) {
    const double e10 = sine_expansion_error(10);
    const double e20 = sine_expansion_error(20);
    const double e40 = sine_expansion_error(40);

    EXPECT_GT(e10, 1e-3);   // the error exists (it is not zero by chance)
    EXPECT_NEAR(observed_order(e10, e20, 2.0), 2.0, 0.1);
    EXPECT_NEAR(observed_order(e20, e40, 2.0), 2.0, 0.1);
}

// Wrong inputs: they must throw std::invalid_argument.
TEST(Thermoelastic1d, ThermalExpansionWrongInputsThrow) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DVector T = {10.0, 20.0, 40.0};

    // T with 2 values instead of 3
    EXPECT_THROW(solve_thermal_expansion(x, 2.0, 0.5, DVector(2, 10.0), 10.0, RightEnd::Free), std::invalid_argument);
    // a bar with a single node has no elements
    EXPECT_THROW(solve_thermal_expansion({0.0}, 2.0, 0.5, {10.0}, 10.0, RightEnd::Free), std::invalid_argument);
    // E must be positive: 0, negative and NaN are not acceptable
    EXPECT_THROW(solve_thermal_expansion(x, 0.0, 0.5, T, 10.0, RightEnd::Free), std::invalid_argument);
    EXPECT_THROW(solve_thermal_expansion(x, -2.0, 0.5, T, 10.0, RightEnd::Free), std::invalid_argument);
    EXPECT_THROW(solve_thermal_expansion(x, std::nan(""), 0.5, T, 10.0, RightEnd::Free), std::invalid_argument);
}

// A NaN in the temperature: the system has no solution and CG cannot converge.
// It must say so with an exception (std::runtime_error), not return meaningless displacements.
TEST(Thermoelastic1d, ThermalExpansionThrowsWhenTheLinearSolverCannotConverge) {
    std::vector<double> x = uniform_mesh(1.0, 10);
    DVector T(x.size(), 100.0);
    T[4] = std::nan("");

    EXPECT_THROW(solve_thermal_expansion(x, 2.0, 0.5, T, 10.0, RightEnd::Free), std::runtime_error);
    EXPECT_THROW(solve_thermal_expansion(x, 2.0, 0.5, T, 10.0, RightEnd::Fixed), std::runtime_error);
}

// ---- Stress in every element -----------------------------------------------------------

// The same hand example as always: x = {0, 1, 3}, E = 2, alpha = 0.5, T_ref = 10, T = {10, 20, 40}.
// Element lengths: {1, 2}. Mean temperature deviations: {5, 20}.
// Strain due to the heat alone (alpha times the mean deviation): {2.5, 10}.
//     sigma = E * ( (u[e+1] - u[e]) / h  -  alpha dT_mean )
//
// a) free bar, u = {0, 2.5, 22.5}: total strain {2.5, 10}, equal to the thermal one.
//    The difference is 0, so sigma = {0, 0}: a bar that is free to expand has no stresses.
// b) clamped bar, u = {0, -5, 0}: total strain {-5, 2.5}, so
//    sigma = 2 * (-5 - 2.5) = -15 and sigma = 2 * (2.5 - 10) = -15. The same stress in the whole bar,
//    and it is negative: compression.
// c) bar that does not move (u = 0): sigma = 2 * (0 - 2.5) = -5 and sigma = 2 * (0 - 10) = -20.
// d) no heat (T = T_ref) and u = {0, 1, 4}: strains 1 and 1.5, so sigma = {2, 3}.
//    Here only the mechanical part matters.
TEST(Thermoelastic1d, ElementStressMatchesHandComputation) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DVector T = {10.0, 20.0, 40.0};

    expect_vector_near(element_stress(x, 2.0, 0.5, T, 10.0, {0.0, 2.5, 22.5}), {0.0, 0.0}, 1e-12);
    expect_vector_near(element_stress(x, 2.0, 0.5, T, 10.0, {0.0, -5.0, 0.0}), {-15.0, -15.0}, 1e-12);
    expect_vector_near(element_stress(x, 2.0, 0.5, T, 10.0, {0.0, 0.0, 0.0}), {-5.0, -20.0}, 1e-12);
    expect_vector_near(element_stress(x, 2.0, 0.5, DVector(3, 10.0), 10.0, {0.0, 1.0, 4.0}), {2.0, 3.0}, 1e-12);
}

// Steel bar of length 1 m (10 elements), heated from 20 to 120 degrees. E = 200e9 Pa, alpha = 12e-6.
// Clamped at both ends it cannot expand: the wall squeezes it by as much as the heat would
// lengthen it, so the mechanical strain is -alpha dT and in every element
//     sigma = -E alpha dT = -200e9 * 12e-6 * 100 = -2.4e8 Pa   (240 MPa of compression).
// It is of the same order as the yield stress of a structural steel (about 250 MPa):
// this is why bridges and rails have expansion joints.
// With the right end free the bar lengthens without effort and the stress is 0.
TEST(Thermoelastic1d, ClampedSteelBarIsCompressedByEAlphaDeltaT) {
    const double E = 200e9, alpha = 12e-6, T_ref = 20.0, T_hot = 120.0;
    const double scale = E * alpha * (T_hot - T_ref);   // 2.4e8 Pa
    std::vector<double> x = uniform_mesh(1.0, 10);
    DVector T(x.size(), T_hot);

    DVector u_fixed = solve_thermal_expansion(x, E, alpha, T, T_ref, RightEnd::Fixed);
    DVector s_fixed = element_stress(x, E, alpha, T, T_ref, u_fixed);
    ASSERT_EQ(s_fixed.size(), 10u);
    for (size_t e = 0; e < s_fixed.size(); ++e) {
        EXPECT_NEAR(s_fixed[e], -scale, 1e-8 * scale) << "clamped, element " << e;
    }

    DVector u_free = solve_thermal_expansion(x, E, alpha, T, T_ref, RightEnd::Free);
    DVector s_free = element_stress(x, E, alpha, T, T_ref, u_free);
    ASSERT_EQ(s_free.size(), 10u);
    for (size_t e = 0; e < s_free.size(); ++e) {
        EXPECT_NEAR(s_free[e], 0.0, 1e-8 * scale) << "free, element " << e;
    }
}

// Free bar, with the same irregular temperature and the same mesh (200 elements of different
// length) as the test FreeBarOnARoughTemperatureExpandsElementByElement. Every element expands on
// its own, so the stress is 0 everywhere, for ANY temperature and ANY mesh.
// "About 0" here means less than 1e-8 times E alpha 100, i.e. the stress the same
// bar would have if it were clamped and heated by 100 degrees.
TEST(Thermoelastic1d, FreeBarHasNoStressOnARoughTemperature) {
    const size_t N = 200;
    const double L = 2.0, E = 7.0e10, alpha = 1.5e-5, T_ref = 300.0;
    std::vector<double> x(N + 1);
    DVector T(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = L * std::pow(static_cast<double>(i) / N, 1.3);
        T[i] = 300.0 + 80.0 * std::sin(1.7 * i) + 30.0 * std::cos(0.37 * i * i);
    }

    DVector u = solve_thermal_expansion(x, E, alpha, T, T_ref, RightEnd::Free);
    DVector sigma = element_stress(x, E, alpha, T, T_ref, u);

    ASSERT_EQ(sigma.size(), N);
    for (size_t e = 0; e < N; ++e) {
        EXPECT_NEAR(sigma[e], 0.0, 1e-8 * E * alpha * 100.0) << "element " << e;
    }
}

// Linear temperature T(x) = 30 + 15 x on a bar of length 2, with the same NON-uniform mesh as the
// test LinearTemperatureOnAGradedMeshIsExactAtTheNodes, but with the right end clamped.
// T_ref = 10, E = 3, alpha = 0.02, so the deviation is 20 + 15 x: the heat would want to expand every
// little piece differently. Yet the stress is the same in every element, because the forces
// must balance along the bar. It equals -E alpha times the mean deviation over the bar:
//     -3 * 0.02 * (20 + 15 * 1) = -2.1.
TEST(Thermoelastic1d, ClampedBarStressIsUniformForALinearTemperatureOnAGradedMesh) {
    const size_t N = 30;
    const double L = 2.0;
    std::vector<double> x(N + 1);
    DVector T(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = L * std::pow(static_cast<double>(i) / N, 1.5);
        T[i] = 30.0 + 15.0 * x[i];
    }

    DVector u = solve_thermal_expansion(x, 3.0, 0.02, T, 10.0, RightEnd::Fixed);
    DVector sigma = element_stress(x, 3.0, 0.02, T, 10.0, u);

    ASSERT_EQ(sigma.size(), N);
    for (size_t e = 0; e < N; ++e) {
        EXPECT_NEAR(sigma[e], -2.1, 1e-8) << "element " << e;
    }
}

// Maximum error on the stress for T(x) = sin(pi x) on a bar of length 1, with T_ref = 0, E = 3,
// alpha = 2 and the right end clamped. Exact stress: -E alpha times the mean of sin(pi x) over
// [0, 1], i.e. -3 * 2 * (2 / pi) = -12 / pi = -3.82.
static double sine_stress_error(size_t N) {
    const double pi = std::acos(-1.0);
    std::vector<double> x = uniform_mesh(1.0, N);
    DVector T(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        T[i] = std::sin(pi * x[i]);
    }

    DVector u = solve_thermal_expansion(x, 3.0, 2.0, T, 0.0, RightEnd::Fixed);
    DVector sigma = element_stress(x, 3.0, 2.0, T, 0.0, u);

    EXPECT_EQ(sigma.size(), N);
    double max_err = 0.0;
    for (double s : sigma) {
        const double err = std::fabs(s + 12.0 / pi);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

// T is not linear, so the computed stress is not exact: the mean of T over the bar is
// computed with the trapezoidal rule, which is off by about pi / (6 N^2). The error on the stress is
// about pi / N^2: halving h it becomes 4 times smaller (second order).
TEST(Thermoelastic1d, ClampedBarStressIsSecondOrderForASineTemperature) {
    const double e10 = sine_stress_error(10);
    const double e20 = sine_stress_error(20);
    const double e40 = sine_stress_error(40);

    EXPECT_GT(e10, 1e-2);   // the error exists (about 0.031): it is not zero by chance
    EXPECT_NEAR(observed_order(e10, e20, 2.0), 2.0, 0.1);
    EXPECT_NEAR(observed_order(e20, e40, 2.0), 2.0, 0.1);
}

// One stress per element: with n nodes there are n - 1 elements.
// A single element (of length 2, E = 3, alpha = 5, temperature deviation 1) held still (u = 0):
// sigma = 3 * (0 - 5 * 1) = -15. With fewer than 2 nodes there are no elements: the result is an
// empty vector, with no errors and no crash.
TEST(Thermoelastic1d, ElementStressHasOneValuePerElement) {
    DVector one = element_stress({0.0, 2.0}, 3.0, 5.0, {1.0, 1.0}, 0.0, {0.0, 0.0});
    ASSERT_EQ(one.size(), 1u);
    EXPECT_NEAR(one[0], -15.0, 1e-12);

    EXPECT_TRUE(element_stress({0.0}, 3.0, 5.0, {1.0}, 0.0, {0.0}).empty());
    EXPECT_TRUE(element_stress({}, 3.0, 5.0, {}, 0.0, {}).empty());
}

// Wrong inputs: T or u with a number of values different from the number of nodes.
TEST(Thermoelastic1d, ElementStressWrongInputsThrow) {
    std::vector<double> x = {0.0, 1.0, 3.0};   // 3 nodes
    DVector T = {10.0, 20.0, 40.0};
    DVector u = {0.0, 2.5, 22.5};

    EXPECT_THROW(element_stress(x, 2.0, 0.5, DVector(2, 10.0), 10.0, u), std::invalid_argument);          // T with 2 values
    EXPECT_THROW(element_stress(x, 2.0, 0.5, DVector(4, 10.0), 10.0, u), std::invalid_argument);          // T with 4 values
    EXPECT_THROW(element_stress(x, 2.0, 0.5, T, 10.0, {0.0, 2.5}), std::invalid_argument);                // u with 2 values
    EXPECT_THROW(element_stress(x, 2.0, 0.5, T, 10.0, {0.0, 2.5, 22.5, 30.0}), std::invalid_argument);    // u with 4 values
}

// ---- The complete chain: heat -> expansion -> stress -----------------------------------

// Hand example, all in one go. Bar of length 2 with 2 elements (x = {0, 1, 2}, h = 1).
//
// HEAT: one backward Euler step (theta = 1), dt = 0.5, rho_c = 3, k = 1.5, q = 2.
// Ends fixed at 6 and 14, initial temperature T0 = {6, 6, 14}. The matrices are
//     M = rho_c/6 * [[2,1,0],[1,4,1],[0,1,2]] = [[1, 0.5, 0], [0.5, 2, 0.5], [0, 0.5, 1]]
//     K = k * [[1,-1,0],[-1,2,-1],[0,-1,1]]   = [[1.5, -1.5, 0], [-1.5, 3, -1.5], [0, -1.5, 1.5]]
// and the heat of the source is F = q h/2 * (1, 2, 1) = (1, 2, 1). The system (M + dt K) T = M T0 + dt F
// has the middle row   -0.25 T_0 + 3.5 T_1 - 0.25 T_2 = 0.5*6 + 2*6 + 0.5*14 + 0.5*2 = 23.
// With T_0 = 6 and T_2 = 14 (ends fixed) what remains is 3.5 T_1 = 23 + 1.5 + 3.5 = 28, so T_1 = 8.
//     T = {6, 8, 14}
//
// MECHANICS: T_ref = 5, E = 4, alpha = 0.25. Deviations dT = {1, 3, 9}, mean deviations {2, 6}.
//  - right end free: each element elongates by alpha dT_mean h = 0.5 and 1.5,
//    so u = {0, 0.5, 2}, and the stress is 0.
//  - right end clamped: the forces are E alpha dT_mean = {2, 6}, so f = {-2, 2-6, 6}.
//    With u_0 = u_2 = 0 the equation of node 1 is E * 2 u_1 = -4, i.e. u_1 = -0.5.
//    Total strains -0.5 and +0.5, thermal 0.5 and 1.5: sigma = 4 (-0.5-0.5) = -4 and 4 (0.5-1.5) = -4.
static ThermoMechanicalProblem hand_problem() {
    ThermoMechanicalProblem p;
    p.x = uniform_mesh(2.0, 2);
    p.rho_c = 3.0;
    p.k = 1.5;
    p.q = 2.0;
    p.T_left = 6.0;
    p.T_right = 14.0;
    p.T0 = {6.0, 6.0, 14.0};
    p.dt = 0.5;
    p.n_steps = 1;
    p.theta = 1.0;
    p.E = 4.0;
    p.alpha = 0.25;
    p.T_ref = 5.0;
    return p;
}

TEST(Thermoelastic1d, ThermoMechanicalHandComputation) {
    ThermoMechanicalProblem p = hand_problem();

    p.right_end = RightEnd::Free;
    ThermoMechanicalResult free_bar = solve_thermomechanical(p);
    expect_vector_near(free_bar.T, {6.0, 8.0, 14.0}, 1e-9);
    expect_vector_near(free_bar.u, {0.0, 0.5, 2.0}, 1e-9);
    expect_vector_near(free_bar.sigma, {0.0, 0.0}, 1e-9);

    p.right_end = RightEnd::Fixed;
    ThermoMechanicalResult clamped_bar = solve_thermomechanical(p);
    expect_vector_near(clamped_bar.T, {6.0, 8.0, 14.0}, 1e-9);
    expect_vector_near(clamped_bar.u, {0.0, -0.5, 0.0}, 1e-9);
    expect_vector_near(clamped_bar.sigma, {-4.0, -4.0}, 1e-9);
}

// The chain must give exactly what the three steps done by hand give, one after the other.
// Here every datum has a different and "awkward" value (none equal to another), the mesh is not
// uniform and theta = 0.6: if the function swaps two arguments of the same type (rho_c and k,
// T_left and T_right, E and alpha...) or passes the initial temperature in place of the
// final one, the numbers change and the test notices.
TEST(Thermoelastic1d, ThermoMechanicalEqualsTheThreeStepsOneByOne) {
    const size_t N = 12;
    ThermoMechanicalProblem p;
    p.x.resize(N + 1);
    p.T0.resize(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        p.x[i] = 1.7 * std::pow(static_cast<double>(i) / N, 1.2);
        p.T0[i] = 20.0 + 3.0 * std::sin(2.0 * i);
    }
    p.rho_c = 1.7;
    p.k = 2.3;
    p.q = 4.1;
    p.T_left = 12.5;
    p.T_right = 31.5;
    p.T0.front() = p.T_left;
    p.T0.back() = p.T_right;
    p.dt = 0.07;
    p.n_steps = 9;
    p.theta = 0.6;
    p.E = 5.3;
    p.alpha = 0.013;
    p.T_ref = 9.1;

    for (RightEnd end : {RightEnd::Free, RightEnd::Fixed}) {
        p.right_end = end;
        ThermoMechanicalResult r = solve_thermomechanical(p);

        DVector T = solve_transient(p.x, p.rho_c, p.k, p.q, p.T_left, p.T_right, p.T0, p.dt, p.n_steps, p.theta);
        DVector u = solve_thermal_expansion(p.x, p.E, p.alpha, T, p.T_ref, end);
        DVector sigma = element_stress(p.x, p.E, p.alpha, T, p.T_ref, u);

        expect_vector_near(r.T, T, 1e-12);
        expect_vector_near(r.u, u, 1e-12);
        expect_vector_near(r.sigma, sigma, 1e-12);
    }
}

// The "decaying sine" problem, already used for the temperature: bar of length 1, rho_c = 1, k = 1,
// no source, ends at 0, T0 = sin(pi x), so T(x, t) = exp(-pi^2 t) sin(pi x).
// With T_ref = 0, E = 3 and alpha = 2 the chain has two exact results, obtained by integrating
// T along the bar (the integral of sin(pi x) between 0 and 1 is 2/pi):
//  - right end free: the tip moves by alpha * integral = 2 alpha exp(-pi^2 t) / pi;
//  - right end clamped: the stress is the same everywhere and equals
//    -E alpha * (mean of T) = -2 E alpha exp(-pi^2 t) / pi.
static ThermoMechanicalProblem decaying_sine_problem(size_t N, double dt, size_t n_steps, double theta,
                                                     RightEnd right_end) {
    const double pi = std::acos(-1.0);
    ThermoMechanicalProblem p;
    p.x = uniform_mesh(1.0, N);
    p.T0.resize(p.x.size());
    for (size_t i = 0; i < p.x.size(); ++i) {
        p.T0[i] = std::sin(pi * p.x[i]);
    }
    p.rho_c = 1.0;
    p.k = 1.0;
    p.q = 0.0;
    p.T_left = 0.0;
    p.T_right = 0.0;
    p.dt = dt;
    p.n_steps = n_steps;
    p.theta = theta;
    p.E = 3.0;
    p.alpha = 2.0;
    p.T_ref = 0.0;
    p.right_end = right_end;
    return p;
}

// Error (in absolute value) of the tip displacement, with the right end free.
static double tip_error(size_t N, double dt, size_t n_steps, double theta) {
    const double pi = std::acos(-1.0);
    ThermoMechanicalResult r = solve_thermomechanical(decaying_sine_problem(N, dt, n_steps, theta, RightEnd::Free));
    EXPECT_EQ(r.u.size(), N + 1);
    if (r.u.size() != N + 1) {
        return 1.0;   // with no displacements there is nothing to compare (the test is already red)
    }
    const double t = n_steps * dt;
    return std::fabs(r.u.back() - 2.0 * 2.0 * std::exp(-pi * pi * t) / pi);
}

// At t = 0.1 (Crank-Nicolson, 100 elements, 100 steps) the chain must give the two exact results:
//  - free bar: the tip moves by 2 alpha exp(-pi^2 t) / pi = 0.4745. The left node does not
//    move, and the stress is 0 in every element (computed with the SAME final temperature from which
//    the displacement comes: with the initial temperature it would not be 0);
//  - clamped bar: the stress is -2 E alpha exp(-pi^2 t) / pi = -1.4236 in every element.
// The discretisation error is about 1.7e-4 in relative terms, so the tolerance is 5e-4.
TEST(Thermoelastic1d, ThermoMechanicalFollowsTheDecayingSine) {
    const double pi = std::acos(-1.0);
    const size_t N = 100, n_steps = 100;
    const double dt = 0.001, t = n_steps * dt;

    ThermoMechanicalResult free_bar = solve_thermomechanical(decaying_sine_problem(N, dt, n_steps, 0.5, RightEnd::Free));
    ASSERT_EQ(free_bar.u.size(), N + 1);
    ASSERT_EQ(free_bar.sigma.size(), N);
    const double exact_tip = 2.0 * 2.0 * std::exp(-pi * pi * t) / pi;
    EXPECT_NEAR(free_bar.u.front(), 0.0, 1e-15);
    EXPECT_NEAR(free_bar.u.back(), exact_tip, 5e-4 * exact_tip);
    for (size_t e = 0; e < N; ++e) {
        EXPECT_NEAR(free_bar.sigma[e], 0.0, 1e-9) << "free, element " << e;
    }

    ThermoMechanicalResult clamped_bar = solve_thermomechanical(decaying_sine_problem(N, dt, n_steps, 0.5, RightEnd::Fixed));
    ASSERT_EQ(clamped_bar.sigma.size(), N);
    const double exact_stress = -2.0 * 3.0 * 2.0 * std::exp(-pi * pi * t) / pi;
    for (size_t e = 0; e < N; ++e) {
        EXPECT_NEAR(clamped_bar.sigma[e], exact_stress, 5e-4 * std::fabs(exact_stress)) << "clamped, element " << e;
    }
}

// ORDERS. The displacement and the stress derive from the temperature: do they inherit its order?
// Crank-Nicolson, N = 10, 20, 40, 80 elements and N steps up to t = 0.1 (dt = 0.1 / N: when
// you halve h you also halve dt). The error in space and the one in time are both second
// order, so the error on the tip must become 4 times smaller at every doubling.
TEST(Thermoelastic1d, ThermoMechanicalIsSecondOrderForCrankNicolson) {
    double previous = tip_error(10, 0.1 / 10, 10, 0.5);
    for (size_t N : {20, 40, 80}) {
        const double current = tip_error(N, 0.1 / N, N, 0.5);
        EXPECT_NEAR(observed_order(previous, current, 2.0), 2.0, 0.05) << "N = " << N / 2 << " -> " << N;
        previous = current;
    }
}

// Backward Euler (theta = 1): first order in time. Very fine mesh (N = 400), so the mesh error
// is negligible; t = 0.1 with 5, 10, 20 steps.
TEST(Thermoelastic1d, ThermoMechanicalIsFirstOrderInTimeForBackwardEuler) {
    const size_t N = 400;
    double previous = tip_error(N, 0.1 / 5, 5, 1.0);
    for (size_t steps : {10, 20}) {
        const double current = tip_error(N, 0.1 / steps, steps, 1.0);
        EXPECT_NEAR(observed_order(previous, current, 2.0), 1.0, 0.1) << steps / 2 << " -> " << steps << " steps";
        previous = current;
    }
}

// Wrong inputs: the errors of the three steps must not disappear. Start from the hand problem
// (valid) and spoil one datum at a time.
TEST(Thermoelastic1d, ThermoMechanicalWrongInputsThrow) {
    const ThermoMechanicalProblem good = hand_problem();
    EXPECT_NO_THROW(solve_thermomechanical(good));

    ThermoMechanicalProblem p = good;   // a T0 with one value fewer (from the heat step)
    p.T0.pop_back();
    EXPECT_THROW(solve_thermomechanical(p), std::invalid_argument);

    p = good;                           // dt = 0 (from the heat step)
    p.dt = 0.0;
    EXPECT_THROW(solve_thermomechanical(p), std::invalid_argument);

    p = good;                           // theta outside [0, 1] (from the heat step)
    p.theta = 1.5;
    EXPECT_THROW(solve_thermomechanical(p), std::invalid_argument);

    p = good;                           // E = 0 and E = NaN (from the mechanics step)
    p.E = 0.0;
    EXPECT_THROW(solve_thermomechanical(p), std::invalid_argument);
    p.E = std::nan("");
    EXPECT_THROW(solve_thermomechanical(p), std::invalid_argument);

    p = good;                           // a bar with a single node has no elements (from the mechanics step)
    p.x = {0.0};
    p.T0 = {6.0};
    EXPECT_THROW(solve_thermomechanical(p), std::invalid_argument);

    p = good;                           // a NaN in the data: CG does not converge (from the heat step)
    p.T0[1] = std::nan("");
    EXPECT_THROW(solve_thermomechanical(p), std::runtime_error);
}

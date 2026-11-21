#ifndef THERMOELASTIC1D_H
#define THERMOELASTIC1D_H

#include <vector>
#include "Fem1D.h" 

// 1D thermo-mechanics: a bar of cross-section 1 that expands when it is heated.
//
// The material has Young's modulus E (> 0) and thermal expansion coefficient alpha.
// T_ref is the temperature at which the bar is stress-free.
// Heated by dT = T - T_ref, the bar would like to lengthen by  alpha dT  per unit of length.
// With u(x) the displacement, the stress is   sigma = E (u' - alpha dT):
// only the part of the strain that is NOT due to heat counts.

// Nodal forces equivalent to the thermal expansion (one number per node).
// Every element is imagined as blocked at both ends: when heated it pushes its two nodes outwards
// with force E alpha dT, where dT = (T_left + T_right) / 2 - T_ref is the mean temperature rise
// of the element (the length of the element cancels out). The left node receives -E alpha dT,
// the right one +E alpha dT; a node shared by two elements adds the two contributions.
// T = temperature at every node (x.size() numbers, linear inside the element).
// Throws std::invalid_argument if T does not have one number per node.
DVector assemble_thermal_load(const std::vector<double>& x, double E, double alpha, const DVector& T, double T_ref);

// Condition at the right end of the bar: free or fixed (displacement 0).
// The left end is always fixed.
enum class RightEnd { Free, Fixed };

// Displacement u at every node of a bar with nodes x (increasing), heated by T (one value per node).
// It solves K u = f, with K the stiffness matrix (E/h for every element, like the K of the heat
// problem with E in place of k) and f the thermal forces. The left end is fixed (u = 0); the right
// end is free or fixed (u = 0), according to right_end.
// Throws std::invalid_argument if x has fewer than 2 nodes, if T does not have one value per node or if E is not > 0,
// and std::runtime_error if CG does not converge.
DVector solve_thermal_expansion(const std::vector<double>& x, double E, double alpha, const DVector& T, double T_ref, RightEnd right_end);

// Stress in every element (x.size() - 1 numbers), given the displacement u of every node:
//   sigma_e = E * ( (u[e+1] - u[e]) / h_e  -  alpha * mean_dT_e )
// h_e is the length of the element and mean_dT_e the mean of the differences (T - T_ref) of its two nodes.
// The first term is the total strain, the second the strain that the heat would produce anyway:
// the stress comes from the difference. Positive = tension, negative = compression.
// A bar with fewer than 2 nodes has no elements: the result is an empty vector.
// Throws std::invalid_argument if T or u do not have one value per node.
DVector element_stress(const std::vector<double>& x, double E, double alpha, const DVector& T, double T_ref, const DVector& u);

// ---- The whole chain: heat -> expansion -> stress ------------------------------------

// All the data of a thermo-mechanical problem in one object, each with its own name.
// Passing 14 arguments one after the other (10 of them double) is dangerous: the compiler does not
// notice if you swap two of the same type, like the arguments of CGSolver. With names it cannot happen.
// The problem is created zero-initialised (ThermoMechanicalProblem p;) and the fields that are needed are filled in.
struct ThermoMechanicalProblem {
    std::vector<double> x;   // nodes of the bar (increasing)

    // Heat: rho_c dT/dt - k T'' = q, fixed ends (as in solve_transient).
    double rho_c = 0.0;
    double k = 0.0;
    double q = 0.0;
    double T_left = 0.0;
    double T_right = 0.0;
    DVector T0;              // initial temperature, one value per node

    // Time: n_steps steps of size dt with the theta scheme (theta = 1: backward Euler).
    double dt = 0.0;
    std::size_t n_steps = 0;
    double theta = 1.0;

    // Mechanics (as in solve_thermal_expansion). The left end is always fixed.
    double E = 0.0;
    double alpha = 0.0;
    double T_ref = 0.0;
    RightEnd right_end = RightEnd::Free;
};

// The result, at the final time t = n_steps * dt.
struct ThermoMechanicalResult {
    DVector T;       // temperature at every node
    DVector u;       // displacement at every node
    DVector sigma;   // stress in every element (one number fewer than the nodes)
};

// Solves the chained problem, in three steps:
//   1. T     = solve_transient(...)           the temperature at the final time
//   2. u     = solve_thermal_expansion(...)   the displacement caused by that temperature
//   3. sigma = element_stress(...)            the resulting stress
// The coupling is one-way: heat moves the bar, the bar heats nothing.
// The exceptions of the three steps (std::invalid_argument, std::runtime_error) are not caught.
ThermoMechanicalResult solve_thermomechanical(const ThermoMechanicalProblem& problem);

#endif

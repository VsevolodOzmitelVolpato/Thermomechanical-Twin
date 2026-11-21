#ifndef SOLVER2D_H
#define SOLVER2D_H

#include <cstddef>
#include <vector>
#include "Mesh2D.h"
#include "CSR_and_Vectors.h"

// Steady heat conduction on a 2D mesh: solves K T = f, where K is the stiffness matrix of conductivity k and f the load
// vector (one value per node), with fixed temperatures `values` at `fixed_nodes`.
// Returns the temperature of every node (n values); at the fixed nodes the values are exactly the imposed ones.
// At least one node must be fixed, otherwise K is singular.
// Throws std::invalid_argument if: k is not positive or is NaN; f does not have n elements; no node is fixed; the number
// of values differs from the number of fixed nodes; a fixed node is >= n or is repeated; or a free node does not belong
// to any triangle. Throws std::runtime_error if CG does not converge.
DVector solve_steady_2d(const Mesh2D &M, double k,const DVector &f, const std::vector<std::size_t> &fixed_nodes,const std::vector<double> &values);

// Transient heat conduction with the theta scheme (theta = 1: backward Euler, theta = 0.5: Crank-Nicolson):
//     (M + theta dt K) T_new = (M - (1 - theta) dt K) T_old + dt f
// It starts from T0 and does n_steps steps of size dt, with fixed temperatures at `fixed_nodes`; the matrices are built
// once, before the time loop. Returns the temperatures at the final time.
// Throws std::invalid_argument if rho_c, k or dt is not positive, if theta is outside [0, 1], if f or T0 does not have
// n elements, if fixed_values and fixed_nodes differ in size, or if a fixed node is >= n or is repeated.
// Throws std::runtime_error if CG does not converge in a step.
DVector solve_transient_2d(const Mesh2D &M, double rho_c, double k, const DVector &f, const std::vector<std::size_t> &fixed_nodes, 
                           const std::vector<double> &fixed_values, const DVector &T0, double dt, std::size_t n_steps, double theta);

#endif

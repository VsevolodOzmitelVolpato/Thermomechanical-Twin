#ifndef THERMOELASTIC2D_H
#define THERMOELASTIC2D_H

#include <cstddef>
#include <vector>
#include "CSR_and_Vectors.h"
#include "Elasticity2D.h"
#include "Mesh2D.h"

struct ThermoMechanicalProblem2D{double rho_c = 0.0, k = 0.0, dt = 0.0, theta = 1.0, E = 0.0, nu = 0.0, alpha = 0.0, T_ref = 0.0;
    DVector f{}, T0{};
    std::vector<std::size_t> fixed_nodes{}, fixed_dofs{};
    std::vector<double>	fixed_temperatures{},	fixed_displacements{};
    std::size_t save_every = 1, n_steps = 0;};

struct ThermoMechanicalState2D{
    double time = 0.0;
    DVector T;
    std::vector<double> u;
    std::vector<Vector3> sigma;
};

std::vector<ThermoMechanicalState2D> solve_thermomechanical_2d(const Mesh2D &M,
                                                               const ThermoMechanicalProblem2D &problem);



#endif
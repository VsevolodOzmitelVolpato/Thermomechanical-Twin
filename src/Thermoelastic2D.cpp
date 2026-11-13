#include "Thermoelastic2D.h"
#include "Elasticity2D.h"
#include "Solver2D.h"
#include <stdexcept>
#include <vector>

namespace{
    ThermoMechanicalState2D make_state(const Mesh2D &M, const Matrix3 &D, const ThermoMechanicalProblem2D &problem,
                                       double time, const std::vector<double> &T){

        std::vector<double> F = assemble_thermal_load_2d(M, D, problem.nu, problem.alpha, T, problem.T_ref);

        std::vector<double> u = solve_displacement_2d(M, D, F, problem.fixed_dofs, problem.fixed_displacements);

        std::vector<Vector3> stress = stress_field_2d(M, D, problem.nu, problem.alpha, u, T, problem.T_ref);

        ThermoMechanicalState2D TMS = {time, T, u, stress};

        return TMS;
    }
}

std::vector<ThermoMechanicalState2D> solve_thermomechanical_2d(const Mesh2D &M,
                                                               const ThermoMechanicalProblem2D &problem){
    if(problem.save_every == 0){
        throw std::invalid_argument("the save every argument is equal to 0");}
    if(problem.n_steps % problem.save_every != 0){
        throw std::invalid_argument("the steps and the save every are not proportional");}

    Matrix3 D = plane_strain_matrix(problem.E, problem.nu);
    DVector T = problem.T0;

    std::vector<ThermoMechanicalState2D> TMS_vec;
    TMS_vec.push_back(make_state(M, D, problem, 0.0, T));

    for (std::size_t step = problem.save_every; step <= problem.n_steps; step += problem.save_every){
        T = solve_transient_2d(M, problem.rho_c, problem.k, problem.f, problem.fixed_nodes, problem.fixed_temperatures,
                               T, problem.dt, problem.save_every, problem.theta);

        ThermoMechanicalState2D TMS = make_state(M, D, problem, step * problem.dt, T);
        TMS_vec.push_back(TMS);
    }

    return TMS_vec;
}
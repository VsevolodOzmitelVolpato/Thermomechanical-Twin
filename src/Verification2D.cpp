#include "Verification2D.h"
#include <cstddef>
#include "CSR_and_Vectors.h"
#include "Mesh2D.h"
#include <stdexcept>
#include "Assembly2D.h"
#include <cmath>
#include <vector>
#include "Solver2D.h"
#include "Elasticity2D.h"

const double pi = std::acos(-1.0);

double l2_error_2d(const Mesh2D &M, const DVector &T_num, const DVector &T_ref){
    if((T_num.size() != M.num_nodes()) or (T_ref.size() != M.num_nodes())){
        throw std::invalid_argument("T_num and T_ref must have one value per node of the mesh");}

    DVector e = T_num - T_ref;

    CSR mass = assemble_mass_2d(M, 1.0);
    double l2 = std::sqrt(dot(e, mass * e));

    return l2;
}

double manufactured_sine_error_2d(std::size_t n, double k){
    if((n < 1) or !(k > 0.0)){
        throw std::invalid_argument("the input data is invalid");}

    Mesh2D M = rectangle_mesh(1.0, 1.0, n, n);
    const auto source = [k](double x, double y) {
        return 2.0 * pi * pi * k * std::sin(pi * x) * std::sin(pi * y);};

    std::vector<double> load = assemble_load_2d(M, source);
    std::vector<std::size_t> fixed = boundary_nodes(M);
    std::vector<double> zeros(fixed.size(), 0.0);

    DVector T = solve_steady_2d(M, k, load, fixed, zeros);
    DVector exact(M.num_nodes(), 0.0);

    for(std::size_t i {0}; i < M.num_nodes(); ++i){
        exact[i] = std::sin(pi * M.x(i)) * std::sin(pi * M.y(i));}

    return l2_error_2d(M, T, exact);
}

double decaying_sine_error_2d(std::size_t n, double dt, std::size_t n_steps, double theta){
    if(n < 1){
        throw std::invalid_argument("the input data is invalid");}

    Mesh2D M = rectangle_mesh(1.0, 1.0, n, n);
    DVector T0;

    for(std::size_t i {0}; i < M.num_nodes(); ++i){
        T0.push_back(sin(pi * M.x(i)) * sin(pi * M.y(i)));}
    
    std::vector<size_t> bn = boundary_nodes(M);
    std::vector<double> zeros(bn.size(), 0.0);
    DVector vec_0(M.num_nodes(), 0.0);

    DVector sol = solve_transient_2d(M, 1.0, 1.0, vec_0, bn, zeros, T0, dt, n_steps, theta);
    double t = n_steps * dt;

    DVector exact;
    for(std::size_t i {0}; i < M.num_nodes(); ++i){
        exact.push_back(std::exp(-2 * pi * pi * t) * sin(pi * M.x(i)) * sin(pi * M.y(i)));}

    return l2_error_2d(M, sol, exact);
}

std::vector<double> bending_beam_exact_displacement_2d(const Mesh2D &M, const BendingBeam &beam){
    double k = (1 + beam.nu) * beam.alpha * beam.gradient;
    std::vector<double> vect(2 * M.num_nodes(), 0.0);

    for(std::size_t i {0}; i < M.num_nodes(); ++i){
        double x = M.x(i);
        double y = M.y(i) - (beam.height) / 2;

        vect[2 * i] = k * x * y;
        vect[2 * i + 1] = k * (y * y - x * x) / 2;
    }

    return vect;
}

BendingBeamErrors bending_beam_errors_2d(const BendingBeam &beam, std::size_t nx, std::size_t ny){
    Mesh2D M = rectangle_mesh(beam.length, beam.height, nx, ny);
    Matrix3 D = plane_strain_matrix(beam.E, beam.nu);

    std::vector<double> T(M.num_nodes(), 0.0);
    for(std::size_t i {0}; i < M.num_nodes(); ++i){
        T[i] = beam.T_ref + beam.gradient * (M.y(i) - beam.height / 2);}

    std::vector<double> F = assemble_thermal_load_2d(M, D, beam.nu, beam.alpha, T, beam.T_ref);
    std::vector<double> exact = bending_beam_exact_displacement_2d(M, beam);

    std::vector<std::size_t> fixed = {0, 1, 2 * nx + 1};
    std::vector<double> values = {exact[0], exact[1], exact[2 * nx + 1]};

    std::vector<double> u = solve_displacement_2d(M, D, F, fixed, values);

    double max_err = 0.0;
    for(std::size_t i {0}; i < M.num_nodes(); ++i){
        const double du = u[2 * i] - exact[2 * i];
        const double dv = u[2 * i + 1] - exact[2 * i + 1];

        max_err = std::max(max_err, std::hypot(du, dv));
    }

    std::vector<Vector3> stress = stress_field_2d(M, D, beam.nu, beam.alpha, u, T, beam.T_ref);

    double max_stress = 0.0;
    for(const auto &s : stress){
        max_stress = std::max(max_stress, std::hypot(s[0], s[1], s[2]));
    }

    BendingBeamErrors binding = {max_err, max_stress};

    return binding;
}

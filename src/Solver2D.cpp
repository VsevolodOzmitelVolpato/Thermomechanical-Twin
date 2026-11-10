#include "Solver2D.h"
#include <cstddef>
#include <vector>
#include "Mesh2D.h"
#include "Triplets.h"
#include "CSR_and_Vectors.h"
#include "CGSolver.h"
#include "Dirichlet2D.h"
#include "Assembly2D.h"
#include <stdexcept>

DVector solve_steady_2d(const Mesh2D &M, double k,const DVector &f, const std::vector<std::size_t> &fixed_nodes,const std::vector<double> &fixed_values){
    std::size_t n = M.num_nodes();

    if(!(k > 0.0)){
        throw std::invalid_argument("K must be positive");}
    if(f.size() != n){
        throw std::invalid_argument("the size of the vector doesnt match the size of the mesh");}
    if(fixed_nodes.empty()){
        throw std::invalid_argument("the fixed nodes are empty");}

    std::vector<Triplet> triplets = stiffness_triplets_2d(M, k);

    CSR K = csr_from_triplets(n, triplets);
    CSR A = csr_from_triplets(n, dirichlet_matrix_triplets_2d(n, triplets, fixed_nodes));

    DVector b = dirichlet_rhs_2d(K, f, fixed_nodes, fixed_values);
    DVector x0(n, 0.0);

    for(std::size_t q {0}; q < fixed_nodes.size(); ++q){
        x0[fixed_nodes[q]] = fixed_values[q];}

    CGSolver CGS(A, b, x0, 1e-12, static_cast<unsigned>(10 * n));
    CGS.solve(true);

    if(!CGS.has_converged()){
        throw std::runtime_error("The matrix has not converged");}

    return CGS.get_solution();
}

namespace {
    std::vector<Triplet> combine_triplets(const std::vector<Triplet> &M_triplets, const std::vector<Triplet> &K_triplets, double factor){
         std::vector<Triplet> result = M_triplets;
        result.reserve(M_triplets.size() + K_triplets.size());

        for (const auto& tk : K_triplets) {
            result.push_back({tk.row, tk.col, tk.val * factor});}

        return result;}
};

DVector solve_transient_2d(const Mesh2D &M, double rho_c, double k, const DVector &f, const std::vector<std::size_t> &fixed_nodes, 
                           const std::vector<double> &fixed_values, const DVector &T0, double dt, std::size_t n_steps, double theta){
    std::size_t n = M.num_nodes();

    if(!(rho_c > 0.0) or !(k > 0.0) or !(dt > 0.0)){
        throw std::invalid_argument("one of the doubles (not theta) is less than 0");}
    if(!(theta >= 0.0 && theta <= 1.0)){
        throw std::invalid_argument("the value of theta is invalid");}
    if((f.size() != n) or (T0.size() != n)){
        throw std::invalid_argument("either one of the two vector doesnt match in size with the mesh");}
    if(fixed_values.size() != fixed_nodes.size()){
        throw std::invalid_argument("the fixed values and nodes are not of the same size");}
    
    std::vector<Triplet> mass_tr = mass_triplets_2d(M, rho_c);
    std::vector<Triplet> stiff_tr = stiffness_triplets_2d(M, k);
    std::vector<Triplet> trip_A = combine_triplets(mass_tr, stiff_tr, theta * dt);
    std::vector<Triplet> trip_B = combine_triplets(mass_tr, stiff_tr, -(1.0 - theta) * dt);

    CSR A = csr_from_triplets(n, trip_A);
    CSR B = csr_from_triplets(n, trip_B);
    CSR A_D = csr_from_triplets(n, dirichlet_matrix_triplets_2d(n, trip_A, fixed_nodes));

    DVector T = T0;

    for(std::size_t i {0}; i < n_steps; ++i){
        DVector rhs = B * T + dt * f;
        DVector b = dirichlet_rhs_2d(A, rhs, fixed_nodes, fixed_values);
        DVector x0 = T;

        for (std::size_t k {0}; k < fixed_nodes.size(); ++k) {
            std::size_t node = fixed_nodes[k];
            x0[node] = fixed_values[k];}

        CGSolver cg(A_D, b, x0, 1e-12, static_cast<unsigned>(10 * n));
        cg.solve(true);

        if (!cg.has_converged()){
            throw std::runtime_error("the CG has not converged");}

        T = cg.get_solution();
    }

    return T;
}
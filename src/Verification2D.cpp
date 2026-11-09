#include "Verification2D.h"
#include <cstddef>
#include "CSR_and_Vectors.h"
#include "Mesh2D.h"
#include <stdexcept>
#include "Assembly2D.h"
#include <cmath>
#include <vector>
#include "Solver2D.h"

const double pi = std::acos(-1.0);

double l2_error_2d(const Mesh2D &M, const DVector &T_num, const DVector &T_ref){
    if((T_num.size() != M.num_nodes()) or (T_ref.size() != M.num_nodes())){
        throw std::invalid_argument("either one of the two vector has not the same size of the mesh");}

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
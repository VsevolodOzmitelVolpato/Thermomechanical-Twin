#include "Fem1D.h"
#include <vector>
#include "CSR_and_Vectors.h"

std::vector<double> uniform_mesh(double L, std::size_t n_elements) {
    std::vector<double> mesh (n_elements + 1);

    for(size_t i {0}; i <= n_elements; ++i){
        mesh[i] = L*i/n_elements;}
    
    return mesh;
    
}

DMatrix assemble_stiffness(const std::vector<double>& x, double k) {
    size_t n = x.size();
    DMatrix AS (n, std::vector<double> (n, 0.0));
    for(size_t i {1}; i < n; ++i){
        double h = x[i] - x[i-1];
        AS[i-1][i-1] += k/h;
        AS[i][i] += k/h;
        AS[i-1][i] = -k/h;
        AS[i][i-1] = -k/h;}

    return AS;
}

DVector assemble_load(const std::vector<double>& x, double q) {
    size_t n = x.size();
    DVector v(n, 0.0);

    for(size_t i {1}; i < n; ++i){
        double h = x[i] - x[i-1];
        v[i-1] += h*q/2;
        v[i] += h*q/2;
    }

    return v;
}
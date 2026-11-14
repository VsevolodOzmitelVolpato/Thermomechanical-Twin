#include "Elasticity2D.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <vector>
#include "Triplets.h"
#include "CGSolver.h"
#include "Dirichlet2D.h"

Matrix3 plane_strain_matrix(double E, double nu){
    if(!(E > 0.0)){
        throw std::invalid_argument("the E values is less then zero");}
    if(!(nu > -1.0 && nu < 0.5)){
        throw std::invalid_argument("the nu value is invalid");}

    const double lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    const double mu = E / (2 * (1 + nu));

    Matrix3 D{};

    D[0][0] = lambda + 2 * mu;
    D[1][1] = lambda + 2 * mu;

    D[1][0] = lambda;
    D[0][1] = lambda;
    D[2][2] = mu;

    return D;
}

Vector3 plane_strain_thermal_strain(double nu, double alpha, double dT){
    if(!(nu > -1.0 && nu < 0.5)){
        throw std::invalid_argument("the nu value is invalid");}

    const double e = (1 + nu) * alpha * dT;

    Vector3 vec{e, e, 0.0};

    return vec;
}

Matrix3x6 strain_displacement_matrix(const Mesh2D &M, std::size_t e){
    Triangle t = M.triangle(e);
    double area = M.area(e);

    std::array<double, 3> x{}, y{}, b{}, c{};

    for(std::size_t i {0}; i < 3; ++i){
        x[i] = M.x(t[i]);
        y[i] = M.y(t[i]);
    }

    for(std::size_t i {0}; i < 3; ++i){
        size_t p = (i + 1) % 3;
        size_t q = (i + 2) % 3;

        b[i] = y[p] - y[q];
        c[i] = x[q] - x[p];
    }

    Matrix3x6 B{};

    for(std::size_t k {0}; k < 3; ++k){
        B[0][2 * k]   = b[k] / (2 * area);
        B[1][2 * k + 1] = c[k] / (2 * area);
        B[2][2 * k]   = c[k] / (2 * area);
        B[2][2 * k + 1] = b[k] / (2 * area);}

    return B;
}

Matrix6 element_elasticity_stiffness(const Mesh2D &M, std::size_t e, const Matrix3 &D){
    Matrix3x6 B = strain_displacement_matrix(M, e);
    double area = M.area(e);

    Matrix3x6 DB{};

    for(std::size_t i {0}; i < 3; ++i){
        for(std::size_t j {0}; j < 6; ++j){
            for(std::size_t k {0}; k < 3; ++k){
                DB[i][j] += D[i][k] * B[k][j];}}}

    Matrix6 K{};

        for(std::size_t i {0}; i < 6; ++i){
            for(std::size_t j {0}; j < 6; ++j){
                for(std::size_t k {0}; k < 3; ++k){
                    K[i][j] += area * B[k][i] * DB[k][j];}}}

    return K;
}

Vector6 element_thermal_load(const Mesh2D &M, std::size_t e, const Matrix3 &D, const Vector3 &eps_th){
    Matrix3x6 B = strain_displacement_matrix(M, e);
    double area = M.area(e);
    
    Vector3 sigma{};

    for(std::size_t i {0}; i < 3; ++i){
        for(std::size_t k {0}; k < 3; ++k){
            sigma[i] += D[i][k] * eps_th[k];}}

    Vector6 f{};

    for(std::size_t j{0}; j < 6; ++j){
        for(std::size_t k{0}; k < 3; ++k){
            f[j] += area * B[k][j] * sigma[k];}}

    return f;
}

std::vector<Triplet> elasticity_stiffness_triplets_2d(const Mesh2D &M, const Matrix3 &D){
    std::vector<Triplet> vec_tr;
    vec_tr.reserve(36 * M.num_triangles());

    for(std::size_t e {0}; e < M.num_triangles(); ++e){
        Triangle t = M.triangle(e);
        Matrix6 Ke = element_elasticity_stiffness(M, e, D);
        for(std::size_t i {0}; i < 6; ++i){
            for(std::size_t j {0}; j < 6; ++j){
                Triplet triplet{2 * t[i / 2] + i % 2, 2 * t[j / 2] + j % 2, Ke[i][j]};
                vec_tr.push_back(triplet);}}
    }
 
    return vec_tr;
}

CSR assemble_elasticity_stiffness_2d(const Mesh2D &M, const Matrix3 &D){
    CSR csr = csr_from_triplets(2 * M.num_nodes(), elasticity_stiffness_triplets_2d(M,D));
    return csr;
}

std::vector<double> assemble_thermal_load_2d(const Mesh2D &M, const Matrix3 &D, double nu, double alpha,
                                             const std::vector<double> &T, double T_ref){
    if(T.size() != M.num_nodes()){
        throw std::invalid_argument("The vector T doesnt match in size with the mesh");}

    std::vector<double> F(2 * M.num_nodes(), 0.0);

    for(std::size_t e {0}; e < M.num_triangles(); ++e){
        Triangle t = M.triangle(e);
        double mean_T = (T[t[0]] + T[t[1]] + T[t[2]]) / 3.0;
        double dT = mean_T - T_ref;

        Vector3 eps_th = plane_strain_thermal_strain(nu, alpha, dT);
        Vector6 f = element_thermal_load(M, e, D, eps_th);

        for(std::size_t i {0}; i < 6; ++i){
            F[2 * t[i / 2] + i % 2] += f[i];}
    }

    return F;
}

std::vector<double> solve_displacement_2d(const Mesh2D &M, const Matrix3 &D, const std::vector<double> &F,
                                          const std::vector<std::size_t> &fixed_dofs, const std::vector<double> &fixed_values){
    std::size_t N = 2 * M.num_nodes();
    if(F.size() != N){
        throw std::invalid_argument("the F vector doesnt matche the size of the mesh");}
    if(fixed_dofs.size() < 3){
        throw std::invalid_argument("the size of the dofs is less than 3");}

    std::vector<Triplet> trip = elasticity_stiffness_triplets_2d(M, D);

    double s = 0.0;
    for (const auto& t : trip) {
        s = std::max(s, std::fabs(t.val));}

    std::vector<double> load = F;

    if(s > 0.0){
        for(auto &t : trip){
            t.val = t.val / s;}
        for(auto &l : load){
            l = l / s;}}

    CSR K = csr_from_triplets(N, trip);
    CSR A = csr_from_triplets(N, dirichlet_matrix_triplets_2d(N, trip, fixed_dofs));

    DVector b = dirichlet_rhs_2d(K, load, fixed_dofs, fixed_values);

    std::vector<double> x0(N, 0.0);

    for(std::size_t q{0}; q < fixed_values.size(); ++q){
        x0[fixed_dofs[q]] = fixed_values[q];}

    CGSolver cg(A, b, x0, 1e-12, 10 * N);
    cg.solve(true);

    if(!cg.has_converged()){
        throw std::runtime_error("the CG has not converged");}

    std::vector<double> u = cg.get_solution();

    if(!(norm(b - A * u) <= 1e-6 * norm(b))){
        throw std::runtime_error("the residual is too large");}

    return u;
}

Vector3 element_stress_2d(const Mesh2D &M, std::size_t e, const Matrix3 &D, const Vector6 &u_e, const Vector3 &eps_th){
    Matrix3x6 B = strain_displacement_matrix(M, e);

    Vector3 eps{};

    for(std::size_t i {0}; i < eps.size(); ++i){
        for(std::size_t j{0}; j < 6; ++j){
            eps[i] += B[i][j] * u_e[j];}
        eps[i] -= eps_th[i];
    }

    Vector3 sigma{};

    for(std::size_t i {0}; i < sigma.size(); ++i){
        for(std::size_t k{0}; k < 3; ++k){
            sigma[i] += D[i][k] * eps[k];}}

    return sigma;
}

std::vector<Vector3> stress_field_2d(const Mesh2D &M, const Matrix3 &D, double nu, double alpha,
                                     const std::vector<double> &u, const std::vector<double> &T, double T_ref){
    if(u.size() != 2 * M.num_nodes()){
        throw std::invalid_argument("the u dimension doesnt match the mesh one");}
    if(T.size() != M.num_nodes()){
        throw std::invalid_argument("the T vector doesnt match the mesh size");}

    std::vector<Vector3> sigma(M.num_triangles());

    for(std::size_t e {0}; e < M.num_triangles(); ++e){
        const Triangle &t = M.triangle(e);
        Vector6 u_e;

        for(std::size_t i {0}; i < 6 ; ++i){
            u_e[i] = u[2 * t[i / 2] + i % 2];}

        double dT = (T[t[0]] + T[t[1]] + T[t[2]]) / 3.0 - T_ref;
        Vector3 eps_th = plane_strain_thermal_strain(nu, alpha, dT);

        sigma[e] = element_stress_2d(M, e, D, u_e, eps_th);
    }

    return sigma;
}

double von_mises_2d(const Vector3& sigma, double sigma_zz){
    double dxy = sigma[0] - sigma[1], dyz = sigma[1] - sigma_zz, dzx = sigma_zz - sigma[0];
    
    return std::sqrt(0.5 * (dxy*dxy + dyz*dyz + dzx*dzx) + 3.0 * sigma[2] * sigma[2]);
}

std::vector<double> von_mises_field_2d(const Mesh2D &M, const std::vector<Vector3> &sigma, double E, double nu,
                                       double alpha, const std::vector<double> &T, double T_ref){
    if(!(E > 0.0)){
        throw std::invalid_argument("the E value is invalid");}
    if(!(nu > -1.0 && nu < 0.5)){
        throw std::invalid_argument("the nu value is invalid");}
    if(sigma.size() != M.num_triangles()){
        throw std::invalid_argument("the sigma argument doesnt match the size of the mesh");}
    if(T.size() != M.num_nodes()){
        throw std::invalid_argument("the T argument doesnt match the size of the mesh");}

    std::vector<double> vm(M.num_triangles());

    for(std::size_t e {0}; e < M.num_triangles() ; ++e){
        const Triangle t = M.triangle(e);

        double dT = (T[t[0]] + T[t[1]] + T[t[2]]) / 3.0 - T_ref;
        double sigma_zz = nu * (sigma[e][0] + sigma[e][1]) - E * alpha * dT;

        vm[e] = von_mises_2d(sigma[e], sigma_zz);
    }

    return vm;
}
#include "Elasticity2D.h"
#include <stdexcept>

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
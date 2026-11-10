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
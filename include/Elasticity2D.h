#ifndef ELASTICITY2D_H
#define ELASTICITY2D_H

#include "Fem2D.h"

Matrix3 plane_strain_matrix(double E, double nu);
Vector3 plane_strain_thermal_strain(double nu, double alpha, double dT);

#endif
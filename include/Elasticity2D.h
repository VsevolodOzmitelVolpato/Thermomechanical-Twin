#ifndef ELASTICITY2D_H
#define ELASTICITY2D_H

#include "Fem2D.h"
#include <array>
#include "Triplets.h"
#include "CSR_and_Vectors.h"
#include <vector>

using Matrix3x6 = std::array<std::array<double, 6>, 3>;
using Vector6 = std::array<double, 6>;
using Matrix6 = std::array<std::array<double, 6>, 6>;

Matrix3 plane_strain_matrix(double E, double nu);

Vector3 plane_strain_thermal_strain(double nu, double alpha, double dT);

Matrix3x6 strain_displacement_matrix(const Mesh2D& M, std::size_t e);

Matrix6 element_elasticity_stiffness(const Mesh2D& mesh, std::size_t e, const Matrix3& D);

Vector6 element_thermal_load(const Mesh2D &M, std::size_t e, const Matrix3 &D, const Vector3 &eps_th);

std::vector<Triplet> elasticity_stiffness_triplets_2d(const Mesh2D &M, const Matrix3 &D);

CSR assemble_elasticity_stiffness_2d(const Mesh2D &M, const Matrix3 &D);

std::vector<double> assemble_thermal_load_2d(const Mesh2D &M, const Matrix3 &D, double nu, double alpha,
                                             const std::vector<double> &T, double T_ref);

std::vector<double> solve_displacement_2d(const Mesh2D &M, const Matrix3 &D, const std::vector<double> &F,
                                          const std::vector<std::size_t> &fixed_dofs, const std::vector<double> &fixed_values);

Vector3 element_stress_2d(const Mesh2D &M, std::size_t e, const Matrix3 &D, const Vector6 &u_e, const Vector3 &eps_th);

std::vector<Vector3> stress_field_2d(const Mesh2D &M, const Matrix3 &D, double nu, double alpha,
                                     const std::vector<double> &u, const std::vector<double> &T, double T_ref);

double von_mises_2d(const Vector3& sigma, double sigma_zz);

std::vector<double> von_mises_field_2d(const Mesh2D& mesh, const std::vector<Vector3>& sigma, double E, double nu,
                                       double alpha, const std::vector<double>& T, double T_ref);

#endif
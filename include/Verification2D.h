#ifndef VERIFICATION2D_H
#define VERIFICATION2D_H

#include <cstddef>
#include "CSR_and_Vectors.h"
#include "Mesh2D.h"
#include <cmath>
#include <vector>

struct BendingBeam { double length; double height; double E; double nu; double alpha; double gradient; double T_ref;};
struct BendingBeamErrors { double displacement; double stress;};

double l2_error_2d(const Mesh2D &M, const DVector &T_num, const DVector &T_ref);

double manufactured_sine_error_2d(std::size_t n, double k);

double decaying_sine_error_2d(std::size_t n, double dt, std::size_t n_steps, double theta);

std::vector<double> bending_beam_exact_displacement_2d(const Mesh2D &M, const BendingBeam &beam);

BendingBeamErrors bending_beam_errors_2d(const BendingBeam &beam, std::size_t nx, std::size_t ny);

#endif
#ifndef VERIFICATION2D_H
#define VERIFICATION2D_H

#include <cstddef>
#include "CSR_and_Vectors.h"
#include "Mesh2D.h"
#include <cmath>

double l2_error_2d(const Mesh2D &M, const DVector &T_num, const DVector &T_ref);

double manufactured_sine_error_2d(std::size_t n, double k);

double decaying_sine_error_2d(std::size_t n, double dt, std::size_t n_steps, double theta);


#endif

#ifndef FEM1D_H
#define FEM1D_H

#include <cstddef>
#include <vector>

#include "CSR_and_Vectors.h"   // per DVector

typedef std::vector<std::vector<double>> DMatrix;

std::vector<double> uniform_mesh(double L, std::size_t n_elements);

DMatrix assemble_stiffness(const std::vector<double>& x, double k);

DVector assemble_load(const std::vector<double>& x, double q);

#endif
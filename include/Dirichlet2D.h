#ifndef DIRICHLET2D_H
#define DIRICHLET2D_H

#include <cstddef>
#include <vector>
#include "Triplets.h"
#include "CSR_and_Vectors.h"

// Matrix side of the Dirichlet conditions, by symmetric elimination. Returns the triplets of the matrix with the fixed
// nodes eliminated: the triplets whose row or column is a fixed node are dropped and the triplet (d, d, 1) is added for
// every fixed node d. Zeros are neither added nor removed, and the list received is not changed.
// Throws std::invalid_argument if a fixed node is >= n, if a fixed node is repeated (the two triplets (d, d, 1) would
// add up to 2), or if a triplet has a row or column >= n.
std::vector<Triplet> dirichlet_matrix_triplets_2d(std::size_t n, const std::vector<Triplet> &T, const std::vector<std::size_t> &fixed_nodes);

// Right-hand side that goes with the matrix above. At a fixed node the entry is the imposed value; at a free node i it
// is f[i] minus the contribution of the fixed nodes (K times the vector that holds the imposed values and zeros
// elsewhere). K and f are not changed.
// Throws std::invalid_argument in four cases: K and f have different sizes; the number of values differs from the
// number of fixed nodes; a fixed node is >= n; a fixed node appears more than once (even if the value is the same).
DVector dirichlet_rhs_2d(const CSR &csr, const DVector &f, const std::vector<std::size_t> &fixed_nodes, const std::vector<double> &fixed_values);

#endif

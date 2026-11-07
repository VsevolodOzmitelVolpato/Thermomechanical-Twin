#ifndef DIRICHLET2D_H
#define DIRICHLET2D_H

#include <cstddef>
#include <vector>
#include "Triplets.h"

//throws out triplets with fixed row or column, adds (d, d, 1)), and it doesn't add, doesn't remove zeros, doesn't change the received list),
// and when it throws std::invalid_argument (fixed node >= n, repeated fixed node, triplet with row or column >= n).
std::vector<Triplet> dirichlet_matrix_triplets_2d(std::size_t n, const std::vector<Triplet> &T, const std::vector<std::size_t> &fixed_nodes);



#endif
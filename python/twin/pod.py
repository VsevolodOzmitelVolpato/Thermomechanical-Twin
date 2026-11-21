"""Proper orthogonal decomposition (POD) of temperature snapshots."""

import operator
import numpy as np

RANK_TOLERANCE = 1e-6

def pod_basis(snapshots, mass, n_modes) :
    """POD basis of a set of snapshots, orthonormal in the inner product of the mass matrix.

    snapshots  (n, m) array, one temperature field (n nodes) per column
    mass       (n, n) mass matrix: the inner product of two fields a and b is  a^T mass b
    n_modes    number of modes to keep, between 1 and m

    Method of the snapshots: the eigenvectors of the small (m, m) matrix  S^T mass S  give the modes, and the square
    roots of its eigenvalues are the singular values. The modes are then orthonormalised again (Cholesky factor of
    their Gram matrix), which removes the round-off of the first step.

    Returns (basis, sigma): basis is (n, n_modes), one mode per column, with  basis^T mass basis = identity;
    sigma holds the m singular values, largest first.
    Raises ValueError for wrong shapes, NaN or Inf, an n_modes outside [1, m], or when the singular value of the
    last mode is not larger than 1e-6 times the first (the snapshots do not hold that many independent modes).
    """
    n_modes = operator.index(n_modes)
    S = np.array(snapshots, dtype=float)

    if S.ndim != 2 :
        raise ValueError("snapshots must be a 2-D array")

    n, m = S.shape

    if not (np.all(np.isfinite(S))):
        raise ValueError("snapshots contain NaN or Inf")
    if ( mass.shape != (n, n) ):
        raise ValueError("mass must have shape (n, n), where n is the number of rows of snapshots")
    if n_modes < 1 or n_modes > m:
        raise ValueError(f"n_modes must be between 1 and {m}")

    gram = S.T @ (mass @ S)

    eigenvalues, vectors = np.linalg.eigh(gram)
    eigenvalues = eigenvalues[::-1]
    vectors = vectors[:, ::-1]

    eigenvalues = np.maximum(eigenvalues, 0.0)
    sigma = np.sqrt(eigenvalues)

    if sigma[n_modes - 1] <= RANK_TOLERANCE * sigma[0]:
        raise ValueError("the last mode is negligible compared with the first: the snapshots do not hold that many independent modes")

    basis = S @ vectors[:, :n_modes] / sigma[:n_modes]

    G = basis.T @ (mass @ basis)
    L = np.linalg.cholesky(G)
    basis = np.linalg.solve(L, basis.T).T

    return basis, sigma

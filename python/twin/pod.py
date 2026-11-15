import operator
import numpy as np

RANK_TOLERANCE = 1e-6

def pod_basis(snapshots, mass, n_modes) :
    n_modes = operator.index(n_modes)
    S = np.array(snapshots, dtype=float)

    if S.ndim != 2 :
        raise ValueError("S dimentions are not in R**2")

    n, m = S.shape

    if not (np.all(np.isfinite(S))):
        raise ValueError("the array contains NaN or Inf")
    if ( mass.shape != (n, n) ):
        raise ValueError("the mass size is not")
    if n_modes < 1 or n_modes > m:
        raise ValueError(f"n_modes should be between 1 e {m}")

    gram = S.T @ (mass @ S)

    eigenvalues, vectors = np.linalg.eigh(gram)
    eigenvalues = eigenvalues[::-1]
    vectors = vectors[:, ::-1]

    eigenvalues = np.maximum(eigenvalues, 0.0)
    sigma = np.sqrt(eigenvalues)

    if sigma[n_modes - 1] <= RANK_TOLERANCE * sigma[0]:
        raise ValueError("the last element of the sigma is below the tolerance")

    basis = S @ vectors[:, :n_modes] / sigma[:n_modes]

    G = basis.T @ (mass @ basis)
    L = np.linalg.cholesky(G)
    basis = np.linalg.solve(L, basis.T).T

    return basis, sigma
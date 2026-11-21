"""Galerkin projection of the heat equation on a basis of thermal modes.

The imposed temperatures (dirichlet_lift), the time stepping of the small system (reduced_theta_scheme), the way back
to the nodes (reconstruct) and the three steps together (solve_reduced_heat).
"""

import numpy as np
import operator

def dirichlet_lift(num_nodes, fixed_nodes, fixed_values):
    """Vector of num_nodes entries with the imposed temperatures at fixed_nodes and zeros elsewhere.

    It is the part of the temperature that is known: the temperature is  T = lift + (a combination of modes), and the
    modes vanish on the fixed nodes, so T has the imposed value there whatever the combination is.
    Raises ValueError if num_nodes < 1, if fixed_values is not 1-D or has a different length from fixed_nodes, or if a
    node is out of range or repeated.
    """
    num_nodes = operator.index(num_nodes)
    if num_nodes < 1:
        raise ValueError("num_nodes must be at least 1")

    nodes = [operator.index(node) for node in fixed_nodes]

    values = np.array(fixed_values, dtype=float)

    if values.ndim != 1 :
            raise ValueError("fixed_values must be a 1-D array")
    if values.shape[0] != len(nodes):
        raise ValueError("fixed_values and fixed_nodes must have the same length")

    seen = set()
    for node in nodes:
        if not (0 <= node < num_nodes):
            raise ValueError(f"node {node} is out of range [0, {num_nodes - 1}]")
        if node in seen:
            raise ValueError(f"node {node} is repeated")
        seen.add(node)

    vect = np.zeros(num_nodes)
    vect[nodes] = values

    return vect
    
def reduced_theta_scheme(mass_r, stiffness_r, load_r, a0, dt, n_steps, theta):
    """Theta scheme for the small system  mass_r a' + stiffness_r a = load_r.

    Every step solves
        (mass_r + theta dt stiffness_r) a_new = (mass_r - (1 - theta) dt stiffness_r) a_old + dt load_r
    (theta = 1: backward Euler, theta = 0.5: Crank-Nicolson). The matrices do not change, so the step
    a_new = P a_old + c  is built once, before the loop.

    mass_r, stiffness_r  (r, r) matrices
    load_r, a0           (r,) vectors: the load and the starting values of a
    dt, n_steps, theta   dt > 0, n_steps >= 0, 0 <= theta <= 1
    Returns the values of a at every step, shape (n_steps + 1, r); row 0 is a0.
    Raises ValueError for inconsistent shapes or invalid dt, theta or n_steps.
    """
    n_steps = operator.index(n_steps)

    mass_r = np.array(mass_r, dtype=float)
    stiffness_r = np.array(stiffness_r, dtype=float)
    load_r = np.array(load_r, dtype=float)
    a0 = np.array(a0, dtype=float)

    if mass_r.ndim != 2 :
        raise ValueError("mass_r must be a 2-D array")

    n = mass_r.shape[0]
    if mass_r.shape != (n, n):
        raise ValueError("mass_r must be a square matrix")
    if stiffness_r.shape != (n, n):
        raise ValueError("stiffness_r must have the same shape as mass_r")
    if a0.shape != (n,) :
        raise ValueError("a0 must have one entry per row of mass_r")
    if load_r.shape != (n,) :
        raise ValueError("load_r must have one entry per row of mass_r")

    if not dt > 0:
        raise ValueError("dt must be positive")
    if not 0 <= theta <= 1:
        raise ValueError("theta must be between 0 and 1")
    if n_steps < 0:
        raise ValueError("n_steps must not be negative")

    lhs = mass_r + theta * dt * stiffness_r
    rhs = mass_r - (1 - theta) * dt * stiffness_r

    P = np.linalg.solve(lhs, rhs)
    c = np.linalg.solve(lhs, dt * load_r)

    coordinates = np.empty((n_steps + 1, n), dtype=float)

    coordinates[0] = a0

    for step in range(n_steps):
        coordinates[step + 1] = P @ coordinates[step] + c

    return coordinates

def reconstruct(basis, coordinates, lift):
    """Back from the coordinates to the nodes:  lift + basis * coordinates.

    basis (n, r), lift (n,), coordinates (r,) for one state or (m, r) for m states.
    Returns an array of shape (n,) or (m, n). Raises ValueError for inconsistent shapes.
    """
    basis = np.asarray(basis, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    lift = np.asarray(lift, dtype=float)

    if basis.ndim != 2 :
        raise ValueError("basis must be a 2-D array")
    if lift.shape != (basis.shape[0],):
        raise ValueError("lift must have one entry per row of basis")
    if coordinates.ndim not in (1, 2) or coordinates.shape[-1] != basis.shape[1]:
        raise ValueError("coordinates must be 1-D or 2-D, with as many entries per row as basis has columns")

    return lift + coordinates @ basis.T

def solve_reduced_heat( basis, mass, stiffness, load, fixed_nodes, fixed_values, T0, dt, n_steps, theta):
    """Heat equation on the reduced basis: project, march in time, go back to the nodes.

    basis                      (n, r) thermal modes, one per column; they must vanish on the fixed nodes
    mass, stiffness, load      (n, n), (n, n) and (n,): the matrices and the load vector of the full problem
    fixed_nodes, fixed_values  the nodes with an imposed temperature, and the temperatures
    T0                         (n,) temperature at time 0
    dt, n_steps, theta         time step, number of steps and theta of the theta scheme

    The temperature is  T = lift + basis a,  where lift holds the imposed values (see dirichlet_lift). Projecting
    mass T' + stiffness T = load  on the basis gives  M_r a' + K_r a = f_r  with  M_r = B^T mass B,
    K_r = B^T stiffness B,  f_r = B^T (load - stiffness lift),  and the starting a is the fit of T0 - lift by the basis
    in the mass inner product.
    Returns the temperature at every step, shape (n_steps + 1, n).
    Raises ValueError for inconsistent shapes or for a basis that does not vanish on the fixed nodes.
    """
    basis = np.array(basis, dtype=float)
    load = np.array(load, dtype=float)
    T0 = np.array(T0, dtype=float)

    if basis.ndim != 2:
        raise ValueError(f"basis must be a 2-D array")

    n = basis.shape[0]

    if mass.shape != (n, n):
        raise ValueError(f"mass must have shape (n, n), where n is the number of rows of basis")
    if stiffness.shape != (n, n):
        raise ValueError(f"stiffness must have shape (n, n), where n is the number of rows of basis")
    if load.shape != (n,):
        raise ValueError(f"load must have {n} entries")

    if T0.shape != (n,):
        raise ValueError(f"T0 must have {n} entries")

    lift = dirichlet_lift(n, fixed_nodes, fixed_values)

    if len(fixed_nodes) > 0:
        fixed_idx = list(fixed_nodes)

        max_fixed = np.max(np.abs(basis[fixed_idx]))
        max_total = np.max(np.abs(basis))

        if max_fixed > 1e-8 * max_total:
            raise ValueError("the basis must vanish at the fixed nodes")

    M_r = basis.T @ (mass @ basis)
    K_r = basis.T @ (stiffness @ basis)
    f_r = basis.T @ (load - stiffness @ lift)

    rhs_a0 = basis.T @ (mass @ (T0 - lift))
    a0 = np.linalg.solve(M_r, rhs_a0)

    coordinates = reduced_theta_scheme(M_r, K_r, f_r, a0, dt, n_steps, theta)

    return reconstruct(basis, coordinates, lift)

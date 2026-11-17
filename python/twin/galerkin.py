import numpy as np
import operator

def dirichlet_lift(num_nodes, fixed_nodes, fixed_values):
    num_nodes = operator.index(num_nodes)
    if num_nodes < 1:
        raise ValueError("the lenhgh of the nodes is invalid")

    nodes = [operator.index(node) for node in fixed_nodes]

    values = np.array(fixed_values, dtype=float)

    if values.ndim != 1 :
            raise ValueError("the values are not in R1")
    if values.shape[0] != len(nodes):
        raise ValueError("The values and the number of nodes dont match in size")

    seen = set()
    for node in nodes:
        if not (0 <= node < num_nodes):
            raise ValueError(f"the {node} is out of range [0, {num_nodes - 1}]")
        if node in seen:
            raise ValueError(f"{node} node is a duplicate")
        seen.add(node)

    vect = np.zeros(num_nodes)
    vect[nodes] = values

    return vect
    
def reduced_theta_scheme(mass_r, stiffness_r, load_r, a0, dt, n_steps, theta):
    n_steps = operator.index(n_steps)

    mass_r = np.array(mass_r, dtype=float)
    stiffness_r = np.array(stiffness_r, dtype=float)
    load_r = np.array(load_r, dtype=float)
    a0 = np.array(a0, dtype=float)

    if mass_r.ndim != 2 :
        raise ValueError("the mass matrix dimention is not 2")

    n = mass_r.shape[0]
    if mass_r.shape != (n, n):
        raise ValueError("the mass matrix is not NxN")
    if stiffness_r.shape != (n, n):
        raise ValueError("the stiffness matrix is not NxN")
    if a0.shape != (n,) :
        raise ValueError("a0 element has different dimention than the other variables")
    if load_r.shape != (n,) :
        raise ValueError("load element has different dimention than the other variables")

    if not dt > 0:
        raise ValueError("dt element must be positive")
    if not 0 <= theta <= 1:
        raise ValueError("theta element must be between 1 and 0")
    if n_steps < 0:
        raise ValueError("the number of steps must be positive")

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
    basis = np.asarray(basis, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    lift = np.asarray(lift, dtype=float)

    if basis.ndim != 2 :
        raise ValueError("the bases matrix dimention is not 2")
    if lift.shape != (basis.shape[0],):
        raise ValueError("the lift shape is not congruent with thr basis one")
    if coordinates.ndim not in (1, 2) or coordinates.shape[-1] != basis.shape[1]:
        raise ValueError("the coordinates iether dosent match the size of the basis or are they havent a diomention of 1 or 2")

    return lift + coordinates @ basis.T

def solve_reduced_heat( basis, mass, stiffness, load, fixed_nodes, fixed_values, T0, dt, n_steps, theta):
    basis = np.array(basis, dtype=float)
    load = np.array(load, dtype=float)
    T0 = np.array(T0, dtype=float)

    if basis.ndim != 2:
        raise ValueError(f"basis must be a 2 dim array")

    n = basis.shape[0]

    if mass.shape != (n, n):
        raise ValueError(f"mass is not a NxN matrix")
    if stiffness.shape != (n, n):
        raise ValueError(f"stiffness is not a NxN matrix")
    if load.shape != (n,):
        raise ValueError(f"load must be {n} dimentional")

    if T0.shape != (n,):
        raise ValueError(f"T0 must be {n} dimentional")

    lift = dirichlet_lift(n, fixed_nodes, fixed_values)

    if len(fixed_nodes) > 0:
        fixed_idx = list(fixed_nodes)

        max_fixed = np.max(np.abs(basis[fixed_idx]))
        max_total = np.max(np.abs(basis))

        if max_fixed > 1e-8 * max_total:
            raise ValueError("POD basis must be null on the fixed nodes")

    M_r = basis.T @ (mass @ basis)
    K_r = basis.T @ (stiffness @ basis)
    f_r = basis.T @ (load - stiffness @ lift)

    rhs_a0 = basis.T @ (mass @ (T0 - lift))
    a0 = np.linalg.solve(M_r, rhs_a0)

    coordinates = reduced_theta_scheme(M_r, K_r, f_r, a0, dt, n_steps, theta)

    return reconstruct(basis, coordinates, lift)
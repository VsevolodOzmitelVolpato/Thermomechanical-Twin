"""Mechanics of the reduced model.

Means over the triangles, the von Mises stress of plane strain, and the displacement and stress that go with every
thermal mode.
"""

import numpy as np
import thermomech

def element_mean(triangles, values):
    """Mean of nodal values over the 3 nodes of every triangle.

    triangles (n_tri, 3) node numbers; values (n,) for one set of nodal values or (m, n) for m sets.
    Returns (n_tri,) or (m, n_tri). Raises ValueError for wrong shapes or node numbers out of range.
    """
    triangles = np.asarray(triangles)
    values = np.asarray(values, dtype=float)

    if triangles.ndim != 2:
        raise ValueError("triangles must be a 2-D array")
    if triangles.shape[1] != 3:
        raise ValueError("triangles must have 3 columns")
    if values.ndim not in (1, 2):
        raise ValueError("values must be a 1-D or 2-D array")
    if np.any((triangles < 0) | (triangles >= values.shape[-1])):
        raise ValueError("triangles refer to nodes that do not exist")

    element_values = (
        values[..., triangles[:, 0]]
        + values[..., triangles[:, 1]]
        + values[..., triangles[:, 2]]
    ) / 3

    return element_values

def von_mises_plane_strain(sigma, temperature_rise, E, nu, alpha):
    """Von Mises stress of plane strain, including the stress along z.

    sigma (..., 3) holds (xx, yy, xy); temperature_rise has the shape of sigma without the last axis (T - T_ref of
    every triangle). The strain along z is zero, so the stress along z is  zz = nu (xx + yy) - E alpha temperature_rise,
    and  vm = sqrt( 1/2 [ (xx - yy)^2 + (yy - zz)^2 + (zz - xx)^2 ] + 3 xy^2 ).
    Returns an array with the shape of temperature_rise. Raises ValueError if E <= 0, if nu is not in (-1, 0.5) or if
    the shapes do not match.
    """
    if not E > 0.0 or not -1.0 < nu < 0.5:
        raise ValueError("E must be positive and nu between -1 and 0.5 (exclusive)")

    sigma = np.asarray(sigma, dtype=float)
    temperature_rise = np.asarray(temperature_rise, dtype=float)
    if sigma.ndim < 1 or sigma.shape[-1] != 3:
        raise ValueError("the last axis of sigma must have length 3")
    if temperature_rise.shape != sigma.shape[:-1]:
        raise ValueError("temperature_rise must have the shape of sigma without its last axis")

    xx, yy, xy = sigma[..., 0], sigma[..., 1], sigma[..., 2]

    zz = nu * (xx + yy) - E * alpha * temperature_rise

    vm_sq = 0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3.0 * (xy ** 2)

    return np.sqrt(np.maximum(vm_sq, 0.0))

def thermoelastic_response(mesh, temperature, E, nu, alpha, T_ref, fixed_dofs, fixed_displacements):
    """Displacement and stress of the plate for a given temperature, with one call to the C++ model.

    temperature (n,) nodal temperature; E, nu, alpha, T_ref the material; fixed_dofs and fixed_displacements the
    supports (the degree of freedom 2 I is the x displacement of node I, and 2 I + 1 the y displacement).
    thermomech.solve_thermomechanical_2d is run with no time steps, so the state it returns is the initial one and
    only the elasticity problem is solved.
    Returns (u, sigma): u has shape (2 n,) and sigma (n_tri, 3).
    """
    n = mesh.num_nodes
    temperature = np.array(temperature, dtype=float)

    if temperature.shape != (n,):
        raise ValueError("temperature must have one value per node")

    state = thermomech.solve_thermomechanical_2d(mesh, 1.0, 1.0, np.zeros(n), [], [], temperature, 1.0, 0, 1.0, 1, E, nu, 
                                             alpha, T_ref, fixed_dofs, fixed_displacements)

    u = np.array(state["u"][0], dtype=float, copy=True)
    sigma = np.array(state["sigma"][0], dtype=float, copy=True)

    return u, sigma

def mechanical_modes(mesh, basis, lift, E, nu, alpha, T_ref, fixed_dofs, fixed_displacements):
    """Displacement and stress modes that go with the thermal modes.

    Displacement and stress are linear in the temperature, so it is enough to solve the elasticity problem for the lift
    and for every mode (r + 1 solves):
        u(lift + B a) = u_lift + u_modes a        sigma(lift + B a) = sigma_lift + sigma_modes a
    where the modes are the differences from the response to the lift.

    basis (n, r) and lift (n,) are the thermal modes and the imposed-temperature part; the material and the supports
    are as in thermoelastic_response.
    Returns (u_lift, u_modes, sigma_lift, sigma_modes) with shapes (2 n,), (2 n, r), (3 n_tri,) and (3 n_tri, r); the
    stress arrays are flattened over (triangle, component).
    """
    n = mesh.num_nodes
    n_tri = mesh.num_triangles

    basis = np.array(basis, dtype=float)
    lift = np.array(lift, dtype= float)

    if basis.ndim != 2:
        raise ValueError("basis must be a 2-D array")
    if basis.shape[0] != n:
        raise ValueError("basis must have one row per node")
    if lift.shape != (n,):
        raise ValueError("lift must have one entry per node")

    r = basis.shape[1]

    u_lift, sigma_lift_2d = thermoelastic_response(
        mesh, lift, E, nu, alpha, T_ref, fixed_dofs, fixed_displacements)
    sigma_lift = sigma_lift_2d.ravel()

    u_modes = np.empty((2 * n, r), dtype=float)
    sigma_modes = np.empty((3 * n_tri, r), dtype=float)

    for i in range(r):
        u_i, sigma_i_2d = thermoelastic_response(
            mesh, lift + basis[:, i], E, nu, alpha, T_ref, fixed_dofs, fixed_displacements)
        
        u_modes[:, i] = u_i - u_lift
        sigma_modes[:, i] = sigma_i_2d.ravel() - sigma_lift

    return u_lift , u_modes, sigma_lift, sigma_modes

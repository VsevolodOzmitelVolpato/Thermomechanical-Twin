import numpy as np
import thermomech

def element_mean(triangles, values):
    triangles = np.asarray(triangles)
    values = np.asarray(values, dtype=float)

    if triangles.ndim != 2:
        raise ValueError("the dimention of the triangles is not 2")
    if triangles.shape[1] != 3:
        raise ValueError("the traingles have not a shape of 3 elements")
    if values.ndim not in (1, 2):
        raise ValueError("the dimension of the values is different either from 1 or 2")
    for tr in triangles:
        for i in tr:
            if i < 0 or i >= values.shape[-1]:
                raise ValueError("either one or more nodes are invalid")

    element_values = ( values[..., triangles[:, 0]] + values[..., triangles[:, 1]] + values[..., triangles[:, 2]]) / 3

    return element_values

def von_mises_plane_strain(sigma, temperature_rise, E, nu, alpha):
    if not E > 0.0 or not -1.0 < nu < 0.5:
        raise ValueError("either E or nu are invalid data")

    sigma = np.asarray(sigma, dtype=float)
    temperature_rise = np.asarray(temperature_rise, dtype=float)
    if sigma.ndim < 1 or sigma.shape[-1] != 3:
        raise ValueError("the final axis of sigma is not 3")
    if temperature_rise.shape != sigma.shape[:-1]:
        raise ValueError("the temperature rise element has different number of elements in respect to sigma")

    xx, yy, xy = sigma[..., 0], sigma[..., 1], sigma[..., 2]

    zz = nu * (xx + yy) - E * alpha * temperature_rise

    vm_sq = 0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3.0 * (xy ** 2)

    return np.sqrt(np.maximum(vm_sq, 0.0))

def thermoelastic_response(mesh, temperature, E, nu, alpha, T_ref, fixed_dofs, fixed_displacements):
    n = mesh.num_nodes
    temperature = np.array(temperature, dtype=float)

    if temperature.shape != (n,):
        raise ValueError("the size of the temperature is invalid")

    state = thermomech.solve_thermomechanical_2d(mesh, 1.0, 1.0, np.zeros(n), [], [], temperature, 1.0, 0, 1.0, 1, E, nu, 
                                             alpha, T_ref, fixed_dofs, fixed_displacements)

    u = np.array(state["u"][0], dtype=float, copy=True)
    sigma = np.array(state["sigma"][0], dtype=float, copy=True)

    return u, sigma

def mechanical_modes(mesh, basis, lift, E, nu, alpha, T_ref, fixed_dofs, fixed_displacements):
    n = mesh.num_nodes
    n_tri = mesh.num_triangles

    basis = np.array(basis, dtype=float)
    lift = np.array(lift, dtype= float)

    if basis.ndim != 2:
        raise ValueError("basis is not 2 dimentional")
    if basis.shape[0] != n:
        raise ValueError("basis has an invalid dimention")
    if lift.shape != (n,):
        raise ValueError("lift has an invalid dimention")

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
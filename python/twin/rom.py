"""Reduced-order model of the thermo-mechanical plate: the heat equation on a POD basis, and the displacement and
stress modes that go with it."""

import operator

import numpy as np
import thermomech

from twin.galerkin import dirichlet_lift, reconstruct, reduced_theta_scheme
from twin.mechanics import element_mean, mechanical_modes, von_mises_plane_strain

BASIS_TOLERANCE = 1e-8


class ThermoMechanicalROM:
    """Reduced-order model of the 2D thermo-mechanical plate.

    The offline phase (the constructor) projects the heat equation on a thermal basis and computes the displacement
    and stress modes with the C++ model; the online phase (solve) then needs only small dense systems and matrix
    products. The five numbers of the study (k, rho_c, q, E, alpha) enter only in the online phase, so one object
    serves all of them. num_modes is the number of thermal modes.
    """

    def __init__(self, mesh, basis, load, fixed_nodes, fixed_values, T0, fixed_dofs, nu, T_ref):
        """OFFLINE phase.

        basis        (n, r) thermal modes, one per column; they must vanish on the fixed nodes
        load         (n,) nodal load of the heat source for q = 1 (the source q is then q * load)
        fixed_nodes  the nodes with an imposed temperature, fixed_values their temperatures
        T0           (n,) temperature at time 0
        fixed_dofs   displacement degrees of freedom held at zero (node I has 2 I for x and 2 I + 1 for y)
        nu, T_ref    Poisson ratio and reference (stress-free) temperature
        """
        n = mesh.num_nodes

        basis = np.array(basis, dtype=float)
        load = np.array(load, dtype=float)
        T0 = np.array(T0, dtype=float)

        if basis.ndim != 2 or basis.shape[0] != n or basis.shape[1] < 1:
            raise ValueError("basis must have shape (n, r) with r >= 1, n = %d" % n)
        if load.shape != (n,) or T0.shape != (n,):
            raise ValueError("load and T0 must have n entries, n = %d" % n)
        
        lift = dirichlet_lift(n, fixed_nodes, fixed_values)

        if not (np.isfinite(basis).all() and np.isfinite(load).all() and np.isfinite(T0).all()
                and np.isfinite(lift).all()):
            raise ValueError("basis, load, T0 and fixed_values must be finite")
        
        fixed = [operator.index(node) for node in fixed_nodes]
        if fixed and np.abs(basis[fixed]).max() > BASIS_TOLERANCE * np.abs(basis).max():
            raise ValueError("the basis must vanish at the fixed nodes")

        # The full equation is  rho_c M1 T' + k K1 T = q f1  with  T = lift + B a  (B = basis).
        # Multiplying by B^T gives the equation of the r numbers a:
        #     rho_c (B^T M1 B) a' + k (B^T K1 B) a = q (B^T f1) - k (B^T K1 lift)
        # Every bracket is free of parameters, so it is computed here, once.
        mass = thermomech.assemble_mass_2d(mesh, 1.0)            # M1: the mass matrix for rho_c = 1
        stiffness = thermomech.assemble_stiffness_2d(mesh, 1.0)  # K1: the stiffness matrix for k = 1
        self._mass_r = basis.T @ (mass @ basis)
        self._stiffness_r = basis.T @ (stiffness @ basis)
        self._load_r = basis.T @ load
        self._lift_r = basis.T @ (stiffness @ lift)  # the imposed temperatures push on the free part of the plate
        # Starting numbers a0: the best fit of T0 - lift by the basis, in the mass inner product.
        # (rho_c would multiply both sides and cancel, so a0 does not depend on it.)
        self._a0 = np.linalg.solve(self._mass_r, basis.T @ (mass @ (T0 - lift)))

        # ---- 3. Mechanics: the displacement and stress modes (the only C++ solves) ---------------------------
        # The plate is in equilibrium for every temperature: K u = F. K is proportional to E and F to E alpha, so E
        # cancels and u is proportional to alpha. The stress is E times a strain, so it is proportional to E alpha.
        # Hence one solve with E = 1 and alpha = 1 is enough for every material.
        # u and sigma are also linear in the temperature, so r + 1 solves (the lift and each mode) give the modes:
        #     u(lift + B a) = u_lift + sum_i a_i u_modes_i        (the same for sigma)
        # The supports must keep the displacement at 0: a non-zero value would not be proportional to alpha.
        dofs = list(fixed_dofs)
        self._u_lift, self._u_modes, self._s_lift, self._s_modes = mechanical_modes(
            mesh, basis, lift, 1.0, nu, 1.0, T_ref, dofs, [0.0] * len(dofs))

        # ---- 4. Keep what the online phase needs -------------------------------------------------------------
        self._basis = basis
        self._lift = lift
        self._triangles = np.array(mesh.triangles)  # a copy: the mesh is not needed any more
        self._nu = nu
        self._T_ref = T_ref
        self.num_modes = basis.shape[1]

    def solve(self, k, rho_c, q, E, alpha, dt, n_steps, theta, save_every=1):
        """ONLINE phase: one run for the given numbers.

        Returns the same dict as thermomech.solve_thermomechanical_2d, with m = n_steps / save_every + 1 states:
            "time" (m,)   "T" (m, n)   "u" (m, 2 n)   "sigma" (m, n_tri, 3)   "von_mises" (m, n_tri)
        The object is not changed: solve can be called as often as you like, in any order.
        """
        # ---- 1. Check the numbers ----------------------------------------------------------------------------
        if not all(np.isfinite(value) for value in (k, rho_c, q, E, alpha, dt)):
            raise ValueError("k, rho_c, q, E, alpha and dt must be finite numbers")
        if not (k > 0.0 and rho_c > 0.0 and E > 0.0 and dt > 0.0):
            raise ValueError("k, rho_c, E and dt must be positive")
        # operator.index accepts integers only (a float such as 3.0 is refused).
        n_steps = operator.index(n_steps)
        save_every = operator.index(save_every)
        if save_every < 1 or n_steps % save_every != 0:
            raise ValueError("save_every must be at least 1 and divide n_steps")
        # n_steps >= 0 and 0 <= theta <= 1 are checked by reduced_theta_scheme.

        # ---- 2. Heat: one run of the theta-scheme on the r numbers a -----------------------------------------
        # The parameters go on the blocks of the offline phase:  rho_c M_r a' + k K_r a = q f_r - k lift_r
        coordinates = reduced_theta_scheme(rho_c * self._mass_r, k * self._stiffness_r,
                                           q * self._load_r - k * self._lift_r, self._a0, dt, n_steps, theta)
        coordinates = coordinates[::save_every]  # keep the steps 0, save_every, 2 save_every, ..., n_steps

        # ---- 3. Back to the plate: every field is "lift part + modes times a" --------------------------------
        T = reconstruct(self._basis, coordinates, self._lift)
        u = alpha * reconstruct(self._u_modes, coordinates, self._u_lift)  # proportional to alpha
        sigma = E * alpha * reconstruct(self._s_modes, coordinates, self._s_lift)  # proportional to E alpha
        sigma = sigma.reshape(len(T), len(self._triangles), 3)  # (state, triangle, xx yy xy)
        # The von Mises stress needs the temperature rise of every triangle: the mean over its 3 nodes minus T_ref.
        rise = element_mean(self._triangles, T) - self._T_ref
        return {
            "time": np.arange(0, n_steps + 1, save_every) * float(dt),
            "T": T,
            "u": u,
            "sigma": sigma,
            "von_mises": von_mises_plane_strain(sigma, rise, E, self._nu, alpha),
        }

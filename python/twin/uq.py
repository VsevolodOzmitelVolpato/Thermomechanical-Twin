"""The three results of the uncertainty study, as a function of the five uncertain numbers."""

import numpy as np

PARAMETER_NAMES = ("k", "rho_c", "q", "E", "alpha")

QUANTITY_NAMES = ("max_T", "max_von_mises", "max_displacement")


def rom_quantities(rom, points, dt, n_steps, theta, save_every=1):
    """Run the reduced model for many sets of numbers and return three results per run, read at the last time.

    rom                              a ThermoMechanicalROM
    points                           (m, 5) array, one row (k, rho_c, q, E, alpha) per run (see PARAMETER_NAMES)
    dt, n_steps, theta, save_every   time stepping, passed to rom.solve

    Returns an (m, 3) array whose columns are QUANTITY_NAMES: the largest nodal temperature, the largest von Mises
    stress and the largest displacement magnitude, all at the final time.
    Raises ValueError if points does not have shape (m, 5).
    """
    points = np.asarray(points, dtype= float)

    if points.ndim !=2 or points.shape[1] != 5:
        raise ValueError("points must have shape (m, 5)")

    m = points.shape[0]
    out = np.empty((m, 3), dtype=float)

    for i in range(m):
        k, rho_c, q, E, alpha = points[i]

        states = rom.solve(k, rho_c, q, E, alpha, dt, n_steps, theta, save_every)

        out[i, 0] = np.max(states["T"][-1])
        out[i, 1] = np.max(states["von_mises"][-1])

        u_last = states["u"][-1]
        ux = u_last[0::2]
        uy = u_last[1::2]
        out[i, 2] = np.max(np.hypot(ux, uy))

    return out

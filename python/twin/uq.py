import numpy as np

PARAMETER_NAMES = ("k", "rho_c", "q", "E", "alpha")

QUANTITY_NAMES = ("max_T", "max_von_mises", "max_displacement")


def rom_quantities(rom, points, dt, n_steps, theta, save_every=1):
    points = np.asarray(points, dtype= float)

    if points.ndim !=2 or points.shape[1] != 5:
        raise ValueError("invalid data for the points")

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
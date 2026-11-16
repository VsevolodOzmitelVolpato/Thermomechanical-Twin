"""Benchmark of the reduced-order model (ROM) against the C++ full model: the tables and the figures of the README.

Usage, from the project folder (the project must be built):
    python scripts/rom_benchmark.py            the full benchmark, a few minutes
    python scripts/rom_benchmark.py --quick    a coarse mesh and few runs, about 20 seconds: only to see that all works
It prints four markdown tables, saves them in docs/rom_tables.md and draws three figures into docs/figures/ (--out to
change it), each one twice: rom_<name>_light.png and rom_<name>_dark.png. The figures need matplotlib.

The problem. A steel plate of 200 mm x 100 mm (plane strain), clamped on its left edge. Its four edges are held at the
ambient temperature, except the left one, which is heated up to +40 K at mid-height. A Gaussian heat source warms the
right half. The temperatures are rises above the ambient one (T_ref = 0). The plate starts from the boundary values and
is followed for 300 s with backward Euler. Five numbers change from run to run: k, rho_c and q (the heat equation) and
E and alpha (the mechanics).

The ROM is built once (the offline phase):
  1. heat runs of the C++ solver, with (k, rho_c, q) taken from a Sobol sequence, give the snapshots;
  2. the POD (twin.pod) turns the snapshots into a few thermal modes;
  3. ThermoMechanicalROM (twin.rom) projects the problem on the modes and computes the mechanical modes.
Then it is tested on parameter sets it has never seen (all five parameters, from another sequence), against the full
C++ model (heat and elasticity), and the two are compared in error and in time.
"""
import argparse
import math
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "build" / "python", ROOT / "python"):
    if folder.is_dir():
        sys.path.insert(0, str(folder))

try:
    import thermomech
    from scipy.stats import qmc
    from twin.pod import pod_basis
    from twin.rom import ThermoMechanicalROM
except ImportError as error:
    sys.exit("Cannot import a module of the project (%s): build the project first (cmake --build build) and run this "
             "script from the project folder." % error)

try:
    import matplotlib

    matplotlib.use("Agg")  # only saves files, no window
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullLocator
    from matplotlib.tri import Triangulation
except ImportError:
    plt = None  # the tables are still written: main() tells what to install

# ---- The problem --------------------------------------------------------------------------------------------------

LENGTH, HEIGHT = 0.20, 0.10          # the plate [m]
EDGE_RISE = 40.0                     # temperature rise of the left edge at mid-height [K]
SPOT = (0.14, 0.05, 0.025)           # the heat source, a Gaussian: centre x, centre y and width [m]
NU, T_REF = 0.30, 0.0                # Poisson ratio; the temperatures are rises above the ambient one
DT, N_STEPS, THETA = 3.0, 100, 1.0   # 300 s of backward Euler
SAVE_EVERY = 10                      # the full model keeps 11 states (and solves the elasticity for each of them)
SNAPSHOT_EVERY = 5                   # a training run gives a snapshot every 5 steps: 21 per run
PARAMETERS = ("k", "rho_c", "q", "E", "alpha")                                    # the order of ROM.solve
NOMINAL = {"k": 45.0, "rho_c": 3.8e6, "q": 2.0e6, "E": 200e9, "alpha": 12e-6}     # a steel
SPREAD = {"k": 0.20, "rho_c": 0.10, "q": 0.30, "E": 0.10, "alpha": 0.10}          # each one is nominal * (1 +- spread)
FIELDS = ("T", "u", "sigma", "von_mises")                                         # what the models return, and compare
RANK_LIMIT = 1e-6                    # pod_basis refuses a mode whose singular value is below this fraction of the first


def sample(count, names, seed):
    """`count` points of a scrambled Sobol sequence, scaled to the box of the parameters `names`."""
    low = np.array([NOMINAL[name] * (1.0 - SPREAD[name]) for name in names])
    high = np.array([NOMINAL[name] * (1.0 + SPREAD[name]) for name in names])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a Sobol sequence whose length is not a power of 2 is a little unbalanced
        unit = qmc.Sobol(len(names), scramble=True, seed=seed).random(count)
    return qmc.scale(unit, low, high)


def build_problem(nx, ny):
    """The mesh, the temperatures held on the boundary, the heat load of the source for q = 1 and the clamped dofs."""
    mesh = thermomech.rectangle_mesh(LENGTH, HEIGHT, nx, ny)
    x, y = np.array(mesh.x), np.array(mesh.y)
    grid = np.arange(mesh.num_nodes).reshape(ny + 1, nx + 1)  # node (i, j) is j * (nx + 1) + i
    left = grid[:, 0]
    boundary = np.unique(np.concatenate([left, grid[:, -1], grid[0, :], grid[-1, :]]))
    values = np.where(x[boundary] == 0.0, EDGE_RISE * np.sin(np.pi * y[boundary] / HEIGHT), 0.0)
    values[np.abs(values) < 1e-9] = 0.0  # sin(pi) is not exactly 0
    lift = np.zeros(mesh.num_nodes)  # the imposed temperatures on the boundary, 0 inside: also the initial state
    lift[boundary] = values
    cx, cy, width = SPOT
    load = thermomech.assemble_load_2d(
        mesh, lambda px, py: math.exp(-((px - cx) ** 2 + (py - cy) ** 2) / (2.0 * width**2)))
    clamped = [dof for node in left for dof in (2 * int(node), 2 * int(node) + 1)]
    return {"mesh": mesh, "x": x, "y": y, "triangles": np.array(mesh.triangles), "fixed": boundary.tolist(),
            "values": values.tolist(), "lift": lift, "load": load, "clamped": clamped}


def heat_snapshots(problem, k, rho_c, q):
    """One training run, from the C++ heat solver: T - lift every SNAPSHOT_EVERY steps (the solver gives only the last
    state of a call, so it is called again from there)."""
    state = problem["lift"].copy()
    states = [state]
    for _ in range(N_STEPS // SNAPSHOT_EVERY):
        state = np.array(thermomech.solve_transient_2d(
            problem["mesh"], rho_c, k, q * problem["load"], problem["fixed"], problem["values"], state, DT,
            SNAPSHOT_EVERY, THETA))
        states.append(state)
    return np.array(states) - problem["lift"]


def full_model(problem, k, rho_c, q, E, alpha):
    """One run of the C++ full model: temperature, displacement, stress and von Mises stress."""
    dofs = problem["clamped"]
    return thermomech.solve_thermomechanical_2d(
        problem["mesh"], rho_c, k, q * problem["load"], problem["fixed"], problem["values"], problem["lift"], DT,
        N_STEPS, THETA, SAVE_EVERY, E, NU, alpha, T_REF, dofs, [0.0] * len(dofs))


def timed(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start


def relative_error(approximation, reference):
    """The L2 norm of the difference over all the nodes (or triangles) and all the saved states, relative to the norm
    of the reference."""
    return float(np.linalg.norm(approximation - reference) / np.linalg.norm(reference))


# ---- The benchmark ------------------------------------------------------------------------------------------------


def run_benchmark(nx, ny, n_train, n_held, modes, repeats):
    problem = build_problem(nx, ny)
    mesh = problem["mesh"]
    print("mesh %d x %d cells: %d nodes, %d triangles" % (nx, ny, mesh.num_nodes, mesh.num_triangles), flush=True)

    # offline 1: the training runs
    train = sample(n_train, PARAMETERS[:3], seed=7)
    snapshots, train_seconds = [], []
    for i, (k, rho_c, q) in enumerate(train, 1):
        states, seconds = timed(heat_snapshots, problem, k, rho_c, q)
        snapshots.append(states)
        train_seconds.append(seconds)
        print("  training run %d of %d: %.1f s" % (i, n_train, seconds), flush=True)
    matrix = np.vstack(snapshots).T  # one column per snapshot

    # offline 2: the POD
    mass = thermomech.assemble_mass_2d(mesh, 1.0)
    sigma = pod_basis(matrix, mass, 1)[1]  # all the singular values
    r_max = int(np.sum(sigma > 1.5 * RANK_LIMIT * sigma[0]))
    modes = [r for r in modes if r <= r_max]
    if not modes:
        sys.exit("The snapshots have no usable mode (only %d above the limit of pod_basis)." % r_max)
    (basis, _), pod_seconds = timed(pod_basis, matrix, mass, max(modes))
    print("POD: %d snapshots, %d usable modes, %.2f s" % (matrix.shape[1], r_max, pod_seconds), flush=True)

    # offline 3: a ROM for each number of modes (r + 1 elasticity solves of the C++ solver each)
    roms, build_seconds = {}, {}
    for r in modes:
        roms[r], build_seconds[r] = timed(
            ThermoMechanicalROM, mesh, basis[:, :r], problem["load"], problem["fixed"], problem["values"],
            problem["lift"], problem["clamped"], NU, T_REF)
        print("  ROM with %d modes: %.1f s" % (r, build_seconds[r]), flush=True)

    # the test: parameters that the basis has never seen, against the C++ full model
    held = sample(n_held, PARAMETERS, seed=99)
    runs = [dict(zip(PARAMETERS, row), dt=DT, n_steps=N_STEPS, theta=THETA, save_every=SAVE_EVERY) for row in held]
    fulls, full_seconds = [], []
    for i, run in enumerate(runs, 1):
        result, seconds = timed(full_model, problem, *(run[name] for name in PARAMETERS))
        fulls.append(result)
        full_seconds.append(seconds)
        print("  full model, run %d of %d: %.1f s" % (i, n_held, seconds), flush=True)

    errors = {r: np.zeros((n_held, len(FIELDS))) for r in modes}
    online = {}
    for r in modes:
        seconds = []
        for i, run in enumerate(runs):
            states = roms[r].solve(**run)
            errors[r][i] = [relative_error(states[name], fulls[i][name]) for name in FIELDS]
            seconds += [timed(roms[r].solve, **run)[1] for _ in range(repeats)]
        online[r] = statistics.median(seconds)

    main = max(modes)
    shown = int(np.argmax(errors[main][:, FIELDS.index("von_mises")]))  # the test run with the largest error
    return {
        "nx": nx, "ny": ny, "n_nodes": mesh.num_nodes, "n_triangles": mesh.num_triangles, "n_train": n_train,
        "n_held": n_held, "n_snapshots": matrix.shape[1], "modes": modes, "main": main, "sigma": sigma, "r_max": r_max,
        "errors": errors, "online": online, "build": build_seconds, "pod": pod_seconds, "train": sum(train_seconds),
        "full": statistics.median(full_seconds), "shown": shown, "shown_run": runs[shown],
        "x": problem["x"], "y": problem["y"], "triangles": problem["triangles"],
        "full_fields": fulls[shown], "rom_fields": roms[main].solve(**runs[shown]),
    }


# ---- The tables ---------------------------------------------------------------------------------------------------


def format_time(seconds):
    """'0.62 ms', '8.8 s', '3.4 min', '2.1 h', '10 days', '1.1 years': two significant digits."""
    units = ((365.25 * 86400.0, "year", "years"), (86400.0, "day", "days"), (3600.0, "h", "h"), (60.0, "min", "min"),
             (1.0, "s", "s"), (1e-3, "ms", "ms"), (1e-6, "us", "us"))
    for size, singular, plural in units:
        if seconds >= size or size == 1e-6:
            value = seconds / size
            text = "%.1f" % value if value < 9.95 else "%.0f" % value
            return "%s %s" % (text, singular if text == "1.0" else plural)


def format_ratio(value):
    """Two significant digits, never an exponent: 0.31, 3.1, 31, 280, 2900."""
    value = float("%.2g" % value)
    return "%d" % value if value >= 10 else "%.2g" % value


def markdown_table(title, headers, rows, align):
    """A '### title' heading and a markdown table whose columns are aligned: 'l' for left, 'r' for right."""
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def line(cells):
        return "| " + " | ".join(c.ljust(w) if a == "l" else c.rjust(w) for c, w, a in zip(cells, widths, align)) + " |"

    dashes = [("-" * (w + 1) + ":") if a == "r" else (":" + "-" * (w + 1)) for w, a in zip(widths, align)]
    separator = "|" + "|".join(dashes)
    return "\n".join(["### " + title, "", line(headers), separator + "|"] + [line(row) for row in rows]) + "\n"


def make_tables(results):
    r = results
    main = r["main"]
    low = {name: NOMINAL[name] * (1.0 - SPREAD[name]) for name in PARAMETERS}
    high = {name: NOMINAL[name] * (1.0 + SPREAD[name]) for name in PARAMETERS}
    cx, cy, width = SPOT
    problem = [
        ("plate", "%g mm x %g mm, steel, plane strain; left edge clamped, the other edges free"
         % (LENGTH * 1e3, HEIGHT * 1e3)),
        ("mesh", "%d x %d cells, %d nodes, %d triangles" % (r["nx"], r["ny"], r["n_nodes"], r["n_triangles"])),
        ("temperature", "rise above ambient; edges held at 0 K except the left one, %g K * sin(pi y / H); the plate "
         "starts from the boundary values" % EDGE_RISE),
        ("heat source", "Gaussian of width %g mm centred at (%g, %g) mm, peak value q"
         % (width * 1e3, cx * 1e3, cy * 1e3)),
        ("time", "backward Euler, dt = %g s, %d steps (%g s), %d states kept" % (
            DT, N_STEPS, DT * N_STEPS, N_STEPS // SAVE_EVERY + 1)),
        ("parameters", "k %g-%g W/(m K), rho_c %g-%g MJ/(m3 K), q %g-%g MW/m3, E %g-%g GPa, alpha %g-%g 1e-6/K" % (
            low["k"], high["k"], low["rho_c"] / 1e6, high["rho_c"] / 1e6, low["q"] / 1e6, high["q"] / 1e6,
            low["E"] / 1e9, high["E"] / 1e9, low["alpha"] / 1e-6, high["alpha"] / 1e-6)),
        ("training", "%d heat runs of the C++ solver, (k, rho_c, q) from a Sobol sequence: %d snapshots (%d per run)"
         % (r["n_train"], r["n_snapshots"], r["n_snapshots"] // r["n_train"])),
        ("test", "%d other runs of the full C++ model, the five parameters from another Sobol sequence" % r["n_held"]),
        ("machine", "%s, Python %s, NumPy %s" % (platform.platform(), platform.python_version(), np.__version__)),
    ]
    tables = [markdown_table("Benchmark problem", ["item", "description"], problem, "ll")]

    build = r["build"][main]
    offline = r["train"] + r["pod"] + build
    steps = [("training: %d heat runs (C++)" % r["n_train"], r["train"]),
             ("POD: basis of %d modes" % main, r["pod"]),
             ("ROM: projection and %d elasticity solves (C++)" % (main + 1), build)]
    def in_runs(seconds):
        value = seconds / r["full"]
        return "<0.01" if value < 0.005 else "%.2f" % value

    rows = [(name, format_time(seconds), in_runs(seconds)) for name, seconds in steps]
    rows.append(("offline, total", format_time(offline), in_runs(offline)))
    rows.append(("one run of the full model (heat and elasticity)", format_time(r["full"]), "1.00"))
    tables.append(markdown_table("Offline cost, paid once (ROM with %d modes)" % main,
                                 ["step", "time", "in full-model runs"], rows, "lrr"))

    rows = []
    for m in r["modes"]:
        worst = r["errors"][m].max(axis=0)
        rows.append([m, "%.2e" % (r["sigma"][m - 1] / r["sigma"][0])] + ["%.2e" % e for e in worst] +
                    ["%.2f ms" % (r["online"][m] * 1e3), format_ratio(r["full"] / r["online"][m])])
    tables.append(markdown_table(
        "Accuracy and speed against the number of modes (worst of %d test runs, relative L2 error)" % r["n_held"],
        ["modes", "sigma_r / sigma_1", "T", "u", "stress", "von Mises", "ROM run", "speed-up"], rows, "rrrrrrrr"))

    online = r["online"][main]
    saved = r["full"] - online
    rows = []
    for n in (1, 10, 100, 1000, 10000, 100000, 1000000):
        full_cost, rom_cost = n * r["full"], offline + n * online
        rows.append(["%d" % n, format_time(full_cost), format_time(rom_cost), format_ratio(full_cost / rom_cost)])
    text = markdown_table("Cost of N evaluations (ROM with %d modes, offline phase included)" % main,
                          ["N", "C++ full model", "ROM", "ratio"], rows, "rrrr")
    r["break_even"] = offline / saved
    text += "\nThe ROM is cheaper than the full model from N = %d evaluations on.\n" % math.ceil(r["break_even"])
    tables.append(text)
    return "\n".join(tables)


# ---- The figures: the look is the one of plot_convergence.py ------------------------------------------------------

WIDTH = 880                          # width of every figure in pixels (the PNG has twice as many)
PX = 0.75                            # one pixel of a 96 dpi screen, in points
BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")      # sequential steps 100 .. 700
ORANGE = ("#fbd7ca", "#f2b098", "#e68764", "#d95923", "#b44005", "#8b2c00", "#631b00")   # the same, in the orange hue
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "muted": "#898781", "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"), "blue": BLUE,
              "orange": ORANGE},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "grid": "#2c2c2a", "axis": "#383835",
             "muted": "#898781", "series": ("#3987e5", "#d95926", "#199e70", "#c98500"), "blue": BLUE[::-1],
             "orange": ORANGE[::-1]},   # on the dark surface the ramps run the other way: zero is the darkest step
}
MARKERS = ("o", "s", "^", "D")       # every series also has its own shape: colour is never the only cue
TIME_TICKS = ((1, "1 s"), (10, "10 s"), (60, "1 min"), (600, "10 min"), (3600, "1 h"), (86400, "1 day"),
              (864000, "10 days"), (8.64e6, "100 days"), (3.156e7, "1 year"))

if plt is not None:
    plt.rcParams["font.family"] = "DejaVu Sans"  # comes with matplotlib: the figures look the same on every machine
    plt.rcParams["mathtext.fontset"] = "dejavusans"


def new_figure(height, theme):
    return plt.figure(figsize=(WIDTH / 96.0, height / 96.0), dpi=96, facecolor=theme["surface"])


def add_axes(fig, height, left, top, width, plot_height):
    """Axes placed in pixels, measured from the top left corner of the figure."""
    return fig.add_axes([left / WIDTH, 1.0 - (top + plot_height) / height, width / WIDTH, plot_height / height])


def tick_label(value, _position=None):
    """1, 2 or 5 times a power of ten, written like the tables do: 5e-3, 1e-2, 1 (never 1e0)."""
    exponent = math.floor(math.log10(value) + 1e-9)
    digit = round(value / 10.0**exponent)
    if digit == 10:
        digit, exponent = 1, exponent + 1
    return str(digit) if exponent == 0 else "%de%d" % (digit, exponent)


def log_axis(axis, low, high):
    """Powers of ten, plus 2 and 5 when the axis spans less than two decades (so there are always a few labels)."""
    subs = (1.0, 2.0, 5.0) if math.log10(high / low) < 2.2 else (1.0,)
    axis.set_major_locator(LogLocator(base=10.0, subs=subs, numticks=40))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(tick_label))


def style_axes(ax, theme, x_label, y_label):
    """Quiet chrome: hairline grid, axes only left and bottom, text in ink."""
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=theme["grid"], linewidth=PX, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(PX)
    ax.set_facecolor(theme["surface"])
    ax.tick_params(axis="both", which="major", length=3, width=PX, color=theme["axis"], labelcolor=theme["ink2"],
                   labelsize=8.5, pad=3)
    ax.tick_params(axis="both", which="minor", length=0)
    ax.set_xlabel(x_label, color=theme["ink2"], fontsize=9, labelpad=5)
    ax.set_ylabel(y_label, color=theme["ink2"], fontsize=9, labelpad=5)


def header(ax, title, subtitle, theme):
    """A title in ink and, under it, one line in lighter ink."""
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 42), textcoords="offset points", ha="left",
                va="top", fontsize=10.5, color=theme["ink"], fontweight="bold")
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 30), textcoords="offset points", ha="left",
                va="top", fontsize=8.5, color=theme["ink2"])


def series_style(theme, index):
    """2 px lines, 8 px markers with a 2 px ring in the colour of the surface."""
    color = theme["series"][index]
    return dict(color=color, linewidth=2 * PX, solid_capstyle="round", solid_joinstyle="round", marker=MARKERS[index],
                markersize=8 * PX, markerfacecolor=color, markeredgecolor=theme["surface"], markeredgewidth=2 * PX)


def spread(positions, gap):
    """The positions (pixels) moved apart so that two neighbours are at least `gap` apart, in the same order."""
    order = np.argsort(positions)
    moved = np.array(positions, dtype=float)[order]
    for _ in range(100):
        done = True
        for i in range(1, len(moved)):
            missing = gap - (moved[i] - moved[i - 1])
            if missing > 1e-9:
                moved[i] += missing / 2.0
                moved[i - 1] -= missing / 2.0
                done = False
        if done:
            break
    result = np.empty_like(moved)
    result[order] = moved
    return result


def label_ends(ax, labels, theme, gap=13.0):
    """Direct labels at the right end of the lines, in ink, moved apart when the lines end close to each other.
    `labels` are (text, x, y) in data coordinates."""
    pixels = [ax.transData.transform((x, y))[1] for _, x, y in labels]
    placed = spread(pixels, gap)
    for (text, x, y), pixel, target in zip(labels, pixels, placed):
        ax.annotate(text, xy=(x, y), xycoords="data", xytext=(8 * PX, (target - pixel) * PX),
                    textcoords="offset points", ha="left", va="center", fontsize=8.5, color=theme["ink2"],
                    annotation_clip=False)


def save(fig, out_dir, name, theme_name, theme):
    path = out_dir / ("rom_%s_%s.png" % (name, theme_name))
    fig.savefig(path, dpi=192, facecolor=theme["surface"])
    plt.close(fig)
    return path


def draw_accuracy(results, theme_name, out_dir):
    """Left: the singular values of the snapshots. Right: the error of the ROM against the number of modes."""
    theme = THEMES[theme_name]
    height = 60 + 250 + 48
    fig = new_figure(height, theme)
    left, gap, right = 68, 72, 84
    width = (WIDTH - left - gap - right) / 2.0

    ax = add_axes(fig, height, left, 60, width, 250)
    sigma = results["sigma"] / results["sigma"][0]
    count = min(len(sigma), results["r_max"] + 4)
    xs, ys = list(range(1, count + 1)), list(sigma[:count])
    ax.axhline(RANK_LIMIT, color=theme["muted"], linewidth=PX, linestyle=(0, (4, 3)), zorder=2)
    ax.plot(xs, ys, zorder=3, **series_style(theme, 0))
    ax.set_yscale("log")
    low = min(min(ys), RANK_LIMIT) / 3.0
    ax.set_xlim(0.5, count + 0.5)
    ax.set_ylim(low, 2.0)
    log_axis(ax.yaxis, low, 2.0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    style_axes(ax, theme, "mode i", r"$\sigma_i\,/\,\sigma_1$")
    ax.annotate("pod_basis stops here (1e-6)", xy=(0.5, RANK_LIMIT), xytext=(4 * PX, 4 * PX),
                textcoords="offset points", ha="left", va="bottom", fontsize=8, color=theme["ink2"])
    header(ax, "Singular values", "%d snapshots: %d runs of %d" % (
        results["n_snapshots"], results["n_train"], results["n_snapshots"] // results["n_train"]), theme)

    ax = add_axes(fig, height, left + width + gap, 60, width, 250)
    modes = results["modes"]
    names = ("T", "u", "stress", "von Mises")
    curves = [[results["errors"][m][:, j].max() for m in modes] for j in range(len(FIELDS))]
    handles = []
    for j, ys in enumerate(curves):
        style = series_style(theme, j)
        ax.plot(modes, ys, zorder=3, **style)
        handles.append(Line2D([0], [0], **style))
    everything = [y for ys in curves for y in ys]
    low, high = min(everything) / 2.0, max(everything) * 2.0
    ax.set_yscale("log")
    ax.set_xlim(modes[0] - 0.5, modes[-1] + 0.5)
    ax.set_ylim(low, high)
    log_axis(ax.yaxis, low, high)
    ax.xaxis.set_major_locator(FixedLocator(modes))
    style_axes(ax, theme, "number of modes r", "relative L2 error")
    label_ends(ax, [(names[j], modes[-1], curves[j][-1]) for j in range(len(FIELDS))], theme)
    ax.legend(handles=handles, labels=list(names), loc="upper right", ncol=2, frameon=False, fontsize=8.5,
              labelcolor=theme["ink2"], handlelength=2.0, columnspacing=1.2, borderaxespad=0.2)
    header(ax, "Error against the full model", "worst of %d test runs" % results["n_held"], theme)
    return save(fig, out_dir, "accuracy", theme_name, theme)


def draw_cost(results, theme_name, out_dir):
    """The time of N evaluations of the full model and of the ROM (its offline phase included)."""
    theme = THEMES[theme_name]
    height = 60 + 290 + 48
    fig = new_figure(height, theme)
    ax = add_axes(fig, height, 96, 60, WIDTH - 96 - 110, 290)
    offline = results["train"] + results["pod"] + results["build"][results["main"]]
    full, online = results["full"], results["online"][results["main"]]
    counts = np.logspace(0.0, 6.0, 200)
    lines = (("C++ full model", counts * full), ("ROM", offline + counts * online))
    for i, (name, ys) in enumerate(lines):
        style = series_style(theme, i)
        ax.plot(counts, ys, zorder=3, **{**style, "marker": None})
    low, high = min(offline, full) / 2.0, lines[0][1][-1] * 2.0
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0, 1e6)
    ax.set_ylim(low, high)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%d" % v if v < 1000 else "1e%d" % round(math.log10(v))))
    ticks = [(value, text) for value, text in TIME_TICKS if low <= value <= high]
    ax.yaxis.set_major_locator(FixedLocator([value for value, _ in ticks]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: dict(ticks).get(v, "")))
    ax.yaxis.set_minor_locator(NullLocator())
    style_axes(ax, theme, "number of evaluations N", "time")
    label_ends(ax, [(name, 1e6, ys[-1]) for name, ys in lines], theme)

    # the break-even point and the cost of 1e5 evaluations (the size of a Sobol analysis)
    even = results["break_even"]
    ax.plot([even], [even * full], linestyle="none", marker="o", markersize=8 * PX, markerfacecolor=theme["ink"],
            markeredgecolor=theme["surface"], markeredgewidth=2 * PX, zorder=4)
    ax.annotate("break-even: %d runs" % math.ceil(even), xy=(even, even * full), xytext=(8 * PX, -8 * PX),
                textcoords="offset points", ha="left", va="top", fontsize=8.5, color=theme["ink"])
    ax.axvline(1e5, color=theme["muted"], linewidth=PX, linestyle=(0, (4, 3)), zorder=2)
    for i, (name, ys) in enumerate(lines):
        value = ys[np.argmin(np.abs(counts - 1e5))]
        ax.plot([1e5], [value], linestyle="none", marker="o", markersize=8 * PX, markerfacecolor=theme["series"][i],
                markeredgecolor=theme["surface"], markeredgewidth=2 * PX, zorder=4)
        above = i == 0  # the full model is the upper line: its label goes up and to the left, the other one down
        ax.annotate(format_time(value), xy=(1e5, value), xytext=((-8 if above else 8) * PX, (6 if above else -6) * PX),
                    textcoords="offset points", ha="right" if above else "left", va="bottom" if above else "top",
                    fontsize=8.5, color=theme["ink"])
    ax.annotate("1e5 runs", xy=(1e5, low), xycoords="data", xytext=(4 * PX, 4 * PX), textcoords="offset points",
                ha="left", va="bottom", fontsize=8, color=theme["ink2"])
    handles = [Line2D([0], [0], **{**series_style(theme, i), "marker": None}) for i in range(2)]
    names = ("C++ full model", "ROM, offline phase included")
    ax.legend(handles=handles, labels=names, loc="upper left", frameon=False, fontsize=8.5,
              labelcolor=theme["ink2"], handlelength=2.0, borderaxespad=0.2)
    header(ax, "Cost of N evaluations", "%d x %d mesh, %d modes: one ROM run is about %s times faster" % (
        results["nx"], results["ny"], results["main"], format_ratio(full / online)), theme)
    return save(fig, out_dir, "cost", theme_name, theme)


def colour_map(theme, name):
    return LinearSegmentedColormap.from_list(name, theme[name])


def draw_fields(results, theme_name, out_dir):
    """The temperature and the von Mises stress at the last state of the test run with the largest error: the full
    model, the ROM, and the difference."""
    theme = THEMES[theme_name]
    side, gap = 14, 18
    plot_width = (WIDTH - 2 * side - 2 * gap) / 3.0
    plot_height = plot_width * HEIGHT / LENGTH
    top, row = 78, plot_height + 62
    height = int(top + 2 * row)
    fig = new_figure(height, theme)
    mesh = Triangulation(results["x"], results["y"], results["triangles"])
    full, rom = results["full_fields"], results["rom_fields"]
    # (label, label of the difference, full model, ROM, scale of the values, scale of the difference, given at nodes?)
    quantities = (
        ("temperature rise [K]", r"difference [$\mu$K]", full["T"][-1], rom["T"][-1], 1.0, 1e6, True),
        ("von Mises stress [MPa]", "difference [Pa]", full["von_mises"][-1], rom["von_mises"][-1], 1e-6, 1.0, False))
    run = results["shown_run"]
    subtitle = ("test run %d of %d, the one with the largest error: k = %.3g W/(m K), rho_c = %.3g MJ/(m3 K), "
                "q = %.3g MW/m3, E = %.3g GPa, alpha = %.3g e-6/K" % (
                    results["shown"] + 1, results["n_held"], run["k"], run["rho_c"] / 1e6, run["q"] / 1e6,
                    run["E"] / 1e9, run["alpha"] / 1e-6))
    fig.text(side / WIDTH, 1.0 - 6.0 / height, "Temperature and von Mises stress at t = %g s" % (DT * N_STEPS),
             ha="left", va="top", fontsize=10.5, color=theme["ink"], fontweight="bold")
    fig.text(side / WIDTH, 1.0 - 24.0 / height, subtitle, ha="left", va="top", fontsize=8.5, color=theme["ink2"])
    titles = ("C++ full model", "ROM, %d modes" % results["main"], "Difference, ROM - full model")
    for r, (label, diff_label, full_values, rom_values, scale, diff_scale, nodal) in enumerate(quantities):
        difference = np.abs(rom_values - full_values) * diff_scale
        panels = ((full_values * scale, label, "blue"), (rom_values * scale, label, "blue"),
                  (difference, diff_label, "orange"))
        for c, (values, bar_label, ramp) in enumerate(panels):
            left = side + c * (plot_width + gap)
            plot_top = top + r * row
            ax = add_axes(fig, height, left, plot_top, plot_width, plot_height)
            maximum = max(full_values.max() * scale if c < 2 else difference.max(), 1e-300)
            cmap = colour_map(theme, ramp)
            if nodal:
                ax.tripcolor(mesh, values, cmap=cmap, vmin=0.0, vmax=maximum, shading="gouraud", rasterized=True)
            else:
                # one colour per triangle; an outline in the colour of the face closes the thin gaps between triangles
                ax.tripcolor(mesh, facecolors=values, cmap=cmap, vmin=0.0, vmax=maximum, edgecolors="face",
                             linewidth=0.7, rasterized=True)
            ax.set_xlim(0.0, LENGTH)
            ax.set_ylim(0.0, HEIGHT)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(theme["surface"])
            for spine in ax.spines.values():
                spine.set_color(theme["axis"])
                spine.set_linewidth(PX)
            wall = (left - 4) / WIDTH  # the clamped edge: a bar just outside the plate, never hidden by the data
            fig.add_artist(Line2D([wall, wall], [1.0 - (plot_top + plot_height) / height, 1.0 - plot_top / height],
                                  color=theme["ink"], linewidth=3 * PX, solid_capstyle="butt"))
            if r == 0:
                ax.annotate(titles[c], xy=(0, 1), xycoords="axes fraction", xytext=(0, 5 * PX),
                            textcoords="offset points", ha="left", va="bottom", fontsize=9.5, color=theme["ink"],
                            fontweight="bold")
            bar = add_axes(fig, height, left, plot_top + plot_height + 10, plot_width, 7)
            bar.imshow(np.linspace(0.0, 1.0, 256)[None, :], aspect="auto", cmap=cmap, extent=(0.0, maximum, 0.0, 1.0))
            bar.set_yticks([])
            bar.set_xticks([0.0, maximum])
            bar.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%.3g" % v))
            bar.tick_params(axis="x", length=3, width=PX, color=theme["axis"], labelcolor=theme["ink2"], labelsize=8,
                            pad=2)
            bar.get_xticklabels()[0].set_horizontalalignment("left")  # the labels stay under the bar
            bar.get_xticklabels()[1].set_horizontalalignment("right")
            for spine in bar.spines.values():
                spine.set_visible(False)
            bar.set_xlabel(bar_label, color=theme["ink2"], fontsize=8.5, labelpad=3)
    return save(fig, out_dir, "fields", theme_name, theme)


# ---- Main ---------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Accuracy and cost of the ROM against the C++ full model.")
    parser.add_argument("--quick", action="store_true", help="a coarse mesh and few runs (about 20 s): only to check")
    parser.add_argument("--nx", type=int, default=64, help="cells along the plate (default: 64)")
    parser.add_argument("--ny", type=int, default=32, help="cells across the plate (default: 32)")
    parser.add_argument("--train", type=int, default=16, help="training runs (default: 16)")
    parser.add_argument("--test", type=int, default=8, help="test runs against the full model (default: 8)")
    parser.add_argument("--modes", default="1,2,3,4,5,6,8,10,12",
                        help="numbers of modes to try (default: 1,2,3,4,5,6,8,10,12)")
    parser.add_argument("--repeats", type=int, default=5, help="timed repetitions of each ROM run (default: 5)")
    parser.add_argument("--out", default="docs/figures", help="folder of the figures (default: docs/figures)")
    parser.add_argument("--tables", default="docs/rom_tables.md",
                        help="file of the tables (default: docs/rom_tables.md)")
    parser.add_argument("--no-figures", action="store_true", help="only the tables")
    args = parser.parse_args()
    if args.quick:
        args.nx, args.ny, args.train, args.test, args.modes = 32, 16, 8, 3, "1,2,4,6"
    modes = sorted({int(text) for text in args.modes.split(",")})
    figures = not args.no_figures
    if figures and plt is None:
        sys.exit("The figures need matplotlib: pip install matplotlib (or run with --no-figures).")

    started = time.perf_counter()
    results = run_benchmark(args.nx, args.ny, args.train, args.test, modes, args.repeats)
    tables = make_tables(results)
    print("\n" + tables)
    path = Path(args.tables) if Path(args.tables).is_absolute() else ROOT / args.tables
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(tables)
    print("wrote", path)
    if figures:
        out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        for draw in (draw_accuracy, draw_cost, draw_fields):
            for theme_name in THEMES:
                print("wrote", draw(results, theme_name, out_dir))
    print("done in %.0f s" % (time.perf_counter() - started))


if __name__ == "__main__":
    main()
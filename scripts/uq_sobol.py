"""Uncertainty quantification of the plate: Sobol indices of the results, computed on the reduced-order model (ROM).

Usage, from the project folder (the project must be built, and scripts/rom_benchmark.py must be next to this file):
    python scripts/uq_sobol.py            the full study, a few minutes
    python scripts/uq_sobol.py --quick    a coarse mesh and a small sample, about a minute: only to see that all works
It prints the tables, saves them in docs/uq_tables.md and draws two figures into docs/figures/ (--out to change it),
each one twice: uq_<name>_light.png and uq_<name>_dark.png. The figures need matplotlib.

The question. The five numbers of the plate (k, rho_c, q, E, alpha) are known only up to a tolerance: each one is taken
uniform in the range nominal * (1 +- spread) of scripts/rom_benchmark.py, independent of the others. How uncertain are
the results, and which of the five numbers is responsible for most of the uncertainty? The results are three numbers
read at the end of the run (t = 300 s): the peak temperature rise at the heat source, the peak von Mises stress and the
peak displacement. (The peak temperature of the whole plate cannot be used: in every run it is the 40 K that the left
edge imposes, so it has no variance and no Sobol index. The script reads the temperature only at the nodes within one
width of the centre of the source, through SourceView.)

The method. twin.sobol.sobol_indices gives, for every result and every number, the first-order index S (the share of
the variance of the result that the number explains by itself) and the total index ST (the number alone and together
with the others), each with a 95 % confidence interval, from n (d + 2) = 7 n runs of the model. That is many runs for
the C++ model and a few seconds for the ROM, so the study is done on the ROM, built as in the benchmark (the same
training runs). Three checks tell how far the answer can be trusted:
  1. the three results of the ROM against those of the C++ full model, at test runs and at the corners of the box of
     (k, rho_c, q), the points farthest from the training runs;
  2. exact properties of the ROM that the indices must show: E and alpha do not change the temperature, E does not
     change the displacement, and the displacement (the stress) is proportional to alpha (to E alpha), so the indices
     of those numbers follow from the mean and the variance of the result alone;
  3. the confidence intervals and the indices against the number of runs n.
"""
import argparse
import itertools
import pickle
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import rom_benchmark as bench  # the same folder: the problem, the full model and the look of the figures

try:
    import scipy
    import thermomech
    from twin.pod import pod_basis
    from twin.rom import ThermoMechanicalROM
    from twin.sobol import sobol_indices
    from twin.uq import rom_quantities
except ImportError as error:
    sys.exit("Cannot import a module of the project (%s): build the project (cmake --build build), check that "
             "python/twin/sobol.py and python/twin/uq.py exist, and run this script from the project folder." % error)

try:
    import matplotlib.pyplot as plt  # rom_benchmark has already chosen the backend that only saves files
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, PathPatch
    from matplotlib.path import Path as OutlinePath
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
except ImportError:
    plt = None  # the tables are still written: main() tells what to install

# ---- What is studied ----------------------------------------------------------------------------------------------

D = len(bench.PARAMETERS)            # the number of uncertain numbers: 5
QUANTITIES = (                       # name in twin.uq, title, unit, factor from SI to the unit
    ("max_T", "peak temperature at the source", "K", 1.0),
    ("max_von_mises", "peak von Mises stress", "MPa", 1e-6),
    ("max_displacement", "peak displacement", "um", 1e6),
)
UNITS = {"k": ("W/(m K)", 1.0), "rho_c": ("MJ/(m3 K)", 1e-6), "q": ("MW/m3", 1e-6), "E": ("GPa", 1e-9),
         "alpha": ("1e-6/K", 1e6)}   # unit and factor from SI to the unit
LABELS = ("k", r"$\rho_c$", "q", "E", r"$\alpha$")   # the numbers in the figures
ZERO = 1e-9                          # an index below this is a zero
SAVED = bench.ROOT / "build" / "uq_results.pkl"      # what the run computed, to draw again without running again


def bounds():
    """The box of the five numbers: nominal * (1 - spread) and nominal * (1 + spread)."""
    low = np.array([bench.NOMINAL[name] * (1.0 - bench.SPREAD[name]) for name in bench.PARAMETERS])
    high = np.array([bench.NOMINAL[name] * (1.0 + bench.SPREAD[name]) for name in bench.PARAMETERS])
    return low, high


def evaluate(model, points):
    """The three results of every row of `points`, read at the last state only (save_every = n_steps)."""
    return rom_quantities(model, points, bench.DT, bench.N_STEPS, bench.THETA, save_every=bench.N_STEPS)


def source_nodes(problem):
    """The nodes within one width of the centre of the heat source."""
    cx, cy, width = bench.SPOT
    return np.flatnonzero(np.hypot(problem["x"] - cx, problem["y"] - cy) <= width)


class SourceView:
    """A model (the ROM or the full model) seen through the nodes of the heat source: solve() returns the fields of the
    model, except the temperature, which is given only at `nodes`. The 'max_T' of rom_quantities is then the peak
    temperature at the source. (Over the whole plate it would always be 40 K, the value that the left edge imposes.)"""

    def __init__(self, model, nodes):
        self.model = model
        self.nodes = nodes

    def solve(self, *arguments):
        states = self.model.solve(*arguments)
        return {**states, "T": np.asarray(states["T"])[:, self.nodes]}


class FullModel:
    """The C++ full model with the interface that rom_quantities expects from the ROM (a method solve); it also
    measures every run."""

    def __init__(self, problem):
        self.problem = problem
        self.seconds = []

    def solve(self, k, rho_c, q, E, alpha, dt, n_steps, theta, save_every=1):
        p = self.problem
        start = time.perf_counter()
        result = thermomech.solve_thermomechanical_2d(
            p["mesh"], rho_c, k, q * p["load"], p["fixed"], p["values"], p["lift"], dt, n_steps, theta, save_every, E,
            bench.NU, alpha, bench.T_REF, p["clamped"], [0.0] * len(p["clamped"]))
        self.seconds.append(time.perf_counter() - start)
        print("  full model, run %d: %.1f s" % (len(self.seconds), self.seconds[-1]), flush=True)
        return result


# ---- The study ----------------------------------------------------------------------------------------------------


def build_rom(nx, ny, n_train, n_modes):
    """The offline phase of the benchmark, with the same training runs: heat runs of the C++ solver, POD, ROM."""
    problem = bench.build_problem(nx, ny)
    mesh = problem["mesh"]
    print("mesh %d x %d cells: %d nodes, %d triangles" % (nx, ny, mesh.num_nodes, mesh.num_triangles), flush=True)
    snapshots, train_seconds = [], 0.0
    for i, (k, rho_c, q) in enumerate(bench.sample(n_train, bench.PARAMETERS[:3], seed=7), 1):
        states, seconds = bench.timed(bench.heat_snapshots, problem, k, rho_c, q)
        snapshots.append(states)
        train_seconds += seconds
        print("  training run %d of %d: %.1f s" % (i, n_train, seconds), flush=True)
    matrix = np.vstack(snapshots).T  # one column per snapshot
    mass = thermomech.assemble_mass_2d(mesh, 1.0)
    sigma = pod_basis(matrix, mass, 1)[1]  # all the singular values
    r_max = int(np.sum(sigma > 1.5 * bench.RANK_LIMIT * sigma[0]))
    n_modes = min(n_modes, r_max)
    (basis, _), pod_seconds = bench.timed(pod_basis, matrix, mass, n_modes)
    rom, build_seconds = bench.timed(
        ThermoMechanicalROM, mesh, basis, problem["load"], problem["fixed"], problem["values"], problem["lift"],
        problem["clamped"], bench.NU, bench.T_REF)
    print("POD and ROM with %d modes (%d usable): %.1f s" % (n_modes, r_max, pod_seconds + build_seconds), flush=True)
    return problem, rom, {"modes": n_modes, "nodes": mesh.num_nodes, "triangles": mesh.num_triangles,
                          "offline": train_seconds + pod_seconds + build_seconds}


def run_study(rom, sizes, seed):
    """The Sobol indices of the three results for every n in `sizes` (powers of 2).

    The largest n goes first. The smaller ones need no new run: their tables of points are the first rows of the tables
    of the largest one (the same seed gives nested Sobol sequences), and the results of every run are kept in memory.
    Returns the indices for every n, the results of all the distinct runs, the seconds spent in the runs of the ROM and
    the seconds of the whole study (the indices and their intervals included)."""
    low, high = bounds()
    memory, rom_seconds = {}, 0.0

    def model(points):
        nonlocal rom_seconds
        keys = [row.tobytes() for row in points]
        new = [i for i, key in enumerate(keys) if key not in memory]
        if new:
            values, seconds = bench.timed(evaluate, rom, points[new])
            rom_seconds += seconds
            for i, row in zip(new, values):
                memory[keys[i]] = row
            print("    %d runs of the ROM: %.1f s" % (len(new), seconds), flush=True)
        return np.array([memory[key] for key in keys])

    start = time.perf_counter()
    study = {}
    for n in sorted(sizes, reverse=True):
        print("  n = %d, %d evaluations" % (n, n * (D + 2)), flush=True)
        study[n] = sobol_indices(model, low, high, n, seed=seed)
    return study, np.array(list(memory.values())), rom_seconds, time.perf_counter() - start


def run_seeds(rom, n, seeds):
    """The Sobol indices again with other seeds, that is, with other points (a different scrambling of the Sobol
    sequence): the repetitions that tell how much the indices depend on the points."""
    low, high = bounds()
    results = []
    for seed in seeds:
        print("  seed %d, n = %d, %d evaluations" % (seed, n, n * (D + 2)), flush=True)
        results.append(sobol_indices(lambda points: evaluate(rom, points), low, high, n, seed=seed))
    return results


def check_quantities(problem, rom, n_test):
    """The three results of the ROM against those of the C++ full model (relative error), at `n_test` runs from another
    Sobol sequence and at the 8 corners of the box of (k, rho_c, q), with E and alpha at their nominal value (the ROM is
    exact in E and alpha). The corners are the points farthest from the training runs: the hardest test.
    `rom` is a SourceView of the ROM; the full model is seen through the same nodes."""
    low, high = bounds()
    corners = np.array([[*corner, bench.NOMINAL["E"], bench.NOMINAL["alpha"]]
                        for corner in itertools.product(*zip(low[:3], high[:3]))])
    sets = (("test", bench.sample(n_test, bench.PARAMETERS, seed=99)), ("corners", corners))
    full, errors = FullModel(problem), {}
    full_view = SourceView(full, rom.nodes)
    for name, points in sets:
        exact = evaluate(full_view, points)
        errors[name] = np.abs(evaluate(rom, points) - exact) / np.abs(exact)
    return {"errors": errors, "full_seconds": statistics.median(full.seconds)}


def check_structure(result):
    """The indices against what the ROM makes certain. Rows: check, expected, estimated, interval, verdict.

    Zeros: the temperature does not depend on E and alpha, the displacement does not depend on E, so their indices are
    zero (to the rounding of the computer). Products: a result X = c * Y with c uniform, nominal c0 * (1 +- s) and
    independent of Y, has the squared coefficient of variation CV2 = s^2 / 3, and then
        S = CV2 * mean^2 / variance,    ST = CV2 / (1 + CV2) * (1 + mean^2 / variance)
    (the displacement is proportional to alpha, the von Mises stress to E and to alpha: apply it to each of them)."""
    names = [q[0] for q in QUANTITIES]
    titles = {q[0]: q[1] for q in QUANTITIES}
    first, total = result.first_order, result.total_order
    rows = []
    for quantity, numbers in (("max_T", ("E", "alpha")), ("max_displacement", ("E",))):
        j = names.index(quantity)
        found = max(abs(index[bench.PARAMETERS.index(name), j]) for name in numbers for index in (first, total))
        rows.append(["%s: S and ST of %s" % (titles[quantity], " and ".join(numbers)), "0", "%.1e" % found, "",
                     "ok" if found < ZERO else "NOT ZERO"])
    for quantity, number in (("max_displacement", "alpha"), ("max_von_mises", "E"), ("max_von_mises", "alpha")):
        j, i = names.index(quantity), bench.PARAMETERS.index(number)
        cv2 = bench.SPREAD[number] ** 2 / 3.0
        ratio = result.mean[j] ** 2 / result.variance[j]
        expected = {"S": cv2 * ratio, "ST": cv2 / (1.0 + cv2) * (1.0 + ratio)}
        for kind, index, interval in (("S", first, result.first_order_ci), ("ST", total, result.total_order_ci)):
            low, high = interval[0, i, j], interval[1, i, j]
            inside = low <= expected[kind] <= high
            rows.append(["%s: %s of %s, from the mean and the variance" % (titles[quantity], kind, number),
                         format_index(expected[kind]), format_index(index[i, j]),
                         "[%s, %s]" % (format_index(low), format_index(high)),
                         "inside" if inside else "OUTSIDE"])
    return rows


def half_widths(result):
    """The largest half-width of the confidence intervals of the 10 indices (S and ST of 5 numbers), for each result."""
    widths = np.concatenate([result.first_order_ci[1] - result.first_order_ci[0],
                             result.total_order_ci[1] - result.total_order_ci[0]]) / 2.0
    return widths.max(axis=0)


# ---- The tables ---------------------------------------------------------------------------------------------------


def format_index(value):
    text = "%.3f" % value
    return "0.000" if text == "-0.000" else text


def make_tables(results):
    r = results
    n, sizes, study = r["n"], r["sizes"], r["study"]
    result = study[n]
    low, high = bounds()
    tables = []

    rows = [
        ("model", "the ROM of the benchmark: %d modes, trained on %d heat runs of the C++ solver; mesh %d x %d cells, "
         "%d nodes, %d triangles" % (r["modes"], r["n_train"], r["nx"], r["ny"], r["nodes"], r["triangles"])),
        ("uncertain numbers", "k, rho_c, q, E, alpha: independent, each uniform on its range (next table)"),
        ("results", "read at t = %g s: the peak temperature at the source (a rise above the ambient one, over the "
         "nodes within %g mm of its centre), the peak von Mises stress (all the triangles) and the peak displacement "
         "(all the nodes)" % (bench.DT * bench.N_STEPS, bench.SPOT[2] * 1e3)),
        ("indices", "S (first order) and ST (total) of every number and every result; estimators of Saltelli and of "
         "Jansen on a scrambled Sobol sequence, seed %d; 95 %% intervals from 200 bootstrap resamples" % r["seed"]),
        ("size", "n = %d: n (d + 2) = %d runs of the model; the convergence is followed from n = %d" % (
            n, n * (D + 2), sizes[0])),
        ("study", "%d distinct runs of the ROM (%.2f ms each, the three results at the last state only): %s; the whole "
         "study, with the indices and their intervals: %s" % (
             len(r["outputs"]), r["rom_seconds"] * 1e3, bench.format_time(r["rom_seconds"] * len(r["outputs"])),
             bench.format_time(r["study_seconds"]))),
        ("machine", "%s, Python %s, NumPy %s, SciPy %s" % (
            platform.platform(), platform.python_version(), np.__version__, scipy.__version__)),
    ]
    tables.append(bench.markdown_table("Uncertainty study", ["item", "description"], rows, "ll"))

    rows = []
    for i, name in enumerate(bench.PARAMETERS):
        unit, factor = UNITS[name]
        rows.append([name, unit, "%g" % (bench.NOMINAL[name] * factor),
                     "%g - %g" % (low[i] * factor, high[i] * factor), "+-%g %%" % (bench.SPREAD[name] * 100.0)])
    tables.append(bench.markdown_table("Uncertain numbers (independent, uniform on the range)",
                                       ["number", "unit", "nominal", "range", "spread"], rows, "llrrr"))

    rows = []
    for j, (name, title, unit, factor) in enumerate(QUANTITIES):
        mean, std = float(result.mean[j]) * factor, float(np.sqrt(result.variance[j])) * factor
        low95, high95 = np.percentile(r["outputs"][:, j] * factor, [2.5, 97.5])
        rows.append([title, unit, "%.4g" % mean, "%.3g" % std, "%.1f %%" % (100.0 * std / mean), "%.4g" % low95,
                     "%.4g" % high95])
    tables.append(bench.markdown_table(
        "Uncertainty of the results (n = %d: mean and standard deviation; the 95 %% range of the %d runs of the study)"
        % (n, len(r["outputs"])), ["result", "unit", "mean", "std", "std / mean", "2.5 %", "97.5 %"], rows, "lrrrrrr"))

    for j, (name, title, unit, factor) in enumerate(QUANTITIES):
        rows = []
        for i, number in enumerate(bench.PARAMETERS):
            rows.append([number, format_index(result.first_order[i, j]),
                         "[%s, %s]" % (format_index(result.first_order_ci[0, i, j]),
                                       format_index(result.first_order_ci[1, i, j])),
                         format_index(result.total_order[i, j]),
                         "[%s, %s]" % (format_index(result.total_order_ci[0, i, j]),
                                       format_index(result.total_order_ci[1, i, j]))])
        rows.append(["sum", format_index(result.first_order[:, j].sum()), "",
                     format_index(result.total_order[:, j].sum()), ""])
        tables.append(bench.markdown_table("Sobol indices of the %s (n = %d)" % (title, n),
                                           ["number", "S", "95 % interval", "ST", "95 % interval"], rows, "lrrrr"))

    rows = [[size, size * (D + 2)] + ["%.4f" % w for w in half_widths(study[size])] for size in sizes]
    tables.append(bench.markdown_table(
        "Convergence with n: the largest half-width of the 95 % intervals of the indices",
        ["n", "runs"] + [title for _, title, _, _ in QUANTITIES], rows, "rrrrr"))

    seeds = r["seeds"]
    runs = [study[seeds["n"]]] + seeds["results"]
    spread_first = np.ptp([run.first_order for run in runs], axis=0).max(axis=0)   # over the seeds, then the numbers
    spread_total = np.ptp([run.total_order for run in runs], axis=0).max(axis=0)
    widths = half_widths(study[seeds["n"]])
    rows = [[title, "%.4f" % widths[j], "%.4f" % spread_first[j], "%.4f" % spread_total[j]]
            for j, (name, title, unit, factor) in enumerate(QUANTITIES)]
    tables.append(bench.markdown_table(
        "Other seeds, n = %d: the largest difference of an index between the %d seeds (%s), against the half-width of "
        "the intervals of seed %d" % (seeds["n"], len(runs), ", ".join(str(s) for s in [r["seed"]] + seeds["seeds"]),
                                      r["seed"]),
        ["result", "half-width", "difference of S", "difference of ST"], rows, "lrrr"))

    errors = r["check"]["errors"]
    rows = [[title, "%.1e" % errors["test"][:, j].max(), "%.1e" % errors["corners"][:, j].max()]
            for j, (name, title, unit, factor) in enumerate(QUANTITIES)]
    tables.append(bench.markdown_table(
        "ROM against the C++ full model, relative error of the three results (worst of %d test runs, worst of the %d "
        "corners of the box)" % (len(errors["test"]), len(errors["corners"])),
        ["result", "test runs", "corners"], rows, "lrr"))

    tables.append(bench.markdown_table("Structure of the indices: what the ROM makes certain (n = %d)" % n,
                                       ["check", "expected", "estimated", "95 % interval", "verdict"],
                                       r["structure"], "lrrrl"))

    rom_seconds, full_seconds = r["rom_seconds"], r["check"]["full_seconds"]
    rows = []
    for size in sizes:
        runs = size * (D + 2)
        rom_cost, full_cost = r["offline"] + runs * rom_seconds, runs * full_seconds
        rows.append([size, runs, bench.format_time(rom_cost), bench.format_time(full_cost),
                     bench.format_ratio(full_cost / rom_cost)])
    text = bench.markdown_table("Cost of the study (ROM: offline phase included; full model: estimated from its runs)",
                                ["n", "runs", "ROM", "C++ full model", "ratio"], rows, "rrrrr")
    text += ("\nOne run of the ROM takes %.2f ms and one run of the full model %s, both with the last state only; the "
             "offline phase of the ROM takes %s.\n" % (rom_seconds * 1e3, bench.format_time(full_seconds),
                                                       bench.format_time(r["offline"])))
    tables.append(text)
    return "\n".join(tables)


# ---- The figures --------------------------------------------------------------------------------------------------

BAR = 18                             # width of a bar in pixels (at most 24)
RADIUS = 4                           # the rounded end of a bar, in pixels


def save_figure(fig, out_dir, name, theme_name, theme):
    path = out_dir / ("uq_%s_%s.png" % (name, theme_name))
    fig.savefig(path, dpi=192, facecolor=theme["surface"])
    plt.close(fig)
    return path


def bar_outline(left, width, height, radius_x, radius_y):
    """The outline of a bar: a flat foot on y = 0 and the two corners at the top rounded."""
    right, top = left + width, height
    vertices = [(left, 0.0), (left, top - radius_y), (left, top), (left + radius_x, top), (right - radius_x, top),
                (right, top), (right, top - radius_y), (right, 0.0), (left, 0.0)]
    codes = [OutlinePath.MOVETO, OutlinePath.LINETO, OutlinePath.CURVE3, OutlinePath.CURVE3, OutlinePath.LINETO,
             OutlinePath.CURVE3, OutlinePath.CURVE3, OutlinePath.LINETO, OutlinePath.CLOSEPOLY]
    return OutlinePath(vertices, codes)


def draw_indices(results, theme_name, out_dir):
    """One panel for each result: the indices S and ST of the five numbers, with their 95 % intervals."""
    theme = bench.THEMES[theme_name]
    px = bench.PX
    n, result = results["n"], results["study"][results["n"]]
    top, plot_height, bottom, y_max = 124, 230, 46, 1.1
    height = top + plot_height + bottom
    fig = bench.new_figure(height, theme)
    left, gap, right = 52, 46, 20
    width = (bench.WIDTH - left - right - 2 * gap) / 3.0
    pitch = width / D
    per_unit = plot_height / y_max   # pixels per unit of index
    colors = (theme["series"][0], theme["series"][1])

    fig.text(left / bench.WIDTH, 1.0 - 6.0 / height, "Sobol indices of the three results", ha="left", va="top",
             fontsize=10.5, color=theme["ink"], fontweight="bold")
    fig.text(left / bench.WIDTH, 1.0 - 24.0 / height,
             "ROM with %d modes, n = %d (%d runs); the lines are the 95%% confidence intervals" % (
                 results["modes"], n, n * (D + 2)), ha="left", va="top", fontsize=8.5, color=theme["ink2"])
    fig.legend(handles=[Patch(facecolor=color, edgecolor="none") for color in colors],
               labels=["S: the number alone", "ST: the number alone and with the others"], loc="upper left",
               bbox_to_anchor=((left - 4) / bench.WIDTH, 1.0 - 44.0 / height), ncol=2, frameon=False, fontsize=8.5,
               labelcolor=theme["ink2"], handlelength=1.0, handleheight=1.0, columnspacing=2.0, borderpad=0.0)

    for j, (name, title, unit, factor) in enumerate(QUANTITIES):
        ax = bench.add_axes(fig, height, left + j * (width + gap), top, width, plot_height)
        ax.set_xlim(0.0, width)    # the x axis counts pixels from the left edge of the panel
        ax.set_ylim(0.0, y_max)
        bench.style_axes(ax, theme, "", "share of the variance" if j == 0 else "")
        ax.grid(False, axis="x")
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else "1" if v == 1 else "%.1f" % v))
        ax.set_xticks([(i + 0.5) * pitch for i in range(D)])
        ax.set_xticklabels(LABELS, fontsize=9.5)
        ax.tick_params(axis="x", length=0, pad=5)
        for series, (index, interval) in enumerate(((result.first_order, result.first_order_ci),
                                                    (result.total_order, result.total_order_ci))):
            for i in range(D):
                x0 = (i + 0.5) * pitch + (-1 - BAR if series == 0 else 1)   # a gap of 2 px between the two bars
                value = max(float(index[i, j]), 0.0)
                radius = min(RADIUS, value * per_unit / 2.0)
                ax.add_patch(PathPatch(bar_outline(x0, BAR, value, radius, radius / per_unit), facecolor=colors[series],
                                       edgecolor="none", zorder=3))
                centre = x0 + BAR / 2.0
                lower, upper = max(float(interval[0, i, j]), 0.0), float(interval[1, i, j])
                if (upper - lower) * per_unit >= 3.0:   # a smaller interval would be a speck on the bar
                    ax.plot([centre, centre], [lower, upper], color=theme["ink"], linewidth=1.5 * px,
                            solid_capstyle="butt", zorder=4)
                    for end in (lower, upper):
                        ax.plot([centre - 2.5, centre + 2.5], [end, end], color=theme["ink"], linewidth=1.5 * px,
                                solid_capstyle="butt", zorder=4)
                if value >= 0.05:   # a number only above the bars that matter
                    ax.annotate(("%.2f" % value).lstrip("0"), xy=(centre, max(value, upper)), xytext=(0, 3 * px),
                                textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=theme["ink"])
        mean, std = float(result.mean[j]) * factor, float(np.sqrt(result.variance[j])) * factor
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 22), textcoords="offset points", ha="left",
                    va="bottom", fontsize=9.5, color=theme["ink"], fontweight="bold")
        ax.annotate("mean %.3g %s, std %.2g %s" % (mean, unit, std, unit), xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 8), textcoords="offset points", ha="left", va="bottom", fontsize=8.5,
                    color=theme["ink2"])
    return save_figure(fig, out_dir, "indices", theme_name, theme)


def draw_convergence(results, theme_name, out_dir):
    """Left: the confidence intervals against n. Right: the time of the study against n, ROM and full model."""
    theme = bench.THEMES[theme_name]
    px = bench.PX
    sizes, study = results["sizes"], results["study"]
    height = 60 + 250 + 48
    fig = bench.new_figure(height, theme)
    left, gap, right = 68, 72, 96
    width = (bench.WIDTH - left - gap - right) / 2.0

    ax = bench.add_axes(fig, height, left, 60, width, 250)
    widths = np.array([half_widths(study[size]) for size in sizes])   # (sizes, results)
    handles = []
    for j in range(len(QUANTITIES)):
        style = bench.series_style(theme, j)
        ax.plot(sizes, widths[:, j], zorder=3, **style)
        handles.append(Line2D([0], [0], **style))
    guide = 0.5 * np.exp(np.log(widths[0]).mean()) * np.sqrt(sizes[0] / np.array(sizes, dtype=float))
    ax.plot(sizes, guide, color=theme["muted"], linewidth=px, linestyle=(0, (4, 3)), zorder=2)   # below the data
    low, high = min(widths.min(), guide.min()) / 1.5, widths.max() * 2.0
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(sizes[0] / 1.2, sizes[-1] * 1.2)
    ax.set_ylim(low, high)
    ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%d" % round(v)))
    bench.log_axis(ax.yaxis, low, high)
    bench.style_axes(ax, theme, "n (the runs are 7 n)", "half-width of the 95% interval")
    ax.annotate(r"$1/\sqrt{n}$", xy=(sizes[-1], guide[-1]), xytext=(0, -5 * px), textcoords="offset points",
                ha="center", va="top", fontsize=8.5, color=theme["ink2"])
    ax.legend(handles=handles, labels=[title for _, title, _, _ in QUANTITIES], loc="upper right", frameon=False,
              fontsize=8.5, labelcolor=theme["ink2"], handlelength=2.0, borderaxespad=0.2)
    bench.header(ax, "Confidence intervals against n", "the widest of the 10 indices (S and ST) of each result", theme)

    ax = bench.add_axes(fig, height, left + width + gap, 60, width, 250)
    runs = np.array(sizes, dtype=float) * (D + 2)
    full, rom = results["check"]["full_seconds"], results["rom_seconds"]
    lines = (("C++ full model", runs * full), ("ROM", results["offline"] + runs * rom))
    for i, (name, ys) in enumerate(lines):
        ax.plot(sizes, ys, zorder=3, **{**bench.series_style(theme, i), "marker": None})
        ax.plot(sizes[-1:], ys[-1:], linestyle="none", marker="o", markersize=8 * px,
                markerfacecolor=theme["series"][i], markeredgecolor=theme["surface"], markeredgewidth=2 * px, zorder=4)
        ax.annotate(bench.format_time(ys[-1]), xy=(sizes[-1], ys[-1]), xytext=(-8 * px, 6 * px),
                    textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=theme["ink"])
    low, high = min(lines[1][1][0], lines[0][1][0]) / 2.0, lines[0][1][-1] * 2.0
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(sizes[0] / 1.2, sizes[-1] * 1.2)
    ax.set_ylim(low, high)
    ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%d" % round(v)))
    ticks = [(value, text) for value, text in bench.TIME_TICKS if low <= value <= high]
    ax.yaxis.set_major_locator(FixedLocator([value for value, _ in ticks]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: dict(ticks).get(v, "")))
    ax.yaxis.set_minor_locator(NullLocator())
    bench.style_axes(ax, theme, "n (the runs are 7 n)", "time of the study")
    bench.label_ends(ax, [(name, sizes[-1], ys[-1]) for name, ys in lines], theme)
    handles = [Line2D([0], [0], **{**bench.series_style(theme, i), "marker": None}) for i in range(2)]
    ax.legend(handles=handles, labels=("C++ full model", "ROM, offline phase included"), loc="upper left",
              frameon=False, fontsize=8.5, labelcolor=theme["ink2"], handlelength=2.0, borderaxespad=0.2)
    bench.header(ax, "Cost of the study", "last state only, for both models: one run takes %s against %.2f ms" % (
        bench.format_time(full), rom * 1e3), theme)
    return save_figure(fig, out_dir, "convergence", theme_name, theme)


# ---- Main ---------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Sobol indices of the plate on the ROM, checked against the full "
                                                 "model.")
    parser.add_argument("--quick", action="store_true",
                        help="a coarse mesh and a small sample (about a minute): only to check")
    parser.add_argument("--nx", type=int, default=64, help="cells along the plate (default: 64)")
    parser.add_argument("--ny", type=int, default=32, help="cells across the plate (default: 32)")
    parser.add_argument("--train", type=int, default=16, help="training runs of the ROM (default: 16)")
    parser.add_argument("--modes", type=int, default=12, help="modes of the ROM (default: 12)")
    parser.add_argument("--n", type=int, default=4096, help="size of the Sobol tables, a power of 2 (default: 4096)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the Sobol sequence (default: 0)")
    parser.add_argument("--test", type=int, default=8,
                        help="test runs against the full model, besides the 8 corners (default: 8)")
    parser.add_argument("--out", default="docs/figures", help="folder of the figures (default: docs/figures)")
    parser.add_argument("--tables", default="docs/uq_tables.md", help="file of the tables (default: docs/uq_tables.md)")
    parser.add_argument("--no-figures", action="store_true", help="only the tables")
    parser.add_argument("--redraw", action="store_true",
                        help="the tables and the figures again from the last run, which is not repeated")
    args = parser.parse_args()
    if args.quick:
        args.nx, args.ny, args.train, args.modes, args.n, args.test = 32, 16, 8, 6, 256, 3
    if args.n < 16 or args.n & (args.n - 1) != 0:
        parser.error("--n must be a power of 2, at least 16")
    if args.test < 1:
        parser.error("--test must be at least 1")
    figures = not args.no_figures
    if figures and plt is None:
        sys.exit("The figures need matplotlib: pip install matplotlib (or run with --no-figures).")

    started = time.perf_counter()
    if args.redraw:
        if not SAVED.is_file():
            sys.exit("No saved run (%s): run the script once, without --redraw." % SAVED)
        with open(SAVED, "rb") as handle:
            results = pickle.load(handle)
    else:
        problem, rom, offline = build_rom(args.nx, args.ny, args.train, args.modes)
        rom = SourceView(rom, source_nodes(problem))   # the ROM seen through the nodes of the source
        sizes = [args.n >> i for i in range(5, -1, -1) if args.n >> i >= 16]
        print("Sobol study, n from %d to %d" % (sizes[0], sizes[-1]), flush=True)
        study, outputs, rom_seconds, study_seconds = run_study(rom, sizes, args.seed)
        print("%d distinct runs of the ROM: %.1f s (%.2f ms each); the whole study: %.1f s" % (
            len(outputs), rom_seconds, rom_seconds / len(outputs) * 1e3, study_seconds), flush=True)
        limit = min(256, args.n // 4)   # the repetitions use one of the sizes of the study, 256 at most
        n_seeds = max([size for size in sizes if size <= limit], default=sizes[0])
        other_seeds = [args.seed + 1, args.seed + 2, args.seed + 3]
        print("the study again at n = %d with the seeds %s" % (n_seeds, other_seeds), flush=True)
        seeds = {"n": n_seeds, "seeds": other_seeds, "results": run_seeds(rom, n_seeds, other_seeds)}
        print("the ROM against the full model, at %d test runs and at the 8 corners of the box" % args.test, flush=True)
        check = check_quantities(problem, rom, args.test)
        results = {
            "nx": args.nx, "ny": args.ny, "n_train": args.train, "modes": offline["modes"], "nodes": offline["nodes"],
            "triangles": offline["triangles"], "offline": offline["offline"], "rom_seconds": rom_seconds / len(outputs),
            "n": args.n, "sizes": sizes, "seed": args.seed, "study": study, "outputs": outputs,
            "study_seconds": study_seconds, "seeds": seeds, "check": check, "structure": check_structure(study[args.n]),
        }
        SAVED.parent.mkdir(parents=True, exist_ok=True)
        with open(SAVED, "wb") as handle:
            pickle.dump(results, handle)

    tables = make_tables(results)
    print("\n" + tables)
    path = Path(args.tables) if Path(args.tables).is_absolute() else bench.ROOT / args.tables
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(tables)
    print("wrote", path)
    verdicts = [row[-1] for row in results["structure"]]
    print("structure checks: %d of %d ok" % (sum(v in ("ok", "inside") for v in verdicts), len(verdicts)))
    if figures:
        out_dir = Path(args.out) if Path(args.out).is_absolute() else bench.ROOT / args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        for draw in (draw_indices, draw_convergence):
            for theme_name in bench.THEMES:
                print("wrote", draw(results, theme_name, out_dir))
    print("done in %.0f s" % (time.perf_counter() - started))


if __name__ == "__main__":
    main()
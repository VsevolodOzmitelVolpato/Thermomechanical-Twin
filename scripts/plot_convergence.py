"""Draws the convergence figures of the README from the tables printed by convergence_study.

Usage, from the project folder:
    python scripts/plot_convergence.py                          runs build/convergence_study(.exe) and plots
    python scripts/plot_convergence.py --from-file tables.md    reads an output that was saved before
It writes the figures into docs/figures/ (--out to change it), each one twice: name_light.png and name_dark.png.
Needs matplotlib (pip install matplotlib).

Each panel shows one table (or one quantity of it, when a table mixes units) on log-log axes. The dashed line is the
slope the method is supposed to have (order 1 or 2): when the data run parallel to it, the method converges at that
order. The numbers are read from the tables, so the figures always agree with them.
"""
import argparse
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # only saves files, no window
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

# ---- Look ---------------------------------------------------------------------------------------------------------

plt.rcParams["font.family"] = "DejaVu Sans"      # comes with matplotlib: the figures look the same on every machine
plt.rcParams["mathtext.fontset"] = "dejavusans"

# Colours of the two versions (blue and orange of the categorical palette, checked for contrast on each surface).
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "muted": "#898781", "series": ("#2a78d6", "#eb6834")},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "grid": "#2c2c2a", "axis": "#383835",
             "muted": "#898781", "series": ("#3987e5", "#d95926")},
}
MARKERS = ("o", "s")      # the second series also has another shape: colour is never the only cue
PX = 0.75                 # one pixel of a 96 dpi screen, in points
WIDTH = 880               # width of every figure in pixels (the PNG has twice as many)
HEADER, PLOT_HEIGHT, FOOTER = 60, 250, 48   # pixels: title and subtitle, the plot itself, tick labels and x label
LEFT, RIGHT = 68, 14      # pixels around the panels

# ---- What to draw -------------------------------------------------------------------------------------------------

# The tables, found by the beginning of their title in the output of convergence_study.
TABLES = {
    "heat_space": "Spatial convergence (",
    "heat_be": "Temporal convergence (backward Euler",
    "heat_cn": "Temporal convergence (Crank-Nicolson",
    "chain_space": "Chain: spatial convergence",
    "chain_be": "Chain: temporal convergence (backward Euler",
    "chain_cn": "Chain: temporal convergence (Crank-Nicolson",
    "plate_steady": "2D plate: manufactured solution",
    "plate_space": "2D plate: decaying sine, spatial convergence",
    "plate_be": "2D plate: decaying sine, temporal convergence (backward Euler",
    "plate_cn": "2D plate: decaying sine, temporal convergence (Crank-Nicolson",
    "beam": "2D beam",
    "slab": "2D slab",
}

# A panel: "series" are the columns of its table to draw; "guides" are (series, order): a dashed line of slope
# "order" drawn under that series. Panels of a figure that share_y have the same y axis, so they can be compared.
FIGURES = [
    {"name": "convergence_heat_1d", "columns": 3, "share_y": True, "ylabel": "L2 error", "panels": [
        {"title": "Space", "table": "heat_space", "series": ["L2 error"], "guides": [(0, 2)]},
        {"title": "Time", "table": "heat_be", "series": ["L2 error"], "guides": [(0, 1)]},
        {"title": "Time", "table": "heat_cn", "series": ["L2 error"], "guides": [(0, 2)]},
    ]},
    {"name": "convergence_chain_1d", "columns": 3, "share_y": True, "ylabel": "error", "panels": [
        {"title": "Space", "table": "chain_space", "series": ["tip error", "stress error"], "guides": [(0, 2)]},
        {"title": "Time", "table": "chain_be", "series": ["tip error", "stress error"], "guides": [(0, 1)]},
        {"title": "Time", "table": "chain_cn", "series": ["tip error", "stress error"], "guides": [(0, 2)]},
    ]},
    {"name": "convergence_heat_2d", "columns": 2, "share_y": True, "ylabel": "L2 error", "panels": [
        {"title": "Steady state", "table": "plate_steady", "series": ["L2 error"], "guides": [(0, 2)],
         "subtitle": "manufactured solution sin(pi x) sin(pi y), k = 1"},
        {"title": "Decaying sine, space", "table": "plate_space", "series": ["L2 error"], "guides": [(0, 2)]},
        {"title": "Decaying sine, time", "table": "plate_be", "series": ["L2 error", "difference"],
         "guides": [(1, 1)]},
        {"title": "Decaying sine, time", "table": "plate_cn", "series": ["L2 error", "difference"],
         "guides": [(1, 2)]},
    ]},
    {"name": "convergence_thermomechanics_2d", "columns": 3, "share_y": False, "panels": [
        {"title": "Beam, displacement", "table": "beam", "series": ["displacement error"], "guides": [(0, 2)],
         "ylabel": "error [m]", "subtitle": "steel beam 4 m x 1 m, G = 100 K/m"},
        {"title": "Beam, stress", "table": "beam", "series": ["stress error"], "guides": [(0, 1)],
         "ylabel": "error [Pa]", "subtitle": "steel beam 4 m x 1 m, G = 100 K/m"},
        {"title": "Slab, von Mises stress", "table": "slab", "series": ["relative error"], "guides": [(0, 2)],
         "ylabel": "relative error", "subtitle": "Crank-Nicolson, dt = tau / 100,\nt = tau = 83.07 s"},
    ]},
]

# ---- Reading the tables -------------------------------------------------------------------------------------------


def split_row(line):
    """'| a | b |' -> ['a', 'b']"""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def to_float(cell):
    """The number in a cell, or None for '-' and for text like '4 x 1'."""
    try:
        return float(cell)
    except ValueError:
        return None


def parse_tables(text):
    """[(title, headers, rows)] for every '### title' followed by a markdown table."""
    tables = []
    title, block = None, []
    for line in text.splitlines() + ["### end"]:
        if line.startswith("###"):
            if title is not None and len(block) > 2:
                tables.append((title, split_row(block[0]), [split_row(row) for row in block[2:]]))
            title, block = line.lstrip("# ").strip(), []
        elif line.startswith("|"):
            block.append(line)
    return tables


def read_output(path):
    """The text of a saved output; PowerShell 5 writes '>' redirections as UTF-16."""
    data = Path(path).read_bytes()
    return data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")


def steps(headers, rows):
    """The step of each row (h or dt) and its axis label; the beam and the slab have k cells in the height: 1 / k."""
    if "h" in headers:
        return [to_float(row[headers.index("h")]) for row in rows], "mesh size h"
    if "dt" in headers:
        return [to_float(row[headers.index("dt")]) for row in rows], "time step dt"
    return [1.0 / to_float(row[0]) for row in rows], "cell size h / H"


# ---- Drawing ------------------------------------------------------------------------------------------------------


def tick_label(value, _position=None):
    """1, 2 or 5 times a power of ten, written like the tables do: 5e-3, 1e-2, 1 (never 1e0)."""
    exponent = math.floor(math.log10(value) + 1e-9)
    digit = round(value / 10.0 ** exponent)
    if digit == 10:
        digit, exponent = 1, exponent + 1
    return str(digit) if exponent == 0 else "%de%d" % (digit, exponent)


def log_axis(axis, low, high):
    """Powers of ten, plus 2 and 5 when the axis spans less than two decades (so there are always a few labels)."""
    subs = (1.0, 2.0, 5.0) if math.log10(high / low) < 2.2 else (1.0,)
    axis.set_major_locator(LogLocator(base=10.0, subs=subs, numticks=40))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(tick_label))


def guide_line(xs, ys, order):
    """Two points of the dashed line y = c x^order: it starts under the finest point (at 0.4 times its error) and goes
    over the finest half of the steps (in log scale); the coarse points are often not on the line yet."""
    x0 = min(xs)
    y0 = 0.4 * ys[xs.index(x0)]
    x1 = x0 * (max(xs) / x0) ** 0.5
    return (x0, x1), (y0, y0 * (x1 / x0) ** order)


def panel_data(panel, tables):
    """The points of every series of a panel and its guides, before any drawing."""
    headers, rows = tables[panel["table"]]
    x, x_label = steps(headers, rows)
    series = []
    for name in panel["series"]:
        column = headers.index(name)
        points = [(a, to_float(row[column])) for a, row in zip(x, rows) if to_float(row[column]) is not None]
        series.append((name, [p[0] for p in points], [p[1] for p in points]))
    guides = []
    for index, order in panel["guides"]:
        _, xs, ys = series[index]
        gx, gy = guide_line(xs, ys, order)
        guides.append((gx, gy, order))
    return series, guides, x_label


def y_range(data):
    """The smallest and largest y of everything that is drawn in the panels of `data`, with some room around."""
    values = []
    for series, guides, _ in data:
        for _, _, ys in series:
            values += ys
        for _, gy, _ in guides:
            values += gy
    return min(values) / 1.5, max(values) * 1.5


def draw_panel(ax, panel, data, subtitle, theme, plot_width, y_limits, show_y_label, y_label):
    series, guides, x_label = data
    surface = theme["surface"]

    all_x = [a for _, xs, _ in series for a in xs]
    x_limits = (min(all_x) / 1.2, max(all_x) * 1.2)

    # the dashed reference slopes, each with its label along the line, just under it
    x_decades = math.log10(x_limits[1] / x_limits[0])
    y_decades = math.log10(y_limits[1] / y_limits[0])
    for gx, gy, order in guides:
        ax.plot(gx, gy, color=theme["muted"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        angle = math.atan2(order * PLOT_HEIGHT / y_decades, plot_width / x_decades)   # of the line on the screen
        along = 0.5                                       # where on the line the label goes
        ax.annotate("order %d" % order, xy=(gx[0] * (gx[1] / gx[0]) ** along, gy[0] * (gy[1] / gy[0]) ** along),
                    xytext=(5 * math.sin(angle), -5 * math.cos(angle)), textcoords="offset points",
                    rotation=math.degrees(angle), rotation_mode="anchor", ha="center", va="top", fontsize=8,
                    color=theme["ink2"])                  # 5 points away from the line, on its lower side

    # the data: 2 px lines, 8 px markers with a 2 px ring in the colour of the surface
    handles = []
    for i, (name, xs, ys) in enumerate(series):
        color, marker = theme["series"][i], MARKERS[i]
        style = dict(color=color, linewidth=2 * PX, solid_capstyle="round", solid_joinstyle="round", marker=marker,
                     markersize=8 * PX, markerfacecolor=color, markeredgecolor=surface, markeredgewidth=2 * PX)
        ax.plot(xs, ys, zorder=3, **style)
        handles.append(Line2D([0], [0], **style))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    log_axis(ax.xaxis, *x_limits)
    log_axis(ax.yaxis, *y_limits)

    # quiet chrome: hairline grid, axes only left and bottom, text in ink
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=theme["grid"], linewidth=PX, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(PX)
    ax.set_facecolor(surface)
    ax.tick_params(axis="both", which="major", length=3, width=PX, color=theme["axis"], labelcolor=theme["ink2"],
                   labelsize=8.5, pad=3)
    ax.tick_params(axis="both", which="minor", length=0)
    ax.set_xlabel(panel.get("xlabel", x_label), color=theme["ink2"], fontsize=9, labelpad=5)
    if show_y_label:
        ax.set_ylabel(y_label, color=theme["ink2"], fontsize=9, labelpad=5)
    else:
        ax.tick_params(axis="y", labelleft=False)

    # a header: title in ink, under it the parameters of the run in lighter ink
    ax.annotate(panel["title"], xy=(0, 1), xycoords="axes fraction", xytext=(0, 42), textcoords="offset points",
                ha="left", va="top", fontsize=10.5, color=theme["ink"], fontweight="bold")
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 30), textcoords="offset points",
                ha="left", va="top", fontsize=8.5, color=theme["ink2"], linespacing=1.3)

    if len(series) > 1:   # a legend as soon as there are two series; a single one is named by the axis
        ax.legend(handles=handles, labels=[name for name, _, _ in series], loc="upper left", frameon=False,
                  fontsize=8.5, labelcolor=theme["ink2"], handlelength=2.0, borderaxespad=0.2)


def subtitle_of(panel, table_title, characters):
    """The subtitle of a panel: its own, or the text in the last parentheses of the title of its table
    ('Crank-Nicolson, dt = 1e-4, t = 0.1'), on at most 2 lines."""
    start, end = table_title.rfind("("), table_title.rfind(")")
    text = panel.get("subtitle", table_title[start + 1:end] if 0 <= start < end else "")
    lines = [line for part in text.split("\n") for line in textwrap.wrap(part, width=characters)]
    if len(lines) > 2:
        sys.exit("Subtitle too long for its panel: " + text)
    return "\n".join(lines)


def draw_figure(figure, tables, titles, theme_name, out_dir):
    theme = THEMES[theme_name]
    columns = figure["columns"]
    panels = figure["panels"]
    rows = (len(panels) + columns - 1) // columns
    gap = 22 if figure["share_y"] else 72                  # between panels: room for the y labels of the next one
    plot_width = (WIDTH - LEFT - RIGHT - (columns - 1) * gap) / columns
    height = rows * (HEADER + PLOT_HEIGHT + FOOTER)
    characters = int(plot_width / 6.6)                     # characters of the subtitle that fit in one line

    fig = plt.figure(figsize=(WIDTH / 96.0, height / 96.0), facecolor=theme["surface"])
    data = [panel_data(panel, tables) for panel in panels]
    limits = {}
    for r in range(rows):                                  # one y range for the whole row when it is shared
        members = range(r * columns, min((r + 1) * columns, len(panels)))
        for i in members:
            limits[i] = y_range([data[j] for j in members]) if figure["share_y"] else y_range([data[i]])

    for i, panel in enumerate(panels):
        r, c = divmod(i, columns)
        x0 = (LEFT + c * (plot_width + gap)) / WIDTH
        y0 = (height - (r + 1) * (HEADER + PLOT_HEIGHT + FOOTER) + FOOTER) / height
        ax = fig.add_axes([x0, y0, plot_width / WIDTH, PLOT_HEIGHT / height])
        show_y_label = (c == 0) or not figure["share_y"]
        draw_panel(ax, panel, data[i], subtitle_of(panel, titles[panel["table"]], characters), theme, plot_width,
                   limits[i], show_y_label, panel.get("ylabel", figure.get("ylabel", "")))

    path = out_dir / ("%s_%s.png" % (figure["name"], theme_name))
    fig.savefig(path, dpi=192, facecolor=theme["surface"])
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Convergence figures of the tables of convergence_study.")
    parser.add_argument("--from-file", help="text file with the output of convergence_study")
    parser.add_argument("--exe", default=None, help="path of convergence_study (default: build/convergence_study)")
    parser.add_argument("--out", default="docs/figures", help="output folder (default: docs/figures)")
    args = parser.parse_args()

    if args.from_file:
        text = read_output(args.from_file)
    else:
        default_name = "convergence_study.exe" if sys.platform.startswith("win") else "convergence_study"
        exe = Path(args.exe) if args.exe else Path("build") / default_name
        if not exe.exists():
            sys.exit("Cannot find " + str(exe) + ": build it first (cmake --build build --target convergence_study)")
        print("Running " + str(exe) + " (it takes a few seconds)...")
        text = subprocess.run([str(exe)], capture_output=True, check=True).stdout.decode("utf-8", "replace")

    # table of each key: the first whose title begins as TABLES says
    parsed = parse_tables(text)
    tables, titles, used = {}, {}, set()
    for key, beginning in TABLES.items():
        for title, headers, rows in parsed:
            if title.startswith(beginning):
                tables[key], titles[key] = (headers, rows), title
                used.add(title)
                break
        else:
            sys.exit("The output has no table titled '" + beginning + "...'. Titles found:\n  " +
                     "\n  ".join(title for title, _, _ in parsed))
    for title, _, _ in parsed:
        if title not in used:
            print("not drawn (no panel for it in FIGURES): " + title)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for figure in FIGURES:
        for theme_name in THEMES:
            print("wrote", draw_figure(figure, tables, titles, theme_name, out_dir))


if __name__ == "__main__":
    main()
"""Static figures (PNG) for a single flight and for a Monte Carlo run.

Uses the object-oriented matplotlib API only, so no GUI backend is ever needed.
Colours follow a small validated palette: one blue series colour, quiet grey chrome.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from .montecarlo import MCResult
from .sim import Flight

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"

PathLike = Union[str, Path]


def _style(ax, *, grid: str = "y", left_spine: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left_spine)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=8.5, pad=6)
    ax.grid(True, axis=grid, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=10, color=INK, pad=10)


def _xlabel(ax, text: str) -> None:
    ax.set_xlabel(text, fontsize=8.5, color=INK_2, labelpad=6)


def _dot(ax, x: float, y: float) -> None:
    ax.plot([x], [y], "o", ms=6.5, color=BLUE, mec=SURFACE, mew=1.5, zorder=5, clip_on=False)


def _note(ax, text: str, xy, offset, ha="left", va="bottom") -> None:
    ax.annotate(text, xy=xy, xytext=offset, textcoords="offset points", ha=ha, va=va,
                fontsize=8.5, color=INK_2, zorder=6)


def _header(fig: Figure, title: str, subtitle: str) -> None:
    fig.text(0.065, 0.945, title, fontsize=14, fontweight="bold", color=INK, va="center")
    fig.text(0.065, 0.905, subtitle, fontsize=9, color=INK_2, va="center")


def _conditions_text(c) -> str:
    text = f"wind {c.wind_speed:.1f} m/s from {c.wind_from:.0f} deg"
    if c.tilt > 0:
        text += f", rail tilt {c.tilt:.0f} deg"
    return text


def plot_flight(flight: Flight, path: PathLike, title: Optional[str] = None, dpi: int = 160) -> Path:
    """Four panels: altitude, speed, load factor, and a side view of the trajectory."""
    ev = flight.events
    t, z, speed = flight.t, flight.altitude, flight.speed
    s = flight.summary()

    fig = Figure(figsize=(11, 7.4), dpi=dpi, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, left=0.065, right=0.975, top=0.83, bottom=0.085,
                          hspace=0.42, wspace=0.17)
    sub = (f"{_conditions_text(flight.conditions)}, ejection delay {flight.delay:g} s  |  "
           f"apogee {s['apogee_m']:.0f} m, drift {s['drift_m']:.0f} m, "
           f"landing speed {s['landing_speed_ms']:.1f} m/s")
    _header(fig, title or f"{flight.rocket.name} on {flight.motor.name}", sub)

    # altitude
    ax = fig.add_subplot(gs[0, 0])
    _style(ax)
    ax.plot(t, z, color=BLUE, lw=2, solid_capstyle="round", solid_joinstyle="round")
    ax.set_ylim(0, max(z.max() * 1.3, 1.0))
    ax.set_xlim(0, t[-1])
    _title(ax, "Altitude (m)")
    if "burnout" in ev:
        b = ev["burnout"]
        _dot(ax, b.t, b.z)
        _note(ax, "burnout", (b.t, b.z), (9, -6), va="top")
    d, a = ev.get("deploy"), ev.get("apogee")
    if d:
        _dot(ax, d.t, d.z)
    if a:
        _dot(ax, a.t, a.z)
    if a and d and abs(a.t - d.t) < 2.0:       # the two events share one label
        gap = a.t - d.t
        when = f"{gap:.1f} s before" if gap > 0 else f"{-gap:.1f} s after"
        _note(ax, f"apogee {a.z:.0f} m\nchute out {when}", (a.t, a.z), (12, 6))
    else:
        if a:
            _note(ax, f"apogee {a.z:.0f} m", (a.t, a.z), (12, 6))
        if d:
            _note(ax, "chute out", (d.t, d.z), (12, 6))

    # speed
    ax = fig.add_subplot(gs[0, 1])
    _style(ax)
    ax.plot(t, speed, color=BLUE, lw=2, solid_capstyle="round", solid_joinstyle="round")
    ax.set_ylim(0, speed.max() * 1.2)
    ax.set_xlim(0, t[-1])
    _title(ax, "Speed (m/s)")
    i = int(np.argmax(speed))
    _dot(ax, t[i], speed[i])
    _note(ax, f"max {speed[i]:.0f} m/s", (t[i], speed[i]), (9, 4))
    _dot(ax, t[-1], speed[-1])
    _note(ax, f"landing {speed[-1]:.1f} m/s", (t[-1], speed[-1]), (-8, 10), ha="right")

    # load factor over the first seconds only (that is where all the action is)
    ax = fig.add_subplot(gs[1, 0])
    _style(ax)
    stop = max(ev["burnout"].t if "burnout" in ev else 0.0,
               ev["deploy"].t if "deploy" in ev else 0.0) + 3.0
    stop = min(stop, float(t[-1]))
    mask = t <= stop
    g = flight.g_axial
    gm = g[mask]
    ax.plot(t[mask], gm, color=BLUE, lw=2, solid_capstyle="round", solid_joinstyle="round")
    ax.axhline(0.0, color=AXIS, lw=1.0, zorder=2)
    ax.set_xlim(0, stop)
    ax.set_ylim(min(gm.min() * 1.5, -0.5), gm.max() * 1.2)
    _title(ax, f"Acceleration along the flight path (g), first {stop:.0f} s")
    _xlabel(ax, "Time (s)")
    j = int(np.argmax(np.where(mask, g, -np.inf)))
    _dot(ax, t[j], g[j])
    _note(ax, f"peak {g[j]:+.1f} g", (t[j], g[j]), (9, 2), va="center")
    k = int(np.argmin(np.where(mask, g, np.inf)))
    _dot(ax, t[k], g[k])
    _note(ax, f"min {g[k]:+.1f} g", (t[k], g[k]), (9, 0), va="center")

    # side view
    ax = fig.add_subplot(gs[1, 1])
    _style(ax, grid="both")
    d = flight.downrange
    ax.plot(d, z, color=BLUE, lw=2, solid_capstyle="round", solid_joinstyle="round")
    span = max(d.max() - min(d.min(), 0.0), z.max(), 1.0)
    ax.margins(0)
    ax.update_datalim([(min(d.min(), 0.0) - 0.12 * span, 0.0), (d.max() + 0.45 * span, z.max() * 1.15)])
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="datalim")
    _title(ax, "Side view (m)")
    _xlabel(ax, "Downrange from pad (m), positive towards the landing point")
    ax.plot([0], [0], "o", ms=6.5, color=INK, mec=SURFACE, mew=1.5, zorder=5, clip_on=False)
    _note(ax, "pad", (0, 0), (8, 6))
    _dot(ax, d[-1], 0.0)
    _note(ax, f"landed {flight.drift:.0f} m away", (d[-1], 0.0), (9, 8))

    out = Path(path)
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    return out


def plot_monte_carlo(result: MCResult, path: PathLike, title: Optional[str] = None,
                     dpi: int = 160) -> Path:
    """Top-down landing scatter with the 95 % radius, plus the apogee distribution."""
    c, u = result.conditions, result.uncertainty
    n = len(result.drift)
    r95 = result.radius(95.0)
    sm = result.summary()

    fig = Figure(figsize=(11, 5.6), dpi=dpi, facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], left=0.065, right=0.975,
                          top=0.79, bottom=0.12, wspace=0.17)
    sub = (f"{_conditions_text(c)} (sd {u.wind_speed_sd:.1f} m/s, {u.wind_dir_sd:.0f} deg), "
           f"thrust sd {u.thrust_sd * 100:.0f}%, delay sd {u.delay_sd:.1f} s"
           f"  |  95% of landings within {r95:.0f} m of the pad")
    _header(fig, title or f"Landing dispersion, {n} simulated flights on {result.motor_name}", sub)

    ax = fig.add_subplot(gs[0, 0])
    _style(ax, grid="both")
    ax.scatter(result.landing[:, 0], result.landing[:, 1], s=26, color=BLUE, alpha=0.55,
               linewidths=0, zorder=3)
    ax.add_patch(Circle((0, 0), r95, fill=False, ec=INK_2, lw=1.2, zorder=4))
    ax.plot([0], [0], "o", ms=6.5, color=INK, mec=SURFACE, mew=1.5, zorder=5)
    lim = max(r95 * 1.25, float(np.abs(result.landing).max()) * 1.08, 10.0)
    ax.margins(0)
    ax.update_datalim([(-lim, -lim), (lim, lim)])
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="datalim")
    _title(ax, "Landing points, top view (m, east / north)")
    ax.annotate(f"95% inside {r95:.0f} m", xy=(0, r95), xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, color=INK_2, zorder=6,
                bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none", alpha=0.9))
    ax.annotate("pad", xy=(0, 0), xytext=(-8, -4), textcoords="offset points", ha="right",
                va="top", fontsize=8.5, color=INK_2, zorder=6)

    ax = fig.add_subplot(gs[0, 1])
    _style(ax)
    counts, _, _ = ax.hist(result.apogee, bins=min(20, max(5, n // 8)), color=BLUE,
                           edgecolor=SURFACE, linewidth=1.5, zorder=3)
    top = counts.max() * 1.22
    ax.set_ylim(0, top)
    mean = sm["apogee_mean_m"]
    ax.axvline(mean, color=INK, lw=1.0, zorder=4)
    ax.annotate(f"mean {mean:.0f} m (sd {sm['apogee_sd_m']:.0f})", xy=(mean, top),
                xytext=(6, -4), textcoords="offset points", ha="left", va="top",
                fontsize=8.5, color=INK_2, zorder=6)
    _title(ax, "Apogee distribution, flights per bin")
    _xlabel(ax, "Apogee (m)")

    out = Path(path)
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    return out

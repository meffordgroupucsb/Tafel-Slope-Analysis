"""
io_plot.py -- shared CSV export and plotting helpers for the microkinetic
simulation scripts (numerical.py, qssa.py, gillespie.py, transient.py).

Each simulation script builds its results into a header + rows table, writes it
to CSV with save_csv, and optionally shows a plot with show_polarization (log|j|
vs overpotential, with error bars for the stochastic method) or show_transient
(current vs time). CSV writing uses only the standard library; matplotlib is
imported lazily inside the plotting helpers, so the scripts still run and still
write their CSV even if matplotlib is not installed.
"""
import csv


def save_csv(path, header, rows):
    """Write a results table to CSV.

    Args:
        path:   output filename.
        header: list of column names.
        rows:   iterable of row tuples/lists (numbers) matching `header`.
    """
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"# wrote {path}")


def _pyplot():
    """Import matplotlib.pyplot, or return None (with a note) if it is not installed."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("# matplotlib not installed; skipping plot (pip install matplotlib)")
        return None


def show_polarization(eta, j, se=None, title=None):
    """Show a polarization plot: |current| (log axis) versus overpotential.

    Args:
        eta:   overpotentials (V).
        j:     current densities (A/cm^2), same length as eta.
        se:    optional standard errors on j (A/cm^2); drawn as error bars (Gillespie).
        title: optional plot title.
    """
    plt = _pyplot()
    if plt is None:
        return
    import numpy as np
    eta = np.asarray(eta, float); j = np.abs(np.asarray(j, float))
    fig, ax = plt.subplots()
    if se is None:
        ax.semilogy(eta, j, "o-")
    else:
        ax.errorbar(eta, j, yerr=np.asarray(se, float), fmt="o-", capsize=3)
    ax.set_xlabel(r"overpotential $\eta$ / V")
    ax.set_ylabel(r"$|j|$ / A cm$^{-2}$")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    plt.show()


def show_transient(t, series, labels, title=None):
    """Show current transients: current versus time, one line per step potential.

    Args:
        t:      time points (s).
        series: list of current arrays (A/cm^2), one per curve, each len(t).
        labels: legend labels, one per series.
        title:  optional plot title.
    """
    plt = _pyplot()
    if plt is None:
        return
    fig, ax = plt.subplots()
    for y, lab in zip(series, labels):
        ax.plot(t, y, "-", label=lab)
    ax.set_xlabel("time / s")
    ax.set_ylabel(r"$j$ / A cm$^{-2}$")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    plt.show()

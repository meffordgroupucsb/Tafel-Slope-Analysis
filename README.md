# Microkinetic simulation of multistep electrochemical reactions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22183015.svg)](https://doi.org/10.5281/zenodo.22183015)

<img src="docs/toc.png" width="640" alt="Cardinal Tafel slopes from quasi-equilibrium can be misleading; robust analysis requires the steady-state approximation, coverage measurements, adsorption energies, and transient kinetic measurements.">

This repository contains scripts that can run microkinetic simulations of an
electrocatalytic mechanism. Each script runs the method specified in its name.
In each script, the mechanism is defined inline in a `MECHANISM` block you
edit, so the same code runs HER, HOR, ORR, OER, or any custom mechanism.

## Installation

```
pip install -r requirements.txt
```

installs everything (numpy, scipy, mpmath, matplotlib, jupyter). The scripts need only numpy and scipy (plus mpmath for `qe_error.py`) if you skip plotting.

| script | method | needs |
|---|---|---|
| `numerical.py` | stiff ODE integration of the coverage equations to steady state | numpy, scipy |
| `qssa.py` | steady-state approximation, SSA (solve `N·v = 0` algebraically) | numpy, scipy |
| `gillespie.py` | stochastic simulation (Gillespie Monte Carlo) on a finite site ensemble | numpy |
| `transient.py` | potentiostatic current transient `i(t)` after a potential step (companion to `numerical.py`) | numpy, scipy |
| `qe_error.py` | quasi-equilibrium levels (QE, QE′, QE″) for chosen slow steps, their error versus the SSA, and its attribution: the four-term breakdown for serial cycles and the integrated degree of rate control for any mechanism | numpy, scipy, mpmath |

`io_plot.py` is a shared helper (imported by the scripts) that writes the
CSV and draws the plots; it is not run directly. Run the scripts from the
repository directory so the import resolves, e.g. `python numerical.py`.
`numerical.py`, `qssa.py`, and `gillespie.py` each print the current density and
surface coverages across a scan of overpotentials. `numerical.py` and `qssa.py`
also report the local Tafel slope. `transient.py` instead prints the current
versus time after a step to a fixed overpotential, sampled over whatever window
and resolution you choose.

`qe_error.py` takes the same `MECHANISM` block plus `SLOW_STEPS`, the slow steps
of a quasi-equilibrium treatment. QE″ needs one slow step per independent route
and equilibrates the others; the script refuses sets that cannot work (the wrong
number of slow steps, or equilibrated steps that close a loop). For each
overpotential it prints the SSA current and degree of rate control of every
step, the current of each QE level and its error `log10(j_level/j_SSA)` in
decades, and where the error comes from. For a serial cycle with one slow step
the errors split exactly into rate-control, reverse-rate, downstream and
fixed-coverage terms. For any mechanism the QE″ error equals the degree of rate
control of the equilibrated steps integrated along the path that speeds them up
at fixed equilibrium constants; the total is exact, while its split among the
steps depends on that path.

## Output: CSV and plots

Each script has two output controls in its config block:

- `CSV_OUT` — filename for a CSV of the results (`None` to skip). Uses only the
  standard library.
- `SHOW_PLOT` — if `True`, opens a plot window: `|j|` versus overpotential for
  the three scan scripts (with error bars for `gillespie.py`), `i(t)` for
  `transient.py`, or the error of each QE level versus overpotential for
  `qe_error.py`. Plotting needs `matplotlib`.

## Tutorials

The `tutorials/` folder contains one Jupyter notebook per reaction in the main text,
each walking through all of the methods on that reaction's mechanism and then
rebuilding the paper's designed rate-limiting-step cases:

| notebook | reaction |
|---|---|
| `01_HER.ipynb` | acid HER (Volmer–Heyrovsky–Tafel) |
| `02_HOR.ipynb` | HOR (anodic branch of the same mechanism) |
| `03_ORR.ipynb` | acid associative 4-e⁻ ORR (5 steps) |
| `04_OER.ipynb` | alkaline associative 4-e⁻ OER (5 steps) |
| `06_qe_error.ipynb` | quasi-equilibrium levels and their error versus the SSA (serial OER, branched 2e⁻/4e⁻ ORR) |

`05_customize_scripts.ipynb` shows how to express any mechanism as the block the standalone scripts read, so you can run your own reaction. `06_qe_error.ipynb` runs `qe_error.py` on the OER and the branched ORR.

They are supported by two modules in the same folder: `tutorials/core.py` and `tutorials/mechanisms.py`.
Run the notebooks from the `tutorials/` folder. They need `numpy`, `scipy`, and `matplotlib` (and `mpmath` for `06_qe_error.ipynb`).

## Defining a mechanism

Edit the `MECHANISM` block near the top of each script:

- `SPECIES` — surface species, including the free site named in `SITE`.
- `STEPS` — each step gives its surface `reactants`/`products` with
  stoichiometric coefficients (a coefficient of `2` is a bimolecular term),
  `n_e` electrons transferred (`0` for a chemical step), symmetry factor `beta`,
  standard rate constants `k0`/`km0`, and optional bulk activities `a_fwd`/`a_rev`.
- `DIRECTION` — `-1` for a cathodic reaction (HER, ORR), `+1` for anodic (HOR, OER).
- `ETAS` — the overpotentials to scan (`STEP_ETAS`, `T_END`, and `N_T` set the
  potentials, time window, and sampling for `transient.py`).

The default mechanism is acid HER (Volmer–Heyrovsky–Tafel).

## Citing

See `CITATION.cff`, or cite the Zenodo archive:
[10.5281/zenodo.22183015](https://doi.org/10.5281/zenodo.22183015).

"""
transient.py -- potentiostatic current transient by stiff ODE integration.

Companion to numerical.py. Where numerical.py reports only the settled
steady-state current, this script keeps the whole time course: it applies a
potential step to a fixed overpotential eta from a clean surface and integrates
the coverage equations dtheta/dt = N . v in time, returning the transient current
i(t). This is the potentiostatic (chronoamperometric) response, and it can be
sampled over whatever time window and resolution the reader chooses by editing
T_END and N_T below.

Define the mechanism in the MECHANISM block (identical in form to numerical.py),
then run:  python transient.py

Only numpy and scipy are required.
"""
import numpy as np
from scipy.integrate import solve_ivp

from io_plot import save_csv, show_transient

# ---- physical constants -----------------------------------------------------
F = 96485                # C/mol
R = 8.3145               # J/mol/K
GAMMA = 1.0e-9           # mol/cm^2, active-site density
T = 298.15               # K
DIRECTION = -1           # -1 = cathodic forward (HER, ORR); +1 = anodic (HOR, OER)

# ============================ MECHANISM (edit me) ============================
# Surface species (include the free site). Steps are written in the forward
# direction set by DIRECTION. Each step lists its surface reactants/products
# with stoichiometric coefficients (a coefficient of 2 is a bimolecular term),
# the electrons transferred from the electrode n_e (0 for a chemical step), the
# symmetry factor beta, the standard forward/reverse rate constants k0/km0, and
# optional bulk-activity products a_fwd/a_rev.
SPECIES = ["*", "H"]    # define species here; list order defines the array index downstream
SITE = "*"              # the empty site

# STEPS stores the mechanism as a list of dicts, one per elementary step. Format:
#   dict(reactants={"reactant": coefficient}, products={"product": coefficient},
#        n_e=electrons transferred, beta=symmetry factor, k0=forward rate constant,
#        km0=reverse rate constant, a_fwd=reactant bulk activity, a_rev=product bulk activity,
#        label=name)
# a_fwd and a_rev are omitted below, so they default to 1.0.
STEPS = [
    dict(reactants={"*": 1}, products={"H": 1}, n_e=1, beta=0.5, k0=1e-3, km0=1.0,  label="Volmer"),
    dict(reactants={"H": 1}, products={"*": 1}, n_e=1, beta=0.5, k0=1.0,  km0=1e-3, label="Heyrovsky"),
    dict(reactants={"H": 2}, products={"*": 2}, n_e=0, beta=0.0, k0=1.0,  km0=1e-6, label="Tafel"),
]

# potentiostatic-step controls
STEP_ETAS = [-0.10, -0.20, -0.30]     # overpotentials to step to (V), one transient each
T_END = 1.0e2                         # length of the time window to follow (s)
N_T = 21                              # number of time points to report over [0, T_END]

# output controls
CSV_OUT   = "transient.csv"   # CSV filename to write, or None to skip
SHOW_PLOT = True              # show an i(t) plot window (needs matplotlib)
# ============================================================================

f = F / (R * T)
idx = {s: i for i, s in enumerate(SPECIES)}    # name -> index lookup (e.g. {"*": 0, "H": 1})
si = idx[SITE]                                 # index of the empty site

# net stoichiometric matrix: Nmat[species, step] = products - reactants
Nmat = np.zeros((len(SPECIES), len(STEPS)))
for i, st in enumerate(STEPS):
    for sp, c in st["reactants"].items():
        Nmat[idx[sp], i] -= c
    for sp, c in st["products"].items():
        Nmat[idx[sp], i] += c
NE = np.array([st["n_e"] for st in STEPS], dtype=float)   # electrons transferred per step


def applied_rates(eta):
    """Forward and reverse rate constants for every step at a given overpotential (Butler-Volmer).

    Args:
        eta: overpotential E - E_eq (V), referenced to the overall reaction.

    Returns:
        (kf, kr): forward and reverse rate-constant arrays, one entry per step.
    """
    kf = np.empty(len(STEPS)); kr = np.empty(len(STEPS))
    for i, st in enumerate(STEPS):
        bv = st["n_e"] * f * eta                              # BV exponent argument (0 for a chemical step)
        b = st["beta"]                                        # symmetry factor
        af = st.get("a_fwd", 1.0); ar = st.get("a_rev", 1.0)  # bulk activities (default 1.0)
        if DIRECTION < 0:                                     # cathodic forward
            kf[i] = st["k0"] * af * np.exp(-b * bv)
            kr[i] = st["km0"] * ar * np.exp((1.0 - b) * bv)
        else:                                                # anodic forward
            kf[i] = st["k0"] * af * np.exp((1.0 - b) * bv)
            kr[i] = st["km0"] * ar * np.exp(-b * bv)
    return kf, kr


def net_rates(theta, eta):
    """Mass-action net rate (forward - reverse) of every step for a coverage vector.

    Args:
        theta: full coverage vector (every species, summing to 1).
        eta:   overpotential (V).

    Returns:
        v: net rate of each step.
    """
    kf, kr = applied_rates(eta)
    v = np.empty(len(STEPS))
    for i, st in enumerate(STEPS):
        fwd = kf[i]
        for sp, c in st["reactants"].items():
            fwd *= theta[idx[sp]] ** c          # forward rate: kf * prod(theta_reactant^coeff)
        rev = kr[i]
        for sp, c in st["products"].items():
            rev *= theta[idx[sp]] ** c          # reverse rate: kr * prod(theta_product^coeff)
        v[i] = fwd - rev
    return v


def full_theta(free):
    """Rebuild the full coverage vector; the site species carries the balance to sum = 1.

    Args:
        free: coverages of every species except the site (the independent variables).

    Returns:
        full: the full coverage vector, including the site coverage.
    """
    full = np.empty(len(SPECIES)); j = 0
    for k in range(len(SPECIES)):
        if k == si:                             # skip the site; it is set by the balance
            continue
        full[k] = free[j]; j += 1
    full[si] = 1.0 - np.delete(full, si).sum()  # site coverage = 1 - sum(the rest)
    return full


def current(theta, eta):
    """Current density from the coverages: j = DIRECTION * F * GAMMA * sum(n_e * v).

    Args:
        theta: full coverage vector.
        eta:   overpotential (V).

    Returns:
        Current density (A/cm^2).
    """
    return DIRECTION * F * GAMMA * float(NE @ net_rates(theta, eta))


def dtheta_dt(free, eta):
    """Coverage time-derivatives of the independent species (the ODE right-hand side).

    Args:
        free: independent coverages (every species except the site).
        eta:  overpotential (V).

    Returns:
        d(theta)/dt for the independent species.
    """
    d = Nmat @ net_rates(full_theta(free), eta)
    return np.delete(d, si)                      # drop the site row; it is slaved by the balance


def transient(eta, t_end=T_END, n=N_T):
    """Potentiostatic current transient after a step to `eta` from a clean surface.

    Integrates the coverage ODEs at fixed eta and evaluates the current on a time
    grid, so the reader gets i(t) rather than only the settled value. Change
    `t_end` / `n` (or pass a custom grid via t_eval in the call) to look at other
    timescales.

    Args:
        eta:   overpotential stepped to at t = 0 (V).
        t_end: length of the time window to follow (s).
        n:     number of evenly spaced time points over [0, t_end].

    Returns:
        (t, i): time array (s) and current-density array (A/cm^2), both length n.
    """
    t_eval = np.linspace(0.0, t_end, n)                        # times at which to report the current
    y0 = np.zeros(len(SPECIES) - 1)                            # clean surface: site = 1, all others = 0
    sol = solve_ivp(lambda t, y: dtheta_dt(y, eta), (0.0, t_end), y0,
                    method="BDF", t_eval=t_eval, rtol=1e-10, atol=1e-13)   # stiff solver, sampled on t_eval
    i = np.array([current(np.clip(full_theta(sol.y[:, k]), 0.0, 1.0), eta)  # current at each sampled time
                  for k in range(sol.y.shape[1])])
    return sol.t, i


if __name__ == "__main__":
    print(f"# potentiostatic transients (ODE/BDF) -- {', '.join(st['label'] for st in STEPS)}")
    print(f"# columns: current density i(t)/A cm^-2 at each step potential; window 0..{T_END:g} s")
    curves = {eta: transient(eta)[1] for eta in STEP_ETAS}     # one transient per step potential
    t = transient(STEP_ETAS[0])[0]                             # shared time grid
    head = "".join(f"{f'eta={eta:+.2f}':>15}" for eta in STEP_ETAS)
    print(f"{'t/s':>9}{head}")
    for k in range(len(t)):                                    # one row per sampled time
        row = "".join(f"{curves[eta][k]:15.3e}" for eta in STEP_ETAS)
        print(f"{t[k]:9.3f}{row}")
    if CSV_OUT:
        header = ["t_s", *[f"i_eta{eta:+.2f}_A_cm2" for eta in STEP_ETAS]]
        table = [(t[k], *[curves[eta][k] for eta in STEP_ETAS]) for k in range(len(t))]
        save_csv(CSV_OUT, header, table)
    if SHOW_PLOT:
        show_transient(t, [curves[eta] for eta in STEP_ETAS],
                       [f"$\\eta$ = {eta:+.2f} V" for eta in STEP_ETAS],
                       title="potentiostatic transients")

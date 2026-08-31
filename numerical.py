"""
numerical.py -- steady-state microkinetics by stiff ODE integration.

Define an electrocatalytic mechanism in the MECHANISM block below, then run:
    python numerical.py

The coverage equations dtheta/dt = N . v are integrated with a stiff (BDF) solver
from a clean surface until they settle. At each overpotential the script prints
the current density, the surface coverages, and the local Tafel slope. This
method makes no rate-limiting-step or quasi-equilibrium assumption.

Only numpy and scipy are required.
"""
import numpy as np
from scipy.integrate import solve_ivp

from io_plot import save_csv, show_polarization

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

ETAS = np.linspace(-0.05, -0.40, 8)   # overpotentials to scan (V)

# output controls
CSV_OUT   = "numerical.csv"   # CSV filename to write, or None to skip
SHOW_PLOT = True              # show a log|j|-vs-eta plot window (needs matplotlib)
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
    """Net rate (forward - reverse) of every step for a coverage vector.

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


def steady_state(eta, t_end=1.0e3):
    """Integrate the coverage ODEs to steady state from a clean surface.

    Args:
        eta:   overpotential (V).
        t_end: integration end time; long enough for the coverages to settle.

    Returns:
        (j, theta): steady-state current density and full coverage vector.
    """
    y0 = np.zeros(len(SPECIES) - 1)              # clean surface: site = 1, all others = 0
    sol = solve_ivp(lambda t, y: dtheta_dt(y, eta), (0.0, t_end), y0,
                    method="BDF", rtol=1e-10, atol=1e-13)   # stiff solver
    theta = np.clip(full_theta(sol.y[:, -1]), 0.0, 1.0)     # final state, clipped to [0, 1]
    return current(theta, eta), theta


def tafel_slope(eta, deta=0.005):
    """Local Tafel slope (mV/dec) at eta from a central finite difference of log|j|.

    Args:
        eta:  overpotential (V).
        deta: half-width of the finite-difference window (V).

    Returns:
        Tafel slope (mV/dec, positive magnitude); inf if the current is flat.
    """
    jm = abs(steady_state(eta - deta)[0]); jp = abs(steady_state(eta + deta)[0])
    dlog = np.log10(jp) - np.log10(jm)
    return abs(2 * deta / dlog) * 1e3 if dlog else float("inf")


if __name__ == "__main__":
    head = "  ".join(f"th_{s}" for s in SPECIES)              # coverage column labels
    print(f"# numerical (ODE/BDF) -- {', '.join(st['label'] for st in STEPS)}")
    print(f"{'eta/V':>7}{'j/A cm^-2':>14}   {head}   {'b/mV dec^-1':>11}")
    rows = []
    for eta in ETAS:                                         # scan the overpotentials
        j, th = steady_state(eta)
        b = tafel_slope(eta)
        cov = "  ".join(f"{t:5.3f}" for t in th)
        print(f"{eta:7.3f}{j:14.3e}   {cov}   {b:11.1f}")
        rows.append((eta, j, *th, b))                        # collect for CSV / plot
    if CSV_OUT:
        save_csv(CSV_OUT, ["eta_V", "j_A_cm2", *[f"theta_{s}" for s in SPECIES], "tafel_mV_dec"], rows)
    if SHOW_PLOT:
        show_polarization([r[0] for r in rows], [r[1] for r in rows], title="numerical (ODE/BDF)")

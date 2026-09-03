"""
gillespie.py -- stochastic microkinetics by Gillespie Monte Carlo.

Define an electrocatalytic mechanism in the MECHANISM block below, then run:
    python gillespie.py

The surface is a finite ensemble of N sites with integer species counts. Every
elementary step (forward and reverse) is assigned a propensity from its rate
constant and the current site occupancies. At each iteration the waiting time is
drawn exponentially and one step fires. After a burn-in the current and coverages
are accumulated as time-weighted ensemble averages, with a standard error from
independent trajectories. In the large-N limit this reduces to the deterministic
balance, so it is an assumption-free cross-check that also resolves finite-size
fluctuations.

Only numpy is required.
"""
import numpy as np

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

# stochastic-simulation controls (larger values = less noise, slower to run)
N_SITES = 2000           # ensemble size
N_BURN = 200000          # burn-in reaction events (erase the initial condition)
N_PROD = 800000          # production reaction events (accumulate averages)
N_TRAJ = 4               # independent trajectories, for the standard error
SEED = 0

# output controls
CSV_OUT   = "gillespie.csv"   # CSV filename to write, or None to skip
SHOW_PLOT = True              # show a log|j|-vs-eta plot window (needs matplotlib)
# ============================================================================

f = F / (R * T)
idx = {s: i for i, s in enumerate(SPECIES)}    # name -> index lookup (e.g. {"*": 0, "H": 1})
si = idx[SITE]                                 # index of the empty site

# integer stoichiometry change per forward event: dcount = products - reactants
DN = np.zeros((len(STEPS), len(SPECIES)), dtype=int)   # initialize the change matrix

# fill DN with the stoichiometric change of each elementary step
for i, st in enumerate(STEPS):
    for sp, c in st["reactants"].items():
        DN[i, idx[sp]] -= c
    for sp, c in st["products"].items():
        DN[i, idx[sp]] += c

# reactant / product (species, coeff) lists and total orders, precomputed
RCT = [[(idx[sp], c) for sp, c in st["reactants"].items()] for st in STEPS]
PRD = [[(idx[sp], c) for sp, c in st["products"].items()] for st in STEPS]
ORD_R = [sum(st["reactants"].values()) for st in STEPS]
ORD_P = [sum(st["products"].values()) for st in STEPS]
NE = [st["n_e"] for st in STEPS]


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


def _combi(count, terms, order, N):
    """How many ways this step's reactants can be chosen from the sites present now
    (n for a single reactant, n(n-1) for a bimolecular one). Dividing by N rescales this
    count so that, with many sites, the event rate matches the ordinary rate law -- the
    rate constant times the coverages.

    Args:
        count: current integer count of each species (indexed as in SPECIES).
        terms: list of (species index, coefficient) pairs for one side of a step
               (RCT[i] for the forward reactants, PRD[i] for the reverse).
        order: total order of that side, the sum of its coefficients (ORD_R[i] or ORD_P[i]).
        N:     ensemble size (number of sites).

    Returns:
        The normalized combinatorial factor that multiplies the rate constant.
    """
    a = 1.0                # running product
    for j, nu in terms:    # j: species index, nu: coefficient
        n = count[j]
        for m in range(nu):
            a *= (n - m)   # falling factorial n(n-1)...(n-nu+1)
    return a / N ** (order - 1)


def propensities(count, kf, kr, N):
    """Build the vector of event rates (propensities) for the current state.

    Args:
        count:  current integer count of each species.
        kf, kr: forward and reverse rate-constant arrays from applied_rates.
        N:      ensemble size (number of sites).

    Returns:
        Array of length 2*len(STEPS): forward rates at even indices, reverse at odd.
    """
    a = np.empty(2 * len(STEPS))                                    # two channels per step: forward + reverse
    for i in range(len(STEPS)):
        # rate constant x number of ways the reactants (products, for the reverse) can react
        a[2 * i]     = kf[i] * _combi(count, RCT[i], ORD_R[i], N)   # forward channel (even index)
        a[2 * i + 1] = kr[i] * _combi(count, PRD[i], ORD_P[i], N)   # reverse channel (odd index; uses products)
    return np.maximum(a, 0.0)                                       # clamp any tiny negatives to 0


def run_trajectory(eta, N, rng):
    """Run one Gillespie trajectory; return its current and time-averaged coverages.

    Args:
        eta: overpotential (V).
        N:   ensemble size (number of sites).
        rng: a numpy random generator (numpy.random.Generator) supplying the random draws.

    Returns:
        (j, theta): current density (A/cm^2) and the time-averaged coverage of each species.
    """
    kf, kr = applied_rates(eta)                                 # rate constants (fixed for this eta)
    count = np.zeros(len(SPECIES), dtype=int); count[si] = N     # clean surface: all N sites start empty
    ne_net = 0.0; t_prod = 0.0                                  # net electrons transferred, elapsed production time
    theta_acc = np.zeros(len(SPECIES))                          # time-weighted coverage accumulator
    for step in range(N_BURN + N_PROD):                         # burn-in events, then production events
        a = propensities(count, kf, kr, N)                      # event rates for the current occupancies
        a0 = a.sum()                                            # total event rate
        if a0 <= 0:                                             # no reaction possible -> trajectory is stuck
            break
        dt = -np.log(rng.random()) / a0                         # waiting time to the next event (exponential)
        mu = np.searchsorted(np.cumsum(a), rng.random() * a0)   # choose a channel, prob proportional to its rate
        i, fwd = mu // 2, (mu % 2 == 0)                         # decode: step index i, forward if even index
        if step >= N_BURN:                                      # only record once past the burn-in
            t_prod += dt                                        # advance the production clock
            theta_acc += (count / N) * dt                       # coverage sample, weighted by the interval dt
            ne_net += NE[i] if fwd else -NE[i]                  # tally electrons (+ forward, - reverse)
        count += DN[i] if fwd else -DN[i]                       # apply the reaction to the site counts
    j = DIRECTION * F * GAMMA * ne_net / (N * t_prod)           # current density from per-site electron turnover
    return j, theta_acc / t_prod                                # coverages = time average over the window


def steady_state(eta, N=N_SITES):
    """Average N_TRAJ independent trajectories; return the mean current, its standard error,
    and the mean coverages.

    Args:
        eta: overpotential (V).
        N:   ensemble size (number of sites); defaults to N_SITES.

    Returns:
        (mean_j, se, mean_theta): mean current density, its seed-to-seed standard error,
        and the mean coverage of each species.
    """
    rng = np.random.default_rng(SEED + hash(round(eta, 6)) % 100000)  # reproducible RNG, one stream per eta
    js, ths = [], []                                           # per-trajectory current and coverages
    for _ in range(N_TRAJ):
        j, th = run_trajectory(eta, N, rng)
        js.append(j); ths.append(th)
    js = np.array(js); ths = np.array(ths)
    se = js.std(ddof=1) / np.sqrt(len(js)) if len(js) > 1 else 0.0    # standard error across trajectories
    return js.mean(), se, ths.mean(axis=0)


if __name__ == "__main__":
    head = "  ".join(f"th_{s}" for s in SPECIES)               # coverage column labels
    print(f"# Gillespie Monte Carlo -- {', '.join(st['label'] for st in STEPS)}"
          f"  (N={N_SITES}, {N_TRAJ} traj)")
    print(f"{'eta/V':>7}{'j/A cm^-2':>14}{'+/- SE':>11}   {head}")
    rows = []
    for eta in ETAS:                                          # scan the overpotentials
        j, se, th = steady_state(eta)
        cov = "  ".join(f"{t:5.3f}" for t in th)
        print(f"{eta:7.3f}{j:14.3e}{se:11.1e}   {cov}")
        rows.append((eta, j, se, *th))                        # collect for CSV / plot
    if CSV_OUT:
        save_csv(CSV_OUT, ["eta_V", "j_A_cm2", "se_A_cm2", *[f"theta_{s}" for s in SPECIES]], rows)
    if SHOW_PLOT:
        show_polarization([r[0] for r in rows], [r[1] for r in rows],
                          se=[r[2] for r in rows], title="Gillespie Monte Carlo")

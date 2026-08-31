"""
core.py -- parameterized microkinetics engine for the tutorial notebooks.

The standalone scripts in the repository root (numerical.py, qssa.py,
gillespie.py, transient.py) each bake one mechanism in at import time. The
tutorials instead define a mechanism per reaction and run every method on it, so
this module wraps the same physics behind a `Mechanism` object plus scan
functions:

    mech = Mechanism(species=["*", "H"], site="*", direction=-1, steps=[...])
    eta, j, theta, b = numerical_scan(mech, etas)     # BDF ODE to steady state
    eta, j, theta, b = qssa_scan(mech, etas)          # algebraic QSSA
    eta, j, se, theta = gillespie_scan(mech, etas)    # stochastic (Gillespie Monte Carlo)
    t, i = transient(mech, eta)                        # potentiostatic i(t)

The equations are identical to the scripts (Butler-Volmer rate constants,
mass-action coverages, the site balance); only the packaging differs so the same
code runs HER, HOR, ORR, OER, or any custom mechanism.
"""
import numpy as np

F = 96485.0   # C/mol
R = 8.3145    # J/mol/K


class Mechanism:
    """A microkinetic mechanism plus the precomputed structures the solvers need.

    Args:
        species:   list of surface species names, including the free site.
        site:      name of the free (empty) site, one of `species`.
        steps:     list of step dicts, each with `reactants`/`products`
                   (name -> stoichiometric coefficient), `n_e`, `beta`, `k0`,
                   `km0`, optional `a_fwd`/`a_rev`, and optional `label`.
        direction: -1 cathodic forward (HER, ORR), +1 anodic forward (HOR, OER).
        T:         temperature (K).
        gamma:     active-site density (mol/cm^2).
    """

    def __init__(self, species, site, steps, direction=-1, T=298.15, gamma=1.0e-9):
        self.species = list(species)
        self.site = site
        self.steps = steps
        self.direction = direction
        self.T = T
        self.gamma = gamma
        self.f = F / (R * T)
        self.idx = {s: i for i, s in enumerate(self.species)}
        self.si = self.idx[site]
        ns, nst = len(self.species), len(self.steps)
        # deterministic net stoichiometric matrix  Nmat[species, step]
        self.Nmat = np.zeros((ns, nst))
        # integer per-event change for the stochastic method  DN[step, species]
        self.DN = np.zeros((nst, ns), dtype=int)
        for i, st in enumerate(steps):
            for sp, c in st["reactants"].items():
                self.Nmat[self.idx[sp], i] -= c
                self.DN[i, self.idx[sp]] -= c
            for sp, c in st["products"].items():
                self.Nmat[self.idx[sp], i] += c
                self.DN[i, self.idx[sp]] += c
        self.NE = np.array([st["n_e"] for st in steps], dtype=float)
        self.RCT = [[(self.idx[sp], c) for sp, c in st["reactants"].items()] for st in steps]
        self.PRD = [[(self.idx[sp], c) for sp, c in st["products"].items()] for st in steps]
        self.ORD_R = [sum(st["reactants"].values()) for st in steps]
        self.ORD_P = [sum(st["products"].values()) for st in steps]
        self.labels = [st.get("label", f"step{i + 1}") for i, st in enumerate(steps)]


# ------------------------------ shared kinetics ------------------------------
def applied_rates(mech, eta):
    """Butler-Volmer forward/reverse rate constants for every step at eta."""
    kf = np.empty(len(mech.steps)); kr = np.empty(len(mech.steps))
    for i, st in enumerate(mech.steps):
        bv = st["n_e"] * mech.f * eta
        b = st["beta"]
        af = st.get("a_fwd", 1.0); ar = st.get("a_rev", 1.0)
        if mech.direction < 0:                       # cathodic forward
            kf[i] = st["k0"] * af * np.exp(-b * bv)
            kr[i] = st["km0"] * ar * np.exp((1.0 - b) * bv)
        else:                                        # anodic forward
            kf[i] = st["k0"] * af * np.exp((1.0 - b) * bv)
            kr[i] = st["km0"] * ar * np.exp(-b * bv)
    return kf, kr


def _net_rates(mech, theta, eta):
    """Mass-action net rate (forward - reverse) of every step for coverages theta."""
    kf, kr = applied_rates(mech, eta)
    v = np.empty(len(mech.steps))
    for i, st in enumerate(mech.steps):
        fwd = kf[i]
        for sp, c in st["reactants"].items():
            fwd *= theta[mech.idx[sp]] ** c
        rev = kr[i]
        for sp, c in st["products"].items():
            rev *= theta[mech.idx[sp]] ** c
        v[i] = fwd - rev
    return v


def _full_theta(mech, free):
    """Rebuild the full coverage vector; the site carries the balance to sum = 1."""
    full = np.empty(len(mech.species)); j = 0
    for k in range(len(mech.species)):
        if k == mech.si:
            continue
        full[k] = free[j]; j += 1
    full[mech.si] = 1.0 - np.delete(full, mech.si).sum()
    return full


def current(mech, theta, eta):
    """Current density (A/cm^2) from the coverages: DIRECTION * F * GAMMA * sum(n_e * v)."""
    return mech.direction * F * mech.gamma * float(mech.NE @ _net_rates(mech, theta, eta))


# ------------------------------ numerical (BDF) ------------------------------
def numerical_state(mech, eta, t_end=1.0e3):
    """Integrate the coverage ODEs to steady state from a clean surface. -> (j, theta)."""
    from scipy.integrate import solve_ivp

    def rhs(t, free):
        d = mech.Nmat @ _net_rates(mech, _full_theta(mech, free), eta)
        return np.delete(d, mech.si)

    y0 = np.zeros(len(mech.species) - 1)
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="BDF", rtol=1e-10, atol=1e-13)
    theta = np.clip(_full_theta(mech, sol.y[:, -1]), 0.0, 1.0)
    return current(mech, theta, eta), theta


# ------------------------------ QSSA (root find) -----------------------------
def qssa_state(mech, eta, seed=None):
    """Solve N.v = 0 algebraically (QSSA). -> (j, theta, free) where free reseeds the next eta."""
    from scipy.optimize import root

    def resid(free):
        d = mech.Nmat @ _net_rates(mech, _full_theta(mech, free), eta)
        return np.delete(d, mech.si)

    y0 = np.zeros(len(mech.species) - 1) if seed is None else seed
    sol = root(resid, y0, method="hybr", tol=1e-13)
    theta = np.clip(_full_theta(mech, sol.x), 0.0, 1.0)
    return current(mech, theta, eta), theta, np.delete(theta, mech.si)


def tafel_slope(mech, eta, method="qssa", deta=0.005, seed=None):
    """Local Tafel slope (mV/dec) by a central finite difference of log|j|."""
    if method == "qssa":
        solve = lambda e: qssa_state(mech, e, seed)[0]
    else:
        solve = lambda e: numerical_state(mech, e)[0]
    jm = abs(solve(eta - deta)); jp = abs(solve(eta + deta))
    dlog = np.log10(jp) - np.log10(jm)
    return abs(2 * deta / dlog) * 1e3 if dlog else float("inf")


def numerical_scan(mech, etas):
    """Run the BDF method across etas. -> (eta, j, theta, tafel) arrays."""
    js, ths, bs = [], [], []
    for eta in etas:
        j, th = numerical_state(mech, eta)
        js.append(j); ths.append(th); bs.append(tafel_slope(mech, eta, method="num"))
    return np.asarray(etas, float), np.array(js), np.array(ths), np.array(bs)


def qssa_scan(mech, etas):
    """Run the QSSA across etas with continuation seeding. -> (eta, j, theta, tafel) arrays."""
    js, ths, bs = [], [], []
    seed = None
    for eta in etas:
        j, th, seed = qssa_state(mech, eta, seed)
        js.append(j); ths.append(th); bs.append(tafel_slope(mech, eta, method="qssa", seed=seed))
    return np.asarray(etas, float), np.array(js), np.array(ths), np.array(bs)


# ------------------------------ Gillespie Monte Carlo ------------------------------
def _combi(count, terms, order, N):
    a = 1.0
    for j, nu in terms:
        n = count[j]
        for m in range(nu):
            a *= (n - m)
    return a / N ** (order - 1)


def _propensities(mech, count, kf, kr, N):
    a = np.empty(2 * len(mech.steps))
    for i in range(len(mech.steps)):
        a[2 * i]     = kf[i] * _combi(count, mech.RCT[i], mech.ORD_R[i], N)
        a[2 * i + 1] = kr[i] * _combi(count, mech.PRD[i], mech.ORD_P[i], N)
    return np.maximum(a, 0.0)


def _gillespie_traj(mech, eta, N, n_burn, n_prod, rng):
    kf, kr = applied_rates(mech, eta)
    count = np.zeros(len(mech.species), dtype=int); count[mech.si] = N
    ne_net = 0.0; t_prod = 0.0
    theta_acc = np.zeros(len(mech.species))
    for step in range(n_burn + n_prod):
        a = _propensities(mech, count, kf, kr, N)
        a0 = a.sum()
        if a0 <= 0:
            break
        dt = -np.log(rng.random()) / a0
        mu = np.searchsorted(np.cumsum(a), rng.random() * a0)
        i, fwd = mu // 2, (mu % 2 == 0)
        if step >= n_burn:
            t_prod += dt
            theta_acc += (count / N) * dt
            ne_net += mech.NE[i] if fwd else -mech.NE[i]
        count += mech.DN[i] if fwd else -mech.DN[i]
    j = mech.direction * F * mech.gamma * ne_net / (N * t_prod)
    return j, theta_acc / t_prod


def gillespie_scan(mech, etas, n_sites=2000, n_burn=200000, n_prod=800000, n_traj=4, seed=0):
    """Run the stochastic method across etas. -> (eta, j, se, theta) arrays.

    n_sites/n_burn/n_prod/n_traj trade run time for lower noise; the defaults
    match gillespie.py. Notebooks use lighter settings so a cell runs in seconds.
    """
    E, J, SE, TH = [], [], [], []
    for eta in etas:
        rng = np.random.default_rng(seed + hash(round(float(eta), 6)) % 100000)
        js, ths = [], []
        for _ in range(n_traj):
            jj, th = _gillespie_traj(mech, eta, n_sites, n_burn, n_prod, rng)
            js.append(jj); ths.append(th)
        js = np.array(js); ths = np.array(ths)
        se = js.std(ddof=1) / np.sqrt(len(js)) if len(js) > 1 else 0.0
        E.append(eta); J.append(js.mean()); SE.append(se); TH.append(ths.mean(axis=0))
    return np.asarray(E, float), np.array(J), np.array(SE), np.array(TH)


# ------------------------------ transient (BDF) ------------------------------
def transient(mech, eta, t_end=1.0e2, n=200):
    """Potentiostatic current transient i(t) after a step to eta. -> (t, i) arrays."""
    from scipy.integrate import solve_ivp

    def rhs(t, free):
        d = mech.Nmat @ _net_rates(mech, _full_theta(mech, free), eta)
        return np.delete(d, mech.si)

    t_eval = np.linspace(0.0, t_end, n)
    y0 = np.zeros(len(mech.species) - 1)
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="BDF", t_eval=t_eval, rtol=1e-10, atol=1e-13)
    i = np.array([current(mech, np.clip(_full_theta(mech, sol.y[:, k]), 0.0, 1.0), eta)
                  for k in range(sol.y.shape[1])])
    return sol.t, i

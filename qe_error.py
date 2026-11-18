"""
qe_error.py -- quasi-equilibrium (QE) levels and their error versus the
steady-state approximation (SSA).

Define an electrocatalytic mechanism in the MECHANISM block below, name the
slow steps of the quasi-equilibrium treatment in SLOW_STEPS, then run:
    python qe_error.py

For each overpotential the script reports

  SSA     the exact steady state (qssa.py's root find, refined in mpmath), and
          the degree of rate control X_i of every step: a central finite
          difference of ln|j| with k_i and k_-i scaled together (K_i fixed).
  QE''    one slow step per independent route; every other step is
          equilibrated. Coverages follow from those equilibria and the site
          balance, the route rates from the slow-step rate laws
          (R = sigma_S^-1 v_S), and the current from the electron counts.
          Works for any mechanism, including branched networks and
          bimolecular steps.
  QE'     serial cycles only, one slow step s: the states up to s are
          equilibrated with the empty site, the downstream states are set to
          zero, and the reverse rate of s is dropped.
  QE      serial cycles only: the reactant of s is saturated (coverage 1), so
          r = k_s'.

and the error of each level, log10(j_level / j_SSA), in decades.

For a serial cycle the errors split exactly into four terms (decades):
    rate control      -log10 X_s
    reverse rate      -log10(1 - 1/Z)
    downstream        -log10(1 - Theta_downstream)
    fixed coverage    -log10 theta_s(QE')
QE'' carries the first term, QE' the first three, QE all four.

For any mechanism the QE'' error is also the integrated rate control of the
equilibrated steps F along the path that scales k_j and k_-j of every j in F
by lambda (K_j fixed):
    ln(J_QE'' / J) = integral_0^inf  sum_{j in F} X_j(lambda)  d ln(lambda).
The integral is evaluated numerically and checked against the direct QE''
error. The total is exact; how it is split among the steps of F depends on the
path (here, all of F scaled together).

numpy, scipy and mpmath are required (matplotlib for the plot). The levels and
the SSA are evaluated with 60-digit arithmetic because coverages and the
weights w_j can span many decades.
"""
import math
import warnings
from fractions import Fraction
from types import SimpleNamespace

import mpmath as mp
import numpy as np

from io_plot import save_csv, _pyplot

mp.mp.dps = 60

# ---- physical constants -----------------------------------------------------
F = 96485                # C/mol
R = 8.3145               # J/mol/K
GAMMA = 1.0e-9           # mol/cm^2, active-site density
T = 298.15               # K
DIRECTION = +1           # -1 = cathodic forward (HER, ORR); +1 = anodic (HOR, OER)

# ============================ MECHANISM (edit me) ============================
# Same format as qssa.py. The default is the 5-step alkaline OER with ET4 as
# the rate-limiting step (upstream steps uphill, K = 0.01; prod K_i = 1).
SPECIES = ["M", "MOH", "MO", "MOOH", "MOOm"]
SITE = "M"

STEPS = [
    dict(reactants={"M": 1},    products={"MOH": 1},  n_e=1, beta=0.5, k0=1.0,  km0=100.0, label="ET1"),
    dict(reactants={"MOH": 1},  products={"MO": 1},   n_e=1, beta=0.5, k0=1.0,  km0=100.0, label="ET2"),
    dict(reactants={"MO": 1},   products={"MOOH": 1}, n_e=1, beta=0.5, k0=1.0,  km0=100.0, label="ET3"),
    dict(reactants={"MOOH": 1}, products={"MOOm": 1}, n_e=0, beta=0.0, k0=1.0,  km0=100.0, label="ChemPT"),
    dict(reactants={"MOOm": 1}, products={"M": 1},    n_e=1, beta=0.5, k0=1e-3, km0=1e-11, label="ET4"),
]

ETAS = np.linspace(0.05, 0.50, 10)   # overpotentials to scan (V)

# Slow steps of the QE treatment (labels). QE'' needs one per independent
# route; QE and QE' need a serial cycle and exactly one slow step.
SLOW_STEPS = ["ET4"]

PATH_INTEGRAL = True          # also evaluate the integrated rate control (slower)

# output controls
CSV_OUT   = "qe_error.csv"    # CSV filename to write, or None to skip
SHOW_PLOT = True              # error-vs-eta plot (needs matplotlib)
# ============================================================================


class QEError(ValueError):
    """Raised for a slow-step set that does not define a quasi-equilibrium state."""


# ------------------------------ exact linear algebra -------------------------
def _rref(rows):
    """Reduced row echelon form over the rationals. -> (matrix, pivot columns)."""
    A = [[Fraction(x) for x in r] for r in rows]
    piv, r = [], 0
    ncol = len(A[0]) if A else 0
    for c in range(ncol):
        p = next((i for i in range(r, len(A)) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        A[r] = [x / A[r][c] for x in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c] != 0:
                A[i] = [a - A[i][c] * b for a, b in zip(A[i], A[r])]
        piv.append(c); r += 1
        if r == len(A):
            break
    return A, piv


def _nullspace(rows, ncol):
    """Integer basis of the right null space of a rational matrix (list of vectors)."""
    A, piv = _rref(rows) if rows else ([], [])
    basis = []
    for fc in (c for c in range(ncol) if c not in piv):
        v = [Fraction(0)] * ncol; v[fc] = Fraction(1)
        for i, pc in enumerate(piv):
            v[pc] = -A[i][fc]
        lcm = 1
        for x in v:
            lcm = lcm * x.denominator // math.gcd(lcm, x.denominator)
        v = [x * lcm for x in v]
        g = 0
        for x in v:
            g = math.gcd(g, int(x))
        v = [int(x) // g for x in v]
        if sum(v) < 0:
            v = [-x for x in v]
        basis.append(v)
    return basis


# ------------------------------ the network ----------------------------------
class Network:
    """Mechanism plus the structures the QE analysis needs.

    Args:
        mech: any object with `species`, `site`, `steps`, `direction` and
              optional `T` (K) and `gamma` (mol/cm^2) -- e.g. a
              tutorials/core.Mechanism, or the MECHANISM block of this script.
    """

    def __init__(self, mech):
        self.mech = mech
        self.species = list(mech.species)
        self.site = mech.site
        self.steps = list(mech.steps)
        self.direction = mech.direction
        self.gamma = getattr(mech, "gamma", GAMMA)
        self.f = F / (R * getattr(mech, "T", T))
        self.labels = [st.get("label", f"step{i + 1}") for i, st in enumerate(self.steps)]
        self.idx = {s: i for i, s in enumerate(self.species)}
        self.si = self.idx[self.site]
        ns, nst = len(self.species), len(self.steps)
        self.N = [[0] * nst for _ in range(ns)]          # N[species][step], net stoichiometry
        for j, st in enumerate(self.steps):
            for sp, c in st["reactants"].items():
                self.N[self.idx[sp]][j] -= c
            for sp, c in st["products"].items():
                self.N[self.idx[sp]][j] += c
        self.NE = [st["n_e"] for st in self.steps]
        self.RCT = [[(self.idx[sp], c) for sp, c in st["reactants"].items()] for st in self.steps]
        self.PRD = [[(self.idx[sp], c) for sp, c in st["products"].items()] for st in self.steps]
        self.free = [k for k in range(ns) if k != self.si]
        # routes: integer basis of the null space of N (steps x routes)
        self.rank = len(_rref(self.N)[1])
        self.routes = _nullspace(self.N, nst)
        self.n_routes = len(self.routes)
        self.sites_conserved = all(sum(self.N[k][j] for k in range(ns)) == 0 for j in range(nst))
        self.cycle = self._serial_cycle()

    # ---- structure ----
    def step_index(self, label):
        if label not in self.labels:
            raise QEError(f"unknown step {label!r}; steps are {self.labels}")
        return self.labels.index(label)

    def _serial_cycle(self):
        """Step indices in cycle order starting at the empty site, or None if not a serial cycle."""
        if self.n_routes != 1 or len(self.steps) != len(self.species):
            return None
        nxt = {}
        for j, st in enumerate(self.steps):
            if (len(st["reactants"]) != 1 or len(st["products"]) != 1
                    or list(st["reactants"].values()) != [1] or list(st["products"].values()) != [1]):
                return None
            a, b = self.idx[next(iter(st["reactants"]))], self.idx[next(iter(st["products"]))]
            if a in nxt or a == b:
                return None
            nxt[a] = (j, b)
        order, state, seen = [], self.si, set()
        while state not in seen and state in nxt:
            seen.add(state)
            j, state = nxt[state]
            order.append(j)
        if state != self.si or len(order) != len(self.steps):
            return None
        return order

    def check_slow(self, slow):
        """Validate a QE'' slow-step set. -> (slow indices, equilibrated indices)."""
        S = sorted({self.step_index(s) if isinstance(s, str) else int(s) for s in slow})
        Fq = [j for j in range(len(self.steps)) if j not in S]
        if not self.sites_conserved:
            raise QEError("a step does not conserve the number of sites; the site balance is not a "
                          "conservation law and QE'' is not implemented for this case")
        if self.rank != len(self.species) - 1:
            raise QEError(f"the mechanism has {len(self.species) - self.rank} conservation laws; "
                          "QE'' is implemented for one site balance only")
        if len(S) != self.n_routes:
            raise QEError(f"this mechanism has {self.n_routes} independent route(s), so QE'' needs "
                          f"exactly {self.n_routes} slow step(s), one per route; got {len(S)} "
                          f"({', '.join(self.labels[j] for j in S) or 'none'})")
        loops = _nullspace([[self.N[k][j] for j in Fq] for k in range(len(self.species))], len(Fq))
        if loops:
            v = loops[0]
            names = [self.labels[Fq[i]] for i, c in enumerate(v) if c]
            prod = " * ".join(f"K_{self.labels[Fq[i]]}" + (f"^{c}" if c != 1 else "")
                              for i, c in enumerate(v) if c)
            raise QEError(f"the equilibrated steps {{{', '.join(names)}}} close a loop: their "
                          f"equilibria would force {prod} = 1, which holds at one potential at most. "
                          "Make one of them slow instead.")
        return S, Fq

    # ---- kinetics (mpmath) ----
    def rate_constants(self, eta, scale=None):
        """Butler-Volmer forward/reverse constants at eta (same convention as qssa.py), as mpf.

        `scale` multiplies k_j and k_-j of each step (dict index -> factor)."""
        kf, kr = [], []
        e = mp.mpf(eta) * mp.mpf(self.f)
        for j, st in enumerate(self.steps):
            bv = st["n_e"] * e
            b = mp.mpf(st["beta"])
            af = mp.mpf(st.get("a_fwd", 1.0)); ar = mp.mpf(st.get("a_rev", 1.0))
            if self.direction < 0:
                f_, r_ = mp.mpf(st["k0"]) * af * mp.exp(-b * bv), mp.mpf(st["km0"]) * ar * mp.exp((1 - b) * bv)
            else:
                f_, r_ = mp.mpf(st["k0"]) * af * mp.exp((1 - b) * bv), mp.mpf(st["km0"]) * ar * mp.exp(-b * bv)
            if scale and j in scale:
                f_ *= scale[j]; r_ *= scale[j]
            kf.append(f_); kr.append(r_)
        return kf, kr

    def net_rates(self, kf, kr, th):
        v = []
        for j in range(len(self.steps)):
            a = kf[j]
            for k, c in self.RCT[j]:
                a *= th[k] ** c
            b = kr[j]
            for k, c in self.PRD[j]:
                b *= th[k] ** c
            v.append(a - b)
        return v

    def rate_jacobian(self, kf, kr, th):
        """dv[j]/dtheta[k] for mass-action rates."""
        n = len(self.species)
        D = [[mp.mpf(0)] * n for _ in self.steps]
        for j in range(len(self.steps)):
            for terms, k0, sgn in ((self.RCT[j], kf[j], 1), (self.PRD[j], kr[j], -1)):
                for k, c in terms:
                    d = k0 * c * th[k] ** (c - 1)
                    for k2, c2 in terms:
                        if k2 != k:
                            d *= th[k2] ** c2
                    D[j][k] += sgn * d
        return D

    def electron_rate(self, v):
        return mp.fsum(ne * x for ne, x in zip(self.NE, v))

    def current(self, v):
        """Current density (A/cm^2): DIRECTION * F * GAMMA * sum(n_e * v)."""
        return float(self.direction * F * self.gamma * self.electron_rate(v))


# ------------------------------ SSA ------------------------------------------
def _qssa_seed(net, kf, kr):
    """Float steady state from qssa.py's root find, run on this mechanism's constants."""
    import qssa as Q
    names = ("SPECIES", "SITE", "STEPS", "DIRECTION", "idx", "si", "Nmat", "NE")
    saved = {k: getattr(Q, k) for k in names}       # restored afterwards
    try:
        steps = [dict(st, k0=float(kf[j]), km0=float(kr[j]), a_fwd=1.0, a_rev=1.0)
                 for j, st in enumerate(net.steps)]
        Q.SPECIES, Q.SITE, Q.STEPS, Q.DIRECTION = net.species, net.site, steps, net.direction
        Q.idx, Q.si = net.idx, net.si
        Q.Nmat = np.array(net.N, dtype=float)
        Q.NE = np.array(net.NE, dtype=float)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            theta = Q.steady_state(0.0)[1]          # eta = 0: constants are already applied
        if np.all(np.isfinite(theta)):
            return [mp.mpf(float(t)) for t in theta]
    except (OverflowError, ValueError, FloatingPointError):
        pass
    finally:
        for k, v in saved.items():
            setattr(Q, k, v)
    return [mp.mpf(1) / len(net.species)] * len(net.species)


def _ssa_system(net, kf, kr, th):
    v = net.net_rates(kf, kr, th)
    D = net.rate_jacobian(kf, kr, th)
    n = len(net.species)
    G = mp.zeros(n, 1); JG = mp.zeros(n, n)
    for r, k in enumerate(net.free):
        G[r] = mp.fsum(net.N[k][j] * v[j] for j in range(len(net.steps)) if net.N[k][j])
        for m in range(n):
            JG[r, m] = mp.fsum(net.N[k][j] * D[j][m] for j in range(len(net.steps)) if net.N[k][j])
    G[n - 1] = mp.fsum(th) - 1
    for m in range(n):
        JG[n - 1, m] = 1
    return v, D, G, JG


def ssa_state(net, kf, kr, seed=None, maxit=200):
    """Steady state N.v = 0 with the site balance, refined by Newton's method in mpmath.

    The starting point is `seed` or, if None, qssa.py's float solution. For a
    mechanism of first-order steps the system is linear and one Newton step is exact.

    Returns:
        (v, theta): net step rates and coverages (mpf lists).
    """
    th = list(seed) if seed is not None else _qssa_seed(net, kf, kr)
    th = [x if x > 0 else mp.mpf(0) for x in th]
    tol = mp.mpf(10) ** (-(mp.mp.dps - 12))
    for _ in range(maxit):
        v, D, G, JG = _ssa_system(net, kf, kr, th)
        d = mp.lu_solve(JG, -G)
        t = mp.mpf(1)
        while any(th[k] + t * d[k] < 0 for k in range(len(th))) and t > mp.mpf(2) ** -40:
            t /= 2
        new = [th[k] + t * d[k] for k in range(len(th))]
        done = all(abs(new[k] - th[k]) <= tol * abs(new[k]) for k in range(len(th)))
        th = [x if x > 0 else mp.mpf(0) for x in new]
        if done:
            break
    else:
        warnings.warn("SSA Newton iteration did not converge")
    return net.net_rates(kf, kr, th), th


def drc_fd(net, eta, h=1e-12, seed=None):
    """Degree of rate control of every step by a central finite difference of ln|j|
    (k_i and k_-i scaled by exp(+-h), K_i fixed)."""
    h = mp.mpf(h)
    kf, kr = net.rate_constants(eta)
    _, th0 = ssa_state(net, kf, kr, seed)
    X = []
    for i in range(len(net.steps)):
        I = []
        for sgn in (1, -1):
            kfs, krs = list(kf), list(kr)
            kfs[i] *= mp.exp(sgn * h); krs[i] *= mp.exp(sgn * h)
            I.append(net.electron_rate(ssa_state(net, kfs, krs, th0)[0]))
        X.append(float((mp.log(abs(I[0])) - mp.log(abs(I[1]))) / (2 * h)))
    return X


def drc_exact(net, kf, kr, v, th):
    """Degree of rate control of every step by implicit differentiation of the
    steady state (used along the integration path)."""
    _, D, _, JG = _ssa_system(net, kf, kr, th)
    n, nst = len(net.species), len(net.steps)
    B = mp.zeros(n, nst)
    for r, k in enumerate(net.free):
        for j in range(nst):
            B[r, j] = -net.N[k][j] * v[j]
    I = net.electron_rate(v)
    LU = mp.inverse(JG)
    dth = LU * B
    X = []
    for j in range(nst):
        dI = net.NE[j] * v[j]
        for i in range(nst):
            if net.NE[i]:
                dI += net.NE[i] * mp.fsum(D[i][m] * dth[m, j] for m in range(n))
        X.append(dI / I)
    return X, I


# ------------------------------ QE levels ------------------------------------
def qepp_state(net, kf, kr, slow):
    """QE'': coverages from the equilibria of all non-slow steps plus the site balance
    (solved in log space), route rates from the slow-step rate laws.

    Returns:
        dict(theta, v, R): coverages, net rates of every step (v = sigma R), route rates.
    """
    S, Fq = net.check_slow(slow)
    n = len(net.species)
    A = mp.zeros(n, n); b = mp.zeros(n, 1)
    for r, j in enumerate(Fq):                  # sum_k N_kj ln(theta_k) = ln K_j
        if kr[j] == 0:
            raise QEError(f"step {net.labels[j]} is irreversible and cannot be equilibrated")
        for k in range(n):
            A[r, k] = net.N[k][j]
        b[r] = mp.log(kf[j] / kr[j])
    A[n - 1, net.si] = 1                         # reference: ln(w_site) = 0
    lw = mp.lu_solve(A, b)
    top = max(lw)
    w = [mp.exp(x - top) for x in lw]
    th = [x / mp.fsum(w) for x in w]
    v_rate = net.net_rates(kf, kr, th)
    sig = net.routes                             # list of route vectors over steps
    nr = net.n_routes
    sS = mp.matrix([[sig[q][j] for q in range(nr)] for j in S])
    Rr = mp.lu_solve(sS, mp.matrix([v_rate[j] for j in S]))
    v = [mp.fsum(sig[q][j] * Rr[q] for q in range(nr)) for j in range(len(net.steps))]
    return dict(theta=th, v=v, R=[Rr[q] for q in range(nr)])


def serial_levels(net, kf, kr, s, r_ssa):
    """Closed-form QE, QE', QE'' and the four-term breakdown for slow step s of a serial cycle.

    Returns rates per cycle (mpf) and the terms in decades (floats), or None if
    the mechanism is not a serial cycle.
    """
    if net.cycle is None:
        return None
    order = net.cycle
    p = order.index(s)                           # position of s in the cycle
    K = [kf[j] / kr[j] for j in order]
    m = len(order)
    lw = [mp.mpf(0)]
    for i in range(m):
        lw.append(lw[-1] + mp.log(K[i]))         # ln w_j (w_0 = site), ln Z = lw[m]
    lZ = lw[m]
    top = max(lw)
    w = [mp.exp(x - top) for x in lw[:m]]        # w_j scaled by exp(-top)
    Zinv = mp.exp(-lZ)
    A = mp.fsum(w[:p + 1])
    B = mp.fsum(w[p + 1:]) * Zinv
    ks = kf[s]
    r_qe = ks
    r_qep = ks * w[p] / A
    r_qepp = ks * w[p] * (1 - Zinv) / (A + B)
    X = r_ssa / r_qepp
    lg = lambda x: float(mp.log10(x)) if x > 0 else float("nan")
    return dict(
        r_qe=r_qe, r_qep=r_qep, r_qepp=r_qepp, X_s=float(X), Z=lZ,
        theta_down=float(B / (A + B)), theta_s_qep=float(w[p] / A),
        rate_control=-lg(X), reverse=-lg(1 - Zinv), downstream=-lg(A / (A + B)),
        coverage=-lg(w[p] / A))


def path_integral(net, eta, slow, j_qepp=None, width=2.0, nodes=10, tol=1e-14, u_max=230.0):
    """Integrated rate control of the equilibrated steps from lambda = 1 to infinity.

    k_j and k_-j of every equilibrated step j are multiplied by lambda; the
    integrand sum_j X_j(lambda) is integrated in u = ln(lambda) with
    Gauss-Legendre panels until it decays below `tol`.

    Returns:
        dict(total, shares, check, ok, note): `total` and `shares` in decades;
        `check` = total - log10(J_QE''/J_SSA), which should be ~0.
    """
    S, Fq = net.check_slow(slow)
    kf0, kr0 = net.rate_constants(eta)
    v0, th = ssa_state(net, kf0, kr0)
    I0 = net.electron_rate(v0)
    if j_qepp is None:
        j_qepp = net.electron_rate(qepp_state(net, kf0, kr0, slow)["v"])
    xg, wg = np.polynomial.legendre.leggauss(nodes)
    xg = [mp.mpf(float(x)) for x in xg]; wg = [mp.mpf(float(x)) for x in wg]
    shares = [mp.mpf(0)] * len(Fq)
    u0, ok, note, last = mp.mpf(0), True, "", None
    sign_change = False
    while True:
        h = mp.mpf(width) / 2
        for x, wq in zip(xg, wg):
            u = u0 + h * (x + 1)
            lam = mp.exp(u)
            kf, kr = net.rate_constants(eta, {j: lam for j in Fq})
            v, th = ssa_state(net, kf, kr, th)
            X, I = drc_exact(net, kf, kr, v, th)
            if mp.sign(I) != mp.sign(I0):
                sign_change = True
            for q, j in enumerate(Fq):
                shares[q] += h * wq * X[j]
            last = (u, [X[j] for j in Fq])
        u0 += width
        if max(abs(x) for x in last[1]) < tol:
            for q in range(len(Fq)):             # tail, assuming X ~ 1/lambda
                shares[q] += last[1][q] * mp.exp(-(u0 - last[0]))
            break
        if u0 > u_max:
            ok, note = False, "integrand did not decay; J(lambda) may not approach a finite J_QE''"
            break
    if sign_change or mp.sign(j_qepp) != mp.sign(I0):
        ok, note = False, "J changes sign along the path; the identity does not hold"
    ln10 = mp.log(10)
    total = mp.fsum(shares)
    direct = mp.log(abs(j_qepp / I0)) if mp.sign(j_qepp) == mp.sign(I0) else mp.nan
    return dict(total=float(total / ln10),
                shares={net.labels[j]: float(shares[q] / ln10) for q, j in enumerate(Fq)},
                check=float((total - direct) / ln10) if ok else float("nan"), ok=ok, note=note)


# ------------------------------ driver ---------------------------------------
def _log_ratio(a, b):
    return float(mp.log10(a / b)) if a != 0 and b != 0 and mp.sign(a) == mp.sign(b) else float("nan")


def analyze(net, eta, slow, path=True):
    """Every quantity at one overpotential. -> dict (currents A/cm^2, errors in decades)."""
    S, Fq = net.check_slow(slow)
    kf, kr = net.rate_constants(eta)
    v, th = ssa_state(net, kf, kr)
    I = net.electron_rate(v)
    out = dict(eta=float(eta), j_ssa=net.current(v), theta_ssa=[float(x) for x in th],
               X=dict(zip(net.labels, drc_fd(net, eta, seed=th))))
    q = qepp_state(net, kf, kr, slow)
    Iq = net.electron_rate(q["v"])
    out.update(j_qepp=net.current(q["v"]), err_qepp=_log_ratio(Iq, I),
               theta_qepp=[float(x) for x in q["theta"]])
    out["serial"] = None
    if net.cycle is not None and len(S) == 1:
        s = S[0]
        L = serial_levels(net, kf, kr, s, v[s])
        ntot = sum(net.NE)
        jl = lambda r: float(net.direction * F * net.gamma * ntot * r)
        out["serial"] = L
        out.update(j_qep=jl(L["r_qep"]), err_qep=_log_ratio(L["r_qep"], v[s]),
                   j_qe=jl(L["r_qe"]), err_qe=_log_ratio(L["r_qe"], v[s]))
    if path:
        out["path"] = path_integral(net, eta, slow, j_qepp=Iq)
    return out


def scan(mech, etas, slow, path=True):
    """Run `analyze` across overpotentials. -> (Network, list of dicts)."""
    net = Network(mech)
    net.check_slow(slow)
    return net, [analyze(net, e, slow, path) for e in etas]


def describe(net, slow):
    """Short text on the routes, the slow set and which levels apply."""
    S, Fq = net.check_slow(slow)
    lines = [f"# routes: {net.n_routes}  "
             + "  ".join("[" + " ".join(f"{c:+d}" for c in r) + "]" for r in net.routes)
             + "  (coefficients over " + ", ".join(net.labels) + ")",
             f"# QE'' slow: {{{', '.join(net.labels[j] for j in S)}}}   "
             f"equilibrated: {{{', '.join(net.labels[j] for j in Fq)}}}"]
    if net.n_routes > 1:
        lines.append("# branched network: QE' is not unique and classical QE is not defined; both skipped")
    elif net.cycle is None:
        lines.append("# single route but not a serial cycle of first-order steps: QE and QE' skipped")
    elif len(S) != 1:
        lines.append("# QE and QE' need exactly one slow step")
    return "\n".join(lines)


def print_report(net, slow, rows):
    labs = net.labels
    print(describe(net, slow))
    print("# degree of rate control X_i (SSA)")
    print(f"{'eta/V':>7}{'j_SSA':>12}  " + "".join(f"{l:>9}" for l in labs))
    for r in rows:
        print(f"{r['eta']:7.3f}{r['j_ssa']:12.3e}  " + "".join(f"{r['X'][l]:9.4f}" for l in labs))
    serial = rows[0]["serial"] is not None
    print("# currents (A/cm^2) and error vs SSA, log10(j_level/j_SSA) in decades")
    head = f"{'eta/V':>7}{'j_SSA':>12}{'j_QE″':>12}{'err':>9}"
    if serial:
        head += f"{'j_QE′':>12}{'err':>9}{'j_QE':>12}{'err':>9}"
    print(head)
    for r in rows:
        line = f"{r['eta']:7.3f}{r['j_ssa']:12.3e}{r['j_qepp']:12.3e}{r['err_qepp']:9.4f}"
        if serial:
            line += f"{r['j_qep']:12.3e}{r['err_qep']:9.4f}{r['j_qe']:12.3e}{r['err_qe']:9.4f}"
        print(line)
    if serial:
        print("# serial breakdown (decades): QE″ = rate control; QE′ = + reverse + downstream; "
              "QE = + fixed coverage")
        print(f"{'eta/V':>7}{'rate ctrl':>11}{'reverse':>11}{'downstrm':>11}{'coverage':>11}{'sum-QE':>10}")
        for r in rows:
            L = r["serial"]
            tot = L["rate_control"] + L["reverse"] + L["downstream"] + L["coverage"]
            print(f"{r['eta']:7.3f}{L['rate_control']:11.4g}{L['reverse']:11.4g}{L['downstream']:11.4g}"
                  f"{L['coverage']:11.4g}{tot - r['err_qe']:10.1e}")
    if "path" in rows[0]:
        fl = list(rows[0]["path"]["shares"])
        print("# integrated rate control of the equilibrated steps (decades). The total is exact;")
        print("# the split among steps depends on the path (all equilibrated steps scaled together).")
        print(f"{'eta/V':>7}{'total':>10}" + "".join(f"{l:>10}" for l in fl) + f"{'total-QE″':>11}")
        for r in rows:
            P = r["path"]
            print(f"{r['eta']:7.3f}{P['total']:10.4f}" + "".join(f"{P['shares'][l]:10.4f}" for l in fl)
                  + f"{P['check']:11.1e}" + (f"   ! {P['note']}" if not P["ok"] else ""))


def table(net, rows):
    """Header and rows for CSV export."""
    labs = net.labels
    header = ["eta_V", "j_SSA", *[f"X_{l}" for l in labs], "j_QEpp", "err_QEpp_dec"]
    serial = rows[0]["serial"] is not None
    if serial:
        header += ["j_QEp", "err_QEp_dec", "j_QE", "err_QE_dec",
                   "term_rate_control", "term_reverse", "term_downstream", "term_coverage"]
    path = "path" in rows[0]
    if path:
        fl = list(rows[0]["path"]["shares"])
        header += ["path_total_dec", *[f"path_{l}" for l in fl], "path_minus_QEpp"]
    out = []
    for r in rows:
        row = [r["eta"], r["j_ssa"], *[r["X"][l] for l in labs], r["j_qepp"], r["err_qepp"]]
        if serial:
            L = r["serial"]
            row += [r["j_qep"], r["err_qep"], r["j_qe"], r["err_qe"],
                    L["rate_control"], L["reverse"], L["downstream"], L["coverage"]]
        if path:
            P = r["path"]
            row += [P["total"], *[P["shares"][l] for l in fl], P["check"]]
        out.append(row)
    return header, out


def show_plot(net, rows, title=None):
    plt = _pyplot()
    if plt is None:
        return
    eta = [r["eta"] for r in rows]
    serial = rows[0]["serial"] is not None
    ncol = 2 if (serial or "path" in rows[0]) else 1
    fig, axs = plt.subplots(1, ncol, figsize=(5 * ncol, 3.8), squeeze=False)
    ax = axs[0, 0]
    ax.plot(eta, [r["err_qepp"] for r in rows], "o-", label="QE″")
    if serial:
        ax.plot(eta, [r["err_qep"] for r in rows], "s-", label="QE′")
        ax.plot(eta, [r["err_qe"] for r in rows], "^-", label="QE")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel(r"overpotential $\eta$ / V"); ax.set_ylabel(r"log$_{10}(j_{level}/j_{SSA})$")
    ax.legend()
    if ncol == 2:
        ax = axs[0, 1]
        if serial:
            for key, lab in (("rate_control", "rate control"), ("reverse", "reverse rate"),
                             ("downstream", "downstream"), ("coverage", "fixed coverage")):
                ax.plot(eta, [r["serial"][key] for r in rows], "o-", label=lab)
            ax.set_ylabel("term / decades")
        else:
            for l in rows[0]["path"]["shares"]:
                ax.plot(eta, [r["path"]["shares"][l] for r in rows], "o-", label=f"∫X_{l}")
            ax.plot(eta, [r["path"]["total"] for r in rows], "k--", label="total")
            ax.set_ylabel("integrated rate control / decades")
        ax.set_xlabel(r"overpotential $\eta$ / V"); ax.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()


MECHANISM = SimpleNamespace(species=SPECIES, site=SITE, steps=STEPS, direction=DIRECTION, T=T, gamma=GAMMA)

if __name__ == "__main__":
    try:
        net, rows = scan(MECHANISM, ETAS, SLOW_STEPS, path=PATH_INTEGRAL)
    except QEError as e:
        raise SystemExit(f"qe_error.py: {e}")
    print(f"# QE levels vs SSA -- {', '.join(net.labels)}")
    print_report(net, SLOW_STEPS, rows)
    if CSV_OUT:
        save_csv(CSV_OUT, *table(net, rows))
    if SHOW_PLOT:
        show_plot(net, rows, title="QE levels vs SSA")

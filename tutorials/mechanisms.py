"""
mechanisms.py -- the four reaction mechanisms of the accompanying paper,
expressed as core.Mechanism objects, together with the paper's designed
rate-limiting-step (RLS) cases.

Every case is built the same way ("delta convention"): the designated RLS gets a
slow forward rate constant (delta = 1e-3) while the other steps stay fast, and
the reverse rate constants are chosen so the product of the equilibrium
constants over the cycle is exactly 1 -- i.e. all cases of a reaction share the
same equilibrium potential and differ ONLY in which step is kinetically
limiting. That is what makes the comparison meaningful: any difference between
the cases is kinetic, not thermodynamic.

Reactions
    HER  acid Volmer-Heyrovsky-Tafel      (cathodic, eta < 0)   2 species
    HOR  the anodic branch of the same    (same mechanism, eta > 0)
    ORR  acid associative 5-step, 4 e-    (cathodic, eta < 0)   5 species
    OER  alkaline associative 5-step, 4 e- with one chemical
         proton-transfer step             (anodic,  eta > 0)    5 species

The rate constants are the ones behind the paper's figures. Unit bulk
activities are used throughout (the a_fwd/a_rev factors default to 1), so the
constants below are the apparent constants directly.
"""
import numpy as np

from core import Mechanism

DELTA = 1e-3   # forward rate constant of a designed RLS ("kinetically slow")


def local_slope(etas, j):
    """Local Tafel slope b(eta) in mV/dec from a scanned polarization curve.

    Args:
        etas: overpotential grid (V), monotone.
        j:    currents on that grid (A/cm^2).

    Returns:
        |d eta / d log10|j|| * 1e3, evaluated pointwise on the grid.
    """
    dlog = np.gradient(np.log10(np.abs(j)), etas)
    with np.errstate(divide="ignore"):
        return np.abs(1.0 / dlog) * 1e3


# ============================== HER / HOR ====================================
# Acid Volmer-Heyrovsky-Tafel, written in the cathodic (HER) forward direction:
#   Volmer:     * + H+ + e-  <->  H*
#   Heyrovsky:  H* + H+ + e- <->  * + H2
#   Tafel:      2 H*         <->  2 * + H2         (chemical)
# HOR is the SAME mechanism run on the anodic branch (eta > 0); no re-writing
# of the steps is needed, the Butler-Volmer factors simply favor the reverse
# rates there.
#
# Thermodynamic consistency across the cycle: K1*K2 = 1 and K1^2*K3 = 1, so
# every case shares eta = 0 as the equilibrium potential.

def her_mechanism(k1, km1, k2, km2, k3, km3):
    """Build the acid HER/HOR Mechanism from the six rate constants."""
    return Mechanism(
        species=["*", "H"], site="*", direction=-1,
        steps=[
            dict(reactants={"*": 1}, products={"H": 1}, n_e=1, beta=0.5,
                 k0=k1, km0=km1, label="Volmer"),
            dict(reactants={"H": 1}, products={"*": 1}, n_e=1, beta=0.5,
                 k0=k2, km0=km2, label="Heyrovsky"),
            dict(reactants={"H": 2}, products={"*": 2}, n_e=0, beta=0.0,
                 k0=k3, km0=km3, label="Tafel"),
        ])


# The paper's canonical HER cases (kinetics.py CASES): each RLS forward = delta,
# non-RLS backwards fixed by K1*K2 = 1 and K1^2*K3 = 1.
HER_CASES = {
    "Volmer":    dict(k1=1e-3, km1=1.0,       k2=1.0,  km2=1e-3,     k3=1.0,  km3=1e-6),
    "Heyrovsky": dict(k1=1.0,  km1=1e-3,      k2=1e-3, km2=1.0,      k3=1.0,  km3=1e6),
    "Tafel":     dict(k1=1.0,  km1=10**-1.5,  k2=1.0,  km2=10**1.5,  k3=1e-3, km3=1.0),
}

# The paper's HOR cases (fig3, CASES_HOR): tuned so each named step controls the
# ANODIC branch; "Parallel" has Heyrovsky, Tafel, and Volmer co-controlling.
HOR_CASES = {
    "Volmer":    dict(k1=1e-6, km1=1e-3, k2=1e3,  km2=1.0,   k3=1e-6, km3=1e-12),
    "Heyrovsky": dict(k1=1.0,  km1=1.0,  k2=1e-3, km2=1e-3,  k3=1e-6, km3=1e-6),
    "Tafel":     dict(k1=1.0,  km1=1e2,  k2=1e-8, km2=1e-10, k3=1.0,  km3=1e-4),
    "Parallel":  dict(k1=0.03, km1=0.03, k2=1.0,  km2=1.0,   k3=1.0,  km3=1.0),
}

HER_ETAS = np.linspace(-0.02, -0.45, 120)   # cathodic scan
HOR_ETAS = np.linspace(+0.01, +0.60, 120)   # anodic scan (same mechanism)


# ================================== ORR ======================================
# Acid associative 4-electron ORR (5 steps, cathodic forward; fig4):
#   1 ChemAds:  M    + O2            <->  MOO             (chemical)
#   2 PCET1:    MOO  + H3O+ + e-     <->  MOOH  (+ H2O)
#   3 PCET2:    MOOH + H3O+ + e-     <->  MO    (+ 2 H2O)
#   4 PCET3:    MO   + H3O+ + e-     <->  MOH   (+ H2O)
#   5 PCET4:    MOH  + H3O+ + e-     <->  M     (+ 2 H2O)
# Overall O2 + 4 H3O+ + 4 e- <-> 6 H2O. Cardinal slopes near onset (beta = 0.5,
# unsaturated RLS reactant): ChemAds -> saturating, PCET1 -> 118, PCET2 -> 39,
# PCET3 -> 24, PCET4 -> 17 mV/dec.

ORR_SPECIES = ["M", "MOO", "MOOH", "MO", "MOH"]
_ORR_STEPS = [("M", "MOO", 0, "ChemAds"), ("MOO", "MOOH", 1, "PCET1"),
              ("MOOH", "MO", 1, "PCET2"), ("MO", "MOH", 1, "PCET3"),
              ("MOH", "M", 1, "PCET4")]
ORR_CASE_NAMES = ("ChemAds", "PCET1", "PCET2", "PCET3", "PCET4")


def orr_mechanism(kf, kb):
    """Build the acid ORR Mechanism from per-step forward/backward constants.

    Args:
        kf, kb: dicts {step index 1..5: rate constant}.
    """
    steps = [dict(reactants={r: 1}, products={p: 1}, n_e=ne,
                  beta=0.5 if ne else 0.0, k0=kf[i + 1], km0=kb[i + 1], label=lab)
             for i, (r, p, ne, lab) in enumerate(_ORR_STEPS)]
    return Mechanism(species=ORR_SPECIES, site="M", direction=-1, steps=steps)


def orr_case(rls, k_up=0.01, fast=1e5, delta=DELTA):
    """The paper's fig4 "fixed-RLS" ORR case for `rls`.

    Cardinal landscape: steps upstream of the RLS are uphill (K = k_up) so the
    RLS reactant stays unsaturated near onset; downstream steps absorb the
    balance so prod(K_i) = 1. The RLS forward is slow (delta) and every NON-RLS
    step is additionally sped up by `fast` (forward and backward together, so no
    K changes), pinning the named step as the unique bottleneck over the whole
    scan. This isolates the regime where the Tafel slope changes with eta while
    the RLS does not.
    """
    rls_i = ORR_CASE_NAMES.index(rls) + 1
    n_down = 5 - rls_i
    Ktar = {i: k_up for i in range(1, rls_i)}
    if n_down > 0:
        Ktar[rls_i] = 1.0
        K_down = (1.0 / k_up ** (rls_i - 1)) ** (1.0 / n_down)
        for i in range(rls_i + 1, 6):
            Ktar[i] = K_down
    else:
        Ktar[rls_i] = 1.0 / k_up ** (rls_i - 1)
    kf = {i: 1.0 for i in range(1, 6)}; kf[rls_i] = delta
    kb = {i: kf[i] / Ktar[i] for i in range(1, 6)}
    for i in range(1, 6):                       # non-RLS steps made fast, K kept
        if i != rls_i:
            kf[i] *= fast; kb[i] *= fast
    return orr_mechanism(kf, kb)


ORR_ETAS = np.linspace(-0.004, -0.55, 200)


# ================================== OER ======================================
# Alkaline associative 4-electron OER (5 steps, anodic forward; fig6):
#   1 ET1:     M    + OH-  <->  MOH  + e-
#   2 ET2:     MOH  + OH-  <->  MO   + H2O + e-
#   3 ET3:     MO   + OH-  <->  MOOH + e-
#   4 ChemPT:  MOOH + OH-  <->  MOO- + H2O          (chemical)
#   5 ET4:     MOO-        <->  M    + O2 + e-
# Overall 4 OH- <-> O2 + 2 H2O + 4 e-. Cardinal slopes near onset:
# ET1 -> 118, ET2 -> 39, ET3 -> 24, ChemPT -> saturating, ET4 -> 17 mV/dec;
# all collapse to 118 once the RLS reactant saturates (the Mefford degeneracy).

OER_SPECIES = ["M", "MOH", "MO", "MOOH", "MOOm"]
_OER_STEPS = [("M", "MOH", 1, "ET1"), ("MOH", "MO", 1, "ET2"),
              ("MO", "MOOH", 1, "ET3"), ("MOOH", "MOOm", 0, "ChemPT"),
              ("MOOm", "M", 1, "ET4")]
OER_CASE_NAMES = ("ET1", "ET2", "ET3", "ChemPT", "ET4")

# The paper's fig6 cardinal landscapes (kinetics_oer_alkaline
# CARDINAL_LANDSCAPES_5): per-step K_i with prod(K_i) = 1, upstream steps
# uphill (K = 0.01) so each RLS reactant stays unsaturated into the anodic
# window.
_K_UP = 0.01
OER_LANDSCAPES = {
    "ET1":    [1.0,   1.0,   1.0,   1.0,   1.0],
    "ET2":    [_K_UP, 1.0,   4.6416, 4.6416, 4.6416],
    "ET3":    [_K_UP, _K_UP, 1.0,   100.0, 100.0],
    "ChemPT": [_K_UP, _K_UP, _K_UP, 1.0,   1.0e6],
    "ET4":    [_K_UP, _K_UP, _K_UP, _K_UP, 1.0e8],
}


def oer_mechanism(kf, kb):
    """Build the alkaline OER Mechanism from per-step forward/backward constants."""
    steps = [dict(reactants={r: 1}, products={p: 1}, n_e=ne,
                  beta=0.5 if ne else 0.0, k0=kf[i + 1], km0=kb[i + 1], label=lab)
             for i, (r, p, ne, lab) in enumerate(_OER_STEPS)]
    return Mechanism(species=OER_SPECIES, site="M", direction=+1, steps=steps)


def oer_case(rls, delta=DELTA):
    """The paper's fig6 cardinal OER case for `rls`: RLS forward = delta, other
    forwards = 1, backwards from the OER_LANDSCAPES equilibrium constants."""
    idx = OER_CASE_NAMES.index(rls) + 1
    Ks = OER_LANDSCAPES[rls]
    kf = {i: (delta if i == idx else 1.0) for i in range(1, 6)}
    kb = {i: kf[i] / Ks[i - 1] for i in range(1, 6)}
    return oer_mechanism(kf, kb)


OER_ETAS = np.linspace(0.004, 0.55, 200)

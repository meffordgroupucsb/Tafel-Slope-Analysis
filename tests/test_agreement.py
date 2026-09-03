"""Smoke test: the independent methods must agree on the default HER mechanism.

Checks that (i) the numerical (BDF) and QSSA steady states of the standalone
scripts agree to near machine precision, (ii) both reproduce reference values,
and (iii) the tutorials engine reproduces the standalone scripts. Runs in a few
seconds with numpy and scipy only; the stochastic method is exercised briefly
at reduced settings.

Run from anywhere:  python tests/test_agreement.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tutorials"))

import numerical as N        # noqa: E402
import qssa as Q             # noqa: E402
import gillespie as G        # noqa: E402
import core                  # noqa: E402
import mechanisms as M       # noqa: E402

ETAS = (-0.10, -0.20, -0.30)

# reference currents for the default HER mechanism (Volmer-RLS delta set)
REF_J = {-0.10: -1.3221e-06, -0.20: -9.4452e-06, -0.30: -6.6154e-05}

failures = []


def check(tag, ok):
    print(("ok   " if ok else "FAIL ") + tag)
    if not ok:
        failures.append(tag)


# (i) numerical vs QSSA, standalone scripts
for eta in ETAS:
    jn = N.steady_state(eta)[0]
    jq = Q.steady_state(eta)[0]
    check(f"numerical == qssa at eta={eta:+.2f}  (rel {abs(jn - jq) / abs(jn):.1e})",
          abs(jn - jq) / abs(jn) < 1e-9)

# (ii) reference values
for eta, ref in REF_J.items():
    jn = N.steady_state(eta)[0]
    check(f"numerical matches reference at eta={eta:+.2f}  ({jn:.4e} vs {ref:.4e})",
          abs(jn - ref) / abs(ref) < 1e-3)

# (iii) tutorials engine vs standalone scripts (same mechanism, same constants)
mech = M.her_mechanism(**M.HER_CASES["Volmer"])
for eta in ETAS:
    jc = core.qssa_state(mech, eta)[0]
    jn = N.steady_state(eta)[0]
    check(f"tutorials core matches scripts at eta={eta:+.2f}  (rel {abs(jc - jn) / abs(jn):.1e})",
          abs(jc - jn) / abs(jn) < 1e-9)

# (iv) stochastic method, brief run: within 10% of deterministic at eta=-0.30
G.N_BURN, G.N_PROD = 20000, 80000       # reduced event budget for a fast test
rng = np.random.default_rng(0)
j_gmc, _ = G.run_trajectory(-0.30, 500, rng)
jn = N.steady_state(-0.30)[0]
check(f"gillespie within 10% at eta=-0.30  ({j_gmc:.3e} vs {jn:.3e})",
      abs(j_gmc - jn) / abs(jn) < 0.10)

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)

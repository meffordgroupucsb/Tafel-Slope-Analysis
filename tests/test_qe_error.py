"""Checks for qe_error.py: QE levels, the serial breakdown, and the path integral.

(i)   SSA in qe_error.py reproduces qssa.py on the default HER mechanism;
(ii)  finite-difference and implicit DRC agree and sum to one;
(iii) the serial four-term breakdown sums to each level's error (1e-10);
(iv)  the QE'' error equals the integrated rate control (path integral);
(v)   QE' equals classical QE when the RLS is the first step;
(vi)  inconsistent slow-step sets are refused;
(vii) reference numbers: OER ET4 budget and branched-ORR current ratios.

Run from anywhere:  python tests/test_qe_error.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tutorials"))

import qssa as Q             # noqa: E402
import qe_error as E         # noqa: E402
import mechanisms as M       # noqa: E402
from types import SimpleNamespace   # noqa: E402

failures = []


def check(tag, ok):
    print(("ok   " if ok else "FAIL ") + tag)
    if not ok:
        failures.append(tag)


# (i) SSA vs qssa.py, default HER mechanism of qssa.py
her = SimpleNamespace(species=Q.SPECIES, site=Q.SITE, steps=Q.STEPS, direction=Q.DIRECTION)
net_her = E.Network(her)
for eta in (-0.10, -0.20, -0.30):
    kf, kr = net_her.rate_constants(eta)
    j = net_her.current(E.ssa_state(net_her, kf, kr)[0])
    jq = Q.steady_state(eta)[0]
    check(f"SSA matches qssa.py at eta={eta:+.2f}  (rel {abs(j - jq) / abs(jq):.1e})",
          abs(j - jq) / abs(jq) < 1e-9)

# (ii) DRC: finite difference vs implicit, sum rule (HER has a bimolecular step)
for net, eta in ((net_her, -0.2), (E.Network(M.oer_case("ET2")), 0.3)):
    kf, kr = net.rate_constants(eta)
    v, th = E.ssa_state(net, kf, kr)
    Xfd = E.drc_fd(net, eta)
    Xex = [float(x) for x in E.drc_exact(net, kf, kr, v, th)[0]]
    check(f"DRC finite difference == implicit ({net.labels[0]}..., eta={eta})  "
          f"(max diff {max(abs(a - b) for a, b in zip(Xfd, Xex)):.1e})",
          max(abs(a - b) for a, b in zip(Xfd, Xex)) < 1e-10)
    check(f"DRC sum rule ({net.labels[0]}..., eta={eta})  (sum {sum(Xfd):.12f})", abs(sum(Xfd) - 1) < 1e-10)

# (iii) serial breakdown sums to the totals; X_s closed form == DRC
for rls in M.OER_CASE_NAMES:
    net = E.Network(M.oer_case(rls))
    for eta in (0.05, 0.25, 0.45):
        r = E.analyze(net, eta, [rls], path=False)
        L = r["serial"]
        e1 = abs(L["rate_control"] - r["err_qepp"])
        e2 = abs(L["rate_control"] + L["reverse"] + L["downstream"] - r["err_qep"])
        e3 = abs(L["rate_control"] + L["reverse"] + L["downstream"] + L["coverage"] - r["err_qe"])
        check(f"OER {rls} eta={eta}: breakdown sums to QE'', QE', QE errors  (max {max(e1, e2, e3):.1e})",
              max(e1, e2, e3) < 1e-10)
        check(f"OER {rls} eta={eta}: X_s = r/r_QE'' equals the DRC  "
              f"({L['X_s']:.10f} vs {r['X'][rls]:.10f})", abs(L["X_s"] - r["X"][rls]) < 1e-9)

# (iv) path integral == QE'' error (serial and branched)
cases = [(M.oer_case("ET4"), ["ET4"], 0.4), (M.oer_case("ET3"), ["ET3"], 0.2)]
orr = M.orr_branched_mechanism()
cases += [(orr, s, eta) for s in (["PCET2", "PCET3"], ["PCET1", "PCET2"], ["PCET1", "PCET3"])
          for eta in (-0.53, -0.93)]
cases += [(her, s, -0.2) for s in (["Volmer", "Tafel"], ["Heyrovsky", "Tafel"])]   # bimolecular Tafel
for mech, slow, eta in cases:
    net = E.Network(mech)
    r = E.analyze(net, eta, slow)
    P = r["path"]
    check(f"path integral == QE'' error  {slow} eta={eta}  ({P['total']:.6f} vs {r['err_qepp']:.6f})",
          P["ok"] and abs(P["total"] - r["err_qepp"]) < 1e-8)

# (v) QE' == classical QE when the RLS is the first step
net = E.Network(M.oer_case("ET1"))
for eta in (0.1, 0.3):
    r = E.analyze(net, eta, ["ET1"], path=False)
    check(f"QE' == QE for first-step RLS at eta={eta}  ({r['j_qep']:.6e} vs {r['j_qe']:.6e})",
          abs(r["j_qep"] - r["j_qe"]) <= 1e-12 * abs(r["j_qe"]))

# (vi) inconsistent slow-step sets are refused
net = E.Network(orr)
for bad, why in ((["PCET3", "PCET4"], "loop"), (["PCET4", "PCET5"], "loop"),
                 (["PCET1"], "route"), (["PCET1", "PCET2", "PCET3"], "route"), (["PCETX", "PCET1"], "unknown")):
    try:
        net.check_slow(bad)
        check(f"refuses {bad}", False)
    except E.QEError as e:
        check(f"refuses {bad} ({why}): {str(e)[:70]}...", why in str(e))

# (vii) reference numbers
#  OER ET4 at eta = 0.4 V: qepp_oer.budget gives QE'' 0.0102231 and QE 2.0145520 decades
#  (its F, R constants differ slightly, hence the 1e-5 tolerance).
r = E.analyze(E.Network(M.oer_case("ET4")), 0.4, ["ET4"], path=False)
check(f"OER ET4 budget at 0.4 V  (QE'' {r['err_qepp']:.7f}, QE {r['err_qe']:.7f})",
      abs(r["err_qepp"] - 0.0102231) < 1e-5 and abs(r["err_qe"] - 2.0145520) < 1e-5)
#  Branched ORR at eta = -0.93 V: QE''/SSA = 4/3 for {PCET1, PCET2}, 2/3 for {PCET1, PCET3}.
for slow, ref in ((["PCET1", "PCET2"], 4 / 3), (["PCET1", "PCET3"], 2 / 3)):
    r = E.analyze(net, -0.93, slow, path=False)
    ratio = 10 ** r["err_qepp"]
    check(f"ORR {slow} j_QE''/j_SSA at -0.93 V = {ratio:.6f} (ref {ref:.6f})", abs(ratio - ref) < 1e-4)

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)

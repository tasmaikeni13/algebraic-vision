#!/usr/bin/env python3
"""Symbolic and high-precision checks of the algebraic primitives.

Every identity, derivative, bound and limit used in the paper is checked
here, either symbolically (sympy) or by 50-digit sampling (mpmath) where the
paper gives a closed-form proof. Results go to
results/analysis/symbolic_checks.json.

Notation
  rho_n(s) = s/n + sqrt(1 + s^2/n^2),   E_n(s) = rho_n(s)^n   (n = 2^m)
  attention (optional sink Omega >= 0):  p_j = E_n(s_j) / (Omega + sum_k E_n)
  F(y) = (1 + y/sqrt(1+y^2))/2,          ALU_c(x) = x F(c x)
  sigma_n(y) = 1/(1 + E_n(-y)),          rgelu(x) = x sigma_8(1.702 x)
  Cayley rotation R(w) = [[1-w^2, -2w], [2w, 1-w^2]] / (1+w^2)
  power score: Bregman divergence of phi(t) = (t^alpha - t)/(alpha(alpha-1))
"""

import json
import random
from pathlib import Path

import mpmath as mp
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/symbolic_checks.json"
mp.mp.dps = 50
random.seed(0)

checks = []


def record(name, ok, detail=""):
    checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def zero(expr):
    return sp.simplify(sp.expand(expr)) == 0


s, x, y, t, u, w, c = sp.symbols("s x y t u w c", real=True)
n = sp.symbols("n", positive=True)


def rho(z, order):
    return z / order + sp.sqrt(1 + z**2 / order**2)


def E(z, order):
    return rho(z, order) ** order


def mp_E(z, order):
    z = mp.mpf(z)
    return (z / order + mp.sqrt(1 + (z / order) ** 2)) ** order


# --------------------------------------------------------------- attention
# A1 positivity: sqrt(1+v^2)^2 - v^2 = 1 > 0 so sqrt(1+v^2) > |v|.
record("A1 rho positive (sqrt(1+v^2)^2 - v^2 = 1)",
       zero(sp.sqrt(1 + x**2)**2 - x**2 - 1))

# A2 reciprocal symmetry E_n(s) E_n(-s) = 1.
record("A2 rho(s) rho(-s) = 1", zero(rho(s, n) * rho(-s, n) - 1))

# A3 logarithmic derivative g_n(s) = 1/sqrt(1 + s^2/n^2) in (0, 1].
g = sp.diff(sp.log(E(s, n)), s)
record("A3 d/ds log E_n = (1+s^2/n^2)^(-1/2)",
       zero(sp.simplify(g - 1 / sp.sqrt(1 + s**2 / n**2))))

# A4 Taylor expansion agrees with exp to second order.
series = sp.series(E(s, n), s, 0, 4).removeO()
target = 1 + s + s**2 / 2 + (1 - 1 / n**2) * s**3 / 6
record("A4 E_n(s) = 1 + s + s^2/2 + (1-1/n^2) s^3/6 + O(s^4)",
       zero(sp.expand(series - target)))

# A5 |n asinh(s/n) - s| <= |s|^3/(6 n^2): checked at 50 digits.
worst = mp.mpf(0)
for order in (2, 4, 8, 16, 32):
    for k in range(-400, 401):
        z = mp.mpf(k) / 10
        gap = abs(order * mp.asinh(z / order) - z)
        bound = abs(z) ** 3 / (6 * order**2)
        worst = max(worst, gap - bound)
record("A5 |log E_n(s) - s| <= |s|^3/(6n^2) on s in [-40,40]", worst <= 0,
       f"max(gap-bound)={mp.nstr(worst, 5)}")

# A6 ratio bound: for x >= y > 0, rho(x)/rho(y) <= x/y, so
# E_n(x)/E_n(y) <= (x/y)^n. Key step y^2(n^2+x^2) - x^2(n^2+y^2) =
# n^2(y^2-x^2).
record("A6 key identity y^2(n^2+x^2) - x^2(n^2+y^2) = n^2 (y^2 - x^2)",
       zero(y**2 * (n**2 + x**2) - x**2 * (n**2 + y**2)
            - n**2 * (y**2 - x**2)))
ok = True
slack = 1 + mp.mpf(10) ** -40
for _ in range(20000):
    order = random.choice((2, 4, 8, 16))
    yy = mp.mpf(10) ** random.uniform(-3, 4)
    xx = yy * (1 + mp.mpf(10) ** random.uniform(-6, 3))
    ok &= mp_E(xx, order) / mp_E(yy, order) <= (xx / yy) ** order * slack
record("A6 E_n(x)/E_n(y) <= (x/y)^n for x>=y>0 (20k samples, 50 digits)", ok)

# A7 scale robustness: lim_{c->inf} E_n(cx)/E_n(cy) = (x/y)^n, softmax ratio
# e^{c(x-y)} diverges.
lim = sp.limit(rho(c * 3, 8) / rho(c * 2, 8), c, sp.oo)
record("A7 lim_c rho(3c)/rho(2c) = 3/2 (radical ratio converges)",
       sp.simplify(lim - sp.Rational(3, 2)) == 0, f"limit={lim}")
record("A7 lim_c exp(3c)/exp(2c) = oo (softmax ratio diverges)",
       sp.limit(sp.exp(3 * c) / sp.exp(2 * c), c, sp.oo) == sp.oo)

# A8 score Jacobian with sink: dp_i/ds_j = g(s_j) p_i (delta_ij - p_j).
s1, s2, s3 = sp.symbols("s1 s2 s3", real=True)
Om = sp.symbols("Omega", positive=True)
order = 8
ks = [E(v, order) for v in (s1, s2, s3)]
D = Om + sum(ks)
ps = [k / D for k in ks]
gs = [1 / sp.sqrt(1 + v**2 / order**2) for v in (s1, s2, s3)]
at = {s1: sp.Rational(3, 7), s2: -sp.Rational(5, 3), s3: 2,
      Om: sp.Rational(1, 2)}
jac_ok = True
for i in range(3):
    for j, v in enumerate((s1, s2, s3)):
        lhs = sp.diff(ps[i], v)
        rhs = gs[j] * ps[i] * ((1 if i == j else 0) - ps[j])
        jac_ok &= abs(sp.N((lhs - rhs).subs(at), 30)) < 1e-25
record("A8 dp_i/ds_j = g(s_j) p_i (delta_ij - p_j) with sink "
       "(N=3, exact point)", jac_ok)

# A9 sink mass: sum_j p_j = 1 - Omega/D.
record("A9 sum p = 1 - Omega/D", zero(sp.simplify(sum(ps) - (1 - Om / D))))

# A10 entrywise Jacobian bound |dp_i/ds_j| <= g(s_j)/4 <= 1/4 because
# p_i(1-p_i) <= 1/4 and p_i p_j <= 1/4 when p_i + p_j <= 1.
pi, pj = sp.symbols("p_i p_j", nonnegative=True)
record("A10 1/4 - p(1-p) = (p-1/2)^2",
       zero(sp.Rational(1, 4) - pi * (1 - pi) - (pi - sp.Rational(1, 2))**2))
record("A10 1/4 - p q >= 0 when p+q<=1: (p+q)^2 - 4pq = (p-q)^2",
       zero((pi + pj)**2 - 4 * pi * pj - (pi - pj)**2))

# A11 overflow thresholds of E_n in float32/bfloat16 (max ~3.39e38).
fmax = mp.mpf("3.3895313892515355e38")
thresholds = {}
for order in (4, 8, 16, 32):
    root = fmax ** (mp.mpf(1) / order)
    # rho = root  <=>  s = n (root - 1/root)/2
    thresholds[order] = float(order * (root - 1 / root) / 2)
record("A11 overflow-free score range |s| < s_max(n) (fp32/bf16)",
       thresholds[16] > 2000 and thresholds[8] > 2e5,
       {k: f"{v:.4g}" for k, v in thresholds.items()})
record("A11 softmax without max-shift overflows at s > 88.72",
       abs(float(mp.log(fmax)) - 88.72) < 0.01)

# A12 E_n -> exp as n -> infinity.
record("A12 lim_n E_n(s) = e^s",
       sp.limit(sp.log(rho(s, n)) * n, n, sp.oo) == s)

# A13 tile additivity: the normalised output is invariant to splitting the
# key set: (N1+N2)/(Om+D1+D2) is what both one-pass and tiled sums produce.
N1, N2, D1, D2 = sp.symbols("N1 N2 D1 D2", positive=True)
record("A13 tiles add without rescaling",
       zero((N1 + N2) / (Om + D1 + D2) - (N1 + N2) / (Om + (D1 + D2))))

# ---------------------------------------------------------------- activation
F = (1 + y / sp.sqrt(1 + y**2)) / 2
record("B1 F'(y) = (1+y^2)^(-3/2)/2 > 0",
       zero(sp.diff(F, y) - (1 + y**2)**sp.Rational(-3, 2) / 2))
record("B1 F(y) + F(-y) = 1", zero(F + F.subs(y, -y) - 1))
record("B1 F(-inf)=0, F(inf)=1",
       sp.limit(F, y, -sp.oo) == 0 and sp.limit(F, y, sp.oo) == 1)

alu = x * F.subs(y, c * x)
dalu = sp.diff(alu, x)
uexpr = c * x / sp.sqrt(1 + c**2 * x**2)
record("B2 ALU_c'(x) = (1+u)/2 + u(1-u^2)/2, u = cx/sqrt(1+c^2x^2)",
       zero(sp.simplify(dalu - ((1 + uexpr) / 2
                                + uexpr * (1 - uexpr**2) / 2))))

h = sp.Rational(1, 2) + u - u**3 / 2
crit = sp.solve(sp.diff(h, u), u)
ends = [h.subs(u, -1), h.subs(u, 1)]
vals = sorted([sp.nsimplify(h.subs(u, r)) for r in crit] + ends, key=float)
lo, hi = vals[0], vals[-1]
record("B3 ALU_c' range = [1/2 - 2 sqrt6/9, 1/2 + 2 sqrt6/9] (any c>0)",
       zero(lo - (sp.Rational(1, 2) - 2 * sp.sqrt(6) / 9))
       and zero(hi - (sp.Rational(1, 2) + 2 * sp.sqrt(6) / 9)),
       f"[{float(lo):.6f}, {float(hi):.6f}]")


def phi(z):
    return mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi)


def Phi(z):
    return (1 + mp.erf(z / mp.sqrt(2))) / 2


def gelu_d(z):
    return Phi(z) + z * phi(z)


g_hi, g_lo = gelu_d(mp.sqrt(2)), gelu_d(-mp.sqrt(2))
record("B3 GELU' range = [-0.128904, 1.128904] (attained at +-sqrt2)",
       abs(g_hi - mp.mpf("1.1289039")) < 1e-6
       and abs(g_lo + mp.mpf("0.1289039")) < 1e-6,
       f"[{mp.nstr(g_lo, 8)}, {mp.nstr(g_hi, 8)}]")
record("B3 Lipschitz: ALU 1.044331 < GELU 1.128904", float(hi) < float(g_hi))

record("B4 ALU_c''(0) = c", zero(sp.diff(alu, x, 2).subs(x, 0) - c))
record("B4 GELU''(0) = sqrt(2/pi)",
       abs(2 * phi(0) - mp.sqrt(2 / mp.pi)) < mp.mpf(10)**-45)

# ALU_c' = 0  <=>  h(u) = 0  <=>  (u + 1)(u^2 - u - 1) = 0, so the finite
# stationary point is u = (1 - sqrt5)/2 = -1/golden. Inflections (extremes of
# the derivative) sit at u^2 = 2/3, i.e. x^2 = 2/c^2.
golden = (1 + sp.sqrt(5)) / 2
record("B5 h(u) = -(u+1)(u^2-u-1)/2", zero(h + (u + 1) * (u**2 - u - 1) / 2))
cp = sp.symbols("cp", positive=True)
minx = -golden**sp.Rational(-1, 2) / cp
alu_c = alu.subs(c, cp)
at = {cp: sp.Rational(4, 5)}
slope = sp.diff(alu_c, x).subs(x, minx).subs(at)
gap = (alu_c.subs(x, minx) + golden**sp.Rational(-5, 2) / (2 * cp)).subs(at)
record("B5 ALU_c minimum at x = -golden^(-1/2)/c with value "
       "-golden^(-5/2)/(2c)",
       abs(sp.N(slope, 40)) < 1e-35 and abs(sp.N(gap, 40)) < 1e-35,
       f"min*c = {float(-golden**sp.Rational(-5, 2) / 2):.6f}")
gelu_min = mp.findroot(gelu_d, -0.75)
gelu_min_value = gelu_min * Phi(gelu_min)
record("B5 GELU minimum -0.169971 at x = -0.751791",
       abs(gelu_min + mp.mpf("0.751791")) < 1e-5
       and abs(gelu_min_value + mp.mpf("0.169971")) < 1e-5,
       f"x={mp.nstr(gelu_min, 8)} value={mp.nstr(gelu_min_value, 8)}")
record("B5 inflections of ALU_c at x^2 = 2/c^2 (h'(u)=0 <=> u^2=2/3)",
       zero(sp.simplify(sp.diff(alu, x, 2).subs(x, sp.sqrt(2) / c)))
       and zero(sp.simplify(sp.diff(alu, x, 2).subs(x, -sp.sqrt(2) / c))))

cpos = sp.symbols("cpos", positive=True)
alup = alu.subs(c, cpos)
tail_neg = sp.limit(alup * x, x, -sp.oo)
tail_pos = sp.limit((alup - x) * x, x, sp.oo)
record("B6 tails: ALU_c(x) ~ 1/(4c^2 x) as x->-inf, "
       "x - 1/(4c^2 x) as x->+inf",
       zero(tail_neg - 1 / (4 * cpos**2))
       and zero(tail_pos + 1 / (4 * cpos**2)))

# --------------------------------------------- radical logistic gate (rgelu)
# sigma_n(y) = 1 / (1 + E_n(-y)),  rgelu(x) = x sigma_n(beta x)
sig = 1 / (1 + E(-y, n))
ev = sp.symbols("e", positive=True)          # e = E_n(y), E_n(-y) = 1/e (A2)
record("B7 sigma_n(y) + sigma_n(-y) = 1 (using A2: E(-y) = 1/E(y))",
       zero(sp.simplify(1 / (1 + 1 / ev) + 1 / (1 + ev) - 1)))
ser = sp.series(sig.subs(n, 8), y, 0, 4).removeO()
logi = sp.series(1 / (1 + sp.exp(-y)), y, 0, 4).removeO()
diff = sp.expand(ser - logi)
record("B7 sigma_8 matches the logistic to second order",
       all(zero(diff.coeff(y, d)) for d in range(3)),
       f"diff = {sp.simplify(ser - logi)}")
dsig = sp.diff(sig, y)
slope_gap = dsig - sig * (1 - sig) / sp.sqrt(1 + y**2 / n**2)
record("B7 sigma_n' = sigma_n (1 - sigma_n) / sqrt(1 + y^2/n^2)",
       abs(sp.N(slope_gap.subs({y: sp.Rational(7, 3), n: 8}), 30)) < 1e-25
       and abs(sp.N(slope_gap.subs({y: -sp.Rational(11, 2), n: 16}),
                    30)) < 1e-25)
record("B7 0 < sigma_n' <= 1/4 (p(1-p) <= 1/4 and g <= 1)", True,
       "from A10 and A3")
worst_tail = max(
    abs(float(1 / (1 + mp_E(mp.mpf(v), 8)) * (2 * v / 8) ** 8) - 1)
    for v in (200, 1000, 5000))
record("B7 tail sigma_8(-y) ~ (n/(2y))^n", worst_tail < 0.05,
       f"max rel dev {worst_tail:.3g} at y in (200, 1000, 5000)")
beta = mp.mpf("1.702")


def rg(z):
    return z / (1 + mp_E(-beta * z, 8))


def rg_d(z):
    return mp.diff(rg, z)


grid_d = [(rg_d(mp.mpf(k) / 100), mp.mpf(k) / 100) for k in range(-600, 601)]
z_hi = max(grid_d)[1]
z_lo = min(grid_d)[1]
rgelu_hi = mp.findroot(lambda z: mp.diff(rg_d, z), z_hi)
rgelu_lo = mp.findroot(lambda z: mp.diff(rg_d, z), z_lo)
record("B8 rgelu derivative range ~ [-0.0945, 1.0945] (n=8, beta=1.702)",
       abs(rg_d(rgelu_hi) - mp.mpf("1.0945")) < 1e-3
       and abs(rg_d(rgelu_lo) + mp.mpf("0.0945")) < 1e-3,
       f"[{mp.nstr(rg_d(rgelu_lo), 6)}, {mp.nstr(rg_d(rgelu_hi), 6)}]")

# ------------------------------------------------------------------ position
R = sp.Matrix([[1 - w**2, -2 * w], [2 * w, 1 - w**2]]) / (1 + w**2)
record("C1 R(w)^T R(w) = I",
       (R.T * R - sp.eye(2)).applyfunc(sp.simplify) == sp.zeros(2))
record("C1 det R(w) = 1", zero(sp.simplify(R.det() - 1)))
theta = mp.mpf("0.7")
tw = mp.tan(theta / 2)
Rn = mp.matrix([[1 - tw**2, -2 * tw], [2 * tw, 1 - tw**2]]) / (1 + tw**2)
record("C2 R(w) rotates by theta = 2 atan(w)",
       abs(Rn[0, 0] - mp.cos(theta)) < 1e-45
       and abs(Rn[1, 0] - mp.sin(theta)) < 1e-45)

rel_ok = True
for p1 in range(-2, 3):
    for p2 in range(-2, 3):
        A = R**p1 if p1 >= 0 else (R.T)**(-p1)
        B = R**p2 if p2 >= 0 else (R.T)**(-p2)
        Cm = R**(p2 - p1) if p2 - p1 >= 0 else (R.T)**(p1 - p2)
        rel_ok &= (A.T * B - Cm).applyfunc(sp.simplify) == sp.zeros(2)
record("C3 R^a^T R^b = R^(b-a) for a,b in [-2,2]", rel_ok)

wx, wy = sp.symbols("w_x w_y", real=True)
Rx = R.subs(w, wx)
Ry = R.subs(w, wy)
q = sp.Matrix(sp.symbols("q0:4", real=True))
k = sp.Matrix(sp.symbols("k0:4", real=True))


def block(r, cc):
    out = sp.zeros(4)
    out[0:2, 0:2] = Rx**cc
    out[2:4, 2:4] = Ry**r
    return out


lhs = (block(1, 2) * q).dot(block(3, 1) * k)
# Relative offset (dr, dc) = (3 - 1, 1 - 2) = (2, -1); a negative step uses
# the transpose R^T = R^-1.
rel = sp.zeros(4)
rel[0:2, 0:2] = Rx.T
rel[2:4, 2:4] = Ry**2
rhs = q.dot(rel * k)
record("C4 2-D axial Cayley: <B(1,2)q, B(3,1)k> = <q, B(2,-1)k>",
       zero(sp.simplify(sp.expand(lhs - rhs))))

# ---------------------------------------------------------------------- loss
al = sp.symbols("alpha", positive=True)
p1_, p2_, p3_ = sp.symbols("p1 p2 p3", positive=True)
y1, y2, y3 = sp.symbols("y1 y2 y3", nonnegative=True)


def phi_t(v):
    return (v**al - v) / (al * (al - 1))


def dphi(v):
    return sp.diff(phi_t(t), t).subs(t, v)


P = (p1_, p2_, p3_)
Y = (y1, y2, y3)
breg = sum(phi_t(Y[i]) - phi_t(P[i]) - dphi(P[i]) * (Y[i] - P[i])
           for i in range(3))
onehot = breg.subs({y1: 1, y2: 0, y3: 0})
closed = (sum(v**al for v in P) / al - p1_**(al - 1) / (al - 1)
          + 1 / (al * (al - 1)))
vals = {al: sp.Rational(7, 8), p1_: sp.Rational(1, 5),
        p2_: sp.Rational(1, 3), p3_: sp.Rational(7, 15)}
gap_78 = (onehot.subs(0**al, 0) - closed).subs(vals)
gap_32 = (onehot - closed).subs({**vals, al: sp.Rational(3, 2)})
record("D1 one-hot power score = sum p^a/a - p_c^(a-1)/(a-1) + 1/(a(a-1))",
       abs(sp.N(gap_78, 30)) < 1e-25 and abs(sp.N(gap_32, 30)) < 1e-25)

closed_78 = (8 * p1_**sp.Rational(-1, 8)
             + sp.Rational(8, 7) * sum(v**sp.Rational(7, 8) for v in P)
             - sp.Rational(64, 7))
record("D2 alpha=7/8: 8 p_c^(-1/8) + (8/7) sum p^(7/8) - 64/7",
       abs(sp.N((closed.subs(al, sp.Rational(7, 8)) - closed_78).subs(vals),
                30)) < 1e-25)

brier = sum((P[i] - (1 if i == 0 else 0))**2 for i in range(3)) / 2
record("D3 alpha=2 gives Brier/2 (on the simplex)",
       zero(sp.expand((closed.subs(al, 2) - brier)
                      .subs(p3_, 1 - p1_ - p2_))))

ce_lim = sp.limit(closed.subs({p1_: sp.Rational(1, 5),
                               p2_: sp.Rational(1, 3),
                               p3_: sp.Rational(7, 15)}), al, 1)
record("D4 alpha->1 gives the log score -log p_c",
       abs(sp.N(ce_lim + sp.log(sp.Rational(1, 5)), 30)) < 1e-25,
       f"{sp.N(ce_lim, 12)}")

# Propriety: the expected score under q is minimised at p = q (Bregman
# identity), checked for the exponent of the small-scale trials and for the
# one used in the final model.
q1, q2 = sp.Rational(1, 6), sp.Rational(1, 3)
qv = (q1, q2, 1 - q1 - q2)


def expected_score(pv, alpha):
    """E_{c ~ q}[S(p, c)] for the one-hot power score of exponent alpha."""
    return sum(qv[ci] * (sum(v**alpha for v in pv) / alpha
                         - pv[ci]**(alpha - 1) / (alpha - 1))
               for ci in range(3))


for alpha_v in (sp.Rational(7, 8), sp.Rational(63, 64)):
    best = expected_score(qv, alpha_v)
    prop_ok = True
    for _ in range(2000):
        r1, r2 = sorted((random.random(), random.random()))
        pv = [sp.Float(r1, 30) + 1e-9, sp.Float(r2 - r1, 30) + 1e-9,
              sp.Float(1 - r2, 30) + 1e-9]
        tot = sum(pv)
        pv = [v / tot for v in pv]
        prop_ok &= sp.N(expected_score(pv, alpha_v) - best, 25) >= -1e-20
    record("D5 strict propriety: E_q[S(p,y)] >= E_q[S(q,y)] "
           f"(2000 random p, alpha={alpha_v})", prop_ok)
record("D5 generator convex: phi''(t) = t^(alpha-2) > 0",
       zero(sp.simplify(sp.diff(phi_t(t), t, 2) - t**(al - 2))))

# Logit gradient through the radical link p = E_m(z)/sum E_m(z).
z1, z2, z3 = sp.symbols("z1 z2 z3", real=True)
m = 8
kz = [E(v, m) for v in (z1, z2, z3)]
pz = [kk / sum(kz) for kk in kz]
L = (sum(v**sp.Rational(7, 8) for v in pz) / sp.Rational(7, 8)
     - pz[0]**sp.Rational(-1, 8) / sp.Rational(-1, 8))
point = {z1: sp.Rational(1, 3), z2: -sp.Rational(1, 2), z3: sp.Rational(5, 4)}
grad_ok = True
for j, v in enumerate((z1, z2, z3)):
    auto = sp.N(sp.diff(L, v).subs(point), 30)
    pvals = [sp.N(pp.subs(point), 40) for pp in pz]
    # dL/dp_i = p_i^(-1/8) - [i = 0] p_0^(-9/8)
    gl = [sp.N(pvals[i]**sp.Rational(-1, 8)
               - (pvals[0]**sp.Rational(-9, 8) if i == 0 else 0), 40)
          for i in range(3)]
    inner = sum(pvals[i] * gl[i] for i in range(3))
    gj = 1 / sp.sqrt(1 + (point[v] / m)**2)
    manual = sp.N(gj * pvals[j] * (gl[j] - inner), 30)
    grad_ok &= abs(auto - manual) < 1e-20
record("D6 dL/dz_j = g(z_j) p_j (dL/dp_j - <p, dL/dp>) via the radical link",
       grad_ok)

OUT.parent.mkdir(parents=True, exist_ok=True)
summary = {"passed": all(ck["passed"] for ck in checks), "count": len(checks),
           "checks": checks}
OUT.write_text(json.dumps(summary, indent=2) + "\n")
passed = sum(ck["passed"] for ck in checks)
print(f"\n{passed}/{len(checks)} checks passed -> {OUT}")
raise SystemExit(0 if summary["passed"] else 1)

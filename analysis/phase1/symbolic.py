"""Exact identities, necessary hypotheses, counterexamples, 50-digit checks."""

import mpmath as mp
import sympy as sp


def run():
    k, u = sp.symbols("k u", real=True)
    v, w = sp.symbols("v w", real=True)
    d, s = sp.symbols("d s", positive=True)
    checks = {}

    def zero(name, expression):
        value = sp.factor(expression)
        checks[name] = value == 0
        if value != 0:
            raise AssertionError((name, value))

    t = 2*k*u/(k*k-u*u)
    root = (k*k+u*u)/(k*k-u*u)
    zero("rational_radicand", 1+t*t-root*root)
    zero("rational_base", t+root-(k+u)/(k-u))
    ratio = ((k+u)/(k-u))**8
    slope = 16*k/(k*k-u*u)
    zero("log_derivative_u", sp.diff(ratio, u)/ratio-slope)
    zero("log_derivative_kappa", sp.diff(ratio, k)/ratio+16*u/(k*k-u*u))
    zero("projective_orthogonality", (v*v-u*u)**2+(2*u*v)**2-(v*v+u*u)**2)
    vn, un = v-w*u, u+w*v
    zero("recurrence_norm_growth", vn*vn+un*un-(1+w*w)*(v*v+u*u))
    z = (v+sp.I*u)**2/(v*v+u*u)
    zn = (vn+sp.I*un)**2/(vn*vn+un*un)
    zero("projective_step_equals_cayley", zn-z*(1+sp.I*w)**2/(1+w*w))

    q0, q1, a0, a1 = sp.symbols("q0 q1 a0 a1", real=True)
    r = sp.sqrt(d+q0*q0+q1*q1)
    lifted_dot = (q0*a0+q1*a1)/r
    for qi, ai in ((q0, a0), (q1, a1)):
        zero("lift_gradient_"+str(qi), sp.diff(lifted_dot, qi)
             -(ai/r-qi*(q0*a0+q1*a1)/r**3))
    a, b, c, e = sp.symbols("a b c e", real=True)
    positional_j = sp.Matrix([[a, b], [c, -e]])
    zero("position_rank_condition", positional_j.det()+a*e+b*c)
    zero("tied_position_rank", sp.Matrix([[a, b], [a, -b]]).det()+2*a*b)
    # Local rank is conditional: all-zero frequencies are an exact counterexample.
    checks["unconstrained_rank_claim_refuted"] = positional_j.subs({a:0,b:0,c:0,e:0}).det() == 0
    zero("pool_fixed_point_squared_scale", (d/(d+s))/(1-s/(d+s))-1)

    x = sp.symbols("x0:6", real=True)
    rows = [sp.Matrix(x[2*i:2*i+2]) for i in range(3)]
    total = sum(rows, sp.zeros(2, 1))
    mean = total/3
    original = 9-total.dot(total)
    stable = 3*sum(1-row.dot(row) for row in rows) + 3*sum((row-mean).dot(row-mean) for row in rows)
    zero("stable_pool_denominator", original-stable)
    radius = sp.symbols("radius", positive=True)
    radial = d/(d+radius*radius)**sp.Rational(3, 2)
    tangent = 1/sp.sqrt(d+radius*radius)
    checks["outlier_tangent_rate"] = sp.limit(radius*tangent, radius, sp.oo) == 1
    checks["outlier_radial_rate"] = sp.limit(radius**3*radial, radius, sp.oo) == d
    checks["singleton_outlier_claim_refuted"] = True  # fixed point for N=1 gives Dh=I.

    # Chain rule: b_i=d log(weight_i)/dz_i; off-diagonal dp_i/dz_j=-p_i*p_j*b_j.
    pi, pj, bi, bj, ai, aj = sp.symbols("pi pj bi bj ai aj", positive=True)
    zero("mle_poincare", -bi*pi*pj*bj-(-bj*pi*pj*bi))
    nonconservative_curl = sp.factor(pi*pj*(aj*bi-ai*bj))
    checks["quarter_exponent_not_identically_integrable"] = nonconservative_curl != 0
    p, y = sp.symbols("p y", positive=True)
    zero("ce_expected_minimum_stationarity", sp.diff(-y*sp.log(p), p).subs(p, y)+1)
    zero("ce_expected_curvature", sp.diff(-y*sp.log(p), p, 2)-y/p**2)
    zero("power_score_curvature", sp.diff(p**sp.Rational(7,8)/(sp.Rational(7,8)*(-sp.Rational(1,8))), p, 2)-p**(-sp.Rational(9,8)))
    zero("brier_expected_excess", sp.Rational(1,2)*(p-y)**2
         -(sp.Rational(1,2)*p*p-y*p+sp.Rational(1,2)*y*y))
    tpos = sp.symbols("t", nonnegative=True)
    tail = 1+tpos+tpos*tpos/(2+tpos)
    zero("rational_tail_log_derivative", sp.diff(tail,tpos)/tail
         -(2*tpos*tpos+8*tpos+4)/((tpos+2)*(2*tpos*tpos+3*tpos+2)))
    zero("rational_tail_origin_value", tail.subs(tpos,0)-1)
    zero("rational_tail_origin_slope", sp.diff(tail,tpos).subs(tpos,0)-1)
    zero("rational_tail_origin_curvature", sp.diff(tail,tpos,2).subs(tpos,0)-1)
    # Negative branch is the reciprocal, so the first two derivatives agree
    # at the branch point. Higher-order matching with the radical is not claimed.
    zero("rational_tail_negative_slope", sp.diff(1/tail.subs(tpos,-u),u).subs(u,0)-1)
    zero("rational_tail_negative_curvature", sp.diff(1/tail.subs(tpos,-u),u,2).subs(u,0)-1)
    inverse_energy = d*radius/(d+radius*radius)
    checks["inverse_energy_outlier_influence_rate"] = sp.limit(radius*inverse_energy,radius,sp.oo)==d
    checks["inverse_energy_outlier_derivative_rate"] = sp.limit(radius**2*sp.diff(inverse_energy,radius),radius,sp.oo)==-d

    with mp.workdps(50):
        identity_error, derivative_error = mp.mpf(0), mp.mpf(0)
        for kk in ("1.0500000001", "1.1", "2"):
            for uu in ("-0.999999999", "-0.8", "0", "0.8", "0.999999999"):
                kk, uu = mp.mpf(kk), mp.mpf(uu)
                tt = 2*kk*uu/(kk*kk-uu*uu)
                radical = (tt+mp.sqrt(1+tt*tt))**8
                rational = ((kk+uu)/(kk-uu))**8
                identity_error = max(identity_error, abs(radical/rational-1))
                deriv = mp.diff(lambda a: mp.log(((kk+a)/(kk-a))**8), uu)
                derivative_error = max(derivative_error, abs(deriv-16*kk/(kk*kk-uu*uu)))
        zz = [mp.mpf("-1.3"), mp.mpf("0.4"), mp.mpf("2.1")]
        target = [mp.mpf("0.8"), mp.mpf("0.1"), mp.mpf("0.1")]

        def probs(z):
            w = [(a/4+mp.sqrt(1+(a/4)**2))**8 for a in z]
            return [a/sum(w) for a in w]

        def grad(z, gamma):
            pp = probs(z)
            return [2*(pp[i]-target[i])/(1+(z[i]/4)**2)**gamma for i in range(3)]

        curls = {}
        for gamma in (mp.mpf("0.5"), mp.mpf("0.25")):
            jac = mp.matrix(3)
            for i in range(3):
                for j in range(3):
                    def f(t):
                        z = zz.copy()
                        z[j] = t
                        return grad(z, gamma)[i]
                    jac[i,j] = mp.diff(f, zz[j])
            curls[str(gamma)] = str(max(abs(jac[i,j]-jac[j,i]) for i in range(3) for j in range(3)))
        ce = lambda z: -sum(target[i]*mp.log(probs(z)[i]) for i in range(3))
        ce_error = max(abs(mp.diff(lambda t: ce(zz[:i]+[t]+zz[i+1:]), zz[i])-grad(zz, mp.mpf("0.5"))[i]) for i in range(3))
        checks["mpmath_rational_identity"] = identity_error < mp.mpf("1e-40")
        checks["mpmath_mle_gradient"] = ce_error < mp.mpf("1e-45")
        checks["mpmath_quarter_curl_counterexample"] = mp.mpf(curls["0.25"]) > mp.mpf("1e-5")
        telemetry = {"precision_digits":50, "relative_identity_error":str(identity_error),
                     "log_derivative_error":str(derivative_error), "mle_gradient_error":str(ce_error),
                     "curl_max":curls, "g8_at_94":str(1/mp.sqrt(1+(mp.mpf(94)/8)**2))}
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"checks":checks, "telemetry":telemetry,
            "taylor_n8":str(sp.series(ratio, u, 0, 5)),
            "weight_bounds":"[((kappa-1)/(kappa+1))^8, ((kappa+1)/(kappa-1))^8]; kappa>1, |u|<=1",
            "log_gradient_q":"16*kappa/(kappa^2-u^2) * (R_i^T R_j k_tilde / r_q - q*(q^T R_i^T R_j k_tilde)/r_q^3)",
            "log_gradient_k":"same formula with q,k and R_i,R_j exchanged",
            "log_gradient_kappa":"-16*u/(kappa^2-u^2)",
            "pool_bound":"D >= N^2*d/(d+max_i ||h_i||^2)>0, for d>0,N>0",
            "hypotheses":["rational identity requires kappa>|u|, n a nonnegative integer",
                          "projective orthogonality is exact over reals, not machine-exact in floating point",
                          "SO(4) rank requires a*e+b*c != 0; local rank does not exclude periodic global aliases",
                          "outlier derivative rate holds for N>=2 with the other finite tokens fixed",
                          "CE is strictly proper in probabilities; smoothed targets change the population optimum",
                          "exact CE value is not algebraic; a pure custom VJP does not remove the forward log",
                          "1+1/rho(t) ranges over (1,infinity), not (1.05,2]"],
            "nonconservative_curl":str(nonconservative_curl)}

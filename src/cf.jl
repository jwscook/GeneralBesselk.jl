# Steed's method for Temme's second continued fraction (CF2), as in Numerical Recipes'
# bessik, generalised to complex order and argument. Gives the *scaled* pair
#   (e^z K_μ(z), e^z K_{μ+1}(z)),  |z| not small, |arg z| <= π/2,
# and an error estimate in units of eps. It is used both with |Re μ| <= 1/2 (followed by
# recurrence) and directly at the target order; the internal sequences are rescaled so
# large orders do not overflow. If it does not converge the estimate is Inf.
const STEED_MAXITERS = 400   # converged cases need ≲150; failures fall through to other methods
function _steed(μ, z, tol)
  b = 2 * (1 + z)
  d = inv(b)
  h = delh = d
  q1 = zero(d)
  q2 = one(d)
  a1 = 1 / 4 - μ * μ
  q = c = a1
  a = -a1
  s = 1 + q * delh
  sa = _absv(s)
  conv = false
  for i in 2:STEED_MAXITERS
    a -= 2 * (i - 1)
    c = -a * c / i
    qnew = (q1 - b * q2) / a
    q1 = q2
    q2 = qnew
    q += c * qnew
    b += 2
    d = inv(b + a * d)
    delh = (b * d - 1) * delh
    h += delh
    dels = q * delh
    s += dels
    sa += _absv(dels)
    # keep c and the q recursion in range: c * q_k is invariant under c/λ, q_k*λ
    ac = _absv(c)
    if ac > 1e100 || (0 < ac < 1e-100)
      λ = ac
      c /= λ
      q1 *= λ
      q2 *= λ
    end
    if _conv(dels, s, tol)
      conv = true
      break
    end
  end
  h = a1 * h
  kμ = sqrt(π / (2z)) / s
  t = μ + z + 1 / 2 - h
  kμ1 = kμ * t / z
  est = sa / _absv(s) + (abs(_val(μ)) + abs(z) + 1 + _absv(h)) / _absv(t)
  (conv && isfinite(est) && isfinite(_absv(kμ)) && isfinite(_absv(kμ1))) || (est = Inf)
  return kμ, kμ1, est
end

# I_{ν+1}(w) / I_ν(w) from the continued fraction 1/(b1 + 1/(b2 + ...)), b_k = 2(ν+k)/w,
# by the modified Lentz method. Converges for all w ≠ 0 once k ≳ |w|.
function _cf1_ratio(ν, w, tol)
  tiny = 1e-300
  iw2 = 2 / w
  f = C = (ν + 1) * iw2
  D = zero(f)
  for k in 2:MAXITERS
    b = (ν + k) * iw2
    D = b + D
    iszero(_val(D)) && (D = tiny + zero(D))
    D = inv(D)
    C = b + inv(C)
    iszero(_val(C)) && (C = tiny + zero(C))
    Δ = C * D
    f *= Δ
    _mag(Δ - 1) <= tol && break
  end
  return inv(f)
end

# Forward recurrence K_{μ+j+1} = K_{μ+j-1} + 2(μ+j)/z K_{μ+j}:
# (K_μ, K_{μ+1}) -> (K_{μ+n}, K_{μ+n+1}); works equally on e^z-scaled values.
# K is usually the dominant solution in this direction, but not always for complex
# orders/arguments (e.g. large |Im ν| near arg z = ±π/2), so a perturbation is carried
# along to measure the error amplification, returned as the third value.
function _recur(μ, z, k0, k1, n)
  iz2 = 2 / z
  e0 = zero(_val(k0))
  e1 = _val(k1)
  r0 = abs(e1) / (_absv(k0) + _absv(k1))     # initial relative size of the perturbation
  scale = 1.0
  ex = 0                                      # values are k * 2^ex
  for j in 1:n
    f = (μ + j) * iz2
    k0, k1 = k1, muladd(f, k1, k0)
    fv = _val(f)
    e0, e1 = e1, muladd(fv, e1, e0)
    if _absv(k1) > 0x1p600                    # keep K in range (exact power-of-two scaling)
      k0 = _ldexp(k0, -600); k1 = _ldexp(k1, -600); ex += 600
    end
    ae = abs(e1)
    if ae > 1e100                             # renormalise the perturbation
      e0 /= ae; e1 /= ae; scale *= ae
    end
  end
  amp = (abs(e0) + abs(e1)) * scale / (_absv(k0) + _absv(k1)) / r0
  return k0, k1, isnan(amp) ? Inf : amp, ex
end

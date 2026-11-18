# Direct series at the full order (DLMF 10.27.4):
#   K_ν(z) = π / (2 sin νπ) (I_{-ν}(z) - I_ν(z)),
#   I_ν(z) = (z/2)^ν / Γ(ν+1) 0F1(; ν+1; z²/4).
# Exact for non-integer ν on the principal branch. It cancels catastrophically near integer
# orders and for large |z| (handled by the other methods), but it is the accurate route
# for small |z| when K_ν is recessive under forward recurrence (large |Im ν|, z off the
# positive real axis). Returns the *unscaled* (K_ν, K_{ν+1}) as mantissas, an error
# estimate and a common power-of-two exponent.

# Σ_k m^k / (k! (b)_k) and Σ_k |term_k|
function _0f1_abs(b, m, tol)
  t = one(b * m)
  s = t
  sa = 1.0
  kmin = max(0.0, -real(_val(b))) + 2        # terms can jump up near k ≈ -Re b
  for k in 1:MAXITERS
    t *= m / (k * (b + (k - 1)))
    s += t
    sa += _absv(t)
    k > kmin && _conv(t, s, tol) && break
  end
  return s, sa
end

function _kseries1(ν, L, m, lsin, tol)
  # A = π/(2 sin νπ) I_{-ν},  B = π/(2 sin νπ) I_ν
  la = log(π / 2) - lsin - ν * L - _loggamma(1 - ν)
  lb = log(π / 2) - lsin + ν * L - _loggamma(1 + ν)
  # split a large common magnitude off as 2^ex so exp neither overflows nor underflows
  big = max(real(_val(la)), real(_val(lb)))
  ex = abs(big) > 500 ? round(Int, big / log(2.0)) : 0
  sh = ex * log(2.0)
  Sa, sa = _0f1_abs(1 - ν, m, tol)
  Sb, sb = _0f1_abs(1 + ν, m, tol)
  A = exp(la - sh) * Sa
  B = exp(lb - sh) * Sb
  k = A - B
  # rounding in the series, in exp of a large exponent, and cancellation in A - B
  ea = sa / _absv(Sa) + _absv(la)
  eb = sb / _absv(Sb) + _absv(lb)
  est = (_absv(A) * ea + _absv(B) * eb) / _absv(k)
  return k, isnan(est) ? Inf : est, ex
end

function _kseries(ν, z, tol, pair::Bool)
  L = log(z / 2)
  m = (z / 2)^2
  k0, e0, x0 = _kseries1(ν, L, m, _logsinpi(ν), tol)
  pair || return k0, k0, e0, x0
  k1, e1, x1 = _kseries1(ν + 1, L, m, _logsinpi(ν + 1), tol)
  x = max(x0, x1)
  return _ldexp(k0, x0 - x), _ldexp(k1, x1 - x), max(e0, e1), x
end

# Direct series at the full order (DLMF 10.27.4):
#   K_ν(z) = π / (2 sin νπ) (I_{-ν}(z) - I_ν(z)),
#   I_ν(z) = (z/2)^ν / Γ(ν+1) 0F1(; ν+1; z²/4).
# Exact for non-integer ν on the principal branch. It cancels catastrophically near integer
# orders and for large |z| (handled by the other methods), but it is the accurate route
# for small |z| when K_ν is recessive under forward recurrence (large |Im ν|, z off the
# positive real axis). Returns the *unscaled* (K_ν, K_{ν+1}) and an error estimate.

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
  Sa, sa = _0f1_abs(1 - ν, m, tol)
  Sb, sb = _0f1_abs(1 + ν, m, tol)
  A = exp(la) * Sa
  B = exp(lb) * Sb
  k = A - B
  # rounding in the series, in exp of a large exponent, and cancellation in A - B
  ea = sa / _absv(Sa) + _absv(la)
  eb = sb / _absv(Sb) + _absv(lb)
  est = (_absv(A) * ea + _absv(B) * eb) / _absv(k)
  return k, isnan(est) ? Inf : est
end

function _kseries(ν, z, tol, pair::Bool)
  L = log(z / 2)
  m = (z / 2)^2
  k0, e0 = _kseries1(ν, L, m, _logsinpi(ν), tol)
  pair || return k0, k0, e0
  k1, e1 = _kseries1(ν + 1, L, m, _logsinpi(ν + 1), tol)
  return k0, k1, max(e0, e1)
end

# The direct series in BigFloat. The precision starts from the cancellation estimated in
# double precision (est, in units of eps) and is raised until the estimate made at the
# working precision (which, unlike the double one, is not saturated by rounding) shows
# the result is good to ~2^-60. Returns scaled values in double.
function _kxseries_big(ν, z, pair::Bool, est)
  bits = 53 + ceil(Int, log2(max(est, 1.0))) + 40
  local res
  for _ in 1:8
    res, ok, ebig = setprecision(BigFloat, bits) do
      νb, zb = _tobig(ν), _tobig(z)
      k0, k1, e = _kseries(νb, zb, eps(Float64) / 4, pair)
      ez = exp(zb)
      (_to64(k0 * ez), _to64(k1 * ez), 1.0), e * eps(BigFloat) < 2.0^-60, e
    end
    ok && return res
    bits = max(2bits, 60 + ceil(Int, log2(max(ebig, 1.0))) + 40)
  end
  return (res[1], res[2], Inf)
end

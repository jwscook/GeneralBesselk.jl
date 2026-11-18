const TOL = eps(Float64)
# error estimates are in units of eps; anything below this is accepted without trying
# other methods
const GOOD_EST = 100.0
# above this estimate (expected error ≳ 1e-13) the BigFloat fallback is used
const BIG_EST = 500.0
# the direct series is only a candidate below this |z| (its terms grow like e^{|z|})
const SERIES_MAX_ABSZ = 200.0

_pick(a, b) = a === nothing ? b : b === nothing ? a : (a[3] <= b[3] ? a : b)

# Scaled pair (e^z K_ν(z), e^z K_{ν+1}(z), error estimate) for complex (possibly Dual) ν
# with Re ν >= 0 and plain complex z with Re z >= 0. With pair = false the second entry is
# not meaningful.
function _kx_right(ν, z::Complex, pair::Bool)
  az = abs(z)
  if az >= ASYM_MIN_ABSZ
    k0 = _hankel_kx(ν, z, TOL)
    if k0 !== nothing
      pair || return k0, k0, 1.0
      k1 = _hankel_kx(ν + 1, z, TOL)
      k1 !== nothing && return k0, k1, 1.0
    end
  end
  n = round(Int, real(_val(ν)))
  μ = ν - n                                     # |Re μ| <= 1/2
  best = nothing
  if az <= TEMME_MAX_ABSZ
    k0, k1, e = _temme(μ, z, TOL)
    k0, k1, amp = _recur(μ, z, k0, k1, n)
    ez = exp(z)
    best = (k0 * ez, k1 * ez, e * amp)
  else
    r = _steed(μ, z, TOL)
    if r !== nothing
      k0, k1, amp = _recur(μ, z, r[1], r[2], n)
      best = (k0, k1, r[3] * amp)
    end
  end
  best !== nothing && best[3] <= GOOD_EST && return best
  # forward recurrence was unstable (or CF2 failed): direct methods at the full order
  if az > TEMME_MAX_ABSZ && n > 0
    best = _pick(best, _steed(ν, z, TOL))
    best !== nothing && best[3] <= GOOD_EST && return best
  end
  if az < SERIES_MAX_ABSZ
    k0, k1, e = _kseries(ν, z, TOL, pair)
    ez = exp(z)
    best = _pick(best, (k0 * ez, k1 * ez, e))
  end
  return best
end

# Scaled pair and error estimate for any z off the cut (Re ν >= 0). Falls back to the
# direct series in BigFloat when no double precision method is accurate.
function _kx_pair(ν, z::Complex, pair::Bool=true)
  best = _kx_pair64(ν, z, pair)
  best[3] <= BIG_EST && return best
  abs(z) < SERIES_MAX_ABSZ || return best
  _, _, es = _kseries(ν, z, TOL, pair)
  isfinite(es) || return best
  return _kxseries_big(ν, z, pair, es)
end

function _kx_pair64(ν, z::Complex, pair::Bool)
  real(z) >= 0 && return _kx_right(ν, z, pair)
  # DLMF 10.34.2 with z = w e^{iσπ}, Re w > 0:
  #   K_ν(z) = e^{-iσνπ} K_ν(w) - iσπ I_ν(w),
  # and I_ν(w) = 1 / (w (K_{ν+1}(w) + r K_ν(w))), r = I_{ν+1}(w)/I_ν(w) (Wronskian + CF1),
  # all stable for Re w > 0. In e^z-scaled form (e^z = e^{-w}) nothing overflows.
  w = -z
  σ = signbit(imag(z)) ? -1 : 1
  kw0, kw1, ew = _kx_right(ν, w, true)
  r = _cf1_ratio(ν, w, TOL)
  den = kw1 + r * kw0
  iI = -im * σ * π / (w * den)                  # -iσπ e^{-w} I_ν(w)
  ph = exp(-2w - im * σ * π * ν)                # e^{-2w} e^{-iσνπ}
  t0 = ph * kw0
  t1 = -ph * kw1
  k0 = t0 + iI
  k1 = t1 + r * iI
  eI = ew * (_absv(kw1) + _absv(r * kw0)) / _absv(den) + 4
  e0 = (_absv(t0) * ew + _absv(iI) * eI) / _absv(k0)
  e1 = (_absv(t1) * ew + _absv(r * iI) * eI) / _absv(k1)
  est = pair ? max(e0, e1) : e0
  # if the I_ν(w) term is itself unreliable its computed size says nothing: give up on it
  eI * TOL > 1e-2 && (est = Inf)
  best = (k0, k1, isnan(est) ? Inf : est)
  if best[3] > GOOD_EST && abs(z) < SERIES_MAX_ABSZ
    s0, s1, es = _kseries(ν, z, TOL, pair)
    ez = exp(z)
    best = _pick(best, (s0 * ez, s1 * ez, es))
  end
  return best
end

# K_ν(z) (scaled = false) or e^z K_ν(z) (scaled = true) for complex or Dual ν (any z).
function _besselk_general(ν, z, scaled::Bool)
  νc = _cplx(ν)
  real(_val(νc)) < 0 && (νc = -νc)               # K_{-ν} = K_ν
  if z isa Dual
    z0 = _cplx(realpart(z))
    iszero(z0) && return Dual(_kinf(νc), _kinf(νc))
    k0, k1, _ = _kx_pair(νc, z0, true)
    # d/dz K_ν = (ν/z) K_ν - K_{ν+1};  d/dz (e^z K_ν) = e^z K_ν + e^z (ν/z) K_ν - e^z K_{ν+1}
    dkx = _val(νc / z0 * k0 - k1)
    ez = scaled ? one(z0) : exp(-z0)
    dk = (scaled ? dkx + _val(k0) : dkx) * ez
    k = k0 * ez
    return k isa Dual ? Dual(realpart(k), dualpart(k) + dk * dualpart(z)) :
      Dual(k, dk * dualpart(z))
  end
  zc = _cplx(z)
  iszero(zc) && return _kinf(νc)
  k0, _, _ = _kx_pair(νc, zc, false)
  return scaled ? k0 : k0 * exp(-zc)
end
_kinf(ν) = complex(Inf, 0.0) + zero(ν)
_kinf(ν::Dual) = Dual(complex(Inf, 0.0), complex(0.0, 0.0))

# results for real ν and real z >= 0 are real
_isrealval(x::Real) = true
_isrealval(x::Complex) = iszero(imag(x))
_isrealval(x::Dual) = _isrealval(realpart(x)) && _isrealval(dualpart(x))
_realify(x::Complex) = real(x)
_realify(x::Dual) = Dual(real(realpart(x)), real(dualpart(x)))
_realify(x::Real) = x

function _kreal_or_complex(ν, z, scaled)
  r = _besselk_general(ν, z, scaled)
  if ν isa Dual && _isrealval(ν) && _isrealval(z) && real(_val(z)) >= 0
    return _realify(r)
  end
  return r
end

"""
    besselk(ν, z)

Modified Bessel function of the second kind, K_ν(z), on the principal branch
(|arg z| < π; for z on the negative real axis the sign of `imag(z)`, including the sign
of a zero, selects the side of the cut).

In addition to the methods in SpecialFunctions this supports
* complex orders `ν`,
* `DualNumbers.Dual` orders (∂K/∂ν propagated through the algorithm), and
* `DualNumbers.Dual` arguments (∂K/∂z = (ν/z)K_ν - K_{ν+1}), in any combination.
"""
besselk(ν::Complex, z::Number) = _besselk_complex_order(ν, z, false)
besselk(ν::Dual, z::Number) = _kreal_or_complex(ν, z, false)
besselk(ν::Real, z::Dual) = _besselk_real_order_dual(ν, z, false)

"""
    besselkx(ν, z)

Scaled modified Bessel function of the second kind, e^z K_ν(z). Accepts the same
argument types as [`besselk`](@ref).
"""
besselkx(ν::Complex, z::Number) = _besselk_complex_order(ν, z, true)
besselkx(ν::Dual, z::Number) = _kreal_or_complex(ν, z, true)
besselkx(ν::Real, z::Dual) = _besselk_real_order_dual(ν, z, true)

function _besselk_complex_order(ν::Complex, z, scaled)
  if iszero(imag(ν))       # real order: SpecialFunctions (AMOS) is accurate everywhere
    νr = real(ν)
    z isa Dual && return _cplx(_besselk_real_order_dual(νr, z, scaled))
    zz = z isa Real && z < 0 ? complex(z) : z
    return complex(scaled ? besselkx(νr, zz) : besselk(νr, zz))
  end
  return _besselk_general(ν, z, scaled)
end

# real order, Dual argument: values from SpecialFunctions, derivative from
#   K_ν' = -(K_{ν-1} + K_{ν+1})/2,   (e^z K_ν)' = e^z K_ν - e^z (K_{ν-1} + K_{ν+1})/2
function _besselk_real_order_dual(ν::Real, z::Dual, scaled)
  z0 = realpart(z)
  z0 isa Real && z0 < 0 && (z0 = complex(z0))
  f = scaled ? besselkx : besselk
  k = f(ν, z0)
  dk = -(f(ν - 1, z0) + f(ν + 1, z0)) / 2
  scaled && (dk += k)
  return Dual(k, dk * dualpart(z))
end

"""
    besselk_v(ν::AbstractVector, z)

`[besselk(νi, z) for νi in ν]`, reusing the result when consecutive orders are equal or
opposite (K_{-ν} = K_ν).
"""
besselk_v(ν::AbstractVector, z::Number) = _bessel_v(besselk, ν, z)

"""
    besselkx_v(ν::AbstractVector, z)

`[besselkx(νi, z) for νi in ν]`, reusing results as in [`besselk_v`](@ref).
"""
besselkx_v(ν::AbstractVector, z::Number) = _bessel_v(besselkx, ν, z)

function _bessel_v(f, ν, z)
  isempty(ν) && return Vector{typeof(f(one(eltype(ν)), z))}()
  first_ = f(first(ν), z)
  out = similar(ν, typeof(first_))
  i0 = firstindex(ν)
  out[i0] = first_
  for i in i0+1:lastindex(ν)
    out[i] = (ν[i] == ν[i-1] || ν[i] == -ν[i-1]) ? out[i-1] : f(ν[i], z)
  end
  return out
end

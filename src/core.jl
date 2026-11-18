const TOL = eps(Float64)
# error estimates are in units of eps; anything below this is accepted without trying
# other methods
const GOOD_EST = 100.0
# the direct series is only a candidate below this |z| (its terms grow like e^{|z|})
const SERIES_MAX_ABSZ = 200.0

# Internally values are carried as (mantissa, exponent) pairs, value = mantissa * 2^e, so
# that intermediate results beyond the Float64 range (e.g. K_{ν+1} when only K_ν is
# needed) neither overflow nor turn into NaN. A candidate result is the tuple
#   (k0, k1, est, e):  e^z K_ν = k0 * 2^e,  e^z K_{ν+1} = k1 * 2^e,
# with est the estimated relative error in units of eps.

_pick(a, b) = a[3] <= b[3] ? a : b

# Scaled pair for complex (possibly Dual) ν with Re ν >= 0 and complex z with Re z >= 0.
# With pair = false the second entry is not meaningful.
function _kx_right(ν, z::ComplexF64, pair::Bool)
  az = abs(z)
  if az >= ASYM_MIN_ABSZ
    k0, ok0 = _hankel_kx(ν, z, TOL)
    if ok0
      pair || return (k0, k0, 1.0, 0)
      k1, ok1 = _hankel_kx(ν + 1, z, TOL)
      ok1 && return (k0, k1, 1.0, 0)
    end
  end
  n = round(Int, real(_val(ν)))
  μ = ν - n                                     # |Re μ| <= 1/2
  if az <= TEMME_MAX_ABSZ
    t0, t1, et = _temme(μ, z, TOL)
    k0, k1, amp, e = _recur(μ, z, t0, t1, n)
    ez = exp(z)
    best = (k0 * ez, k1 * ez, et * amp, e)
  else
    s0, s1, es = _steed(μ, z, TOL)
    k0, k1, amp, e = _recur(μ, z, s0, s1, n)
    best = (k0, k1, es * amp, e)
  end
  best[3] <= GOOD_EST && return best
  # forward recurrence was unstable (or CF2 failed): direct methods at the full order
  if az > TEMME_MAX_ABSZ && n > 0
    s0, s1, es = _steed(ν, z, TOL)
    best = _pick(best, (s0, s1, es, 0))
    best[3] <= GOOD_EST && return best
  end
  if az < SERIES_MAX_ABSZ
    best = _pick(best, _kxseries(ν, z, TOL, pair))
  end
  return best
end

# the direct series, scaled by e^z
function _kxseries(ν, z::ComplexF64, tol, pair::Bool)
  k0, k1, est, e = _kseries(ν, z, tol, pair)
  ez = exp(z)
  return (k0 * ez, k1 * ez, est, e)
end

# Scaled pair for any z off the cut (Re ν >= 0).
function _kx_pair(ν, z::ComplexF64, pair::Bool)
  real(z) >= 0 && return _kx_right(ν, z, pair)
  # DLMF 10.34.2 with z = w e^{iσπ}, Re w > 0:
  #   K_ν(z) = e^{-iσνπ} K_ν(w) - iσπ I_ν(w),
  # and I_ν(w) = 1 / (w (K_{ν+1}(w) + r K_ν(w))), r = I_{ν+1}(w)/I_ν(w) (Wronskian + CF1),
  # all stable for Re w > 0. In e^z-scaled form (e^z = e^{-w}):
  #   e^z K_ν(z) = e^{-2w} e^{-iσνπ} e^w K_ν(w) - iσπ / (w (e^w K_{ν+1}(w) + r e^w K_ν(w))).
  # With e^w K(w) = kw * 2^p the first term scales as 2^p and the second as 2^-p.
  w = -z
  σ = signbit(imag(z)) ? -1 : 1
  kw0, kw1, ew, p = _kx_right(ν, w, true)
  r = _cf1_ratio(ν, w, TOL)
  den = kw1 + r * kw0
  iI = _ldexp(-im * σ * π / (w * den), -2p)    # -iσπ e^{-w} I_ν(w), in units of 2^p
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
  (eI * TOL > 1e-2 || isnan(est)) && (est = Inf)
  best = (k0, k1, est, p)
  if est > GOOD_EST && abs(z) < SERIES_MAX_ABSZ
    best = _pick(best, _kxseries(ν, z, TOL, pair))
  end
  return best
end

# k * 2^e as a value, after multiplying by e^{-z} if unscaled. Done on a normalised
# mantissa with the large part of e^{-z} split off as a power of two, so the result
# overflows/underflows cleanly (±Inf, 0) rather than giving Inf - Inf = NaN.
function _assemble(k, e::Int, z::ComplexF64, scaled::Bool)
  k, e = _normalise(k, e)
  scaled && return _ldexp(k, e)
  j = abs(real(z)) > 500 ? round(Int, -real(z) / log(2.0)) : 0
  return _ldexp(k * exp(-z - j * log(2.0)), e + j)
end

# K_ν(z) (scaled = false) or e^z K_ν(z) (scaled = true) for complex or Dual ν, plain z.
function _besselk_general(ν, z::Number, scaled::Bool)
  νc = _cplx(ν)
  real(_val(νc)) < 0 && (νc = -νc)               # K_{-ν} = K_ν
  zc = ComplexF64(z)
  iszero(zc) && return _kinf(νc)
  k0, _, _, e = _kx_pair(νc, zc, false)
  return _assemble(k0, e, zc, scaled)
end

# Dual argument. Derivative from the symmetric form
#   dK_ν/dz = -(K_{ν-1} + K_{ν+1})/2,   d(e^z K_ν)/dz = e^z K_ν - e^z (K_{ν-1} + K_{ν+1})/2,
# with K_{ν-1} computed independently: no cancellation for real ν and z > 0, and better
# conditioned than (ν/z) K_ν - K_{ν+1} for complex arguments.
function _besselk_general(ν, z::Dual, scaled::Bool)
  νc = _cplx(ν)
  real(_val(νc)) < 0 && (νc = -νc)
  z0 = ComplexF64(realpart(z))
  iszero(z0) && return Dual(_kinf(νc), _kinf(νc)) + zero(νc)
  k0, k1, ep, e = _kx_pair(νc, z0, true)
  νm = _val(νc) - 1                              # only the value of ν matters here
  real(νm) < 0 && (νm = -νm)                     # K_{ν-1} = K_{1-ν}
  km, _, emest, em = _kx_pair(νm, z0, false)
  E = max(e, em)
  a = _ldexp(km, em - E)                         # e^z K_{ν-1} in units of 2^E
  b = _ldexp(_val(k1), e - E)                    # e^z K_{ν+1}
  d = -(a + b) / 2                               # e^z dK_ν/dz in units of 2^E
  # K_{ν-1} can land in a harder regime than K_ν, K_{ν+1}; if the estimates say so, use
  # the recurrence form (ν/z) K_ν - K_{ν+1}, which needs only the pair
  c = _ldexp(_val(νc) / z0 * _val(k0), e - E)
  err_sym = (abs(a) * emest + abs(b) * ep) / abs(a + b)
  err_rec = (abs(c) + abs(b)) * ep / abs(c - b)
  err_rec < err_sym && (d = c - b)
  scaled && (d += _ldexp(_val(k0), e - E))
  k = _assemble(k0, e, z0, scaled)
  dk = _mulsafe(_assemble(d, E, z0, scaled), dualpart(z))
  return k isa Dual ? Dual(realpart(k), dualpart(k) + dk) : Dual(k, dk)
end

_kinf(ν) = complex(Inf, 0.0) + zero(ν)
_kinf(ν::Dual) = Dual(complex(Inf, 0.0), complex(0.0, 0.0))

# a * b without IEEE complex-multiply NaNs (Inf * 0 in the cross terms) when b is real-valued
_mulsafe(a, b::Complex) = iszero(imag(b)) ? a * real(b) : a * b
_mulsafe(a, b) = a * b

_realify(x::Complex) = real(x)
_realify(x::Dual) = Dual(real(realpart(x)), real(dualpart(x)))

# real Dual order with a real (or real Dual) argument: the result is real, like
# SpecialFunctions' besselk(::Real, ::Real), which also rejects negative arguments
function _besselk_realdual(ν::Dual, z, scaled::Bool)
  real(_val(z)) < 0 && throw(DomainError(_val(z),
    "a real argument must be non-negative; pass a complex argument instead"))
  return _realify(_besselk_general(ν, z, scaled))
end

const _RealOrRealDual = Union{Real,Dual{<:Real}}

"""
    besselk(ν, z)

Modified Bessel function of the second kind, K_ν(z), on the principal branch
(|arg z| < π; for z on the negative real axis the sign of `imag(z)`, including the sign
of a zero, selects the side of the cut).

In addition to the methods in SpecialFunctions this supports
* complex orders `ν`,
* `DualNumbers.Dual` orders (∂K/∂ν propagated through the algorithm), and
* `DualNumbers.Dual` arguments (∂K/∂z = -(K_{ν-1} + K_{ν+1})/2), in any combination.

The result is real (`Float64` or `Dual{Float64}`) when the order and the argument are
both real, and complex otherwise. Computation is in `Float64`.
"""
besselk(ν::Complex, z::Number) = _besselk_complex_order(ν, z, false)
besselk(ν::Dual, z::Number) = _besselk_general(ν, z, false)
besselk(ν::Dual{<:Real}, z::_RealOrRealDual) = _besselk_realdual(ν, z, false)
besselk(ν::Real, z::Dual) = _besselk_real_order_dual(ν, z, false)

"""
    besselkx(ν, z)

Scaled modified Bessel function of the second kind, e^z K_ν(z). Accepts the same
argument types as [`besselk`](@ref).
"""
besselkx(ν::Complex, z::Number) = _besselk_complex_order(ν, z, true)
besselkx(ν::Dual, z::Number) = _besselk_general(ν, z, true)
besselkx(ν::Dual{<:Real}, z::_RealOrRealDual) = _besselk_realdual(ν, z, true)
besselkx(ν::Real, z::Dual) = _besselk_real_order_dual(ν, z, true)

function _besselk_complex_order(ν::Complex, z, scaled::Bool)
  if iszero(imag(ν))       # real order: fast large-|z| path, else SpecialFunctions (AMOS)
    νr = Float64(real(ν))
    z isa Dual && return _besselk_real_order_dual(νr, _cplx(z), scaled)
    return _besselk_real_order(νr, ComplexF64(z), scaled)
  end
  return _besselk_general(ν, z, scaled)
end

# real order, complex argument
function _besselk_real_order(ν::Float64, z::ComplexF64, scaled::Bool)
  kx, _, ok = _kx_fast(ν, z, Val(false))
  ok && return scaled ? kx : kx * exp(-z)        # Re z >= 0 here: no overflow
  return scaled ? besselkx(ν, z) : besselk(ν, z)
end

# Integer orders with a ComplexF64 argument get the fast path too. (A Float64 order with a
# ComplexF64 argument matches a concrete SpecialFunctions method, which can't be replaced
# without overwriting it.)
besselk(ν::Integer, z::ComplexF64) = _besselk_real_order(Float64(ν), z, false)
besselkx(ν::Integer, z::ComplexF64) = _besselk_real_order(Float64(ν), z, true)

# real order, Dual argument. Large |z|: value and derivative from the fast expansion.
# Otherwise values from SpecialFunctions and the derivative from
#   K_ν' = -(K_{ν-1} + K_{ν+1})/2,   (e^z K_ν)' = e^z K_ν - e^z (K_{ν-1} + K_{ν+1})/2
function _besselk_real_order_dual(ν::Real, z::Dual, scaled::Bool)
  z0 = float(realpart(z))
  kx, dkx, ok = _kx_fast(Float64(ν), ComplexF64(z0), Val(true))
  if ok
    if scaled
      k, dk = kx, dkx
    else
      ez = exp(-ComplexF64(z0))
      k, dk = kx * ez, (dkx - kx) * ez             # d/dz (e^{-z} Kx) = e^{-z}(Kx' - Kx)
    end
    return Dual(_like(k, z0), _mulsafe(_like(dk, z0), dualpart(z)))
  end
  f = scaled ? besselkx : besselk
  k = f(ν, z0)
  dk = -(f(ν - 1, z0) + f(ν + 1, z0)) / 2
  scaled && (dk += k)
  return Dual(k, _mulsafe(dk, dualpart(z)))
end
# a real argument gives a real result (the fast path only runs for z0 >= 0)
_like(x::Complex, ::Real) = real(x)
_like(x::Complex, ::Complex) = x

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

function _bessel_v(f::F, ν, z) where {F}
  isempty(ν) && return similar(ν, Base.promote_op(f, eltype(ν), typeof(z)))
  i0 = firstindex(ν)
  first_ = f(ν[i0], z)
  out = similar(ν, typeof(first_))
  out[i0] = first_
  for i in i0+1:lastindex(ν)
    out[i] = (_same(ν[i], ν[i-1]) || _same(ν[i], -ν[i-1])) ? out[i-1] : f(ν[i], z)
  end
  return out
end
# Dual == compares only values; reuse requires identical ε-parts too
_same(a, b) = a == b
_same(a::Dual, b::Dual) = realpart(a) == realpart(b) && dualpart(a) == dualpart(b)

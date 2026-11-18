# Fast large-|z| path for real orders: a fixed number of Hankel terms (DLMF 10.40.2),
#   e^z K_ν(z) ≈ sqrt(π/(2z)) S(1/z),   S(w) = Σ_{k<N} a_k(ν) w^k,
#   a_k(ν) = a_{k-1}(ν) (4ν² - (2k-1)²) / (8k),
# with the real coefficients built in one pass and S evaluated by `evalpoly` (Goertzel's
# algorithm for a complex argument: mostly real arithmetic). No per-term branches; a single
# check on the first neglected term decides afterwards whether the truncation error is
# below eps. For real ν and |arg z| <= π/2 the remainder is bounded by the first neglected
# term (DLMF 10.40(ii)), so the path is only used for Re z >= 0. With 24 terms it accepts
# |z| ≳ 22 for ν ≲ 10, and the threshold rises automatically with ν; for |z| >= 60 a
# 12-term evaluation is tried first.
#
# The z-derivative comes from the same coefficients:
#   d/dz [e^z K_ν(z)] = -sqrt(π/(2z)) / z (S/2 + Σ_k k a_k w^k).
const _ODDSQ = ntuple(k -> Float64((2k - 1)^2), 24)     # (2k-1)²
const _INV8K = ntuple(k -> 1 / (8k), 24)               # 1/(8k)

# a_1 .. a_24 for a given 4ν², fully unrolled (no tuple through memory, no divisions)
@inline function _hankel_coeffs(μ4::Float64)
  Base.Cartesian.@nexprs 24 k -> r_k = (μ4 - _ODDSQ[k]) * _INV8K[k]
  a_1 = r_1
  Base.Cartesian.@nexprs 23 k -> a_{k + 1} = a_k * r_{k + 1}
  return Base.Cartesian.@ntuple 24 k -> a_k
end

# precomputed coefficients for integer orders |ν| <= 16 (a table lookup instead of the
# 24-step dependent multiply chain)
const _HANKEL_INT = ntuple(n -> _hankel_coeffs(4.0 * (n - 1)^2), 17)

# (e^z K_ν(z), d/dz e^z K_ν(z), accepted); the derivative is only computed if D.
# Large |z| needs only half the terms, so try 12 first and use all 24 only if needed.
@inline function _kx_fast(ν::Float64, z::ComplexF64, ::Val{D}) where {D}
  aν = abs(ν)
  # a_1 .. a_24
  a = (aν <= 16 && isinteger(aν)) ? @inbounds(_HANKEL_INT[Int(aν) + 1]) : _hankel_coeffs(4ν * ν)
  if abs2(z) >= 3600.0                           # |z| >= 60: 12 terms usually suffice
    r = _kx_fast_n(a, z, Val(12), Val(D))
    r[3] && return r
  end
  return _kx_fast_n(a, z, Val(24), Val(D))
end

# the first N-1 coefficients summed, a_N the first neglected term
@inline function _kx_fast_n(a::NTuple{24,Float64}, z::ComplexF64, ::Val{N}, ::Val{D}) where {N,D}
  w = inv(z)
  s = @inline evalpoly(w, (1.0, ntuple(k -> a[k], Val(N - 1))...))
  # N |a_N w^N| <= eps/2 |s|, compared in squares, |w|^(2N) by repeated squaring
  l1s = abs(real(s)) + abs(imag(s))
  ok = (real(z) >= 0) & ((N * a[N])^2 * _pow2n(abs2(w), Val(N)) <= (0.5eps() * l1s)^2)
  pre = sqrt(π / 2 * w)
  kx = pre * s
  D || return kx, zero(kx), ok
  sd = w * @inline evalpoly(w, ntuple(k -> k * a[k], Val(N - 1)))
  return kx, -pre * w * (s / 2 + sd), ok
end
@inline _pow2n(q, ::Val{12}) = (q2 = q * q; q4 = q2 * q2; q8 = q4 * q4; q8 * q4)
@inline _pow2n(q, ::Val{24}) = (q2 = q * q; q4 = q2 * q2; q8 = q4 * q4; q16 = q8 * q8; q16 * q8)

# ---- with the order derivative as well ----
# d/dν of r_k = (4ν² - (2k-1)²)/(8k) is ν/k, so by the product rule
#   a'_{k+1} = a'_k r_{k+1} + a_k ν/(k+1),
# with no divisions (fine at half-integer ν, where some r_k vanish), and
#   ∂/∂ν e^z K_ν(z) ≈ sqrt(π/(2z)) Σ_k a'_k(ν) w^k.
const _INVK = ntuple(k -> 1.0 / k, 24)

@inline function _hankel_coeffs_dν(ν::Float64)
  μ4 = 4ν * ν
  Base.Cartesian.@nexprs 24 k -> r_k = (μ4 - _ODDSQ[k]) * _INV8K[k]
  a_1 = r_1
  da_1 = ν
  Base.Cartesian.@nexprs 23 k -> begin
    a_{k + 1} = a_k * r_{k + 1}
    da_{k + 1} = muladd(da_k, r_{k + 1}, a_k * (ν * _INVK[k + 1]))
  end
  return (Base.Cartesian.@ntuple 24 k -> a_k), (Base.Cartesian.@ntuple 24 k -> da_k)
end

# (e^z K_ν, ∂/∂ν e^z K_ν, ∂/∂z e^z K_ν, accepted)
@inline function _kx_fast_dν(ν::Float64, z::ComplexF64)
  a, da = _hankel_coeffs_dν(ν)
  if abs2(z) >= 3600.0
    r = _kx_fast_dν_n(a, da, z, Val(12))
    r[4] && return r
  end
  return _kx_fast_dν_n(a, da, z, Val(24))
end

@inline function _kx_fast_dν_n(a, da, z::ComplexF64, ::Val{N}) where {N}
  kx, kxz, ok = _kx_fast_n(a, z, Val(N), Val(true))
  w = inv(z)
  sν = w * @inline evalpoly(w, ntuple(k -> da[k], Val(N - 1)))
  # same truncation test for the order derivative (ν = 0: all a'_k = 0, accepted)
  l1 = abs(real(sν)) + abs(imag(sν))
  okν = (N * da[N])^2 * _pow2n(abs2(w), Val(N)) <= (0.5eps() * l1)^2
  return kx, sqrt(π / 2 * w) * sν, kxz, ok & okν
end

# Helpers that let the same algorithms run on plain numbers and on DualNumbers.Dual.
# Branch decisions are always taken on the value (real part of the Dual); convergence
# checks look at the value and the derivative separately.

_val(x) = x
_val(x::Dual) = realpart(x)

# all computation is in Float64
_cplx(x) = ComplexF64(x)
_cplx(x::Dual) = Dual(ComplexF64(realpart(x)), ComplexF64(dualpart(x)))

# cheap norm |re| + |im| (within √2 of abs; no hypot in the inner loops)
_l1(x::Real) = abs(x)
_l1(x::Complex) = abs(real(x)) + abs(imag(x))

# size for steering iterations (e.g. "terms are still decreasing")
_mag(x) = _l1(x)
_mag(x::Dual) = _l1(realpart(x)) + _l1(dualpart(x))

# has a series with latest term t and running sum s converged to relative tolerance tol?
_conv(t, s, tol) = _l1(t) <= tol * _l1(s)
function _conv(t::Dual, s::Dual, tol)
  rt, rs = _l1(realpart(t)), _l1(realpart(s))
  dt, ds = _l1(dualpart(t)), _l1(dualpart(s))
  return rt <= tol * rs && dt <= tol * (ds + tol * rs)
end
_conv(t::Dual, s, tol) = _conv(t, Dual(s, zero(s)), tol)
_conv(t, s::Dual, tol) = _conv(Dual(t, zero(t)), s, tol)

# Taylor coefficients of 1/Γ(1+x) about x = 0 (entire function)
const _RGAMMA1 = (1.0, 0.5772156649015329, -0.6558780715202539, -0.04200263503409524,
  0.16653861138229148, -0.04219773455554433, -0.009621971527876973, 0.0072189432466631,
  -0.0011651675918590652, -0.00021524167411495098, 0.0001280502823881162,
  -2.013485478078824e-05, -1.2504934821426706e-06, 1.133027231981696e-06,
  -2.056338416977607e-07, 6.116095104481416e-09, 5.002007644469223e-09,
  -1.18127457048702e-09, 1.0434267116911005e-10, 7.782263439905071e-12,
  -3.696805618642206e-12, 5.100370287454476e-13, -2.0583260535665066e-14,
  -5.348122539423018e-15, 1.2267786282382608e-15, -1.1812593016974588e-16,
  1.1866922547516004e-18, 1.4123806553180319e-18, -2.29874568443537e-19,
  1.7144063219273374e-20, 1.337351730493693e-22, -2.0542335517666728e-22)
const _RG_EVEN = _RGAMMA1[1:2:end]          # c0, c2, c4, ...
const _RG_ODD = _RGAMMA1[2:2:end]           # c1, c3, c5, ...

# Temme's gamma combinations for the order μ:
#   gampl = 1/Γ(1+μ), gammi = 1/Γ(1-μ),
#   Γ1 = (gammi - gampl) / (2μ), Γ2 = (gammi + gampl) / 2,
# with Γ1 computed without cancellation as μ -> 0.
function _temme_gammas(μ)
  if abs(_val(μ)) < 1
    μ² = μ * μ
    Γ2 = evalpoly(μ², _RG_EVEN)
    Γ1 = -evalpoly(μ², _RG_ODD)
    return Γ2 - μ * Γ1, Γ2 + μ * Γ1, Γ1, Γ2
  else
    gampl = inv(gamma(1 + μ))
    gammi = inv(gamma(1 - μ))
    return gampl, gammi, (gammi - gampl) / (2μ), (gammi + gampl) / 2
  end
end

# sinh(x)/x
function _sinhc(x)
  abs(_val(x)) < 0.05 || return sinh(x) / x
  return evalpoly(x * x, (1.0, 1 / 6, 1 / 120, 1 / 5040, 1 / 362880, 1 / 39916800))
end

# x / sin(x) evaluated at x = πμ
function _pimu_csc(μ)
  abs(_val(μ)) < 0.01 || return π * μ / sinpi(μ)
  x² = (π * μ)^2
  return evalpoly(x², (1.0, 1 / 6, 7 / 360, 31 / 15120, 127 / 604800, 73 / 3421440))
end

# |value| as a plain float (used for error estimates)
_absv(x) = _l1(_val(x))

# 1/x without the scaling and branches of Base's robust complex division: conj(x)/|x|²,
# falling back to inv(x) when |x|² is near the ends of the Float64 range
@inline function _finv(x::Complex{Float64})
  s = abs2(x)
  return (1e-290 < s < 1e290) ? conj(x) * inv(s) : inv(x)
end
@inline _finv(x::Real) = inv(x)
@inline function _finv(x::Dual)
  r = _finv(realpart(x))
  return Dual(r, -dualpart(x) * (r * r))
end

# log Γ for Dual arguments (SpecialFunctions has none)
_loggamma(x) = loggamma(x)
_loggamma(x::Dual) = Dual(loggamma(realpart(x)), digamma(realpart(x)) * dualpart(x))

# log(sin(πν)) without overflow for large |Im ν| (branch irrelevant: only exponentiated)
function _logsinpi(ν)
  y = imag(_val(ν))
  abs(y) < 1 && return log(sinpi(ν))
  # sin(πν) = (e^{iπν} - e^{-iπν}) / 2i; factor out the larger exponential
  if y > 0
    return -im * π * ν + log((1 - exp(2im * π * ν)) / (-2im))
  else
    return im * π * ν + log((1 - exp(-2im * π * ν)) / (2im))
  end
end

# x * 2^e, componentwise, so it overflows to ±Inf / underflows to 0 without NaN
_ldexp(x::Real, e::Int) = ldexp(x, e)
_ldexp(x::Complex, e::Int) = complex(ldexp(real(x), e), ldexp(imag(x), e))
_ldexp(x::Dual, e::Int) = Dual(_ldexp(realpart(x), e), _ldexp(dualpart(x), e))

# (x, e) -> (x * 2^-m, e + m) with the mantissa of order one (no-op for 0, Inf, NaN)
function _normalise(x, e::Int)
  a = _absv(x)
  (iszero(a) || !isfinite(a)) && return x, e
  m = exponent(a)
  return _ldexp(x, -m), e + m
end

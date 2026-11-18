# Helpers that let the same algorithms run on plain numbers and on DualNumbers.Dual.
# Branch decisions are always taken on the value (real part of the Dual); convergence
# checks look at the value and the derivative separately.

_val(x) = x
_val(x::Dual) = realpart(x)

_cplx(x) = complex(float(x))
_cplx(x::Dual) = Dual(complex(float(realpart(x))), complex(float(dualpart(x))))

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

# ---- BigFloat support for the high-precision fallback ----

# Bernoulli numbers B_2, B_4, ..., B_{2n} (Akiyama–Tanigawa), cached
const _BERNOULLI = Rational{BigInt}[]
function _bernoulli_even(n)
  length(_BERNOULLI) >= n && return _BERNOULLI
  N = 2n
  a = Vector{Rational{BigInt}}(undef, N + 1)
  B = Vector{Rational{BigInt}}(undef, N + 1)
  for m in 0:N
    a[m+1] = 1 // big(m + 1)
    for j in m:-1:1
      a[j] = j * (a[j] - a[j+1])
    end
    B[m+1] = a[1]                      # B_m (with B_1 = +1/2, unused)
  end
  empty!(_BERNOULLI)
  append!(_BERNOULLI, B[3:2:end])
  return _BERNOULLI
end

# ψ(z) for Complex{BigFloat}: shift to large Re z, then the asymptotic series
#   ψ(z) ~ log z - 1/(2z) - Σ_k B_{2k} / (2k z^{2k})
function _digamma_big(z::Complex{BigFloat})
  p = precision(BigFloat)
  shift = zero(z)
  while real(z) < p / 2
    shift -= inv(z)
    z += 1
  end
  B = _bernoulli_even(60)
  iz² = inv(z * z)
  s = zero(z)
  zk = one(z)
  tol = eps(BigFloat)
  for k in 1:60
    zk *= iz²
    t = B[k] / (2k) * zk
    s += t
    abs(t) <= tol * abs(s) && break
  end
  return shift + log(z) - inv(2z) - s
end
_loggamma(x::Dual{Complex{BigFloat}}) =
  Dual(loggamma(realpart(x)), _digamma_big(realpart(x)) * dualpart(x))

_tobig(x::Complex) = Complex{BigFloat}(x)
_tobig(x::Dual) = Dual(_tobig(realpart(x)), _tobig(dualpart(x)))
_to64(x::Complex) = ComplexF64(x)
_to64(x::Dual) = Dual(_to64(realpart(x)), _to64(dualpart(x)))

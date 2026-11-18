# Real order with Re z > 0: the same algorithms with a real Dual{Float64} order carrying
# ∂/∂ν (they are generic). For real x everything is real arithmetic, about 5× cheaper than
# a Dual{ComplexF64} order through the complex code; for complex z the order-dependent
# quantities stay real, about 1.6× cheaper. Forward recurrence is stable for real ν and
# Re z > 0, so the direct-series candidate is not needed; if the error estimate is poor
# the caller falls back to the general complex path.

# (e^z K_ν(z) mantissa, exponent, accepted) for ν >= 0, Re z > 0
function _kx_real(ν, x::Union{Float64,ComplexF64})
  if abs(x) >= ASYM_MIN_ABSZ
    k, ok = _hankel_kx(ν, x, TOL)
    ok && return k, 0, true
  end
  n = round(Int, _val(ν))
  μ = ν - n                                     # |μ| <= 1/2
  if abs(x) <= TEMME_MAX_ABSZ
    t0, t1, est = _temme(μ, x, TOL)
    k0, _, amp, e = _recur(μ, x, t0, t1, n)
    return k0 * exp(x), e, est * amp <= GOOD_EST
  end
  s0, s1, est = _steed(μ, x, TOL)
  k0, _, amp, e = _recur(μ, x, s0, s1, n)
  return k0, e, est * amp <= GOOD_EST
end

# K or e^x K from a mantissa and exponent (overflow -> Inf, underflow -> 0, never NaN)
function _assemble_real(k, e::Int, x::Float64, scaled::Bool)
  k, e = _normalise(k, e)
  scaled && return _ldexp(k, e)
  j = x > 500 ? round(Int, -x / log(2.0)) : 0
  return _ldexp(k * exp(-x - j * log(2.0)), e + j)
end

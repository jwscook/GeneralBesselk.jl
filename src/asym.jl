# Large-|z| Hankel expansion (DLMF 10.40.2), valid for |arg z| < 3π/2:
#   e^z K_ν(z) ~ sqrt(π/(2z)) Σ_k a_k(ν) / z^k,
#   a_k(ν)/z^k = a_{k-1}(ν)/z^{k-1} (4ν² - (2k-1)²) / (8 k z).
# The series is divergent: terms are summed while they decrease and the result is
# accepted (second return value true) only if the last term is below tol relative to the sum.
# For half-integer ν the series terminates and is exact.
function _hankel_kx(ν, z, tol)
  μ4 = 4 * ν * ν
  iz8 = inv(8 * z)
  term = one(μ4 * iz8)
  s = term
  prev = Inf
  for k in 1:500
    term *= (μ4 - (2k - 1)^2) * (iz8 / k)
    m = _mag(term)
    m > prev && return sqrt(π / (2z)) * s, false
    prev = m
    s += term
    _conv(term, s, tol) && return sqrt(π / (2z)) * s, true
  end
  return sqrt(π / (2z)) * s, false
end

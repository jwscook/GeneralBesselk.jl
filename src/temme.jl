# Temme's series (N. M. Temme, J. Comput. Phys. 19, 324 (1975)) for K_μ(z) and K_{μ+1}(z),
# |Re μ| <= 1/2, small |z|. Uniform in μ (no cancellation as μ -> integer) and valid for
# any complex μ and any z off the branch cut, using the principal log(z/2).
# Returns the *unscaled* pair and an estimate of the relative error in units of eps.
function _temme(μ, z, tol)
  x2 = z / 2
  d = -log(x2)
  e = μ * d
  fact = _pimu_csc(μ)
  fact2 = _sinhc(e)
  gampl, gammi, Γ1, Γ2 = _temme_gammas(μ)
  ff = fact * (Γ1 * cosh(e) + Γ2 * fact2 * d)
  s = ff
  ee = exp(e)
  p = ee / (2 * gampl)                 # (z/2)^(-μ) Γ(1+μ) / 2
  q = 1 / (2 * ee * gammi)             # (z/2)^(+μ) Γ(1-μ) / 2
  c = one(x2)
  x2² = x2 * x2
  s1 = p
  μ² = μ * μ
  sa = _absv(s)
  sa1 = _absv(s1)
  for i in 1:MAXITERS
    iden = inv(i * i - μ²)               # 1/((i-μ)(i+μ)), shared by all three updates
    ff = (i * ff + p + q) * iden
    c *= x2² / i
    p *= (i + μ) * iden
    q *= (i - μ) * iden
    del = c * ff
    s += del
    del1 = c * (p - i * ff)
    s1 += del1
    sa += _absv(del)
    sa1 += _absv(del1)
    _conv(del, s, tol) && _conv(del1, s1, tol) && break
  end
  est = max(sa / _absv(s), sa1 / _absv(s1)) + _absv(e)
  return s, s1 * 2 / z, est
end

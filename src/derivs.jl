# The derivative core shared by every AD interface (DualNumbers here, ForwardDiff and
# ChainRulesCore in package extensions): value and both first derivatives at plain Float64
# arguments,
#   _kderivs(ν, z, scaled, needν, needz) -> (K, ∂K/∂ν, ∂K/∂z)       (ComplexF64 each)
# or the same for e^z K when scaled. Derivatives not asked for are returned as 0 and not
# computed. Any number of AD partials then follows by the chain rule
#   dK = ∂K/∂ν dν + ∂K/∂z dz          (K is analytic in z),
# so the cost is independent of the number of partials.

function _kderivs(ν::Float64, z::ComplexF64, scaled::Bool, needν::Bool, needz::Bool)
  # large |z|: everything from one fixed-length Hankel series
  kx, kxν, kxz, ok = needν ? _kx_fast_dν(ν, z) : _kx_fast_nodν(ν, z, needz)
  if ok
    scaled && return kx, kxν, kxz
    ez = exp(-z)                                   # Re z >= 0 here
    return kx * ez, kxν * ez, (kxz - kx) * ez      # d/dz (e^{-z} Kx) = e^{-z}(Kx' - Kx)
  end
  if needν
    # K and ∂K/∂ν with a real Dual order (K_{-ν} = K_ν, so ∂/∂ν changes sign with ν; at
    # ν = 0 it vanishes)
    νd = Dual(abs(ν), sign(ν))
    x = real(z)
    done = false
    K, Kν = zero(z), zero(z)
    if x > 0 && iszero(imag(z))                    # real arithmetic throughout
      kr, e, done = _kx_real(νd, x)
      kd = _assemble_real(kr, e, x, scaled)
      K, Kν = complex(realpart(kd)), complex(dualpart(kd))
    elseif x > 0
      kc, e, done = _kx_real(νd, z)
      kd = _assemble(kc, e, z, scaled)
      K, Kν = realpart(kd), dualpart(kd)
    end
    if !done
      kc, _ = _kz_parts(_cplx(νd), z, scaled, false)
      K, Kν = realpart(kc), dualpart(kc)
    end
    needz || return K, Kν, zero(K)
    _, Kz, ok = _amos_kz(ν, z, scaled)
    ok && return K, Kν, Kz
    return K, Kν, _kz_parts(complex(abs(ν)), z, scaled, true)[2]
  end
  # real order, no order derivative: AMOS values, symmetric z-derivative from one call
  if needz
    k, kz, ok = _amos_kz(ν, z, scaled)
    ok && return k, zero(k), kz
  end
  f = scaled ? besselkx : besselk
  k = f(ν, z)
  needz || return k, zero(k), zero(k)
  kz = -(f(ν - 1, z) + f(ν + 1, z)) / 2
  return k, zero(k), scaled ? kz + k : kz
end

@inline function _kx_fast_nodν(ν::Float64, z::ComplexF64, needz::Bool)
  kx, kxz, ok = needz ? _kx_fast(ν, z, Val(true)) : _kx_fast(ν, z, Val(false))
  return kx, zero(kx), kxz, ok
end

function _kderivs(ν::ComplexF64, z::ComplexF64, scaled::Bool, needν::Bool, needz::Bool)
  # a complex order on the real axis: the real-order routes (K is analytic in ν, so the
  # real-direction derivative is the complex derivative)
  iszero(imag(ν)) && return _kderivs(real(ν), z, scaled, needν, needz)
  return _kderivs_general(ν, z, scaled, needν, needz)
end

function _kderivs_general(ν::ComplexF64, z::ComplexF64, scaled::Bool, needν::Bool, needz::Bool)
  if needν
    # ∂K/∂ν propagated through the algorithm by a DualNumbers order; reflecting to
    # Re ν >= 0 (K_{-ν} = K_ν) keeps the chain rule sign inside the Dual
    νd = Dual(ν, one(ν))
    real(ν) < 0 && (νd = -νd)
    k, kz = _kz_parts(νd, z, scaled, needz)
    return realpart(k), dualpart(k), kz
  end
  k, kz = _kz_parts(real(ν) < 0 ? -ν : ν, z, scaled, needz)
  return k, zero(k), kz
end

_coreν(ν::Real) = Float64(ν)
_coreν(ν::Complex) = ComplexF64(ν)

# a real order with a real argument gives real results, and needs a non-negative argument
function _check_real_domain(z)
  real(z) < 0 && throw(DomainError(z,
    "a real argument must be non-negative; pass a complex argument instead"))
  return nothing
end

# ---- DualNumbers ----
# Any combination of DualNumbers.Dual order and argument (one ε): one core call.
function _dn_besselk(ν, z, scaled::Bool)
  hasν, hasz = ν isa Dual, z isa Dual
  νv = hasν ? realpart(ν) : ν
  zv = hasz ? realpart(z) : z
  K, Kν, Kz = _kderivs(_coreν(νv), ComplexF64(zv), scaled, hasν, hasz)
  d = zero(K)
  hasν && (d += _mulsafe(Kν, dualpart(ν)))
  hasz && (d += _mulsafe(Kz, dualpart(z)))
  return Dual(K, d)
end

# (K, ∂K/∂ν, ∂K/∂z) at plain real or complex arguments, with the package's type rule:
# real order and real argument give real results (and need a non-negative argument).
# Used by the ForwardDiff and ChainRulesCore extensions.
function _kderivs_typed(ν::Union{Real,Complex}, z::Union{Real,Complex}, scaled::Bool,
    needν::Bool, needz::Bool)
  if ν isa Real && z isa Real
    _check_real_domain(z)
    K, Kν, Kz = _kderivs(Float64(ν), ComplexF64(z), scaled, needν, needz)
    return real(K), real(Kν), real(Kz)
  end
  return _kderivs(_coreν(ν), ComplexF64(z), scaled, needν, needz)
end

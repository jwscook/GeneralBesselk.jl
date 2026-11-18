module GeneralBesselkForwardDiffExt

# ForwardDiff support. One layer of ForwardDiff.Dual (one tag) is stripped from the order
# and/or argument, (K, ∂K/∂ν, ∂K/∂z) are computed once at the values, and every partial
# follows by the chain rule dK = ∂K/∂ν dν + ∂K/∂z dz (K is analytic in z), so the cost
# does not grow with the number of partials. If the values are themselves Duals (nested
# differentiation) K and the derivatives are obtained by calling besselk again with the
# symmetric form in z, which recurses through the same code, so any order of z-derivative
# and mixed ∂ⁿ/∂zⁿ⁻¹∂ν work. ∂²K/∂ν² has no such identity and is not supported.

using GeneralBesselk, ForwardDiff
using ForwardDiff: Dual, Partials, value, partials
import GeneralBesselk: besselk, besselkx
const GB = GeneralBesselk

const CDual = Complex{<:Dual}

struct NoTag end
_tag(::Dual{T}) where {T} = T
_tag(::Complex{<:Dual{T}}) where {T} = T
_tag(_) = NoTag
# the outer layer: as in ForwardDiff's binary operators, the tag that is not ≺ the other
_outer(::Type{NoTag}, ::Type{T}) where {T} = T
_outer(::Type{T}, ::Type{NoTag}) where {T} = T
_outer(::Type{T}, ::Type{T}) where {T} = T
_outer(::Type{NoTag}, ::Type{NoTag}) = NoTag            # unreachable; resolves ambiguity
_outer(::Type{T1}, ::Type{T2}) where {T1,T2} = ForwardDiff.:≺(T1, T2) ? T2 : T1

# (value, partials) of x for tag T; partials are `nothing` if x is constant in this layer
_strip(::Type{T}, x::Dual{T}) where {T} = value(x), partials(x)
_strip(::Type{T}, x::Complex{<:Dual{T}}) where {T} =
  complex(value(real(x)), value(imag(x))), (partials(real(x)), partials(imag(x)))
_strip(::Type{T}, x) where {T} = x, nothing

# i-th directional derivative of the input
_dir(p::Partials, i) = p[i]
_dir(p::Tuple{<:Partials,<:Partials}, i) = complex(p[1][i], p[2][i])
# the number of partials, by dispatch so that it is a compile-time constant
_npartials(::Partials{N}) where {N} = Val(N)
_npartials(::Tuple{Partials{N,V},Partials{N,V}}) where {N,V} = Val(N)
_npartials(::Nothing, zp) = _npartials(zp)
_npartials(νp, _) = _npartials(νp)

_isplain(x) = x isa Union{Real,Complex} && !(x isa Dual) && !(x isa CDual)

function _fd(f::F, scaled::Bool, ν, z) where {F}
  T = _outer(_tag(ν), _tag(z))
  νv, νp = _strip(T, ν)
  zv, zp = _strip(T, z)
  needν, needz = νp !== nothing, zp !== nothing
  N = _npartials(νp, zp)
  K, Kν, Kz = _parts(f, scaled, νv, zv, needν, needz)
  ps = ntuple(N) do i
    d = zero(K)
    needν && (d += Kν * _dir(νp, i))
    needz && (d += Kz * _dir(zp, i))
    d
  end
  return _rebuild(T, K, ps)
end
# (K, ∂K/∂ν, ∂K/∂z) at the stripped values (a separate function so that nothing the
# partials closure captures is reassigned, which would box it)
function _parts(f::F, scaled::Bool, νv, zv, needν::Bool, needz::Bool) where {F}
  _isplain(νv) && _isplain(zv) && return GB._kderivs_typed(νv, zv, scaled, needν, needz)
  # nested Duals: recurse through besselk itself
  needν && !_isplain(νv) && throw(ArgumentError("nested ForwardDiff differentiation " *
    "with respect to the order of besselk/besselkx is not supported (it needs ∂²K/∂ν²)"))
  K, Kν = needν ? _kν(f, scaled, νv, zv) : (f(νv, zv), nothing)
  Kz = needz ? _kz(f, scaled, νv, zv, K) : zero(K)
  return K, needν ? Kν : zero(K), Kz
end

# d/dz of K (or of e^z K) from the symmetric form
_kz(f::F, scaled::Bool, ν, z, K) where {F} =
  (Kz = -(f(ν - 1, z) + f(ν + 1, z)) / 2; scaled ? Kz + K : Kz)

# (K, ∂K/∂ν) at a plain order and a possibly nested Dual argument: the z-partials of ∂K/∂ν
# come from differentiating the symmetric form in ν, ∂/∂z ∂K/∂ν = -(∂K_{ν-1}/∂ν + ∂K_{ν+1}/∂ν)/2
function _kν(f::F, scaled::Bool, ν, z) where {F}
  if _isplain(z)
    K, Kν, _ = GB._kderivs_typed(ν, z, scaled, true, false)
    return K, Kν
  end
  T = _tag(z)
  zv, zp = _strip(T, z)
  K0, Kν0 = _kν(f, scaled, ν, zv)
  Kz = _kz(f, scaled, ν, zv, K0)
  Kνz = -(_kν(f, scaled, ν - 1, zv)[2] + _kν(f, scaled, ν + 1, zv)[2]) / 2
  scaled && (Kνz += Kν0)
  N = _npartials(zp)
  return _rebuild(T, K0, ntuple(i -> Kz * _dir(zp, i), N)),
         _rebuild(T, Kν0, ntuple(i -> Kνz * _dir(zp, i), N))
end

_rebuild(::Type{T}, K::Real, ps) where {T} = Dual{T}(K, Partials(ps))
_rebuild(::Type{T}, K::Complex, ps) where {T} =
  complex(Dual{T}(real(K), Partials(map(real, ps))), Dual{T}(imag(K), Partials(map(imag, ps))))

# Methods for every order/argument combination containing a ForwardDiff.Dual.
#
# Real order and real argument: ForwardDiff itself defines besselk(::Dual, ::R),
# (::R, ::Dual) for R in its AMBIGUOUS_TYPES, and (::Dual, ::Dual), through DiffRules (with
# ∂/∂ν = NaN). Those cannot be redefined, so the methods here are strictly more specific:
# Duals whose value type is a float (FDual). Nested real Duals therefore keep ForwardDiff's
# rule. The R list mirrors ForwardDiff's to avoid ambiguities.
#
# Complex order or argument: ForwardDiff defines nothing, so these are only made more
# specific than SpecialFunctions' (Real, Complex) and this package's (Complex, Number).
const FDual{T} = Dual{T,<:AbstractFloat}

for (f, scaled) in ((:besselk, false), (:besselkx, true))
  for R in (Real, AbstractFloat, Integer, Rational, Irrational)
    @eval $f(ν::FDual, z::$R) = _fd($f, $scaled, ν, z)
    @eval $f(ν::$R, z::FDual) = _fd($f, $scaled, ν, z)
  end
  @eval $f(ν::FDual{T}, z::FDual{T}) where {T} = _fd($f, $scaled, ν, z)
  @eval $f(ν::FDual{T1}, z::FDual{T2}) where {T1,T2} = _fd($f, $scaled, ν, z)
  # a float-valued Dual meeting a nested one (overlap with ForwardDiff's (Dual, Dual))
  @eval $f(ν::FDual{T}, z::Dual{T}) where {T} = _fd($f, $scaled, ν, z)
  @eval $f(ν::Dual{T}, z::FDual{T}) where {T} = _fd($f, $scaled, ν, z)
  @eval $f(ν::FDual{T1}, z::Dual{T2}) where {T1,T2} = _fd($f, $scaled, ν, z)
  @eval $f(ν::Dual{T1}, z::FDual{T2}) where {T1,T2} = _fd($f, $scaled, ν, z)
  for (Tν, Tz) in ((Dual, Complex), (Dual, CDual), (Real, CDual), (Complex, Dual),
                   (Complex, CDual), (CDual, Number), (CDual, Dual), (CDual, CDual))
    @eval $f(ν::$Tν, z::$Tz) = _fd($f, $scaled, ν, z)
  end
end

end # module

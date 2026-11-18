module GeneralBesselkForwardDiffExt

# ForwardDiff support. One layer of ForwardDiff.Dual (one tag) is stripped from the order
# and/or argument, (K, ∂K/∂ν, ∂K/∂z) are computed once at the values, and every partial
# follows by the chain rule dK = ∂K/∂ν dν + ∂K/∂z dz (K is analytic in z), so the cost
# does not grow with the number of partials. If the values are themselves Duals (nested
# differentiation) K and ∂K/∂z are obtained by calling besselk again, which recurses
# through the same code. ∂K/∂ν has no such identity, so nested Duals with an order
# derivative are not supported.

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
_outer(::Type{T1}, ::Type{T2}) where {T1,T2} = ForwardDiff.:≺(T1, T2) ? T2 : T1

# (value, partials) of x for tag T; partials are `nothing` if x is constant in this layer
_strip(::Type{T}, x::Dual{T}) where {T} = value(x), partials(x)
_strip(::Type{T}, x::Complex{<:Dual{T}}) where {T} =
  complex(value(real(x)), value(imag(x))), (partials(real(x)), partials(imag(x)))
_strip(::Type{T}, x) where {T} = x, nothing

# i-th directional derivative of the input
_dir(p::Partials, i) = p[i]
_dir(p::Tuple{<:Partials,<:Partials}, i) = complex(p[1][i], p[2][i])
_npartials(p::Partials{N}) where {N} = N
_npartials(p::Tuple{<:Partials{N},<:Partials}) where {N} = N

_isplain(x) = x isa Union{Real,Complex} && !(x isa Dual) && !(x isa CDual)

function _fd(f::F, scaled::Bool, ν, z) where {F}
  T = _outer(_tag(ν), _tag(z))
  νv, νp = _strip(T, ν)
  zv, zp = _strip(T, z)
  needν, needz = νp !== nothing, zp !== nothing
  N = needν ? _npartials(νp) : _npartials(zp)
  if _isplain(νv) && _isplain(zv)
    K, Kν, Kz = GB._kderivs_typed(νv, zv, scaled, needν, needz)
  else                                            # nested Duals: recurse
    needν && throw(ArgumentError("nested ForwardDiff differentiation with respect to " *
      "the order of besselk/besselkx is not supported (it needs ∂²K/∂ν²)"))
    K = f(νv, zv)
    Kν = zero(K)
    Kz = needz ? -(f(νv - 1, zv) + f(νv + 1, zv)) / 2 : zero(K)
    scaled && needz && (Kz += K)
  end
  ps = ntuple(Val(N)) do i
    d = zero(K)
    needν && (d += Kν * _dir(νp, i))
    needz && (d += Kz * _dir(zp, i))
    d
  end
  return _rebuild(T, K, ps)
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
  for (Tν, Tz) in ((Dual, Complex), (Dual, CDual), (Real, CDual), (Complex, Dual),
                   (Complex, CDual), (CDual, Number), (CDual, Dual), (CDual, CDual))
    @eval $f(ν::$Tν, z::$Tz) = _fd($f, $scaled, ν, z)
  end
end

end # module

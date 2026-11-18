module GeneralBesselkChainRulesCoreExt

# ChainRulesCore rules for besselk and besselkx, differentiable in both the order and the
# argument, for Float64 / ComplexF64 inputs. SpecialFunctions' own rules are declared for
# (Number, Number) and mark the order as not differentiable; these more specific methods
# take precedence without overwriting them (other input types keep SpecialFunctions'
# rules). K is holomorphic in z and in ν, so the pullback multiplies by the conjugate
# derivatives, and ProjectTo takes the real part for real inputs.

using GeneralBesselk, ChainRulesCore
import GeneralBesselk: besselk, besselkx
const GB = GeneralBesselk

const F64 = Union{Float64,ComplexF64}

_tangent(Δ) = unthunk(Δ)
_tangent(::AbstractZero) = false          # strong zero: false * Inf == 0

for (f, scaled) in ((:besselk, false), (:besselkx, true))
  @eval function ChainRulesCore.frule((_, Δν, Δz), ::typeof($f), ν::F64, z::F64)
    needν, needz = !(Δν isa AbstractZero), !(Δz isa AbstractZero)
    K, Kν, Kz = GB._kderivs_typed(ν, z, $scaled, needν, needz)
    return K, Kν * _tangent(Δν) + Kz * _tangent(Δz)
  end
  @eval function ChainRulesCore.rrule(::typeof($f), ν::F64, z::F64)
    K, Kν, Kz = GB._kderivs_typed(ν, z, $scaled, true, true)
    pν, pz = ProjectTo(ν), ProjectTo(z)
    function besselk_pullback(ΔK)
      Δ = unthunk(ΔK)
      return NoTangent(), pν(conj(Kν) * Δ), pz(conj(Kz) * Δ)
    end
    return K, besselk_pullback
  end
end

end # module

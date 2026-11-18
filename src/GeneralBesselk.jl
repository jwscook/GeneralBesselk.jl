module GeneralBesselk

export besselk, besselkx, besselk_v, besselkx_v

using DualNumbers, SpecialFunctions
import SpecialFunctions: besselk, besselkx

const MAXITERS = 2^14
# |z| above which the Hankel expansion is tried (it is used only if it converges to eps)
const ASYM_MIN_ABSZ = 17.0
# |z| at or below which Temme's power series is used
const TEMME_MAX_ABSZ = 2.0

include("utils.jl")
include("temme.jl")
include("cf.jl")
include("series.jl")
include("asym.jl")
include("fastasym.jl")
include("core.jl")

end # module GeneralBesselk

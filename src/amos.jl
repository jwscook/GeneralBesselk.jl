# AMOS zbesk returns a whole sequence K_{fnu}, K_{fnu+1}, ..., K_{fnu+n-1} from one call
# (it recurs forwards internally, the stable direction for K), whereas SpecialFunctions
# only ever asks for one value. The symmetric z-derivative needs K_{ν-1}, K_ν, K_{ν+1}, so
# this turns three AMOS calls into one (two for |ν| < 1, where K_{ν-1} = K_{1-ν} is not in
# the sequence).

# (values, ok): ok is false on any AMOS error or underflow in the sequence
@inline function _zbesk(fnu::Float64, z::ComplexF64, scaled::Bool, ::Val{N}) where {N}
  yr = Ref{NTuple{N,Float64}}()
  yi = Ref{NTuple{N,Float64}}()
  nz = Ref{Int32}(0)
  ierr = Ref{Int32}(0)
  ccall((:zbesk_, libopenspecfun), Cvoid,
    (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
     Ref{NTuple{N,Float64}}, Ref{NTuple{N,Float64}}, Ref{Int32}, Ref{Int32}),
    real(z), imag(z), fnu, scaled ? Int32(2) : Int32(1), Int32(N), yr, yi, nz, ierr)
  ok = (ierr[] == 0 || ierr[] == 3) && nz[] == 0
  return ntuple(k -> complex(yr[][k], yi[][k]), Val(N)), ok
end

# (K, dK/dz) for real ν (or e^z K and its derivative if scaled), and whether AMOS succeeded
function _amos_kz(ν::Float64, z::ComplexF64, scaled::Bool)
  νa = abs(ν)
  if νa >= 1
    (km, k, kp), ok = _zbesk(νa - 1, z, scaled, Val(3))
  else
    (k, kp), ok1 = _zbesk(νa, z, scaled, Val(2))
    (km,), ok2 = _zbesk(1 - νa, z, scaled, Val(1))
    ok = ok1 & ok2
  end
  kz = -(km + kp) / 2
  return k, scaled ? kz + k : kz, ok
end

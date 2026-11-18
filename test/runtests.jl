using DualNumbers, GeneralBesselk, Random, SpecialFunctions, Test
import ChainRulesCore, ForwardDiff
const G = GeneralBesselk
const FD = ForwardDiff
const CRC = ChainRulesCore
Random.seed!(0)

# (ν, z, K_ν(z), ∂K_ν(z)/∂ν) from mpmath at 40 digits
include("reference_values.jl")
# (ν, z, e^z K_ν(z), d/dz e^z K_ν(z)) for real ν, large |z|, from mpmath
include("reference_fast.jl")
# (ν, z, K, ∂K/∂ν, ∂K/∂z) for real ν, small and moderate |z|, any angle, from mpmath
include("reference_real.jl")

@testset "GeneralBesselk" begin
  @testset "complex order vs mpmath" begin
    for (ν, z, K, dK) in REFERENCE
      @test besselk(ν, z) ≈ K rtol = 1e-12
      @test besselkx(ν, z) ≈ exp(z) * K rtol = 1e-12
      # ∂K/∂ν through a Dual order; ∂K/∂ν -> 0 as ν -> 0 (K is even in ν), so allow a
      # floor relative to |K|
      kd = besselk(Dual(ν, one(ν)), z)
      @test realpart(kd) ≈ K rtol = 1e-12
      @test abs(dualpart(kd) - dK) <= 1e-9 * abs(dK) + 1e-13 * abs(K)
    end
  end

  @testset "real order: general algorithm vs SpecialFunctions (AMOS)" begin
    for _ in 1:2000
      ν = (rand() - 0.5) * 2 * rand((1, 5, 30))
      r = rand((0.1, 1.5, 3.0, 8.0, 15.0, 25.0, 60.0)) * (0.5 + rand())
      z = r * cis((rand() - 0.5) * 2 * 0.999π)
      @test G._besselk_general(complex(ν), z, true) ≈ besselkx(ν, z) rtol = 1e-12
    end
  end

  @testset "real orders are delegated to SpecialFunctions" begin
    for ν in (-3.0, 0.0, 0.5, 2.0, 7.25), z in (0.3, 2.0, 1.0 + 2im, -3.0 + 0.5im)
      @test besselk(complex(ν), z) ≈ besselk(ν, complex(z))
      @test besselkx(complex(ν), z) ≈ besselkx(ν, complex(z))
    end
    @test besselk(2.0 + 0im, 1.5) isa ComplexF64
  end

  @testset "identities" begin
    for _ in 1:500
      ν = randn(ComplexF64) * 5
      z = randn(ComplexF64) * 8
      k = besselk(ν, z)
      @test besselk(-ν, z) ≈ k rtol = 1e-12                            # K_{-ν} = K_ν
      @test besselkx(ν, z) ≈ exp(z) * k rtol = 1e-12
      # recurrence K_{ν+1} - K_{ν-1} = (2ν/z) K_ν (skip when it cancels badly)
      kp, km = besselk(ν + 1, z), besselk(ν - 1, z)
      if abs(kp - km) > 1e-3 * max(abs(kp), abs(km))
        @test kp - km ≈ 2ν / z * k rtol = 1e-9
      end
    end
    # conjugate symmetry K_{ν̄}(z̄) = conj(K_ν(z))
    @test besselk(conj(1.3 + 2im), conj(0.7 + 1.1im)) ≈ conj(besselk(1.3 + 2im, 0.7 + 1.1im))
  end

  @testset "Dual argument" begin
    for _ in 1:300
      ν = randn(ComplexF64) * 4
      z = randn(ComplexF64) * 8
      dK = -(besselk(ν - 1, z) + besselk(ν + 1, z)) / 2
      kd = besselk(ν, Dual(z, one(z)))
      @test realpart(kd) ≈ besselk(ν, z) rtol = 1e-12
      @test dualpart(kd) ≈ dK rtol = 1e-10
      kxd = besselkx(ν, Dual(z, one(z)))
      @test realpart(kxd) ≈ besselkx(ν, z) rtol = 1e-12
      @test dualpart(kxd) ≈ exp(z) * (besselk(ν, z) + dK) rtol = 1e-10
    end
    # real order: SpecialFunctions values, real Dual result
    for ν in (0.0, 1.5, 3.0, -2.25), x in (0.5, 3.0, 20.0)
      kd = besselk(ν, Dual(x, 1.0))
      @test kd isa Dual{Float64}
      @test realpart(kd) ≈ besselk(ν, x)
      @test dualpart(kd) ≈ -(besselk(ν - 1, x) + besselk(ν + 1, x)) / 2
      @test dualpart(besselkx(ν, Dual(x, 1.0))) ≈
        besselkx(ν, x) - (besselkx(ν - 1, x) + besselkx(ν + 1, x)) / 2
    end
  end

  @testset "Dual order" begin
    # real Dual order, real argument: real result, derivative vs central differences
    for ν in (0.3, 0.5, 1.0, 1.5, 2.5, 3.5, 7.0), x in (0.2, 1.5, 3.0, 6.0, 30.0)
      kd = besselk(Dual(ν, 1.0), x)
      @test kd isa Dual{Float64}
      @test realpart(kd) ≈ besselk(ν, x) rtol = 1e-12
      h = 1e-5
      fd = (besselk(ν + h, x) - besselk(ν - h, x)) / 2h
      @test dualpart(kd) ≈ fd rtol = 1e-7
    end
    # integer order is *not* short-circuited for Duals (isinteger(Dual(2.0, 1)) == true)
    @test dualpart(besselk(Dual(2.0, 1.0), 1.0)) != 0
    # at ν = 0 the order derivative vanishes
    @test abs(dualpart(besselk(Dual(0.0, 1.0), 1.3))) < 1e-14
    # Dual order and Dual argument together (same ε): total derivative
    ν, z = 0.7 + 0.4im, 1.9 - 0.6im
    k = besselk(Dual(ν, one(ν)), Dual(z, one(z)))
    ∂ν = dualpart(besselk(Dual(ν, one(ν)), z))
    ∂z = dualpart(besselk(ν, Dual(z, one(z))))
    @test dualpart(k) ≈ ∂ν + ∂z rtol = 1e-12
    # Cauchy-Riemann: complex order derivative = derivative along the real direction
    h = 1e-6
    @test dualpart(besselk(Dual(ν, one(ν)), z)) ≈
      (besselk(ν + h, z) - besselk(ν - h, z)) / 2h rtol = 1e-7
  end

  @testset "recessive K: forward recurrence unstable, direct series chosen" begin
    # large |Im ν|, z in the left half plane near the imaginary axis (mpmath values)
    ν, z = 2.2938420186840323 + 38.9968051490219im, -1.0639458484627917 + 2.974619693153284im
    @test besselk(ν, z) ≈ -194162554.209400323105638765367 + 92645341.8725884190695375764002im rtol = 1e-12
    ν, z = 11.795042701406187 - 7.521244633798166im, -0.20722034040980836 + 0.128876585942133im
    @test besselk(ν, z) ≈ -149025056.9685776846982098317562816668779 - 199513358.3121503187519713348019283715096im rtol = 1e-12
  end

  @testset "symmetric derivative in z" begin
    # real order and argument: -(K_{ν-1} + K_{ν+1})/2 has no cancellation
    for ν in (0.3 + 0.0001im, 5.5 + 1e-3im, 20.2 + 0.5im), x in (0.05, 1.0, 10.0)
      d = dualpart(besselk(ν, Dual(complex(x), 1.0 + 0im)))
      @test d ≈ -(besselk(ν - 1, x) + besselk(ν + 1, x)) / 2 rtol = 1e-13
    end
    # K_{ν-1} in a harder regime than K_ν, K_{ν+1}: falls back to (ν/z)K_ν - K_{ν+1}
    ν, z = -0.955561850644782 - 8.229566433395934im, -5.6398676012624795 - 56.92530386591847im
    @test dualpart(besselk(ν, Dual(z, one(z)))) ≈
      -35.1327599999145036156464386303 - 25.2908130070246816839240039184im rtol = 1e-10
  end

  @testset "overflow gives Inf, never NaN" begin
    # K_ν finite but K_{ν+1} (needed internally) beyond floatmax
    ν, z = 82.47797718461617 + 0.06650902909717046im, -0.011113380500254271 + 0.004035924914047084im
    @test besselk(ν, z) ≈ -1.5978706850528e305 + 6.953890020002e304im rtol = 1e-12
    k1 = besselk(ν + 1, z)                     # mpmath: 2.4267e309 - 1.48990234827458864e308im
    @test real(k1) == Inf
    @test imag(k1) ≈ -1.48990234827458864e308 rtol = 1e-10
    d = besselk(ν, Dual(z, one(z)))
    @test !isnan(realpart(d)) && !isnan(dualpart(d))
    # value beyond floatmax: Inf in the right quadrant (mpmath phase 1.229)
    k = besselk(300.0 + 1im, 0.01 + 0.01im)
    @test real(k) == Inf && imag(k) == Inf
    @test !any(isnan, reim(dualpart(besselk(Dual(300.0 + 1im, 1.0 + 0im), 0.01 + 0.01im))))
    @test besselk(Dual(300.0, 1.0), 0.01) == Dual(Inf, Inf)
    # unscaling with huge |Re z|: scaled value finite, unscaled overflows/underflows cleanly
    @test isfinite(besselkx(0.3 + 0.2im, -800.0 + 1im))
    k = besselk(0.3 + 0.2im, -800.0 + 1im)
    @test isinf(real(k)) && isinf(imag(k)) && !isnan(real(k))
    @test besselk(0.3 + 0.2im, 800.0 + 1im) == 0
  end

  @testset "fast large-|z| path for real orders" begin
    # against mpmath, including the scaled derivative (which AMOS-based formulas can only
    # get by cancellation)
    nfast = 0
    for (ν, z, kx, dkx) in REFERENCE_FAST
      f, d, ok = G._kx_fast(ν, z, Val(true))
      ok || continue
      nfast += 1
      @test f ≈ kx rtol = 2e-15
      @test d ≈ dkx rtol = 1e-13
      # public paths that use it (the Dual path may pick the longer tier, so compare it
      # with mpmath directly)
      r = besselkx(ν, Dual(z, one(z)))
      @test realpart(r) ≈ kx rtol = 2e-15
      @test dualpart(r) ≈ dkx rtol = 1e-13
      @test besselkx(ν + 0im, z) ≈ f rtol = 1e-15
      isinteger(ν) && @test besselkx(Int(ν), z) ≈ f rtol = 1e-15
    end
    @test nfast > 400
    # against AMOS on random points: values, and the unscaled derivative (no cancellation)
    for _ in 1:2000
      ν = rand((2, 3, rand(-10:10), 40 * (rand() - 0.5)))
      z = ComplexF64((15 + 600rand()^2) * cis((rand() - 0.5) * π))
      G._kx_fast(Float64(ν), z, Val(false))[3] || continue
      @test besselkx(ν + 0im, z) ≈ besselkx(Float64(ν), z) rtol = 2e-15
      @test besselk(ν + 0im, z) ≈ besselk(Float64(ν), z) rtol = 4e-15      # + rounding in e^{-z}
      ν isa Integer && @test besselk(ν, z) ≈ besselk(Float64(ν), z) rtol = 4e-15
      d = besselk(Float64(ν), Dual(z, one(z)))
      @test dualpart(d) ≈ -(besselk(ν - 1.0, z) + besselk(ν + 1.0, z)) / 2 rtol = 1e-14
    end
    # acceptance: ν = 2 from |z| ≈ 22, never for Re z < 0, small |z| or 0; the threshold
    # rises with the order
    @test G._kx_fast(2.0, 25.0 + 0im, Val(false))[3]
    @test !G._kx_fast(2.0, 15.0 + 0im, Val(false))[3]
    @test !G._kx_fast(2.0, -100.0 + 1im, Val(false))[3]
    @test !G._kx_fast(30.0, 100.0 + 0im, Val(false))[3]
    @test !G._kx_fast(2.0, 0.0im, Val(false))[3]
    # integer table and the general recurrence agree
    @test G._HANKEL_INT[4] == G._hankel_coeffs(36.0)
    # real Dual argument stays real on the fast path
    @test besselk(2, Dual(30.0, 1.0)) isa Dual{Float64}
    @test dualpart(besselk(2, Dual(30.0, 1.0))) ≈ -(besselk(1, 30.0) + besselk(3, 30.0)) / 2 rtol = 1e-14
  end

  @testset "type stability" begin
    νs = (0.3 + 0.4im, 2.0 + 0im, Dual(0.3 + 0.4im, 1.0 + 0im), Dual(1.5, 1.0), 1.5, 2)
    zs = (1.5, 2, 1.5 + 0.5im, -3.0 + 1im, 40.0 + 3im, Dual(1.5, 1.0), Dual(40.0, 1.0),
      Dual(1.5 + 0.5im, 1.0 + 0im), Dual(40.0 + 3im, 1.0 + 0im), 1.5f0 + 0.5f0im)
    for ν in νs, z in zs, f in (besselk, besselkx)
      (ν isa AbstractFloat && !(z isa Dual)) && continue   # plain SpecialFunctions
      (ν isa Integer && !(z isa Union{Dual,ComplexF64})) && continue
      @test (@inferred f(ν, z)) isa Number
    end
    # real Dual order with real argument is real, and rejects negative arguments
    @test besselk(Dual(1.5, 1.0), 2.0) isa Dual{Float64}
    @test besselk(Dual(1.5, 1.0), Dual(2.0, 1.0)) isa Dual{Float64}
    @test_throws DomainError besselk(Dual(1.5, 1.0), -2.0)
    @test besselk(Dual(1.5, 1.0), -2.0 + 0im) isa Dual{ComplexF64}
    @test (@inferred besselk_v([0.3 + 0.4im, 1.0 + 2im], 1.0 + 1im)) isa Vector{ComplexF64}
    @test (@inferred besselk_v([Dual(0.3, 1.0)], 1.0)) isa Vector{Dual{Float64}}
  end

  @testset "real order derivatives vs mpmath" begin
    for (ν, z, K, Kν, Kz) in REFERENCE_REAL
      k, kν, kz = G._kderivs(ν, z, false, true, true)
      @test k ≈ K rtol = 1e-13
      @test abs(kν - Kν) <= 1e-12 * abs(Kν) + 1e-15 * abs(K)
      @test kz ≈ Kz rtol = 1e-12
      # public routes
      @test dualpart(besselk(Dual(ν, 1.0), z)) ≈ kν
      @test dualpart(besselk(ν, Dual(z, one(z)))) ≈ kz
      r = besselk(FD.Dual(ν, 1.0, 0.0), complex(FD.Dual(real(z), 0.0, 1.0), FD.Dual(imag(z), 0.0, 0.0)))
      @test complex(FD.partials(real(r), 1), FD.partials(imag(r), 1)) ≈ kν
      @test complex(FD.partials(real(r), 2), FD.partials(imag(r), 2)) ≈ kz
      if iszero(imag(z)) && real(z) > 0
        @test besselk(Dual(ν, 1.0), real(z)) isa Dual{Float64}
        @test dualpart(besselk(Dual(ν, 1.0), real(z))) ≈ real(kν)
      end
    end
  end

  @testset "ForwardDiff" begin
    ext = Base.get_extension(GeneralBesselk, :GeneralBesselkForwardDiffExt)
    @test ext !== nothing
    @test isempty(Test.detect_ambiguities(ext; recursive = true))
    # complex order: two partials (Re ν, Im ν); K is holomorphic in ν, so
    # ∂/∂Re ν = ∂K/∂ν and ∂/∂Im ν = i ∂K/∂ν (mpmath)
    for (ν, z, K, dK) in REFERENCE[1:4:end]
      νd = complex(FD.Dual(real(ν), 1.0, 0.0), FD.Dual(imag(ν), 0.0, 1.0))
      r = besselk(νd, z)
      v = complex(FD.value(real(r)), FD.value(imag(r)))
      d1 = complex(FD.partials(real(r), 1), FD.partials(imag(r), 1))
      d2 = complex(FD.partials(real(r), 2), FD.partials(imag(r), 2))
      @test v ≈ K rtol = 1e-12
      @test abs(d1 - dK) <= 1e-9 * abs(dK) + 1e-13 * abs(K)
      @test abs(d2 - im * dK) <= 1e-9 * abs(dK) + 1e-13 * abs(K)
    end
    # complex argument, real order: ∂/∂Re z = dKx/dz and ∂/∂Im z = i dKx/dz (mpmath)
    for (ν, z, kx, dkx) in REFERENCE_FAST[1:5:end]
      zd = complex(FD.Dual(real(z), 1.0, 0.0), FD.Dual(imag(z), 0.0, 1.0))
      r = besselkx(ν, zd)
      @test complex(FD.value(real(r)), FD.value(imag(r))) ≈ kx rtol = 5e-14   # AMOS where the fast path declines
      @test complex(FD.partials(real(r), 1), FD.partials(imag(r), 1)) ≈ dkx rtol = 1e-13
      @test complex(FD.partials(real(r), 2), FD.partials(imag(r), 2)) ≈ im * dkx rtol = 1e-13
    end
    # real order and argument: ∂/∂ν (ForwardDiff alone gives NaN) and ∂/∂z, against
    # central differences
    cd(f, x, h = 1e-6) = (f(x + h) - f(x - h)) / 2h
    for ν in (0.0, 0.3, 1.5, 2.0, 7.25, -3.6), x in (0.05, 0.9, 3.0, 30.0)
      for f in (besselk, besselkx)
        gν = FD.derivative(v -> f(v, x), ν)
        gz = FD.derivative(y -> f(ν, y), x)
        @test isfinite(gν) && isfinite(gz)
        @test gν ≈ cd(v -> f(v, x), ν) rtol = 1e-6 atol = 1e-8 * abs(f(ν, x))
        @test gz ≈ cd(y -> f(ν, y), x) rtol = 1e-6
        @test FD.gradient(p -> f(p[1], p[2]), [ν, x]) ≈ [gν, gz]
      end
    end
    # several partials through one call, with both inputs dual
    p = [1.2, 0.4, 2.0, -0.7]
    J = FD.jacobian(q -> (r = besselk(complex(q[1], q[2]), complex(q[3], q[4]));
                          [real(r), imag(r)]), p)
    ν, z = complex(p[1], p[2]), complex(p[3], p[4])
    Kν = dualpart(besselk(Dual(ν, one(ν)), z))
    Kz = -(besselk(ν - 1, z) + besselk(ν + 1, z)) / 2
    @test complex(J[1, 1], J[2, 1]) ≈ Kν
    @test complex(J[1, 2], J[2, 2]) ≈ im * Kν
    @test complex(J[1, 3], J[2, 3]) ≈ Kz
    @test complex(J[1, 4], J[2, 4]) ≈ im * Kz
    # nested: second z-derivative, K'' = (K_{ν-2} + 2K_ν + K_{ν+2})/4
    for (ν, x) in ((2.3, 3.0), (0.4, 40.0))
      @test FD.derivative(y -> FD.derivative(t -> besselk(ν, t), y), x) ≈
        (besselk(ν - 2, x) + 2besselk(ν, x) + besselk(ν + 2, x)) / 4 rtol = 1e-12
    end
    z0 = 2.0 + 1.0im
    h2 = FD.derivative(y -> (r = FD.derivative(t -> besselk(1.5 + 0.5im, complex(t, imag(z0))), y);
                             r), real(z0))
    @test h2 ≈ (besselk(-0.5 + 0.5im, z0) + 2besselk(1.5 + 0.5im, z0) + besselk(3.5 + 0.5im, z0)) / 4 rtol = 1e-11
    # nested, mixed: ∂²K/∂z∂ν = ∂/∂ν (-(K_{ν-1} + K_{ν+1})/2)
    mixed = FD.derivative(y -> FD.derivative(v -> besselk(v, y), 2.3), 3.0)
    @test mixed ≈ -dualpart(besselk(Dual(1.3, 1.0), 3.0) + besselk(Dual(3.3, 1.0), 3.0)) / 2 rtol = 1e-10
    # mixed with a complex argument, and third order: ∂³K/∂z²∂ν
    zz = 2.0 + 1.0im
    m2 = FD.derivative(y -> FD.derivative(v -> besselk(v, complex(y, imag(zz))), 1.3), real(zz))
    @test m2 ≈ -dualpart(besselk(Dual(0.3, 1.0), zz) + besselk(Dual(2.3, 1.0), zz)) / 2 rtol = 1e-10
    m3 = FD.derivative(x -> FD.derivative(y -> FD.derivative(v -> besselk(v, y), 2.3), x), 3.0)
    @test m3 ≈ dualpart(besselk(Dual(0.3, 1.0), 3.0) + 2besselk(Dual(2.3, 1.0), 3.0) +
                         besselk(Dual(4.3, 1.0), 3.0)) / 4 rtol = 1e-9
    # ∂²K/∂ν² is not available
    @test_throws ArgumentError FD.derivative(u -> FD.derivative(v -> besselk(complex(v), 2.0 + 1im), u), 1.3)
    # types
    νf = FD.Dual{Nothing}(2.3, 1.0, 0.0)
    zf = complex(FD.Dual{Nothing}(3.0, 0.0, 1.0), FD.Dual{Nothing}(0.5, 0.0, 0.0))
    for f in (besselk, besselkx), (ν, z) in ((νf, zf), (νf, 3.0), (νf, 3.0 + 1im), (2.3, zf),
        (2, zf), (2.3 + 1im, zf), (2.3 + 1im, FD.Dual(3.0, 1.0)), (complex(νf, νf), 3.0),
        (complex(νf, νf), zf), (νf, FD.Dual{Nothing}(3.0, 1.0, 1.0)), (2.3, FD.Dual(3.0, 1.0)))
      @test (@inferred f(ν, z)) isa Union{FD.Dual,Complex{<:FD.Dual}}
    end
    @test besselk(νf, 3.0) isa FD.Dual{Nothing,Float64,2}
    @test_throws DomainError besselk(νf, -3.0)
    @test besselk(νf, -3.0 + 0im) isa Complex{FD.Dual{Nothing,Float64,2}}
  end

  @testset "ChainRulesCore" begin
    @test Base.get_extension(GeneralBesselk, :GeneralBesselkChainRulesCoreExt) !== nothing
    ν, z = 1.2 + 0.4im, 2.0 - 0.7im
    Kν = dualpart(besselk(Dual(ν, one(ν)), z))
    for (f, scaled) in ((besselk, false), (besselkx, true))
      K = f(ν, z)
      Kz = -(f(ν - 1, z) + f(ν + 1, z)) / 2 + (scaled ? K : 0)
      Kνf = scaled ? exp(z) * Kν : Kν
      y, ẏ = CRC.frule((CRC.NoTangent(), 0.3 - 0.1im, 0.2 + 0.5im), f, ν, z)
      @test y ≈ K && ẏ ≈ Kνf * (0.3 - 0.1im) + Kz * (0.2 + 0.5im)
      y, ẏ = CRC.frule((CRC.NoTangent(), CRC.ZeroTangent(), 1.0), f, ν, z)
      @test ẏ ≈ Kz
      y, back = CRC.rrule(f, ν, z)
      @test y ≈ K
      _, gν, gz = back(1.0 + 0im)
      @test gν ≈ conj(Kνf) && gz ≈ conj(Kz)
    end
    # real inputs: real cotangents, matching ForwardDiff
    for (f, ν, x) in ((besselk, 2.3, 3.0), (besselkx, 0.5, 40.0))
      y, back = CRC.rrule(f, ν, x)
      _, gν, gx = back(1.0)
      @test y isa Float64 && gν isa Float64 && gx isa Float64
      @test gν ≈ FD.derivative(v -> f(v, x), ν) && gx ≈ FD.derivative(t -> f(ν, t), x)
    end
    # this package's rule takes precedence over SpecialFunctions' (which has no ∂/∂ν)
    @test isfinite(CRC.rrule(besselk, 2.3, 3.0)[2](1.0)[2])
  end

  @testset "edge cases" begin
    @test isinf(abs(besselk(0.5 + 1im, 0.0)))
    # negative real argument with complex order: principal branch (upper side of the cut)
    @test besselk(0.3 + 0.2im, -2.0) ≈ besselk(0.3 + 0.2im, complex(-2.0, 0.0))
    @test besselk(0.3 + 0.2im, complex(-2.0, -0.0)) ≈
      conj(besselk(0.3 - 0.2im, complex(-2.0, 0.0)))
    # K_{1/2}(x) = sqrt(π/(2x)) e^{-x}
    @test realpart(besselkx(Dual(0.5, 0.0), 3.0)) ≈ sqrt(π / 6)
  end

  @testset "vector of orders" begin
    for _ in 1:50
      a = randn(ComplexF64, 5) * 5
      z = randn(ComplexF64) * 5
      @test besselk_v(a, z) ≈ besselk.(a, z)
      @test besselkx_v(a, z) ≈ besselkx.(a, z)
    end
    b = 1.3 + 0.7im
    @test besselk_v([b, -b], 2.0) ≈ [besselk(b, 2.0), besselk(b, 2.0)]
    @test besselk_v([1.5, 2.5], 1.0 + 1im) ≈ besselk.([1.5, 2.5], 1.0 + 1im)
    @test isempty(besselk_v(ComplexF64[], 1.0))
  end
end

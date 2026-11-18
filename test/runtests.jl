using DualNumbers, GeneralBesselk, Random, SpecialFunctions, Test
const G = GeneralBesselk
Random.seed!(0)

# (ν, z, K_ν(z), ∂K_ν(z)/∂ν) from mpmath at 40 digits
include("reference_values.jl")

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

  @testset "type stability" begin
    νs = (0.3 + 0.4im, 2.0 + 0im, Dual(0.3 + 0.4im, 1.0 + 0im), Dual(1.5, 1.0), 1.5, 2)
    zs = (1.5, 2, 1.5 + 0.5im, -3.0 + 1im, Dual(1.5, 1.0), Dual(1.5 + 0.5im, 1.0 + 0im),
      1.5f0 + 0.5f0im)
    for ν in νs, z in zs, f in (besselk, besselkx)
      (ν isa Real && !(z isa Dual)) && continue   # plain SpecialFunctions
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

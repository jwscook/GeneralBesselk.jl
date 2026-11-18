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

  @testset "hard cases use the BigFloat fallback correctly" begin
    # large |Im ν|, z near the imaginary axis in the left half plane, large |z|
    ν, z = 3.3967753376286183 + 19.82808237016364im, -11.170138958251217 + 147.42661384800755im
    @test besselk(ν, z) ≈ -9985.07898913385437987700531429 - 7735.7127845610428113410155245im rtol = 1e-12
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

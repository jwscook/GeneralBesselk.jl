# GeneralBesselk.jl

The modified Bessel function of the second kind, `besselk(ν, z)`, and its scaled form
`besselkx(ν, z) = exp(z) * besselk(ν, z)`. This extends the SpecialFunctions methods to:

* **complex orders** `ν` (SpecialFunctions/AMOS only does real orders),
* **`DualNumbers.Dual` orders**: ∂K/∂ν is carried through the algorithm,
* **`DualNumbers.Dual` arguments**: ∂K/∂z = −(K_{ν−1} + K_{ν+1})/2,

in any combination, for complex `z` anywhere on the principal branch (|arg z| < π).
It is the `besselk` counterpart of
[GeneralBesselj.jl](https://github.com/jwscook/GeneralBesselj.jl).

```julia
using GeneralBesselk, DualNumbers

besselk(1.5 + 2im, 0.7 - 0.3im)         # complex order and argument
besselkx(0.3 + 4im, -3.0 + 1im)         # scaled, left half plane
besselk(Dual(2.5, 1.0), 3.0)            # value and ∂K/∂ν (a real Dual)
besselk(1.2 + 1im, Dual(2.0 + 0im, 1))  # value and ∂K/∂z
besselk_v([0.5 + 1im, -0.5 - 1im], 2.0) # many orders at one z
```

## Types

All calls are type-stable, and `@inferred` checks every argument-type combination in the
tests. Computation is in `Float64`, and the result type depends only on the argument types:

* the result is real (`Float64` or `Dual{Float64}`) when the order and the argument are both
  real. As in SpecialFunctions, a real argument must then be non-negative (`DomainError`
  otherwise); pass a complex argument for the other side of the cut;
* otherwise the result is `ComplexF64` or `Dual{ComplexF64}`.

Real (non-Dual) orders go straight to SpecialFunctions, including a complex `ν` with zero
imaginary part. On the negative real axis the sign of `imag(z)`, including a signed zero,
picks the side of the cut.

## Method

With `Re ν ≥ 0` (K_{−ν} = K_ν), write `ν = μ + n` with `|Re μ| ≤ 1/2`:

| region | method |
|---|---|
| `\|z\| ≥ 17` | Hankel asymptotic expansion, used only if it converges to `eps` |
| `\|z\| ≤ 2` | Temme's series for `K_μ`, `K_{μ+1}` (uniform as `μ → 0`), then forward recurrence |
| `Re z ≥ 0` otherwise | Steed's method for Temme's CF2, then forward recurrence |
| `Re z < 0` | continuation from `w = −z`: `K_ν(z) = e^{∓iνπ}K_ν(w) ∓ iπI_ν(w)`, with `I_ν(w)` from CF1 and the Wronskian |

For complex orders, forward recurrence is not always stable: with large `|Im ν|` and `z`
off the positive real axis, `K_ν` can be the recessive solution. Every method therefore
returns an estimate of its own rounding-error amplification. The recurrence estimate
comes from a perturbation carried alongside it. If the estimate is poor, two other
candidates are tried: Steed at the full order, and the direct series
`K_ν = π/(2 sin νπ) (I_{−ν} − I_ν)`. The one with the best estimate is returned. There is
no arbitrary-precision fallback: every call runs a bounded amount of `Float64` work.

**Derivative in z.** By default this uses the symmetric form −(K_{ν−1} + K_{ν+1})/2, with
K_{ν−1} computed independently. For real ν and z > 0 its two terms add, so there is no
cancellation. The recurrence form (ν/z)K_ν − K_{ν+1} subtracts, with up to ~3× error
amplification. If K_{ν−1} falls in a harder regime than K_ν and K_{ν+1}, the error
estimates detect it and the recurrence form is used instead.

**Overflow.** Internally, values are carried as a mantissa and a power-of-two exponent,
and only combined at the end. Results beyond the `Float64` range become `±Inf` in the
correct quadrant (or 0), never `NaN`. Intermediate quantities that overflow, such as
K_{ν+1} when only K_ν is wanted, do not affect the result. `besselkx` stays finite where
`besselk` over- or underflows because of `e^{−z}`.

## Accuracy

Compared with mpmath on 490 stored reference values (`test/`) and two independent random
sets of 3000 and 1800 points (|ν| up to ~150, |Im ν| up to ~100, |z| from 0.01 to 3000,
including points within 1e-12 of the branch cut):

* median relative error ~1e-15, 99th percentile ~1e-13, for values, ∂K/∂z and ∂K/∂ν;
* real orders agree with SpecialFunctions (AMOS) to ~1e-13.

**Limitation:** about 0.3% of those random points lose accuracy. All of them combine
|Im ν| ≳ 8, z in the left half-plane near the imaginary axis, and |z| ≳ 30, where every
`Float64` method available here cancels badly. Errors there range from ~1e-11 up to
complete loss of accuracy at the extremes (|Im ν| ≈ 20–30, |z| ≈ 80–150).

## Cost

Calls don't allocate. On the 3000-point random set the median is ~1.3 μs (~3 μs with a
Dual order or argument), the 99th percentile ~20 μs (~60 μs), and the maximum ~30 μs
(~95 μs). This is a few times slower than AMOS for comparable real-order calls. Iteration
counts are capped, so no input can trigger an unbounded or arbitrary-precision computation.

## Notes

* `besselk_v` and `besselkx_v` reuse a result when consecutive orders are equal or opposite.
* Run the tests with `] test`. `test/gen_reference.py` regenerates the reference table
  (needs mpmath).

# GeneralBesselk.jl

The modified Bessel function of the second kind, `besselk(ν, z)`, and its scaled form
`besselkx(ν, z) = exp(z) * besselk(ν, z)`. This extends the SpecialFunctions methods to:

* **complex orders** `ν` (SpecialFunctions/AMOS only does real orders),
* **`DualNumbers.Dual` orders**: ∂K/∂ν is carried through the algorithm,
* **`DualNumbers.Dual` arguments**: uses ∂K/∂z = (ν/z)K_ν − K_{ν+1},

in any combination, for complex `z` anywhere on the principal branch (|arg z| < π).
It is the `besselk` counterpart of
[GeneralBesselj.jl](https://github.com/jwscook/GeneralBesselj.jl).

```julia
using GeneralBesselk, DualNumbers

besselk(1.5 + 2im, 0.7 - 0.3im)        # complex order and argument
besselkx(0.3 + 4im, -3.0 + 1im)        # scaled, left half plane
besselk(Dual(2.5, 1.0), 3.0)           # value and ∂K/∂ν (a real Dual)
besselk(1.2 + 1im, Dual(2.0 + 0im, 1)) # value and ∂K/∂z
besselk_v([0.5 + 1im, -0.5 - 1im], 2.0) # many orders at one z
```

Real (non-Dual) orders are passed straight to SpecialFunctions, including a complex `ν`
with zero imaginary part. On the negative real axis the sign of `imag(z)`, including a
signed zero, picks the side of the cut. A plain real negative `z` with a complex order
gives the value on the upper side.

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
`K_ν = π/(2 sin νπ) (I_{−ν} − I_ν)`. The best one is kept. If even that is worse than
about `1e-13`, the direct series is rerun in `BigFloat`, with the precision raised until
its own estimate is good. This happens only for extreme parameters, such as
`|Im ν| ≳ 20` with large `|z|` near the imaginary axis.

## Accuracy

Tested against mpmath on 490 reference values (stored in `test/`) and a separate 3000-point
random set with `|ν|` up to ~150 and `|z|` from 0.01 to 150:

* relative error: median ~1e-15, max ~2e-13, for both values and derivatives;
* real orders agree with SpecialFunctions (AMOS) to ~1e-13.

## Notes and limitations

* Computation is in `Float64`.
* Typical cost is 0.2–3 μs per call, a few times slower than AMOS for comparable real-order
  calls. The rare `BigFloat` fallback takes much longer.
* `besselk_v` reuses a result when consecutive orders are equal or opposite.
* Run tests with `] test`. `test/gen_reference.py` regenerates the reference table (needs mpmath).

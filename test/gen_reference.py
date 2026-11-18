# Regenerates reference_values.jl: run from the test/ directory with `python3 gen_reference.py` (needs mpmath)
import mpmath as mp, random, math, cmath
mp.mp.dps = 40
random.seed(12345)
rows = []
def add(nu, z):
    nu = complex(nu); z = complex(z)
    n = mp.mpc(nu); w = mp.mpc(z)
    K = mp.besselk(n, w)
    dK = mp.diff(lambda v: mp.besselk(v, w), n)
    rows.append((nu, z, complex(K), complex(dK)))
regions = [(0.05, 2.0), (2.0, 17.0), (17.0, 60.0)]
for i in range(400):
    lo, hi = random.choice(regions)
    r = math.exp(random.uniform(math.log(lo), math.log(hi)))
    th = random.uniform(-0.98, 0.98) * math.pi
    s = random.choice([1, 3, 10, 30])
    nu = complex(random.gauss(0, s), random.gauss(0, s / 2))
    add(nu, cmath.rect(r, th))
# near-integer and near-zero complex orders, imaginary orders, half-integers + i·small
for nu in [1e-12j, 1e-8 + 1e-9j, 2 + 1e-10j, -3 + 1e-6j, 0.5 + 1e-7j, 5j, 20j, 0.3 + 40j, 10.5 + 0.5j]:
    for z in [0.1, 1.0, 1.9, 2.1 + 0.3j, 5j, -4 + 1j, -4 - 1j, 10.0, 25 - 5j, -30 + 2j]:
        add(nu, z)
with open("reference_values.jl", "w") as f:
    f.write("# (ν, z, K_ν(z), ∂K_ν(z)/∂ν) generated with mpmath %s at 40 digits\n" % mp.__version__)
    f.write("const REFERENCE = [\n")
    for nu, z, K, dK in rows:
        f.write("  (" + ", ".join("complex(%r, %r)" % (x.real, x.imag) for x in (nu, z, K, dK)) + "),\n")
    f.write("]\n")
print(len(rows))

# reference_fast.jl: real orders at large |z| (the fast Hankel path), with the derivative
def gen_fast(n=500):
    random.seed(99)
    mp.mp.dps = 30
    with open("reference_fast.jl", "w") as f:
        f.write("const REFERENCE_FAST = [\n")
        for i in range(n):
            nu = random.choice([2, 3, random.randint(-10, 10), random.uniform(-40, 40)])
            r = math.exp(random.uniform(math.log(15), math.log(1e4))) if i % 2 else random.uniform(18, 60)
            z = cmath.rect(r, random.uniform(-0.5, 0.5) * math.pi)
            w = mp.mpc(z)
            kx = mp.exp(w) * mp.besselk(nu, w)
            dkx = kx - mp.exp(w) * (mp.besselk(nu - 1, w) + mp.besselk(nu + 1, w)) / 2
            f.write("  (%r, complex(%r, %r), complex(%r, %r), complex(%r, %r)),\n" % (
                float(nu), z.real, z.imag, float(kx.real), float(kx.imag), float(dkx.real), float(dkx.imag)))
        f.write("]\n")
gen_fast()

# reference_real.jl: real orders at small and moderate |z|, any angle, with ∂K/∂ν and ∂K/∂z
def gen_real(n=400):
    random.seed(777)
    mp.mp.dps = 30
    with open("reference_real.jl", "w") as f:
        f.write("const REFERENCE_REAL = [\n")
        for i in range(n):
            nu = random.choice([0.0, 0.5, 1.0, 2.0, random.randint(-12, 12), random.uniform(-3, 3), random.uniform(-40, 40)])
            r = math.exp(random.uniform(math.log(0.03), math.log(60)))
            th = random.choice([0.0, random.uniform(-0.5, 0.5), random.uniform(-0.99, 0.99)]) * math.pi
            z = cmath.rect(r, th)
            w = mp.mpc(z)
            K = mp.besselk(nu, w)
            Kn = mp.diff(lambda v: mp.besselk(v, w), nu)
            Kz = -(mp.besselk(nu - 1, w) + mp.besselk(nu + 1, w)) / 2
            f.write("  (%r, complex(%r, %r), complex(%r, %r), complex(%r, %r), complex(%r, %r)),\n" % (
                float(nu), z.real, z.imag, float(K.real), float(K.imag), float(Kn.real), float(Kn.imag),
                float(Kz.real), float(Kz.imag)))
        f.write("]\n")
gen_real()

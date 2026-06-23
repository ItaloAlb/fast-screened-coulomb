# Fast Screened Coulomb

**C library for screened electron–hole interaction potentials in layered dielectric systems.**

Implements the Electrostatic Transfer Matrix (ETM) method from  
Cavalcante et al., [*Phys. Rev. B* **97**, 125427 (2018)](https://doi.org/10.1103/PhysRevB.97.125427).

Computes V(ρ) — the screened Coulomb potential between two charge carriers
in a stack of dielectric slabs (van der Waals materials).
Designed for high-performance applications where V(ρ) is called millions of times per
second: the fitted Chebyshev expansion evaluates in **~0.01 µs per call** —
over **40,000× faster** than numerical integration.

---

## Physics summary

A point charge in a layered dielectric produces a screened potential that differs
substantially from the bare 1/ρ Coulomb form. The ETM method solves the Poisson
equation exactly for an arbitrary stack of N dielectric slabs by propagating
2×2 transfer matrices across each interface.

The real-space potential is the Fourier–Bessel integral:

```
V(ρ) = 2 ∫₀^∞ J₀(kρ) / ε(k)  dk          [Rydberg]
```

where ε(k) is the effective dielectric function computed from the transfer
matrices. For **direct excitons** (electron and hole in the same layer),
asymptotic subtraction accelerates convergence: ε(k→∞) = ε_c (the dielectric
constant of the source layer). For **indirect excitons** (different layers),
ε(k→∞) → ∞ and the integral converges naturally.

### Units

| Quantity | Unit |
|----------|------|
| Energy | Rydberg (1 Ry ≈ 13.606 eV) |
| Distance | Bohr radius a₀ (1 a₀ ≈ 0.529 Å) |

---

## Files

```
src/
  fast_screened_coulomb.h    Public API (C, callable from C++/Fortran)
  fast_screened_coulomb.c    Full implementation
  main.py                    Python reference implementation
tests/
  test_fsc.c                 TDD validation suite (135 tests)
  benchmark_fsc.c            CLI tool comparing all expansion methods
  _timing_full.py            Python timing comparison
  _timing_test.py            Python timing benchmarks
Makefile                     Build
README.md                    This file
```

---

## Dependencies

**None.** C99 standard library only (`<math.h>`, `<stdlib.h>`, `<string.h>`).
No BLAS, LAPACK, GSL, or other external libraries required. The Bessel function
J₀ is provided by a portable rational approximation — the system `j0()` is not
needed.

---

## Compilation

```sh
# Library + test suite
gcc -std=c99 -Wall -O2 -Isrc -o tests/test_fsc tests/test_fsc.c src/fast_screened_coulomb.c -lm
./tests/test_fsc

# Benchmark CLI
gcc -std=c99 -Wall -O2 -Isrc -o tests/benchmark_fsc tests/benchmark_fsc.c src/fast_screened_coulomb.c -lm
./tests/benchmark_fsc

# Or with make
make test
make bench
```

To use in your own project:
```sh
gcc -std=c99 -O2 -c src/fast_screened_coulomb.c
gcc -std=c99 -O2 your_qmc.c fast_screened_coulomb.o -lm
```

---

## Quick start

```c
#include "fast_screened_coulomb.h"

int main(void)
{
    /* 1. Define your dielectric stack (bottom → top) */
    double epsilons[] = {1.0, 14.0, 14.0, 1.0};  /* vacuum, MoS₂, MoS₂, vacuum */
    double d[]        = {-3.0, 3.0, 9.0};         /* interface positions (a₀) */

    /* 2. Create context */
    FSCContext *ctx = fsc_create(4, epsilons, d, 2, 0, 0.0);

    /* 3. Auto-tune integration parameters from tolerances */
    double k_max, rho_max; int n_quad;
    fsc_optimal_params(ctx, 1e-4, 1e-3, 1.5, &k_max, &n_quad, &rho_max);

    /* 4. Fit: "give me 0.5% accuracy on [0.1, rho_max] a₀" */
    FSCFit *V = fsc_fit_auto(ctx, 0.1, rho_max, 0.5, k_max, 1e-8);

    /* 5. Evaluate in your high-performance application inner loop (~0.01 µs/call) */
    for (int step = 0; step < n_mc_steps; step++) {
        double rho = distance_between_particles(...);
        double Vij = fsc_fit_eval(V, rho);
        energy += Vij;
    }

    /* 6. Clean up */
    fsc_fit_free(V);
    fsc_free(ctx);
    return 0;
}
```

---

## API reference

### Lifecycle

```c
FSCContext *fsc_create(int n_layers, const double *epsilons, const double *d,
                       int c, int t, double z_t);
void        fsc_free(FSCContext *ctx);
```

| Parameter | Description |
|-----------|-------------|
| `n_layers` | Number of dielectric layers N |
| `epsilons` | Dielectric constants [N] (bottom → top) |
| `d` | Interface positions [N−1]. Origin at centre of source layer |
| `c` | Source-charge layer (1-based) |
| `t` | Observation layer (1-based), or 0 for direct exciton (t = c) |
| `z_t` | z-coordinate of observation point (only used when t ≠ c) |

### Dielectric function

```c
double fsc_epsilon_k(const FSCContext *ctx, double k);
```

Returns ε(k) — the effective dielectric function (Eq. 3 of the paper).

### Automatic parameter search

```c
void fsc_optimal_params(const FSCContext *ctx,
                        double tol, double tol_V, double safety,
                        double *k_max_out, int *n_quad_out,
                        double *rho_max_out);
```

Automatically determines safe integration parameters from tolerances — no
manual tuning required. Scans k-space to find where the integrand envelope
decays below `tol`, then estimates the physical cutoff `rho_max` where
|V(ρ)| < `tol_V`. Uses an exponential + bisection search (~12 ε(k) evaluations)
instead of the Python reference's 200-point grid scan.

| Parameter | Typical value | Description |
|-----------|--------------|-------------|
| `tol` | 1e-4 | Target relative accuracy on the k-integration |
| `tol_V` | 1e-3 (≈ 14 meV) | Potential threshold below which V(ρ) ≈ 0 |
| `safety` | 1.5 | Multiplier on k_envelope (1.5× is standard) |

Works for direct and indirect excitons, any layer count, any dielectric contrast.
For the homogeneous limit (N=1) no scan is needed — returns immediately.

### Exact potential (slow — for reference/validation)

```c
double fsc_potential(const FSCContext *ctx, double rho,
                     double k_max, double tol);
void   fsc_potential_array(const FSCContext *ctx, const double *rhos, int n,
                           double *V_out, double k_max, double tol);
```

Computes V(ρ) via adaptive Gauss–Kronrod integration of the Fourier–Bessel
integral. Speed: ~0.4 ms/call. Use for generating reference data or debugging.

| Parameter | Typical value | Description |
|-----------|--------------|-------------|
| `k_max` | 20.0 (direct), 5.0 (indirect) | Upper integration cutoff (a₀⁻¹) |
| `tol` | 1e-8 | Integration tolerance |

### Fast evaluation (recommended for QMC)

```c
FSCFit *fsc_fit_auto(const FSCContext *ctx,
                     double rho_min, double rho_max,
                     double target_err_pct,
                     double k_max, double tol);
```

Automatically selects the Chebyshev degree to meet `target_err_pct`% error.
Sweeps degrees 3, 5, 7, 9, 11, 15, 20, 25, 30 and returns the smallest-degree
fit meeting the target.

```c
FSCFit *fsc_fit_chebyshev(const FSCContext *ctx, int degree,
                          double rho_min, double rho_max,
                          double k_max, double tol);
```

Manual Chebyshev fit of given degree. Evaluation uses Clenshaw recurrence —
the fastest method in compiled code (~0.01 µs/call).

```c
FSCFit *fsc_fit_bspline(const FSCContext *ctx, int n_interior_knots,
                        double rho_min, double rho_max,
                        double k_max, double tol);
```

Cubic B-spline fit on ρ·V(ρ). Best accuracy-per-parameter ratio. Useful when
you need <0.1% error.

```c
FSCFit *fsc_fit_legendre(const FSCContext *ctx, int degree, ...);
FSCFit *fsc_fit_pade(const FSCContext *ctx, int m, int n, ...);
FSCFit *fsc_fit_image_chg(const FSCContext *ctx, int n_images, ...);
```

Additional expansion methods (for comparison/benchmarking).

### Evaluating a fit

```c
double fsc_fit_eval(const FSCFit *fit, double rho);
void   fsc_fit_eval_array(const FSCFit *fit, const double *rhos, int n,
                          double *V_out);
```

`fsc_fit_eval` is thread-safe (read-only on the fit object, no global state,
zero allocation). Safe to call from OpenMP-parallel loops.

### Inspecting a fit

```c
FSCMethod   fsc_fit_method(const FSCFit *fit);
int         fsc_fit_n_params(const FSCFit *fit);
const char *fsc_fit_label(const FSCFit *fit);
void        fsc_fit_free(FSCFit *fit);
```

### Benchmark

```c
FSCBenchEntry *fsc_benchmark(const FSCContext *ctx,
                             double rho_min, double rho_max,
                             int n_fit_pts, int n_eval_pts,
                             const double *targets, int n_targets,
                             double k_max, double tol,
                             int *n_entries_out, FILE *fp);
void fsc_bench_free(FSCBenchEntry *entries);
```

Runs all five expansion methods, sweeps parameter counts, and reports which
method reaches each accuracy target with the fewest parameters. Outputs a
formatted table to `fp` (if non-NULL).

---

## Performance (bilayer MoS₂, ρ ∈ [0.1, 30] a₀)

| Method | Params | Max Error | Eval Time | Speedup vs Exact |
|--------|--------|-----------|-----------|------------------|
| Chebyshev d=7 | 8 | 0.13% | 0.02 µs | 19,500× |
| Legende d=7 | 8 | 0.13% | 0.04 µs | 11,300× |
| **B-spline k=5** | 9 | **0.07%** | 0.04 µs | 9,500× |
| Image chg n=12 | 12 | 0.15% | 0.07 µs | 6,100× |
| Pade [1/2] | 4 | 0.84% | 0.11 µs | 3,600× |
| Exact integral | — | — | ~400 µs | 1× |

Chebyshev is the default — it provides the best speed at all accuracy targets
above ~0.1%. B-spline should be used when <0.1% error is required.

---

## Indirect excitons (electron and hole in different layers)

```c
/* Source in layer 2, observation in layer 3, vertical offset z_t */
double z_t = d_a0;  /* interlayer separation in Bohr radii */
FSCContext *ctx = fsc_create(4, epsilons, d,
                              2,       /* c — source layer */
                              3,       /* t — observation layer */
                              z_t);    /* vertical separation */

/* fsc_optimal_params handles indirect automatically —
   ε(k) → ∞ ensures fast convergence, smaller k_max */
double k_max, rho_max; int n_quad;
fsc_optimal_params(ctx, 1e-4, 1e-3, 1.5, &k_max, &n_quad, &rho_max);
FSCFit *V = fsc_fit_auto(ctx, 0.1, rho_max, 0.5, k_max, 1e-8);
```

---

## Thread safety

```c
FSCFit *V = fsc_fit_auto(ctx, rho_min, rho_max, 0.5, k_max, tol);

#pragma omp parallel for
for (int i = 0; i < N; i++) {
    double Vij = fsc_fit_eval(V, rho[i]);  /* safe — read-only, no allocations */
}
```

`FSCFit` is immutable after fitting. Evaluate from any number of threads.

---

## Validation

The library is verified against the Python reference implementation (SciPy
adaptive quadrature). The TDD test suite (`test_fsc.c`) passes 135/135 tests:

| Test | Checks | Tolerance |
|------|--------|-----------|
| ε(k) vs exact N=3 formula | 6 k-points | 1e-10 |
| V(ρ) homogeneous limit | 4 ρ-points | 2e-5 |
| V(ρ) bilayer (direct) | 4 ρ-points | 1e-5 |
| V(ρ) bilayer (indirect) | 4 ρ-points | 1e-5 |
| B-spline fit | 10 ρ-points | 1% |
| Chebyshev fit | 10 ρ-points | 1% |
| Legendre fit | 10 ρ-points | 1% |
| Image charges fit | 10 ρ-points | 5% |
| Pade fit | 10 ρ-points | 3% |
| Auto-fit (Chebyshev default) | accuracy targets | meets target |
| Benchmark | all methods run | B-spline <1% |
| Edge cases | ρ→0, ρ→∞, N=1, high contrast | inf/finite/valid |
| **Auto-parameter search (N3, N4, indirect)** | k_max, rho_max, n_quad | analytic relation |
| **Safety factor & tol sensitivity** | doubling safety, tightening tol | exact scaling |
| **N=1 homogeneous** | single-layer edge case | exact formula |
| **Convergence sweep (4 diverse systems)** | 12 (ρ, system) pairs | 50× tol vs reference |
| **V(ρ) plateau vs k_max** | 3 ρ × 9 k_max values | error flatlines past k_envelope |
| **Adaptive GK vs dense Simpson** | 4 ρ-values, same k_max | 0.2% cross-method agreement |

---

## Edge cases

- **ρ → 0**: returns `HUGE_VAL` (∞)
- **N = 1** (homogeneous medium): ε(k) = ε_c, V(ρ) = 2/(ε_c·ρ)
- **High dielectric contrast** (ε ratios up to 100): safe — exponentials capped
  at exp(±500) to prevent float64 overflow

---

## References

1. Cavalcante, L. S. R., Chaves, A., Van Duppen, B., Peeters, F. M., &
   Reichman, D. R. (2018). Electrostatics of electron–hole interactions in
   van der Waals heterostructures. *Physical Review B*, 97(12), 125427.

2. Rytova, N. S. (1967). Screened potential of a point charge in a thin film.
   *Proceedings of Moscow Physical-Technical Institute*, 30–44.

3. Keldysh, L. V. (1979). Coulomb interaction in thin semiconductor and
   semimetal films. *JETP Letters*, 29(11), 658–661.

---

## License

Provided as-is for research use. See the paper reference above for the
underlying method.

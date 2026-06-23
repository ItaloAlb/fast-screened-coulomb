/* fast_screened_coulomb.h — Fast Screened Coulomb C library
 *
 * Electrostatic Transfer Matrix (ETM) method for screened electron-hole
 * interaction potentials in layered dielectric systems.
 *
 * Reference: Cavalcante et al., Phys. Rev. B 97, 125427 (2018).
 *
 * Units: Rydberg (energy), Bohr radii a₀ (distance).
 *
 * Dependencies: C99 standard library (<math.h> for exp, j0, fabs, sqrt).
 */

#ifndef FAST_SCREENED_COULOMB_H
#define FAST_SCREENED_COULOMB_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>   /* FILE* for benchmark output */

/* ── Opaque types ── */
typedef struct FSCContext FSCContext;
typedef struct FSCFit     FSCFit;

/* ── Expansion method identifier ── */
typedef enum {
    FSC_METHOD_PADE = 0,
    FSC_METHOD_CHEBYSHEV,
    FSC_METHOD_LEGENDRE,
    FSC_METHOD_IMAGE_CHARGES,
    FSC_METHOD_BSPLINE,
    FSC_METHOD_COUNT
} FSCMethod;

/* ── Benchmark result entry ── */
typedef struct {
    FSCMethod  method;
    char       label[32];
    int        n_params;
    double     max_err_pct;
    double     t_eval_us;
} FSCBenchEntry;

/* ─────────────────────────────────────────────────────────────────────
 *  Lifecycle
 * ───────────────────────────────────────────────────────────────────── */

/* Create a screened-Coulomb calculator for a dielectric stack.
 *
 *   n_layers : number of dielectric layers  N
 *   epsilons : dielectric constants [N]  (bottom → top)
 *   d        : interface positions [N-1]; origin at centre of source layer
 *   c        : source-charge layer (1-based)
 *   t        : observation layer (1-based), or 0 for direct exciton (t=c)
 *   z_t      : z-coordinate of observation point (only used when t ≠ c)
 */
FSCContext *fsc_create(int n_layers, const double *epsilons, const double *d,
                       int c, int t, double z_t);

void fsc_free(FSCContext *ctx);

/* ─────────────────────────────────────────────────────────────────────
 *  Dielectric function  ε(k)
 * ───────────────────────────────────────────────────────────────────── */

/* Effective dielectric function ε_{t,c}(k)  — Eq. (3) of the PRB paper.
 * Returns ε for the configured stack at wave-vector magnitude k (a₀⁻¹).
 */
double fsc_epsilon_k(const FSCContext *ctx, double k);

/* ─────────────────────────────────────────────────────────────────────
 *  Exact potential  V(ρ)  via Fourier–Bessel integration
 * ───────────────────────────────────────────────────────────────────── */

/* Compute V_eh(ρ) in Rydberg using adaptive Gauss-Kronrod integration.
 *
 *   ctx   : configured stack
 *   rho   : in-plane distance (a₀)
 *   k_max : upper integration cutoff (a₀⁻¹)
 *   tol   : relative integration tolerance
 *
 * For direct excitons asymptotic subtraction is used (ε(∞)=ε_c).
 * For indirect excitons the integral is direct (ε(∞)→∞ naturally).
 */
double fsc_potential(const FSCContext *ctx, double rho,
                     double k_max, double tol);

/* Evaluate V(ρ) at multiple points (batched for efficiency). */
void fsc_potential_array(const FSCContext *ctx, const double *rhos, int n,
                         double *V_out, double k_max, double tol);

/* Query / reset internal ε(k) evaluation counter (for benchmarking). */
long fsc_get_eval_count(void);
void fsc_reset_eval_count(void);

/* ─────────────────────────────────────────────────────────────────────
 *  Automatic parameter search
 * ───────────────────────────────────────────────────────────────────── */

/* Find optimal integration parameters (k_max, rho_max) from tolerances.
 *
 * Scans k-space to determine where the integrand envelope falls below tol,
 * and estimates the physical cutoff rho_max where |V(ρ)| < tol_V.
 *
 *   ctx      : configured stack
 *   tol      : target relative accuracy on k-integration (default 1e-4)
 *   tol_V    : potential threshold in Ry (default 1e-3 Ry ≈ 14 meV)
 *   safety   : safety factor applied to k_envelope (>1, default 1.5)
 *   k_max_out: recommended upper integration limit (a₀⁻¹)
 *   n_quad_out: recommended quadrature limit
 *   rho_max_out: physical cutoff where V(rho) < tol_V (a₀)
 *
 * For direct excitons: monitors |1/ε(k) - 1/ε_c| < tol/ε_c.
 * For indirect excitons: monitors 1/ε(k) < tol/ε_c. */
void fsc_optimal_params(const FSCContext *ctx,
                        double tol, double tol_V, double safety,
                        double *k_max_out, int *n_quad_out,
                        double *rho_max_out);

/* ─────────────────────────────────────────────────────────────────────
 *  Expansion fits  (fast evaluation for QMC)
 * ─────────────────────────────────────────────────────────────────────
 *
 *  RECOMMENDED DEFAULT: Chebyshev polynomial expansion (fsc_fit_chebyshev
 *  or the automatic fsc_fit_auto).  Chebyshev evaluation via Clenshaw
 *  recurrence is the fastest method in compiled code (~0.01–0.02 µs/pt),
 *  with competitive accuracy.  Use fsc_fit_auto() to select the minimal
 *  degree for a target error tolerance.
 */

/* Automatically fit a Chebyshev expansion to meet a target accuracy.
 *
 * Sweeps degrees 3,5,7,9,11,15,20,25,30 and returns the fit for the
 * smallest degree whose max relative error falls below target_err_pct.
 * If no degree meets the target, returns the best (degree=30) fit.
 *
 * If achieved_err_pct is non-NULL, it receives the actual max relative
 * error (in percent) of the returned fit.  Callers can compare this against
 * target_err_pct to detect when the target was not met.
 *
 * This is the recommended entry point for QMC: you specify the accuracy
 * you need and the library picks the cheapest expansion that achieves it. */
FSCFit *fsc_fit_auto(const FSCContext *ctx,
                     double rho_min, double rho_max,
                     double target_err_pct,
                     double k_max, double tol,
                     double *achieved_err_pct);

/* Fit a Pade [m/n] rational approximant on [rho_min, rho_max].
 *   V(ρ) ≈ (a₀ + a₁ρ + … + a_m·ρ^m) / (b₁ρ + b₂ρ² + … + b_n·ρ^n)
 * Uses Levenberg-Marquardt nonlinear least-squares. */
FSCFit *fsc_fit_pade(const FSCContext *ctx, int m, int n,
                     double rho_min, double rho_max,
                     double k_max, double tol);

/* Fit a Chebyshev polynomial expansion (degree d) to ρ·V(ρ).
 * Evaluates as V(ρ) = Σ cⱼ Tⱼ(x) / ρ  with x ∈ [-1,1].
 * Uses Clenshaw recurrence — typically ~0.01–0.02 µs/pt. */
FSCFit *fsc_fit_chebyshev(const FSCContext *ctx, int degree,
                          double rho_min, double rho_max,
                          double k_max, double tol);

/* Fit a Legendre polynomial expansion (degree d) to ρ·V(ρ). */
FSCFit *fsc_fit_legendre(const FSCContext *ctx, int degree,
                         double rho_min, double rho_max,
                         double k_max, double tol);

/* Fit image charges: V(ρ) = Σ qᵢ / √(ρ² + zᵢ²). */
FSCFit *fsc_fit_image_chg(const FSCContext *ctx, int n_images,
                          double rho_min, double rho_max,
                          double k_max, double tol);

/* Fit a cubic B-spline to ρ·V(ρ) on [rho_min, rho_max].
 *   n_interior_knots : interior knots (0 = minimal). Typical 5–20. */
FSCFit *fsc_fit_bspline(const FSCContext *ctx, int n_interior_knots,
                        double rho_min, double rho_max,
                        double k_max, double tol);

/* ── Evaluate any fit ── */
double fsc_fit_eval(const FSCFit *fit, double rho);
void   fsc_fit_eval_array(const FSCFit *fit, const double *rhos, int n,
                          double *V_out);

/* ── Inspect a fit ── */
FSCMethod   fsc_fit_method(const FSCFit *fit);
int         fsc_fit_n_params(const FSCFit *fit);
const char *fsc_fit_label(const FSCFit *fit);
void        fsc_fit_free(FSCFit *fit);

/* ─────────────────────────────────────────────────────────────────────
 *  Benchmark
 * ───────────────────────────────────────────────────────────────────── */

/* Run the full expansion benchmark (reproduces Python fit_benchmark).
 *
 * Returns a heap-allocated array of FSCBenchEntry; *n_entries_out is set.
 * If fp is non-NULL, a formatted table is also written to fp.
 */
FSCBenchEntry *fsc_benchmark(const FSCContext *ctx,
                             double rho_min, double rho_max,
                             int n_fit_pts, int n_eval_pts,
                             const double *targets, int n_targets,
                             double k_max, double tol,
                             int *n_entries_out, FILE *fp);

void fsc_bench_free(FSCBenchEntry *entries);

#ifdef __cplusplus
}
#endif

#endif /* FAST_SCREENED_COULOMB_H */

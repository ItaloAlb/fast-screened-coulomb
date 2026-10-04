/* fast_screened_coulomb.c — Implementation
 *
 * Electrostatic Transfer Matrix (ETM) method for screened electron-hole
 * interaction potentials in layered dielectric systems.
 *
 * Reference: Cavalcante et al., Phys. Rev. B 97, 125427 (2018).
 *
 * Implements:
 *   - Transfer-matrix engine (ε(k) for arbitrary dielectric stacks)
 *   - Adaptive Gauss-Kronrod G7/K15 integration for the Fourier–Bessel
 *     integral → exact V(ρ)
 *   - Five expansion methods: Pade, Chebyshev, Legendre, Image charges,
 *     B-spline (cubic)
 *   - Benchmark function comparing all methods
 */

#include "fast_screened_coulomb.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>    /* fprintf, snprintf */

/* ═════════════════════════════════════════════════════════════════════
 *  Portable Bessel J₀  (C99 j0() not available on some Windows toolchains)
 * ═════════════════════════════════════════════════════════════════════ */

/* Polynomial approximation of J₀(x) accurate to ~1e-8 for all x.
 * Uses rational approximation for x ≤ 8 and asymptotic expansion for x > 8.
 * Reference: Abramowitz & Stegun, 9.4.1, 9.4.3. */
static double bessel_j0(double x)
{
    double ax = fabs(x);
    if (ax < 1e-12) return 1.0;

    if (ax <= 8.0) {
        /* Rational approximation for |x| ≤ 8  (Hart, 1968) */
        double y = ax * ax;
        double num = 57568490574.0 + y * (-13362590354.0
                    + y * (651619640.7 + y * (-11214424.18
                    + y * (77392.33017 + y * (-184.9052456)))));
        double den = 57568490411.0 + y * (1029532985.0
                    + y * (9494680.718 + y * (59272.64853
                    + y * (267.8532712 + y * 1.0))));
        return num / den;
    } else {
        /* Asymptotic expansion for |x| > 8  (Abramowitz 9.4.3) */
        double z = 8.0 / ax;
        double y = z * z;
        double xx = ax - 0.785398164; /* x - π/4 */

        double P0 = 1.0 + y * (-0.001098628627 + y * (0.00002734510407
                        + y * (-0.000002073370639 + y * 2.09388711e-7)));
        double Q0 = -0.01562499995 + y * (0.0001430488765
                        + y * (-0.000006911147651 + y * (7.621095161e-7
                        + y * -9.34945152e-8)));

        return sqrt(0.636619772 / ax)   /* sqrt(2/π) */
               * (cos(xx) * P0 - z * sin(xx) * Q0);
    }
}

/* ═════════════════════════════════════════════════════════════════════
 *  Internal constants
 * ═════════════════════════════════════════════════════════════════════ */

#define MAX_DEPTH   60        /* max recursion depth for adaptive integrator */
#define MAX_PARAMS  80        /* max parameters in any fit */
#define MAX_FIT_PTS 600       /* max fitting points */
#define BSPLINE_ORDER 4       /* cubic B-spline */

/* ═════════════════════════════════════════════════════════════════════
 *  Internal types
 * ═════════════════════════════════════════════════════════════════════ */

struct FSCContext {
    int    n_layers;
    double *epsilons;    /* [n_layers] */
    double *d;           /* [n_layers-1] interface positions */
    int    c;            /* source layer (1-based) */
    int    t;            /* observation layer (1-based), never 0 internally */
    double z_t;
    double eps_c;        /* cached ε of source layer */
    int    is_direct;    /* t==c and z_t≈0 */
};

struct FSCFit {
    FSCMethod method;
    char      label[32];
    int       n_params;
    double    *param;    /* generic parameter array */
    /* domain */
    double    rho_min, rho_max;
    /* method-specific data */
    int       order;     /* B-spline order, or Pade [m,n] m=order, n=extra */
    int       extra;     /* Pade denominator degree, or Chebyshev/Legendre degree */
    double    *knots;    /* B-spline knot vector */
    int       n_knots;
    /* evaluation function pointer */
    double  (*eval)(const struct FSCFit *fit, double rho);
};

/* ═════════════════════════════════════════════════════════════════════
 *  ε(k): generalized reflection coefficients (numerically stable ETM)
 * ═════════════════════════════════════════════════════════════════════
 *
 * Same physics as the transfer-matrix Eqs. (4)–(7) of Cavalcante et al.,
 * PRB 97, 125427 (2018), reorganised so that EVERY exponential decays.
 *
 * Why: the transfer-matrix products multiply factors e^{+2k d_n} and cancel
 * them again later.  In double precision that cancellation loses everything
 * once k·d ≳ 20 (interlayer values came out wrong by 10^10, even negative),
 * and the raw exp(k d) overflows to NaN past k·d ≈ 709 — well inside the
 * k range a periodic (Ewald) sum needs.
 *
 * ── Potential "waves" ──────────────────────────────────────────────────
 * In each layer the Fourier-space potential is a sum of an up-going part
 * e^{-kz} (decaying upward) and a down-going part e^{+kz}.  At an interface
 * between ε_lo (below) and ε_hi (above), continuity of φ and of ε ∂φ/∂z gives
 * the single-interface reflection of an up-going wave
 *
 *     r = (ε_lo − ε_hi)/(ε_lo + ε_hi)            (the image-charge factor),
 *
 * and −r for a down-going one.  Stacking a layer of thickness t on top turns
 * that into the generalized reflection
 *
 *     R = (r + R' e^{−2kt}) / (1 + r R' e^{−2kt}),
 *
 * R' being the generalized reflection at the far side of that layer — the
 * standard recursion for layered media (e.g. Chew, "Waves and Fields in
 * Inhomogeneous Media", §2.1), here for the static (Laplace) case.
 *
 * ── Source layer ───────────────────────────────────────────────────────
 * Unit source at z = 0 in layer c, bounded by z_l < 0 < z_u:
 *     φ_c(z) = e^{−k|z|} + A e^{kz} + B e^{−kz}      (all over ε_c)
 * Reflection at z_u and z_l: A e^{k z_u} = R_u (1 + B) e^{−k z_u} and
 * B e^{−k z_l} = R_d (1 + A) e^{k z_l}.  With U = R_u e^{−2k z_u}, D = R_d e^{2k z_l}
 * (both ≤ 1 in magnitude, decaying):
 *     A = U (1 + D)/(1 − U D),    B = D (1 + U)/(1 − U D).
 *
 * ── Carrying φ into another layer ──────────────────────────────────────
 * Going up into layer n (bottom z_a, top z_b, R' its generalized reflection
 * at z_b, t = z_b − z_a), the potential at z_a fixes the up-going amplitude
 * P = φ(z_a)/(1 + R' e^{−2kt}), and inside the layer
 *     φ(z) = P [ e^{−k(z − z_a)} + R' e^{−k(2 z_b − z_a − z)} ],
 * both exponents ≤ 0.  Going down is the mirror image.
 *
 * ε_{t,c}(k) = ε_c / φ(z_t).  At large k every R e^{−2kt} underflows to 0, so
 * ε → ε_c for a same-layer pair and 1/ε → 0 (as e^{−k|z_t|}) across layers:
 * the right limits, reached without ever forming a large number. */

/* Generalized reflection of an UP-going wave at interface j (between layer j
 * below and j+1 above), 1-based j in [1, N-1], seen from layer j. */
static double reflect_up(const FSCContext *ctx, double k, int j)
{
    const int N = ctx->n_layers;
    double R = 0.0;                      /* nothing above the top interface */
    for (int i = N - 1; i >= j; i--) {
        const double lo = ctx->epsilons[i - 1], hi = ctx->epsilons[i];
        const double r  = (lo - hi) / (lo + hi);
        /* layer i+1 lies between interfaces i and i+1 (absent for i = N-1) */
        const double x  = (i < N - 1)
                          ? R * exp(-2.0 * k * (ctx->d[i] - ctx->d[i - 1])) : 0.0;
        R = (r + x) / (1.0 + r * x);
    }
    return R;
}

/* Generalized reflection of a DOWN-going wave at interface j, seen from
 * layer j+1. */
static double reflect_down(const FSCContext *ctx, double k, int j)
{
    double R = 0.0;                      /* nothing below the bottom interface */
    for (int i = 1; i <= j; i++) {
        const double lo = ctx->epsilons[i - 1], hi = ctx->epsilons[i];
        const double r  = (hi - lo) / (lo + hi);
        /* layer i lies between interfaces i-1 and i (absent for i = 1) */
        const double x  = (i > 1)
                          ? R * exp(-2.0 * k * (ctx->d[i - 1] - ctx->d[i - 2])) : 0.0;
        R = (r + x) / (1.0 + r * x);
    }
    return R;
}

double fsc_epsilon_k(const FSCContext *ctx, double k)
{
    if (k < 1e-15) k = 1e-15;

    const int    N = ctx->n_layers, c = ctx->c, t = ctx->t;
    const double *d = ctx->d;
    if (N == 1) return ctx->eps_c;

    /* Source layer: its bounding interfaces and the reflections off them. */
    const int    has_up = c < N, has_dn = c > 1;
    const double z_u = has_up ? d[c - 1] : 0.0;
    const double z_l = has_dn ? d[c - 2] : 0.0;
    const double Ru  = has_up ? reflect_up(ctx, k, c) : 0.0;
    const double Rd  = has_dn ? reflect_down(ctx, k, c - 1) : 0.0;
    const double U   = has_up ? Ru * exp(-2.0 * k * z_u) : 0.0;
    const double D   = has_dn ? Rd * exp(2.0 * k * z_l) : 0.0;
    const double den = 1.0 - U * D;
    const double A   = U * (1.0 + D) / den;
    const double B   = D * (1.0 + U) / den;

    double phi;
    if (t == c) {
        /* e^{kz} and e^{-kz} are bounded by the reflections that multiply
         * them: A e^{kz} = U(...) e^{k z}, U carrying e^{-2k z_u}. */
        const double z = ctx->z_t;
        phi = exp(-k * fabs(z)) + A * exp(k * z) + B * exp(-k * z);
    } else if (t > c) {
        /* φ at the top of the source layer, then up through layers c+1..t. */
        double phi_a = (1.0 + B) * exp(-k * z_u) * (1.0 + Ru);
        phi = 0.0;
        for (int n = c + 1; n <= t; n++) {
            const double za   = d[n - 2];
            const int    top  = n < N;
            const double zb   = top ? d[n - 1] : 0.0;
            const double Rp   = top ? reflect_up(ctx, k, n) : 0.0;
            const double e2t  = top ? exp(-2.0 * k * (zb - za)) : 0.0;
            const double P    = phi_a / (1.0 + Rp * e2t);
            if (n == t) {
                const double z = ctx->z_t;
                phi = P * (exp(-k * (z - za)) +
                           (top ? Rp * exp(-k * (2.0 * zb - za - z)) : 0.0));
            } else {
                phi_a = P * exp(-k * (zb - za)) * (1.0 + Rp);
            }
        }
    } else {
        /* Mirror: φ at the bottom of the source layer, then down. */
        double phi_b = (1.0 + A) * exp(k * z_l) * (1.0 + Rd);
        phi = 0.0;
        for (int n = c - 1; n >= t; n--) {
            const double zb   = d[n - 1];
            const int    bot  = n > 1;
            const double za   = bot ? d[n - 2] : 0.0;
            const double Rp   = bot ? reflect_down(ctx, k, n - 1) : 0.0;
            const double e2t  = bot ? exp(-2.0 * k * (zb - za)) : 0.0;
            const double P    = phi_b / (1.0 + Rp * e2t);
            if (n == t) {
                const double z = ctx->z_t;
                phi = P * (exp(k * (z - zb)) +
                           (bot ? Rp * exp(-k * (z - 2.0 * za + zb)) : 0.0));
            } else {
                phi_b = P * exp(-k * (zb - za)) * (1.0 + Rp);
            }
        }
    }
    return ctx->eps_c / phi;
}

/* ═════════════════════════════════════════════════════════════════════
 *  Adaptive Gauss-Kronrod G7/K15 integrator
 * ═════════════════════════════════════════════════════════════════════ */

/* Gauss nodes (positive half, including centre).  Nodes on [-1,1]
 * are ±x_i with weight w_i.  f(0) gets w_0, f(±x_i) each get w_i. */

static const double g7_x[4] = {
    0.0,
    0.4058451513773971669,
    0.7415311855993944399,
    0.9491079123427585245
};
static const double g7_w[4] = {
    0.4179591836734693878,
    0.3818300505051189449,
    0.2797053914892766679,
    0.1294849661688696933
};

static const double k15_x[8] = {
    0.0,
    0.2077849550078984676,
    0.4058451513773971669,
    0.5860872354676911303,
    0.7415311855993944399,
    0.8648644233597690728,
    0.9491079123427585245,
    0.9914553711208126392
};
static const double k15_w[8] = {
    0.2094821410847278280,
    0.2044329400752988924,
    0.1903505780647854099,
    0.1690047266392679028,
    0.1406532597152589187,
    0.1047900103222501838,
    0.0630920926299785533,
    0.0229353220105292250
};

typedef double (*integrand_fn)(double x, void *params);

/* Evaluate G7 integral on interval [a, b] */
static double g7_quad(integrand_fn f, void *p, double a, double b)
{
    double mid = 0.5 * (a + b);
    double half = 0.5 * (b - a);
    double s = g7_w[0] * f(mid, p);
    for (int i = 1; i < 4; i++) {
        double x = half * g7_x[i];
        s += g7_w[i] * (f(mid + x, p) + f(mid - x, p));
    }
    return half * s;
}

/* Evaluate K15 integral on interval [a, b] */
static double k15_quad(integrand_fn f, void *p, double a, double b)
{
    double mid = 0.5 * (a + b);
    double half = 0.5 * (b - a);
    double s = k15_w[0] * f(mid, p);
    for (int i = 1; i < 8; i++) {
        double x = half * k15_x[i];
        s += k15_w[i] * (f(mid + x, p) + f(mid - x, p));
    }
    return half * s;
}

/* Recursive adaptive Gauss-Kronrod integration.
 *   original_len : length of the full integration interval (for tol scaling)
 */
static double gk_adaptive(integrand_fn f, void *p,
                          double a, double b, double tol,
                          double original_len, int depth)
{
    double Ig7  = g7_quad(f, p, a, b);
    double Ik15 = k15_quad(f, p, a, b);
    double err  = fabs(Ik15 - Ig7);

    double scaled_tol = tol * (b - a) / original_len;
    if (err < scaled_tol || depth >= MAX_DEPTH) {
        return Ik15;
    }

    double mid = 0.5 * (a + b);
    return gk_adaptive(f, p, a, mid, tol, original_len, depth + 1) +
           gk_adaptive(f, p, mid, b, tol, original_len, depth + 1);
}

/* ═════════════════════════════════════════════════════════════════════
 *  Fourier–Bessel integrands
 * ═════════════════════════════════════════════════════════════════════ */

typedef struct {
    const FSCContext *ctx;
    double rho;
} fb_params;

static long fsc_eval_counter = 0;   /* counts fsc_epsilon_k calls */

long fsc_get_eval_count(void) { return fsc_eval_counter; }
void fsc_reset_eval_count(void) { fsc_eval_counter = 0; }

/* Direct exciton: (1/ε − 1/ε_c) · J₀(kρ) */
static double fb_direct(double k, void *p)
{
    fb_params *fp = (fb_params *)p;
    double ks = (k < 1e-15) ? 1e-15 : k;
    double ek = fsc_epsilon_k(fp->ctx, ks);
    double eps_c = fp->ctx->eps_c;
    fsc_eval_counter++;
    return (1.0 / ek - 1.0 / eps_c) * bessel_j0(ks * fp->rho);
}

/* General / indirect: J₀(kρ) / ε(k) */
static double fb_general(double k, void *p)
{
    fb_params *fp = (fb_params *)p;
    double ks = (k < 1e-15) ? 1e-15 : k;
    double ek = fsc_epsilon_k(fp->ctx, ks);
    fsc_eval_counter++;
    return bessel_j0(ks * fp->rho) / ek;
}

/* ═════════════════════════════════════════════════════════════════════
 *  Public: exact V(ρ)
 * ═════════════════════════════════════════════════════════════════════ */

double fsc_potential(const FSCContext *ctx, double rho,
                     double k_max, double tol)
{
    if (rho < 1e-15) return HUGE_VAL;

    fb_params fp;
    fp.ctx = ctx;
    fp.rho = rho;

    double integral;

    if (ctx->is_direct) {
        /* asymptotic subtraction: V = 2[∫(1/ε−1/ε_c)J₀ dk + 1/(ε_c·ρ)] */
        integral = gk_adaptive(fb_direct, &fp, 0.0, k_max,
                               tol, k_max, 0);
        return 2.0 * (integral + 1.0 / (ctx->eps_c * rho));
    } else {
        /* direct integration: ε(∞)→∞, so 1/ε→0 naturally */
        integral = gk_adaptive(fb_general, &fp, 0.0, k_max,
                               tol, k_max, 0);
        return 2.0 * integral;
    }
}

void fsc_potential_array(const FSCContext *ctx, const double *rhos, int n,
                         double *V_out, double k_max, double tol)
{
    for (int i = 0; i < n; i++) {
        V_out[i] = fsc_potential(ctx, rhos[i], k_max, tol);
    }
}

/* ═════════════════════════════════════════════════════════════════════
 *  Linear algebra: Cholesky decomposition + solver
 * ═════════════════════════════════════════════════════════════════════ */

/* In-place Cholesky A = L·Lᵀ (lower triangle stored).
 * A is n×n symmetric positive definite, row-major.
 * Returns 0 on success, -1 if not SPD. */
static int cholesky_decomp(double *A, int n)
{
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int k = 0; k < j; k++) {
            sum += A[j*n + k] * A[j*n + k];
        }
        double diag = A[j*n + j] - sum;
        if (diag <= 0.0) return -1;
        A[j*n + j] = sqrt(diag);
        double inv_diag = 1.0 / A[j*n + j];
        for (int i = j + 1; i < n; i++) {
            sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += A[i*n + k] * A[j*n + k];
            }
            A[i*n + j] = (A[i*n + j] - sum) * inv_diag;
        }
    }
    return 0;
}

/* Solve L·Lᵀ·x = b after Cholesky. L is lower-triangular (stored in A).
 * b is overwritten with x. */
static void cholesky_solve(const double *A, double *b, int n)
{
    /* Forward substitution: L·y = b */
    for (int i = 0; i < n; i++) {
        double sum = b[i];
        for (int k = 0; k < i; k++) {
            sum -= A[i*n + k] * b[k];
        }
        b[i] = sum / A[i*n + i];
    }
    /* Back substitution: Lᵀ·x = y */
    for (int i = n - 1; i >= 0; i--) {
        double sum = b[i];
        for (int k = i + 1; k < n; k++) {
            sum -= A[k*n + i] * b[k];
        }
        b[i] = sum / A[i*n + i];
    }
}

/* Solve normal equations:  (AᵀA)·x = Aᵀy
 * A is m×n (m data points, n parameters), y is [m].
 * Result x is written to y[0..n-1] (in-place with workspace).
 * Workspace w[n*n + n] must be provided. */
static int solve_normal_eqns(const double *A, double *y, int m, int n,
                             double *workspace)
{
    double *ATA = workspace;       /* n×n */
    double *ATy = workspace + n*n; /* n */

    /* Form AᵀA and Aᵀy */
    memset(ATA, 0, n * n * sizeof(double));
    memset(ATy, 0, n * sizeof(double));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double a_ij = A[i*n + j];
            ATy[j] += a_ij * y[i];
            for (int k = 0; k <= j; k++) {
                ATA[j*n + k] += a_ij * A[i*n + k];
            }
        }
    }
    /* Fill upper triangle */
    for (int j = 0; j < n; j++) {
        for (int k = j + 1; k < n; k++) {
            ATA[j*n + k] = ATA[k*n + j];
        }
    }

    /* Regularisation for near-singular systems */
    double reg = 1e-12 * ATA[0]; /* scale to diagonal magnitude */
    for (int i = 0; i < n; i++) ATA[i*n + i] += reg;

    if (cholesky_decomp(ATA, n) != 0) return -1;

    memcpy(y, ATy, n * sizeof(double));
    cholesky_solve(ATA, y, n);
    return 0;
}

/* ═════════════════════════════════════════════════════════════════════
 *  Helper: generate log-spaced fitting grid
 * ═════════════════════════════════════════════════════════════════════ */

/* Map ρ ∈ [rho_min, rho_max] → x ∈ [-1, 1] with a log stretch so that
 * log-spaced sample points become uniformly spaced in x.  This gives the
 * Chebyshev/Legendre basis uniform resolving power across ln(ρ) rather
 * than concentrating it at large ρ (the linear-map pathology). */
static double rho_to_x_log(double rho, double rho_min, double rho_max)
{
    double log_ratio = log(rho_max / rho_min);
    return 2.0 * log(rho / rho_min) / log_ratio - 1.0;
}

/* Inverse of rho_to_x_log: x ∈ [-1,1] → ρ ∈ [rho_min, rho_max] */
static double x_to_rho_log(double x, double rho_min, double rho_max)
{
    double log_ratio = log(rho_max / rho_min);
    return rho_min * exp(0.5 * (x + 1.0) * log_ratio);
}

/* Generate fitting grid at Chebyshev nodes in the log-mapped coordinate.
 * This gives optimal conditioning for the Chebyshev/Legendre basis
 * (discrete orthogonality) and clusters points at the endpoints where
 * the function has the most structure. */
static void cheb_fit_grid(double *rhos, int n, double rho_min, double rho_max)
{
    double pi = 3.14159265358979323846;
    for (int i = 0; i < n; i++) {
        /* Chebyshev nodes of the first kind (roots of T_n) */
        double u = cos(pi * (i + 0.5) / n);
        rhos[i] = x_to_rho_log(u, rho_min, rho_max);
    }
}

static void logspace(double *out, double start, double stop, int n)
{
    double log_start = log(start);
    double log_stop  = log(stop);
    double step = (log_stop - log_start) / (n - 1.0);
    for (int i = 0; i < n; i++) {
        out[i] = exp(log_start + step * i);
    }
}

/* ═════════════════════════════════════════════════════════════════════
 *  Helper: compute V_exact on a grid (used by all fit methods)
 * ═════════════════════════════════════════════════════════════════════ */

static void compute_V_grid(const FSCContext *ctx,
                           const double *rhos, int n,
                           double *V, double k_max, double tol)
{
    fsc_potential_array(ctx, rhos, n, V, k_max, tol);
}

/* ═════════════════════════════════════════════════════════════════════
 *  Public: automatic parameter search
 * ═════════════════════════════════════════════════════════════════════ */

void fsc_optimal_params(const FSCContext *ctx,
                        double tol, double tol_V, double safety,
                        double *k_max_out, int *n_quad_out,
                        double *rho_max_out)
{
    /* ── helper: integrand envelope at k ── */
    int    is_direct = ctx->is_direct;
    double eps_c     = ctx->eps_c;
    double target    = tol / eps_c;

    /* ── exponential search: find k where envelope drops below target ──
     * Since dev(k) is monotonic decreasing, double k until we find the
     * threshold, then refine with bisection.  Typically ~10–15 ε(k)
     * evaluations instead of a 100-point linear scan. */
    double k_envelope = 0.0;
    double k_lo = 0.001, k_hi;

    /* phase 1: exponential probe — find an upper bound */
    double k_probe = 0.01;
    int found = 0;
    for (int step = 0; step < 20; step++) {
        double ek  = fsc_epsilon_k(ctx, k_probe);
        double dev = is_direct ? fabs(1.0 / ek - 1.0 / eps_c) : (1.0 / ek);
        if (dev < target) {
            k_hi  = k_probe;
            found = 1;
            break;
        }
        k_lo = k_probe;
        k_probe *= 2.0;
    }
    if (!found) {
        k_hi = k_probe;   /* fallback: use largest k probed */
    }

    /* phase 2: bisection refinement (5 iterations → 3% precision) */
    for (int iter = 0; iter < 8; iter++) {
        double k_mid = 0.5 * (k_lo + k_hi);
        double ek    = fsc_epsilon_k(ctx, k_mid);
        double dev   = is_direct ? fabs(1.0 / ek - 1.0 / eps_c) : (1.0 / ek);
        if (dev < target) {
            k_hi = k_mid;
        } else {
            k_lo = k_mid;
        }
    }
    k_envelope = k_hi;

    *k_max_out  = k_envelope * safety;
    *n_quad_out = 200;

    /* ── estimate rho_max from asymptotic V(ρ) ~ 2/(ε₀·ρ) ── */
    double eps_0 = fsc_epsilon_k(ctx, 1e-4);
    if (eps_0 < 1e-15) eps_0 = eps_c;  /* fallback */
    *rho_max_out = 2.0 / (eps_0 * tol_V);
}

/* ═════════════════════════════════════════════════════════════════════
 *  Fit method: B-spline (cubic, order=4)
 * ═════════════════════════════════════════════════════════════════════ */

/* Find knot interval: largest i such that knots[i] <= x < knots[i+1].
 * For x == knots[n_knots-1], return n_knots - order - 1. */
static int bspline_find_interval(double x, const double *knots, int n_knots,
                                 int order)
{
    if (x >= knots[n_knots - 1]) return n_knots - order - 1;
    if (x <= knots[0]) return order - 1;
    /* Binary search */
    int lo = 0, hi = n_knots - 2;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (knots[mid] <= x) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

/* Evaluate one B-spline basis function B_{i,order}(x) via Cox–de Boor recurrence. */
static double bspline_basis(int i, int k, double x, const double *knots)
{
    if (k == 1) {
        return (knots[i] <= x && x < knots[i+1]) ? 1.0 : 0.0;
    }
    double v = 0.0;
    double denom1 = knots[i+k-1] - knots[i];
    if (denom1 > 1e-30) {
        v += (x - knots[i]) / denom1 * bspline_basis(i, k-1, x, knots);
    }
    double denom2 = knots[i+k] - knots[i+1];
    if (denom2 > 1e-30) {
        v += (knots[i+k] - x) / denom2 * bspline_basis(i+1, k-1, x, knots);
    }
    return v;
}

/* de Boor algorithm for evaluating a B-spline S(x) = Σ ctrl[i]·B_{i,k}(x). */
static double bspline_deboor(double x, int order, const double *knots,
                             int n_ctrl, const double *ctrl)
{
    if (x <= knots[0]) return ctrl[0];
    if (x >= knots[n_ctrl + order - 1]) return ctrl[n_ctrl - 1];

    int i = bspline_find_interval(x, knots, n_ctrl + order, order);

    double d[10]; /* order <= 8 is safe; we use 4 */
    for (int j = 0; j < order; j++) {
        d[j] = ctrl[i - order + 1 + j];
    }

    for (int r = 1; r < order; r++) {
        for (int j = order - 1; j >= r; j--) {
            int idx = i - order + 1 + j;
            double alpha = (x - knots[idx]) /
                           (knots[idx + order - r] - knots[idx] + 1e-30);
            d[j] = (1.0 - alpha) * d[j-1] + alpha * d[j];
        }
    }
    return d[order - 1];
}

/* Build clamped knot vector for log-spaced data. */
static void bspline_make_knots(double *knots, int n_knots, int order,
                               const double *x_data, int n_data,
                               double x_min, double x_max)
{
    int n_interior = n_knots - 2 * order;
    /* Clamped at ends: first `order` knots at x_min */
    for (int i = 0; i < order; i++) knots[i] = x_min;
    /* Interior knots at quantiles */
    if (n_interior > 0) {
        for (int j = 0; j < n_interior; j++) {
            double pct = (j + 1.0) / (n_interior + 1.0);
            int idx = (int)(pct * (n_data - 1));
            if (idx < 0) idx = 0;
            if (idx >= n_data) idx = n_data - 1;
            knots[order + j] = x_data[idx];
        }
    }
    /* Clamped at end */
    for (int i = order + n_interior; i < n_knots; i++) knots[i] = x_max;
}

static double bspline_eval_wrapper(const FSCFit *fit, double rho)
{
    return bspline_deboor(rho, fit->order, fit->knots,
                          fit->n_params, fit->param) / rho;
}

static FSCFit *fit_bspline(const FSCContext *ctx,
                           double rho_min, double rho_max,
                           int n_interior_knots,
                           double k_max, double tol)
{
    int order = BSPLINE_ORDER;
    int n_knots = 2 * order + n_interior_knots;
    int n_ctrl = n_knots - order;  /* = order + n_interior_knots */
    int n_fit = (n_ctrl * 8 > MAX_FIT_PTS) ? MAX_FIT_PTS : n_ctrl * 8;
    if (n_fit < 50) n_fit = 50;

    double *rhos  = malloc(n_fit * sizeof(double));
    double *V     = malloc(n_fit * sizeof(double));
    double *rhoV  = malloc(n_fit * sizeof(double));
    double *knots = malloc(n_knots * sizeof(double));
    double *A     = malloc((size_t)n_fit * n_ctrl * sizeof(double));
    double *ws    = malloc((size_t)(n_ctrl*n_ctrl + n_ctrl) * sizeof(double));

    FSCFit *fit = NULL;
    if (!rhos || !V || !rhoV || !knots || !A || !ws) goto cleanup;

    /* Generate fitting grid */
    logspace(rhos, rho_min, rho_max, n_fit);
    compute_V_grid(ctx, rhos, n_fit, V, k_max, tol);
    for (int i = 0; i < n_fit; i++) rhoV[i] = rhos[i] * V[i];

    /* Knot vector */
    bspline_make_knots(knots, n_knots, order, rhos, n_fit, rho_min, rho_max);

    /* Design matrix */
    for (int i = 0; i < n_fit; i++) {
        for (int j = 0; j < n_ctrl; j++) {
            A[i*n_ctrl + j] = bspline_basis(j, order, rhos[i], knots);
        }
    }

    /* Solve normal equations */
    if (solve_normal_eqns(A, rhoV, n_fit, n_ctrl, ws) != 0) goto cleanup;

    /* Build fit object */
    fit = malloc(sizeof(FSCFit));
    if (!fit) goto cleanup;
    fit->method   = FSC_METHOD_BSPLINE;
    snprintf(fit->label, sizeof(fit->label), "B-spline k=%d", n_interior_knots);
    fit->n_params = n_ctrl;
    fit->param    = malloc(n_ctrl * sizeof(double));
    fit->order    = order;
    fit->extra    = n_interior_knots;
    fit->knots    = knots;  knots = NULL;
    fit->n_knots  = n_knots;
    fit->rho_min  = rho_min;
    fit->rho_max  = rho_max;
    fit->eval     = bspline_eval_wrapper;
    if (!fit->param) { free(fit); fit = NULL; goto cleanup; }
    memcpy(fit->param, rhoV, n_ctrl * sizeof(double));

cleanup:
    free(rhos); free(V); free(rhoV); free(knots); free(A); free(ws);
    return fit;
}

/* ═════════════════════════════════════════════════════════════════════
 *  Fit method: Chebyshev
 * ═════════════════════════════════════════════════════════════════════ */

/* Evaluate Chebyshev sum via direct recurrence (matching Legendre pattern) */
static double cheb_eval(double x, const double *c, int n)
{
    /* Σ_{j=0}^{n-1} c[j] T_j(x) */
    if (n == 0) return 0.0;
    double s = c[0];          /* T_0 = 1 */
    if (n == 1) return s;
    double T_prev = 1.0;      /* T_0 */
    double T_cur  = x;        /* T_1 */
    s += c[1] * T_cur;
    for (int j = 2; j < n; j++) {
        double T_next = 2.0 * x * T_cur - T_prev;
        s += c[j] * T_next;
        T_prev = T_cur;
        T_cur  = T_next;
    }
    return s;
}

static double cheb_fit_eval_wrapper(const FSCFit *fit, double rho)
{
    /* x ∈ [-1, 1] via log-stretched map — matches the fitting grid */
    double x = rho_to_x_log(rho, fit->rho_min, fit->rho_max);
    double rhoV = cheb_eval(x, fit->param, fit->n_params);
    return rhoV / rho;
}

static FSCFit *fit_chebyshev(const FSCContext *ctx, int degree,
                             double rho_min, double rho_max,
                             double k_max, double tol)
{
    int n_params = degree + 1;
    int n_fit = n_params * 10;
    if (n_fit < 30) n_fit = 30;

    double *rhos = malloc(n_fit * sizeof(double));
    double *V    = malloc(n_fit * sizeof(double));
    double *rhoV = malloc(n_fit * sizeof(double));
    double *A    = malloc((size_t)n_fit * n_params * sizeof(double));
    double *ws   = malloc((size_t)(n_params*n_params + n_params) * sizeof(double));

    FSCFit *fit = NULL;
    if (!rhos || !V || !rhoV || !A || !ws) goto cleanup;

    /* Generate fitting grid at Chebyshev nodes in log-mapped coordinate.
     * This clusters points near the endpoints (±1 → rho_min, rho_max) where
     * Chebyshev basis has the most resolving power, and gives a
     * well-conditioned design matrix (discrete orthogonality). */
    cheb_fit_grid(rhos, n_fit, rho_min, rho_max);
    compute_V_grid(ctx, rhos, n_fit, V, k_max, tol);
    for (int i = 0; i < n_fit; i++) rhoV[i] = rhos[i] * V[i];

    /* Build design matrix: A[i][j] = T_j(x_i)  with x_i in [-1,1] */
    for (int i = 0; i < n_fit; i++) {
        double x = rho_to_x_log(rhos[i], rho_min, rho_max);
        /* Chebyshev basis: T_0=1, T_1=x, T_{j+1}=2x·T_j - T_{j-1} */
        if (n_params >= 1) A[i*n_params + 0] = 1.0;   /* T_0 */
        if (n_params >= 2) A[i*n_params + 1] = x;     /* T_1 */
        double T_prev = 1.0, T_cur = x;
        for (int j = 2; j < n_params; j++) {
            double T_next = 2.0 * x * T_cur - T_prev;
            A[i*n_params + j] = T_next;
            T_prev = T_cur;
            T_cur = T_next;
        }
    }

    if (solve_normal_eqns(A, rhoV, n_fit, n_params, ws) != 0) goto cleanup;

    fit = malloc(sizeof(FSCFit));
    if (!fit) goto cleanup;
    fit->method   = FSC_METHOD_CHEBYSHEV;
    snprintf(fit->label, sizeof(fit->label), "Chebyshev d=%d", degree);
    fit->n_params = n_params;
    fit->param    = malloc(n_params * sizeof(double));
    fit->order    = 0;
    fit->extra    = degree;
    fit->knots    = NULL;
    fit->n_knots  = 0;
    fit->rho_min  = rho_min;
    fit->rho_max  = rho_max;
    fit->eval     = cheb_fit_eval_wrapper;
    if (!fit->param) { free(fit); fit = NULL; goto cleanup; }
    memcpy(fit->param, rhoV, n_params * sizeof(double));

cleanup:
    free(rhos); free(V); free(rhoV); free(A); free(ws);
    return fit;
}

/* ═════════════════════════════════════════════════════════════════════
 *  Fit method: Legendre
 * ═════════════════════════════════════════════════════════════════════ */

static double legendre_eval(double x, const double *c, int n)
{
    /* Σ c[j] P_j(x) using recurrence: P_0=1, P_1=x,
     * j·P_j = (2j-1)·x·P_{j-1} - (j-1)·P_{j-2} */
    if (n == 0) return 0.0;
    double s = c[0];
    if (n == 1) return s;
    double P_prev = 1.0;
    double P_cur  = x;
    s += c[1] * P_cur;
    for (int j = 2; j < n; j++) {
        double P_next = ((2.0*j - 1.0) * x * P_cur - (j - 1.0) * P_prev) / j;
        s += c[j] * P_next;
        P_prev = P_cur;
        P_cur  = P_next;
    }
    return s;
}

static double legendre_fit_eval_wrapper(const FSCFit *fit, double rho)
{
    double x = rho_to_x_log(rho, fit->rho_min, fit->rho_max);
    double rhoV = legendre_eval(x, fit->param, fit->n_params);
    return rhoV / rho;
}

static FSCFit *fit_legendre(const FSCContext *ctx, int degree,
                            double rho_min, double rho_max,
                            double k_max, double tol)
{
    int n_params = degree + 1;
    int n_fit = n_params * 10;
    if (n_fit < 30) n_fit = 30;

    double *rhos = malloc(n_fit * sizeof(double));
    double *V    = malloc(n_fit * sizeof(double));
    double *rhoV = malloc(n_fit * sizeof(double));
    double *A    = malloc((size_t)n_fit * n_params * sizeof(double));
    double *ws   = malloc((size_t)(n_params*n_params + n_params) * sizeof(double));

    FSCFit *fit = NULL;
    if (!rhos || !V || !rhoV || !A || !ws) goto cleanup;

    /* Chebyshev nodes in log-mapped coordinate → well-conditioned design matrix */
    cheb_fit_grid(rhos, n_fit, rho_min, rho_max);
    compute_V_grid(ctx, rhos, n_fit, V, k_max, tol);
    for (int i = 0; i < n_fit; i++) rhoV[i] = rhos[i] * V[i];

    for (int i = 0; i < n_fit; i++) {
        double x = rho_to_x_log(rhos[i], rho_min, rho_max);
        double Pj = 1.0, Pj_prev = 0.0;
        for (int j = 0; j < n_params; j++) {
            A[i*n_params + j] = Pj;
            double Pj_next;
            if (j == 0) {
                Pj_next = x;
            } else {
                Pj_next = ((2.0*j + 1.0) * x * Pj - j * Pj_prev) / (j + 1.0);
            }
            Pj_prev = Pj;
            Pj = Pj_next;
        }
    }

    if (solve_normal_eqns(A, rhoV, n_fit, n_params, ws) != 0) goto cleanup;

    fit = malloc(sizeof(FSCFit));
    if (!fit) goto cleanup;
    fit->method   = FSC_METHOD_LEGENDRE;
    snprintf(fit->label, sizeof(fit->label), "Legendre d=%d", degree);
    fit->n_params = n_params;
    fit->param    = malloc(n_params * sizeof(double));
    fit->order    = 0;
    fit->extra    = degree;
    fit->knots    = NULL;
    fit->n_knots  = 0;
    fit->rho_min  = rho_min;
    fit->rho_max  = rho_max;
    fit->eval     = legendre_fit_eval_wrapper;
    if (!fit->param) { free(fit); fit = NULL; goto cleanup; }
    memcpy(fit->param, rhoV, n_params * sizeof(double));

cleanup:
    free(rhos); free(V); free(rhoV); free(A); free(ws);
    return fit;
}

/* ═════════════════════════════════════════════════════════════════════
 *  Fit method: Image charges
 * ═════════════════════════════════════════════════════════════════════ */

static double img_eval_wrapper(const FSCFit *fit, double rho)
{
    double s = 0.0;
    for (int i = 0; i < fit->n_params; i++) {
        double z = fit->knots[i]; /* abuse knots array for z-positions */
        s += fit->param[i] / sqrt(rho*rho + z*z);
    }
    return s;
}

static FSCFit *fit_image_charges(const FSCContext *ctx, int n_images,
                                 double rho_min, double rho_max,
                                 double k_max, double tol)
{
    int n_fit = n_images * 8;
    if (n_fit < 50) n_fit = 50;

    double *rhos  = malloc(n_fit * sizeof(double));
    double *V     = malloc(n_fit * sizeof(double));
    double *z_pos = malloc(n_images * sizeof(double));
    double *A     = malloc((size_t)n_fit * n_images * sizeof(double));
    double *ws    = malloc((size_t)(n_images*n_images + n_images) * sizeof(double));

    FSCFit *fit = NULL;
    if (!rhos || !V || !z_pos || !A || !ws) goto cleanup;

    logspace(rhos, rho_min, rho_max, n_fit);
    compute_V_grid(ctx, rhos, n_fit, V, k_max, tol);

    /* z positions on log grid */
    logspace(z_pos, rho_min * 0.5 + 1e-12, rho_max * 2.0, n_images);

    /* Design matrix: A_ij = 1/sqrt(rho_i^2 + z_j^2) */
    for (int i = 0; i < n_fit; i++) {
        for (int j = 0; j < n_images; j++) {
            A[i*n_images + j] = 1.0 / sqrt(rhos[i]*rhos[i] + z_pos[j]*z_pos[j]);
        }
    }

    if (solve_normal_eqns(A, V, n_fit, n_images, ws) != 0) goto cleanup;

    fit = malloc(sizeof(FSCFit));
    if (!fit) goto cleanup;
    fit->method   = FSC_METHOD_IMAGE_CHARGES;
    snprintf(fit->label, sizeof(fit->label), "Image charges n=%d", n_images);
    fit->n_params = n_images;
    fit->param    = malloc(n_images * sizeof(double));
    fit->order    = 0;
    fit->extra    = n_images;
    fit->knots    = z_pos;  z_pos = NULL;
    fit->n_knots  = n_images;
    fit->rho_min  = rho_min;
    fit->rho_max  = rho_max;
    fit->eval     = img_eval_wrapper;
    if (!fit->param) { free(fit); fit = NULL; goto cleanup; }
    memcpy(fit->param, V, n_images * sizeof(double));

cleanup:
    free(rhos); free(V); free(z_pos); free(A); free(ws);
    return fit;
}

/* ═════════════════════════════════════════════════════════════════════
 *  Fit method: Pade [m/n]  (Levenberg-Marquardt)
 * ═════════════════════════════════════════════════════════════════════ */

static double pade_eval(double rho, const double *p, int m, int n)
{
    double num = 0.0;
    for (int i = 0; i <= m; i++) {
        num += p[i] * pow(rho, (double)i);
    }
    double den = 0.0;
    for (int j = 0; j < n; j++) {
        den += p[m + 1 + j] * pow(rho, (double)(j + 1));
    }
    return num / den;
}

static double pade_fit_eval_wrapper(const FSCFit *fit, double rho)
{
    return pade_eval(rho, fit->param, fit->order, fit->extra);
}

/* Simple Levenberg-Marquardt for small nonlinear least-squares.
 * Minimises  Σᵢ [ f(x_i; p) - y_i ]²  w.r.t. p[0..np-1].
 * f(x, p) = pade_eval(x, p, m, n).
 * On entry, p holds initial guess; on exit, p holds optimised parameters. */
static int lm_fit(void (*f_eval)(double x, const double *p, int m, int n,
                                  double *out),
                  const double *x, const double *y, int n_pts,
                  double *p, int m, int n,
                  int max_iter, double lambda0)
{
    int np = m + n + 1;
    double *r    = malloc(n_pts * sizeof(double));
    double *J    = malloc((size_t)n_pts * np * sizeof(double));
    double *JTJ  = malloc((size_t)np * np * sizeof(double));
    double *JTr  = malloc(np * sizeof(double));
    double *delta = malloc(np * sizeof(double));
    double *p_try = malloc(np * sizeof(double));

    if (!r || !J || !JTJ || !JTr || !delta || !p_try) {
        free(r); free(J); free(JTJ); free(JTr); free(delta); free(p_try);
        return -1;
    }

    double lambda = lambda0;

    for (int iter = 0; iter < max_iter; iter++) {
        /* Compute residuals */
        double err = 0.0;
        for (int i = 0; i < n_pts; i++) {
            double f_val;
            f_eval(x[i], p, m, n, &f_val);
            r[i] = y[i] - f_val;
            err += r[i] * r[i];
        }

        /* Jacobian via finite differences */
        double h = 1e-6;
        for (int j = 0; j < np; j++) {
            double p_save = p[j];
            p[j] = p_save + h;
            for (int i = 0; i < n_pts; i++) {
                double f_plus;
                f_eval(x[i], p, m, n, &f_plus);
                J[i*np + j] = (f_plus - (f_plus - r[i] + y[i] - y[i])) / h;
                /* simplified: J = (f(p+h) - f(p)) / h */
            }
            p[j] = p_save;
            /* recompute properly: */
            for (int i = 0; i < n_pts; i++) {
                double f_base;
                f_eval(x[i], p, m, n, &f_base);
                p[j] = p_save + h;
                double f_plus;
                f_eval(x[i], p, m, n, &f_plus);
                p[j] = p_save;
                J[i*np + j] = (f_plus - f_base) / h;
            }
        }

        /* Form JTJ and JTr */
        memset(JTJ, 0, np * np * sizeof(double));
        memset(JTr, 0, np * sizeof(double));
        for (int i = 0; i < n_pts; i++) {
            for (int j = 0; j < np; j++) {
                double J_ij = J[i*np + j];
                JTr[j] += J_ij * r[i];
                for (int k = 0; k < np; k++) {
                    JTJ[j*np + k] += J_ij * J[i*np + k];
                }
            }
        }

        /* Damped normal equations: (JTJ + λ·diag(JTJ))·δ = JTr */
        memcpy(delta, JTr, np * sizeof(double));
        double JTJ_copy[50*50]; /* small */
        memcpy(JTJ_copy, JTJ, np * np * sizeof(double));
        for (int j = 0; j < np; j++) {
            JTJ_copy[j*np + j] += lambda * JTJ_copy[j*np + j];
        }

        if (cholesky_decomp(JTJ_copy, np) != 0) {
            lambda *= 10.0;
            continue;
        }
        cholesky_solve(JTJ_copy, delta, np);

        /* Try step */
        for (int j = 0; j < np; j++) p_try[j] = p[j] + delta[j];

        double err_try = 0.0;
        for (int i = 0; i < n_pts; i++) {
            double f_val;
            f_eval(x[i], p_try, m, n, &f_val);
            double ri = y[i] - f_val;
            err_try += ri * ri;
        }

        if (err_try < err) {
            memcpy(p, p_try, np * sizeof(double));
            lambda *= 0.5;
            if (err - err_try < 1e-12 * err) break; /* converged */
        } else {
            lambda *= 5.0;
        }
    }

    free(r); free(J); free(JTJ); free(JTr); free(delta); free(p_try);
    return 0;
}

static void pade_eval_cb(double x, const double *p, int m, int n, double *out)
{
    *out = pade_eval(x, p, m, n);
}

static FSCFit *fit_pade(const FSCContext *ctx, int m, int n,
                        double rho_min, double rho_max,
                        double k_max, double tol)
{
    int n_params = m + n + 1;
    int n_fit = n_params * 10;
    if (n_fit < 40) n_fit = 40;

    double *rhos = malloc(n_fit * sizeof(double));
    double *V    = malloc(n_fit * sizeof(double));
    double *p    = malloc(n_params * sizeof(double));

    FSCFit *fit = NULL;
    if (!rhos || !V || !p) goto cleanup;

    logspace(rhos, rho_min, rho_max, n_fit);
    compute_V_grid(ctx, rhos, n_fit, V, k_max, tol);

    /* Initial guess: match small-rho asymptotic V ~ 2/(eps_c * rho)
     * For Pade [m/n]: V ~ a_0 / (b_1 * rho) as rho → 0
     * Want a_0 / b_1 ≈ 2 / eps_c */
    double a0_guess = 2.0 / ctx->eps_c;
    for (int i = 0; i <= m; i++) p[i] = (i == 0) ? a0_guess : 0.0;
    for (int j = 0; j < n; j++) p[m+1+j] = (j == 0) ? 1.0 : 0.0;

    lm_fit(pade_eval_cb, rhos, V, n_fit, p, m, n, 50, 1e-2);

    fit = malloc(sizeof(FSCFit));
    if (!fit) goto cleanup;
    fit->method   = FSC_METHOD_PADE;
    snprintf(fit->label, sizeof(fit->label), "Pade [%d/%d]", m, n);
    fit->n_params = n_params;
    fit->param    = p;  p = NULL;
    fit->order    = m;
    fit->extra    = n;
    fit->knots    = NULL;
    fit->n_knots  = 0;
    fit->rho_min  = rho_min;
    fit->rho_max  = rho_max;
    fit->eval     = pade_fit_eval_wrapper;

cleanup:
    free(rhos); free(V); free(p);
    return fit;
}

/* ═════════════════════════════════════════════════════════════════════
 *  Public: automatic Chebyshev fit (recommended default)
 * ═════════════════════════════════════════════════════════════════════ */

FSCFit *fsc_fit_auto(const FSCContext *ctx,
                     double rho_min, double rho_max,
                     double target_err_pct,
                     double k_max, double tol,
                     double *achieved_err_pct)
{
    /* Sweep Chebyshev degrees: smallest degree meeting target wins */
    int degrees[] = {3, 5, 7, 9, 11, 15, 20, 25, 30};
    int n_deg = sizeof(degrees) / sizeof(degrees[0]);

    /* Generate evaluation grid for error checking */
    int n_eval = 100;
    double *rho_eval = malloc(n_eval * sizeof(double));
    double *V_exact  = malloc(n_eval * sizeof(double));
    if (!rho_eval || !V_exact) {
        free(rho_eval); free(V_exact);
        return fsc_fit_chebyshev(ctx, 7, rho_min, rho_max, k_max, tol);
    }
    logspace(rho_eval, rho_min, rho_max, n_eval);
    compute_V_grid(ctx, rho_eval, n_eval, V_exact, k_max, tol * 0.1);

    FSCFit *best_fit = NULL;
    double   best_err = 1e100;

    for (int i = 0; i < n_deg; i++) {
        FSCFit *fit = fsc_fit_chebyshev(ctx, degrees[i],
                                        rho_min, rho_max, k_max, tol);
        if (!fit) continue;

        double max_err = 0.0;
        for (int j = 0; j < n_eval; j++) {
            double Ve = fsc_fit_eval(fit, rho_eval[j]);
            double re = fabs(Ve - V_exact[j]) / fmax(fabs(V_exact[j]), 1e-15);
            if (re > max_err) max_err = re;
        }

        if (max_err * 100.0 <= target_err_pct) {
            /* Smallest degree that meets target — done */
            fsc_fit_free(best_fit);
            if (achieved_err_pct) *achieved_err_pct = max_err * 100.0;
            free(rho_eval); free(V_exact);
            return fit;
        }

        /* Keep best-so-far as fallback */
        if (max_err < best_err) {
            best_err = max_err;
            fsc_fit_free(best_fit);
            best_fit = fit;
        } else {
            fsc_fit_free(fit);
        }
    }

    if (achieved_err_pct) *achieved_err_pct = best_err * 100.0;
    free(rho_eval); free(V_exact);
    if (best_fit) return best_fit;
    return fsc_fit_chebyshev(ctx, 7, rho_min, rho_max, k_max, tol);
}

/* ═════════════════════════════════════════════════════════════════════
 *  Public: fit constructors
 * ═════════════════════════════════════════════════════════════════════ */

FSCFit *fsc_fit_pade(const FSCContext *ctx, int m, int n,
                     double rho_min, double rho_max,
                     double k_max, double tol)
{
    return fit_pade(ctx, m, n, rho_min, rho_max, k_max, tol);
}

FSCFit *fsc_fit_chebyshev(const FSCContext *ctx, int degree,
                          double rho_min, double rho_max,
                          double k_max, double tol)
{
    return fit_chebyshev(ctx, degree, rho_min, rho_max, k_max, tol);
}

FSCFit *fsc_fit_legendre(const FSCContext *ctx, int degree,
                         double rho_min, double rho_max,
                         double k_max, double tol)
{
    return fit_legendre(ctx, degree, rho_min, rho_max, k_max, tol);
}

FSCFit *fsc_fit_image_chg(const FSCContext *ctx, int n_images,
                          double rho_min, double rho_max,
                          double k_max, double tol)
{
    return fit_image_charges(ctx, n_images, rho_min, rho_max, k_max, tol);
}

FSCFit *fsc_fit_bspline(const FSCContext *ctx, int n_interior_knots,
                        double rho_min, double rho_max,
                        double k_max, double tol)
{
    return fit_bspline(ctx, rho_min, rho_max, n_interior_knots, k_max, tol);
}

/* ═════════════════════════════════════════════════════════════════════
 *  Public: fit evaluation
 * ═════════════════════════════════════════════════════════════════════ */

double fsc_fit_eval(const FSCFit *fit, double rho)
{
    if (!fit || !fit->eval) return HUGE_VAL;
    return fit->eval(fit, rho);
}

void fsc_fit_eval_array(const FSCFit *fit, const double *rhos, int n,
                        double *V_out)
{
    for (int i = 0; i < n; i++) {
        V_out[i] = fsc_fit_eval(fit, rhos[i]);
    }
}

FSCMethod fsc_fit_method(const FSCFit *fit) { return fit->method; }
int       fsc_fit_n_params(const FSCFit *fit) { return fit->n_params; }
const char *fsc_fit_label(const FSCFit *fit) { return fit->label; }

void fsc_fit_free(FSCFit *fit)
{
    if (!fit) return;
    free(fit->param);
    free(fit->knots);
    free(fit);
}

/* ═════════════════════════════════════════════════════════════════════
 *  Public: benchmark
 * ═════════════════════════════════════════════════════════════════════ */

FSCBenchEntry *fsc_benchmark(const FSCContext *ctx,
                             double rho_min, double rho_max,
                             int n_fit_pts, int n_eval_pts,
                             const double *targets, int n_targets,
                             double k_max, double tol,
                             int *n_entries_out, FILE *fp)
{
    /* Allocate result array (max: 5 methods × 30 param counts × 6 targets) */
    int max_entries = 5 * 30;
    FSCBenchEntry *entries = malloc(max_entries * sizeof(FSCBenchEntry));
    int n_entries = 0;
    if (!entries) { *n_entries_out = 0; return NULL; }

    /* Generate evaluation grid */
    double *rho_fit  = malloc(n_fit_pts * sizeof(double));
    double *rho_eval = malloc(n_eval_pts * sizeof(double));
    double *V_exact  = malloc(n_eval_pts * sizeof(double));
    if (!rho_fit || !rho_eval || !V_exact) {
        free(rho_fit); free(rho_eval); free(V_exact);
        free(entries); *n_entries_out = 0; return NULL;
    }
    logspace(rho_fit,  rho_min, rho_max, n_fit_pts);
    logspace(rho_eval, rho_min, rho_max, n_eval_pts);
    compute_V_grid(ctx, rho_eval, n_eval_pts, V_exact, k_max, tol);

    /* Time exact integral */
    clock_t t0 = clock();
    for (int rep = 0; rep < 50; rep++) {
        fsc_potential(ctx, rho_eval[0], k_max, tol);
    }
    double t_exact_ms = (double)(clock() - t0) / CLOCKS_PER_SEC / 20.0 * 1000.0;

    if (fp) {
        fprintf(fp, "=== Fast Screened Coulomb — Benchmark ===\n");
        fprintf(fp, "  exact integral: %.3f ms/point (1 point)\n", t_exact_ms);
        fprintf(fp, "  k_max=%.1f  tol=%.1e  rho=[%.2f, %.2f]\n\n",
                k_max, tol, rho_min, rho_max);
    }

    /* ── Sweep all methods ── */
    /* Pade [m/n] */
    int pade_sweep[][2] = {{1,1},{1,2},{2,2},{2,3},{3,3}};
    int n_pade = 5;
    for (int s = 0; s < n_pade; s++) {
        int m = pade_sweep[s][0], n_p = pade_sweep[s][1];
        FSCFit *fit = fsc_fit_pade(ctx, m, n_p, rho_min, rho_max, k_max, tol);
        if (!fit) continue;
        /* error */
        double max_err = 0.0;
        for (int i = 0; i < n_eval_pts; i++) {
            double Ve = fsc_fit_eval(fit, rho_eval[i]);
            double re = fabs(Ve - V_exact[i]) / fmax(fabs(V_exact[i]), 1e-15);
            if (re > max_err) max_err = re;
        }
        /* timing */
        clock_t t_start = clock(), t_now;
        int n_reps = 0;
        do {
            for (int batch = 0; batch < 1000; batch++)
                fsc_fit_eval(fit, rho_eval[batch % n_eval_pts]);
            n_reps += 1000;
            t_now = clock();
        } while ((double)(t_now - t_start) / CLOCKS_PER_SEC < 0.02 && n_reps < 5000000);
        double t_us = (double)(t_now - t_start) / CLOCKS_PER_SEC / (double)n_reps * 1e6;

        for (int ti = 0; ti < n_targets; ti++) {
            if (max_err * 100.0 <= targets[ti]) {
                if (n_entries < max_entries) {
                    entries[n_entries].method = FSC_METHOD_PADE;
                    snprintf(entries[n_entries].label,
                             sizeof(entries[n_entries].label),
                             "Pade [%d/%d]", m, n_p);
                    entries[n_entries].n_params = m + n_p + 1;
                    entries[n_entries].max_err_pct = max_err * 100.0;
                    entries[n_entries].t_eval_us = t_us;
                    n_entries++;
                }
                break; /* first (cheapest) hit for this target */
            }
        }
        fsc_fit_free(fit);
    }

    /* Chebyshev */
    int cheb_degrees[] = {3, 5, 7, 9, 11, 15, 20, 25, 30};
    int n_cheb = sizeof(cheb_degrees) / sizeof(cheb_degrees[0]);
    for (int s = 0; s < n_cheb; s++) {
        int d = cheb_degrees[s];
        FSCFit *fit = fsc_fit_chebyshev(ctx, d, rho_min, rho_max, k_max, tol);
        if (!fit) continue;
        double max_err = 0.0;
        for (int i = 0; i < n_eval_pts; i++) {
            double Ve = fsc_fit_eval(fit, rho_eval[i]);
            double re = fabs(Ve - V_exact[i]) / fmax(fabs(V_exact[i]), 1e-15);
            if (re > max_err) max_err = re;
        }
        clock_t t_start = clock(), t_now;
        int n_reps = 0;
        do {
            for (int batch = 0; batch < 1000; batch++)
                fsc_fit_eval(fit, rho_eval[batch % n_eval_pts]);
            n_reps += 1000;
            t_now = clock();
        } while ((double)(t_now - t_start) / CLOCKS_PER_SEC < 0.02 && n_reps < 5000000);
        double t_us = (double)(t_now - t_start) / CLOCKS_PER_SEC / (double)n_reps * 1e6;

        for (int ti = 0; ti < n_targets; ti++) {
            if (max_err * 100.0 <= targets[ti]) {
                if (n_entries < max_entries) {
                    entries[n_entries].method = FSC_METHOD_CHEBYSHEV;
                    snprintf(entries[n_entries].label,
                             sizeof(entries[n_entries].label),
                             "Chebyshev d=%d", d);
                    entries[n_entries].n_params = d + 1;
                    entries[n_entries].max_err_pct = max_err * 100.0;
                    entries[n_entries].t_eval_us = t_us;
                    n_entries++;
                }
                break;
            }
        }
        fsc_fit_free(fit);
    }

    /* Legendre, Image charges, B-spline sweeps */
    int legendre_degrees[] = {3, 5, 7, 9, 11, 15, 20, 25, 30};
    int n_legendre = sizeof(legendre_degrees) / sizeof(legendre_degrees[0]);
    for (int s = 0; s < n_legendre; s++) {
        int d = legendre_degrees[s];
        FSCFit *fit = fsc_fit_legendre(ctx, d, rho_min, rho_max, k_max, tol);
        if (!fit) continue;
        double max_err = 0.0;
        for (int i = 0; i < n_eval_pts; i++) {
            double Ve = fsc_fit_eval(fit, rho_eval[i]);
            double re = fabs(Ve - V_exact[i]) / fmax(fabs(V_exact[i]), 1e-15);
            if (re > max_err) max_err = re;
        }
        clock_t t_start = clock(), t_now;
        int n_reps = 0;
        do {
            for (int batch = 0; batch < 1000; batch++)
                fsc_fit_eval(fit, rho_eval[batch % n_eval_pts]);
            n_reps += 1000;
            t_now = clock();
        } while ((double)(t_now - t_start) / CLOCKS_PER_SEC < 0.02 && n_reps < 5000000);
        double t_us = (double)(t_now - t_start) / CLOCKS_PER_SEC / (double)n_reps * 1e6;

        for (int ti = 0; ti < n_targets; ti++) {
            if (max_err * 100.0 <= targets[ti]) {
                if (n_entries < max_entries) {
                    entries[n_entries].method = FSC_METHOD_LEGENDRE;
                    snprintf(entries[n_entries].label,
                             sizeof(entries[n_entries].label),
                             "Legendre d=%d", d);
                    entries[n_entries].n_params = d + 1;
                    entries[n_entries].max_err_pct = max_err * 100.0;
                    entries[n_entries].t_eval_us = t_us;
                    n_entries++;
                }
                break;
            }
        }
        fsc_fit_free(fit);
    }

    int img_counts[] = {2, 3, 4, 5, 6, 8, 10, 12, 15, 20};
    int n_img = sizeof(img_counts) / sizeof(img_counts[0]);
    for (int s = 0; s < n_img; s++) {
        int ni = img_counts[s];
        FSCFit *fit = fsc_fit_image_chg(ctx, ni, rho_min, rho_max, k_max, tol);
        if (!fit) continue;
        double max_err = 0.0;
        for (int i = 0; i < n_eval_pts; i++) {
            double Ve = fsc_fit_eval(fit, rho_eval[i]);
            double re = fabs(Ve - V_exact[i]) / fmax(fabs(V_exact[i]), 1e-15);
            if (re > max_err) max_err = re;
        }
        clock_t t_start = clock(), t_now;
        int n_reps = 0;
        do {
            for (int batch = 0; batch < 1000; batch++)
                fsc_fit_eval(fit, rho_eval[batch % n_eval_pts]);
            n_reps += 1000;
            t_now = clock();
        } while ((double)(t_now - t_start) / CLOCKS_PER_SEC < 0.02 && n_reps < 5000000);
        double t_us = (double)(t_now - t_start) / CLOCKS_PER_SEC / (double)n_reps * 1e6;

        for (int ti = 0; ti < n_targets; ti++) {
            if (max_err * 100.0 <= targets[ti]) {
                if (n_entries < max_entries) {
                    entries[n_entries].method = FSC_METHOD_IMAGE_CHARGES;
                    snprintf(entries[n_entries].label,
                             sizeof(entries[n_entries].label),
                             "Image charges n=%d", ni);
                    entries[n_entries].n_params = ni;
                    entries[n_entries].max_err_pct = max_err * 100.0;
                    entries[n_entries].t_eval_us = t_us;
                    n_entries++;
                }
                break;
            }
        }
        fsc_fit_free(fit);
    }

    int bs_knots_sweep[] = {0, 1, 2, 3, 5, 8, 12, 20};
    int n_bs = sizeof(bs_knots_sweep) / sizeof(bs_knots_sweep[0]);
    for (int s = 0; s < n_bs; s++) {
        int nk = bs_knots_sweep[s];
        FSCFit *fit = fsc_fit_bspline(ctx, nk, rho_min, rho_max, k_max, tol);
        if (!fit) continue;
        double max_err = 0.0;
        for (int i = 0; i < n_eval_pts; i++) {
            double Ve = fsc_fit_eval(fit, rho_eval[i]);
            double re = fabs(Ve - V_exact[i]) / fmax(fabs(V_exact[i]), 1e-15);
            if (re > max_err) max_err = re;
        }
        clock_t t_start = clock(), t_now;
        int n_reps = 0;
        do {
            for (int batch = 0; batch < 1000; batch++)
                fsc_fit_eval(fit, rho_eval[batch % n_eval_pts]);
            n_reps += 1000;
            t_now = clock();
        } while ((double)(t_now - t_start) / CLOCKS_PER_SEC < 0.02 && n_reps < 5000000);
        double t_us = (double)(t_now - t_start) / CLOCKS_PER_SEC / (double)n_reps * 1e6;

        for (int ti = 0; ti < n_targets; ti++) {
            if (max_err * 100.0 <= targets[ti]) {
                if (n_entries < max_entries) {
                    entries[n_entries].method = FSC_METHOD_BSPLINE;
                    snprintf(entries[n_entries].label,
                             sizeof(entries[n_entries].label),
                             "B-spline k=%d", nk);
                    entries[n_entries].n_params = BSPLINE_ORDER + nk;
                    entries[n_entries].max_err_pct = max_err * 100.0;
                    entries[n_entries].t_eval_us = t_us;
                    n_entries++;
                }
                break;
            }
        }
        fsc_fit_free(fit);
    }

    /* ── Print table ── */
    if (fp) {
        for (int ti = 0; ti < n_targets; ti++) {
            fprintf(fp, "  --- target: max error < %.1f%% ---\n", targets[ti]);
            fprintf(fp, "  %20s  %7s  %10s  %10s  %10s\n",
                    "method", "params", "max err %", "t_eval", "speedup");
            fprintf(fp, "  %s\n",
                    "------------------------------------------------------------------");
            for (int mi = 0; mi < 5; mi++) {
                /* Find best (fewest params) entry for this method+target */
                int best_idx = -1;
                int best_np = 999999;
                for (int e = 0; e < n_entries; e++) {
                    if ((int)entries[e].method == mi &&
                        entries[e].max_err_pct <= targets[ti] &&
                        entries[e].n_params < best_np) {
                        best_np = entries[e].n_params;
                        best_idx = e;
                    }
                }
                if (best_idx >= 0) {
                    double speedup = t_exact_ms * 1000.0 /
                                     fmax(entries[best_idx].t_eval_us, 1e-12);
                    fprintf(fp, "  %20s  %7d  %10.4f  %8.2f us  %9.0fx\n",
                            entries[best_idx].label,
                            entries[best_idx].n_params,
                            entries[best_idx].max_err_pct,
                            entries[best_idx].t_eval_us,
                            speedup);
                }
            }
            fprintf(fp, "\n");
        }
    }

    free(rho_fit);
    free(rho_eval);
    free(V_exact);
    *n_entries_out = n_entries;
    return entries;
}

void fsc_bench_free(FSCBenchEntry *entries)
{
    free(entries);
}

/* ═════════════════════════════════════════════════════════════════════
 *  Lifecycle
 * ═════════════════════════════════════════════════════════════════════ */

FSCContext *fsc_create(int n_layers, const double *epsilons, const double *d,
                       int c, int t, double z_t)
{
    if (n_layers < 1 || !epsilons) return NULL;
    if (n_layers > 1 && !d) return NULL;
    if (c < 1 || c > n_layers) return NULL;

    FSCContext *ctx = malloc(sizeof(FSCContext));
    if (!ctx) return NULL;

    ctx->n_layers = n_layers;
    ctx->epsilons = malloc(n_layers * sizeof(double));
    ctx->d        = (n_layers > 1) ? malloc((n_layers - 1) * sizeof(double)) : NULL;
    if (!ctx->epsilons || (n_layers > 1 && !ctx->d)) {
        free(ctx->epsilons); free(ctx->d); free(ctx);
        return NULL;
    }
    memcpy(ctx->epsilons, epsilons, n_layers * sizeof(double));
    if (n_layers > 1) {
        memcpy(ctx->d, d, (n_layers - 1) * sizeof(double));
    }

    if (t <= 0) {
        ctx->t = c;
        ctx->z_t = 0.0;
    } else {
        ctx->t = t;
        ctx->z_t = z_t;
    }
    ctx->c = c;
    ctx->eps_c = epsilons[c - 1];
    ctx->is_direct = (ctx->t == c && fabs(ctx->z_t) < 1e-15);

    return ctx;
}

void fsc_free(FSCContext *ctx)
{
    if (!ctx) return;
    free(ctx->epsilons);
    free(ctx->d);
    free(ctx);
}

/* Ordinary derivatives of the existing fit evaluators. Differentiate their
 * recurrences (including de Boor's regularizer), leaving all value code intact.
 * Polynomial fits use V=S(x(rho))/rho with dx/drho=2/(rho*log(max/min)).
 * Thus V'=(2*S_x/log(max/min)-S)/rho^2. No new physical approximation.
 */
int fsc_fit_supports_d1(const FSCFit *fit)
{
    if (!fit) return 0;
    switch (fit->method) {
    case FSC_METHOD_PADE:
    case FSC_METHOD_CHEBYSHEV:
    case FSC_METHOD_LEGENDRE:
    case FSC_METHOD_IMAGE_CHARGES:
    case FSC_METHOD_BSPLINE:
        return 1;
    default:
        return 0;
    }
}

static double polynomial_fit_d1(const FSCFit *fit, double rho)
{
    const double x = rho_to_x_log(rho, fit->rho_min, fit->rho_max);
    double sum = fit->param[0], sum_d1 = 0.0;
    double prev = 1.0, current = x, prev_d1 = 0.0, current_d1 = 1.0;
    if (fit->n_params > 1) {
        sum += fit->param[1] * current;
        sum_d1 += fit->param[1];
    }
    for (int j = 2; j < fit->n_params; ++j) {
        double next, next_d1;
        if (fit->method == FSC_METHOD_CHEBYSHEV) {
            next = 2.0 * x * current - prev;
            next_d1 = 2.0 * (current + x * current_d1) - prev_d1;
        } else {
            next = ((2*j-1) * x * current - (j-1) * prev) / j;
            next_d1 = ((2*j-1) * (current + x * current_d1) -
                       (j-1) * prev_d1) / j;
        }
        sum += fit->param[j] * next;
        sum_d1 += fit->param[j] * next_d1;
        prev = current; current = next;
        prev_d1 = current_d1; current_d1 = next_d1;
    }
    return ((2.0 / log(fit->rho_max / fit->rho_min)) * sum_d1 - sum) / rho / rho;
}

static double bspline_fit_d1(const FSCFit *fit, double rho)
{
    const int order = fit->order, n_ctrl = fit->n_params;
    const double *knots = fit->knots;
    if (rho <= knots[0]) return -fit->param[0] / rho / rho;
    if (rho >= knots[n_ctrl + order - 1]) return -fit->param[n_ctrl-1] / rho / rho;
    const int i = bspline_find_interval(rho, knots, n_ctrl + order, order);
    double value[10], derivative[10]; /* [order], same bound as value de Boor */
    for (int j = 0; j < order; ++j) {
        value[j] = fit->param[i-order+1+j];
        derivative[j] = 0.0;
    }
    for (int r = 1; r < order; ++r)
        for (int j = order-1; j >= r; --j) {
            const int idx = i-order+1+j;
            const double denom = knots[idx+order-r] - knots[idx] + 1e-30;
            const double alpha = (rho-knots[idx]) / denom;
            derivative[j] = (1.0-alpha)*derivative[j-1] + alpha*derivative[j] +
                            (value[j]-value[j-1]) / denom;
            value[j] = (1.0-alpha)*value[j-1] + alpha*value[j];
        }
    return (derivative[order-1] - value[order-1]/rho) / rho;
}

double fsc_fit_d1(const FSCFit *fit, double rho)
{
    if (!fsc_fit_supports_d1(fit) || !isfinite(rho) || rho <= 0.0) return NAN;
    switch (fit->method) {
    case FSC_METHOD_CHEBYSHEV:
    case FSC_METHOD_LEGENDRE:
        return polynomial_fit_d1(fit, rho);
    case FSC_METHOD_BSPLINE:
        return bspline_fit_d1(fit, rho);
    case FSC_METHOD_PADE: {
        const int m = fit->order, n = fit->extra;
        double num = 0.0, den = 0.0, num_d1 = 0.0, den_d1 = 0.0;
        for (int i = 0; i <= m; ++i) {
            num += fit->param[i] * pow(rho, i);
            if (i > 0) num_d1 += i * fit->param[i] * pow(rho, i-1);
        }
        for (int j = 0; j < n; ++j) {
            den += fit->param[m+1+j] * pow(rho, j+1);
            den_d1 += (j+1) * fit->param[m+1+j] * pow(rho, j);
        }
        /* Quotient rule without forming den^2. Poles remain nonfinite. */
        return (num_d1 - (num/den)*den_d1) / den;
    }
    case FSC_METHOD_IMAGE_CHARGES: {
        double derivative = 0.0;
        for (int i = 0; i < fit->n_params; ++i) {
            const double radius = hypot(rho, fit->knots[i]);
            derivative -= (fit->param[i]/radius) / radius * (rho/radius);
        }
        return derivative;
    }
    default:
        return NAN;
    }
}

void fsc_fit_d1_array(const FSCFit *fit, const double *rhos, int n, double *dV_out)
{
    for (int i = 0; i < n; ++i) dV_out[i] = fsc_fit_d1(fit, rhos[i]);
}

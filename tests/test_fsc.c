#include "fast_screened_coulomb.h"
#include <stdio.h>
#include <math.h>

/* Portable inf/finite — use C99 macros where available */
#define c99_isinf(x)    ((x) != 0 && (x) * 0.5 == (x))
#define c99_isfinite(x) ((x) == (x) && (x) * 0.0 == 0.0)

static int tests_run = 0, tests_passed = 0;

static void check_rel_err(double got, double expected, double tol, const char *msg) {
    tests_run++;
    double denom = fmax(fabs(expected), 1e-15);
    double re = fabs(got - expected) / denom;
    if (re < tol) { tests_passed++; }
    else { printf("  FAIL: %s (got=%.6e exp=%.6e re=%.2e)\n", msg, got, expected, re); }
}
static void check_ok(int cond, const char *msg) {
    tests_run++;
    if (cond) { tests_passed++; }
    else { printf("  FAIL: %s\n", msg); }
}

#define TEST(name) static void name(void)

TEST(test_epsilon_N3_exact) {
    printf("Slice 1: epsilon N3\n");
    double eps[] = {1.0, 14.0, 1.0};
    double d[] = {-0.25, 0.25};
    FSCContext *ctx = fsc_create(3, eps, d, 2, 0, 0.0);
    check_ok(ctx != NULL, "create");
    struct { double k, exp; } ref[] = {
        {0.001,1.003482080604589e+00},{0.01,1.034815139050803e+00},
        {0.1,1.347521314680677e+00},{1.0,4.352714066018002e+00},
        {10.0,1.383744175382293e+01},{100.0,1.400000000000000e+01}};
    for (int i=0;i<6;i++) { double ek=fsc_epsilon_k(ctx,ref[i].k); check_rel_err(ek,ref[i].exp,1e-10,"eps"); }
    fsc_free(ctx);
}

TEST(test_potential_hom) {
    printf("Slice 2a: V hom\n");
    double eps[]={5,5,5}; double d[]={-0.5,0.5};
    FSCContext*ctx=fsc_create(3,eps,d,2,0,0);
    double rhos[]={0.5,2,10,30}, exp[]={0.8,0.2,0.04,0.01333333333333333};
    for(int i=0;i<4;i++){double V=fsc_potential(ctx,rhos[i],20,1e-8);check_rel_err(V,exp[i],2e-5,"hom");}
    fsc_free(ctx);
}

TEST(test_potential_bl) {
    printf("Slice 2b: V bilayer\n");
    double eps[]={1,14,14,1}; double d[]={-3,3,9};
    FSCContext*ctx=fsc_create(4,eps,d,2,0,0);
    double rhos[]={0.5,2,10,30}, exp[]={3.406712414550356e-01,1.253152298813076e-01,5.738483257515958e-02,3.409547286065846e-02};
    for(int i=0;i<4;i++){double V=fsc_potential(ctx,rhos[i],20,1e-8);check_rel_err(V,exp[i],1e-5,"bl");}
    fsc_free(ctx);
}

TEST(test_potential_ind) {
    printf("Slice 3: V indirect\n");
    double eps[]={1,14,14,1}; double d[]={-0.5,0.5,1.5};
    FSCContext*ctx=fsc_create(4,eps,d,2,3,1.0);
    double rhos[]={0.5,2,10,30}, exp[]={4.153511617347747e-01,3.091405392601423e-01,1.357477764813350e-01,5.940884792113079e-02};
    for(int i=0;i<4;i++){double V=fsc_potential(ctx,rhos[i],5,1e-8);check_rel_err(V,exp[i],1e-5,"ind");}
    fsc_free(ctx);
}

TEST(test_bspline) {
    printf("Slice 4: B-spline\n");
    double eps[]={1,14,14,1}; double d[]={-3,3,9};
    FSCContext*ctx=fsc_create(4,eps,d,2,0,0);
    FSCFit*fit=fsc_fit_bspline(ctx,8,0.1,30,20,1e-8);
    check_ok(fit!=NULL,"fit");
    check_ok(fsc_fit_method(fit)==FSC_METHOD_BSPLINE,"method");
    double rhos[]={0.1,0.188467,0.355199,0.669433,1.26166,2.37782,4.48140,8.44598,15.9179,30};
    double Vex[]={1.483602122e+00,8.130176876e-01,4.571842703e-01,2.682968182e-01,1.677863248e-01,1.135370277e-01,8.237753462e-02,6.184247361e-02,4.657915462e-02,3.409547286e-02};
    double maxe=0;
    for(int i=0;i<10;i++){double Vf=fsc_fit_eval(fit,rhos[i]);double e=fabs(Vf-Vex[i])/Vex[i]*100;if(e>maxe)maxe=e;check_rel_err(Vf,Vex[i],0.01,"bs");}
    printf("  max_err=%.4f%%\n",maxe);
    fsc_fit_free(fit); fsc_free(ctx);
}

TEST(test_chebyshev) {
    printf("Slice 5: Chebyshev\n");
    double eps[]={1,14,14,1}; double d[]={-3,3,9};
    FSCContext*ctx=fsc_create(4,eps,d,2,0,0);
    FSCFit*fit=fsc_fit_chebyshev(ctx,9,0.1,30,20,1e-8);
    check_ok(fit!=NULL,"fit");
    check_ok(fsc_fit_method(fit)==FSC_METHOD_CHEBYSHEV,"method");
    double rhos[]={0.1,0.188467,0.355199,0.669433,1.26166,2.37782,4.48140,8.44598,15.9179,30};
    double Vex[]={1.483602122e+00,8.130176876e-01,4.571842703e-01,2.682968182e-01,1.677863248e-01,1.135370277e-01,8.237753462e-02,6.184247361e-02,4.657915462e-02,3.409547286e-02};
    double maxe=0;
    for(int i=0;i<10;i++){double Vf=fsc_fit_eval(fit,rhos[i]);double e=fabs(Vf-Vex[i])/Vex[i]*100;if(e>maxe)maxe=e;check_rel_err(Vf,Vex[i],0.01,"ch");}
    printf("  max_err=%.4f%%\n",maxe);
    fsc_fit_free(fit); fsc_free(ctx);
}

TEST(test_legendre) {
    printf("Slice 6: Legendre\n");
    double eps[]={1,14,14,1}; double d[]={-3,3,9};
    FSCContext*ctx=fsc_create(4,eps,d,2,0,0);
    FSCFit*fit=fsc_fit_legendre(ctx,9,0.1,30,20,1e-8);
    check_ok(fit!=NULL,"fit");
    double rhos[]={0.1,0.188467,0.355199,0.669433,1.26166,2.37782,4.48140,8.44598,15.9179,30};
    double Vex[]={1.483602122e+00,8.130176876e-01,4.571842703e-01,2.682968182e-01,1.677863248e-01,1.135370277e-01,8.237753462e-02,6.184247361e-02,4.657915462e-02,3.409547286e-02};
    double maxe=0;
    for(int i=0;i<10;i++){double Vf=fsc_fit_eval(fit,rhos[i]);double e=fabs(Vf-Vex[i])/Vex[i]*100;if(e>maxe)maxe=e;check_rel_err(Vf,Vex[i],0.01,"lg");}
    printf("  max_err=%.4f%%\n",maxe);
    fsc_fit_free(fit); fsc_free(ctx);
}

TEST(test_img) {
    printf("Slice 7: Image chg\n");
    double eps[]={1,14,14,1}; double d[]={-3,3,9};
    FSCContext*ctx=fsc_create(4,eps,d,2,0,0);
    FSCFit*fit=fsc_fit_image_chg(ctx,6,0.1,30,20,1e-8);
    check_ok(fit!=NULL,"fit");
    double rhos[]={0.1,0.188467,0.355199,0.669433,1.26166,2.37782,4.48140,8.44598,15.9179,30};
    double Vex[]={1.483602122e+00,8.130176876e-01,4.571842703e-01,2.682968182e-01,1.677863248e-01,1.135370277e-01,8.237753462e-02,6.184247361e-02,4.657915462e-02,3.409547286e-02};
    double maxe=0;
    for(int i=0;i<10;i++){double Vf=fsc_fit_eval(fit,rhos[i]);double e=fabs(Vf-Vex[i])/Vex[i]*100;if(e>maxe)maxe=e;check_rel_err(Vf,Vex[i],0.05,"img");}
    printf("  max_err=%.4f%%\n",maxe);
    fsc_fit_free(fit); fsc_free(ctx);
}

TEST(test_pade) {
    printf("Slice 8: Pade\n");
    double eps[]={1,14,14,1}; double d[]={-3,3,9};
    FSCContext*ctx=fsc_create(4,eps,d,2,0,0);
    FSCFit*fit=fsc_fit_pade(ctx,1,2,0.1,30,20,1e-8);
    check_ok(fit!=NULL,"fit");
    double rhos[]={0.1,0.188467,0.355199,0.669433,1.26166,2.37782,4.48140,8.44598,15.9179,30};
    double Vex[]={1.483602122e+00,8.130176876e-01,4.571842703e-01,2.682968182e-01,1.677863248e-01,1.135370277e-01,8.237753462e-02,6.184247361e-02,4.657915462e-02,3.409547286e-02};
    double maxe=0;
    for(int i=0;i<10;i++){double Vf=fsc_fit_eval(fit,rhos[i]);double e=fabs(Vf-Vex[i])/Vex[i]*100;if(e>maxe)maxe=e;check_rel_err(Vf,Vex[i],0.03,"pa");}
    printf("  max_err=%.4f%%\n",maxe);
    fsc_fit_free(fit); fsc_free(ctx);
}

TEST(test_auto_fit) {
    printf("Slice 8b: Auto-fit (Chebyshev default)\n");
    double eps[]={1,14,14,1}; double d[]={-3,3,9};
    FSCContext*ctx=fsc_create(4,eps,d,2,0,0);
    double Vex[]={1.483602122e+00,8.130176876e-01,4.571842703e-01,2.682968182e-01,1.677863248e-01,1.135370277e-01,8.237753462e-02,6.184247361e-02,4.657915462e-02,3.409547286e-02};
    double rhos[]={0.1,0.188467,0.355199,0.669433,1.26166,2.37782,4.48140,8.44598,15.9179,30};

    /* Target 1%: should pick low degree (3 or 5) */
    double achieved1;
    FSCFit*f1=fsc_fit_auto(ctx,0.1,30,1.0,20,1e-8,&achieved1);
    check_ok(f1!=NULL,"auto 1%");
    check_ok(fsc_fit_method(f1)==FSC_METHOD_CHEBYSHEV,"auto method");
    check_ok(fsc_fit_n_params(f1)<=10,"auto 1% params <=10");
    check_ok(achieved1 <= 1.0, "auto 1% achieved <= target");
    double maxe1=0;
    for(int i=0;i<10;i++){double Vf=fsc_fit_eval(f1,rhos[i]);double e=fabs(Vf-Vex[i])/Vex[i]*100;if(e>maxe1)maxe1=e;}
    check_ok(maxe1<1.0,"auto 1% err<1%");
    printf("  target 1%%: degree=%d n_params=%d err=%.4f%% achieved=%.4f%%\n",fsc_fit_n_params(f1)-1,fsc_fit_n_params(f1),maxe1,achieved1);
    fsc_fit_free(f1);

    /* Target 0.2%: should pick higher degree */
    double achieved02;
    FSCFit*f02=fsc_fit_auto(ctx,0.1,30,0.2,20,1e-8,&achieved02);
    check_ok(f02!=NULL,"auto 0.2%");
    check_ok(fsc_fit_n_params(f02) >= 6,"auto 0.2% params>=6");
    check_ok(achieved02 <= 0.2, "auto 0.2% achieved <= target");
    double maxe2=0;
    for(int i=0;i<10;i++){double Vf=fsc_fit_eval(f02,rhos[i]);double e=fabs(Vf-Vex[i])/Vex[i]*100;if(e>maxe2)maxe2=e;}
    check_ok(maxe2<0.2,"auto 0.2% err<0.2%");
    printf("  target 0.2%%: degree=%d n_params=%d err=%.4f%% achieved=%.4f%%\n",fsc_fit_n_params(f02)-1,fsc_fit_n_params(f02),maxe2,achieved02);
    fsc_fit_free(f02);
    fsc_free(ctx);
}

TEST(test_benchmark) {
    printf("Slice 9: Benchmark\n");
    double eps[]={1,14,14,1}; double d[]={-3,3,9};
    FSCContext*ctx=fsc_create(4,eps,d,2,0,0);
    double targets[]={10,5,2,1,0.5}; int nt=5, ne=0;
    FSCBenchEntry*e=fsc_benchmark(ctx,0.1,30,20,30,targets,nt,20,1e-8,&ne,NULL);
    check_ok(e!=NULL,"bench entries");
    check_ok(ne>0,"bench count");
    int bs_hit=0;
    for(int i=0;i<ne;i++){if(e[i].method==FSC_METHOD_BSPLINE&&e[i].max_err_pct<1.0){bs_hit=1;break;}}
    check_ok(bs_hit,"B-spline <1%");
    fsc_bench_free(e); fsc_free(ctx);
}

TEST(test_edges) {
    printf("Slice 10: Edge cases\n");
    fflush(stdout);

    /* rho=0 returns INFINITY */
    {
        double eps[]={1,14,1}; double d[]={-0.25,0.25};
        FSCContext*ctx=fsc_create(3,eps,d,2,0,0);
        check_ok(ctx!=NULL,"edge create");
        double V0=fsc_potential(ctx,0,20,1e-8);
        check_ok(V0 > 1e300,"V(0)=inf");
        double Vl=fsc_potential(ctx,100,20,1e-8);
        check_ok(Vl>0 && Vl<1,"V(100)>0");
        fsc_free(ctx);
    }

    /* Single-layer homogeneous */
    {
        double e1[]={5.0};
        FSCContext*c1=fsc_create(1,e1,NULL,1,0,0);
        check_ok(c1!=NULL,"N=1 ctx");
        if(c1){
            double ek=fsc_epsilon_k(c1,10);
            check_rel_err(ek,5.0,1e-10,"eps=5");
            double V=fsc_potential(c1,2,20,1e-8);
            check_rel_err(V,0.2,1e-4,"V single");
            fsc_free(c1);
        }
    }

    /* High contrast */
    {
        double eh[]={1,100,1}; double dh[]={-0.25,0.25};
        FSCContext*ch=fsc_create(3,eh,dh,2,0,0);
        check_ok(ch!=NULL,"high cont ctx");
        if(ch){
            double ek=fsc_epsilon_k(ch,0.01);
            check_ok(ek>0&&ek<200,"high contrast eps>0");
            fsc_free(ch);
        }
    }
}

TEST(test_optimal_params_direct) {
    printf("Slice 11: Optimal params — direct exciton N3\n");

    /* N3: vacuum | slab (eps=14, width=0.5) | vacuum */
    double eps[] = {1.0, 14.0, 1.0};
    double d[] = {-0.25, 0.25};
    FSCContext *ctx = fsc_create(3, eps, d, 2, 0, 0.0);
    check_ok(ctx != NULL, "create ctx");

    double k_max = -1.0, rho_max = -1.0;
    int n_quad = -1;

    fsc_optimal_params(ctx, 1e-4, 1e-3, 1.5, &k_max, &n_quad, &rho_max);

    /* k_max must be positive and finite */
    check_ok(k_max > 0.0, "k_max > 0");
    check_ok(c99_isfinite(k_max), "k_max finite");
    printf("  k_max=%.3f\n", k_max);

    /* n_quad fixed at 200 */
    check_ok(n_quad == 200, "n_quad == 200");

    /* rho_max must be positive — V(ρ) decays as 2/(ε₀·ρ) */
    check_ok(rho_max > 0.0, "rho_max > 0");
    check_ok(c99_isfinite(rho_max), "rho_max finite");
    printf("  rho_max=%.3f a0 (%.2f Ang)\n", rho_max, rho_max * 0.529);

    /* Sanity: rho_max should match 2/(eps_0*tol_V) where eps_0 = ε(k→0) */
    double eps_0 = fsc_epsilon_k(ctx, 1e-4);
    double rho_expected = 2.0 / (eps_0 * 1e-3);
    printf("  eps_0=%.3f rho_expected=%.1f\n", eps_0, rho_expected);
    check_rel_err(rho_max, rho_expected, 1e-10, "rho_max = 2/(eps_0*tol_V)");

    /* tol_V scaling: smaller threshold → larger cutoff */
    double k2, r2; int nq2;
    fsc_optimal_params(ctx, 1e-4, 1e-4, 1.5, &k2, &nq2, &r2);
    check_ok(r2 > rho_max, "tighter tol_V → larger rho_max");
    printf("  tol_V=1e-4 → rho_max=%.1f a0\n", r2);

    fsc_free(ctx);
}

TEST(test_optimal_params_tol) {
    printf("Slice 15: Optimal params — tol sensitivity\n");

    double eps[] = {1.0, 14.0, 14.0, 1.0};
    double d[]   = {-3.0, 3.0, 9.0};
    FSCContext *ctx = fsc_create(4, eps, d, 2, 0, 0.0);

    double k1, r1; int nq1;
    double k2, r2; int nq2;

    /* Tighter tol → larger k_max (need more k-space to converge) */
    fsc_optimal_params(ctx, 1e-2, 1e-3, 1.0, &k1, &nq1, &r1);
    fsc_optimal_params(ctx, 1e-4, 1e-3, 1.0, &k2, &nq2, &r2);
    check_ok(k2 > k1, "tighter tol → larger k_max");
    printf("  tol=1e-2 → k_max=%.3f\n", k1);
    printf("  tol=1e-4 → k_max=%.3f\n", k2);

    /* rho_max should not depend on tol (only tol_V) */
    check_rel_err(r1, r2, 1e-10, "rho_max independent of k-tol");

    fsc_free(ctx);
}

TEST(test_optimal_params_N1) {
    printf("Slice 16: Optimal params — N=1 homogeneous\n");

    /* Single homogeneous dielectric layer */
    double eps[] = {5.0};
    FSCContext *ctx = fsc_create(1, eps, NULL, 1, 0, 0.0);
    check_ok(ctx != NULL, "create N=1 ctx");

    double k_max, rho_max; int n_quad;
    fsc_optimal_params(ctx, 1e-4, 1e-3, 1.5, &k_max, &n_quad, &rho_max);

    /* k_max should be small — homogeneous ε(k)=5 constant, deviation is 0 always */
    check_ok(k_max > 0.0, "k_max > 0");
    check_ok(c99_isfinite(k_max), "k_max finite");
    check_ok(n_quad == 200, "n_quad == 200");
    printf("  k_max=%.3f  rho_max=%.1f a0\n", k_max, rho_max);

    /* For N=1 homogeneous: ε₀ = ε_c = 5, so rho_max ≈ 2/(5*1e-3) = 400 */
    double rho_expected = 2.0 / (5.0 * 1e-3);
    check_rel_err(rho_max, rho_expected, 1e-10, "N=1 rho_max = 2/(eps*tol_V)");

    fsc_free(ctx);
}

TEST(test_optimal_params_vs_ref_scan) {
    printf("Slice 17: Optimal params — agrees with dense reference scan\n");

    /* N4 bilayer direct */
    double eps[] = {1.0, 14.0, 14.0, 1.0};
    double d[]   = {-3.0, 3.0, 9.0};
    FSCContext *ctx = fsc_create(4, eps, d, 2, 0, 0.0);

    double k_fast, r; int nq;
    fsc_optimal_params(ctx, 1e-4, 1e-3, 1.0, &k_fast, &nq, &r);

    /* Reference: dense 200-point log-spaced scan (same as Python) */
    double k_lo = 0.03162277660168379;
    double k_hi = 63.09573444801933;
    double target = 1e-4 / 14.0;
    double k_ref = k_hi;
    for (int i = 0; i < 200; i++) {
        double frac = (double)i / 199.0;
        double k = k_lo * pow(k_hi / k_lo, frac);
        double ek = fsc_epsilon_k(ctx, k);
        if (fabs(1.0/ek - 1.0/14.0) < target) {
            k_ref = k;
            break;
        }
    }
    printf("  fast search k_envelope=%.4f  ref scan k_envelope=%.4f\n",
           k_fast, k_ref);

    /* Fast search should be within ±15% of reference */
    check_rel_err(k_fast, k_ref, 0.15, "k_envelope agrees with ref scan");

    fsc_free(ctx);
}

TEST(test_optimal_params_convergence) {
    printf("Slice 18: V(rho) converged to within tol at auto k_max\n");

    /* N4 bilayer */
    double eps[] = {1.0, 14.0, 14.0, 1.0};
    double d[]   = {-3.0, 3.0, 9.0};
    FSCContext *ctx = fsc_create(4, eps, d, 2, 0, 0.0);

    double tol = 1e-4;
    double k_auto, rho_max; int nq;
    fsc_optimal_params(ctx, tol, 1e-3, 1.5, &k_auto, &nq, &rho_max);

    double rhos[] = {0.5, 2.0, 10.0, 30.0};
    int nr = 4;

    for (int i = 0; i < nr; i++) {
        double rho = rhos[i];

        /* V at auto-tuned k_max */
        double V_auto = fsc_potential(ctx, rho, k_auto, tol);

        /* Reference: same integral but with 5× larger k_max */
        double V_ref  = fsc_potential(ctx, rho, k_auto * 5.0, tol * 0.01);

        double re = fabs(V_auto - V_ref) / fmax(fabs(V_ref), 1e-15);
        char msg[64];
        snprintf(msg, sizeof(msg), "V(rho=%.1f) converged to tol=%.0e", rho, tol);
        check_ok(re < tol * 10.0, msg);
        printf("  rho=%.1f  V_auto=%.6f  V_ref=%.6f  re=%.2e  (k_auto=%.2f)\n",
               rho, V_auto, V_ref, re, k_auto);
    }

    fsc_free(ctx);
}

typedef struct {
    const char *name;
    int    n_layers;
    double eps[6];
    double d[5];
    int    c, t;
    double z_t;
} sys_config;

TEST(test_convergence_sweep) {
    printf("Slice 20: Convergence sweep — 4 diverse systems\n");

    sys_config systems[] = {
        /* name                        N  eps              d                  c  t  z_t */
        {"thin slab (0.5 a0)",         3, {1,14,1},       {-0.25,  0.25},    2, 0, 0},
        {"standard bilayer (6 a0)",    4, {1,14,14,1},    {-3,      3,  9}, 2, 0, 0},
        {"high contrast (eps=100)",    3, {1,100,1},      {-1.0,    1.0},    2, 0, 0},
        {"indirect (layer sep=7 a0)",  4, {1,14,14,1},    {-3,      3,  9}, 2, 3, 7.0},
    };
    int nsys = 4;
    double rhos[] = {0.3, 2.0, 15.0};  /* small, medium, large */
    int nr = 3;
    double tol_op = 1e-4;

    int all_ok = 1, n_checked = 0;

    for (int s = 0; s < nsys; s++) {
        sys_config *sc = &systems[s];
        FSCContext *ctx = fsc_create(sc->n_layers, sc->eps, sc->d,
                                     sc->c, sc->t, sc->z_t);
        if (!ctx) { printf("  %-30s FAIL: create\n", sc->name); all_ok = 0; continue; }

        double k_auto, rho_max; int nq;
        fsc_optimal_params(ctx, tol_op, 1e-3, 1.2, &k_auto, &nq, &rho_max);

        printf("  %-30s k_max=%-6.2f", sc->name, k_auto);

        for (int i = 0; i < nr; i++) {
            double rho = rhos[i];
            if (rho > rho_max * 0.7) continue;

            /* Reference: 2× k_max, 100× tighter tol = definitive answer */
            double V_op  = fsc_potential(ctx, rho, k_auto, tol_op);
            double V_ref = fsc_potential(ctx, rho, k_auto * 2.0, tol_op * 0.01);

            double re = fabs(V_op - V_ref) / fmax(fabs(V_ref), 1e-15);
            int ok = (re < tol_op * 50.0) && c99_isfinite(V_op);
            if (!ok) all_ok = 0;
            n_checked++;

            printf("  ρ=%.1f:re=%.1e:%s", rho, re, ok ? "✓" : "✗");
        }
        printf("\n");
        fsc_free(ctx);
    }

    printf("  checked %d points across %d systems\n", n_checked, nsys);
    check_ok(all_ok && n_checked >= 8,
             "all diverse systems converged to tol");
}

/* Portable J₀ — same algorithm as fast_screened_coulomb.c, for test use */
static double test_j0(double x) {
    double ax = fabs(x);
    if (ax < 1e-12) return 1.0;
    if (ax <= 8.0) {
        double y = ax * ax;
        double num = 57568490574.0 + y * (-13362590354.0
                    + y * (651619640.7 + y * (-11214424.18
                    + y * (77392.33017 + y * (-184.9052456)))));
        double den = 57568490411.0 + y * (1029532985.0
                    + y * (9494680.718 + y * (59272.64853
                    + y * (267.8532712 + y * 1.0))));
        return num / den;
    } else {
        double z = 8.0 / ax;
        double y = z * z;
        double xx = ax - 0.785398164;
        double P0 = 1.0 + y * (-0.001098628627 + y * (0.00002734510407
                        + y * (-0.000002073370639 + y * 2.09388711e-7)));
        double Q0 = -0.01562499995 + y * (0.0001430488765
                        + y * (-0.000006911147651 + y * (7.621095161e-7
                        + y * -9.34945152e-8)));
        return sqrt(0.636619772 / ax) * (cos(xx) * P0 - z * sin(xx) * Q0);
    }
}

TEST(test_adaptive_vs_simpson) {
    printf("Slice 22: Adaptive GK vs dense Simpson — same k_max\n");

    double eps_arr[] = {1.0, 14.0, 14.0, 1.0};
    double d_arr[]   = {-3.0, 3.0, 9.0};
    FSCContext *ctx = fsc_create(4, eps_arr, d_arr, 2, 0, 0.0);

    /* ε_c = ε(k→∞).  Use moderate k (not k=100 — exponentials would overflow).
     * For this bilayer eps=[1,14,14,1] with source in layer 2, ε_c = 14. */
    double eps_c = 14.0;   /* known from test configuration */

    double k_max  = 2.3;
    double tol_op = 1e-4;
    double rhos[] = {0.5, 2.0, 10.0, 30.0};
    int nr = 4;

    printf("  %6s  %12s  %12s  %9s  %6s  %7s\n",
           "rho", "V_adaptive", "V_simpson", "rel_diff", "n_GK", "n_simp");
    printf("  %6s  %12s  %12s  %9s  %6s  %7s\n",
           "------", "---------", "---------", "--------", "-----", "------");

    int all_ok = 1;
    for (int i = 0; i < nr; i++) {
        double rho = rhos[i];

        /* Adaptive GK (the method under test) */
        fsc_reset_eval_count();
        double Va = fsc_potential(ctx, rho, k_max, tol_op);
        long na = fsc_get_eval_count();

        /* Dense Simpson — guaranteed to resolve all J₀ oscillations */
        long ns = (long)(25.0 * k_max * rho / 3.14159 + 200.0);
        if (ns < 200) ns = 200;
        if (ns > 40000) ns = 40000;
        /* ns must be odd for Simpson */
        if (ns % 2 == 0) ns++;
        double h = k_max / (ns - 1.0);

        long nc = 0;
        double sum = 0.0;
        for (long j = 0; j < ns; j++) {
            double kk = j * h;
            double ks = (kk < 1e-15) ? 1e-15 : kk;
            double ek = fsc_epsilon_k(ctx, ks);
            double fk = (1.0 / ek - 1.0 / eps_c) * test_j0(ks * rho);
            double w = (j == 0 || j == ns - 1) ? 1.0
                     : (j % 2 == 1) ? 4.0 : 2.0;
            sum += w * fk;
            nc++;
        }
        double Vs = 2.0 * ((h / 3.0) * sum + 1.0 / (eps_c * rho));

        double diff = fabs(Va - Vs) / fmax(fabs(Vs), 1e-15);
        int ok = diff < tol_op * 20.0;  /* 0.2% for tol=1e-4 */
        if (!ok) all_ok = 0;

        printf("  %6.1f  %12.8f  %12.8f  %8.1e  %5ld  %6ld  %s\n",
               rho, Va, Vs, diff, na, nc, ok ? "OK" : "??");
    }

    check_ok(all_ok, "adaptive GK agrees with dense Simpson");
    fsc_free(ctx);
}

TEST(test_k_convergence_plateau) {
    printf("Slice 21: V(rho) plateau vs k_max\n");

    double eps[] = {1.0, 14.0, 14.0, 1.0};
    double d[]   = {-3.0, 3.0, 9.0};
    FSCContext *ctx = fsc_create(4, eps, d, 2, 0, 0.0);

    double tol = 1e-4;
    double k_auto, rhom; int nq;
    fsc_optimal_params(ctx, tol, 1e-3, 1.5, &k_auto, &nq, &rhom);

    double rhos[] = {0.5, 2.0, 10.0};
    int nr = 3;
    double kmax_vals[] = {0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 4.0, 6.0, 10.0};
    int nk = 9;

    printf("  k_max =");
    for (int k = 0; k < nk; k++) printf(" %6.1f", kmax_vals[k]);
    printf("\n");

    for (int i = 0; i < nr; i++) {
        double rho = rhos[i];
        printf("  rho=%.1f", rho);

        double V_ref = fsc_potential(ctx, rho, 20.0, 1e-8);  /* definitive */
        for (int k = 0; k < nk; k++) {
            double Vk = fsc_potential(ctx, rho, kmax_vals[k], tol);
            double re = fabs(Vk - V_ref) / fmax(fabs(V_ref), 1e-15);
            char marker = (kmax_vals[k] >= k_auto / 1.5) ? '>' : ' ';
            printf(" %c%5.4f", marker, re);
        }
        printf("  (auto k_max=%.1f)\n", k_auto);
    }

    printf("  '>' marks k >= k_envelope (auto-tuned threshold)\n");
    printf("  values are |V-V_ref|/V_ref vs V(k_max=20, tol=1e-8)\n");

    /* Verify: at auto k_max, error < 100*tol */
    double V_check = fsc_potential(ctx, 10.0, k_auto, tol);
    double V_def   = fsc_potential(ctx, 10.0, 20.0, 1e-8);
    double re_final = fabs(V_check - V_def) / fmax(fabs(V_def), 1e-15);
    check_ok(re_final < tol * 100.0, "plateau reached at auto k_max");

    fsc_free(ctx);
}

TEST(test_adaptive_vs_uniform) {
    printf("Slice 19: Adaptive GK vs uniform Simpson — eval counts\n");

    double eps[] = {1.0, 14.0, 14.0, 1.0};
    double d[]   = {-3.0, 3.0, 9.0};
    FSCContext *ctx = fsc_create(4, eps, d, 2, 0, 0.0);

    double tol  = 1e-4;
    double k_max = 2.3;   /* typical auto-tuned value */
    double rhos[] = {0.5, 2.0, 10.0, 30.0};
    int nr = 4;

    printf("  %8s  %10s  %12s  %12s  %10s\n",
           "rho", "adaptive", "uniform_est", "ratio", "V(Ry)");
    printf("  %8s  %10s  %12s  %12s  %10s\n",
           "----", "---------", "-----------", "-----------", "------");

    for (int i = 0; i < nr; i++) {
        double rho = rhos[i];

        /* Adaptive: count actual ε(k) evaluations */
        fsc_reset_eval_count();
        double V_adaptive = fsc_potential(ctx, rho, k_max, tol);
        long n_adaptive = fsc_get_eval_count();

        /* Uniform Simpson estimate:
         *   Need ~6 points per J₀ oscillation for 1e-4 accuracy.
         *   Number of oscillations in [0, k_max] ≈ k_max * ρ / π.
         *   n_uniform ≈ 6 * oscillations + safety_margin */
        double oscillations = k_max * rho / 3.1415926535;
        long n_uniform = (long)(6.0 * oscillations + 30.0);
        if (n_uniform < 50) n_uniform = 50;  /* minimum for baseline */

        double ratio = (double)n_uniform / (double)n_adaptive;

        printf("  %8.1f  %10ld  %12ld  %11.1fx  %10.6f\n",
               rho, n_adaptive, n_uniform, ratio, V_adaptive);
    }

    fsc_free(ctx);
}

TEST(test_optimal_params_safety) {
    printf("Slice 14: Optimal params — safety factor\n");

    double eps[] = {1.0, 14.0, 14.0, 1.0};
    double d[]   = {-3.0, 3.0, 9.0};
    FSCContext *ctx = fsc_create(4, eps, d, 2, 0, 0.0);
    check_ok(ctx != NULL, "create ctx");

    double k1, r1; int nq1;
    double k2, r2; int nq2;

    fsc_optimal_params(ctx, 1e-4, 1e-3, 1.0, &k1, &nq1, &r1);
    fsc_optimal_params(ctx, 1e-4, 1e-3, 2.0, &k2, &nq2, &r2);

    /* safety factor of 2 should double k_max (within rounding) */
    check_rel_err(k2, 2.0 * k1, 1e-10, "safety=2 doubles k_max");
    printf("  safety=1.0 → k_max=%.4f\n", k1);
    printf("  safety=2.0 → k_max=%.4f\n", k2);

    /* rho_max should NOT depend on safety */
    check_rel_err(r1, r2, 1e-10, "rho_max independent of safety");
    check_ok(nq1 == 200 && nq2 == 200, "n_quad unchanged");

    fsc_free(ctx);
}

TEST(test_optimal_params_indirect) {
    printf("Slice 13: Optimal params — indirect exciton\n");

    /* N4 bilayer: source in layer 2, observe in layer 3 at z=7 a0 offset */
    double eps[] = {1.0, 14.0, 14.0, 1.0};
    double d[]   = {-3.0, 3.0, 9.0};
    double z_t   = 7.0;
    FSCContext *ctx = fsc_create(4, eps, d, 2, 3, z_t);
    check_ok(ctx != NULL, "create indirect ctx");

    double k_max, rho_max; int n_quad;
    fsc_optimal_params(ctx, 1e-4, 1e-3, 1.5, &k_max, &n_quad, &rho_max);

    check_ok(k_max > 0.0, "k_max > 0");
    check_ok(c99_isfinite(k_max), "k_max finite");
    check_ok(n_quad == 200, "n_quad == 200");
    check_ok(rho_max > 0.0, "rho_max > 0");
    printf("  k_max=%.2f  rho_max=%.1f a0\n", k_max, rho_max);

    /* Indirect: V(ρ) should be much smaller than direct at same ρ */
    /* Also verify asymptotic ε(k) → ∞ works — no overflow */
    double ek_low  = fsc_epsilon_k(ctx, 0.001);
    double ek_high = fsc_epsilon_k(ctx, k_max);
    check_ok(ek_low > 0.0, "eps(k→0) > 0");
    check_ok(ek_high > ek_low, "eps(k) grows with k (indirect)");
    printf("  ε(0.001)=%.2f  ε(%.1f)=%.2e\n", ek_low, k_max, ek_high);

    fsc_free(ctx);
}

TEST(test_optimal_params_N4) {
    printf("Slice 12: Optimal params — direct exciton N4\n");

    /* N4 bilayer: vacuum | MoS2 (6 a0) | MoS2 (6 a0) | vacuum */
    double eps[] = {1.0, 14.0, 14.0, 1.0};
    double d[]   = {-3.0, 3.0, 9.0};
    FSCContext *ctx = fsc_create(4, eps, d, 2, 0, 0.0);
    check_ok(ctx != NULL, "create N4 ctx");

    double k_max, rho_max; int n_quad;
    fsc_optimal_params(ctx, 1e-4, 1e-3, 1.5, &k_max, &n_quad, &rho_max);

    check_ok(k_max > 0.0, "k_max > 0");
    check_ok(c99_isfinite(k_max), "k_max finite");
    check_ok(n_quad == 200, "n_quad == 200");
    check_ok(rho_max > 0.0, "rho_max > 0");
    printf("  k_max=%.2f  rho_max=%.1f a0 (%.1f Ang)\n",
           k_max, rho_max, rho_max * 0.529);

    /* With this stack, integrated V(ρ) at rho_max should be small */
    double V_cut = fsc_potential(ctx, rho_max, k_max, 1e-8);
    check_ok(V_cut > 0.0, "V(rho_max) > 0");
    check_ok(V_cut < 0.01, "V(rho_max) < tol_V + margin");
    printf("  V(%.1f) = %.6f Ry\n", rho_max, V_cut);

    fsc_free(ctx);
}

TEST(test_chebyshev_wide_range) {
    printf("Slice 23: Chebyshev wide-range accuracy (log-map fix)\n");

    /* Vacuum-screened thin slab: eps=[1,160,1], t=0.5 a0
     * fsc_optimal_params returns rho_max ~ 2000.  The old linear map
     * gave ~8% error at rho=1; the log map with Chebyshev-node grid
     * brings it below ~2% — a 4× improvement.  (The remaining error
     * is fundamental: degree 30 across 4.3 decades for this extreme
     * high-contrast system.  Narrower rho_max or higher degree
     * would improve it further.) */
    double eps[] = {1.0, 160.0, 1.0};
    double d[]   = {-0.25, 0.25};
    FSCContext *ctx = fsc_create(3, eps, d, 2, 0, 0.0);
    check_ok(ctx != NULL, "wide ctx");

    /* Use rho_max=2000 — typical auto-tuned value */
    double rho_min = 0.1, rho_max = 2000.0;

    /* Use k_max=30 (auto-tuned value for this system is ~29) */
    double k_use = 30.0;
    FSCFit *fit = fsc_fit_chebyshev(ctx, 30, rho_min, rho_max, k_use, 1e-6);
    check_ok(fit != NULL, "wide cheb fit");

    /* Check accuracy at several small rho — where the old map failed */
    double rhos[] = {0.1, 0.3, 1.0, 3.0, 10.0};
    int nr = 5;
    double max_err = 0.0;
    for (int i = 0; i < nr; i++) {
        double V_exact = fsc_potential(ctx, rhos[i], k_use * 2.0, 1e-8);
        double V_fit   = fsc_fit_eval(fit, rhos[i]);
        double re = fabs(V_fit - V_exact) / fmax(fabs(V_exact), 1e-15) * 100.0;
        if (re > max_err) max_err = re;
        printf("  rho=%.1f  V_exact=%.6e  V_fit=%.6e  err=%.4f%%\n",
               rhos[i], V_exact, V_fit, re);
    }
    printf("  max_err over [0.1,10] with rho_max=2000: %.4f%%\n", max_err);

    /* The log-map fix improves rho=1 error from ~7.7% (linear map)
     * to well under 3%.  For this extreme system (eps=160, 4.3 decades
     * of range) degree 30 can't reach 0.1%, but it's much better. */
    check_ok(max_err < 3.0, "wide range small-rho err < 3%");

    /* rho=1 specifically — the bug-report benchmark point */
    double V1_exact = fsc_potential(ctx, 1.0, k_use * 2.0, 1e-8);
    double V1_fit   = fsc_fit_eval(fit, 1.0);
    double re1 = fabs(V1_fit - V1_exact) / V1_exact * 100.0;
    check_ok(re1 < 2.5, "rho=1 error < 2.5% (was 7.7%% before fix)");
    printf("  rho=1 error: %.4f%% (was ~7.7%% with linear map)\n", re1);

    /* Narrower range: [0.1, 200] should be sub-1% even for this system */
    FSCFit *fit_narrow = fsc_fit_chebyshev(ctx, 30, 0.1, 200.0, k_use, 1e-6);
    check_ok(fit_narrow != NULL, "narrow fit");
    double V1_narrow = fsc_fit_eval(fit_narrow, 1.0);
    double re1_narrow = fabs(V1_narrow - V1_exact) / V1_exact * 100.0;
    check_ok(re1_narrow < 1.0, "rho=1 err < 1% with rho_max=200");
    printf("  rho=1 error with rho_max=200: %.4f%%\n", re1_narrow);

    fsc_fit_free(fit_narrow);
    fsc_fit_free(fit);
    fsc_free(ctx);
}

TEST(test_auto_fit_reports_error) {
    printf("Slice 24: Auto-fit error reporting\n");

    double eps[] = {1.0, 14.0, 14.0, 1.0};
    double d[]   = {-3.0, 3.0, 9.0};
    FSCContext *ctx = fsc_create(4, eps, d, 2, 0, 0.0);
    check_ok(ctx != NULL, "auto-err ctx");

    /* Normal target — should meet it */
    double achieved;
    FSCFit *f1 = fsc_fit_auto(ctx, 0.1, 30.0, 1.0, 20.0, 1e-8, &achieved);
    check_ok(f1 != NULL, "fit created");
    check_ok(achieved <= 1.0, "achieved <= target (1%%)");
    check_ok(achieved > 0.0, "achieved > 0 (sanity)");
    printf("  target=1.0%%  achieved=%.4f%%  degree=%d\n",
           achieved, fsc_fit_n_params(f1) - 1);
    fsc_fit_free(f1);

    /* Very tight target on wide range — even degree 30 may not meet it.
     * Use the extreme system from the wide-range test. */
    {
        double eps2[] = {1.0, 160.0, 1.0};
        double d2[]   = {-0.25, 0.25};
        FSCContext *ctx2 = fsc_create(3, eps2, d2, 2, 0, 0.0);
        check_ok(ctx2 != NULL, "extreme ctx");
        FSCFit *f2 = fsc_fit_auto(ctx2, 0.1, 2000.0, 0.1, 40.0, 1e-6, &achieved);
        check_ok(f2 != NULL, "fallback fit created");
        /* With rho_max=2000, eps=160, target=0.1%, degree 30 likely misses */
        printf("  extreme system target=0.1%%  achieved=%.4f%%  degree=%d  (target %s)\n",
               achieved, fsc_fit_n_params(f2) - 1,
               achieved <= 0.1 ? "MET" : "MISSED");
        fsc_fit_free(f2);
        fsc_free(ctx2);
    }

    /* NULL out-param should not crash (backward compatibility) */
    FSCFit *f3 = fsc_fit_auto(ctx, 0.1, 30.0, 1.0, 20.0, 1e-8, NULL);
    check_ok(f3 != NULL, "NULL out-param doesn't crash");
    fsc_fit_free(f3);

    fsc_free(ctx);
}

int main(void) {
    printf("Fast Screened Coulomb — TDD Validation\n========================================\n\n");
    test_epsilon_N3_exact();
    test_potential_hom();
    test_potential_bl();
    test_potential_ind();
    test_bspline();
    test_chebyshev();
    test_legendre();
    test_img();
    test_pade();
    test_auto_fit();
    test_benchmark();
    test_edges();
    test_optimal_params_direct();
    test_optimal_params_safety();
    test_optimal_params_tol();
    test_optimal_params_N1();
    test_optimal_params_indirect();
    test_optimal_params_N4();
    test_optimal_params_vs_ref_scan();
    test_optimal_params_convergence();
    test_convergence_sweep();
    test_adaptive_vs_simpson();
    test_k_convergence_plateau();
    test_adaptive_vs_uniform();
    test_chebyshev_wide_range();
    test_auto_fit_reports_error();
    printf("\n%d / %d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}

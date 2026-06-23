/* benchmark_fsc.c — CLI tool reproducing Python fit_benchmark() output */
#include "fast_screened_coulomb.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    /* Default bilayer from Python example */
    double epsilons[] = {1.0, 14.0, 14.0, 1.0};
    double d[]        = {-3.0, 3.0, 9.0};
    int    c          = 2;
    double rho_min    = 0.1;
    double rho_max    = 30.0;
    int    n_fit      = 50;
    int    n_eval     = 200;
    double k_max      = 20.0;
    double tol        = 1e-8;
    double targets[]  = {10.0, 5.0, 2.0, 1.0, 0.5, 0.2};

    /* Allow user to specify a different configuration */
    if (argc >= 2) {
        if (argc < 5) {
            fprintf(stderr, "Usage: %s [eps2 eps3 d_half k_max]\n", argv[0]);
            fprintf(stderr, "  Default: bilayer eps=[1,14,14,1] d=[-3,3,9]\n");
            return 1;
        }
        double eps2  = atof(argv[1]);
        double eps3  = atof(argv[2]);
        double d_half = atof(argv[3]);
        k_max = atof(argv[4]);
        epsilons[1] = eps2;
        epsilons[2] = eps3;
        d[0] = -d_half;
        d[1] =  d_half;
        d[2] =  d_half + 6.0;
    }

    FSCContext *ctx = fsc_create(4, epsilons, d, c, 0, 0.0);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    printf("Fast Screened Coulomb — Expansion Benchmark\n");
    printf("===========================================\n");
    printf("  epsilons = [%.1f, %.1f, %.1f, %.1f]\n",
           epsilons[0], epsilons[1], epsilons[2], epsilons[3]);
    printf("  d        = [%.1f, %.1f, %.1f]\n", d[0], d[1], d[2]);
    printf("  c        = %d\n", c);
    printf("  rho      = [%.2f, %.2f] a0\n", rho_min, rho_max);
    printf("  k_max    = %.1f  tol = %.1e\n", k_max, tol);
    printf("  fit_pts  = %d   eval_pts = %d\n\n", n_fit, n_eval);

    int n_entries;
    FSCBenchEntry *entries = fsc_benchmark(ctx,
                                           rho_min, rho_max,
                                           n_fit, n_eval,
                                           targets,
                                           sizeof(targets)/sizeof(targets[0]),
                                           k_max, tol,
                                           &n_entries, stdout);

    fsc_bench_free(entries);
    fsc_free(ctx);
    return 0;
}

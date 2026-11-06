/*
 * SPDX-FileCopyrightText: 2026 The Fast Screened Coulomb Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Independent uniform/slab image-charge accuracy checks in Rydberg units.
 * Compile and run with make test; no fitted scalar is used as a reference.
 * Image lengths follow electrostatic reflection at a slab's two interfaces.
 */
#include "fast_screened_coulomb.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    const double stacks[3][4]={{1.5,1.5,1.5,1.5},{1.,4.,4.,1.},{1.,4.,4.,2.}};
    const double boundaries[3]={-1.,1.,3.};
    int kind;
    for (kind=0; kind<3; ++kind) {
        int direct;
        for (direct=0; direct<=1; ++direct) {
            FSCAccuracy accuracy;
            FSCContext *ctx=fsc_create(4,stacks[kind],boundaries,2,direct?0:3,direct?0.:2.);
            FSCControlledFit *fit=fsc_fit_controlled(ctx,.01,1333.,.01,.01,&accuracy);
            printf("kind=%d direct=%d status=%d segments=%d references=%d value=%g force=%g qv=%g qf=%g tv=%g tf=%g\n",kind,direct,
                accuracy.status,accuracy.segments,accuracy.reference_points,accuracy.value_error_pct,accuracy.force_error_pct,
                accuracy.quadrature_value_pct,accuracy.quadrature_force_pct,accuracy.cutoff_value_pct,accuracy.cutoff_force_pct);
            fflush(stdout);
            if (!fit) return 1;
            {
                int point;
                double mv=0.,mf=0.;
                const double eps=stacks[kind][1],rd=(eps-stacks[kind][0])/(eps+stacks[kind][0]);
                const double ru=(eps-stacks[kind][3])/(eps+stacks[kind][3]);
                for (point=0; point<513; ++point) {
                    const double rho=point==512 ? 1333. : .01*pow(1333./.01,point/512.);
                    const double z=direct?0.:2.;
                    double v=0.,dv=0.,power=1.; int n;
                    for (n=0; n<200; ++n) {
                        const double heights[4]={z+8*n,6-z+8*n,z+2+8*n,8-z+8*n};
                        const double weights[4]={1.,ru,rd,rd*ru}; int image;
                        for (image=0; image<4; ++image) {
                            const double radius=hypot(rho,heights[image]);
                            v+=2/eps*power*weights[image]/radius;
                            dv-=2/eps*power*weights[image]/radius/radius*(rho/radius);
                        }
                        power*=rd*ru;
                    }
                    if (!isfinite(fsc_controlled_eval(fit,rho)) || !isfinite(fsc_controlled_d1(fit,rho))) return 3;
                    mv=fmax(mv,fabs(fsc_controlled_eval(fit,rho)-v)/fabs(v));
                    mf=fmax(mf,fabs(fsc_controlled_d1(fit,rho)-dv)/fabs(dv));
                }
                printf("independent images max value fraction=%g force fraction=%g\n",mv,mf); fflush(stdout);
                if (mv>1e-4 || mf>1e-4) return 2;
            }
            fsc_controlled_free(fit); fsc_free(ctx);
        }
    }
    return 0;
}

import sys; sys.path.insert(0, '../src')
import numpy as np, time, warnings
from main import optimal_parameters, V_eh_direct, V_eh

ang = 1.0 / 0.529177
rho_test = [0.5, 2.0, 10.0, 30.0]

configs = [
    ('direct d=3A',  [1,14,1],     [-1.5*ang, 1.5*ang],  2, None, 0),
    ('direct d=7A',  [1,14,1],     [-3.5*ang, 3.5*ang],  2, None, 0),
    ('direct N4',    [1,14,14,1],  [-3.0, 3.0, 9.0],     2, None, 0),
    ('indirect d=7A',[1,14,14,1],  [-3.5*ang, 3.5*ang, 10.5*ang], 2, 3, 7*ang),
]

print('Exact integral time per point:')
print(f'  {"config":>18s}  {"k_max":>6s}  {"ms/pt":>8s}  {"fit speedup":>12s}')
print('  ' + '-' * 52)
for name, eps, d_arr, c, t, zt in configs:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        km, nq, _, _ = optimal_parameters(eps, d_arr, c, t=t, z_t=zt, tol=1e-4, tol_V=1e-2)
    
    n_repeat = 80
    t0 = time.perf_counter()
    for rho in rho_test * (n_repeat // len(rho_test)):
        if t is None:
            V_eh_direct([rho], eps, d_arr, c, k_max=km, n_quad=200)
        else:
            V_eh([rho], eps, d_arr, c, t=t, z_t=zt, k_max=km, n_quad=200)
    t_ms = (time.perf_counter() - t0) / n_repeat * 1000
    
    speedup = t_ms * 1000 / 0.105  # vs B-spline at 0.105 us
    print(f'  {name:>18s}  {km:6.2f}  {t_ms:8.2f}  {speedup:10.0f}x')

